"""端到端两阶段训练：解冻 ConvNeXt 最后一个 stage + BERT 最后两层。

阶段 1（--stage1-epochs，默认 2）：编码器全冻结，只训练两个投影层。
阶段 2（剩余 epoch）：解冻 ConvNeXt.stages[-1] + ConvNeXt.norm + BERT.layers[-2:]，
    编码器用小 lr，投影层 lr 降一档，联合训练。

注意 1 个 epoch = 把每张图的 caption 遍历 --caption-passes 遍（默认 5）。
Flickr30k 每图 5 条 caption，所以默认一轮就用满全部 145000 条图文对；
想回到「每图每轮只抽 1 条」的老行为用 --caption-passes 1。

--------------------------------------------------------------------------
启动参数预设（--preset，可切换）
--------------------------------------------------------------------------
    auto            默认。Windows 用 local-12g，Linux 用 cloud-5090
    local-12g       单卡 12GB 笔记本（batch 64 / workers 8 / 12 epoch）
    cloud-5090      RTX 5090 32GB + 25 核 + 90GB 内存（batch 256 / workers 16 / 20 epoch）
    cloud-5090-long 同上，更大 batch 更长训练，冲分数（batch 384 / 30 epoch）
    smoke           冒烟测试，几十秒验证环境和显存

    # 预设 + 覆盖单个参数
    python src/train_dual_tower.py --preset cloud-5090 --epochs 30
    bash scripts/train.sh cloud-5090

--------------------------------------------------------------------------
产物与命名（多次训练全部保留，不会互相覆盖）
--------------------------------------------------------------------------
每次运行生成一个 tag（时间戳，可用 --run-name 追加可读后缀），所有文件共用它：

    train_<tag>.log          完整控制台输出（每步 loss、评估指标、异常栈）
    best_<tag>.pt            按 val 指标选出的最优权重
    last_<tag>.pt            最近一个 epoch 的权重（用于 --resume）
    metrics_<tag>.csv        每个 epoch 一行，可直接画曲线
    train_log_<tag>.jsonl    同上但为 JSON，保留全部字段
    config_<tag>.json        本次全部超参 + 环境信息，便于复现

例：--run-name lr3e5 → best_20261128_060102_lr3e5.pt 对应
    train_20261128_060102_lr3e5.log

--------------------------------------------------------------------------
云端依赖
--------------------------------------------------------------------------
RTX 50 系是 sm_120，必须 torch >= 2.7 且 cu128/cu130 的 wheel；
CUDA <= 13.2 的环境请用 cu130 或 cu128，不要用 cu134：

    pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
    pip install transformers
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.dataset import Flickr30kDataset, ImageListDataset, UniqueImageBatchSampler
from data.process import get_image_transform, get_train_transform
from model.DualTower import DualTowerModel
from model.model import (
    load_bert,
    load_convnext,
    trainable_parameter_summary,
    unfreeze_bert_last_layers,
    unfreeze_convnext_last_stages,
)
from utils.loss import info_nce_loss
from utils.recall import ndcg_i2t, ndcg_t2i, recall_i2t, recall_t2i

PROJECT_ROOT = SRC_ROOT.parent

# ConvNeXt-T 官方权重（ImageNet-1k），data/raw 不入库，云端需要单独下载
CONVNEXT_URL = "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth"

# 每个预设只写「和默认不同的项」，其余沿用 argparse 默认值
PRESETS = {
    "local-12g": dict(
        batch_size=64, accum=1, workers=8, eval_batch_size=128, prefetch_factor=4,
        epochs=12, stage1_epochs=2, lr_proj=1e-3, lr_enc=2e-5,
    ),
    "cloud-5090": dict(
        batch_size=256, accum=1, workers=16, eval_batch_size=256, prefetch_factor=2,
        epochs=20, stage1_epochs=2, lr_proj=1e-3, lr_enc=3e-5,
    ),
    "cloud-5090-long": dict(
        batch_size=512, accum=1, workers=20, eval_batch_size=256, prefetch_factor=2,
        epochs=30, stage1_epochs=2, lr_proj=1.5e-3, lr_enc=4e-5,caption_passes=5
    ),
    "smoke": dict(
        batch_size=8, accum=1, workers=2, eval_batch_size=32, prefetch_factor=2,
        epochs=2, stage1_epochs=1, limit_batches=3, limit_val=64,
    ),
}


def resolve_preset(name):
    """auto -> 按操作系统挑默认预设。"""
    if name == "auto":
        return "local-12g" if os.name == "nt" else "cloud-5090"
    if name not in PRESETS:
        raise SystemExit(f"未知 preset: {name}，可选 {sorted(PRESETS)} 或 auto")
    return name


def parse_args(argv=None):
    # 第一遍只取 --preset，用它改掉第二遍的默认值
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", default="auto")
    known, _ = pre.parse_known_args(argv)
    preset = resolve_preset(known.preset)

    p = argparse.ArgumentParser(description="双塔两阶段端到端训练")
    p.add_argument("--preset", default=preset, choices=sorted(PRESETS) + ["auto"])
    # 数据
    p.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "annotations" / "captions_flickr30k.csv"))
    p.add_argument("--images", default=str(PROJECT_ROOT / "data" / "raw" / "flickr30k" / "flickr30k-images"))
    p.add_argument("--convnext-weights", default=str(PROJECT_ROOT / "data" / "raw" / "convnext_tiny_1k_224_ema.pth"))
    p.add_argument("--max-len", type=int, default=32)
    # 训练
    p.add_argument("--epochs", type=int, default=12, help="总 epoch 数")
    p.add_argument("--stage1-epochs", type=int, default=2, help="只训投影的 epoch 数")
    p.add_argument("--batch-size", type=int, default=64, help="单卡 batch（InfoNCE 负例数 = 它）")
    p.add_argument("--accum", type=int, default=1, help="梯度累积步数（不增加负例数）")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--embed-dim", type=int, default=256)
    # 学习率
    p.add_argument("--lr-proj", type=float, default=1e-3, help="投影层 lr（阶段1）")
    p.add_argument("--lr-enc", type=float, default=2e-5, help="编码器 lr（阶段2）")
    p.add_argument("--proj-lr-scale", type=float, default=0.3, help="阶段2 投影层 lr 的缩放系数")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    # 解冻
    p.add_argument("--unfreeze-conv-stages", type=int, default=1, help="解冻 ConvNeXt 最后几个 stage")
    p.add_argument("--unfreeze-bert-layers", type=int, default=2, help="解冻 BERT 最后几层")
    # 性能
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto")
    p.add_argument("--workers", type=int, default=-1, help="-1 表示自动（Windows 0，其它 8）")
    p.add_argument("--prefetch-factor", type=int, default=4, help="每个 worker 预取的 batch 数")
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # 记录
    p.add_argument("--out", default=str(PROJECT_ROOT / "output" / "dual_tower"))
    p.add_argument("--log-file", default="", help="空表示 <out>/train_<tag>.log；可含 {tag} 占位；none 表示不写文件")
    p.add_argument("--run-name", default="",
                   help="本次运行的名字，会拼进所有产物文件名（如 best_<tag>_lr3e5.pt），便于区分多次实验")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=1)
    # 评估
    p.add_argument("--eval-train", action=argparse.BooleanOptionalAction, default=True,
                   help="每次评估时一并算训练集指标（用 --no-eval-train 关掉）")
    p.add_argument("--eval-train-subset", type=int, default=0,
                   help="训练集评估的候选池大小；0 表示与 val 对齐（同样的图片数，数字才可比）")
    # 数据
    p.add_argument("--caption-passes", type=int, default=1,
                   help="每个 epoch 遍历几遍 caption（Flickr30k 每图 5 条，5 = 一轮用完；1 = 老行为）")
    # 杂项
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="")
    p.add_argument("--limit-batches", type=int, default=0, help="每个 epoch 最多跑几个 batch（冒烟测试用）")
    p.add_argument("--limit-val", type=int, default=0, help="评估时最多用多少张图（冒烟测试用）")

    p.set_defaults(**PRESETS[preset])  # 预设覆盖 argparse 默认值，命令行参数仍可再覆盖
    args = p.parse_args(argv)
    args.preset = preset
    return args


class _Tee:
    """把 stdout/stderr 同时写到控制台和日志文件。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_amp(choice: str, device: torch.device):
    """返回 autocast 的 dtype；None 表示关闭混合精度。"""
    if choice == "none" or device.type != "cuda":
        return None
    if choice == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[choice]


def resolve_workers(n: int) -> int:
    if n >= 0:
        return n
    if os.name == "nt":
        return 0
    return min(8, os.cpu_count() or 1)


def describe_env(device):
    """打印并校验运行环境；显卡/显存/磁盘有问题时给出能直接照做的提示。"""
    print(f"[env] torch {torch.__version__} cuda {torch.version.cuda}")
    print(f"[env] device={device}")
    if device.type != "cuda":
        print("[env][warn] 没检测到 CUDA，将在 CPU 上训练（会非常慢）")
        return
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[env] gpu={props.name} {props.total_memory / 2**30:.1f}GB sm_{cap[0]}{cap[1]}")
    try:
        probe = torch.randn(128, 128, device=device)
        (probe @ probe).sum().item()
    except RuntimeError as exc:
        raise SystemExit(
            f"CUDA kernel 启动失败：{exc}\n"
            "RTX 50 系是 sm_120，需要 torch>=2.7 且 cu128/cu130 的 wheel；\n"
            "  pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision"
        )


def check_disk(path: Path, need_gb: float = 6.0):
    """checkpoint 约 0.8GB/份（model+optimizer），空间不够时提前告警。"""
    try:
        free_gb = shutil.disk_usage(path).free / 2**30
    except OSError:
        return
    print(f"[env] {path} 可用空间 {free_gb:.1f}GB")
    if free_gb < need_gb:
        print(f"[env][warn] 可用空间不足 {need_gb}GB，checkpoint 可能写不下")


def build_optimizer(model, args, lr_proj):
    """按「投影 / 编码器-需衰减 / 编码器-不衰减」分三组，给不同 lr。

    bias 和 LayerNorm 的 1 维参数不做 weight decay（BERT 微调的常规做法）。
    """
    proj, enc_decay, enc_nodecay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("proj_"):
            proj.append(p)
        elif p.ndim == 1:
            enc_nodecay.append(p)
        else:
            enc_decay.append(p)
    groups = [{"params": proj, "lr": lr_proj, "weight_decay": args.weight_decay}]
    if enc_decay:
        groups.append({"params": enc_decay, "lr": args.lr_enc, "weight_decay": args.weight_decay})
    if enc_nodecay:
        groups.append({"params": enc_nodecay, "lr": args.lr_enc, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def make_scheduler(opt, total_steps, warmup_steps):
    """线性 warmup + cosine 衰减，按优化器步计数。"""
    total_steps = max(1, total_steps)
    warmup_steps = max(1, min(warmup_steps, total_steps - 1))

    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


@torch.no_grad()
def evaluate(model, ds, args, device, amp_dtype, workers, max_images=0):
    """在一个划分上算 T2I / I2T 的 Recall 与 NDCG（候选池 = 该划分的图片）。

    Args:
        ds: Flickr30kDataset。只用它的 rows 和预编码好的 input_ids/attention_mask，
            不会调用 __getitem__，所以传 train 集也不会带入随机增广。
        max_images: 只用前 N 张图当候选池（0 表示全用）。
            train 集有 29000 张图，全上会构造 145000x29000 的相似度矩阵；
            要和 val 的数字可比，就取同样多的图（默认 1014）。
    """
    model.eval()
    indexed = list(enumerate(ds.rows))
    filenames = list(dict.fromkeys(r["filename"] for _, r in indexed))
    if max_images:
        filenames = filenames[:max_images]
    pos = {fn: i for i, fn in enumerate(filenames)}
    sel = [(i, r) for i, r in indexed if r["filename"] in pos]
    gt = torch.tensor([pos[r["filename"]] for _, r in sel], dtype=torch.long)

    def autocast_ctx():
        return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None)

    # ---------- 图库 ----------
    img_ds = ImageListDataset(filenames, args.images, get_image_transform())
    img_loader = DataLoader(
        img_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=workers, pin_memory=(device.type == "cuda"),
    )
    img_embs = []
    for images in img_loader:
        images = images.to(device, non_blocking=True)
        with autocast_ctx():
            img_embs.append(model.encode_image(images).float())
    img_embs = torch.cat(img_embs, dim=0)

    # ---------- 查询（该划分的全部 caption） ----------
    if ds.input_ids is not None:
        row_ids = [i for i, _ in sel]
        ids_all = ds.input_ids[row_ids]
        mask_all = ds.attention_mask[row_ids]
    else:
        enc = ds.tokenizer(
            [r["caption"] for _, r in sel], padding="max_length", truncation=True,
            max_length=ds.max_len, return_tensors="pt",
        )
        ids_all, mask_all = enc["input_ids"], enc["attention_mask"]

    txt_embs = []
    for s in range(0, ids_all.shape[0], args.eval_batch_size):
        ids = ids_all[s : s + args.eval_batch_size].to(device, non_blocking=True)
        msk = mask_all[s : s + args.eval_batch_size].to(device, non_blocking=True)
        with autocast_ctx():
            txt_embs.append(model.encode_text(ids, msk).float())
    txt_embs = torch.cat(txt_embs, dim=0)

    # 指标计算放 CPU：矩阵只有几 MB，还能避开设备不一致
    sim = (txt_embs @ img_embs.T).cpu()
    n_img = sim.shape[1]
    metrics = {}
    for k in (1, 5, 10):
        if k > n_img:  # 候选池太小（比如 --limit-val）时跳过过大的 k
            continue
        metrics[f"T2I@{k}"] = recall_t2i(sim, gt, k)
        metrics[f"I2T@{k}"] = recall_i2t(sim, gt, k)
    if n_img >= 10:
        metrics["T2I-NDCG@10"] = ndcg_t2i(sim, gt, 10)
        metrics["I2T-NDCG@10"] = ndcg_i2t(sim, gt, 10)
    return metrics


def save_checkpoint(path, model, opt, sched, scaler, epoch, best, args, stage):
    """保存完整状态 + 兼容旧工具（load_projection / bench_hnsw）的投影权重。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict() if opt is not None else None,
        "scheduler": sched.state_dict() if sched is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best": best,
        "stage": stage,
        "args": vars(args),
        "proj_img": model.proj_img.state_dict(),
        "proj_txt": model.proj_txt.state_dict(),
        "embed_dim": model.embed_dim,
    }
    torch.save(payload, path)


def append_jsonl(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_metrics_csv(path, history):
    """把每个 epoch 的指标重写成一张表（行数很少，直接整表重写最省心）。"""
    if not history:
        return
    cols = []
    for rec in history:
        for key in rec:
            if key not in cols:
                cols.append(key)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for rec in history:
            writer.writerow({k: rec.get(k, "") for k in cols})


def check_inputs(args):
    """开跑前把缺失的输入一次性列全。

    data/raw/ 与 data/annotations/ 里的大文件都在 .gitignore 中，
    换机器/上云时必须单独上传或下载，否则会在加载权重时抛一个很深的
    FileNotFoundError，看不出到底缺什么。
    """
    problems = []
    weights = Path(args.convnext_weights)
    if not weights.is_file():
        problems.append(
            f"缺少 ConvNeXt 权重: {weights}\n"
            f"     mkdir -p '{weights.parent}' && curl -L -o '{weights}' {CONVNEXT_URL}\n"
            f"     # 没有 curl 就用: wget -O '{weights}' {CONVNEXT_URL}"
        )
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        problems.append(f"缺少标注文件: {csv_path}")
    images = Path(args.images)
    if not images.is_dir():
        problems.append(f"缺少图片目录: {images}")
    elif not any(images.iterdir()):
        problems.append(f"图片目录是空的: {images}")
    if problems:
        raise SystemExit(
            "[输入检查失败]\n  - " + "\n  - ".join(problems)
            + "\n\ndata/raw/ 和 data/annotations/ 不入库，需要单独上传或下载。"
        )
    print(f"[data] 权重/标注/图片都在位: {weights.name}, {csv_path.name}, {images.name}/")


def run(args):
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    amp_dtype = pick_amp(args.amp, device)
    workers = resolve_workers(args.workers)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"[args] preset={args.preset}")
    print(f"[args] {json.dumps({k: v for k, v in vars(args).items() if k not in ('preset',)}, ensure_ascii=False)}")
    describe_env(device)
    check_disk(out_dir)
    check_inputs(args)
    print(f"[env] amp={amp_dtype} workers={workers} prefetch={args.prefetch_factor}")

    # ---------- 模型 ----------
    convnext = load_convnext(args.convnext_weights)
    bert = load_bert()
    model = DualTowerModel(convnext, bert, embed_dim=args.embed_dim).to(device)
    n_train, n_total = trainable_parameter_summary(model)
    print(f"[model] 可训练 {n_train/1e6:.2f}M / 总计 {n_total/1e6:.2f}M")

    # ---------- 数据 ----------
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    print("[data] 读取并预编码 caption ...")
    train_ds = Flickr30kDataset(
        args.csv, args.images, get_train_transform(), tokenizer,
        split="train", max_len=args.max_len, pretokenize=True,
    )
    val_ds = Flickr30kDataset(
        args.csv, args.images, get_image_transform(), tokenizer,
        split="val", max_len=args.max_len, pretokenize=True,
    )
    print(f"[data] train caption {len(train_ds)} 条 / val caption {len(val_ds)} 条")

    sampler = UniqueImageBatchSampler(
        train_ds.rows, args.batch_size, seed=args.seed, passes=args.caption_passes
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler, num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=workers > 0,
        prefetch_factor=args.prefetch_factor if workers > 0 else None,
    )

    batches_per_epoch = len(train_loader)
    if args.limit_batches:
        batches_per_epoch = min(batches_per_epoch, args.limit_batches)
    steps_per_epoch = math.ceil(batches_per_epoch / args.accum)
    print(f"[data] {len(sampler)} 个 batch/epoch（每 batch {args.batch_size} 图，"
          f"caption 遍历 {args.caption_passes} 遍），实际每 epoch {batches_per_epoch} 步")

    # ---------- 断点续训 ----------
    start_epoch, best, resume_ckpt = 0, -1.0, None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ckpt["model"])
        start_epoch = int(resume_ckpt["epoch"]) + 1
        best = float(resume_ckpt["best"])
        print(f"[resume] 从 epoch {start_epoch} 继续，best={best:.4f}")

    stage1_end = min(args.stage1_epochs, args.epochs)

    def stage_of(epoch):
        return 1 if epoch < stage1_end else 2

    stage = stage_of(start_epoch)
    if stage == 2:
        c = unfreeze_convnext_last_stages(model.convnext, args.unfreeze_conv_stages)
        b = unfreeze_bert_last_layers(model.bert, args.unfreeze_bert_layers)
        print(f"[unfreeze] convnext {c/1e6:.2f}M + bert {b/1e6:.2f}M")
    lr_proj = args.lr_proj * (args.proj_lr_scale if stage == 2 else 1.0)
    opt = build_optimizer(model, args, lr_proj)
    total_steps = steps_per_epoch * max(1, args.epochs - start_epoch)
    sched = make_scheduler(opt, total_steps, int(total_steps * args.warmup_ratio))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    if resume_ckpt is not None and resume_ckpt.get("stage") == stage:
        try:
            opt.load_state_dict(resume_ckpt["optimizer"])
            if resume_ckpt.get("scheduler") is not None:
                sched.load_state_dict(resume_ckpt["scheduler"])
            if resume_ckpt.get("scaler") is not None and scaler.is_enabled():
                scaler.load_state_dict(resume_ckpt["scaler"])
            print("[resume] 优化器状态已恢复")
        except (ValueError, KeyError) as exc:
            print(f"[resume] 优化器状态不兼容，按新优化器继续：{exc}")

    # ---------- 记录文件（全部带 tag，多次训练互不覆盖） ----------
    tag = args.run_tag
    log_path = out_dir / f"train_log_{tag}.jsonl"
    csv_path = out_dir / f"metrics_{tag}.csv"
    cfg_path = out_dir / f"config_{tag}.json"
    best_path = out_dir / f"best_{tag}.pt"
    last_path = out_dir / f"last_{tag}.pt"
    env_info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "amp": str(amp_dtype),
        "gpu": torch.cuda.get_device_properties(0).name if device.type == "cuda" else None,
        "args": {k: v for k, v in vars(args).items()},
    }
    cfg_path.write_text(json.dumps(env_info, ensure_ascii=False, indent=2), encoding="utf-8")
    history = []

    # ---------- 训练 ----------
    for epoch in range(start_epoch, args.epochs):
        epoch_stage = stage_of(epoch)
        if epoch_stage != stage:
            # 进入微调阶段：解冻 + 换优化器（不同 lr）
            c = unfreeze_convnext_last_stages(model.convnext, args.unfreeze_conv_stages)
            b = unfreeze_bert_last_layers(model.bert, args.unfreeze_bert_layers)
            n_train, _ = trainable_parameter_summary(model)
            print(f"[unfreeze] convnext {c/1e6:.2f}M + bert {b/1e6:.2f}M，可训练 {n_train/1e6:.2f}M")
            stage = epoch_stage
            opt = build_optimizer(model, args, args.lr_proj * args.proj_lr_scale)
            total_steps = steps_per_epoch * max(1, args.epochs - epoch)
            sched = make_scheduler(opt, total_steps, int(total_steps * args.warmup_ratio))

        sampler.set_epoch(epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        running, seen, step = 0.0, 0, 0
        t0 = time.perf_counter()
        params = [p for g in opt.param_groups for p in g["params"]]

        for step, (images, ids, mask, _) in enumerate(train_loader):
            if args.limit_batches and step >= args.limit_batches:
                break
            images = images.to(device, non_blocking=True)
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                img_emb, txt_emb = model(images, ids, mask)
                loss = info_nce_loss(img_emb, txt_emb, args.temperature)

            scaled = loss / args.accum
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            running += loss.item()
            seen += images.shape[0]

            is_last = (step + 1) == batches_per_epoch
            if (step + 1) % args.accum == 0 or is_last:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

                if args.log_interval and ((step + 1) // args.accum) % args.log_interval == 0:
                    lr_now = opt.param_groups[0]["lr"]
                    speed = seen / max(1e-9, time.perf_counter() - t0)
                    print(f"  ep{epoch+1} step{step+1}/{batches_per_epoch} "
                          f"loss {running/(step+1):.4f} lr_proj {lr_now:.2e} {speed:.1f} img/s")

        epoch_loss = running / max(1, min(batches_per_epoch, step + 1))
        epoch_time = time.perf_counter() - t0
        record = {"epoch": epoch + 1, "stage": stage, "loss": round(epoch_loss, 5),
                  "time_s": round(epoch_time, 1),
                  "img_per_s": round(batches_per_epoch * args.batch_size / max(1e-9, epoch_time), 1)}
        if device.type == "cuda":
            record["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        print(f"[epoch {epoch+1}/{args.epochs}] stage{stage} loss {epoch_loss:.4f} {epoch_time:.1f}s"
              + (f" peak {record['peak_gb']}GB" if "peak_gb" in record else ""))

        if (epoch + 1) % args.val_every == 0:
            n_eval_img = args.limit_val or 1014  # val 划分就是 1014 张（Flickr30k 1k）
            metrics = evaluate(model, val_ds, args, device, amp_dtype, workers,
                               max_images=args.limit_val)
            print("  [val]   " + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
            score = 0.5 * (metrics["T2I@1"] + metrics["I2T@1"])
            record.update({k: round(v, 5) for k, v in metrics.items()})
            record["score"] = round(score, 5)

            if args.eval_train:
                # 候选池规模和 val 保持一致，否则两边的 R@10 没法直接比
                subset = args.eval_train_subset or n_eval_img
                train_metrics = evaluate(model, train_ds, args, device, amp_dtype, workers,
                                         max_images=subset)
                print(f"  [train] " + " ".join(f"{k} {v:.4f}" for k, v in train_metrics.items())
                      + f"  (前 {subset} 张图)")
                record.update({f"train_{k}": round(v, 5) for k, v in train_metrics.items()})
                for key in ("T2I@10", "I2T@10"):
                    if key in metrics and key in train_metrics:
                        record[f"gap_{key}"] = round(
                            train_metrics[key] - metrics[key], 5)
                gaps = [f"{k} {record['gap_' + k]:+.4f}"
                        for k in ("T2I@10", "I2T@10") if "gap_" + k in record]
                if gaps:
                    print("  [gap]   " + " ".join(gaps) + "  (train - val)")

            if score > best:
                best = score
                save_checkpoint(best_path, model, opt, sched, scaler, epoch, best, args, stage)
                print(f"  -> 新的 best ({best:.4f}) 已保存 {best_path.name}")

        history.append(record)
        append_jsonl(log_path, record)
        write_metrics_csv(csv_path, history)
        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_checkpoint(last_path, model, opt, sched, scaler, epoch, best, args, stage)

    save_checkpoint(last_path, model, opt, sched, scaler, args.epochs - 1, best, args, stage)
    print(f"[done] best={best:.4f}")
    if best_path.exists():
        print(f"[done] tag={tag} 权重 {best_path.name} / {last_path.name}")
    else:
        print(f"[done] tag={tag} 本次没有超过历史 best，未生成 {best_path.name}；"
              f"最终权重在 {last_path.name}")
    print(f"[done] 日志与指标 {log_path.name} / {csv_path.name} / {cfg_path.name}，目录 {out_dir}")


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 本次运行的所有产物（权重/日志/指标）共用同一个 tag，互不覆盖，
    # 文件名一一对应：train_<tag>.log ↔ best_<tag>.pt / last_<tag>.pt
    tag = time.strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        tag = f"{tag}_{args.run_name}"
    args.run_tag = tag

    log_file = None
    old_out, old_err = sys.stdout, sys.stderr
    if args.log_file.lower() != "none":
        if args.log_file:
            log_path = Path(args.log_file.replace("{tag}", tag))
        else:
            log_path = out_dir / f"train_{tag}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = _Tee(old_out, log_file)
        sys.stderr = _Tee(old_err, log_file)
        print(f"[log] tag={tag} 完整输出同时写入 {log_path}")

    try:
        run(args)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = old_out, old_err
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()