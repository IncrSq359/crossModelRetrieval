"""投影层训练的共享模块：数据加载、采样器、损失、Recall 评估、日志。

这个文件里只放「不随超参数变化」的东西，调参脚本 tune_projection.py 会 import 它。
"""

import os
import random
from datetime import datetime

from sympy import true
import torch.nn as nn
import torch
import torch.nn.functional as F

# 项目根目录 = 本文件（src/proj_lib.py）所在目录的上一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# data_store 与「挑战实验」同级，即项目根目录的上一级
DATA_STORE = os.path.join(os.path.dirname(PROJECT_ROOT), "data_store")

IMG_PATH = os.path.join(PROJECT_ROOT, "output", "features", "image_features.pth")
TXT_PATH = os.path.join(PROJECT_ROOT, "output", "features", "text_features_v2.pth")


def load_data(img_path=IMG_PATH, txt_path=TXT_PATH):
    """加载缓存特征，并建好训练 / 评估要用的索引。

    Args:
        img_path: 图像特征文件路径。
        txt_path: 文本特征文件路径。
    Returns:
        字典，键为：
            img_feat: 图像特征 [31014, 768]。
            txt_feat: 文本特征 [155070, 1536]（masked mean 与 CLS 拼接）。
            caption_rows_of: 图像行号 -> 该图 5 条 caption 的行号列表。
            train_img_rows: 训练集图像行号列表（29000 个）。
            val_img_rows: 验证集图像行号列表（1014 个）。
    """
    img_data = torch.load(img_path, map_location="cpu")
    txt_data = torch.load(txt_path, map_location="cpu")

    # 文件名 -> 图像特征行号
    img_index = {fn: i for i, fn in enumerate(img_data["filenames"])}

    # 图像行号 -> 该图 5 条 caption 的行号
    caption_rows_of = {}
    for txt_row, fn in enumerate(txt_data["filenames"]):
        img_row = img_index[fn]  # 这条 caption 属于哪张图
        caption_rows_of.setdefault(img_row, []).append(txt_row)

    splits = img_data["splits"]
    train_img_rows = [i for i, s in enumerate(splits) if s == "train"]
    val_img_rows = [i for i, s in enumerate(splits) if s == "val"]

    txt_mean = txt_data["mean"]
    txt_cls = txt_data["cls"]
    txt_filenames = txt_data["filenames"]
    if txt_mean.shape[0] != len(txt_filenames) or txt_cls.shape[0] != len(
        txt_filenames
    ):
        raise ValueError(
            "Text feature rows must match caption metadata: "
            f"mean={txt_mean.shape[0]}, cls={txt_cls.shape[0]}, "
            f"filenames={len(txt_filenames)}"
        )
    if txt_mean.shape != txt_cls.shape:
        raise ValueError(
            f"Mean and CLS feature shapes must match, got "
            f"{tuple(txt_mean.shape)} and {tuple(txt_cls.shape)}"
        )
    txt_feat = torch.cat((txt_mean, txt_cls), dim=1)

    return {
        "img_feat": img_data["features"],
        "txt_feat": txt_feat,
        "caption_rows_of": caption_rows_of,
        "train_img_rows": train_img_rows,
        "val_img_rows": val_img_rows,
    }


def make_epoch_batches(train_img_rows, caption_rows_of, batch_size, seed):
    """为一个 epoch 生成 batch：每个 batch 里每张图最多一条 caption。

    这样同一 batch 内不会出现「同一张图配两条 caption」，避免 InfoNCE 的假负样本。

    Args:
        train_img_rows: 训练集的图像行号列表。
        caption_rows_of: 图像行号 -> 该图 5 条 caption 行号的列表。
        batch_size: 每个 batch 的大小。
        seed: 随机种子，每个 epoch 传不同的值。
    Returns:
        batches: 列表，每个元素是 batch_size 个 (图像行号, caption 行号)。
    """
    rng = random.Random(seed)  # 独立的随机数生成器，不影响全局随机状态
    # 每张图随机抽 1 条 caption
    pairs = [
        (img_row, rng.choice(caption_rows_of[img_row])) for img_row in train_img_rows
    ]
    rng.shuffle(pairs)

    batches = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        if len(batch) == batch_size:  # 丢掉最后不足一个 batch 的尾巴
            batches.append(batch)
    return batches


def info_nce_loss(img_emb, txt_emb, temperature=0.07):
    """双向 InfoNCE 损失。

    Args:
        img_emb: 图像向量 [B, D]，已 L2 归一化。
        txt_emb: 文本向量 [B, D]，已 L2 归一化。
        temperature: 温度系数，越小越「尖锐」。
    Returns:
        标量损失。
    """
    logits = img_emb @ txt_emb.T / temperature  # [B, B]，对角线是正样本
    labels = torch.arange(img_emb.shape[0])  # 第 i 行的正确答案是第 i 列
    loss_i2t = F.cross_entropy(logits, labels)  # 按行：图找文
    loss_t2i = F.cross_entropy(logits.T, labels)  # 按列：文找图
    return (loss_i2t + loss_t2i) / 2


def recall_t2i(sim, gt, k):
    """文搜图的 Recall@K：正确图片是否排进前 K 名。

    Args:
        sim: 相似度矩阵 [num_caption, num_img]。
        gt: 每条 caption 的正确图片列号 [num_caption]。
        k: 取前几名。
    Returns:
        命中率（0 到 1 的浮点数）。
    """
    topk_idx = sim.topk(k, dim=1).indices  # 每条 caption 的前 K 张图
    hit = (topk_idx == gt.unsqueeze(1)).any(dim=1)
    return hit.float().mean().item()


def ndcg_t2i(sim, gt, k=10):
    """文搜图的 NDCG@K（每条文字只有 1 张正确图片）。

    Args:
        sim: [n_txt, n_img] 相似度矩阵，第 i 行是第 i 条文字对所有图片的分数。
        gt: [n_txt] 每条文字对应的正确图片的列号。
    Returns:
        NDCG@K 的平均值（float）。
    """
    rows = torch.arange(len(gt))
    correct_score = sim[rows, gt]
    bigger = sim > correct_score.unsqueeze(1)
    rank = bigger.sum(dim=1) + 1
    gain = 1.0 / torch.log2(rank.float() + 1)
    gain = torch.where(rank <= k, gain, torch.zeros_like(gain))
    return gain.mean().item()


def recall_i2t(sim, gt, k):
    """图搜文的 Recall@K：前 K 名 caption 里是否有属于这张图的。

    Args:
        sim: 相似度矩阵 [num_caption, num_img]。
        gt: 每条 caption 对应的图片列号 [num_caption]。
        k: 取前几名。
    Returns:
        命中率（0 到 1 的浮点数）。
    """
    topk_idx = sim.T.topk(k, dim=1).indices  # 每张图的前 K 条 caption 的序号
    pred_img = gt[topk_idx]  # 这些 caption 各自属于哪张图
    self_pos = torch.arange(sim.shape[1]).unsqueeze(1)  # 每张图自己的列号
    hit = (pred_img == self_pos).any(dim=1)
    return hit.float().mean().item()


def evaluate(proj_img, proj_txt, data, img_rows_subset):
    """对一批图像及其 caption 计算 Recall（候选池 = 这批图像）。

    Args:
        proj_img: 图像投影层。
        proj_txt: 文本投影层。
        data: load_data 返回的字典。
        img_rows_subset: 参与评估的图像行号列表。
    Returns:
        字典，键如 "T2I@1"、"I2T@10"。
    """
    proj_img.eval()
    proj_txt.eval()
    pos = {img_row: p for p, img_row in enumerate(img_rows_subset)}
    txt_rows = []
    gt = []
    for img_row in img_rows_subset:
        for txt_row in data["caption_rows_of"][img_row]:
            txt_rows.append(txt_row)
            gt.append(pos[img_row])  # 这条 caption 的正确图在候选池里的位置

    with torch.no_grad():  # 评估不需要梯度
        img_emb = project_and_normalize(data["img_feat"][img_rows_subset], proj_img)
        txt_emb = project_and_normalize(data["txt_feat"][txt_rows], proj_txt)

    sim = txt_emb @ img_emb.T  # [num_caption, num_img]
    gt = torch.tensor(gt)

    recalls = {}
    for k in (1, 5, 10):
        recalls[f"T2I@{k}"] = recall_t2i(sim, gt, k)
        recalls[f"I2T@{k}"] = recall_i2t(sim, gt, k)
    return recalls


def write_train_log(
    name, hparams, n_params, loss_history, epoch_times, total_time, recalls
):
    """把一次训练的结果写入 data_store 下的独立 txt 文件。

    Args:
        name: 本次实验的名字，会放进文件名。
        hparams: 超参数字典。
        n_params: 可训练参数总数。
        loss_history: 每个 epoch 的平均 loss 列表。
        epoch_times: 每个 epoch 的耗时（秒）列表。
        total_time: 总用时（秒）。
        recalls: Recall 字典，键如 "val_T2I@10"、"train_T2I@10"。
    Returns:
        写入的文件路径。
    """
    os.makedirs(DATA_STORE, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(DATA_STORE, f"train_{stamp}_{name}.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("===== 超参数 =====\n")
        for key, value in hparams.items():
            f.write(f"{key}: {value}\n")
        f.write(f"可训练参数: {n_params}\n")
        f.write(f"总用时: {total_time:.1f}s\n\n")

        f.write("===== 每个 epoch =====\n")
        for i, (loss, t) in enumerate(zip(loss_history, epoch_times), start=1):
            f.write(f"epoch {i}  loss {loss:.4f}  用时 {t:.1f}s\n")

        f.write("\n===== Recall =====\n")
        for key, value in recalls.items():
            f.write(f"{key}: {value:.4f}\n")
    return path


def load_projection(path: str):
    """加载训练好的投影层权重。

    Args:
        path: projection_flickr30k.pth 的路径。

    Returns:
        (proj_img, proj_txt)：两个载入了权重的 nn.Linear。
    """
    ckpt = torch.load(path, map_location="cpu")

    proj_img = nn.Linear(ckpt["proj_img"]["weight"].shape[1], ckpt["embed_dim"])
    proj_txt = nn.Linear(ckpt["proj_txt"]["weight"].shape[1], ckpt["embed_dim"])

    proj_img.load_state_dict(ckpt["proj_img"])
    proj_txt.load_state_dict(ckpt["proj_txt"])

    return proj_img, proj_txt


@torch.no_grad()
def project_and_normalize(features: torch.Tensor, proj: nn.Module) -> torch.Tensor:
    """先投影，再 L2 归一化。

    Args:
        features: [N, 768] 原始特征。
        proj: 投影层（图像侧或文本侧）。

    Returns:
        [N, 256]，每行长度为 1。
    """
    input_layer = next(
        (layer for layer in proj.modules() if isinstance(layer, nn.Linear)), None
    )
    if input_layer is not None and features.shape[-1] != input_layer.in_features:
        raise ValueError(
            f"Feature dimension {features.shape[-1]} does not match projection "
            f"input dimension {input_layer.in_features}"
        )
    out = proj(features)
    out = F.normalize(out, dim=-1)
    return out


def build_eval_set(data, proj_img, proj_txt, img_rows):
    """构造评测用的图库、查询向量和标准答案。

    图库是全部图片投影后的向量；查询是 img_rows 这些图片的所有
    caption 投影后的向量；标准答案是每条 caption 所属图片在图库中的行号。

    Args:
        data: load_data() 返回的字典。
        proj_img: 图像投影层。
        proj_txt: 文本投影层。
        img_rows: 参与评测的图片行号列表，例如 data["val_img_rows"]。

    Returns:
        (gallery, queries, gt) 三元组：
        gallery 形状 [N, 256]，queries 形状 [Q, 256]，gt 形状 [Q]。
    """
    gallery = project_and_normalize(data["img_feat"], proj_img)

    txt_rows = []
    gt = []
    for img_row in img_rows:
        for txt_row in data["caption_rows_of"][img_row]:
            txt_rows.append(txt_row)
            gt.append(img_row)

    queries = project_and_normalize(data["txt_feat"][txt_rows], proj_txt)
    return gallery, queries, torch.tensor(gt)


if __name__ == "__main__":
    # 自测：数据规模应为 [31014, 768] / [155070, 768] / 31014 29000 1014
    d = load_data()
    print(d["img_feat"].shape, d["txt_feat"].shape)
    print(len(d["caption_rows_of"]), len(d["train_img_rows"]), len(d["val_img_rows"]))
    feats = torch.load("output/features/image_features.pth")["features"]
    sim = torch.tensor(
        [
            [0.9, 0.1, 0.3, 0.2],  # 第 0 条文字
            [0.2, 0.5, 0.8, 0.1],  # 第 1 条文字
            [0.4, 0.3, 0.2, 0.7],  # 第 2 条文字
        ]
    )
    gt = torch.tensor([0, 1, 3])  # 每条文字对应的正确图片的列号
    print(ndcg_t2i(sim, gt, k=1))  # 期望约 0.667（三条里两条命中，各得 1，一条得 0）
    print(ndcg_t2i(sim, gt, k=10))  # 期望约 0.877（1.000 + 0.631 + 1.000 取平均）
