"""投影层训练的共享模块：数据加载、采样器、损失、Recall 评估、日志。

这个文件里只放「不随超参数变化」的东西，调参脚本 tune_projection.py 会 import 它。
"""

import os
import random
from datetime import datetime

import torch.nn as nn
import torch
import torch.nn.functional as F

# 项目根目录 = 本文件（src/proj_lib.py）所在目录的上一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# data_store 与「挑战实验」同级，即项目根目录的上一级
DATA_STORE = os.path.join(os.path.dirname(PROJECT_ROOT), "data_store")

IMG_PATH = os.path.join(PROJECT_ROOT, "output", "features", "image_features.pth")
TXT_PATH = os.path.join(PROJECT_ROOT, "output", "features", "text_features.pth")


def load_data(img_path=IMG_PATH, txt_path=TXT_PATH):
    """加载缓存特征，并建好训练 / 评估要用的索引。

    Args:
        img_path: 图像特征文件路径。
        txt_path: 文本特征文件路径。
    Returns:
        字典，键为：
            img_feat: 图像特征 [31014, 768]。
            txt_feat: 文本特征 [155070, 768]。
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

    return {
        "img_feat": img_data["features"],
        "txt_feat": txt_data["features"],
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
        img_emb = F.normalize(proj_img(data["img_feat"][img_rows_subset]), dim=1)
        txt_emb = F.normalize(proj_txt(data["txt_feat"][txt_rows]), dim=1)

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

    proj_img = nn.Linear(768, ckpt["embed_dim"])
    proj_txt = nn.Linear(768, ckpt["embed_dim"])

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
    out = proj(features)
    out = F.normalize(out, dim=-1)
    return out


if __name__ == "__main__":
    # 自测：数据规模应为 [31014, 768] / [155070, 768] / 31014 29000 1014
    d = load_data()
    print(d["img_feat"].shape, d["txt_feat"].shape)
    print(len(d["caption_rows_of"]), len(d["train_img_rows"]), len(d["val_img_rows"]))
    feats = torch.load("output/features/image_features.pth")["features"]
