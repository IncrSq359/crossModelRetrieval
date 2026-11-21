import random

import torch
import torch.nn.functional as F

IMG_PATH = "output/features/image_features.pth"
TXT_PATH = "output/features/text_features.pth"


def make_epoch_batches(train_img_rows, caption_rows_of, batch_size, seed):
    """为一个 epoch 生成 batch：每个 batch 里每张图最多一条 caption。

    Args:
        train_img_rows: 训练集的图像行号列表（29000 个）。
        caption_rows_of: 图像行号 -> 该图 5 条 caption 行号的字典。
        batch_size: 每个 batch 的大小。
        seed: 随机种子，每个 epoch 传不同的值。
    Returns:
        batches: 列表，每个元素是一个 batch，里面是 batch_size 个 (图像行号, caption行号)。
    """
    rng = random.Random(seed)
    pairs = []
    for img_row in train_img_rows:
        txt_row = rng.choice(caption_rows_of[img_row])
        pairs.append((img_row, txt_row))

    rng.shuffle(pairs)

    batches = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        if len(batch) == batch_size:
            batches.append(batch)
    return batches


def info_nce_loss(img_emb, txt_emb, temperature=0.07):
    """双向 InfoNCE 损失。

    Args:
        img_emb: 图像向量 [B, 256]，已 L2 归一化。
        txt_emb: 文本向量 [B, 256]，已 L2 归一化。
        temperature: 温度系数。
    Returns:
        标量损失。
    """
    logits = img_emb @ txt_emb.T / temperature
    labels = torch.arange(img_emb.shape[0])
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


img_data = torch.load(IMG_PATH, map_location="cpu")
txt_data = torch.load(TXT_PATH, map_location="cpu")
img_feat = img_data["features"]  # [31014, 768]
txt_feat = txt_data["features"]  # [155070, 768]

# 文件名 -> 图像特征行号
img_index = {fn: i for i, fn in enumerate(img_data["filenames"])}

# 每条 caption 对应的图像行号 [155070]
rows = []
for fn in txt_data["filenames"]:
    rows.append(img_index[fn])
txt_img_rows = torch.tensor(rows)

# 训练集的图像行号（29000 张）
train_img_rows = []
for i, s in enumerate(img_data["splits"]):
    if s == "train":
        train_img_rows.append(i)

# 图像行号 -> 该图 5 条 caption 的行号
caption_rows_of = {}
for txt_row, img_row in enumerate(txt_img_rows.tolist()):
    caption_rows_of.setdefault(img_row, []).append(txt_row)