import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from datetime import datetime

# 项目根目录 = 本文件（src/train_projection.py）所在目录的上一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# data_store 与「挑战实验」同级，即项目根目录的上一级
DATA_STORE = os.path.join(os.path.dirname(PROJECT_ROOT), "data_store")


IMG_PATH = "output/features/image_features.pth"
TXT_PATH = "output/features/text_features.pth"


def write_train_log(hparams, n_params, loss_history, epoch_times, total_time, recalls):
    """把一次训练的结果写入 data_store 下的独立 txt 文件。

    Args:
        hparams: 超参数字典，例如 {"lr": 0.001, "epochs": 20}。
        n_params: 可训练参数总数。
        loss_history: 每个 epoch 的平均 loss 列表。
        epoch_times: 每个 epoch 的耗时（秒）列表。
        total_time: 总用时（秒）。
    Returns:
        写入的文件路径。
    """
    os.makedirs(DATA_STORE, exist_ok=True)  # 文件夹不存在就创建
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 时间戳，保证文件名唯一
    path = os.path.join(DATA_STORE, f"train_{stamp}.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("===== 超参数 =====\n")
        for name, value in hparams.items():
            f.write(f"{name}: {value}\n")
        f.write(f"可训练参数: {n_params}\n")
        f.write(f"总用时: {total_time:.1f}s\n\n")

        f.write("===== 每个 epoch =====\n")
        # zip 把两个列表按位置配对，enumerate 给出序号（start=1 表示从 1 开始）
        for i, (loss, t) in enumerate(zip(loss_history, epoch_times), start=1):
            # TODO 5-b：写一行，格式如 "epoch 1  loss 4.8123  用时 1.2s"，别忘了换行符
            f.write(f"epoch {i}  loss {loss:.4f}  用时 {t:.1f}s\n")
        f.write("\n===== val Recall =====\n")
        # TODO 10-a：遍历 recalls 字典，每行写 "名字: 数值"，数值保留 4 位小数
        for name, value in recalls.items():
            f.write(f"{name}: {value:.4f}\n")
    return path


def recall_i2t(sim, gt, k):
    """图搜文的 Recall@K：前 K 名 caption 里是否有属于这张图的。

    Args:
        sim: 相似度矩阵 [5070, 1014]（行是 caption，列是图）。
        gt: 每条 caption 对应的图位置 [5070]。
        k: 取前几名。
    Returns:
        命中率（0 到 1 的浮点数）。
    """
    topk_idx = sim.T.topk(k, dim=1).indices

    pred_img = gt[topk_idx]

    self_pos = torch.arange(sim.shape[1]).unsqueeze(1)

    hit = (pred_img == self_pos).any(dim=1)
    return hit.float().mean().item()


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


def recall_t2i(sim, gt, k):
    """文搜图的 Recall@K：正确图片是否排进前 K 名。

    Args:
        sim: 相似度矩阵 [5070, 1014]，行是 caption，列是图片。
        gt: 每条 caption 的正确图片列号 [5070]。
        k: 取前几名。
    Returns:
        命中率（0 到 1 的浮点数）。
    """
    topk_idx = sim.topk(k, dim=1).indices

    hit = (topk_idx == gt.unsqueeze(1)).any(dim=1)  # 每行是否命中，形状 [5070]

    return hit.float().mean().item()


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

# ======超参数======
EMBED_DIM = 256
LR = 1e-3
EPOCHS = 20
BATCH_SIZE = 128
BASE_SEED = 42
TEMPERATURE = 0.07
SAVE_MODEL = True

torch.manual_seed(BASE_SEED)
proj_img = nn.Linear(768, EMBED_DIM)
proj_txt = nn.Linear(768, EMBED_DIM)

optimizer = torch.optim.Adam(
    list(proj_img.parameters()) + list(proj_txt.parameters()),
    lr=LR,
)

n_params = sum(
    p.numel() for p in list(proj_img.parameters()) + list(proj_txt.parameters())
)
print("可训练参数:", n_params)

hparams = {
    "lr": LR,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "embed_dim": EMBED_DIM,
    "base_seed": BASE_SEED,
    "temperature": TEMPERATURE,
    "optimizer": "Adam",
}

loss_history = []  # 每个 epoch 的平均 loss
epoch_times = []  # 每个 epoch 的耗时（秒）
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()  # 这个 epoch 开始的时刻

    batches = make_epoch_batches(
        train_img_rows, caption_rows_of, BATCH_SIZE, BASE_SEED + epoch
    )

    epoch_loss_sum = 0.0
    for batch in batches:
        img_rows = [img_row for img_row, txt_row in batch]
        txt_rows = [txt_row for img_row, txt_row in batch]
        img_in = img_feat[img_rows]
        txt_in = txt_feat[txt_rows]

        img_emb = F.normalize(proj_img(img_in), dim=1)
        txt_emb = F.normalize(proj_txt(txt_in), dim=1)
        loss = info_nce_loss(img_emb, txt_emb, TEMPERATURE)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss_sum += loss.item()  # item() 取出普通数字，不保留计算图

    avg_loss = epoch_loss_sum / len(batches)
    loss_history.append(avg_loss)
    epoch_times.append(time.time() - epoch_start)
    print(
        f"epoch {epoch + 1}/{EPOCHS}  loss {avg_loss:.4f}  用时 {epoch_times[-1]:.1f}s"
    )

total_time = time.time() - start_time

# ===== 验证集评估 =====
val_img_rows = [i for i, s in enumerate(img_data["splits"]) if s == "val"]

val_pos = {img_row: p for p, img_row in enumerate(val_img_rows)}

val_txt_rows = []
val_txt_gt = []
for img_row in val_img_rows:
    for txt_row in caption_rows_of[img_row]:
        val_txt_rows.append(txt_row)
        val_txt_gt.append(val_pos[img_row])

with torch.no_grad():
    val_img_emb = F.normalize(proj_img(img_feat[val_img_rows]), dim=1)
    val_txt_emb = F.normalize(proj_txt(txt_feat[val_txt_rows]), dim=1)

# 每条 caption 对每张图的相似度
sim = val_txt_emb @ val_img_emb.T
print(len(val_img_rows), len(val_txt_rows))
print(val_img_emb.shape, val_txt_emb.shape, sim.shape)

val_txt_gt = torch.tensor(val_txt_gt)  # [5070]

recalls = {}
for k in (1, 5, 10):
    recalls[f"T2I@{k}"] = recall_t2i(sim, val_txt_gt, k)
    recalls[f"I2T@{k}"] = recall_i2t(sim, val_txt_gt, k)
print(recalls)

log_path = write_train_log(
    hparams, n_params, loss_history, epoch_times, total_time, recalls
)
print("日志:", log_path)

if SAVE_MODEL:
    save_path = os.path.join(
        PROJECT_ROOT, "output", "models", "projection_flickr30k.pth"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(
        {
            "proj_img": proj_img.state_dict(),
            "proj_txt": proj_txt.state_dict(),
            "embed_dim": EMBED_DIM,
        },
        save_path,
    )
    print("权重已保存:", save_path)
