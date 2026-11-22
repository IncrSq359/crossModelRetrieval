"""投影层调参脚本：在 CONFIGS 里加配置，循环训练，每组各写一份日志。

运行（在项目根目录）：python src/tune_projection.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from proj_lib import (
    PROJECT_ROOT,
    evaluate,
    info_nce_loss,
    load_data,
    make_epoch_batches,
    write_train_log,
)

# 默认超参数（基线）。CONFIGS 里每组只需写「和基线不同的项」。
BASE = {
    "name": "baseline",
    "lr": 1e-3,
    "epochs": 20,
    "batch_size": 128,
    "temperature": 0.07,
    "weight_decay": 0.0,
    "embed_dim": 256,
    "base_seed": 42,
    "save": False,  # 调参期间不要覆盖已保存的权重
}

# 要对比的配置：每次只改一个变量，才能看出是谁起的作用
CONFIGS = [
    {"name": "baseline"},
    {"name": "lr3e-3", "lr": 3e-3},
    {"name": "temp0.03", "temperature": 0.03},
    {"name": "epochs40", "epochs": 40},
]


def train(hp, data):
    """按一组超参数训练两个线性投影层，并在 val 和 train 子集上评估。

    Args:
        hp: 完整的超参数字典（已和 BASE 合并）。
        data: load_data 返回的字典。
    Returns:
        recalls 字典，键如 "val_T2I@10"、"train_T2I@10"。
    """
    img_feat, txt_feat = data["img_feat"], data["txt_feat"]
    in_dim_img, in_dim_txt = img_feat.shape[1], txt_feat.shape[1]

    torch.manual_seed(hp["base_seed"])  # 固定初始化，保证结果可复现
    proj_img = nn.Linear(in_dim_img, hp["embed_dim"])
    proj_txt = nn.Linear(in_dim_txt, hp["embed_dim"])
    params = list(proj_img.parameters()) + list(proj_txt.parameters())
    n_params = sum(p.numel() for p in params)

    optimizer = torch.optim.Adam(params, lr=hp["lr"], weight_decay=hp["weight_decay"])

    loss_history = []
    epoch_times = []
    start_time = time.time()

    for epoch in range(hp["epochs"]):
        epoch_start = time.time()
        # 每个 epoch 换 seed，保证每张图 5 条 caption 都有机会被抽到
        batches = make_epoch_batches(
            data["train_img_rows"],
            data["caption_rows_of"],
            hp["batch_size"],
            hp["base_seed"] + epoch,
        )

        loss_sum = 0.0
        for batch in batches:
            img_rows = [p[0] for p in batch]
            txt_rows = [p[1] for p in batch]

            img_emb = F.normalize(proj_img(img_feat[img_rows]), dim=1)
            txt_emb = F.normalize(proj_txt(txt_feat[txt_rows]), dim=1)
            loss = info_nce_loss(img_emb, txt_emb, hp["temperature"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        avg_loss = loss_sum / len(batches)
        loss_history.append(avg_loss)
        epoch_times.append(time.time() - epoch_start)
        print(f"[{hp['name']}] epoch {epoch + 1}/{hp['epochs']}  loss {avg_loss:.4f}")

    total_time = time.time() - start_time

    # 评估：val（1014 张）和 train 前 1014 张（候选池规模相同，才可比）
    val_rows = data["val_img_rows"]
    train_rows = data["train_img_rows"][: len(val_rows)]
    recalls = {}
    for split_name, rows in (("val", val_rows), ("train", train_rows)):
        for key, value in evaluate(proj_img, proj_txt, data, rows).items():
            recalls[f"{split_name}_{key}"] = value

    log_path = write_train_log(
        hp["name"], hp, n_params, loss_history, epoch_times, total_time, recalls
    )
    print("日志:", log_path)

    if hp["save"]:
        save_path = os.path.join(
            PROJECT_ROOT, "output", "models", "projection_flickr30k.pth"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "proj_img": proj_img.state_dict(),
                "proj_txt": proj_txt.state_dict(),
                "embed_dim": hp["embed_dim"],
            },
            save_path,
        )
        print("权重已保存:", save_path)

    return recalls


def main():
    """加载一次数据，依次跑 CONFIGS 里的每组配置，最后打印对比表。"""
    data = load_data()  # 数据只加载一次，所有配置共用
    results = []
    for cfg in CONFIGS:
        hp = {**BASE, **cfg}  # 后面的覆盖前面的：基线 + 本组改动
        results.append((hp["name"], train(hp, data)))

    # 汇总对比表
    cols = ["val_T2I@10", "val_I2T@10", "train_T2I@10", "train_I2T@10"]
    print("\n" + "name".ljust(14) + "".join(c.rjust(14) for c in cols))
    for name, rec in results:
        print(name.ljust(14) + "".join(f"{rec[c]:14.4f}" for c in cols))


if __name__ == "__main__":
    main()
