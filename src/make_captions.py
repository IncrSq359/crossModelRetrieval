"""生成"多样化"图片描述(caption)的脚本。

与 exp1 的单一模板不同：这里为每张图准备多种句式，并保证同一类别的
10 张图描述互不重复，以缓解 In-Batch Negatives 中的"假负样本"问题。
输出仍是 data/texts/captions.csv，表头 image,text,label。
"""

import csv
import random
from data_loader import get_image_data   # 复用：扫描图片，拿到 path 和 label

# 模板池：数量必须 >= 每类图片数(10)，这里给12个。
# {label} 会被替换成类别词，{art} 会被替换成 a / an。
CAPTION_TEMPLATES = [
    "a photo of {art} {label}",
    "a close-up photo of {art} {label}",
    "a picture showing {art} {label}",
    "an image of {art} {label}",
    "{art} {label} in a photograph",
    "a photograph featuring {art} {label}",
    "a clear shot of {art} {label}",
    "{art} {label} seen up close",
    "there is {art} {label} in this picture",
    "a camera capture of {art} {label}",
    "this picture contains {art} {label}",
    "a view of {art} {label}",
]


def pick_article(label):
    """根据类别词首字母返回不定冠词 a / an。

    元音字母(a,e,i,o,u)开头用 an，否则用 a。
    """
    return "an" if label[0] in "aeiou" else "a"


def make_captions(image_dir, csv_path, per_class=10, seed=42):
    """为每张图片生成一句互不雷同的描述并写入 CSV。

    Args:
        image_dir: 图片根目录（如 "data/images"）。
        csv_path: 输出 CSV 路径（如 "data/texts/captions.csv"）。
        per_class: 每个类别有多少张图，决定要抽几个不同模板，默认10。
        seed: 随机种子，固定后结果可复现，默认42。
    """
    random.seed(seed)
    records = get_image_data(image_dir)

    # 第1步：按标签把记录分组 → {类别名: [该类的记录,...]}
    grouped = {}
    for rec in records:
        grouped.setdefault(rec["label"], []).append(rec)

    rows = []
    # 第2步：逐类处理
    for label, recs in grouped.items():
        pretty_label = label.replace("_", " ")          # dollar_bill -> dollar bill
        art = pick_article(pretty_label)                # 决定 a / an

        # 从模板池"不放回"抽取 per_class 个，保证同类各句不同
        chosen_templates = random.sample(CAPTION_TEMPLATES, per_class)

        # 把模板和图片一一配对，生成一行行数据
        for rec, template in zip(recs, chosen_templates):
            text = template.format(art=art, label=pretty_label)
            rows.append({
                "image": str(rec["path"]),
                "text": text,
                "label": label,
            })

    # 第3步：写入 CSV（newline='' 是 Windows 下避免空行的标准写法）
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"已生成 {len(rows)} 条多样化描述 → {csv_path}")


if __name__ == "__main__":
    make_captions(
        image_dir="data/images",
        csv_path="data/texts/captions.csv",
        per_class=10,
        seed=42,
    )
