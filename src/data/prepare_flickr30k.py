# -*- coding: utf-8 -*-
"""
实验三数据准备脚本：下载并整理 Flickr30k 大规模图文数据集。
=====================================================================

为什么选 Flickr30k：
    - exp1/exp2 使用的 Caltech-101 只有图片、没有真实文本，caption 是用
      类别名套模板“人造”的；实验三要做大规模真实场景下的检索与 ANN 索引，
      必须换成自带人工标注的图文数据集。
    - Flickr30k 包含 31014 张图片，每张图片配有 5 条独立的人工英文描述
      （共 155070 条 caption），并带学术界通用的 Karpathy train/val/test 划分。

数据来源（HuggingFace 数据集仓库，镜像站免鉴权）：
    https://hf-mirror.com/datasets/nlphuji/flickr30k
    - flickr30k-images.zip        全部图片压缩包（约 4.4GB）
    - flickr_annotations_30k.csv  官方标注（约 13MB）

脚本分三步，每一步都可重复执行（已完成的部分会自动跳过，下载支持断点续传）：
    1. download : 下载压缩包与官方标注 CSV
    2. extract  : 解压图片到 data/raw/flickr30k/flickr30k-images/
    3. annotate : 把“一行一张图、caption 塞在 JSON 列表里”的官方标注，
                  展开成“一行一条 caption”的标准化 CSV：
                      data/annotations/captions_flickr30k.csv
                  列定义：filename, img_id, split, caption_idx, caption

直接运行整个流程：
    python src/prepare_flickr30k.py
只运行其中某几步（例如图片已解压，只需重建标注）：
    python src/prepare_flickr30k.py annotate verify
"""

import argparse          # 标准库：解析命令行参数
import csv               # 标准库：读写 CSV，无需安装 pandas
import json              # 标准库：解析标注中的 JSON 字符串（caption 列表）
import os                # 标准库：读取/设置环境变量、路径判断
import sys               # 标准库：在导入 huggingface_hub 前退出报错等
import zipfile           # 标准库：解压 zip
from pathlib import Path  # 标准库：面向对象的路径处理，比拼字符串更安全

# ---------------------------------------------------------------------
# 环境变量必须在 `import huggingface_hub` 之前设置才会生效：
#   HF_ENDPOINT          把下载站点指到国内镜像，避免直连 huggingface.co 超时
#   HF_HUB_DISABLE_XET=1 关闭新版 xet 传输通道，强制走普通 HTTPS 断点续传，
#                        对镜像站的兼容性最好
# 若用户已在系统里设置过同名环境变量，则保留用户的值，不强行覆盖。
# ---------------------------------------------------------------------
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import hf_hub_download  # HuggingFace 官方下载工具（自动断点续传）

# ---------------------------------------------------------------------
# 路径常量：以本文件位置（src/prepare_flickr30k.py）反推项目根目录，
# 这样无论从哪个目录运行脚本，路径都不会错。
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]          # 项目根目录
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "flickr30k"       # 原始下载物目录
IMAGES_DIR = RAW_DIR / "flickr30k-images"                   # 解压后的图片目录
RAW_CSV = RAW_DIR / "flickr_annotations_30k.csv"            # 官方标注 CSV
ZIP_PATH = RAW_DIR / "flickr30k-images.zip"                 # 图片压缩包
OUT_CSV = PROJECT_ROOT / "data" / "annotations" / "captions_flickr30k.csv"  # 标准化标注输出

# HuggingFace 数据集仓库标识
REPO_ID = "nlphuji/flickr30k"
REPO_TYPE = "dataset"
# 仓库中的两个文件名
ZIP_NAME = "flickr30k-images.zip"
ANNO_NAME = "flickr_annotations_30k.csv"

# Flickr30k 官方图片总数（用于解压后核对；压缩包里比有标注的图片略多几张）
EXPECTED_TOTAL_IMAGES = 31783
# 带标注的图片数 / caption 数（官方 Karpathy 划分的标准规模）
EXPECTED_ANNOTATED_IMAGES = 31014
EXPECTED_CAPTIONS = 155070
# Karpathy 划分中各 split 的图片数（train/val/test）
EXPECTED_SPLIT_COUNTS = {"train": 29000, "val": 1014, "test": 1000}


def step_download() -> None:
    """第一步：从镜像站下载官方标注 CSV（约 13MB）和图片压缩包（约 4.4GB）。

    hf_hub_download 自带两个关键能力：
      1. 本地已存在且校验通过时直接返回路径（不重复下载）；
      2. 大文件下载中断后再次运行会从断点继续。
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)  # 确保下载目录存在

    print("[1/4] 下载 Flickr30k 文件（来源：HF 镜像 %s）" % os.environ["HF_ENDPOINT"])
    for name in (ANNO_NAME, ZIP_NAME):  # 先下小文件，尽早暴露网络/权限问题
        print("  -> 开始获取 %s ..." % name)
        # local_dir 指定落盘目录；下载完成后文件就在 data/raw/flickr30k/ 下
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=name,
            repo_type=REPO_TYPE,
            local_dir=str(RAW_DIR),
        )
        size_gb = Path(path).stat().st_size / (1024 ** 3)
        print("     完成：%s（%.2f GB）" % (Path(path).name, size_gb))


def step_extract() -> None:
    """第二步：把图片压缩包解压到 data/raw/flickr30k/ 下。

    压缩包内部自带顶层文件夹 flickr30k-images/，解压后图片路径形如
    data/raw/flickr30k/flickr30k-images/xxx.jpg。
    若检测到图片目录已存在且图片数量达标，则跳过解压（避免重复解压 3 万张图）。
    """
    print("[2/4] 解压图片压缩包")
    if not ZIP_PATH.exists():
        sys.exit("错误：找不到压缩包 %s，请先运行 download 步骤。" % ZIP_PATH)

    # glob("*.jpg") 统计已解压的 jpg 数量，作为“是否已解压”的判据
    existing = list(IMAGES_DIR.glob("*.jpg")) if IMAGES_DIR.exists() else []
    if len(existing) >= EXPECTED_ANNOTATED_IMAGES:
        print("  -> 已检测到 %d 张图片，跳过解压。" % len(existing))
        return

    print("  -> 正在解压（约 3.2 万张图片，请耐心等待）...")
    with zipfile.ZipFile(ZIP_PATH, mode="r") as zf:
        # 安全检查：压缩包不应含有绝对路径或 ../ 之类的越界路径
        for member in zf.namelist():
            if member.startswith(("/", "\\")) or ".." in Path(member).parts:
                sys.exit("错误：压缩包中存在可疑路径，已中止：%s" % member)
        zf.extractall(RAW_DIR)  # 解压到 raw 目录，自动生成 flickr30k-images 子目录
    n = len(list(IMAGES_DIR.glob("*.jpg")))
    print("  -> 解压完成，共得到 %d 张 jpg 图片。" % n)


def step_annotate() -> None:
    """第三步：把官方标注展开为“一行一条 caption”的标准化 CSV。

    官方 CSV 中每张图片占一行，5 条 caption 以 JSON 字符串形式塞在 raw 列里，
    形如：["A man...","Two dogs...", ...]。
    训练/检索时按“图文对”取样，一行一条 caption 的形式更方便直接使用。
    """
    print("[3/4] 生成标准化标注 CSV")
    if not RAW_CSV.exists():
        sys.exit("错误：找不到官方标注 %s，请先运行 download 步骤。" % RAW_CSV)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # newline="" 是 Python csv 模块在 Windows 上的标准写法，避免空行问题
    with open(RAW_CSV, "r", encoding="utf-8", newline="") as fin, \
         open(OUT_CSV, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)  # 按表头把每行读成字典
        # 输出列：文件名、图片id、所属划分、第几条caption、caption 文本
        writer = csv.DictWriter(
            fout,
            fieldnames=["filename", "img_id", "split", "caption_idx", "caption"],
        )
        writer.writeheader()

        n_images = 0       # 图片（行）计数
        n_captions = 0     # caption 计数
        for row in reader:
            captions = json.loads(row["raw"])  # raw 列是 JSON 列表字符串 -> Python list
            filename = row["filename"].strip()
            # img_id 列若存在则直接用，否则用文件名去掉后缀作为 id（做了兼容）
            img_id = (row.get("img_id") or Path(filename).stem).strip()
            split = row["split"].strip()       # train / val / test

            for idx, text in enumerate(captions):  # 每张图的 5 条描述各写一行
                writer.writerow({
                    "filename": filename,
                    "img_id": img_id,
                    "split": split,
                    "caption_idx": idx,
                    "caption": text.strip(),
                })
                n_captions += 1
            n_images += 1

            # 每处理 1 万张图打印一次进度
            if n_images % 10000 == 0:
                print("     已处理 %d 张图片 ..." % n_images)

    print("  -> 输出：%s" % OUT_CSV)
    print("     图片 %d 张，caption %d 条。" % (n_images, n_captions))


def step_verify() -> None:
    """第四步：核对数据集完整性与划分规模。

    检查项：
      1. 标准化标注的图片数 / caption 总数是否等于官方规模；
      2. 每张图片是否恰好 5 条 caption；
      3. train/val/test 图片数是否与 Karpathy 标准划分一致；
      4. 标注引用的每张图片文件是否真实存在于解压目录。
    """
    print("[4/4] 校验数据完整性")
    if not OUT_CSV.exists():
        sys.exit("错误：找不到标准化标注 %s，请先运行 annotate 步骤。" % OUT_CSV)

    per_image_count = {}   # filename -> 该图片的 caption 条数
    split_count = {}       # split -> 图片数（用集合去重）
    split_images = {"train": set(), "val": set(), "test": set()}
    total_rows = 0

    with open(OUT_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            fn = row["filename"]
            per_image_count[fn] = per_image_count.get(fn, 0) + 1
            split_images[row["split"]].add(fn)

    # --- 数值核对 ---
    n_images = len(per_image_count)
    bad_caption_num = {fn: c for fn, c in per_image_count.items() if c != 5}
    print("  图片数：%d（期望 %d）" % (n_images, EXPECTED_ANNOTATED_IMAGES))
    print("  caption 总数：%d（期望 %d）" % (total_rows, EXPECTED_CAPTIONS))
    print("  caption 数不等于 5 的图片：%d 张" % len(bad_caption_num))

    ok = (n_images == EXPECTED_ANNOTATED_IMAGES
          and total_rows == EXPECTED_CAPTIONS
          and not bad_caption_num)

    # --- 划分核对 ---
    for split in ("train", "val", "test"):
        actual = len(split_images[split])
        expect = EXPECTED_SPLIT_COUNTS[split]
        mark = "OK" if actual == expect else "!!"
        print("  split=%-5s 图片数：%5d（期望 %5d） [%s]" % (split, actual, expect, mark))
        split_count[split] = actual
        ok = ok and (actual == expect)

    # --- 图片文件核对（只在图片已解压时进行）---
    if IMAGES_DIR.exists():
        missing = [fn for fn in per_image_count if not (IMAGES_DIR / fn).exists()]
        n_jpg = len(list(IMAGES_DIR.glob("*.jpg")))
        print("  解压目录 jpg 文件数：%d" % n_jpg)
        print("  标注引用但磁盘缺失的图片：%d 张" % len(missing))
        if missing[:3]:
            print("    示例：%s" % ", ".join(missing[:3]))
        ok = ok and not missing
    else:
        print("  （图片尚未解压，跳过图片文件存在性检查）")

    print("=" * 50)
    print("校验结果：%s" % ("全部通过 ✅" if ok else "存在异常 ❌，请检查上面的 !! 项"))


# 各步骤的注册表：命令行用名字选择要执行的步骤
STEPS = {
    "download": step_download,
    "extract": step_extract,
    "annotate": step_annotate,
    "verify": step_verify,
}


def main() -> None:
    """命令行入口：默认按顺序执行全部步骤，也可只执行指定步骤。"""
    parser = argparse.ArgumentParser(description="下载并整理 Flickr30k 图文数据集")
    parser.add_argument(
        "steps",
        nargs="*",
        choices=list(STEPS.keys()),
        # 注意：这里不能把默认值直接写成 list —— 在部分运行环境下该列表会被
        # 当成一个字符串参数传入而报错；默认值放到 parse_args 之后用 or 补全。
        default=None,
        help="要执行的步骤（默认全部按顺序执行）",
    )
    args = parser.parse_args()
    # 不带任何参数时，args.steps 为 None -> 执行全部步骤
    steps = args.steps or list(STEPS.keys())

    print("项目根目录：%s" % PROJECT_ROOT)
    for name in steps:
        STEPS[name]()  # 按命令行给出的顺序依次调用
    print("全部步骤执行完毕。")


if __name__ == "__main__":
    main()
