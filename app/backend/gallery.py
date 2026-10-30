"""图库模块：负责文件名与 caption 的对应关系。"""
import csv

def load_captions(csv_path):
    """读取标注 CSV，按图片文件名归拢 caption。

    Args:
        csv_path: 标注 CSV 路径。
    Returns:
        dict：键是文件名，值是该图的 caption 列表（共 5 条）。
    """
    captions={}
    with open(csv_path,"r",encoding ="utf-8",newline="") as f:
        reader=csv.DictReader(f)
        for row in reader:
            captions.setdefault(row["filename"],[]).append(row["caption"])
    return captions