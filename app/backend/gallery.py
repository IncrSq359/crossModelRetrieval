"""图库模块：负责文件名与 caption 的对应关系。"""
import csv
import torch
from torch.nn import functional as F

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

def build_filename_index(filenames):
    """建立 文件名 -> 特征矩阵行号 的映射。

    Args:
        filenames: 文件名列表，第 i 个元素对应特征矩阵第 i 行。
    Returns:
        dict：键是文件名，值是行号（从 0 开始）。
    """
    return {fn : i for i, fn in enumerate(filenames)}

def project_gallery(features,proj_img):
    """把图库的 2048 维原始特征投影成 256 维单位向量矩阵。

    Args:
        features: 图库原始特征 [N, 2048]。
        proj_img: 图像投影层（与查询侧是同一个模块）。
    Returns:
        图库矩阵 [N, 256]，每行是单位向量。
    """
    with torch.no_grad():
        emb=proj_img(features)
        emb=F.normalize(emb,dim=1,p=2)
    return emb