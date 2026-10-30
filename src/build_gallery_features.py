"""图库特征提取：把全部图库图片过一遍 ResNet，缓存 2048 维原始特征。"""
import csv

def list_gallery_images(csv_path):
    """从标注 CSV 中按首次出现顺序取出不重复的图片文件名及其划分。

    Args:
        csv_path: 标注 CSV 路径。
    Returns:
        (filenames, splits)：两个等长列表，第 i 项对应同一张图。
    """
    filenames=[]
    splits=[]
    seen=set()
    with open(csv_path,"r",encoding="utf-8",newline="")as f:
        reader=csv.DictReader(f)
        for row in reader:
            fn=row["filename"]
            if fn in seen:
                continue
            seen.add(fn)
            filenames.append(fn)
            splits.append(row["split"])
    return filenames,splits