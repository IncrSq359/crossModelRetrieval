"""提取 Flickr30k 全部图像的 ConvNeXt 特征并缓存到磁盘。"""
import csv
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[1]
CSV_PATH=PROJECT_ROOT/"data"/"annotations"/"captions_flickr30k.csv"

def load_unique_images(csv_path):
    """从标注 CSV 中按图片去重，保持首次出现的顺序。

    Args:
        csv_path: captions_flickr30k.csv 的路径。
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
        
if __name__=="__main__":
    filenames,splits=load_unique_images(CSV_PATH)
    print("图片数：", len(filenames))
    print("前3张：", filenames[:3], splits[:3])