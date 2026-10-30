"""图库特征提取：把全部图库图片过一遍 ResNet，缓存 2048 维原始特征。"""
import csv
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

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

class GalleryImageDataset(Dataset):
    """图库图片数据集：按给定的文件名顺序，逐张读图并预处理。"""

    def __init__(self, images_dir, filenames, transform):
        """
        Args:
            images_dir: 图片所在目录。
            filenames: 文件名列表，顺序即特征矩阵的行顺序。
            transform: 图像预处理流水线（与查询侧同一个函数产生）。
        """
        self.images_dir = Path(images_dir)
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)                 

    def __getitem__(self, idx):
        image_path = self.images_dir/self.filenames[idx]           
        image = Image.open(image_path).convert("RGB") 
        return self.transform(image)   