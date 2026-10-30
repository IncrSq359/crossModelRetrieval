"""图库特征提取：把全部图库图片过一遍 ResNet，缓存 2048 维原始特征。"""
import csv
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torch
from preprocess import get_image_transform
from model import load_resnet

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

def extract_features(resnet, dataset, batch_size=32, num_workers=0):
    """批量提取图库图片的 ResNet 原始特征。

    Args:
        resnet: load_resnet 返回的冻结 ResNet（输出 2048 维）。
        dataset: GalleryImageDataset，顺序即特征矩阵的行顺序。
        batch_size: 每批图片数。
        num_workers: 读图子进程数，0 表示在主进程中读。
    Returns:
        特征矩阵 [N, 2048]，第 i 行对应 dataset 的第 i 张图。
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    chunks = []
    with torch.no_grad():
        for images in loader:
            feat = resnet(images)
            chunks.append(feat)
    return torch.cat(chunks, dim=0)

def main(limit=None):
    """脚本入口：提取图库全部特征，并按契约格式保存。

    Args:
        limit: 只处理前 limit 张图，用于快速测试；None 表示全部。
    """
    csv_path = "data/annotations/captions_flickr30k.csv"
    images_dir = "data/raw/flickr30k/flickr30k-images"
    resnet_path = "data/raw/resnet50_timm.pth"
    out_path = Path("output/features/image_features.pth")

    filenames, splits = list_gallery_images(csv_path)
    if limit is not None:            
        filenames, splits = filenames[:limit], splits[:limit]
    dataset = GalleryImageDataset(images_dir, filenames, get_image_transform())
    resnet = load_resnet(resnet_path)
    features = extract_features(resnet, dataset, batch_size=32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "filenames": filenames, "splits": splits}, out_path)
    print("saved:", features.shape, "->", out_path)


if __name__ == "__main__":
    main()