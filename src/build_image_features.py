"""提取 Flickr30k 全部图像的 ConvNeXt 特征并缓存到磁盘。"""
from ast import Dict
import csv
from pathlib import Path
from sys import path
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT=Path(__file__).resolve().parents[1]
CSV_PATH=PROJECT_ROOT/"data"/"annotations"/"captions_flickr30k.csv"
IMAGES_DIR=PROJECT_ROOT/"data"/"raw"/"flickr30k"/"flickr30k-images"

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
 
class ImageDataset(Dataset):
    """按文件名列表读取图片并做预处理，只返回图像张量。"""
    def __init__(self,filenames,image_dir,transform):
        """
        Args:
            filenames: 去重后的图片文件名列表。
            images_dir: 图片所在目录。
            transform: 图像预处理流水线（来自 preprocess.py）。
        """
        self.filenames=filenames
        self.image_dir=Path(image_dir)
        self.transform=transform
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        image_path=self.image_dir/self.filenames[idx]
        image=Image.open(image_path).convert("RGB")
        return self.transform(image)
        
def extract_image_features(model,dataset,batch_size=32):
    """用冻结的 ConvNeXt 批量提取图像特征。

    Args:
        model: load_convnext 返回的冻结图像编码器，输出 [B, 768]。
        dataset: ImageDataset，每个样本是 [3, 224, 224] 的张量。
        batch_size: 每批图片数。
    Returns:
        特征矩阵 [len(dataset), 768]，行顺序与 dataset 中的图片顺序一致。
    """
    loader= DataLoader(dataset,batch_size=batch_size,shuffle=False)
    all_feats=[]
    with torch.no_grad():
        for i,images in enumerate(loader):
            feats=model(images)
            all_feats.append(feats)
            if (i+1)%20==0:
                print("已处理批数：",i+1,"/",len(loader))
    return torch.cat(all_feats,dim=0)
    
def save_features(features, filenames, splits, out_path):
    """把特征矩阵和对应的元信息一起保存。

    Args:
        features: 特征矩阵 [N, 768]。
        filenames: 与矩阵行一一对应的文件名列表。
        splits: 与矩阵行一一对应的划分列表。
        out_path: 输出文件路径（.pth）。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)   # 目录不存在就创建
    torch.save({"features":features,"filenames":filenames,"splits":splits
        },
        out_path)
    
if __name__=="__main__":
    from preprocess import get_image_transform
    from model import load_convnext
    
    filenames,splits=load_unique_images(CSV_PATH)
    ds=ImageDataset(filenames,IMAGES_DIR,get_image_transform())
    model=load_convnext(PROJECT_ROOT/"data"/"raw"/"convnext_tiny_1k_224_ema.pth")
    
    feats=extract_image_features(model,ds)
    print("特征形状：",feats.shape)
    
    out_path=PROJECT_ROOT/"output"/"features"/"image_features.pth"
    save_features(feats,filenames,splits,out_path)