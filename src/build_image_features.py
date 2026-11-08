"""提取 Flickr30k 全部图像的 ConvNeXt 特征并缓存到磁盘。"""
import csv
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

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
        
if __name__=="__main__":
    filenames,splits=load_unique_images(CSV_PATH)
    print("图片数：", len(filenames))
    print("前3张：", filenames[:3], splits[:3])
    from preprocess import get_image_transform
    ds = ImageDataset(filenames, IMAGES_DIR, get_image_transform())
    print(len(ds), ds[0].shape)
    
