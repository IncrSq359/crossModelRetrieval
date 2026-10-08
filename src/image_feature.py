"""图像特征模块：用预训练 ResNet-50 把图片编码成 2048 维特征，并负责存取。"""

from os import path
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torch
from torchvision import models
import torch.nn as nn

class ImageDataset(Dataset):
    """图像数据集：按索引懒加载图片、做预处理并返回张量。

    继承 torch 的 Dataset，配合 DataLoader 可实现按批次自动读取。
    所谓"懒加载"指不在一开始读入全部图片，而是用到第几张才打开第几张，省内存。
    """

    def __init__(self,records,transform):
        """
        Args:
            records: get_image_data 返回的记录列表，每条含 path 和 label。
            transform: get_image_transform 返回的预处理流水线。
        """
        self.records=records
        self.transform=transform

    def __len__(self):
        """返回数据集中图片的总张数。"""
        return len(self.records)
    
    def __getitem__(self,idx):
        """取第 idx 张图片，打开并预处理。

        Returns:
            三元组 (图像张量, 标签字符串, 图片路径字符串)。
        """
        rec=self.records[idx]
        img=Image.open(rec["path"]).convert('RGB')
        tensor=self.transform(img)
        return tensor ,rec["label"],str(rec["path"])
    
def load_resnet():
    """加载本地权重的 ResNet-50，并改造成特征提取器。

    去掉最后的全连接分类层(fc)，替换为 Identity（原样输出），
    这样模型输出的是倒数第二层 2048 维特征向量，而非1000类分数。

    Returns:
        处于评估模式(eval)的 ResNet-50 模型，输出 [批次, 2048]。
    """
    model=models.resnet50(weights=None)
    state_dict=torch.load("data/raw/resnet50_timm.pth",
                          map_location="cpu",weights_only=True)
    model.load_state_dict(state_dict)
    model.fc=nn.Identity()
    model.eval()
    return model
        
def extract_image_features(dataset,batch_size=32):
    """对整个数据集批量提取图像特征。

    Args:
        dataset: ImageDataset 实例。
        batch_size: 每批处理的图片张数，默认 32。

    Returns:
        三元组：
        feature_matrix: 全部特征拼接成的张量，形状 [图片数, 2048]；
        labels: 标签列表（与特征逐行对应）；
        paths: 图片路径列表（与特征逐行对应）。
    """
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    model = load_resnet()
    all_features=[]
    all_labels=[]
    all_paths=[]
    with torch.no_grad():
        for images,labels,paths in loader:
            features=model(images)
            all_features.append(features)
            all_labels.extend(labels)
            all_paths.extend(paths)
    feature_matrix=torch.cat(all_features,dim=0)
    return feature_matrix,all_labels,all_paths

def save_features(feature_matrix,labels,paths,save_path):
    """把特征、标签、路径打包成字典保存到磁盘。

    Args:
        feature_matrix: 特征张量。
        labels: 标签列表。
        paths: 路径列表。
        save_path: 目标文件路径（如 "output/features/image_features.pth"），
            所在目录不存在时会自动创建。
    """
    Path(save_path).parent.mkdir(parents=True,exist_ok=True)
    data={
        "features":feature_matrix,
        "labels":labels,
        "paths":paths
    }
    torch.save(data,save_path)
    print(f"特征已保存到 {save_path}")

def load_features(load_path):
    """从磁盘读回 save_features 保存的特征数据。

    Args:
        load_path: 特征文件路径。

    Returns:
        三元组 (feature_matrix, labels, paths)。
    """
    data=torch.load(load_path,map_location='cpu',weights_only=True)
    feature_matrix=data["features"]
    labels=data["labels"]
    paths=data["paths"]
    return feature_matrix,labels,paths
