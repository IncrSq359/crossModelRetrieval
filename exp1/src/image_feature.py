from os import path
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torch
from torchvision import models
import torch.nn as nn

class ImageDataset(Dataset):
    def __init__(self,records,transform):
        self.records=records
        self.transform=transform

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self,idx):
        rec=self.records[idx]
        img=Image.open(rec["path"]).convert('RGB')
        tensor=self.transform(img)
        return tensor ,rec["label"],str(rec["path"])
    
def load_resnet():
    model=models.resnet50(weights=None)
    state_dict=torch.load("data/raw/resnet50_timm.pth",
                          map_location="cpu",weights_only=True)
    model.load_state_dict(state_dict)
    model.fc=nn.Identity()
    model.eval()
    return model
        
def extract_image_features(dataset,batch_size=32):
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
    Path(save_path).parent.mkdir(parents=True,exist_ok=True)
    data={
        "features":feature_matrix,
        "labels":labels,
        "paths":paths
    }
    torch.save(data,save_path)
    print(f"特征已保存到 {save_path}")

def load_features(load_path):
    data=torch.load(load_path,map_location='cpu',weights_only=True)
    feature_matrix=data["features"]
    labels=data["labels"]
    paths=data["paths"]
    return feature_matrix,labels,paths

if __name__ == "__main__":
    from data_loader import get_image_data      # 导入：拿到图像记录的函数
    from preprocess import get_image_transform  # 导入：拿到预处理流水线的函数

    records = get_image_data("data/images")     # 第1步：拿到100条记录
    transform = get_image_transform()           # 第2步：造出预处理流水线
    dataset = ImageDataset(records, transform)  # 第3步：把记录和流水线"递"给Dataset

    # 第4步：批量提取全部特征（权重已预下载到缓存，这里直接使用）
    feature_matrix, labels, paths = extract_image_features(dataset, batch_size=32)

    print("特征矩阵形状：", feature_matrix.shape)   # 期待 torch.Size([100, 2048])
    print("标签数量：", len(labels))                # 期待 100
    print("路径数量：", len(paths))                 # 期待 100
    print("前3个标签：", labels[:3])                # 期待前3个都是 ant

        # 保存到 output/features 目录（该目录已被gitignore忽略，不进仓库）
    save_features(feature_matrix, labels, paths, "output/features/image_features.pth")

    # 立即读回，验证存取一致
    fm2, labels2, paths2 = load_features("output/features/image_features.pth")
    print("读回特征形状：", fm2.shape)              # 仍应是 [100, 2048]
    print("存取是否完全一致：", torch.equal(feature_matrix, fm2))   # True
