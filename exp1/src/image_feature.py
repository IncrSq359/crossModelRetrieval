from torch.utils.data import Dataset
from PIL import Image

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
        
if __name__ == "__main__":
    from data_loader import get_image_data      # 导入：拿到图像记录的函数
    from preprocess import get_image_transform  # 导入：拿到预处理流水线的函数

    records = get_image_data("data/images")     # 第1步：拿到100条记录
    transform = get_image_transform()           # 第2步：造出预处理流水线
    dataset = ImageDataset(records, transform)  # 第3步：把记录和流水线"递"给Dataset

    print("数据总数：", len(dataset))            # 自动调用 __len__

    tensor, label, path = dataset[0]            # 自动调用 __getitem__(0)
    print("第0条图片张量形状：", tensor.shape)
    print("第0条类别：", label)
    print("第0条路径：", path)