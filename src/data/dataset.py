import csv
import random
from pathlib import Path
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch import nn

class Flickr30kDataset(Dataset):
    def __init__(self,csv_path,images_dir,transform,tokenizer,split=None,max_len=32):
        self.images_dir=Path(images_dir)
        self.transform=transform
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.split=split
        self.rows=[]
        with open(csv_path,"r",encoding="utf-8",newline="") as f:
            reader =csv.DictReader(f)
            for row in reader:
                if self.split is not None and row["split"]!=self.split:
                    continue
                self.rows.append(row)
            
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self,idx):
        row = self.rows[idx]
        image_path = self.images_dir/row["filename"]
        image = Image.open(image_path).convert("RGB")
        image=self.transform(image)
        encoded=self.tokenizer(
            row["caption"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids=encoded["input_ids"][0]
        attention_mask=encoded["attention_mask"][0]
        return image,input_ids,attention_mask,str(image_path)
    
def make_one_caption_batches(dataset,batch_size,seed):
    rng=random.Random(seed)
    groups={}
    for idx,row in enumerate(dataset.rows):
        groups.setdefault(row["filename"],[]).append(idx)
    chosen=[rng.choice(indices) for indices in groups.values()]
    rng.shuffle(chosen)
    batches=[]
    for start in range(0,len(chosen),batch_size):
        batch =chosen[start:start+batch_size]
        if len(batch)==batch_size:
            batches.append(batch)
    return batches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_PATH = os.path.join(PROJECT_ROOT, "output", "features", "image_features.pth")
TXT_PATH = os.path.join(PROJECT_ROOT, "output", "features", "text_features_v2.pth")

def load_data(img_path=IMG_PATH, txt_path=TXT_PATH):
    """加载缓存特征，并建好训练 / 评估要用的索引。

    Args:
        img_path: 图像特征文件路径。
        txt_path: 文本特征文件路径。
    Returns:
        字典，键为：
            img_feat: 图像特征 [31014, 768]。
            txt_feat: 文本特征 [155070, 1536]（masked mean 与 CLS 拼接）。
            caption_rows_of: 图像行号 -> 该图 5 条 caption 的行号列表。
            train_img_rows: 训练集图像行号列表（29000 个）。
            val_img_rows: 验证集图像行号列表（1014 个）。
    """
    img_data = torch.load(img_path, map_location="cpu")
    txt_data = torch.load(txt_path, map_location="cpu")

    # 文件名 -> 图像特征行号
    img_index = {fn: i for i, fn in enumerate(img_data["filenames"])}

    # 图像行号 -> 该图 5 条 caption 的行号
    caption_rows_of = {}
    for txt_row, fn in enumerate(txt_data["filenames"]):
        img_row = img_index[fn]  # 这条 caption 属于哪张图
        caption_rows_of.setdefault(img_row, []).append(txt_row)

    splits = img_data["splits"]
    train_img_rows = [i for i, s in enumerate(splits) if s == "train"]
    val_img_rows = [i for i, s in enumerate(splits) if s == "val"]

    txt_mean = txt_data["mean"]
    txt_cls = txt_data["cls"]
    txt_filenames = txt_data["filenames"]
    if txt_mean.shape[0] != len(txt_filenames) or txt_cls.shape[0] != len(
        txt_filenames
    ):
        raise ValueError(
            "Text feature rows must match caption metadata: "
            f"mean={txt_mean.shape[0]}, cls={txt_cls.shape[0]}, "
            f"filenames={len(txt_filenames)}"
        )
    if txt_mean.shape != txt_cls.shape:
        raise ValueError(
            f"Mean and CLS feature shapes must match, got "
            f"{tuple(txt_mean.shape)} and {tuple(txt_cls.shape)}"
        )
    txt_feat = torch.cat((txt_mean, txt_cls), dim=1)

    return {
        "img_feat": img_data["features"],
        "txt_feat": txt_feat,
        "caption_rows_of": caption_rows_of,
        "train_img_rows": train_img_rows,
        "val_img_rows": val_img_rows,
    }

def load_projection(path: str):
    """加载训练好的投影层权重。

    Args:
        path: projection_flickr30k.pth 的路径。

    Returns:
        (proj_img, proj_txt)：两个载入了权重的 nn.Linear。
    """
    ckpt = torch.load(path, map_location="cpu")

    proj_img = nn.Linear(ckpt["proj_img"]["weight"].shape[1], ckpt["embed_dim"])
    proj_txt = nn.Linear(ckpt["proj_txt"]["weight"].shape[1], ckpt["embed_dim"])

    proj_img.load_state_dict(ckpt["proj_img"])
    proj_txt.load_state_dict(ckpt["proj_txt"])

    return proj_img, proj_txt