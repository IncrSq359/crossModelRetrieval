"""文本特征模块：读取图片描述，用预训练 BERT 把句子编码成 768 维特征，并负责存取。"""

import os
import csv
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 下载时走国内镜像
os.environ["HF_HUB_OFFLINE"] = "1"                             # 模型已下全，直接读缓存、不联网

import torch
from transformers import BertModel #可删
from transformers import BertTokenizer #保留

def load_captions(csv_path):
    """读取 captions.csv，取出文本、标签和图片路径。

    Args:
        csv_path: CSV 文件路径，表头含 image、text、label 三列。

    Returns:
        三元组 (texts, labels, image_paths)，均为长度相同的列表、逐行对应。
    """
    texts=[]
    labels=[]
    image_paths=[]
    with open(csv_path,"r",encoding="utf-8")as f:
        reader=csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"])
            image_paths.append(row["image"])            
    return texts,labels,image_paths

def load_bert():
    """加载预训练 BERT 模型并设为评估模式。

    Returns:
        处于 eval 模式的 BertModel。
    """
    model=BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    return model

def extract_text_features(texts,max_length=16):
    """把一批句子编码成 BERT 的 CLS 句向量。

    Args:
        texts: 字符串列表（每个元素是一句描述）。
        max_length: 分词后保留的最大词元数，超出截断、不足补 PAD，默认 16。

    Returns:
        特征张量，形状 [句子数, 768]，取每个句子第一个位置(CLS)的输出作为整句表示。
    """
    tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
    model=load_bert()
    # encoded保留,其中encoded["input_ids"]和encoded["attention_mask"]作为模型输入
    encoded=tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        output=model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
    features=output.last_hidden_state[:,0,:]
    return features

def save_features(feature_matrix,labels,paths,save_path):
    """把特征、标签、路径打包成字典保存到磁盘。

    Args:
        feature_matrix: 特征张量。
        labels: 标签列表。
        paths: 图片路径列表。
        save_path: 目标文件路径，所在目录不存在时会自动创建。
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
