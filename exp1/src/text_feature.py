import os
import csv
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 下载时走国内镜像
os.environ["HF_HUB_OFFLINE"] = "1"                             # 模型已下全，直接读缓存、不联网

import torch
from transformers import BertModel
from transformers import BertTokenizer

def load_captions(csv_path):
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
    model=BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    return model

def extract_text_features(texts,max_length=16):
    tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
    model=load_bert()
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
    # 第1步：读入captions.csv，拿到100句文本及对应的标签、图片路径
    texts, labels, image_paths = load_captions("data/texts/captions.csv")
    print("句子数量：", len(texts))                  # 期待 100

    # 第2步：一次编码全部句子、BERT提特征、取CLS
    text_features = extract_text_features(texts)
    print("文本特征形状：", text_features.shape)      # 期待 torch.Size([100, 768])

    print("标签数量：", len(labels))                 # 期待 100
    print("前3个标签：", labels[:3])                 # ['ant','ant','ant']
    print("第0句：", texts[0])                       # a photo of a ant
        # 第3步：把文本特征连同标签、路径一起存盘
    save_features(text_features, labels, image_paths, "output/features/text_features.pth")

    # 第4步：立即读回，验证存取完全一致
    tf2, labels2, paths2 = load_features("output/features/text_features.pth")
    print("读回特征形状：", tf2.shape)                         # 仍应 [100, 768]
    print("存取是否完全一致：", torch.equal(text_features, tf2))  # True