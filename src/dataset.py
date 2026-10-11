"""图文配对数据集与按类采样器，供双塔对比学习使用。"""

import csv
from torch.utils.data import Dataset
from PIL import Image
import random

class ImageTextDataset(Dataset):
    """图文配对数据集：CSV 的一行 = 一个训练样本。

    每个样本返回 (图像张量, 词编号, 注意力掩码, 标签)。
    图像预处理和文本分词器都从外部传入（依赖注入），数据集本身
    只负责"按索引取一对图文"。
    """

    def __init__(self, csv_path, transform,tokenizer,max_len=16):
        """
        Args:
            csv_path: 图文清单 CSV 路径，含 image/text/label 三列。
            transform: 图像预处理流水线（get_image_transform 的返回值）。
            tokenizer: BERT 分词器，把句子变成词编号。
            max_len: 句子统一长度（含 CLS/SEP），不足补 PAD，默认16。
        """
        self.csv_path=csv_path
        self.transform=transform
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.rows=[]
        with open(csv_path,'r',encoding='utf-8') as f:
            reader=csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

    def __len__(self):
        """返回样本（图文对）总数。"""
        return len(self.rows)

    def __getitem__(self,idx):
        """取第 idx 个图文对。

        Returns:
            image: 图像张量 [3, 224, 224]。
            input_ids: 词编号张量 [max_len]。
            attention_mask: 注意力掩码 [max_len]，真词位置为1、PAD为0。
            label: 类别名字符串。
        """
        row=self.rows[idx]
        img=Image.open(row['image']).convert("RGB")
        image=self.transform(img)
        encoded=self.tokenizer(
            row['text'],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        # tokenizer 返回带批次维 [1, max_len]，取 [0] 去掉多余的那一维
        input_ids=encoded["input_ids"][0]
        attention_mask=encoded["attention_mask"][0]
        return image, input_ids, attention_mask, row["label"]

def make_class_balanced_batches(labels,seed=42):
    """构造"每批类别互不相同"的 batch 索引列表，避免假负样本。

    先按类别把样本索引分组并在类内打乱，再按"列"抽取：
    第 k 个 batch 由每个类别的第 k 个样本组成。要求每类样本数相同
    （本数据为 10 类 × 10 张）。

    Args:
        labels: 长度 N 的标签列表，第 i 项是第 i 个样本的类别。
        seed: 随机种子，固定后分组打乱结果可复现，默认42。
    Returns:
        list[list[int]]：每个元素是一个 batch（含每类各一个样本索引）。
        直接传给 DataLoader 的 batch_sampler 参数。
    """
    random.seed(seed)
    grouped={}
    for idx,label in enumerate(labels):
        grouped.setdefault(label,[]).append(idx)
    for label in grouped:
        random.shuffle(grouped[label])
    label_names=list(grouped.keys())
    num_batches=len(grouped[label_names[0]])
    batches=[]
    for k in range(num_batches):
       batch=[grouped[name][k] for name in label_names]
       batches.append(batch)
    return batches
