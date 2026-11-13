"""离线提取 Flickr30k 全部 caption 的 BERT 特征并缓存到磁盘。"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import csv
from transformers import BertTokenizer
from model import load_bert

CSV_PATH = "data/annotations/captions_flickr30k.csv"
SAVE_PATH = "output/features/text_features.pth"
MAX_LEN = 32
BATCH_SIZE = 128


def read_captions(csv_path):
    """按 CSV 原始行顺序读出 filename / caption_idx / caption 三个列表。

    Returns:
        (filenames, caption_idxs, captions) 三个等长的 list
    """
    filenames, caption_idxs, captions = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename, caption_idx, caption = (
                row["filename"],
                row["caption_idx"],
                row["caption"],
            )
            filenames.append(filename)
            caption_idxs.append(int(caption_idx))
            captions.append(caption)
    return filenames, caption_idxs, captions


def main():
    filenames, caption_idxs, captions = read_captions(CSV_PATH)
    print("caption 总数：", len(captions))  # 应该是 155070

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = load_bert()
    bert.eval()  # 关闭 dropout，保证特征可复现

    all_feats = []
    with torch.no_grad():  # 不需要梯度，省内存省时间
        for start in range(0, len(captions), BATCH_SIZE):
            batch_texts = captions[start : start + BATCH_SIZE]

            enc = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            feats = bert(enc["input_ids"], enc["attention_mask"])  # [B, 768]
            feats = feats[:, 0, :]  # 取 CLS 向量，得到 [B, 768]
            all_feats.append(feats)
            done = min(start + BATCH_SIZE, len(captions))  # 已完成的条数
            if (start // BATCH_SIZE) % 50 == 0 or done == len(captions):
                print(f"进度 {done}/{len(captions)}")

    all_feats = torch.cat(all_feats, dim=0)  # [N, 768]
    print("all_feats.shape:", all_feats.shape)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(
        {
            "features": all_feats,  # [155070, 768]
            "filenames": filenames,  # 与 features 行对齐
            "caption_idxs": caption_idxs,  # 与 features 行对齐
        },
        SAVE_PATH,
    )


if __name__ == "__main__":
    main()
