"""离线提取 Flickr30k 全部 caption 的 BERT 特征并缓存到磁盘。"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import csv
from transformers import BertTokenizer
from model import load_bert, masked_mean_pool

CSV_PATH = "data/annotations/captions_flickr30k.csv"
SAVE_PATH = "output/features/text_features_v2.pth"
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
    all_cls = []
    all_mean = []
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
            last_hidden = bert(
                enc["input_ids"], enc["attention_mask"]
            )  # [B, 32, 768]，注意现在是完整序列
            cls_feats = last_hidden[:, 0, :]
            mean_feats = masked_mean_pool(last_hidden, enc["attention_mask"])
            all_mean.append(mean_feats)
            all_cls.append(cls_feats)
            done = min(start + BATCH_SIZE, len(captions))  # 已完成的条数
            if (start // BATCH_SIZE) % 50 == 0 or done == len(captions):
                print(f"进度 {done}/{len(captions)}")

    all_cls = torch.cat(all_cls, dim=0)  # [N, 768]
    all_mean = torch.cat(all_mean, dim=0)
    print("all_cls.shape:", all_cls.shape, "all_mean.shape:", all_mean.shape)
    if len(all_cls) != len(filenames) or len(all_mean) != len(filenames):
        raise ValueError(
            "Text feature rows must match caption metadata: "
            f"cls={len(all_cls)}, mean={len(all_mean)}, filenames={len(filenames)}"
        )
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(
        {
            "cls": all_cls,
            "mean": all_mean,
            "filenames": filenames,  # 与 features 行对齐
            "caption_idxs": caption_idxs,  # 与 features 行对齐
        },
        SAVE_PATH,
    )


if __name__ == "__main__":
    main()
