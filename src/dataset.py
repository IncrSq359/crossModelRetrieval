import csv
from torch.utils.data import Dataset
from PIL import Image
import random

class ImageTextDataset(Dataset):
    def __init__(self, csv_path, transform,tokenizer,max_len=16):
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
        return len(self.rows)

    def __getitem__(self,idx):
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
        input_ids=encoded["input_ids"][0]
        attention_mask=encoded["attention_mask"][0]
        return image, input_ids, attention_mask, row["label"]

def make_class_balanced_batches(labels,seed=42):
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
