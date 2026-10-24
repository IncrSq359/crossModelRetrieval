import csv
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

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