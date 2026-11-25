from pathlib import Path
import csv
import os

os.environ["HF_HUB_OFFLINE"]="1"
from transformers import BertTokenizer

PROJECT_ROOT=Path(__file__).resolve().parents[1]
csv_path=PROJECT_ROOT/"data"/"annotations"/"captions_flickr30k.csv"

tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")

lengths=[]
with open(csv_path,"r",encoding="utf-8",newline="")as f:
    reader =csv.DictReader(f)
    for i,row in enumerate(reader):
        if row["split"]=="train":
            enc=tokenizer(row["caption"])
            lengths.append(len(enc["input_ids"]))
            if(len(lengths)%50000==0):
                print("已处理",len(lengths))
lengths.sort()
n=len(lengths)
def pct(q):
    return lengths[int(q*(n-1))]

over_32=sum(1 for l in lengths if l>32)
print ( "train caption 条数：" , n)
print ( "token 数 min / max：" , lengths[ 0 ], lengths[- 1 ])
print ( "P50 / P95 / P99：" , pct( 0.5 ), pct( 0.95 ), pct( 0.99 ))
print ( "超过 32（会被截断）的条数：" , over_32, "占比：{:.2%}" . format (over_32 / n))