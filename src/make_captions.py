from pathlib import Path
import csv
from data_loader import get_image_data

def make_captions(image_dir,csv_path):
    records=get_image_data(image_dir)
    rows=[]
    for rec in records:
        label=rec["label"].replace("_"," ")
        text=f"a photo of a {label}"
        rows.append({
            "image":rec["path"],
            "text":text,
            "label":rec["label"]
        })
    with open(csv_path,"w",newline="",encoding="utf-8")as f:
        writer=csv.DictWriter(f,fieldnames=["image","text","label"])
        writer.writeheader()
        writer.writerows(rows)
if __name__ == "__main__":
    make_captions(
        image_dir="data/images",
        csv_path="data/texts/captions.csv",
    )
    print("CSV 生成完成")