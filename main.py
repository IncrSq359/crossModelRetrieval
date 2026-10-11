import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/"src"))

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from preprocess import get_image_transform
from dataset import ImageTextDataset,make_class_balanced_batches 
from model import load_bert,load_resnet,DualTowerModel
from train import train
from retrieve import evaluate

def main():
    tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
    transform=get_image_transform()
    dataset=ImageTextDataset("data/texts/captions.csv",transform,tokenizer)
    labels=[row["label"] for row in dataset.rows]
    train_loader=DataLoader(dataset,batch_sampler=make_class_balanced_batches(labels))
    eval_loader=DataLoader(dataset,batch_size=10,shuffle=False)
    resnet=load_resnet("data/raw/resnet50_timm.pth")
    bert=load_bert()
    model=DualTowerModel(resnet,bert)
    loss_history=train(model,train_loader,num_epoches=30,lr=1e-3)
    result=evaluate(model,eval_loader)
    for name,value in result.items():
        print(f"{name}: {value:.4f}")
    out_dir=Path("output/models")
    out_dir.mkdir(parents=True,exist_ok=True)
    torch.save({
        "proj_img":model.proj_img.state_dict(),
        "proj_txt":model.proj_txt.state_dict(),
        "loss_history":loss_history
    },out_dir/"model.pth")

if __name__=="__main__":
    main()