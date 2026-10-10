import os
os.environ["HF_HUB_OFFLINE"]="1"

import torch
from torch import nn
from torchvision import models
from transformers import BertModel
from torch.nn import functional as F

def load_resnet(weights_path):
    net=models.resnet50(weights=None)
    state=torch.load(weights_path,map_location="cpu",weights_only=True)
    net.load_state_dict(state)
    net.fc=nn.Identity()
    net.eval()
    for p  in net.parameters():
        p.requires_grad=False
    return net

def load_bert():
    bert=BertModel.from_pretrained("bert-base-uncased")
    bert.eval()
    for p in bert.parameters():
        p.requires_grad=False
    return bert

class DualTowerModel(nn.Module):
    def __init__(self,resnet,bert,embed_dim=256):
        super().__init__()
        self.resnet=resnet
        self.bert=bert
        self.embed_dim=embed_dim
        self.proj_img=nn.Linear(2048,embed_dim)
        self.proj_txt=nn.Linear(768,embed_dim)

    def forward(self,images,input_ids,attention_mask):
        img_feat=self.resnet(images)
        txt_feat=self.bert(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0,:]
        img_emb=self.proj_img(img_feat)
        txt_emb=self.proj_txt(txt_feat)
        img_emb=F.normalize(img_emb,dim=1,p=2)
        txt_emb=F.normalize(txt_emb,dim=1,p=2)
        return img_emb,txt_emb