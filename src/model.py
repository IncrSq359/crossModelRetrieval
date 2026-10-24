"""双塔模型：冻结的 ResNet/BERT 编码器 + 可训练的投影层。

两座塔分别把图像（2048维）和文本（768维）投影到同一个 256 维空间，
投影结果做 L2 归一化，使点积等于余弦相似度。编码器全程冻结，
训练时只有两个投影层的参数更新。
"""
import os
os.environ["HF_HUB_OFFLINE"]="1"   # 必须在 import transformers 之前设置：库在导入时读一次环境变量并记死

import torch
from torch import nn
from torchvision import models
from transformers import BertModel
from torch.nn import functional as F

def load_resnet(weights_path):
    """加载本地权重的 ResNet-50，改造成 2048 维特征提取器并冻结。

    去掉最后的 1000 类分类头 fc（替换为 Identity），输出卷积主干
    全局池化后的 2048 维特征。

    Args:
        weights_path: 本地权重文件路径（如 data/raw/resnet50_timm.pth）。
    Returns:
        冻结且处于 eval 模式的 ResNet-50，输出 [B, 2048]。
    """
    net=models.resnet50(weights=None)
    state=torch.load(weights_path,map_location="cpu",weights_only=True)
    net.load_state_dict(state)
    net.fc=nn.Identity()
    net.eval()
    for p  in net.parameters():
        p.requires_grad=False
    return net

def load_bert():
    """加载 bert-base-uncased 并冻结，作为 768 维文本编码器。

    Returns:
        冻结且处于 eval 模式的 BertModel；前向时取 CLS 位置向量，
        输出 [B, 768]。
    """
    bert=BertModel.from_pretrained("bert-base-uncased")
    bert.eval()
    for p in bert.parameters():
        p.requires_grad=False
    return bert

class DualTowerModel(nn.Module):
    """双塔模型：图像塔和文本塔各接一个可训练投影层。

    两个投影层是本模型唯一参与训练的部分（约72万参数）；
    ResNet 与 BERT 仅作固定特征提取器。
    """

    def __init__(self,resnet,bert,embed_dim=256):
        """
        Args:
            resnet: load_resnet 返回的冻结图像编码器（输出2048维）。
            bert: load_bert 返回的冻结文本编码器（输出768维）。
            embed_dim: 共享空间维度，默认256。
        """
        super().__init__()
        self.resnet=resnet
        self.bert=bert
        self.embed_dim=embed_dim
        self.proj_img=nn.Linear(2048,embed_dim)   # 图像投影层（可训练）
        self.proj_txt=nn.Linear(768,embed_dim)    # 文本投影层（可训练）

    def forward(self,images,input_ids,attention_mask):
        """前向：图文分别编码、投影、L2归一化。

        Args:
            images: 图像批 [B, 3, 224, 224]。
            input_ids: 词编号批 [B, max_len]。
            attention_mask: 注意力掩码批 [B, max_len]。
        Returns:
            (img_emb, txt_emb)：两个 [B, embed_dim] 的 L2 归一化向量，
            二者点积即为余弦相似度。
        """
        img_feat=self.resnet(images)             # [B, 2048]
        txt_feat=self.bert(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state[:,0,:]  # [B, 768] 取CLS
        img_emb=self.proj_img(img_feat)          # [B, 256]
        txt_emb=self.proj_txt(txt_feat)          # [B, 256]
        img_emb=F.normalize(img_emb,dim=1,p=2)   # 每行除以模长 → 单位向量
        txt_emb=F.normalize(txt_emb,dim=1,p=2)
        return img_emb,txt_emb
