"""双塔模型：冻结的 ConvNeXt/BERT 编码器 + 可训练的投影层。

图像特征为 768 维；文本特征将 masked mean 与 CLS 拼接为 1536 维，
然后分别投影到同一个 256 维空间，
投影结果做 L2 归一化，使点积等于余弦相似度。编码器全程冻结，
训练时只有两个投影层的参数更新。
"""

import os

os.environ["HF_HUB_OFFLINE"] = (
    "1"  # 必须在 import transformers 之前设置：库在导入时读一次环境变量并记死
)

import torch
from torch import nn
from bert_model.BERT import BERT
from bert_model.load_bert_weights import load_hf_weights
from torch.nn import functional as F
from Convnext import ConvNeXt


def masked_mean_pool(last_hidden, attention_mask):
    """对有效 token 求平均，忽略 padding。"""
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return summed / count


def load_convnext(weights_path):
    """加载本地权重的 ConvNeXt-T，改造成 768 维特征提取器并冻结。

    Args:
        weights_path: 本地权重文件路径（如 data/raw/convnext_tiny_1k_224_ema.pth）。
    Returns:
        冻结且处于 eval 模式的 ConvNeXt-T，输出 [B, 768]。
    """
    net = ConvNeXt(out_class=1000)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    net.load_state_dict(state, strict=True)
    net.head = nn.Identity()
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    return net


def load_bert():
    """加载手写 BERT 并载入 bert-base-uncased 预训练权重，冻结后作为 768 维文本编码器。

    Returns:
        冻结且处于 eval 模式的手写 BERT；输出完整序列，[B, L, 768]。
    """
    bert = BERT(30522, 768, 12, 12, 0.1)
    bert = load_hf_weights(bert)
    bert.eval()
    for p in bert.parameters():
        p.requires_grad = False
    return bert


class DualTowerModel(nn.Module):
    """双塔模型：图像塔和文本塔各接一个可训练投影层。

    两个投影层是本模型唯一参与训练的部分；
    ConvNeXt 与 BERT 仅作固定特征提取器。
    """

    def __init__(self, convnext, bert, embed_dim=256):
        """
        Args:
            convnext: load_convnext 返回的冻结图像编码器（输出768维）。
            bert: load_bert 返回的冻结文本编码器（输出 [B, L, 768]）。
            embed_dim: 共享空间维度，默认256。
        """
        super().__init__()
        self.convnext = convnext
        self.bert = bert
        self.embed_dim = embed_dim
        self.proj_img = nn.Linear(768, embed_dim)  # 图像投影层（可训练）
        self.proj_txt = nn.Linear(1536, embed_dim)  # masked mean + CLS

    def forward(self, images, input_ids, attention_mask):
        """前向：图文分别编码、投影、L2归一化。

        Args:
            images: 图像批 [B, 3, 224, 224]。
            input_ids: 词编号批 [B, max_len]。
            attention_mask: 注意力掩码批 [B, max_len]。
        Returns:
            (img_emb, txt_emb)：两个 [B, embed_dim] 的 L2 归一化向量，
            二者点积即为余弦相似度。
        """
        img_feat = self.convnext(images)  # [B, 768]
        last_hidden = self.bert(input_ids, attention_mask)
        cls_feat = last_hidden[:, 0, :]
        mean_feat = masked_mean_pool(last_hidden, attention_mask)
        txt_feat = torch.cat((mean_feat, cls_feat), dim=1)  # [B, 1536]
        img_emb = self.proj_img(img_feat)  # [B, 256]
        txt_emb = self.proj_txt(txt_feat)  # [B, embed_dim]
        img_emb = F.normalize(img_emb, dim=1, p=2)  # 每行除以模长 → 单位向量
        txt_emb = F.normalize(txt_emb, dim=1, p=2)
        return img_emb, txt_emb
