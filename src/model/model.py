"""双塔模型：冻结的 Convnext/BERT 编码器 + 可训练的投影层。

图像特征为 768 维；文本特征拼接 masked mean 与 CLS，得到 1536 维。
两座塔投影到同一个 256 维空间，
投影结果做 L2 归一化，使点积等于余弦相似度。编码器全程冻结，
训练时只有两个投影层的参数更新。
"""

import os

os.environ["HF_HUB_OFFLINE"] = (
    "1"  # 必须在 import transformers 之前设置：库在导入时读一次环境变量并记死
)

import torch
from torch import nn
from transformers import BertTokenizer
from bert_model.BERT import BERT
from bert_model.load_bert_weights import load_hf_weights
from torch.nn import functional as F
from Convnext import ConvNeXt


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



