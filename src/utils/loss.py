import torch
from torch import functional as F

def info_nce_loss(img_emb, txt_emb, temperature=0.07):
    """双向 InfoNCE 损失。

    Args:
        img_emb: 图像向量 [B, D]，已 L2 归一化。
        txt_emb: 文本向量 [B, D]，已 L2 归一化。
        temperature: 温度系数，越小越「尖锐」。
    Returns:
        标量损失。
    """
    logits = img_emb @ txt_emb.T / temperature  # [B, B]，对角线是正样本
    labels = torch.arange(img_emb.shape[0])  # 第 i 行的正确答案是第 i 列
    loss_i2t = F.cross_entropy(logits, labels)  # 按行：图找文
    loss_t2i = F.cross_entropy(logits.T, labels)  # 按列：文找图
    return (loss_i2t + loss_t2i) / 2
