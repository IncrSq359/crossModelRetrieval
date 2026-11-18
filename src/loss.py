"""InfoNCE 对比损失：用 In-Batch Negatives 实现图文特征对齐。"""

import torch
from .utils.CrossEntropy import cross_entropy


def dual_infonce_loss(img_emb, txt_emb, temperature=0.07):
    """双向 InfoNCE 损失：同时监督"以图搜文"和"以文搜图"。

    一个 batch 内，同编号的图文对是正样本（相似度矩阵对角线），
    其余 B-1 对是 In-Batch 负样本。对相似度矩阵的行、列各做一次
    softmax 交叉熵后取平均。随机模型初始损失约为 ln(B)。

    Args:
        img_emb: 图像向量 [B, D]，已 L2 归一化。
        txt_emb: 文本向量 [B, D]，已 L2 归一化。
        temperature: 温度系数 τ，越小 softmax 越尖锐、对比越严格，默认0.07。
    Returns:
        标量损失（0维张量），可直接 backward()。
    """
    logits = img_emb @ txt_emb.T / temperature  # [B,B] 缩放后的相似度矩阵
    targets = torch.arange(
        img_emb.shape[0], device=img_emb.device
    )  # 对角线是正确答案 [0,1,...,B-1]
    loss_i2t = cross_entropy(logits, targets)  # 以图搜文：每行正确答案在对角列
    loss_t2i = cross_entropy(logits.T, targets)  # 以文搜图：转置后每列正确答案在对角行
    return (loss_i2t + loss_t2i) / 2
