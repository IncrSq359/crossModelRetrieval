from utils.transformer_encoder import transformer_encoder as encoder
from utils.embedding import embedding
import torch
from torch import nn, Tensor


class BERT(nn.Module):
    """BERT模型\n
    input:
        x (Tensor): shape[batch,max_len],第一维为句子数量，第二维为句子长度，存token ID
        mask (Tensor): 掩饰码,shape[batch,max_len],与每条句子一一对应
    Args:
        num (int): 词典大小
        dim (int): token维度
        L (int): encoder层数
        H (int): 注意力头
    """

    def __init__(self, num, dim, L, H, dropout):
        super().__init__()
        self.embedding = embedding(num, dim, dropout)
        self.layers = nn.ModuleList()
        ffn = 4 * dim
        for i in range(0, L):
            self.layers.append(encoder(dim, H, ffn, dropout))

    def forward(self, x: Tensor, mask: Tensor):
        x = self.embedding(x)
        mask = mask.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, mask)
        return x
