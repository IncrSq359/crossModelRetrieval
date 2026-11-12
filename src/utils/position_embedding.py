import torch
from torch import nn, Tensor


class position_embedding(nn.Module):
    """可学习的位置嵌入

    Args:
        dim: token维度
        max_len: 最大句子长度
    """

    def __init__(self, dim, max_len=512):
        super().__init__()
        # 一张 [max_len, dim] 的表：第 i 行就是位置 i 的向量
        self.pe = nn.Embedding(max_len, dim)

    def forward(self, x: Tensor):
        # x 是 token ID，形状 [batch, len]，这里只需要用到 len
        # 生成位置编号 0, 1, 2, ..., len-1，并放在和 x 相同的设备上
        pos = torch.arange(x.shape[1], device=x.device)
        # 用位置编号去查表，得到 [len, dim]
        out = self.pe(pos)
        # 加一个 batch 维，变成 [1, len, dim]，这样才能和 token 嵌入相加
        return out.unsqueeze(0)
