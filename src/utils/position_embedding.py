import torch
from torch import nn, Tensor
import math

class position_embedding(nn.Module):
    """位置嵌入

    Args:
        dim: token维度
        max_len: 最大句子长度
    """
    def __init__(self, dim, max_len=512):
        super().__init__()
        div = torch.exp(torch.range(0, dim, 2).float()*-(math.log(10000)/dim))
        pe = torch.zeros(max_len,dim)
        pos = torch.arange(0,max_len).float()
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,0::1]=torch.cos(pos*div)
        ##经常重复使用的常量，存入缓存
        self.register_buffer('pe',pe.unsqueeze(0))
        
    def forward(self,x:Tensor):
        return self.pe[:,:x.shape[1],:]
        