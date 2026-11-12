import torch
from torch import nn
from torch import Tensor
class layer_norm(nn.Module):
    """Layer Normalization\n
    对指定单维度进行层规范化
    Args:
        dim: 规范维度
        features: 维度长度
        eps
    """

    def __init__(self, dim,features, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor):
        ave = x.mean(dim=self.dim, keepdim=True)
        s = (x - ave).pow(2).mean(dim=self.dim, keepdim=True)
        x = (x - ave) / (s + self.eps).sqrt()
        shape = [1]*x.dim()
        shape[self.dim]=-1
        x = x * self.weight.view(shape)+ self.bias.view(shape)
        return x
