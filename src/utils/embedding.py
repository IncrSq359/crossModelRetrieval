from .position_embedding import position_embedding as pe
from .layer_normalize import layer_norm as ln
import torch
from torch import nn, Tensor


class token(nn.Embedding):
    """token ID转token
    Args:
        num: 词典大小
        dim: token维度
    """

    def __init__(self, num, dim):
        super().__init__(num, dim, padding_idx=0)


class embedding(nn.Module):
    def __init__(self, num, dim, dropout):
        super().__init__()
        self.token = token(num, dim)
        self.pos = pe(dim)
        self.token_type = nn.Embedding(2, dim)
        self.ln = ln(-1, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(x)
        token_embedding = self.token(x)
        position_embedding = self.pos(x)
        type_embedding = self.token_type(token_type_ids)
        x = token_embedding + position_embedding + type_embedding
        x = self.dropout(self.ln(x))
        return x
