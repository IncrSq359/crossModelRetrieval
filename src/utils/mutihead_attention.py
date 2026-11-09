import torch
from torch import nn, Tensor,functional
import math

class mutihead_attention(nn.Module):
    """自注意力的多头注意力

    Args:
        d_modules: token维度
        heads: 注意力头
    """
    def __init__(self, d_modules, heads):
        super().__init__()
        assert d_modules % heads == 0
        self.seglen = d_modules
        self.head = heads
        self.dim = d_modules//heads
        self.wq = nn.Linear(d_modules,d_modules)
        self.wk = nn.Linear(d_modules,d_modules)
        self.wv = nn.Linear(d_modules,d_modules)
        self.wo = nn.Linear(d_modules,d_modules)
    
    def transpose(self,x:Tensor):
        # input [batch,len,d_modules]
        # output [batch*heads,len,dim]
        x.reshape(x.shape[0],x.shape[1],self.head,-1)
        x.permute(0,2,1,3)
        x.reshape(-1,x.shape[2],x.shape[3])
        return x
    
    def transpose_inv(self,x:Tensor):
        # input [batch*heads,len,d_modules]
        # output [batch,len,d_modules]
        x.reshape(-1,self.head,x.shape[1],x.shape[2])
        x.permute(0,2,1,3)
        x.reshape(x.shape[0],x.shape[1],-1)
        return x
    
    def forward(self,q,k,v,mask):
        q = self.transpose(self.wq(q))
        k = self.transpose(self.wk(k))
        v = self.transpose(self.wv(v))
        attn_w = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.dim)
        if mask is not None:
            attn_w = attn_w.masked_fill(mask==0,1e-9)
        attn_w = torch.softmax(attn_w,dim=-1)
        output = torch.matmul(attn_w,v)
        output = self.transpose_inv(output)
        return self.wo(output)
        
