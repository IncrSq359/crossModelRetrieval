from torch import nn,Tensor
from .mutihead_attention import mutihead_attention as att
from .feed_forward import feed_forward as ffn
from .layer_normalize import layer_norm as ln

class transformer_encoder(nn.Module):
    def __init__(self,d_modules,heads,ffn_features,dropout):
        super().__init__()
        self.att = att(d_modules,heads)
        self.ffn = ffn(d_modules,ffn_features)
        self.ln1=ln(dim=-1,features=d_modules)
        self.ln2=ln(dim=-1,features=d_modules)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x:Tensor,mask:Tensor):
        src =x
        x=self.att(x,x,x,mask)
        x = self.ln1(self.dropout(x)+src)
        src = x
        x = self.ffn(x)
        x = self.ln2(self.dropout(x)+src)
        return x