import torch
from torch.nn import functional as F

def dual_infonce_loss(img_emb,txt_emb,temperature=0.07):
    logits=img_emb@txt_emb.T/temperature
    targets=torch.arange(img_emb.shape[0],device=img_emb.device)
    loss_i2t=F.cross_entropy(logits,targets)
    loss_t2i=F.cross_entropy(logits.T,targets)
    return (loss_i2t+loss_t2i)/2
