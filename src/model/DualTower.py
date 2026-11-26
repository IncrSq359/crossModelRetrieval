import torch
from torch import nn,functional as F

def masked_mean_pool(last_hidden, attention_mask):
    """对有效 token 求平均，忽略 padding。"""
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return summed / count


class DualTowerModel(nn.Module):
    """双塔模型：图像塔和文本塔各接一个可训练投影层。

    两个投影层是本模型唯一参与训练的部分；
    ConvNeXt 与 BERT 仅作固定特征提取器。
    """

    def __init__(self, convnext, bert, embed_dim=256):
        """
        Args:
            convnext: load_convnext 返回的冻结图像编码器（输出768维）。
            bert: load_bert 返回的冻结文本编码器（输出 [B, L, 768]）。
            embed_dim: 共享空间维度，默认256。
        """
        super().__init__()
        self.convnext = convnext
        self.bert = bert
        self.embed_dim = embed_dim
        self.proj_img = nn.Linear(768, embed_dim)  # 图像投影层（可训练）
        self.proj_txt = nn.Linear(1536, embed_dim)  # mean + CLS 文本投影层（可训练）

    def forward(self, images, input_ids, attention_mask):
        """前向：图文分别编码、投影、L2归一化。

        Args:
            images: 图像批 [B, 3, 224, 224]。
            input_ids: 词编号批 [B, max_len]。
            attention_mask: 注意力掩码批 [B, max_len]。
        Returns:
            (img_emb, txt_emb)：两个 [B, embed_dim] 的 L2 归一化向量，
            二者点积即为余弦相似度。
        """
        img_feat = self.convnext(images)  # [B, 768]
        last_hidden = self.bert(input_ids, attention_mask)
        cls_feat = last_hidden[:, 0, :]
        mean_feat = masked_mean_pool(last_hidden, attention_mask)
        txt_feat = torch.cat((mean_feat, cls_feat), dim=1)  # [B, 1536]
        img_emb = self.proj_img(img_feat)  # [B, 256]
        txt_emb = self.proj_txt(txt_feat)  # [B, embed_dim]
        img_emb = F.normalize(img_emb, dim=1, p=2)  # 每行除以模长 → 单位向量
        txt_emb = F.normalize(txt_emb, dim=1, p=2)
        return img_emb, txt_emb