"""双塔模型：图像塔和文本塔各接一个可训练投影层。

两个投影层在「只训投影」阶段是本模型唯一参与训练的部分；
ConvNeXt 与 BERT 默认冻结，微调阶段再由
model.model.unfreeze_convnext_last_stages / unfreeze_bert_last_layers 解冻。
"""

import torch
from torch import nn
import torch.nn.functional as F


def masked_mean_pool(last_hidden, attention_mask):
    """对有效 token 求平均，忽略 padding。"""
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)
    return summed / count


def _has_trainable(module: nn.Module) -> bool:
    return any(p.requires_grad for p in module.parameters())


class DualTowerModel(nn.Module):
    """双塔模型：图像塔和文本塔各接一个可训练投影层。

    两个投影层在「只训投影」阶段是本模型唯一参与训练的部分；
    ConvNeXt 与 BERT 默认仅作固定特征提取器。
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
        self.dropout = nn.Dropout(0.1)

    def encode_image(self, images):
        """图像 -> 单位向量 [B, embed_dim]。"""
        img_feat = self.convnext(images)  # [B, 768]
        img_feat = self.dropout(img_feat)
        return F.normalize(self.proj_img(img_feat), dim=1, p=2)

    def encode_text(self, input_ids, attention_mask):
        """文本 -> 单位向量 [B, embed_dim]（masked mean 与 CLS 拼接后投影）。"""
        last_hidden = self.bert(input_ids, attention_mask)
        cls_feat = last_hidden[:, 0, :]
        mean_feat = masked_mean_pool(last_hidden, attention_mask)
        txt_feat = torch.cat((mean_feat, cls_feat), dim=1)  # [B, 1536]
        txt_feat = self.dropout(txt_feat)
        return F.normalize(self.proj_txt(txt_feat), dim=1, p=2)

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
        return self.encode_image(images), self.encode_text(input_ids, attention_mask)

    def train(self, mode: bool = True):
        """切换训练模式后，把「完全冻结」的子树重新切回 eval。

        BERT 有 dropout：冻结层若跟着 model.train() 打开 dropout，
        冻结特征每步抖动，投影层只能去拟合噪声（train loss 降、val 不涨）。
        这里按「子模块是否含可训练参数」逐个同步，未冻结层的 dropout 照常开启。
        """
        super().train(mode)
        if not mode:
            return self

        # 先按容器整体同步，再逐个 stage / layer 修正（子树的覆盖优先）
        self.convnext.train(_has_trainable(self.convnext))
        for stage in self.convnext.stages:
            stage.train(_has_trainable(stage))
        self.convnext.downsample_layers.train(_has_trainable(self.convnext.downsample_layers))
        self.convnext.norm.train(_has_trainable(self.convnext.norm))

        self.bert.train(_has_trainable(self.bert))
        self.bert.embedding.train(_has_trainable(self.bert.embedding))
        for layer in self.bert.layers:
            layer.train(_has_trainable(layer))
        return self