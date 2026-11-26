"""投影层训练的共享模块：数据加载、采样器、损失、Recall 评估、日志。

这个文件里只放「不随超参数变化」的东西，调参脚本 tune_projection.py 会 import 它。
"""

import os
from ..data.dataset import load_data
import torch.nn as nn
import torch
import torch.nn.functional as F

# 项目根目录 = 本文件（src/proj_lib.py）所在目录的上一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_proj(in_dim, hidden_dim, out_dim, dropout=0.0):
    """构建投影层：hidden_dim 为 0 时是单层线性，否则是两层 MLP。

    Args:
        in_dim: 输入维度（768）。
        hidden_dim: 隐藏层宽度，0 表示不用隐藏层。
        out_dim: 输出维度（embed_dim）。
    Returns:
        一个 nn.Module。
    """
    if hidden_dim == 0:
        return nn.Linear(in_dim, out_dim)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


@torch.no_grad()
def project_and_normalize(features: torch.Tensor, proj: nn.Module) -> torch.Tensor:
    """先投影，再 L2 归一化。

    Args:
        features: [N, 768] 原始特征。
        proj: 投影层（图像侧或文本侧）。

    Returns:
        [N, 256]，每行长度为 1。
    """
    input_layer = next(
        (layer for layer in proj.modules() if isinstance(layer, nn.Linear)), None
    )
    if input_layer is not None and features.shape[-1] != input_layer.in_features:
        raise ValueError(
            f"Feature dimension {features.shape[-1]} does not match projection "
            f"input dimension {input_layer.in_features}"
        )
    out = proj(features)
    out = F.normalize(out, dim=-1)
    return out

if __name__ == "__main__":
    # 自测：数据规模应为 [31014, 768] / [155070, 768] / 31014 29000 1014
    d = load_data()
    print(d["img_feat"].shape, d["txt_feat"].shape)
    print(len(d["caption_rows_of"]), len(d["train_img_rows"]), len(d["val_img_rows"]))
    feats = torch.load("output/features/image_features.pth")["features"]
