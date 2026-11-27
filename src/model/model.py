"""双塔模型的编码器加载 + 冻结/解冻控制。

图像编码器是 ConvNeXt-T（ImageNet-1k 权重），输出 768 维；
文本编码器是手写 BERT（bert-base-uncased 权重），输出 [B, L, 768]。

默认全部冻结（只训练投影层）。要微调时按「从后往前」解冻：

    unfreeze_convnext_last_stages(model.convnext, 1)   # 最后一个 stage + norm
    unfreeze_bert_last_layers(model.bert, 2)           # 最后 2 层 encoder

先冻结、临到第二阶段再解冻，比构造时就解冻更灵活，
因为两阶段训练需要换优化器（不同参数组给不同 lr）。
"""

import os

# 云端首次运行需要联网拉 bert-base-uncased；本地已有缓存时会直接用缓存
# （load_hf_weights 里先试 local_files_only=True）。想强制离线：HF_HUB_OFFLINE=1
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import torch
from torch import nn
from .Convnext import ConvNeXt
from .bert_model.BERT import BERT
from .bert_model.load_bert_weights import load_hf_weights


def freeze_all(module: nn.Module) -> None:
    """把整个模块的所有参数设为不可训练。"""
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_convnext_last_stages(net: ConvNeXt, n_stages: int = 1) -> int:
    """解冻 ConvNeXt 最后 n_stages 个 stage，以及末尾的 norm。

    stage 之间的 downsample_layers 保持冻结：参数量小、对最终特征影响次要，
    少解冻一点能明显降低过拟合风险。

    Args:
        net: ConvNeXt 实例。
        n_stages: 解冻最后几个 stage（0~4），0 表示保持全冻结。
    Returns:
        本次解冻的参数个数。
    """
    freeze_all(net)
    if n_stages <= 0:
        return 0
    n_stages = min(n_stages, len(net.stages))
    for stage in list(net.stages)[-n_stages:]:
        for p in stage.parameters():
            p.requires_grad = True
    # norm 直接作用在输出上，跟着最后一个 stage 一起解冻
    for p in net.norm.parameters():
        p.requires_grad = True
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def unfreeze_bert_last_layers(bert: BERT, n_layers: int = 2) -> int:
    """解冻 BERT 最后 n_layers 个 encoder 层（embedding 始终冻结）。

    embedding 里是 30522x768 的词表，解冻它既费显存又容易过拟合，
    微调时通常不动。

    Args:
        bert: 手写 BERT 实例。
        n_layers: 解冻最后几层，0 表示保持全冻结。
    Returns:
        本次解冻的参数个数。
    """
    freeze_all(bert)
    if n_layers <= 0:
        return 0
    n_layers = min(n_layers, len(bert.layers))
    for layer in list(bert.layers)[-n_layers:]:
        for p in layer.parameters():
            p.requires_grad = True
    return sum(p.numel() for p in bert.parameters() if p.requires_grad)


def load_convnext(weights_path, device=None):
    """加载本地权重的 ConvNeXt-T，改造成 768 维特征提取器并冻结。

    Args:
        weights_path: 本地权重文件路径（如 data/raw/convnext_tiny_1k_224_ema.pth）。
        device: 目标设备；None 表示留在 CPU（由调用方之后统一 .to(device)）。
    Returns:
        冻结且处于 eval 模式的 ConvNeXt-T，输出 [B, 768]。
    """
    net = ConvNeXt(out_class=1000)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    net.load_state_dict(state, strict=True)
    net.head = nn.Identity()
    freeze_all(net)
    net.eval()
    if device is not None:
        net.to(device)
    return net


def load_bert(device=None):
    """加载手写 BERT 并载入 bert-base-uncased 预训练权重，冻结后作为 768 维文本编码器。

    Args:
        device: 目标设备；None 表示留在 CPU。
    Returns:
        冻结且处于 eval 模式的手写 BERT；输出完整序列，[B, L, 768]。
    """
    bert = BERT(30522, 768, 12, 12, 0.1)
    bert = load_hf_weights(bert)
    freeze_all(bert)
    bert.eval()
    if device is not None:
        bert.to(device)
    return bert


def trainable_parameter_summary(model: nn.Module):
    """返回 (可训练参数量, 总参数量)，用于打印/记录。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total