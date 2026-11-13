"""查询侧编码模块：把用户输入的图片或文本变成 256 维单位向量。"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from src.model import load_convnext, load_bert, DualTowerModel
import torch
from torch.nn import functional as F


def load_model(convnext_path, proj_path):
    """加载双塔模型：冻结编码器 + 训练好的投影层权重。

    Args:
        convnext_path: ConvNeXt-T 本地权重路径。
        proj_path: 投影层 checkpoint 路径。
    Returns:
        处于 eval 模式的 DualTowerModel。
    """
    convnext = load_convnext(convnext_path)
    bert = load_bert()
    model = DualTowerModel(convnext, bert)
    ckpt = torch.load(proj_path, map_location="cpu", weights_only=True)
    model.proj_img.load_state_dict(ckpt["proj_img"])
    model.proj_txt.load_state_dict(ckpt["proj_txt"])
    model.eval()
    return model


def encode_image(model, transform, image):
    """把一张 PIL 图片编码成 256 维单位向量（查询侧）。

    Args:
        model: load_model 返回的 DualTowerModel（eval 模式）。
        transform: get_image_transform() 返回的预处理流水线，必须与图库侧是同一个。
        image: PIL.Image 对象。
    Returns:
        形状 [256] 的 L2 归一化向量。
    """
    image = image.convert("RGB")
    with torch.no_grad():
        x = transform(image)
        x = x.unsqueeze(0)
        feat = model.convnext(x)
        emb = model.proj_img(feat)
        emb = F.normalize(emb, dim=1, p=2)
    return emb[0]


def encode_text(model, tokenizer, text, max_len=32):
    """把一句文本编码成 256 维单位向量（查询侧）。

    Args:
        model: load_model 返回的 DualTowerModel（eval 模式）。
        tokenizer: BertTokenizer，参数必须与训练时一致。
        text: 查询字符串。
        max_len: 截断/填充长度，必须与训练时一致（32）。
    Returns:
        形状 [256] 的 L2 归一化向量。
    """
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model.bert(
            encoded["input_ids"], encoded["attention_mask"]
        )
        feat = out[:, 0, :]
        emb = model.proj_txt(feat)
        emb = F.normalize(emb, dim=1, p=2)
    return emb[0]
