"""查询侧编码模块：把用户输入的图片或文本变成 256 维单位向量。"""
from src.model import load_resnet,load_bert,DualTowerModel
import torch

def load_model(resnet_path,proj_path):
    """加载双塔模型：冻结编码器 + 训练好的投影层权重。

    Args:
        resnet_path: ResNet-50 本地权重路径。
        proj_path: 投影层 checkpoint 路径。
    Returns:
        处于 eval 模式的 DualTowerModel。
    """
    resnet = load_resnet(resnet_path)
    bert=load_bert()
    model=DualTowerModel(resnet,bert)
    ckpt=torch.load(proj_path,map_location="cpu",weights_only=True)
    model.proj_img.load_state_dict(ckpt["proj_img"])
    model.proj_txt.load_state_dict(ckpt["proj_txt"])
    model.eval()
    return model