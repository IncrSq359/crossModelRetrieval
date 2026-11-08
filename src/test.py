"""临时测试：确认 ConvNeXt 权重加载成功，且输出形状正确。"""
import torch
from image_feature import load_convnext  # 如果模块名不同，按你的文件名改

# 加载模型：这一步内部用 strict=True，权重键名对不上会直接报错
model = load_convnext()

# 造一个随机输入：2 张图、3 通道、224x224，形状和真实预处理后的一致
x = torch.randn(2, 3, 224, 224)

# 推理时不需要梯度，关掉可以省内存、更快
with torch.no_grad():
    out1 = model(x)
    out2 = model(x)

# 1. 形状：应该是 torch.Size([2, 768])
print("输出形状：", out1.shape)

# 2. 有没有 nan（数值异常）：应该是 False
print("含 nan：", torch.isnan(out1).any().item())

# 3. 同一输入跑两次，结果应该完全一致（说明已经是 eval 模式，没有随机性）
print("两次输出一致：", torch.allclose(out1, out2))