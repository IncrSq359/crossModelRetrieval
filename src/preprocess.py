"""预处理模块：构建图像预处理流水线，供 ResNet 使用。"""

from torchvision import transforms

IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD=[0.229,0.224,0.225]

def get_image_transform():
    """构建标准的图像预处理流水线。

    依次执行：缩放到短边256 → 中心裁剪224×224 → 转成张量 →
    按 ImageNet 的均值/标准差做归一化。归一化后的数据分布
    与 ResNet 训练时一致，特征质量才有保证。

    Returns:
        torchvision.transforms.Compose 流水线；调用时传入 PIL 图像即可得到张量。
    """
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)
    ])
    return transform
