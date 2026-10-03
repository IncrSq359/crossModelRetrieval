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
if __name__ == "__main__":
    from PIL import Image                          # Pillow：用来打开图片

    img_path = "data/images/ant/image_0003.jpg"    # 选一张子集里的真实图片（先确认这个文件存在）
    img = Image.open(img_path).convert("RGB")      # 打开图片并转成 RGB 模式（去掉可能的透明通道）
    print("处理前尺寸（宽, 高）：", img.size)        # PIL 的 size 是 (宽, 高) 顺序

    transform = get_image_transform()              # 拿到我们刚写的流水线
    tensor = transform(img)                        # 把图片送进流水线
    print("处理后形状 [C, H, W]：", tensor.shape)
    print("数值范围：最小", round(tensor.min().item(), 3), " 最大", round(tensor.max().item(), 3))
    print("三个通道各自的平均值：", tensor.mean(dim=(1, 2)).tolist())
