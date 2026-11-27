import torch
import random
from torchvision import transforms

try:
    # python -m src.train_dual_tower（以项目根为顶层包）
    from ..utils.proj_lib import project_and_normalize
except ImportError:
    # python src/xxx.py（以 src 为顶层，data/utils/model 互为平级包）
    from utils.proj_lib import project_and_normalize

IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD=[0.229,0.224,0.225]

"""预处理模块：构建图像预处理流水线，供 Convnext 使用。"""
def get_image_transform():
    """构建标准的图像预处理流水线。

    依次执行：缩放到短边256 → 中心裁剪224×224 → 转成张量 →
    按 ImageNet 的均值/标准差做归一化。归一化后的数据分布
    与 Convnext 训练时一致，特征质量才有保证。

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

def get_train_transform():
    """构建训练用的随机增广流水线（解冻编码器时必用，否则小数据必然过拟合）。

    只用「不改变语义」的增广：随机裁剪缩放 + 水平翻转。
    不做颜色抖动——颜色在图文检索里本身是有效线索。

    Returns:
        torchvision.transforms.Compose 流水线。
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_set(data, proj_img, proj_txt, img_rows):
    """构造评测用的图库、查询向量和标准答案。

    图库是全部图片投影后的向量；查询是 img_rows 这些图片的所有
    caption 投影后的向量；标准答案是每条 caption 所属图片在图库中的行号。

    Args:
        data: load_data() 返回的字典。
        proj_img: 图像投影层。
        proj_txt: 文本投影层。
        img_rows: 参与评测的图片行号列表，例如 data["val_img_rows"]。

    Returns:
        (gallery, queries, gt) 三元组：
        gallery 形状 [N, 256]，queries 形状 [Q, 256]，gt 形状 [Q]。
    """
    gallery = project_and_normalize(data["img_feat"], proj_img)

    txt_rows = []
    gt = []
    for img_row in img_rows:
        for txt_row in data["caption_rows_of"][img_row]:
            txt_rows.append(txt_row)
            gt.append(img_row)

    queries = project_and_normalize(data["txt_feat"][txt_rows], proj_txt)
    return gallery, queries, torch.tensor(gt)


def make_epoch_batches(train_img_rows, caption_rows_of, batch_size, seed):
    """为一个 epoch 生成 batch：每个 batch 里每张图最多一条 caption。

    这样同一 batch 内不会出现「同一张图配两条 caption」，避免 InfoNCE 的假负样本。

    Args:
        train_img_rows: 训练集的图像行号列表。
        caption_rows_of: 图像行号 -> 该图 5 条 caption 行号的列表。
        batch_size: 每个 batch 的大小。
        seed: 随机种子，每个 epoch 传不同的值。
    Returns:
        batches: 列表，每个元素是 batch_size 个 (图像行号, caption 行号)。
    """
    rng = random.Random(seed)  # 独立的随机数生成器，不影响全局随机状态
    # 每张图随机抽 1 条 caption
    pairs = [
        (img_row, rng.choice(caption_rows_of[img_row])) for img_row in train_img_rows
    ]
    rng.shuffle(pairs)

    batches = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        if len(batch) == batch_size:  # 丢掉最后不足一个 batch 的尾巴
            batches.append(batch)
    return batches