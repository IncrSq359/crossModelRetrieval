"""检查图像/文本特征缓存是否完整、对齐。"""
from collections import Counter

import torch

IMG_PATH = "output/features/image_features.pth"
TXT_PATH = "output/features/text_features_v2.pth"

# 加载两份缓存（图像 features 与文本 mean / cls / filenames）
img = torch.load(IMG_PATH)
txt = torch.load(TXT_PATH)

# 1. 形状检查
print("图像特征:", img["features"].shape)   # 期望 [31014, 768]
print("文本 CLS 特征:", txt["cls"].shape)   # 期望 [155070, 768]
print("文本 mean 特征:", txt["mean"].shape) # 期望 [155070, 768]

# 2. 数值检查：有 nan 说明提取过程出过问题
print("图像特征含 nan:", torch.isnan(img["features"]).any().item())   # 期望 False
print("文本 CLS 特征含 nan:", torch.isnan(txt["cls"]).any().item())   # 期望 False
print("文本 mean 特征含 nan:", torch.isnan(txt["mean"]).any().item()) # 期望 False

# 3. 行数与文件名列表长度必须一致（features 和 filenames 是按行对齐的）
print("图像行数 == 文件名数:", img["features"].shape[0] == len(img["filenames"]))   # 期望 True
print("文本 CLS 行数 == 文件名数:", txt["cls"].shape[0] == len(txt["filenames"]))   # 期望 True
print("文本 mean 行数 == 文件名数:", txt["mean"].shape[0] == len(txt["filenames"])) # 期望 True
print("CLS / mean 特征形状一致:", txt["cls"].shape == txt["mean"].shape)  # 期望 True

# 4. 对齐检查：文本侧每个文件名都必须能在图像侧找到
img_index = {fn: i for i, fn in enumerate(img["filenames"])}   # 文件名 -> 图像特征行号
missing = [fn for fn in txt["filenames"] if fn not in img_index]
print("文本侧在图像侧找不到的文件名数:", len(missing))   # 期望 0

# 5. 每张图应恰好有 5 条 caption
counts = Counter(txt["filenames"])
print("图片数:", len(counts))   # 期望 31014
print("每张图都恰好 5 条:", all(c == 5 for c in counts.values()))   # 期望 True