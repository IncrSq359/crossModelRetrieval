import torch
import time
import numpy as np
from ANN import Index
from proj_lib import (
    load_data,
    load_projection,
    evaluate,
    project_and_normalize,
    recall_t2i,
)

# 读缓存特征和索引表
data = load_data()

proj_img, proj_txt = load_projection("output/models/projection_flickr30k.pth")
recalls = evaluate(proj_img, proj_txt, data, data["val_img_rows"])

gallery = project_and_normalize(data["img_feat"], proj_img)

txt_rows = []
gt = []
for img_row in data["val_img_rows"]:
    for txt_row in data["caption_rows_of"][img_row]:
        txt_rows.append(txt_row)
        gt.append(img_row)
gt = torch.tensor(gt)

queries = project_and_normalize(data["txt_feat"][txt_rows], proj_txt)

sim = queries @ gallery.T
brute_top10 = sim.topk(10, dim=1).indices

hit = (brute_top10 == gt.unsqueeze(1)).any(dim=1)
brute_recall = hit.float().mean().item()

val_rows = torch.tensor(data["val_img_rows"])  # 1014 个图库行号
sub_sim = sim[:, val_rows]

sub_idx = sub_sim.topk(10, dim=1).indices

sub_top10 = val_rows[sub_idx]

# ---------- HNSW：一组参数 ----------
M = 16  # 每点邻居数
EF_CONSTRUCTION = 200  # 建图候选数
EF_SEARCH = 64  # 查询候选数
TOP_K = 10

# ---------- 2. 转成 faiss 需要的格式 ----------
# faiss 只接受 float32 且内存连续的 numpy 数组
gallery_np = np.ascontiguousarray(gallery.cpu().numpy().astype(np.float32))
queries_np = np.ascontiguousarray(queries.cpu().numpy().astype(np.float32))

# ---------- 3. 建索引并计时 ----------
index = Index(gallery_np.shape[1], M, EF_CONSTRUCTION, EF_SEARCH)

t0 = time.perf_counter()
index.add(gallery_np)
build_time = time.perf_counter() - t0
print(f"建索引耗时: {build_time:.2f} s")

# ---------- 4. 批量查询并计时 ----------
t0 = time.perf_counter()
D, I = index.search(queries_np, TOP_K)
search_time = time.perf_counter() - t0
print(
    f"查询总耗时: {search_time:.3f} s，平均每条 {search_time / len(queries_np) * 1000:.3f} ms"
)

# ---------- 5. 指标一：和暴力检索 top10 的重合率 ----------
brute_np = (
    brute_top10.cpu().numpy()
    if hasattr(brute_top10, "cpu")
    else np.asarray(brute_top10)
)
overlap = np.mean([len(set(I[i]) & set(brute_np[i])) / TOP_K for i in range(len(I))])
print(f"与暴力检索 top10 的重合率: {overlap:.4f}")

# ---------- 6. 指标二：Recall@10（top10 里有没有正确图片） ----------
gt_np = gt.cpu().numpy() if hasattr(gt, "cpu") else np.asarray(gt)
hit = (I == gt_np[:, None]).any(axis=1)
print(f"HNSW Recall@10: {hit.mean():.4f}（暴力检索基线 0.1456）")

check = (sub_top10 == gt.unsqueeze(1)).any(dim=1).float().mean().item()
print(f"限定 val 后 Recall@10 = {check:.4f}  （应为 0.6434）")

print(gallery.shape, queries.shape)
assert queries.shape[0] == len(gt) == 5070

print(sim.shape, brute_top10.shape)
# 期望 [5070, 31014]  [5070, 10]

print(f"暴力检索 Recall@10（全图库 31014 张）= {brute_recall:.4f}")

print(f"T2I@10 = {recalls['T2I@10']:.4f}  （训练时 0.6434）")
print(f"I2T@10 = {recalls['I2T@10']:.4f}  （训练时 0.7525）")
