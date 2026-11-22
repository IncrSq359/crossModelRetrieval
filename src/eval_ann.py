import torch
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

val_rows = torch.tensor(data["val_img_rows"])   # 1014 个图库行号
sub_sim = sim[:, val_rows]

sub_idx = sub_sim.topk(10, dim=1).indices

sub_top10 = val_rows[sub_idx]

check = (sub_top10 == gt.unsqueeze(1)).any(dim=1).float().mean().item()
print(f"限定 val 后 Recall@10 = {check:.4f}  （应为 0.6434）")

print(gallery.shape, queries.shape)
assert queries.shape[0] == len(gt) == 5070

print(sim.shape, brute_top10.shape)
# 期望 [5070, 31014]  [5070, 10]

print(f"暴力检索 Recall@10（全图库 31014 张）= {brute_recall:.4f}")

print(f"T2I@10 = {recalls['T2I@10']:.4f}  （训练时 0.6434）")
print(f"I2T@10 = {recalls['I2T@10']:.4f}  （训练时 0.7525）")
