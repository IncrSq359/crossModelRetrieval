import torch
from proj_lib import project_and_normalize

def recall_t2i(sim, gt, k):
    """文搜图的 Recall@K：正确图片是否排进前 K 名。

    Args:
        sim: 相似度矩阵 [num_caption, num_img]。
        gt: 每条 caption 的正确图片列号 [num_caption]。
        k: 取前几名。
    Returns:
        命中率（0 到 1 的浮点数）。
    """
    topk_idx = sim.topk(k, dim=1).indices  # 每条 caption 的前 K 张图
    hit = (topk_idx == gt.unsqueeze(1)).any(dim=1)
    return hit.float().mean().item()


def ndcg_t2i(sim, gt, k=10):
    """文搜图的 NDCG@K（每条文字只有 1 张正确图片）。

    Args:
        sim: [n_txt, n_img] 相似度矩阵，第 i 行是第 i 条文字对所有图片的分数。
        gt: [n_txt] 每条文字对应的正确图片的列号。
    Returns:
        NDCG@K 的平均值（float）。
    """
    rows = torch.arange(len(gt))
    correct_score = sim[rows, gt]
    bigger = sim > correct_score.unsqueeze(1)
    rank = bigger.sum(dim=1) + 1
    gain = 1.0 / torch.log2(rank.float() + 1)
    gain = torch.where(rank <= k, gain, torch.zeros_like(gain))
    return gain.mean().item()


def recall_i2t(sim, gt, k):
    """图搜文的 Recall@K：前 K 名 caption 里是否有属于这张图的。

    Args:
        sim: 相似度矩阵 [num_caption, num_img]。
        gt: 每条 caption 对应的图片列号 [num_caption]。
        k: 取前几名。
    Returns:
        命中率（0 到 1 的浮点数）。
    """
    topk_idx = sim.T.topk(k, dim=1).indices  # 每张图的前 K 条 caption 的序号
    pred_img = gt[topk_idx]  # 这些 caption 各自属于哪张图
    self_pos = torch.arange(sim.shape[1]).unsqueeze(1)  # 每张图自己的列号
    hit = (pred_img == self_pos).any(dim=1)
    return hit.float().mean().item()


def ndcg_i2t(sim, gt, k=10):
    """图搜文的 NDCG@K（每张图有 5 条正确 caption）。

    Args:
        sim: [num_caption, num_img] 相似度矩阵。
        gt: [num_caption] 每条 caption 对应的图片列号。
        k: 只统计前 k 名。
    Returns:
        NDCG@K 的平均值（float）。
    """
    topk_idx = sim.T.topk(k, dim=1).indices
    pred_img = gt[topk_idx]
    self_pos = torch.arange(sim.shape[1]).unsqueeze(1)
    topk_rel = (pred_img == self_pos).float()
    ranks = torch.arange(1, k + 1).float()
    discount = 1.0 / torch.log2(ranks.float() + 1)
    dcg = (topk_rel * discount).sum(dim=1)
    n_rel = sim.shape[0] // sim.shape[1]
    ideal_hits = min(n_rel, k)
    idcg = discount[:ideal_hits].sum()
    ndcg = dcg / idcg
    return ndcg.mean().item()


def evaluate(proj_img, proj_txt, data, img_rows_subset):
    """对一批图像及其 caption 计算 Recall（候选池 = 这批图像）。

    Args:
        proj_img: 图像投影层。
        proj_txt: 文本投影层。
        data: load_data 返回的字典。
        img_rows_subset: 参与评估的图像行号列表。
    Returns:
        字典，键如 "T2I@1"、"I2T@10"。
    """
    proj_img.eval()
    proj_txt.eval()
    pos = {img_row: p for p, img_row in enumerate(img_rows_subset)}
    txt_rows = []
    gt = []
    for img_row in img_rows_subset:
        for txt_row in data["caption_rows_of"][img_row]:
            txt_rows.append(txt_row)
            gt.append(pos[img_row])  # 这条 caption 的正确图在候选池里的位置

    with torch.no_grad():  # 评估不需要梯度
        img_emb = project_and_normalize(data["img_feat"][img_rows_subset], proj_img)
        txt_emb = project_and_normalize(data["txt_feat"][txt_rows], proj_txt)

    sim = txt_emb @ img_emb.T  # [num_caption, num_img]
    gt = torch.tensor(gt)

    recalls = {}
    for k in (1, 5, 10):
        recalls[f"T2I@{k}"] = recall_t2i(sim, gt, k)
        recalls[f"I2T@{k}"] = recall_i2t(sim, gt, k)
    recalls["T2I-NDCG@10"] = ndcg_t2i(sim, gt, 10)
    recalls["I2T-NDCG@10"] = ndcg_i2t(sim, gt, 10)
    return recalls
