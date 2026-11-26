from re import search
from sys import flags
import time
import numpy as np
import torch
from ANN import Index
from utils.proj_lib import build_eval_set, load_projection, load_data

# ---------- 准备数据 ----------
data = load_data()
proj_img, proj_txt = load_projection("output/models/projection_flickr30k.pth")
gallery, queries, gt = build_eval_set(data, proj_img, proj_txt, data["val_img_rows"])

TOP_K = 10

gallery_np = np.ascontiguousarray(gallery.cpu().numpy().astype(np.float32))
queries_np = np.ascontiguousarray(queries.cpu().numpy().astype(np.float32))
gt_np = gt.cpu().numpy()

with torch.no_grad():
    brute_np = (queries @ gallery.T).topk(TOP_K, dim=1).indices.numpy()


def calc_overlap(found, reference, top_k):
    """计算近似检索结果与暴力检索结果的平均重合率。

    Args:
        found: 近似检索返回的行号，形状 [Q, top_k]。
        reference: 暴力检索返回的行号，形状 [Q, top_k]。
        top_k: 每条查询返回的结果数。

    Returns:
        所有查询的平均重合率，范围 0 到 1。
    """
    return float(
        np.mean(
            [len(set(found[i]) & set(reference[i])) / top_k for i in range(len(found))]
        )
    )


def calc_recall(found, gt_rows):
    """计算 Recall@K：返回的 top-K 里是否包含正确图片。

    Args:
        found: 检索返回的行号，形状 [Q, top_k]。
        gt_rows: 每条查询对应的正确图库行号，形状 [Q]。

    Returns:
        命中查询所占比例。
    """
    return float((found == gt_rows[:, None]).any(axis=1).mean())


def run_brute():
    """对全图库做暴力余弦检索并计时。

    Returns:
        字典，含 name、build_time、search_time、overlap、recall。
    """
    with torch.no_grad():
        t0 = time.perf_counter()
        top = (queries @ gallery.T).topk(TOP_K, dim=1).indices
        search_time = time.perf_counter() - t0
    return {
        "name": "brute",
        "build_time": 0.0,  # 暴力检索无需建索引
        "search_time": search_time,
        "overlap": 1.0,  # 和自己比，必为 1
        "recall": calc_recall(top.numpy(), gt_np),
    }


def build_hnsw(m, ef_construction):
    """建 HNSW 索引并计时。

    Args:
        m: 每个点的邻居数。
        ef_construction: 建图时的候选数。

    Returns:
        (index, build_time) 二元组。
    """
    # efSearch 这里随便给一个值，查询时会用 search 的临时参数覆盖
    index = Index(gallery_np.shape[1], m, ef_construction, 64)

    t0 = time.perf_counter()
    index.add(gallery_np)  # HNSW 在 add 时建图
    build_time = time.perf_counter() - t0
    return index, build_time


def query_hnsw(index, name, build_time, ef_search):
    """在已建好的索引上，用指定 efSearch 查询并计算指标。

    Args:
        index: build_hnsw 返回的索引。
        name: 结果里显示的前缀名。
        build_time: 建索引耗时，原样放进结果。
        ef_search: 本次查询的候选数。

    Returns:
        字典，含 name、build_time、search_time、overlap、recall。
    """
    t0 = time.perf_counter()
    _, found = index.search(queries_np, TOP_K, efSearch=ef_search)  # 临时覆盖 efSearch
    search_time = time.perf_counter() - t0

    return {
        "name": f"{name} efS={ef_search}",
        "build_time": build_time,
        "search_time": search_time,
        "overlap": calc_overlap(found, brute_np, TOP_K),
        "recall": calc_recall(found, gt_np),
    }


def show(r):
    """打印一行结果。

    Args:
        r: run_brute 或 run_hnsw 返回的字典。
    """
    per_query_ms = r["search_time"] / len(queries_np) * 1000
    print(
        f"{r['name']:<32} 建索引 {r['build_time']:.2f}s | "
        f"查询 {r['search_time']:.3f}s ({per_query_ms:.4f} ms/条) | "
        f"重合率 {r['overlap']:.4f} | Recall@10 {r['recall']:.4f}"
    )


if __name__ == "__main__":
    show(run_brute())

    # 索引只建一次，之后只换 efSearch
    index, build_time = build_hnsw(16, 200)
    for ef_s in [10, 16, 32, 64, 128, 256]:
        show(query_hnsw(index, "hnsw m=16 efC=200", build_time, ef_s))