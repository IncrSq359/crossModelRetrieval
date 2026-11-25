"""检索模块：在图库特征矩阵上做暴力余弦检索。"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
import torch
from ANN import Index


class BruteForceRetriever:
    """暴力检索器：与整个图库逐行做点积，取分数最高的 K 个。"""

    def __init__(self, gallery):
        """
        Args:
            gallery: 图库特征矩阵 [N, 256]，每行已投影并 L2 归一化。
        """
        self.gallery = gallery

    def search_vector(self, q_vec, top_k):
        """检索与查询向量最相似的 top_k 张图。

        Args:
            q_vec: 查询向量 [256]，已 L2 归一化。
            top_k: 返回的结果数。
        Returns:
            (indices, scores)：两个 [top_k] 的张量，行号与分数，按分数降序。
        """
        with torch.no_grad():
            scores = self.gallery @ q_vec
            values, indices = torch.topk(scores, top_k)
            return indices, values


class HNSWRetriever:
    """HNSW 近似检索器：接口与 BruteForceRetriever 完全一致。"""

    def __init__(self, gallery, m=16, ef_construction=200, ef_search=64):
        """
        Args:
            gallery: 图库特征 [N, 256]，已投影并 L2 归一化。
            ...
        """
        d = gallery.shape[1]
        self.index = Index(d, m, ef_construction, ef_search)
        g = gallery.contiguous().float()
        self.index.add(g)

    def search_vector(self, q_vec, top_k):
        q = q_vec.unsqueeze(0)
        D, I = self.index.search(q, top_k)
        indices = I[0]
        scores = 1 - D[0] / 2
        return indices, scores
