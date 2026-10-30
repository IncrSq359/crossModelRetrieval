"""检索模块：在图库特征矩阵上做暴力余弦检索。"""
import torch

class BruteForceRetriever:
    """暴力检索器：与整个图库逐行做点积，取分数最高的 K 个。"""
    
    def __init__(self,gallery):
        """
        Args:
            gallery: 图库特征矩阵 [N, 256]，每行已投影并 L2 归一化。
        """
        self.gallery=gallery   

    def search_vector(self,q_vec,top_k):
        """检索与查询向量最相似的 top_k 张图。

        Args:
            q_vec: 查询向量 [256]，已 L2 归一化。
            top_k: 返回的结果数。
        Returns:
            (indices, scores)：两个 [top_k] 的张量，行号与分数，按分数降序。
        """
        with torch.no_grad():
            scores=self.gallery@q_vec
            values,indices=torch.topk(scores,top_k)
            return indices,values