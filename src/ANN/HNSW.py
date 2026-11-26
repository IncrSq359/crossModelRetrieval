import torch

from . import hnsw_cpp

import faiss
import faiss.contrib.torch_utils
from torch import Tensor

class Index():
    
    def __init__(self,d,m,efConstruction,efSearch,c=0,flag=0):
        """基于Faiss的HNSW构建
            Args:
                d: 向量长度
                m: 每个节点的最大相邻点数
                efConstruction: 构建索引中的候选池大小
                efSearch: 搜索时的候选池大小
                c: 0时选择faiss版本,1时选择自实现版本
                flag: 选择自实现版本可用，0时采用欧氏距离，1时采用余弦距离
        """
        self.c = c
        if(c==0):
            self.index = faiss.IndexHNSWFlat(d,m)
            self.index.hnsw.efConstruction = efConstruction
            self.index.hnsw.efSearch = efSearch
        else:
            self.index = hnsw_cpp.HNSWindex(d,m,efConstruction,efSearch,flag)
    
    def add(self,x:Tensor):
        if(self.c==0):
            self.index.add(x)
        else:
            self.index.Insert(x)

    def search(self,q:Tensor,k,efSearch=-1):
        """
        Args:
            q (Tensor): 查询矩阵
            k (int): top-k
            efSearch (int, optional): 临时更改搜索候选池大小. Defaults to -1.

        Returns:
            返回nq*k矩阵D, I, nq为查询矩阵的行数\n
            I中第i行为第i个查询返回的k个向量id,按距离升序排序\n
            D为对应的向量与查询向量的平方距离
        """
        if(self.c==0):
            originefs = self.index.hnsw.efSearch
            if(efSearch!=-1):
                self.index.hnsw.efSearch=efSearch
            D,I = self.index.search(q,k)
            self.index.hnsw.efSearch=originefs
        else:
            D,I = self.index.Search(torch.as_tensor(q).float().cpu(),k,efSearch)
        return D,I
        