import faiss
import faiss.contrib.torch_utils
import numpy as np
from torch import Tensor

class Index():
    def __init__(self,d,m,efConstruction,efSearch):
        self.index = faiss.IndexHNSWFlat(d,m)
        self.index.hnsw.efConstruction = efConstruction
        self.index.hnsw.efSearch = efSearch
    
    def add(self,x:Tensor):
        self.index.add(x)
        
    def search(self,q:Tensor,k):
        D,I = self.index.search(q,k)
        return D,I
        