import torch
from torch.nn import functional as F
from ANN import Index

N, D = 31014, 256
gallery = F.normalize(torch.randn(N, D), dim=1)
query = gallery[:5]

index = Index(d=D, m=32, efConstruction=200, efSearch=64)

index.add(gallery)

dist, ids = index.search(query, k=5)

print(ids[:, 0])  # 每个查询的第 1 名编号
print(dist[:, 0])  # 对应的距离
