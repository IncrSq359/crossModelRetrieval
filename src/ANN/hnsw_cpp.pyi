"""hnsw_cpp 原生扩展的存根。

Pyright/Pylance 不索引 .pyd，只有同目录下放一个 .pyi 才能解析
`from . import hnsw_cpp` 的类型和补全。
"""

import torch
from numpy import ndarray

class HNSWindex:
    def __init__(
        self, dim: int, m: int, efConstruction: int, efSearch: int, flag: int = 0
    ) -> None: ...
    def Insert(self, nq: ndarray) -> None: ...
    def Search(
        self, q: torch.Tensor, k: int, efSearch: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

# 兼容旧名字，和 bindings.cpp 里的 m.attr("HNSW") 对应
HNSW = HNSWindex
