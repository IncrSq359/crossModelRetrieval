"""构建 HNSW 的 Python 扩展模块 hnsw_cpp。

用法（在 src/ANN 目录下）::
    python setup.py build_ext --inplace

需要 pybind11 与 torch：bindings.cpp 里用到 at::Tensor，
所以必须链接 torch 的 C++ 库，否则链接阶段会报未解析的外部符号。
"""

import sys
from pathlib import Path

import torch
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
from torch.utils.cpp_extension import include_paths as torch_include_paths

IS_WINDOWS = sys.platform == "win32"

torch_lib_dir = str(Path(torch.__file__).resolve().parent / "lib")

# MSVC 不认 -O3（只会产生 D9002 警告），Release 下 setuptools 自带 /O2；
# torch 头文件会刷大量 C4251/C4275，顺手静音。
if IS_WINDOWS:
    extra_compile_args = ["/O2", "/wd4251", "/wd4275"]
    runtime_library_dirs = []
else:
    extra_compile_args = ["-O3"]
    runtime_library_dirs = [torch_lib_dir]

ext_modules = [
    Pybind11Extension(
        "hnsw_cpp",
        ["bindings.cpp"],
        # 让头文件变更也能触发重编译（setuptools 默认只看 .cpp 的 mtime）
        depends=["HNSW.hpp"],
        # 由 torch 自己给出 include/lib 路径，避免写死某个 conda 环境
        include_dirs=list(torch_include_paths()),
        library_dirs=[torch_lib_dir],
        libraries=["torch_python", "torch", "torch_cpu", "c10"],
        runtime_library_dirs=runtime_library_dirs,
        cxx_std=20,
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="hnsw_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)