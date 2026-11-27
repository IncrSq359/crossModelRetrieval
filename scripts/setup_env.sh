#!/usr/bin/env bash
# 在云平台（Linux）创建本项目的 conda 环境。
#
#   bash scripts/setup_env.sh                    # 默认 CUDA 13.0 的 wheel（cu130）
#   CUDA_TAG=cu128 bash scripts/setup_env.sh     # 驱动只支持 CUDA 12.x 时
#   ENV_NAME=cmr bash scripts/setup_env.sh       # 换环境名
#   WITH_HNSW=1 bash scripts/setup_env.sh        # 顺便编译 src/ANN 的 C++ 扩展
#
# 为什么不用 requirements.txt：那是 conda 在 Windows 上 `conda env list --export`
# 导出的显式清单（platform: win-64、cuda 13.5 的 win 构建、torch 标注为 pypi_0），
# 在 linux-64 上无法解析，也不能用 `conda env create -f` 安装。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-cmr}"
CUDA_TAG="${CUDA_TAG:-cu130}"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
WITH_HNSW="${WITH_HNSW:-0}"

echo "==> 项目目录: $ROOT"
echo "==> 环境名: $ENV_NAME   torch 索引: $TORCH_INDEX"

# ---------- 1. conda 是否可用 ----------
if ! command -v conda >/dev/null 2>&1; then
  for p in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    if [ -f "$p/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1090
      . "$p/etc/profile.d/conda.sh"
      break
    fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  cat <<'EOF'
[错误] 找不到 conda。先装一个 Miniconda（不需要 root）：

  curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
  "$HOME/miniconda3/bin/conda" init bash

然后再跑一次: bash scripts/setup_env.sh
EOF
  exit 1
fi
eval "$(conda shell.bash hook)"

# ---------- 2. 创建 / 更新环境 ----------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> 环境 $ENV_NAME 已存在，更新 python/pip（不加 --prune，避免误删已装的包）"
  conda env update -n "$ENV_NAME" -f environment.yml
else
  echo "==> 创建环境 $ENV_NAME"
  conda env create -n "$ENV_NAME" -f environment.yml
fi
conda activate "$ENV_NAME"
python -V
pip -V

# ---------- 3. torch（必须按 CUDA 版本装） ----------
# 不写死 torchvision 版本，让 pip 按 torch 的 Requires-Dist 自动配对的版本
echo "==> 安装 torch / torchvision（$CUDA_TAG）"
pip install --index-url "$TORCH_INDEX" torch
pip install --index-url "$TORCH_INDEX" torchvision

# ---------- 4. 其余依赖 ----------
echo "==> 安装其余依赖"
pip install -r requirements-linux.txt

# ---------- 5. 自检 ----------
echo "==> 环境自检"
python - <<'PY'
import sys
print("python    ", sys.version.split()[0])

import torch
print("torch     ", torch.__version__, "| cuda runtime", torch.version.cuda)
avail = torch.cuda.is_available()
print("cuda avail", avail)
if not avail:
    print("[warn] 没检测到 GPU：确认驱动、以及装的是 CUDA 版 torch")
else:
    p = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"gpu       {p.name} {p.total_memory / 2**30:.1f}GB  sm_{cap[0]}{cap[1]}")
    print("bf16      ", torch.cuda.is_bf16_supported())
    a = torch.randn(512, 512, device="cuda")
    ok = float((a @ a).sum()) != 0.0
    print("matmul    ", "ok" if ok else "failed")
    if not ok:
        sys.exit("GPU kernel 跑不通")
    import torch.nn.functional as F
    with torch.autocast("cuda", dtype=torch.bfloat16):
        F.layer_norm(a, (512,))
    print("autocast   ok")

import transformers
print("transformers", transformers.__version__)
try:
    import faiss
    print("faiss     ", getattr(faiss, "__version__", "?"))
except ImportError:
    print("faiss      未安装（只影响 HNSW 对比实验）")
PY

# ---------- 6. 可选：编译 C++ HNSW 扩展 ----------
if [ "$WITH_HNSW" = "1" ]; then
  echo "==> 编译 hnsw_cpp"
  if cmake -S . -B build -DPython3_EXECUTABLE="$(command -v python)" \
       && cmake --build build -j "$(nproc 2>/dev/null || echo 4)"; then
    echo "    生成 src/ANN/hnsw_cpp*.so"
  else
    echo "[warn] hnsw_cpp 编译失败；训练双塔不需要它，可稍后再处理"
  fi
fi

cat <<EOF

============================================================
完成。后续步骤：

  conda activate $ENV_NAME
  bash scripts/train.sh smoke      # 1 分钟验证训练链路
  bash scripts/train.sh            # 正式训练（Linux 默认 cloud-5090 预设）

跑之前确认这些文件在位（数据不入库，需要自己上传）：
  data/annotations/captions_flickr30k.csv
  data/raw/flickr30k/flickr30k-images/          (31783 张，约 4.4GB)
  data/raw/convnext_tiny_1k_224_ema.pth         (114MB)

BERT 权重（bert-base-uncased，约 440MB）首次运行会自动下载；
离线机器请先把 ~/.cache/huggingface 拷过来并设 HF_HUB_OFFLINE=1。
============================================================
EOF
