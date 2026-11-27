#!/usr/bin/env bash
# 训练启动脚本（云端：25 核 / RTX 5090 32GB / 90GB 内存 / Linux）
#
#   bash scripts/train.sh                       # auto：Linux 上等价于 cloud-5090
#   bash scripts/train.sh cloud-5090
#   bash scripts/train.sh cloud-5090-long       # 更大 batch + 更长训练
#   bash scripts/train.sh cloud-5090 --epochs 30 --batch-size 384
#   bash scripts/train.sh smoke                 # 上机前 1 分钟验证环境
#   bash scripts/train.sh local-12g             # 12GB 显存机器
#
# 断点续训：
#   bash scripts/train.sh cloud-5090 --resume output/dual_tower/last.pt
set -euo pipefail

cd "$(dirname "$0")/.."

PRESET="auto"
# 第一个参数不是 -开头 就当作 preset（允许 bash scripts/train.sh --epochs 30）
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
  PRESET="$1"
  shift
fi

# 25 核：主进程 + 十几个 dataloader worker 都要吃 CPU，
# 同时给 BLAS 线程留几个核，避免它们和 worker 抢满
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
# 全部 caption 已在主进程里一次性预编码，worker 里不需要再并行分词
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "[launch] preset=${PRESET} cwd=$(pwd) cpus=$(nproc 2>/dev/null || echo '?')"
echo "[launch] disk free: $(df -h . | awk 'NR==2 {print $4}')"

exec python -u src/train_dual_tower.py --preset "${PRESET}" "$@"
