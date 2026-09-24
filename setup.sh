#!/bin/bash
# ========== 用法 ==========
# ./setup.sh <PROJECT_DIR>
# 例如: ./setup.sh ~/Trustworthy-AI-HW-1
# ==========================

set -e

if [ -z "$1" ]; then
    echo "用法: $0 <PROJECT_DIR>" >&2
    exit 1
fi

PROJECT_DIR="$1"

echo "=== 安裝基礎工具 ==="
cd "${PROJECT_DIR}"
sudo apt update
sudo apt install -y build-essential git git-lfs nvtop

echo "=== 確定submodule裡的minimind有沒有裝好 ==="
git submodule update --init --recursive

echo "=== 建立虛擬環境 ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== 偵測 GPU 與驅動資訊 ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "找不到 nvidia-smi,此機器可能沒有 NVIDIA GPU 或驅動未安裝,改裝 CPU 版本"
    pip install torch torchvision torchaudio --break-system-packages
    exit 0
fi

# 抓出 nvidia-smi 顯示的驅動最高支援 CUDA 版本,例如 "12.6"
DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')

if [ -z "$DRIVER_CUDA" ]; then
    echo "無法解析 CUDA 版本,請手動檢查 nvidia-smi 輸出"
    exit 1
fi

echo "偵測到驅動支援的最高 CUDA 版本: $DRIVER_CUDA"

# 把版本轉成整數方便比較,例如 12.6 -> 126, 12.8 -> 128
DRIVER_CUDA_INT=$(echo "$DRIVER_CUDA" | awk -F. '{printf "%d%02d", $1, $2}')

# 依驅動支援上限,由高到低挑選對應的 PyTorch wheel(可依 pytorch.org 最新清單增減)
if [ "$DRIVER_CUDA_INT" -ge 128 ]; then
    PT_TAG="cu128"
elif [ "$DRIVER_CUDA_INT" -ge 126 ]; then
    PT_TAG="cu126"
elif [ "$DRIVER_CUDA_INT" -ge 118 ]; then
    PT_TAG="cu118"
else
    echo "驅動版本過舊($DRIVER_CUDA),PyTorch 官方可能已不支援,建議先更新驅動"
    exit 1
fi

echo "=== 安裝對應版本的 PyTorch: $PT_TAG ==="
pip install torch --index-url https://download.pytorch.org/whl/${PT_TAG}

echo "=== 驗證安裝 ==="
python3 -c "
import torch
print('PyTorch 版本:', torch.__version__)
print('CUDA 是否可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU 型號:', torch.cuda.get_device_name(0))
"

pip install "transformers>=4.44" datasets accelerate huggingface_hub

echo "=== 安裝minimind套件 ==="
cd "${PROJECT_DIR}/minimind"
VERSION_PIN=$(grep -inE '^\s*(torch|torchvision|nvidia|triton)' requirements.txt || true)
if [ -z "$VERSION_PIN" ]; then
    pip install -r requirements.txt
else
    grep -ivE '^\s*(torch|torchvision|torchaudio|nvidia|triton)' requirements.txt > /tmp/req.txt
    pip install -r /tmp/req.txt
fi
python3 -c 'import torch; print("請確認是否與git clone前安裝的版本相同" ,torch.__version__, torch.cuda.is_available())'

echo "=== 冒煙測試，看一下能不能動 ==="
cd "${PROJECT_DIR}"
hf download jingyaogong/minimind-3 --local-dir "${PROJECT_DIR}/minimind-3"
python3 eval_tmmluplus.py eval --model_path ./minimind-3 --limit 20 --output smoke.json

mkdir -p "${PROJECT_DIR}/minimind/dataset"
cd "${PROJECT_DIR}/minimind/dataset"
hf download jingyaogong/minimind_dataset --repo-type dataset \
    --include "pretrain_t2t_mini.jsonl" "sft_t2t_mini.jsonl" --local-dir .
ls -lh