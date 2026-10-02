#!/bin/bash
set -e

git clone https://github.com/GaryLee93/Trustworthy-AI-HW-1.git
cd Trustworthy-AI-HW-1
git submodule update --init --recursive

CONDA_DIR="${CONDA_DIR:-$(dirname "$(pwd)")/miniconda3}"
export TMPDIR="$(dirname "$CONDA_DIR")/tmp"
mkdir -p "$TMPDIR"
# 以 conda.sh 是否存在判斷是否已安裝(目錄存在但為空或安裝不完整時也要重新安裝)
if [ ! -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    # -u: 目錄已存在時仍允許安裝進去，否則安裝程式會因目錄已存在而報錯
    # 清理暫存檔失敗時安裝程式會回傳非 0,但 base 環境其實已裝好，因此改以 conda.sh 是否存在判斷成功與否
    bash miniconda.sh -b -u -p "$CONDA_DIR" || true
    rm -f miniconda.sh
    if [ ! -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
        echo "Miniconda 安裝失敗:$CONDA_DIR"
        exit 1
    fi
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda create -y -n minigpt -c conda-forge --override-channels python=3.11
conda activate minigpt

if ! command -v nvidia-smi &> /dev/null; then
    echo "找不到 nvidia-smi,此機器可能沒有 NVIDIA GPU 或驅動未安裝,改裝 CPU 版本"
    pip install torch torchvision torchaudio
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
cd minimind
VERSION_PIN=$(grep -inE '^\s*(torch|torchvision|nvidia|triton)' requirements.txt || true)
if [ -z "$VERSION_PIN" ]; then
    pip install -r requirements.txt
else
    grep -ivE '^\s*(torch|torchvision|torchaudio|nvidia|triton)' requirements.txt > /tmp/req.txt
    pip install -r /tmp/req.txt
fi
python3 -c 'import torch; print("請確認是否與git clone前安裝的版本相同" ,torch.__version__, torch.cuda.is_available())'

echo "=== 冒煙測試，看一下能不能動 ==="
cd ..
hf download jingyaogong/minimind-3 --local-dir ./minimind-3
python3 eval_tmmluplus.py eval --model_path ./minimind-3 --limit 20 --output smoke.json

echo "=== 安裝和處理資料集 ==="
python prepare_wiki_pretrain.py \
    --target_mb 2048 \
    --seed 824 \
    --min_length 150 \
    --max_chars 400 \
    --out_path minimind/dataset/zhtw_wikipedia_pretrain.jsonl

python prepare_tmmluplus_sft.py \
    --seed 824 \
    --holdout_frac 0.2 \
    --revision v1.1 \
    --holdout_out_path minimind/dataset/tmmluplus_holdout_check.jsonl \
    --out_path minimind/dataset/tmmluplus_sft.jsonl

echo "=== 開始訓練，模型會存在Trustworthy-AI-HW-1/minimind/out ==="
cd minimind/trainer
python3 train_pretrain.py \
    --log_interval 1000 \
    --save_interval 2000 \
    --max_seq_len 500 \
    --data_path ../dataset/zhtw_wikipedia_pretrain.jsonl \
    --seed 824 \

python3 train_full_sft.py \
    --log_interval 500 \
    --save_interval 1000 \
    --data_path ../dataset/tmmluplus_sft.jsonl \
    --epochs 10 \
    --seed 824 \

