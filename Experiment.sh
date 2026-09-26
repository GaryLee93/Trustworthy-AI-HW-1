set -e

# ========== 用法 ==========
# ./baseline.sh <PROJECT_DIR> <OUTPUT_DIR>
# 例如: ./baseline.sh ~/Trustworthy-AI-HW-1 /vault/baseline
# ==========================
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "用法: $0 <PROJECT_DIR> <OUTPUT_DIR>" >&2
    exit 1
fi

PROJECT_DIR="$1"
OUTPUT_DIR="$2"

cd "${PROJECT_DIR}"
source /tmp/b11902090/miniconda3/etc/profile.d/conda.sh
conda activate minigpt
cd "${PROJECT_DIR}/minimind/trainer"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/checkpoints"
ln -sfn "${OUTPUT_DIR}" "${PROJECT_DIR}/minimind/out"
ln -sfn "${OUTPUT_DIR}/checkpoints" "${PROJECT_DIR}/minimind/checkpoints"

swanlab login -k iXDM58vG5zuP1CHeKtVPh

python3 train_pretrain.py \
    --save_dir "${OUTPUT_DIR}" \
    --log_interval 1000 \
    --save_interval 2000 \
    --data_path ../dataset/zhtw_wikipedia_pretrain.jsonl \
    --seed 824 \
    --use_wandb \
    --wandb_project "Trustworthy-AI-HW-1"

python3 train_full_sft.py \
    --save_dir "${OUTPUT_DIR}" \
    --log_interval 500 \
    --save_interval 1000 \
    --data_path ../dataset/tmmluplus_sft.jsonl \
    --epochs 5 \
    --seed 824 \
    --use_wandb \
    --wandb_project "Trustworthy-AI-HW-1"