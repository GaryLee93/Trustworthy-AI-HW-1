set -e

PROJECT_DIR="$1"
OUTPUT_DIR="$2"

cd ~/Trustworthy-AI-HW-1
source /tmp/b11902090/miniconda3/etc/profile.d/conda.sh
conda activate minigpt
cd "${PROJECT_DIR}/minimind/trainer"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/checkpoints"
ln -sfn "${OUTPUT_DIR}" "${PROJECT_DIR}/minimind/out"
ln -sfn "${OUTPUT_DIR}/checkpoints" "${PROJECT_DIR}/minimind/checkpoints"
swanlab login -k iXDM58vG5zuP1CHeKtVPh

python3 train_full_sft.py \
    --save_dir "{$OUTPUT_DIR}" \
    --log_interval 1000 \
    --save_interval 2000 \
    --seed 824 \
    --from_resume 1\
    --use_wandb \
    --wandb_project "Trustworthy-AI-HW-1"