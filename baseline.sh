set -e

cd ~/Trustworthy-AI-HW-1
source .venv/bin/activate
cd ~/Trustworthy-AI-HW-1/minimind/trainer
mkdir -p /vault/baseline
mkdir -p /vault/baseline/checkpoints
ln -sfn /vault/baseline ~/Trustworthy-AI-HW-1/minimind/out
ln -sfn /vault/baseline/checkpoints ~/Trustworthy-AI-HW-1/minimind/checkpoints

swanlab login -k iXDM58vG5zuP1CHeKtVPh

python3 train_pretrain.py \
    --save_dir /vault/baseline \
    --log_interval 1000 \
    --save_interval 2000 \
    --seed 824 \
    --use_wandb \
    --wandb_project "Trustworthy-AI-HW-1"

python3 train_full_sft.py \
    --save_dir /vault/baseline \
    --log_interval 1000 \
    --save_interval 2000 \
    --seed 824 \
    --use_wandb \
    --wandb_project "Trustworthy-AI-HW-1"
