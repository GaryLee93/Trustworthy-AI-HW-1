set -e

cd ~/Trustworthy-AI-HW-1
source .venv/bin/activate
cd ~/minimind/trainer
mkdir -p /vault/baseline
ln -sfn /vault/baseline ~/minimind/out
ln -sfn /vault/baseline/checkpoints ~/minimind/checkpoints
mkdir -p /vault/baseline/checkpoints

swanlab login -k iXDM58vG5zuP1CHeKtVPh

python3 train_pretrain.py \
    --save_dir /vault/baseline \
    --log_interval 10 \
    --seed 824 \
    --use_wandb \
    --wandb_project "Trustworthy-AI-HW-1"

python3 train_full_SFT.py \
    --save_dir /vault/baseline \
    --log_interval 10 \
    --seed 824 \
    --use_wandb \
    --wandb_project "Trustworthy-AI-HW-1"
