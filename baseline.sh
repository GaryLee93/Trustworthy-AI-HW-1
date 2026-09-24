set -e

cd ~/minimind/trainer
mkdir -p /vault/baseline
[ ! -e ~/minimind/out ] && ln -s /vault/baseline ~/minimind/out
[ ! -e ~/minimind/checkpoints ] && ln -s /vault/baseline/checkpoints ~/minimind/checkpoints
mkdir -p /vault/baseline/checkpoints

swanlab login -k iXDM58vG5zuP1CHeKtVPh

python3 train_pretrain.py \
    --save_dir /vault/baseline \
    --seed 824 \
    --use_wandb \
    --wandb_project "Trustworthy-AI-HW-1"

python3 train_full_SFT.py \
    --save_dir /vault/baseline \
    --seed 824 \
    --use_wandb \
    --wandb_project "Trustworthy-AI-HW-1"
