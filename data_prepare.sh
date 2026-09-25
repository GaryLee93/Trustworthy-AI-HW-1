source /tmp/b11902090/miniconda3/etc/profile.d/conda.sh
conda activate minigpt

python prepare_tmmluplus_pretrain.py \
    --seed 824 \
    --holdout_frac 0.2 \
    --revision v1.1 \
    --out_dir ../dataset