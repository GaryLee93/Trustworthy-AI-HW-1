source /tmp/b11902090/miniconda3/etc/profile.d/conda.sh
conda activate minigpt

python prepare_zhtw_wikipedia_pretrain.py \
    --target_mb 2048 \
    --seed 824 \
    --min_length 100 \
    --out_path /tmp/Trustworthy-AI-HW-1/minimind/dataset/zhtw_wikipedia_pretrain.jsonl

python prepare_tmmluplus_sft.py \
    --seed 824 \
    --holdcut_frec 0.2 \
    --revision 1.1 \
    --holdout_out_path /tmp/Trustworthy-AI-HW-1/minimind/dataset/tmmluplus_holdout_check.jsonl \
    --out_path /tmp/Trustworthy-AI-HW-1/minimind/dataset/tmmluplus_sft.jsonl