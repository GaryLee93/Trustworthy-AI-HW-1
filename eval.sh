#!/bin/bash
set -e

source /tmp/b11902090/miniconda3/etc/profile.d/conda.sh
conda activate minigpt

OUTPUT_DIR="$1"

cd /tmp/b11902090/Trustworthy-AI-HW-1/minimind/scripts
python3 convert_model.py

mkdir -p "${OUTPUT_DIR}"/eval
cd /tmp/b11902090/Trustworthy-AI-HW-1
python3 eval_tmmluplus.py eval \
    --model_path /tmp/b11902090/Trustworthy-AI-HW-1/minimind-3 \
    --save_predictions "${OUTPUT_DIR}"/eval/p.json \
    --output "${OUTPUT_DIR}"/eval/eval.json

python3 eval_tmmluplus.py eval \
    --model_path /tmp/b11902090/Trustworthy-AI-HW-1/minimind-3 \
    --chat_template \
    --save_predictions "${OUTPUT_DIR}"/eval/p_template.json \
    --output "${OUTPUT_DIR}"/eval/eval_template.json

python3 eval_tmmluplus.py eval \
    --model_path /tmp/b11902090/Trustworthy-AI-HW-1/minimind-3 \
    --num_fewshot 5 \
    --save_predictions "${OUTPUT_DIR}"/eval/p_fewshot.json \
    --output "${OUTPUT_DIR}"/eval/eval_fewshot.json

python3 eval_tmmluplus.py eval \
    --model_path /tmp/b11902090/Trustworthy-AI-HW-1/minimind-3 \
    --chat_template \
    --num_fewshot 5 \
    --save_predictions "${OUTPUT_DIR}"/eval/p_template_fewshot.json \
    --output "${OUTPUT_DIR}"/eval/eval_fewshot_template.json