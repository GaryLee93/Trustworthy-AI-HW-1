#!/bin/bash
set -e

source /tmp/b11902090/miniconda3/etc/profile.d/conda.sh
conda activate minigpt

OUTPUT_DIR="$1"

cd /tmp/b11902090/Trustworthy-AI-HW-1/minimind/scripts
python3 convert_model.py

cd /tmp/b11902090/Trustworthy-AI-HW-1
python3 eval_tmmluplus.py eval \
    --model_path ../../minimind-3 \
    --output "${OUTPUT_DIR}"/eval.json

python3 eval_tmmluplus.py eval \
    --model_path ../../minimind-3 \
    --chat_template \
    --output "${OUTPUT_DIR}"/eval_template.json

python3 eval_tmmluplus.py eval \
    --model_path ../../minimind-3 \
    --num_fewshot 5 \
    --output "${OUTPUT_DIR}"/eval_fewshot.json

python3 eval_tmmluplus.py eval \
    --model_path ../../minimind-3 \
    --chat_template \
    --num_fewshot 5 \
    --output "${OUTPUT_DIR}"/eval_fewshot_template.json