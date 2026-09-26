#!/bin/bash
set -e

source /tmp/b11902090/miniconda3/etc/profile.d/conda.sh
conda activate minigpt

OUTPUT_DIR="$1"
 
if [ -z "${OUTPUT_DIR}" ]; then
    echo "用法: $0 <OUTPUT_DIR>" >&2
    exit 1
fi
 
REPO_DIR="/tmp/b11902090/Trustworthy-AI-HW-1"
 
# Point minimind/out at THIS OUTPUT_DIR ourselves instead of trusting that
# Experiment.sh (or a manual ln -sfn) already pointed it here. convert_model.py
# reads ../out/full_sft_768.pth blindly -- if that symlink was last set for a
# different run (e.g. you evaluated baseline right after training experiment,
# without re-linking), it silently converts and scores THAT other model under
# whatever label you pass here. Setting it explicitly on every eval.sh call
# removes that footgun.
ln -sfn "${OUTPUT_DIR}" "${REPO_DIR}/minimind/out"
 
cd "${REPO_DIR}/minimind/scripts"
python3 convert_model.py
 
mkdir -p "${OUTPUT_DIR}"/eval
cd "${REPO_DIR}"
 
# Record which checkpoint this eval run actually converted and scored, so a
# saved eval.json can always be traced back to its source weights -- catches
# this class of mismatch after the fact even if the symlink gets reused wrong
# again some other way.
{
    echo "evaluated_at: $(date -Iseconds)"
    echo "output_dir: ${OUTPUT_DIR}"
    echo "checkpoint: ${OUTPUT_DIR}/full_sft_768.pth"
    echo "checkpoint_md5: $(md5sum "${OUTPUT_DIR}/full_sft_768.pth" 2>/dev/null | cut -d' ' -f1)"
    echo "minimind_out_symlink: $(readlink -f "${REPO_DIR}/minimind/out")"
} | tee "${OUTPUT_DIR}"/eval/provenance.txt

mkdir -p "${OUTPUT_DIR}"/eval
cd /tmp/b11902090/Trustworthy-AI-HW-1
python3 eval_tmmluplus.py eval \
    --model_path /tmp/b11902090/Trustworthy-AI-HW-1/minimind/minimind-3 \
    --save_predictions "${OUTPUT_DIR}"/eval/p.json \
    --output "${OUTPUT_DIR}"/eval/eval.json

python3 eval_tmmluplus.py eval \
    --model_path /tmp/b11902090/Trustworthy-AI-HW-1/minimind/minimind-3 \
    --chat_template \
    --save_predictions "${OUTPUT_DIR}"/eval/p_template.json \
    --output "${OUTPUT_DIR}"/eval/eval_template.json

python3 eval_tmmluplus.py eval \
    --model_path /tmp/b11902090/Trustworthy-AI-HW-1/minimind/minimind-3 \
    --num_fewshot 5 \
    --save_predictions "${OUTPUT_DIR}"/eval/p_fewshot.json \
    --output "${OUTPUT_DIR}"/eval/eval_fewshot.json

python3 eval_tmmluplus.py eval \
    --model_path /tmp/b11902090/Trustworthy-AI-HW-1/minimind/minimind-3 \
    --chat_template \
    --num_fewshot 5 \
    --save_predictions "${OUTPUT_DIR}"/eval/p_template_fewshot.json \
    --output "${OUTPUT_DIR}"/eval/eval_fewshot_template.json