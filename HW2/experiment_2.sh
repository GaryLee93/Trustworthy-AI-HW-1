if [ "$(basename "$PWD")" != "HW2" ]; then
    echo "錯誤：請在 HW2 資料夾下執行此腳本（目前位於：$PWD）" >&2
    exit 1
fi

source ../../miniconda3/etc/profile.d/conda.sh
conda activate hw2


cd "../minimind/trainer"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/checkpoints"
ln -sfn "${OUTPUT_DIR}" "${PROJECT_DIR}/minimind/out"
ln -sfn "${OUTPUT_DIR}/checkpoints" "${PROJECT_DIR}/minimind/checkpoints"


swanlab login -k iXDM58vG5zuP1CHeKtVPh
python3 train_full_sft.py \
    --save_dir "${OUTPUT_DIR}" \
    --log_interval 1000 \
    --save_interval 2000 \
    --seed 824 \
    --use_wandb \
    --wandb_project "Trustworthy-AI-HW-1"