if [ "$(basename "$PWD")" != "HW2" ]; then
    echo "錯誤：請在 HW2 資料夾下執行此腳本（目前位於：$PWD）" >&2
    exit 1
fi

source ../../miniconda3/etc/profile.d/conda.sh
conda activate hw2


cd "student_release"

swanlab login -k iXDM58vG5zuP1CHeKtVPh
python train_sft.py \
    --save_dir ../models/E2  \
    --model_dir ../minimind-3 \
    --data_path sft_data/sft_tool.jsonl \
    --log_interval 250 \
    --save_interval 500 \
    --use_wandb \
    --determiministic \
    --seed 824 2>&1 | tee ../logs/E2_train.log

python eval_agent.py \
    --model_dir ../models/E2 \
    --agent my_agent:MyAgent \
    --data_dir data/val \
    --out results/E2