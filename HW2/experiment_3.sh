if [ "$(basename "$PWD")" != "HW2" ]; then
    echo "錯誤：請在 HW2 資料夾下執行此腳本（目前位於：$PWD）" >&2
    exit 1
fi

source ../../miniconda3/etc/profile.d/conda.sh
conda activate hw2
# 任何一步失敗就停止；pipefail 讓 python ... | tee 的錯誤不會被 tee 吃掉
set -eo pipefail

mkdir -p logs
cd "student_release"

# 產生 tool + reading 的混合 SFT 資料（只用 data/train，結果是決定性的）
python build_sft_data.py --data_dir data/train --out_dir sft_data
md5sum sft_data/*.jsonl | tee ../logs/E3_data_md5.txt

swanlab login -k iXDM58vG5zuP1CHeKtVPh
# 從原本的 minimind-3 開始訓練；除了資料以外，設定都與 Experiment 2 相同
python train_sft.py \
    --save_dir ../models/E3 \
    --model_dir ../minimind-3 \
    --data_path sft_data/sft_mix.jsonl \
    --epochs 1 \
    --log_interval 250 \
    --save_interval 500 \
    --use_wandb \
    --deterministic \
    --seed 824 2>&1 | tee ../logs/E3_train.log

python eval_agent.py \
    --model_dir ../models/E3 \
    --agent my_agent:MyAgent \
    --data_dir data/val \
    --out results/E3
