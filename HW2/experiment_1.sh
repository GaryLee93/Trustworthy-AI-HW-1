if [ "$(basename "$PWD")" != "HW2" ]; then
    echo "錯誤：請在 HW2 資料夾下執行此腳本（目前位於：$PWD）" >&2
    exit 1
fi

source ../../miniconda3/etc/profile.d/conda.sh
conda activate hw2

cd student_release
python eval_agent.py --model_dir ../minimind-3 --agent my_agent:MyAgent --data_dir data/val --out results/E1_update3