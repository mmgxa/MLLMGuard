MODEL_PATH="$1"
MODEL_NAME="$2"


categories=("privacy" "bias" "toxicity" "non-exisitent" "position-swapping" "noise-injection" "legality")

for category in "${categories[@]}"
do
    save_path="results/${category}_${MODEL_NAME}.jsonl"

    # 检查 save_path 是否已存在
    if [ -f "$save_path" ]; then
        echo "Skipping $category. File $save_path already exists."
        continue
    fi

    python evaluate.py --model $MODEL_PATH \
                    --save_path "$save_path" \
                    --data_path "data/${category}" \
                    --log_file "logs/evaluate-${category}_${MODEL_NAME}.log"
done