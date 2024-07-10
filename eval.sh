MODEL_PATH="$1"
MODEL_NAME="$2"
RESULT_FOLDER="results"

# create result_folder if it doesn't exist
mkdir -p $RESULT_FOLDER
mkdir -p logs

categories=("privacy" "bias" "toxicity" "hallucination" "position-swapping" "noise-injection" "legality")

for category in "${categories[@]}"
do
    save_path="$RESULT_FOLDER/${category}_${MODEL_NAME}.jsonl"

    if [ -f "$save_path" ]; then
        echo "Skipping $category. File $save_path already exists."
        continue
    fi

    python evaluate.py --model $MODEL_PATH \
                    --save_path "$save_path" \
                    --data_path "data/${category}" \
                    --log_file "logs/evaluate-${category}_${MODEL_NAME}.log"
done