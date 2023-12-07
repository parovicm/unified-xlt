#!/bin/bash
#cd ../scripts/text-classification
cd $SRC_DIR/scripts/text-classification

source activate "$CONDA_ENV"
if [[ $? != 0 ]]; then
    echo "Failed to activate conda environment." >&2
    exit 1
fi

#printenv

PROJECT_ROOT=/mnt/nas_home/fs-xlt

if [[ "$ADAPTATION_METHOD" == "sft" ]]; then
    #ADAPTATION_PATTERN="cambridgeltl/${BASE_MODEL_SFT_IDENTIFIER}-lang-sft-{}-small"
    ADAPTATION_PATTERN="$HDD/experiments/fs-xlt/lang_sft/${BASE_MODEL_SHORT}_{}_params-7667712_steps-100000_reg"
else
    # fill in for adapters
    ADAPTATION_PATTERN="/mnt/nas_home/mp939/nusa_new_${BASE_MODEL_SFT_IDENTIFIER}_adapters/{}"
fi

MODEL_DIR="${PROJECT_ROOT}/experiments/sa_ttrain_all/${ADAPTATION_METHOD}_${BASE_MODEL_SHORT}_la-${LANGUAGE_ADAPTATION}_epochs-${FULL_FT_EPOCHS}_${SPARSE_FT_EPOCHS}"

EVAL_SPLIT="validation"
RESULTS_DIR="${MODEL_DIR}/results/${EVAL_SPLIT}/${TARGET_LANG}"
mkdir -p "$RESULTS_DIR"

python run_text_classification.py \
  --model_name_or_path "$BASE_MODEL" \
  --task_ft "$MODEL_DIR" \
  --load_adapter "${MODEL_DIR}/sa" \
  --task_name sa \
  --target_lang "$TARGET_LANG" \
  --dataset_name indonlp/NusaX-senti \
  --label_file "${PROJECT_ROOT}/datasets/smsa-trimmed/labels.json" \
  --target_adaptation "$TARGET_ADAPTATION" \
  --adaptation_method "$ADAPTATION_METHOD" \
  --lang_adapt_format "$ADAPTATION_PATTERN" \
  --target_num_examples -1 \
  --include_target_in_evaluation yes \
  --output_dir "$RESULTS_DIR" \
  --do_eval \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --overwrite_output_dir no \
  --eval_metric f1 \
  --eval_split "$EVAL_SPLIT" #> "$RESULTS_DIR/log.txt" 2>&1
