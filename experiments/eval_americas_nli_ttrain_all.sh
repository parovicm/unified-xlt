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
    ADAPTATION_PATTERN="cambridgeltl/${BASE_MODEL_SFT_IDENTIFIER}-lang-sft-{}-small"
else
    # fill in for adapters
    ADAPTATION_PATTERN="/mnt/nas_home/mp939/alan_${BASE_MODEL_SHORT}_lang_adapters/{}"
fi

MODEL_DIR="${PROJECT_ROOT}/experiments/nli_ttrain_all/${ADAPTATION_METHOD}_${BASE_MODEL_SHORT}_ts-${NUM_TARGET_SHOTS}_la-${LANGUAGE_ADAPTATION}_epochs-${FULL_FT_EPOCHS}_${SPARSE_FT_EPOCHS}"

EVAL_SPLIT="test"
RESULTS_DIR="${MODEL_DIR}/results/${EVAL_SPLIT}/${TARGET_LANG}"
mkdir -p "$RESULTS_DIR"

python run_text_classification.py \
  --model_name_or_path "$BASE_MODEL" \
  --task_ft "$MODEL_DIR" \
  --load_adapter "${MODEL_DIR}/nli" \
  --task_name nli \
  --target_lang "$TARGET_LANG" \
  --target_test_file "${PROJECT_ROOT}/datasets/AmericasNLI/test/${TARGET_LANG}.tsv" \
  --label_file "${PROJECT_ROOT}/datasets/AmericasNLI/labels.json" \
  --input_columns "premise" "hypothesis" \
  --adaptation_method "$ADAPTATION_METHOD" \
  --target_adaptation "$LANGUAGE_ADAPTATION" \
  --lang_adapt_format "$ADAPTATION_PATTERN" \
  --target_num_examples -1 \
  --include_target_in_evaluation yes \
  --output_dir "$RESULTS_DIR" \
  --do_eval \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --eval_metric xnli \
  --overwrite_output_dir no \
  --eval_split "$EVAL_SPLIT" #> "$RESULTS_DIR/log.txt" 2>&1
