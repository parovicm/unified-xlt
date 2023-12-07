#!/bin/bash
#cd ../scripts/text-classification
cd $SRC_DIR/scripts/text-classification

source activate "$CONDA_ENV"
if [[ $? != 0 ]]; then
    echo "Failed to activate conda environment." >&2
    exit 1
fi

printenv

PROJECT_ROOT=/mnt/nas_home/fs-xlt
DATA_DIR="${PROJECT_ROOT}/datasets/multi_nli/translated-${SOURCE_LANG}"

if [[ "$ADAPTATION_METHOD" == "sft" ]]; then
    ADAPTATION_PATTERN="cambridgeltl/${BASE_MODEL_SFT_IDENTIFIER}-lang-sft-{}-small"
else
    # fill in for adapters
    ADAPTATION_METHOD=
fi

python run_text_classification.py \
  --model_name_or_path "$BASE_MODEL" \
  --source_lang "$SOURCE_LANG" \
  --source_train_file "${DATA_DIR}/train.json" \
  --source_validation_file "${DATA_DIR}/validation.json" \
  --label_file "${PROJECT_ROOT}/datasets/AmericasNLI/labels.json" \
  --input_columns "premise" "hypothesis" \
  --adaptation_method "$ADAPTATION_METHOD" \
  --source_adaptation "$SOURCE_ADAPTATION" \
  --lang_adapt_format "$ADAPTATION_PATTERN" \
  --source_num_examples -1 \
  --output_dir "$DIR" \
  --do_train \
  --do_eval \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration "$FULL_FT_EPOCHS" \
  --sparse_ft_max_epochs_per_iteration "$SPARSE_FT_EPOCHS" \
  --num_train_epochs "$FULL_FT_EPOCHS" \
  --ft_params_num "$SFT_PARAMS_NUM" \
  --freeze_layer_norm \
  --learning_rate "$LEARNING_RATE" \
  --evaluation_strategy steps \
  --eval_steps "$EVAL_STEPS" \
  --save_steps "$EVAL_STEPS" \
  --eval_metric xnli \
  --metric_for_best_model eval_accuracy \
  --load_best_model_at_end \
  --eval_split validation \
  --save_total_limit 2 #> "$MODEL_DIR/log.txt" 2>&1
if [[ $? != 0 ]]; then
    exit 1
fi

EVAL_SPLIT="test"
RESULTS_DIR="${DIR}/results/${EVAL_SPLIT}/${SOURCE_LANG}"
mkdir -p "$RESULTS_DIR"

python run_text_classification.py \
  --model_name_or_path "$BASE_MODEL" \
  --task_ft "$DIR" \
  --source_lang "$SOURCE_LANG" \
  --source_test_file "${PROJECT_ROOT}/datasets/AmericasNLI/test/${SOURCE_LANG}.tsv" \
  --label_file "${PROJECT_ROOT}/datasets/AmericasNLI/labels.json" \
  --input_columns "premise" "hypothesis" \
  --adaptation_method "$ADAPTATION_METHOD" \
  --source_adaptation "$SOURCE_ADAPTATION" \
  --lang_adapt_format "$ADAPTATION_PATTERN" \
  --source_num_examples -1 \
  --output_dir "$RESULTS_DIR" \
  --do_eval \
  --eval_metric xnli \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --overwrite_output_dir no \
  --eval_split "$EVAL_SPLIT" #> "$RESULTS_DIR/log.txt" 2>&1
