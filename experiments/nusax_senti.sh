#!/bin/bash

# This can eventually be a slurm array task, which considers all settings of sfts/adapters, number of shots, target language etc.

cd ../scripts/text-classification

PROJECT_ROOT=/mnt/nas_home/fs-xlt

BASE_MODEL=xlm-roberta-base
BASE_MODEL_SHORT=xlmrbase
BASE_MODEL_SFT_IDENTIFIER=xlmr

SOURCE_LANG=id
TARGET_LANG=ace

ADAPTATION_METHOD=sft # or adapters
SOURCE_ADAPTATION=yes
TARGET_ADAPTATION=yes
ADAPTATION_PATTERN="cambridgeltl/${BASE_MODEL_SFT_IDENTIFIER}-lang-sft-{}-small"

NUM_SOURCE_SHOTS=-1
NUM_TARGET_SHOTS=100
if [[ "$NUM_SOURCE_SHOTS" == -1 ]]; then
    NUM_SOURCE_SHOTS_STR=all
else
    NUM_SOURCE_SHOTS_STR="$NUM_SOURCE_SHOTS"
fi
if [[ "$NUM_TARGET_SHOTS" == -1 ]]; then
    NUM_TARGET_SHOTS_STR=all
else
    NUM_TARGET_SHOTS_STR="$NUM_TARGET_SHOTS"
fi

TARGET_UPSAMPLING=10

EPOCHS=5
LTSFT_PHASE1_EPOCHS=5

MODEL_DIR="${PROJECT_ROOT}/experiments/sa/${ADAPTATION_METHOD}/${BASE_MODEL_SHORT}_${SOURCE_LANG}_${TARGET_LANG}_sa-${SOURCE_ADAPTATION}_ta-${TARGET_ADAPTATION}_shots-${NUM_SOURCE_SHOTS_STR}-${NUM_TARGET_SHOTS_STR}_upsampling-${TARGET_UPSAMPLING}"
mkdir -p $MODEL_DIR

python run_text_classification.py \
  --model_name_or_path "$BASE_MODEL" \
  --source_lang "$SOURCE_LANG" \
  --target_lang "$TARGET_LANG" \
  --source_train_file "${PROJECT_ROOT}/datasets/smsa-trimmed/train.tsv" \
  --source_validation_file "${PROJECT_ROOT}/datasets/smsa-trimmed/validation.tsv" \
  --target_dataset indonlp/NusaX-senti \
  --label_file "${PROJECT_ROOT}/datasets/smsa-trimmed/labels.json" \
  --adaptation_method "$ADAPTATION_METHOD" \
  --source_adaptation "$SOURCE_ADAPTATION" \
  --target_adaptation "$TARGET_ADAPTATION" \
  --lang_adapt_format "$ADAPTATION_PATTERN" \
  --source_num_examples "$NUM_SOURCE_SHOTS" \
  --target_num_examples "$NUM_TARGET_SHOTS" \
  --target_upsampling "$TARGET_UPSAMPLING" \
  --include_target_in_evaluation no \
  --output_dir "$MODEL_DIR" \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration "$EPOCHS" \
  --sparse_ft_max_epochs_per_iteration "$LTSFT_PHASE1_EPOCHS" \
  --num_train_epochs "$EPOCHS" \
  --ft_params_num 14155776 \
  --freeze_layer_norm \
  --learning_rate 2e-5 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --eval_metric f1 \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --eval_split validation \
  --save_total_limit 2 #> "$MODEL_DIR/log.txt" 2>&1
if [[ $? != 0 ]]; then
    exit 1
fi

EVAL_SPLIT="validation"
RESULTS_DIR="${MODEL_DIR}/results/${EVAL_SPLIT}/${TARGET_LANG}"
mkdir -p "$RESULTS_DIR"

python run_text_classification.py \
  --model_name_or_path "$BASE_MODEL" \
  --task_ft "$MODEL_DIR" \
  --source_lang "$SOURCE_LANG" \
  --target_lang "$TARGET_LANG" \
  --dataset_name indonlp/NusaX-senti \
  --label_file "${PROJECT_ROOT}/datasets/smsa-trimmed/labels.json" \
  --adaptation_method "$ADAPTATION_METHOD" \
  --source_adaptation "$SOURCE_ADAPTATION" \
  --target_adaptation "$TARGET_ADAPTATION" \
  --lang_adapt_format "$ADAPTATION_PATTERN" \
  --source_num_examples 0 \
  --target_num_examples -1 \
  --include_target_in_evaluation yes \
  --output_dir "$RESULTS_DIR" \
  --do_eval \
  --eval_metric f1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --overwrite_output_dir no \
  --eval_split "$EVAL_SPLIT" #> "$RESULTS_DIR/log.txt" 2>&1
