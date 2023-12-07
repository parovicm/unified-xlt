#!/bin/bash
#cd ../scripts/text-classification
cd $SRC_DIR/scripts/text-classification

#source activate "$CONDA_ENV"
#if [[ $? != 0 ]]; then
#    echo "Failed to activate conda environment." >&2
#    exit 1
#fi

#printenv

PROJECT_ROOT=/mnt/nas_home/fs-xlt

if [[ "$ADAPTATION_METHOD" == "sft" ]]; then
    ADAPTATION_PATTERN="cambridgeltl/${BASE_MODEL_SFT_IDENTIFIER}-lang-sft-{}-small"
else
    # fill in for adapters
    ADAPTATION_PATTERN="/mnt/nas_home/mp939/nusa_new_${BASE_MODEL_SFT_IDENTIFIER}_adapters/{}"
fi
mkdir -p "$DIR"

python run_text_classification.py \
  --model_name_or_path "$BASE_MODEL" \
  --task_name sa \
  --source_lang "$SOURCE_LANG" \
  --multisource_data nusa_fs_multisource.json \
  --label_file "${PROJECT_ROOT}/datasets/smsa-trimmed/labels.json" \
  --language_adaptation "$LANGUAGE_ADAPTATION" \
  --adaptation_method "$ADAPTATION_METHOD" \
  --lang_adapt_format "$ADAPTATION_PATTERN" \
  --source_num_examples "$NUM_SOURCE_SHOTS" \
  --target_num_examples "$NUM_TARGET_SHOTS" \
  --target_upsampling "$TARGET_UPSAMPLING" \
  --include_target_in_evaluation no \
  --output_dir "$DIR" \
  --do_train \
  --do_eval \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --overwrite_output_dir \
  --full_ft_max_epochs_per_iteration "$FULL_FT_EPOCHS" \
  --sparse_ft_max_epochs_per_iteration "$SPARSE_FT_EPOCHS" \
  --num_train_epochs "$SPARSE_FT_EPOCHS" \
  --ft_params_num "$SFT_PARAMS_NUM" \
  --freeze_layer_norm \
  --learning_rate "$LEARNING_RATE" \
  --evaluation_strategy steps \
  --eval_steps "$EVAL_STEPS" \
  --save_steps "$EVAL_STEPS" \
  --eval_metric f1 \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --eval_split validation \
  --save_total_limit 2 > "$DIR/log.txt" 2>&1
