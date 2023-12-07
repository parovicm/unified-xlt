#!/bin/bash
cd ../scripts/text-classification
#cd $SRC_DIR/scripts/text-classification
#
#source activate "$CONDA_ENV"
#if [[ $? != 0 ]]; then
#    echo "Failed to activate conda environment." >&2
#    exit 1
#fi
#
#printenv

PROJECT_ROOT=/mnt/nas_home/fs-xlt

if [[ "$ADAPTATION_METHOD" == "sft" ]]; then
    ADAPTATION_PATTERN="cambridgeltl/${BASE_MODEL_SFT_IDENTIFIER}-lang-sft-{}-small"
else
    # fill in for adapters
    ADAPTATION_METHOD=
fi

if [[ "$TARGET_LANG" ]]; then
    TARGET_LANG_ARG="--target_lang $TARGET_LANG"
    TARGET_ADAPTATION_ARG="--target_adaptation $TARGET_ADAPTATION"
    TARGET_SHOTS_ARG="--target_num_examples $NUM_TARGET_SHOTS"
    TARGET_UPSAMPLING_ARG="--target_upsampling $TARGET_UPSAMPLING"
else
    TARGET_LANG_ARG=""
    TARGET_ADAPTATION_ARG=""
    TARGET_SHOTS_ARG=""
    TARGET_UPSAMPLING_ARG=""
fi
python run_text_classification.py \
  --model_name_or_path "$BASE_MODEL" \
  --source_lang "$SOURCE_LANG" \
  $TARGET_LANG_ARG \
  --source_dataset multi_nli.py \
  --source_has_config no \
  --target_train_file "${PROJECT_ROOT}/datasets/AmericasNLI/dev/${TARGET_LANG}.tsv" \
  --label_file "${PROJECT_ROOT}/datasets/AmericasNLI/labels.json" \
  --input_columns "premise" "hypothesis" \
  --adaptation_method "$ADAPTATION_METHOD" \
  --source_adaptation "$SOURCE_ADAPTATION" \
  $TARGET_ADAPTATION_ARG \
  --lang_adapt_format "$ADAPTATION_PATTERN" \
  --source_num_examples "$NUM_SOURCE_SHOTS" \
  $TARGET_SHOTS_ARG \
  $TARGET_UPSAMPLING_ARG \
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
  --num_train_epochs "$FULL_FT_EPOCHS" \
  --ft_params_num "$SFT_PARAMS_NUM" \
  --freeze_layer_norm \
  --learning_rate "$LEARNING_RATE" \
  --save_steps "$EVAL_STEPS" \
  --evaluation_strategy steps \
  --eval_steps "$EVAL_STEPS" \
  --eval_metric xnli \
  --metric_for_best_model eval_accuracy \
  --load_best_model_at_end \
  --eval_split validation \
  --save_total_limit 2 #> "$MODEL_DIR/log.txt" 2>&1
if [[ $? != 0 ]]; then
    exit 1
fi

if ! [[ "$TARGET_LANG" ]]; then
    exit 0
fi

EVAL_SPLIT="test"
RESULTS_DIR="${DIR}/results/${EVAL_SPLIT}/${TARGET_LANG}"
mkdir -p "$RESULTS_DIR"

python run_text_classification.py \
  --model_name_or_path "$BASE_MODEL" \
  --task_ft "$DIR" \
  --target_lang "$TARGET_LANG" \
  --target_test_file "${PROJECT_ROOT}/datasets/AmericasNLI/test/${TARGET_LANG}.tsv" \
  --label_file "${PROJECT_ROOT}/datasets/AmericasNLI/labels.json" \
  --input_columns "premise" "hypothesis" \
  --adaptation_method "$ADAPTATION_METHOD" \
  --target_adaptation "$TARGET_ADAPTATION" \
  --lang_adapt_format "$ADAPTATION_PATTERN" \
  --target_num_examples -1 \
  --include_target_in_evaluation yes \
  --output_dir "$RESULTS_DIR" \
  --do_eval \
  --eval_metric xnli \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --overwrite_output_dir no \
  --eval_split "$EVAL_SPLIT" #> "$RESULTS_DIR/log.txt" 2>&1
