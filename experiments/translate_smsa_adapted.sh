#!/bin/bash
#SBATCH --gres=gpu:1,gpu-ram:24G

cd $SRC_DIR/scripts/translation
source activate "$CONDA_ENV"
if [[ $? != 0 ]]; then
    echo "Failed to activate conda environment." >&2
    exit 1
fi

printenv

PROJECT_ROOT=/mnt/nas_home/fs-xlt
DATA_DIR="${PROJECT_ROOT}/datasets/smsa-trimmed"
SOURCE_FILE="${DATA_DIR}/${SPLIT}.tsv"

ADAPTER_DIR="${PROJECT_ROOT}/experiments/nllb_adapt_ltsft/nllb1.3B_${TARGET_LANG}_epochs-5_5"
OUTPUT_DIR="${PROJECT_ROOT}/experiments/nllb_adapt_ltsft/nllb1.3B_epochs-5_5_translations/${TARGET_LANG}"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="${OUTPUT_DIR}/${SPLIT}.json"

python run_translation.py \
    --do_predict \
    --source_lang "${SOURCE_LANG}_${SOURCE_SCRIPT}" \
    --target_lang "${TARGET_LANG}_${TARGET_SCRIPT}" \
    --forced_bos_token "${TARGET_LANG}_${TARGET_SCRIPT}" \
    --test_file $SOURCE_FILE \
    --columns_to_translate text \
    --predictions_file $OUTPUT_FILE \
    --model_name_or_path $BASE_MODEL \
    --lang_ft $ADAPTER_DIR \
    --per_device_eval_batch_size 32 \
    --predict_with_generate \
    --output_dir $DIR 2>&1

