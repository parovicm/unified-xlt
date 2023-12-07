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
OUTPUT_DIR="${TRANSLATION_ROOT}/${TARGET_LANG}"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="${OUTPUT_DIR}/${SPLIT}.json"
#if [[ "$LANGUAGE_ADAPTER" == "yes" ]]; then
#    ADAPTER_DIR_ARG="--lang_ft ${PROJECT_ROOT}/experiments/nllb_adapt/nllb600M_${TARGET_LANG}_epochs-5_5"
#else
#    ADAPTER_DIR_ARG=""
#fi

python run_translation.py \
    --do_predict \
    --source_lang "${SOURCE_LANG}_${SOURCE_SCRIPT}" \
    --target_lang "${TARGET_LANG}_${TARGET_SCRIPT}" \
    --forced_bos_token "${TARGET_LANG}_${TARGET_SCRIPT}" \
    --test_file $SOURCE_FILE \
    --columns_to_translate text \
    --predictions_file $OUTPUT_FILE \
    --model_name_or_path ${TRANSLATION_MODEL} \
    --per_device_eval_batch_size 32 \
    --predict_with_generate \
    --output_dir $DIR
    #$ADAPTER_DIR_ARG \
