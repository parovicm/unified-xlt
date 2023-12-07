#!/bin/bash
#SBATCH --gres=gpu:1,gpu-ram:48G

source activate $CONDA_ENV

cd $SRC_DIR/scripts/translation

#SOURCE_LANG=spa_Latn
#SOURCE_LANG_NAME=spanish
#SOURCE_COLUMN_NAME=es
#TARGET_LANG=cni_Latn
#TARGET_LANG_NAME=ashaninka
#TARGET_COLUMN_NAME=cni

PROJECT_ROOT=/mnt/nas_home/fs-xlt
#DATA_DIR="${PROJECT_ROOT}/datasets/nusax/datasets/mt"
DATA_DIR="${DATA_ROOT}/${LANG1}-${LANG2}"

#MODEL_DIR="${PROJECT_ROOT}/experiments/mt-adapt/test-${SOURCE_LANG}-${TARGET_LANG}"
#mkdir -p $MODEL_DIR

python run_translation.py \
    --do_train \
    --source_lang $SOURCE_LANG \
    --source_column_name $SOURCE_COLUMN_NAME \
    --target_lang $TARGET_LANG \
    --target_column_name $TARGET_COLUMN_NAME \
    --forced_bos_token $TARGET_LANG \
    --model_name_or_path $BASE_MODEL \
    --train_file "${DATA_DIR}/train.tsv" \
    --validation_file "${DATA_DIR}/dev.tsv" \
    --output_dir $DIR \
    --optim adafactor \
    --fp16 \
    --ft_params_proportion 0.08 \
    --full_ft_max_epochs_per_iteration 5 \
    --sparse_ft_max_epochs_per_iteration 5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 16 \
    --max_source_length 256 \
    --max_target_length 256 \
    --learning_rate 2e-5 \
    --logging_steps 50 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --predict_with_generate \
    --overwrite_output_dir \
    --save_total_limit 1 2>&1

    #--target_embedding_init $SOURCE_LANG \
    #--train_file "${PROJECT_ROOT}/datasets/korpusnusantara/bbc.csv" \
    #--validation_file "${DATA_DIR}/bbc/valid.csv" \
