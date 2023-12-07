#!/bin/bash

cd ../scripts/translation

SOURCE_LANG=eng_Latn
TARGET_LANG=spa_Latn

OUTPUT_DIR=test-translation
mkdir -p $OUTPUT_DIR

python run_translation.py \
    --do_predict \
    --source_lang $SOURCE_LANG \
    --target_lang $TARGET_LANG \
    --model_name_or_path facebook/nllb-200-distilled-600M \
    --forced_bos_token $TARGET_LANG \
    --test_file test_sentences.tsv \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --overwrite_output_dir

