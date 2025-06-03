#!/bin/bash
set -x -e

# conda activate mengjin_DL

# Run this on the lambda machines, since the code is not running on package, but refers to local code.
export PYTHONPATH=$PYTHONPATH:/home/dongm/ADNI_Whole_brain/RegionalDeepAtrophy/src
cd /home/dongm/ADNI_Whole_brain/RegionalDeepAtrophy/src

# Test if I need this to run with the package.
# export PYTHONPATH=$PYTHONPATH:/home/mengjin/anaconda3/envs/mengjin_DL/lib/python3.12/site-packages/deepatrophy
# cd /home/mengjin/anaconda3/envs/mengjin_DL/lib/python3.12/site-packages/deepatrophy

# Check how this was set up on pmacs.

DATA_DIR='/home/dongm/ADNI_Whole_brain/input_csv_list_ds_RDA'
OUT_DIR='/home/dongm/ADNI_Whole_brain/RDA_out'

##### run regional deep atrophy training
    # --test-double-pairs $DATA_DIR/csv_list_test_double_pair.csv \
    # --train-pairs $DATA_DIR/csv_list_train_pair.csv \
    # --eval-pairs $DATA_DIR/csv_list_eval_pair.csv \
    # --test-pairs $DATA_DIR/csv_list_test_pair.csv \

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1  python3 -m regional_deep_atrophy run_training \
    --train-double-pairs $DATA_DIR/csv_list_train_double_pair.csv \
    --eval-double-pairs $DATA_DIR/csv_list_eval_double_pair.csv \
    --ROOT ${OUT_DIR} \
    --batch-size 64 \
    --input-D 80 \
    --input-H 64 \
    --input-W 64 \
#     --load-model ${OUT_DIR}/Model/${prefix}/last_checkpoint_${epoch}.pt \


# prefix='2025-05-05_14-42' # registration descrete values (NCC, wp=0.01), lr = 1e-5
# epoch=0007

# prefix='2025-05-07_17-51' # registration contineous values, lr = 1e-4
# epoch=0034

prefix='2025-05-15_14-27' # registration contineous values, lr = 1e-4, patch size is 80*64*64
epoch=0042

# ##### run regional deep atrophy testing
    # --train-double-pairs $DATA_DIR/csv_list_train_double_pair.csv \
    # --eval-double-pairs $DATA_DIR/csv_list_eval_double_pair.csv \
    # --test-double-pairs $DATA_DIR/csv_list_test_double_pair.csv \
    #     --train-pairs $DATA_DIR/csv_list_train_pair.csv \
    # --eval-pairs $DATA_DIR/csv_list_eval_pair.csv \

CUDA_VISIBLE_DEVICES=0 python3 -m regional_deep_atrophy run_test \
    --test-pairs $DATA_DIR/csv_list_test_pair.csv \
    --ROOT "${OUT_DIR}/Model/${prefix}/${prefix}_last_checkpoint_${epoch}_eval" \
    --batch-size 128 \
    --input-D 80 \
    --input-H 64 \
    --input-W 64 \
    --load-model ${OUT_DIR}/Model/${prefix}/last_checkpoint_${epoch}.pt \

# ##### run regional deep atrophy results and generate scatter plots and group difference maps

python3 -m deepatrophy PAIIR \
        --train-pair-spreadsheet ${OUT_DIR}/Model/${prefix}_train_pair.csv \
        --test-pair-spreadsheet ${OUT_DIR}/Model/${prefix}_test_pair.csv \
        --test-double-pair-spreadsheet ${OUT_DIR}/Model/${prefix}_test_double_pair.csv \
        --workdir $OUT_DIR/Model/${prefix} \
        --prefix $prefix \
        --min-date 180 \
        --max-date 400 \

python3 -m deepatrophy PAIIR \
        --train-pair-spreadsheet ${OUT_DIR}/Model/${prefix}_train_pair.csv \
        --test-pair-spreadsheet ${OUT_DIR}/Model/${prefix}_test_pair.csv \
        --test-double-pair-spreadsheet ${OUT_DIR}/Model/${prefix}_test_double_pair.csv \
        --workdir $OUT_DIR/Model/${prefix} \
        --prefix $prefix \
        --min-date 400 \
        --max-date 800 \

