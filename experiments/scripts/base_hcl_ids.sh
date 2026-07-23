#!/bin/bash

#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH -c 8

source C:/Users/sakura/anaconda3/etc/profile.d/conda.sh
conda activate apkencoder
set -aex

SEQ=001
SEED=1
LR=0.003
OPT=sgd
SCH=step
DECAY=0.95
E=250
WLR=0.00015
WE=100
DATA=gen_apigraph_drebin
TRAIN_START=2012-01
TRAIN_END=2012-12
TEST_START=2013-01
TEST_END=2018-12
RESULT_DIR=results/hcl_ids_combined
AL_OPT=adam

CNT=300
D=6
MODELDIM="512-384-256-128"
SAMPLER=half
BATCH_SIZE=1024
LOSS=hi-dist-xent
TS=$(date "+%m.%d-%H.%M.%S")
MODEL_DIR=models/hcl_ids_combined/${CNT}
RUN_DIR=experiments/${RESULT_DIR}/${CNT}/${TS}
RUN_NAME=gen_apigraph_hcl_ids_combined_cnt${CNT}_${SEQ}_seed${SEED}_test_${TEST_START}_${TEST_END}
mkdir -p ${RUN_DIR} ${MODEL_DIR}

python -u base.py                                           \
    --method pseudo                                         \
    --data ${DATA}                                          \
    --benign_zero                                           \
    --mdate 20240312                                        \
    --train_start ${TRAIN_START}                            \
    --train_end ${TRAIN_END}                                \
    --test_start ${TEST_START}                              \
    --test_end ${TEST_END}                                  \
    --encoder simple-enc-mlp                                \
    --classifier simple-enc-mlp                             \
    --loss_func ${LOSS}                                     \
    --enc-hidden ${MODELDIM}                                \
    --mlp-hidden 100-100                                    \
    --mlp-dropout 0.2                                       \
    --sampler ${SAMPLER}                                    \
    --bsize ${BATCH_SIZE}                                   \
    --optimizer ${OPT}                                      \
    --scheduler ${SCH}                                      \
    --learning_rate ${LR}                                   \
    --lr_decay_rate ${DECAY}                                \
    --lr_decay_epochs "10,500,10"                           \
    --epochs ${E}                                           \
    --encoder_retrain                                       \
    --al_optimizer ${AL_OPT}                                \
    --warm_learning_rate ${WLR}                             \
    --al_epochs ${WE}                                       \
    --xent-lambda 100                                       \
    --display-interval 180                                  \
    --al                                                    \
    --count ${CNT}                                          \
    --F1D ${D}                                              \
    --local_pseudo_loss                                     \
    --reduce none                                           \
    --sample_reduce mean                                    \
    --seed ${SEED}                                          \
    --model-dir ${MODEL_DIR}                                \
    --ids                                                   \
    --ids-lambda 0.001                                      \
    --ids-gamma 1.0                                         \
    --ids-ema-decay 0.6                                     \
    --ids-proxy-lr 1.0                                      \
    --ids-robust-weight 1.0                                 \
    --ids-hcl-update-mode combined                          \
    --ids-batch-size 192                                    \
    --ids-warmup 5                                          \
    --ids-grad-clip 1.0                                     \
    --result ${RUN_DIR}/${RUN_NAME}.csv                     \
    --log_path ${RUN_DIR}/${RUN_NAME}.log
