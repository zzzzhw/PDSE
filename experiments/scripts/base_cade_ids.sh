#!/bin/bash

#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 8

source C:/Users/sakura/anaconda3/etc/profile.d/conda.sh
conda activate apkencoder
set -aex

SEQ=127
SEED=1
LR=0.001
OPT=adam
SCH=cosine
DECAY=1
E=100
WLR=5e-05
WE=50
MWLR=5e-05
MWE=50
DATA=gen_apigraph_drebin
TRAIN_START=2012-01
TRAIN_END=2012-12
TEST_START=2013-01
TEST_END=2018-12
RESULT_DIR=results/cade_ids
MODEL_DATE=20230501

CNT=300
D=6
MODELDIM="512-384-256-128"
SAMPLER=triplet
BATCH_SIZE=1536
LOSS=triplet-mse
TS=$(date "+%m.%d-%H.%M.%S")
RUN_DIR=experiments/${RESULT_DIR}/${CNT}/${TS}
MODEL_DIR=models/cade_ids/${CNT}
RUN_NAME=cade_ids_apigraph_cnt${CNT}_${SEQ}_seed${SEED}_test_${TEST_START}_${TEST_END}
mkdir -p ${RUN_DIR} ${MODEL_DIR}

python -u base.py                                           \
    --method cade                                           \
    --data ${DATA}                                          \
    --benign_zero                                           \
    --mdate ${MODEL_DATE}                                   \
    --train_start ${TRAIN_START}                            \
    --train_end ${TRAIN_END}                                \
    --test_start ${TEST_START}                              \
    --test_end ${TEST_END}                                  \
    --encoder cae                                           \
    --enc-hidden ${MODELDIM}                                \
    --loss_func ${LOSS}                                     \
    --sampler ${SAMPLER}                                    \
    --bsize ${BATCH_SIZE}                                   \
    --optimizer ${OPT}                                      \
    --scheduler ${SCH}                                      \
    --learning_rate ${LR}                                   \
    --lr_decay_rate ${DECAY}                                \
    --lr_decay_epochs "10,500,10"                           \
    --epochs ${E}                                           \
    --cae-lambda 0.1                                        \
    --display-interval 100                                  \
    --classifier mlp                                        \
    --cls-feat encoded                                      \
    --mlp-hidden 100-100                                    \
    --mlp-dropout 0.2                                       \
    --mlp-batch-size 32                                     \
    --mlp-lr 0.001                                          \
    --mlp-epochs 50                                         \
    --al                                                    \
    --ood                                                   \
    --count ${CNT}                                          \
    --F1D ${D}                                              \
    --encoder_retrain                                       \
    --warm_learning_rate ${WLR}                             \
    --al_epochs ${WE}                                       \
    --mlp-warm-lr ${MWLR}                                   \
    --mlp-warm-epochs ${MWE}                                \
    --seed ${SEED}                                          \
    --model-dir ${MODEL_DIR}                                \
    --ids                                                   \
    --ids-lambda 0.001                                      \
    --ids-gamma 1.0                                         \
    --ids-ema-decay 0.6                                     \
    --ids-proxy-lr 1.0                                      \
    --ids-robust-weight 1.0                                 \
    --ids-batch-size 192                                    \
    --ids-warmup 5                                          \
    --ids-grad-clip 1.0                                     \
    --result ${RUN_DIR}/${RUN_NAME}.csv                     \
    --log_path ${RUN_DIR}/${RUN_NAME}.log
