#! /bin/bash

#SBATCH -t 03:00:00

#SBATCH -n 1

#SBATCH -c 8
# win
# 红伟电脑运行脚本
source C:/Users/sakura/anaconda3/etc/profile.d/conda.sh
conda activate apkencoder
set a-ex
SEQ=001
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
RESULT_DIR=results/pseudo
AL_OPT=adam

CNT=200
D=6
modeldim="512-384-256-128"
S='half'
B=1024
LOSS='hi-dist-xent'
TS=$(date "+%m.%d-%H.%M.%S")
mkdir -p experiments/${RESULT_DIR}/${CNT}/${TS}

nohup python -u base.py	                                \
            --method 'pseudo'                                 \
            --data ${DATA}                                  \
            --benign_zero                                   \
            --mdate 20240312                                \
            --train_start ${TRAIN_START}                    \
            --train_end ${TRAIN_END}                        \
            --test_start ${TEST_START}                      \
            --test_end ${TEST_END}                          \
            --encoder simple-enc-mlp                        \
            --classifier simple-enc-mlp                     \
            --loss_func ${LOSS}                             \
            --enc-hidden ${modeldim}                        \
            --mlp-hidden 100-100                            \
            --mlp-dropout 0.2                               \
            --sampler ${S}                                  \
            --bsize ${B}                                    \
            --optimizer ${OPT}                              \
            --scheduler ${SCH}                              \
            --learning_rate ${LR}                           \
            --lr_decay_rate ${DECAY}                        \
            --lr_decay_epochs "10,500,10"                   \
            --epochs ${E}                                   \
            --encoder_retrain                               \
            --al_optimizer ${AL_OPT}                        \
            --warm_learning_rate ${WLR}                     \
            --al_epochs ${WE}                               \
            --xent-lambda 100                               \
            --display-interval 180                          \
            --al                                            \
            --count ${CNT}                                  \
            --F1D ${D}                                     \
            --local_pseudo_loss                             \
            --reduce "none"                                 \
            --sample_reduce 'mean'                          \
            --result experiments/${RESULT_DIR}/${CNT}/${TS}/gen_apigraph_cnt${CNT}_${SEQ}_warm_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_${AL_OPT}_wlr${WLR}_we${WE}_test_${TEST_START}_${TEST_END}_cnt${CNT}.csv \
            --log_path experiments/${RESULT_DIR}/${CNT}/${TS}/gen_apigraph_cnt${CNT}_${SEQ}_warm_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_${AL_OPT}_wlr${WLR}_we${WE}_test_${TEST_START}_${TEST_END}_cnt${CNT}.log \
            >> experiments/${RESULT_DIR}/${CNT}/${TS}/gen_apigraph_cnt${CNT}_${SEQ}_warm_lr${LR}_${OPT}_${SCH}_${DECAY}_e${E}_${AL_OPT}_wlr${WLR}_we${WE}_test_${TEST_START}_${TEST_END}_cnt${CNT}.log 2>&1 &

wait