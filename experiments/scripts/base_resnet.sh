#! /bin/bash

#SBATCH -t 03:00:00

#SBATCH -n 1

#SBATCH -c 8

# win
# 红伟电脑运行脚本
source C:/Users/sakura/anaconda3/etc/profile.d/conda.sh
conda activate apkencoder
set a-ex

SEQ=002
LR=0.0009
E=25
WLR=0.000045
WE=25
DATA=gen_apigraph_drebin
TRAIN_START=2012-01
TRAIN_END=2012-12
TEST_START=2013-01
TEST_END=2018-12
RESULT_DIR=results/resnet
MODEL_DATE=20240412
CNT=200
D=6


TS=$(date "+%m.%d-%H.%M.%S")
mkdir -p experiments/${RESULT_DIR}/${CNT}/${TS}

nohup python -u base.py	                                \
            --method 'resnet'                                 \
            --data ${DATA}                                  \
            --mdate ${MODEL_DATE}                           \
            --train_start ${TRAIN_START}                    \
            --train_end ${TRAIN_END}                        \
            --test_start ${TEST_START}                      \
            --test_end ${TEST_END}                          \
            --classifier res                                \
            --mlp-hidden 100-100                            \
            --mlp-dropout 0.2                               \
            --mlp-batch-size 32                             \
            --mlp-lr ${LR}                                  \
            --mlp-epochs ${E}                               \
            --mlp-warm-lr ${WLR}                            \
            --mlp-warm-epochs ${WE}                         \
            --al                                            \
            --unc                                           \
            --count ${CNT}                                  \
            --F1D ${D}                                     \
            --result experiments/${RESULT_DIR}/${CNT}/${TS}/gen_apigraph_cnt${CNT}_${SEQ}_warm_lr${LR}_e${E}_wlr${WLR}_we${WE}_test_${TEST_START}_${TEST_END}_cnt${CNT}.csv \
            --log_path experiments/${RESULT_DIR}/${CNT}/${TS}/gen_apigraph_cnt${CNT}_${SEQ}_warm_lr${LR}_e${E}_wlr${WLR}_we${WE}_test_${TEST_START}_${TEST_END}_cnt${CNT}.log \
            >> experiments/${RESULT_DIR}/${CNT}/${TS}/gen_apigraph_cnt${CNT}_${SEQ}_warm_lr${LR}_e${E}_wlr${WLR}_we${WE}_test_${TEST_START}_${TEST_END}_cnt${CNT}.log 2>&1 &

wait