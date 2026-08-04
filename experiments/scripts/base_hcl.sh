#! /bin/bash

#SBATCH -t 03:00:00

#SBATCH -n 1

#SBATCH -c 8
# win
# 红伟电脑运行脚本
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
DATA=${1:-${DATA:-bodmas}}
case ${DATA} in
    gen_apigraph_drebin)
        DATA_TAG=gen_apigraph
        TRAIN_START=2012-01
        TRAIN_END=2012-12
        TEST_START=2013-01
        TEST_END=2018-12
        ;;
    bodmas)
        DATA_TAG=bodmas
        TRAIN_START=2007-01
        TRAIN_END=2019-09
        TEST_START=2019-10
        TEST_END=2020-09
        ;;
    *)
        echo "Unsupported dataset: ${DATA}" >&2
        exit 2
        ;;
esac
AL_OPT=adam
# HCL_ENHANCEMENT=sam
HCL_ENHANCEMENT=${HCL_ENHANCEMENT:-none}

case ${HCL_ENHANCEMENT} in
    none|mixup|focal|sam) ;;
    *)
        echo "Unsupported HCL_ENHANCEMENT: ${HCL_ENHANCEMENT}" >&2
        exit 2
        ;;
esac

if [ "${HCL_ENHANCEMENT}" = "none" ]; then
    RUN_TAG=hcl
else
    RUN_TAG=hcl_${HCL_ENHANCEMENT}
fi
RESULT_DIR=results/${RUN_TAG}

CNT=25
D=6
modeldim="512-384-256-128"
S='half'
B=1024
LOSS='hi-dist-xent'
TS=$(date "+%m.%d-%H.%M.%S")
MODEL_DIR=models/${RUN_TAG}/${DATA}/${CNT}
RUN_DIR=experiments/${RESULT_DIR}/${CNT}/${TS}
RUN_NAME=${DATA_TAG}_${RUN_TAG}_cnt${CNT}_${SEQ}_seed${SEED}_test_${TEST_START}_${TEST_END}
mkdir -p ${RUN_DIR} ${MODEL_DIR}

nohup python -u base.py	                                \
            --method 'hcl'                                 \
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
            --hcl-enhancement ${HCL_ENHANCEMENT}            \
            --xent-lambda 100                               \
            --display-interval 180                          \
            --al                                            \
            --count ${CNT}                                  \
            --F1D ${D}                                     \
            --local_pseudo_loss                             \
            --reduce "none"                                 \
            --sample_reduce 'mean'                          \
            --seed ${SEED}                                  \
            --model-dir ${MODEL_DIR}                        \
            --result ${RUN_DIR}/${RUN_NAME}.csv             \
            --log_path ${RUN_DIR}/${RUN_NAME}.log           \
            >> ${RUN_DIR}/${RUN_NAME}.log 2>&1 &

wait
