#!/bin/bash

#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH -c 8

source C:/Users/sakura/anaconda3/etc/profile.d/conda.sh
conda activate apkencoder
set -aex

ABLATION=${1:-${ABLATION:-no-triplet}}
DATA=${2:-${DATA:-gen_apigraph_drebin}}
case ${ABLATION} in
    no-triplet|no-perturbation) ;;
    *)
        echo "Unsupported ablation: ${ABLATION}" >&2
        echo "Use no-triplet or no-perturbation" >&2
        exit 2
        ;;
esac

case ${DATA} in
    gen_apigraph_drebin)
        DATA_TAG=gen_apigraph
        TRAIN_START=2012-01
        TRAIN_END=2012-12
        DEFAULT_TEST_START=2013-01
        DEFAULT_TEST_END=2018-12
        ;;
    bodmas)
        DATA_TAG=bodmas
        TRAIN_START=2007-01
        TRAIN_END=2019-09
        DEFAULT_TEST_START=2019-10
        DEFAULT_TEST_END=2020-09
        ;;
    kronodroid|kronodroid_real_2008_2014)
        DATA_TAG=kronodroid_real
        TRAIN_START=2008-01
        TRAIN_END=2010-12
        DEFAULT_TEST_START=2011-01
        DEFAULT_TEST_END=2014-12
        ;;
    *)
        echo "Unsupported dataset: ${DATA}" >&2
        exit 2
        ;;
esac

SEQ=${SEQ:-001}
SEED=${SEED:-1}
LR=${LR:-0.003}
OPT=${OPT:-sgd}
SCH=${SCH:-step}
DECAY=${DECAY:-0.95}
E=${E:-250}
WLR=${WLR:-0.00015}
WE=${WE:-100}
CNT=${CNT:-50}
TEST_START=${TEST_START_OVERRIDE:-${DEFAULT_TEST_START}}
TEST_END=${TEST_END_OVERRIDE:-${DEFAULT_TEST_END}}
PDSE_WARMUP=${PDSE_WARMUP:-5}

RESULT_DIR=${RESULT_DIR:-results/hcl_pdse_ablation/${ABLATION}/${DATA_TAG}}
MODEL_DIR=models/hcl_pdse_ablation/${ABLATION}/${DATA}/${CNT}
AL_OPT=${AL_OPT:-adam}
MODELDIM=${MODELDIM:-512-384-256-128}
SAMPLER=${SAMPLER:-half}
BATCH_SIZE=${BATCH_SIZE:-1024}
LOSS=${LOSS:-hi-dist-xent}
MODEL_DATE=${MODEL_DATE:-20240312}
PDSE_LAMBDA=${PDSE_LAMBDA:-0.001}
PDSE_GAMMA=${PDSE_GAMMA:-1.0}
PDSE_EMA_DECAY=${PDSE_EMA_DECAY:-0.6}
PDSE_PROXY_LR=${PDSE_PROXY_LR:-1.0}
PDSE_ROBUST_WEIGHT=${PDSE_ROBUST_WEIGHT:-1.0}
PDSE_BATCH_SIZE=${PDSE_BATCH_SIZE:-192}
PDSE_GRAD_CLIP=${PDSE_GRAD_CLIP:-1.0}
TS=$(date "+%m.%d-%H.%M.%S")
RUN_DIR=experiments/${RESULT_DIR}/${CNT}/${TS}
RUN_NAME=${DATA_TAG}_hcl_pdse_${ABLATION}_cnt${CNT}_${SEQ}_seed${SEED}_test_${TEST_START}_${TEST_END}
mkdir -p ${RUN_DIR} ${MODEL_DIR}

python -u base.py                                           \
    --method hcl                                            \
    --data ${DATA}                                          \
    --benign_zero                                           \
    --mdate ${MODEL_DATE}                                   \
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
    --local_pseudo_loss                                     \
    --reduce none                                           \
    --sample_reduce mean                                    \
    --seed ${SEED}                                          \
    --model-dir ${MODEL_DIR}                                \
    --pdse                                                  \
    --pdse-ablation ${ABLATION}                             \
    --pdse-lambda ${PDSE_LAMBDA}                            \
    --pdse-gamma ${PDSE_GAMMA}                              \
    --pdse-ema-decay ${PDSE_EMA_DECAY}                      \
    --pdse-proxy-lr ${PDSE_PROXY_LR}                        \
    --pdse-robust-weight ${PDSE_ROBUST_WEIGHT}              \
    --pdse-hcl-update-mode combined                         \
    --pdse-batch-size ${PDSE_BATCH_SIZE}                    \
    --pdse-warmup ${PDSE_WARMUP}                            \
    --pdse-grad-clip ${PDSE_GRAD_CLIP}                      \
    --result ${RUN_DIR}/${RUN_NAME}.csv                     \
    --log_path ${RUN_DIR}/${RUN_NAME}.log
