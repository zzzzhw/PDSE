#!/bin/bash

#SBATCH -t 03:00:00
#SBATCH -n 1
#SBATCH -c 8

set -o pipefail

CONDA_SH=${CONDA_SH:-C:/Users/sakura/anaconda3/etc/profile.d/conda.sh}
source "${CONDA_SH}"
conda activate "${CONDA_ENV:-apkencoder}"
set -eu

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

SEQ=${SEQ:-088}
SEED=${SEED:-1}
LR=${LR:-0.003}
OPT=${OPT:-sgd}
SCH=${SCH:-step}
DECAY=${DECAY:-0.95}
E=${E:-100}
WLR=${WLR:-0.00015}
WE=${WE:-100}

DATA=${1:-${DATA:-gen_androzoo_drebin}}
# DATA=${1:-${DATA:-gen_apigraph_drebin}}

case ${DATA} in
    gen_androzoo_drebin)
        DATA_TAG=gen_androzoo
        TRAIN_START=2019-01
        TRAIN_END=2019-12
        DEFAULT_TEST_START=2020-01
        DEFAULT_TEST_END=2021-12
        ;;
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

TEST_START=${TEST_START_OVERRIDE:-${DEFAULT_TEST_START}}
TEST_END=${TEST_END_OVERRIDE:-${DEFAULT_TEST_END}}
AL_OPT=${AL_OPT:-adam}

CNT=${CNT:-200}

D=${D:-6}
MODELDIM=${MODELDIM:-512-384-256-128}
SAMPLER=${SAMPLER:-half}
BATCH_SIZE=${BATCH_SIZE:-1024}
LOSS=${LOSS:-hi-dist}
MODEL_DATE=${MODEL_DATE:-20230501}

TS=$(date "+%m.%d-%H.%M.%S")
MODEL_DIR=models/svm/${DATA}/${CNT}
RUN_DIR=experiments/results/svm/${CNT}/${TS}
RUN_NAME=${DATA_TAG}_svm_cnt${CNT}_${SEQ}_seed${SEED}_${LOSS}_transcend_test_${TEST_START}_${TEST_END}
mkdir -p "${RUN_DIR}" "${MODEL_DIR}"

nohup python -u base.py                                      \
    --method svm                                             \
    --data "${DATA}"                                       \
    --benign_zero                                            \
    --mdate "${MODEL_DATE}"                                \
    --train_start "${TRAIN_START}"                         \
    --train_end "${TRAIN_END}"                             \
    --test_start "${TEST_START}"                           \
    --test_end "${TEST_END}"                               \
    --encoder enc                                            \
    --classifier svm                                         \
    --cls-feat encoded                                       \
    --cls-retrain 1                                          \
    --cold-start                                             \
    --loss_func "${LOSS}"                                  \
    --enc-hidden "${MODELDIM}"                             \
    --sampler "${SAMPLER}"                                 \
    --bsize "${BATCH_SIZE}"                                \
    --optimizer "${OPT}"                                   \
    --scheduler "${SCH}"                                   \
    --learning_rate "${LR}"                                \
    --lr_decay_rate "${DECAY}"                             \
    --lr_decay_epochs "10,500,10"                           \
    --epochs "${E}"                                        \
    --encoder_retrain                                        \
    --al_optimizer "${AL_OPT}"                             \
    --warm_learning_rate "${WLR}"                          \
    --al_epochs "${WE}"                                    \
    --display-interval 180                                   \
    --al                                                     \
    --count "${CNT}"                                       \
    --F1D "${D}"                                           \
    --transcend                                              \
    --criteria cred+conf                                     \
    --seed "${SEED}"                                       \
    --model-dir "${MODEL_DIR}"                             \
    --result "${RUN_DIR}/${RUN_NAME}.csv"                  \
    --log_path "${RUN_DIR}/${RUN_NAME}.log"                \
    >> "${RUN_DIR}/${RUN_NAME}.log" 2>&1 &

wait
