#!/bin/bash

#SBATCH -t 12:00:00
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

SEQ=${SEQ:-127}
SEED=${SEED:-1}
LR=${LR:-0.0001}
OPT=${OPT:-adam}
SCH=${SCH:-cosine}
DECAY=${DECAY:-1}
E=${E:-100}
WLR=${WLR:-0.00005}
WE=${WE:-50}
MWLR=${MWLR:-0.00005}
MWE=${MWE:-50}

DATA=${1:-${DATA:-kronodroid}}
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
SAMPLER=${SAMPLER:-triplet}
BATCH_SIZE=${BATCH_SIZE:-1536}
LOSS=${LOSS:-triplet-mse}
MODEL_DATE=${MODEL_DATE:-20230501}

TS=$(date "+%m.%d-%H.%M.%S")
MODEL_DIR=models/cade/${DATA}/${CNT}
RUN_DIR=experiments/results/cade/${CNT}/${TS}
RUN_NAME=${DATA_TAG}_cade_cnt${CNT}_${SEQ}_seed${SEED}_test_${TEST_START}_${TEST_END}
mkdir -p "${RUN_DIR}" "${MODEL_DIR}"

nohup python -u base.py                                      \
    --method cade                                            \
    --data "${DATA}"                                       \
    --benign_zero                                            \
    --mdate "${MODEL_DATE}"                                \
    --train_start "${TRAIN_START}"                         \
    --train_end "${TRAIN_END}"                             \
    --test_start "${TEST_START}"                           \
    --test_end "${TEST_END}"                               \
    --encoder cae                                            \
    --enc-hidden "${MODELDIM}"                             \
    --loss_func "${LOSS}"                                  \
    --sampler "${SAMPLER}"                                 \
    --bsize "${BATCH_SIZE}"                                \
    --optimizer "${OPT}"                                   \
    --scheduler "${SCH}"                                   \
    --learning_rate "${LR}"                                \
    --lr_decay_rate "${DECAY}"                             \
    --lr_decay_epochs "10,500,10"                           \
    --epochs "${E}"                                        \
    --cae-lambda 0.1                                         \
    --display-interval 100                                   \
    --classifier mlp                                         \
    --cls-feat encoded                                       \
    --mlp-hidden 100-100                                     \
    --mlp-dropout 0.2                                        \
    --mlp-batch-size 32                                      \
    --mlp-lr 0.001                                           \
    --mlp-epochs "${MWE}"                                  \
    --al                                                     \
    --ood                                                    \
    --count "${CNT}"                                       \
    --F1D "${D}"                                           \
    --encoder_retrain                                        \
    --al_optimizer "${AL_OPT}"                             \
    --warm_learning_rate "${WLR}"                          \
    --al_epochs "${WE}"                                    \
    --mlp-warm-lr "${MWLR}"                                \
    --mlp-warm-epochs "${MWE}"                             \
    --seed "${SEED}"                                       \
    --model-dir "${MODEL_DIR}"                             \
    --result "${RUN_DIR}/${RUN_NAME}.csv"                  \
    --log_path "${RUN_DIR}/${RUN_NAME}.log"                \
    >> "${RUN_DIR}/${RUN_NAME}.log" 2>&1 &

wait
