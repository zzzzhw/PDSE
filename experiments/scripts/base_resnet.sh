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

SEQ=${SEQ:-002}
SEED=${SEED:-1}
LR=${LR:-0.0009}
E=${E:-25}
WLR=${WLR:-0.000045}
WE=${WE:-25}

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

CNT=${CNT:-200}

D=${D:-6}
MODEL_DATE=${MODEL_DATE:-20240412}

TS=$(date "+%m.%d-%H.%M.%S")
MODEL_DIR=models/resnet/${DATA}/${CNT}
RUN_DIR=experiments/results/resnet/${CNT}/${TS}
RUN_NAME=${DATA_TAG}_resnet_cnt${CNT}_${SEQ}_seed${SEED}_test_${TEST_START}_${TEST_END}
mkdir -p "${RUN_DIR}" "${MODEL_DIR}"

nohup python -u base.py                                      \
    --method resnet                                          \
    --data "${DATA}"                                       \
    --benign_zero                                            \
    --mdate "${MODEL_DATE}"                                \
    --train_start "${TRAIN_START}"                         \
    --train_end "${TRAIN_END}"                             \
    --test_start "${TEST_START}"                           \
    --test_end "${TEST_END}"                               \
    --classifier res                                         \
    --mlp-hidden 100-100                                     \
    --mlp-dropout 0.2                                        \
    --mlp-batch-size 32                                      \
    --mlp-lr "${LR}"                                       \
    --mlp-epochs "${E}"                                    \
    --mlp-warm-lr "${WLR}"                                 \
    --mlp-warm-epochs "${WE}"                              \
    --al                                                     \
    --unc                                                    \
    --count "${CNT}"                                       \
    --F1D "${D}"                                           \
    --seed "${SEED}"                                       \
    --model-dir "${MODEL_DIR}"                             \
    --result "${RUN_DIR}/${RUN_NAME}.csv"                  \
    --log_path "${RUN_DIR}/${RUN_NAME}.log"                \
    >> "${RUN_DIR}/${RUN_NAME}.log" 2>&1 &

wait
