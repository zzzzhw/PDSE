#!/usr/bin/env bash

#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

CONDA_ENV=${CONDA_ENV:-apkencoder}
CONDA_SH=${CONDA_SH:-C:/Users/sakura/anaconda3/etc/profile.d/conda.sh}
if [[ -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
fi
PYTHON_BIN=${PYTHON_BIN:-python}

SEED=${SEED:-1}
DATA=${DATA:-gen_apigraph_drebin}
TRAIN_START=${TRAIN_START:-2012-01}
TRAIN_END=${TRAIN_END:-2012-12}
TEST_START=${TEST_START:-2013-01}
TEST_END=${TEST_END:-2018-12}
COUNT=${COUNT:-200}
F1D=${F1D:-6}
MODEL_DATE=${MODEL_DATE:-20230501}

LEARNING_RATE=${LEARNING_RATE:-0.001}
INITIAL_EPOCHS=${INITIAL_EPOCHS:-250}
WARM_LEARNING_RATE=${WARM_LEARNING_RATE:-0.00005}
MONTHLY_EPOCHS=${MONTHLY_EPOCHS:-50}
MLP_LEARNING_RATE=${MLP_LEARNING_RATE:-0.001}
MLP_WARM_LEARNING_RATE=${MLP_WARM_LEARNING_RATE:-0.00005}
MLP_EPOCHS=${MLP_EPOCHS:-50}

IDS_LAMBDA=${IDS_LAMBDA:-0.001}
IDS_GAMMA=${IDS_GAMMA:-1.0}
IDS_EMA_DECAY=${IDS_EMA_DECAY:-0.6}
IDS_PROXY_LR=${IDS_PROXY_LR:-1.0}
IDS_ROBUST_WEIGHT=${IDS_ROBUST_WEIGHT:-1.0}
IDS_BATCH_SIZE=${IDS_BATCH_SIZE:-192}
IDS_WARMUP=${IDS_WARMUP:-5}
IDS_GRAD_CLIP=${IDS_GRAD_CLIP:-1.0}

TIMESTAMP=${TIMESTAMP:-$(date "+%Y%m%d-%H%M%S")}
RESULT_ROOT=${RESULT_ROOT:-experiments/results/cade_ids_apigraph/bash}
RUN_DIR=${RUN_DIR:-${RESULT_ROOT}/${TIMESTAMP}}
MODEL_DIR=${MODEL_DIR:-models/experiments/cade_ids_apigraph/bash/${TIMESTAMP}/${COUNT}}
RUN_NAME="cade_ids_apigraph_seed${SEED}_cnt${COUNT}_${TEST_START}_${TEST_END}"
RESULT_PATH="${RUN_DIR}/${RUN_NAME}.csv"
LOG_PATH="${RUN_DIR}/${RUN_NAME}.log"
CONSOLE_PATH="${RUN_DIR}/${RUN_NAME}.console.log"

mkdir -p "${RUN_DIR}" "${MODEL_DIR}"

printf 'Repository: %s\n' "${REPO_ROOT}"
printf 'Result:     %s\n' "${RESULT_PATH}"
printf 'Log:        %s\n' "${LOG_PATH}"
printf 'Models:     %s\n' "${MODEL_DIR}"

"${PYTHON_BIN}" -u base.py                              \
    --method cade                                       \
    --data "${DATA}"                                   \
    --benign_zero                                       \
    --mdate "${MODEL_DATE}"                            \
    --train_start "${TRAIN_START}"                     \
    --train_end "${TRAIN_END}"                         \
    --test_start "${TEST_START}"                       \
    --test_end "${TEST_END}"                           \
    --encoder cae                                       \
    --enc-hidden "512-384-256-128"                     \
    --loss_func triplet-mse                             \
    --sampler triplet                                   \
    --bsize 1536                                        \
    --optimizer adam                                    \
    --scheduler cosine                                  \
    --learning_rate "${LEARNING_RATE}"                 \
    --lr_decay_rate 1                                   \
    --lr_decay_epochs "10,500,10"                      \
    --epochs "${INITIAL_EPOCHS}"                       \
    --cae-lambda 0.1                                    \
    --display-interval 100                              \
    --classifier mlp                                    \
    --cls-feat encoded                                  \
    --mlp-hidden "100-100"                             \
    --mlp-dropout 0.2                                   \
    --mlp-batch-size 32                                 \
    --mlp-lr "${MLP_LEARNING_RATE}"                    \
    --mlp-epochs "${MLP_EPOCHS}"                       \
    --al                                                \
    --ood                                               \
    --count "${COUNT}"                                 \
    --F1D "${F1D}"                                     \
    --encoder_retrain                                   \
    --warm_learning_rate "${WARM_LEARNING_RATE}"       \
    --al_epochs "${MONTHLY_EPOCHS}"                    \
    --mlp-warm-lr "${MLP_WARM_LEARNING_RATE}"          \
    --mlp-warm-epochs "${MLP_EPOCHS}"                  \
    --seed "${SEED}"                                   \
    --model-dir "${MODEL_DIR}"                         \
    --ids                                               \
    --ids-lambda "${IDS_LAMBDA}"                       \
    --ids-gamma "${IDS_GAMMA}"                         \
    --ids-ema-decay "${IDS_EMA_DECAY}"                 \
    --ids-proxy-lr "${IDS_PROXY_LR}"                   \
    --ids-robust-weight "${IDS_ROBUST_WEIGHT}"         \
    --ids-batch-size "${IDS_BATCH_SIZE}"               \
    --ids-warmup "${IDS_WARMUP}"                       \
    --ids-grad-clip "${IDS_GRAD_CLIP}"                 \
    --result "${RESULT_PATH}"                          \
    --log_path "${LOG_PATH}"                           \
    > "${CONSOLE_PATH}" 2>&1

printf 'CADE+IDS completed. Result: %s\n' "${RESULT_PATH}"
