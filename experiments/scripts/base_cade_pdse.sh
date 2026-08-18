#!/bin/bash

#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 8

set -o pipefail

CONDA_SH=${CONDA_SH:-C:/Users/sakura/anaconda3/etc/profile.d/conda.sh}
source "${CONDA_SH}"
conda activate "${CONDA_ENV:-apkencoder}"

# Enable strict unset-variable checks only after Conda activation scripts run.
set -eu

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

SEQ=${SEQ:-001}
SEED=${SEED:-1}
DATA=${DATA:-gen_apigraph_drebin}
TRAIN_START=${TRAIN_START:-2012-01}
TRAIN_END=${TRAIN_END:-2012-12}
TEST_START=${TEST_START:-2013-01}
TEST_END=${TEST_END:-2018-12}

CNT=${CNT:-200}
F1D=${F1D:-6}
MODEL_DATE=${MODEL_DATE:-20230501}
MODELDIM=${MODELDIM:-512-384-256-128}
SAMPLER=${SAMPLER:-triplet}
BATCH_SIZE=${BATCH_SIZE:-1536}

OPT=${OPT:-adam}
SCH=${SCH:-cosine}
LR=${LR:-0.0001}
DECAY=${DECAY:-1.0}
EPOCHS=${EPOCHS:-100}
AL_OPT=${AL_OPT:-adam}
WARM_LR=${WARM_LR:-0.00005}
AL_EPOCHS=${AL_EPOCHS:-50}
MLP_WARM_LR=${MLP_WARM_LR:-0.00005}
MLP_EPOCHS=${MLP_EPOCHS:-50}
MLP_WARM_EPOCHS=${MLP_WARM_EPOCHS:-50}

PDSE_LAMBDA=${PDSE_LAMBDA:-0.001}
PDSE_GAMMA=${PDSE_GAMMA:-1.0}
PDSE_EMA_DECAY=${PDSE_EMA_DECAY:-0.6}
PDSE_PROXY_LR=${PDSE_PROXY_LR:-1.0}
PDSE_ROBUST_WEIGHT=${PDSE_ROBUST_WEIGHT:-1.0}
PDSE_BATCH_SIZE=${PDSE_BATCH_SIZE:-192}
PDSE_WARMUP=${PDSE_WARMUP:-5}
PDSE_GRAD_CLIP=${PDSE_GRAD_CLIP:-1.0}

TS=$(date "+%m.%d-%H.%M.%S")
RESULT_ROOT=${RESULT_ROOT:-experiments/results/cade_pdse_combined}
MODEL_ROOT=${MODEL_ROOT:-models/cade_pdse_combined/${DATA}/${CNT}}
RUN_DIR=${RESULT_ROOT}/${CNT}/${TS}
RUN_NAME=${DATA}_cade_pdse_cnt${CNT}_${SEQ}_seed${SEED}_test_${TEST_START}_${TEST_END}
mkdir -p "${RUN_DIR}" "${MODEL_ROOT}"

python -u base.py                                           \
    --method cade                                           \
    --data "${DATA}"                                       \
    --benign_zero                                           \
    --mdate "${MODEL_DATE}"                                \
    --train_start "${TRAIN_START}"                         \
    --train_end "${TRAIN_END}"                             \
    --test_start "${TEST_START}"                           \
    --test_end "${TEST_END}"                               \
    --encoder cae                                           \
    --enc-hidden "${MODELDIM}"                             \
    --loss_func triplet-mse                                 \
    --sampler "${SAMPLER}"                                 \
    --bsize "${BATCH_SIZE}"                                \
    --optimizer "${OPT}"                                   \
    --scheduler "${SCH}"                                   \
    --learning_rate "${LR}"                                \
    --lr_decay_rate "${DECAY}"                             \
    --lr_decay_epochs "10,500,10"                          \
    --epochs "${EPOCHS}"                                   \
    --cae-lambda 0.1                                        \
    --display-interval 100                                  \
    --classifier mlp                                        \
    --cls-feat encoded                                      \
    --mlp-hidden 100-100                                    \
    --mlp-dropout 0.2                                       \
    --mlp-batch-size 32                                     \
    --mlp-lr 0.001                                          \
    --mlp-epochs "${MLP_EPOCHS}"                           \
    --al                                                    \
    --ood                                                   \
    --count "${CNT}"                                       \
    --F1D "${F1D}"                                         \
    --encoder_retrain                                       \
    --al_optimizer "${AL_OPT}"                             \
    --warm_learning_rate "${WARM_LR}"                      \
    --al_epochs "${AL_EPOCHS}"                             \
    --mlp-warm-lr "${MLP_WARM_LR}"                         \
    --mlp-warm-epochs "${MLP_WARM_EPOCHS}"                 \
    --seed "${SEED}"                                       \
    --model-dir "${MODEL_ROOT}"                            \
    --pdse                                                  \
    --pdse-lambda "${PDSE_LAMBDA}"                         \
    --pdse-gamma "${PDSE_GAMMA}"                           \
    --pdse-ema-decay "${PDSE_EMA_DECAY}"                   \
    --pdse-proxy-lr "${PDSE_PROXY_LR}"                     \
    --pdse-robust-weight "${PDSE_ROBUST_WEIGHT}"            \
    --pdse-cade-update-mode combined                        \
    --pdse-batch-size "${PDSE_BATCH_SIZE}"                 \
    --pdse-warmup "${PDSE_WARMUP}"                         \
    --pdse-grad-clip "${PDSE_GRAD_CLIP}"                   \
    --result "${RUN_DIR}/${RUN_NAME}.csv"                  \
    --log_path "${RUN_DIR}/${RUN_NAME}.log"
