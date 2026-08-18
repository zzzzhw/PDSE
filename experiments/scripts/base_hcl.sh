#! /bin/bash

#SBATCH -t 03:00:00

#SBATCH -n 1

#SBATCH -c 8
# win
# 红伟电脑运行脚本
source C:/Users/sakura/anaconda3/etc/profile.d/conda.sh
conda activate apkencoder
set -aex
SEQ=${SEQ:-001}
SEED=${SEED:-2}
OPT=${OPT:-sgd}
SCH=${SCH:-step}
DECAY=${DECAY:-0.95}
E=${E:-250}
WLR=${WLR:-0.00015}
WE=${WE:-100}
# DATA=${1:-${DATA:-gen_androzoo_drebin}}
DATA=${1:-${DATA:-gen_apigraph_drebin}}
case ${DATA} in
    gen_androzoo_drebin)
        DATA_TAG=gen_androzoo
        TRAIN_START=2019-01
        TRAIN_END=2019-12
        DEFAULT_TEST_START=2020-01
        DEFAULT_TEST_END=2021-12
        # AndroZoo has 16k+ sparse features.  The APIGraph learning rate
        # causes the HCL encoder to overflow before the first checkpoint.
        DEFAULT_LEARNING_RATE=0.0003
        DEFAULT_HCL_GRAD_CLIP=1.0
        ;;
    gen_apigraph_drebin)
        DATA_TAG=gen_apigraph
        TRAIN_START=2012-01
        TRAIN_END=2012-12
        DEFAULT_TEST_START=2013-01
        DEFAULT_TEST_END=2018-12
        DEFAULT_LEARNING_RATE=0.003
        DEFAULT_HCL_GRAD_CLIP=0
        ;;
    *)
        echo "Unsupported dataset: ${DATA}" >&2
        exit 2
        ;;
esac
TEST_START=${TEST_START_OVERRIDE:-${DEFAULT_TEST_START}}
TEST_END=${TEST_END_OVERRIDE:-${DEFAULT_TEST_END}}
LR=${LR:-${DEFAULT_LEARNING_RATE}}
HCL_GRAD_CLIP=${HCL_GRAD_CLIP:-${DEFAULT_HCL_GRAD_CLIP}}
AL_OPT=${AL_OPT:-adam}

HCL_ENHANCEMENT=${HCL_ENHANCEMENT:-none}

HCL_RWP_GAMMA=${HCL_RWP_GAMMA:-0.01}
HCL_RWP_ALPHA=${HCL_RWP_ALPHA:-0.5}
HCL_GAUSSIAN_NOISE_STD=${HCL_GAUSSIAN_NOISE_STD:-0.01}
HCL_DAMP_STD=${HCL_DAMP_STD:-0.2}

case ${HCL_ENHANCEMENT} in
    none|mixup|focal|sam|rwp|gaussian_noise|damp) ;;
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

CNT=${CNT:-100}
D=${D:-6}
MODELDIM=${MODELDIM:-512-384-256-128}
SAMPLER=${SAMPLER:-half}
BATCH_SIZE=${BATCH_SIZE:-1024}
LOSS=${LOSS:-hi-dist-xent}
MODEL_DATE=${MODEL_DATE:-20240312}
TS=$(date "+%m.%d-%H.%M.%S")
MODEL_DIR=models/${RUN_TAG}/${DATA}/${CNT}
RUN_DIR=experiments/${RESULT_DIR}/${CNT}/${TS}
RUN_NAME=${DATA_TAG}_${RUN_TAG}_cnt${CNT}_${SEQ}_seed${SEED}_test_${TEST_START}_${TEST_END}
mkdir -p ${RUN_DIR} ${MODEL_DIR}

nohup python -u base.py	                                \
            --method 'hcl'                                 \
            --data ${DATA}                                  \
            --benign_zero                                   \
            --mdate ${MODEL_DATE}                           \
            --train_start ${TRAIN_START}                    \
            --train_end ${TRAIN_END}                        \
            --test_start ${TEST_START}                      \
            --test_end ${TEST_END}                          \
            --encoder simple-enc-mlp                        \
            --classifier simple-enc-mlp                     \
            --loss_func ${LOSS}                             \
            --enc-hidden ${MODELDIM}                        \
            --mlp-hidden 100-100                            \
            --mlp-dropout 0.2                               \
            --sampler ${SAMPLER}                            \
            --bsize ${BATCH_SIZE}                           \
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
            --hcl-rwp-gamma ${HCL_RWP_GAMMA}                \
            --hcl-rwp-alpha ${HCL_RWP_ALPHA}                \
            --hcl-gaussian-noise-std ${HCL_GAUSSIAN_NOISE_STD} \
            --hcl-damp-std ${HCL_DAMP_STD}                  \
            --hcl-grad-clip ${HCL_GRAD_CLIP}                \
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
