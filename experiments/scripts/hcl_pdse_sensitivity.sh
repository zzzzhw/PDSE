#!/usr/bin/env bash

#SBATCH -t 3-00:00:00
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1

set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${PROJECT_ROOT}"

if [[ -f "C:/Users/sakura/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "C:/Users/sakura/anaconda3/etc/profile.d/conda.sh"
    conda activate apkencoder
fi

# Conda activation hooks may read optional variables before defining them.
set -u

PYTHON=${PYTHON:-python}
DATA=gen_apigraph_drebin
TRAIN_START=2012-01
TRAIN_END=2012-12
VALID_START=${VALID_START:-2013-01}
VALID_END=${VALID_END:-2013-06}
COUNT=${COUNT:-25}
SEEDS=${SEEDS:-"1"}
FORCE=${FORCE:-0}

LR=0.003
OPT=sgd
SCHEDULER=step
DECAY=0.95
EPOCHS=250
WARM_LR=0.00015
WARM_EPOCHS=100
AL_OPT=adam
MODEL_DIMS="512-384-256-128"
TRAIN_BATCH_SIZE=1024

DEFAULT_LAMBDA=0.001
DEFAULT_GAMMA=1.0
DEFAULT_RHO=0.6
DEFAULT_BETA=1.0
DEFAULT_PROXY_LR=1.0
DEFAULT_PDSE_BATCH_SIZE=192
DEFAULT_WARMUP=5
GRAD_CLIP=1.0

LAMBDAS=(0.0001 0.0003 0.001 0.003 0.01)
GAMMAS=(0 0.1 1 10)
RHOS=(0 0.3 0.6 0.9)
BETAS=(0.25 0.5 1 2)
PROXY_LRS=(0.1 0.3 1 3)
PDSE_BATCH_SIZES=(48 96 192 384)
WARMUPS=(0 5 10 20)

SEARCH_ROOT=${SEARCH_ROOT:-"experiments/results/hcl_pdse_sensitivity/count_${COUNT}/valid_${VALID_START}_${VALID_END}"}
RUN_ROOT=${RUN_ROOT:-"${SEARCH_ROOT}/staged_search"}
MODEL_ROOT=${MODEL_ROOT:-"models/hcl_pdse_sensitivity/${DATA}/count_${COUNT}"}
STATUS_FILE="${RUN_ROOT}/run_status.tsv"

mkdir -p "${RUN_ROOT}" "${MODEL_ROOT}"
printf 'stage\tconfig\tseed\tstatus\n' > "${STATUS_FILE}"

seed_count() {
    local count=0
    local unused
    for unused in ${SEEDS}; do
        count=$((count + 1))
    done
    printf '%s\n' "${count}"
}

config_id() {
    local lambda=$1 gamma=$2 rho=$3 beta=$4 proxy_lr=$5 batch_size=$6 warmup=$7
    printf 'lam%s_gam%s_rho%s_beta%s_plr%s_pb%s_wu%s' \
        "${lambda}" "${gamma}" "${rho}" "${beta}" \
        "${proxy_lr}" "${batch_size}" "${warmup}"
}

run_one() {
    local stage=$1 lambda=$2 gamma=$3 rho=$4 beta=$5 proxy_lr=$6
    local pdse_batch_size=$7 warmup=$8 seed=$9
    local config
    config=$(config_id "${lambda}" "${gamma}" "${rho}" "${beta}" \
        "${proxy_lr}" "${pdse_batch_size}" "${warmup}")

    local run_dir="${RUN_ROOT}/${stage}/${config}/seed_${seed}"
    local result_file="${run_dir}/result.csv"
    local method_log="${run_dir}/run.log"
    local console_log="${run_dir}/console.log"
    local model_dir="${MODEL_ROOT}/seed_${seed}"
    mkdir -p "${run_dir}" "${model_dir}"

    printf 'lambda\tgamma\trho\tbeta\tproxy_lr\tbatch_size\twarmup\n' \
        > "${run_dir}/config.tsv"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${lambda}" "${gamma}" "${rho}" "${beta}" "${proxy_lr}" \
        "${pdse_batch_size}" "${warmup}" >> "${run_dir}/config.tsv"

    if [[ "${FORCE}" != "1" && -f "${result_file}" && -f "${method_log}" ]] \
        && [[ $(wc -l < "${result_file}") -ge 8 ]] \
        && grep -q 'time elapsed:' "${method_log}"; then
        printf '%s\t%s\t%s\tskipped_complete\n' \
            "${stage}" "${config}" "${seed}" >> "${STATUS_FILE}"
        return 0
    fi

    echo "[${stage}] ${config} seed=${seed}"
    if "${PYTHON}" -u base.py \
        --method hcl \
        --data "${DATA}" \
        --benign_zero \
        --mdate 20240312 \
        --train_start "${TRAIN_START}" \
        --train_end "${TRAIN_END}" \
        --test_start "${VALID_START}" \
        --test_end "${VALID_END}" \
        --encoder simple-enc-mlp \
        --classifier simple-enc-mlp \
        --loss_func hi-dist-xent \
        --enc-hidden "${MODEL_DIMS}" \
        --mlp-hidden 100-100 \
        --mlp-dropout 0.2 \
        --sampler half \
        --bsize "${TRAIN_BATCH_SIZE}" \
        --optimizer "${OPT}" \
        --scheduler "${SCHEDULER}" \
        --learning_rate "${LR}" \
        --lr_decay_rate "${DECAY}" \
        --lr_decay_epochs "10,500,10" \
        --epochs "${EPOCHS}" \
        --encoder_retrain \
        --al_optimizer "${AL_OPT}" \
        --warm_learning_rate "${WARM_LR}" \
        --al_epochs "${WARM_EPOCHS}" \
        --xent-lambda 100 \
        --display-interval 180 \
        --al \
        --count "${COUNT}" \
        --F1D 6 \
        --local_pseudo_loss \
        --reduce none \
        --sample_reduce mean \
        --seed "${seed}" \
        --model-dir "${model_dir}" \
        --pdse \
        --pdse-lambda "${lambda}" \
        --pdse-gamma "${gamma}" \
        --pdse-ema-decay "${rho}" \
        --pdse-proxy-lr "${proxy_lr}" \
        --pdse-robust-weight "${beta}" \
        --pdse-hcl-update-mode combined \
        --pdse-batch-size "${pdse_batch_size}" \
        --pdse-warmup "${warmup}" \
        --pdse-grad-clip "${GRAD_CLIP}" \
        --result "${result_file}" \
        --log_path "${method_log}" \
        > "${console_log}" 2>&1; then
        printf '%s\t%s\t%s\tcompleted\n' \
            "${stage}" "${config}" "${seed}" >> "${STATUS_FILE}"
    else
        printf '%s\t%s\t%s\tfailed\n' \
            "${stage}" "${config}" "${seed}" >> "${STATUS_FILE}"
        echo "WARNING: failed ${stage}/${config}/seed_${seed}; see ${console_log}" >&2
    fi
}

run_config() {
    local stage=$1 lambda=$2 gamma=$3 rho=$4 beta=$5 proxy_lr=$6
    local pdse_batch_size=$7 warmup=$8
    local seed
    for seed in ${SEEDS}; do
        run_one "${stage}" "${lambda}" "${gamma}" "${rho}" "${beta}" \
            "${proxy_lr}" "${pdse_batch_size}" "${warmup}" "${seed}"
    done
}

summarize_stage() {
    local stage=$1
    local stage_dir="${RUN_ROOT}/${stage}"
    local expected_seeds
    expected_seeds=$(seed_count)

    "${PYTHON}" - "${stage_dir}" "${expected_seeds}" \
        "${VALID_START}" "${VALID_END}" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

stage_dir = Path(sys.argv[1])
expected_seeds = int(sys.argv[2])
valid_start = sys.argv[3]
valid_end = sys.argv[4]
parameter_names = (
    "lambda", "gamma", "rho", "beta", "proxy_lr", "batch_size", "warmup"
)

grouped = defaultdict(list)
for config_path in stage_dir.glob("*/seed_*/config.tsv"):
    run_dir = config_path.parent
    result_path = run_dir / "result.csv"
    log_path = run_dir / "run.log"
    if not result_path.is_file() or not log_path.is_file():
        continue
    if "time elapsed:" not in log_path.read_text(encoding="utf-8", errors="replace"):
        continue

    with config_path.open(newline="", encoding="utf-8") as stream:
        config = next(csv.DictReader(stream, delimiter="\t"))
    with result_path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream, delimiter="\t")
            if valid_start <= row["date"] <= valid_end
        ]
    if not rows:
        continue

    f1_values = [float(row["F1"]) for row in rows]
    fnr_values = [float(row["FNR"]) for row in rows]
    key = tuple(config[name] for name in parameter_names)
    grouped[key].append({
        "mean_f1": statistics.fmean(f1_values),
        "mean_fnr": statistics.fmean(fnr_values),
        "min_f1": min(f1_values),
    })

summary = []
for key, seed_metrics in grouped.items():
    if len(seed_metrics) != expected_seeds:
        continue
    row = dict(zip(parameter_names, key))
    seed_f1 = [metric["mean_f1"] for metric in seed_metrics]
    row.update({
        "seeds": len(seed_metrics),
        "mean_f1": statistics.fmean(seed_f1),
        "std_f1": statistics.pstdev(seed_f1),
        "mean_fnr": statistics.fmean(metric["mean_fnr"] for metric in seed_metrics),
        "mean_min_f1": statistics.fmean(metric["min_f1"] for metric in seed_metrics),
    })
    summary.append(row)

if not summary:
    raise SystemExit(f"No complete configurations found in {stage_dir}")

summary.sort(key=lambda row: (
    -row["mean_f1"], row["mean_fnr"], -row["mean_min_f1"]
))
fieldnames = list(parameter_names) + [
    "seeds", "mean_f1", "std_f1", "mean_fnr", "mean_min_f1"
]
with (stage_dir / "summary.tsv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(summary)

best = summary[0]
print("\t".join(best[name] for name in parameter_names))
PY
}

read_best() {
    local best_line=$1
    IFS=$'\t' read -r BEST_LAMBDA BEST_GAMMA BEST_RHO BEST_BETA \
        BEST_PROXY_LR BEST_PDSE_BATCH_SIZE BEST_WARMUP <<< "${best_line}"
}

select_stage_best() {
    local stage=$1
    local best_line
    if ! best_line=$(summarize_stage "${stage}"); then
        echo "ERROR: could not select a complete configuration for ${stage}" >&2
        exit 1
    fi
    if [[ -z "${best_line}" ]]; then
        echo "ERROR: empty best configuration for ${stage}" >&2
        exit 1
    fi
    read_best "${best_line}"
}

echo "Search results: ${RUN_ROOT}"
echo "Validation stream: ${VALID_START} to ${VALID_END}"
echo "Label budget: ${COUNT}; seeds: ${SEEDS}"

# Stage 1: jointly tune perturbation strength and representation constraint.
for lambda in "${LAMBDAS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
        run_config stage1_lambda_gamma "${lambda}" "${gamma}" \
            "${DEFAULT_RHO}" "${DEFAULT_BETA}" "${DEFAULT_PROXY_LR}" \
            "${DEFAULT_PDSE_BATCH_SIZE}" "${DEFAULT_WARMUP}"
    done
done
select_stage_best stage1_lambda_gamma
echo "Stage 1 best: lambda=${BEST_LAMBDA}, gamma=${BEST_GAMMA}"

# Stage 2: tune EMA smoothing and robust-loss weight around the stage-1 winner.
for rho in "${RHOS[@]}"; do
    for beta in "${BETAS[@]}"; do
        run_config stage2_rho_beta "${BEST_LAMBDA}" "${BEST_GAMMA}" \
            "${rho}" "${beta}" "${DEFAULT_PROXY_LR}" \
            "${DEFAULT_PDSE_BATCH_SIZE}" "${DEFAULT_WARMUP}"
    done
done
select_stage_best stage2_rho_beta
echo "Stage 2 best: rho=${BEST_RHO}, beta=${BEST_BETA}"

# Stage 3: jointly tune proxy learning rate and exposure batch size.
for proxy_lr in "${PROXY_LRS[@]}"; do
    for pdse_batch_size in "${PDSE_BATCH_SIZES[@]}"; do
        run_config stage3_proxy_batch "${BEST_LAMBDA}" "${BEST_GAMMA}" \
            "${BEST_RHO}" "${BEST_BETA}" "${proxy_lr}" \
            "${pdse_batch_size}" "${DEFAULT_WARMUP}"
    done
done
select_stage_best stage3_proxy_batch
echo "Stage 3 best: proxy_lr=${BEST_PROXY_LR}, batch=${BEST_PDSE_BATCH_SIZE}"

# Stage 4: tune warmup after fixing all other parameters.
for warmup in "${WARMUPS[@]}"; do
    run_config stage4_warmup "${BEST_LAMBDA}" "${BEST_GAMMA}" \
        "${BEST_RHO}" "${BEST_BETA}" "${BEST_PROXY_LR}" \
        "${BEST_PDSE_BATCH_SIZE}" "${warmup}"
done
select_stage_best stage4_warmup

BEST_FILE="${RUN_ROOT}/best_params.tsv"
printf 'lambda\tgamma\trho\tbeta\tproxy_lr\tbatch_size\twarmup\n' > "${BEST_FILE}"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${BEST_LAMBDA}" "${BEST_GAMMA}" "${BEST_RHO}" "${BEST_BETA}" \
    "${BEST_PROXY_LR}" "${BEST_PDSE_BATCH_SIZE}" "${BEST_WARMUP}" \
    >> "${BEST_FILE}"

echo "Best validation parameters:"
echo "  lambda=${BEST_LAMBDA}"
echo "  gamma=${BEST_GAMMA}"
echo "  rho=${BEST_RHO}"
echo "  beta=${BEST_BETA}"
echo "  proxy_lr=${BEST_PROXY_LR}"
echo "  batch_size=${BEST_PDSE_BATCH_SIZE}"
echo "  warmup=${BEST_WARMUP}"
echo "Saved to ${BEST_FILE}"
