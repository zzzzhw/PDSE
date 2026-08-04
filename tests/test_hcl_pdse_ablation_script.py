from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "scripts"
    / "base_hcl_pdse_ablation.sh"
)


def test_ablation_script_exposes_both_modes_and_temporal_datasets():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "no-triplet|no-perturbation" in script
    assert "--pdse-ablation ${ABLATION}" in script
    assert "gen_apigraph_drebin)" in script
    assert "bodmas)" in script
    assert "kronodroid|kronodroid_real_2008_2014)" in script
    assert "TEST_START=${TEST_START_OVERRIDE:-${DEFAULT_TEST_START}}" in script
    assert "TEST_END=${TEST_END_OVERRIDE:-${DEFAULT_TEST_END}}" in script


def test_ablation_script_keeps_static_pdse_defaults_overridable():
    script = SCRIPT.read_text(encoding="utf-8")

    assert "PDSE_LAMBDA=${PDSE_LAMBDA:-0.001}" in script
    assert "PDSE_GAMMA=${PDSE_GAMMA:-1.0}" in script
    assert "PDSE_EMA_DECAY=${PDSE_EMA_DECAY:-0.6}" in script
    assert "PDSE_PROXY_LR=${PDSE_PROXY_LR:-1.0}" in script
    assert "PDSE_ROBUST_WEIGHT=${PDSE_ROBUST_WEIGHT:-1.0}" in script
    assert "PDSE_BATCH_SIZE=${PDSE_BATCH_SIZE:-192}" in script
    assert "PDSE_WARMUP=${PDSE_WARMUP:-5}" in script
    assert "PDSE_GRAD_CLIP=${PDSE_GRAD_CLIP:-1.0}" in script
