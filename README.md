# PDSE

This repository contains the PDSE implementation and scripts for reproducing the experiments. The Conda environment is defined in `environment.yml`, and experiments are launched through the Bash scripts in `experiments/scripts/`.

## 1. Create the Experimental Environment

Using Conda or Mamba, run the following commands from the repository root:

```bash
conda env create -f environment.yml
conda activate apkencoder
```

The environment name defined in `environment.yml` is `apkencoder`. To update an existing environment from the file, run:

```bash
conda env update -n apkencoder -f environment.yml --prune
```

The environment includes CUDA and PyTorch dependencies. Ensure that your GPU driver is compatible with the required CUDA runtime.

## 2. Prepare the Dataset

Download the dataset from:

[Google Drive dataset](https://drive.google.com/file/d/1O0upEcTolGyyvasCPkZFY86FNclk29XO/view)

After downloading, extract the archive under `data/` at the repository root, preserving the directory structure in the archive. The code expects paths such as:

```text
data/
├── gen_androzoo_drebin/
├── gen_apigraph_drebin/
```

Dataset files must retain the expected `.npz` names, for example, `2012-01to2012-12_selected.npz`. If a run raises `FileNotFoundError`, verify that the archive was extracted under `data/` and that its directory name matches the selected dataset.

## 3. Run Experiments

All experiment entry points are located in `experiments/scripts/`. Run the scripts with Bash from the repository root. Linux, macOS, WSL, and Git Bash can run these `.sh` scripts.

```bash
bash experiments/scripts/base_hcl_pdse.sh gen_apigraph_drebin
```

The first positional argument is usually the dataset name. Supported datasets vary by script; common values include:

- `gen_androzoo_drebin`
- `gen_apigraph_drebin`

The main experiment scripts are:

| Script | Method |
| --- | --- |
| `base_svm.sh` | SVM baseline |
| `base_resnet.sh` | Uncertainty baseline |
| `base_cade.sh` | CADE baseline |
| `base_cade_pdse.sh` | CADE + PDSE |
| `base_hcl.sh` | HCL |
| `base_hcl_pdse.sh` | HCL + PDSE |
| `base_hcl_pdse_ablation.sh` | HCL + PDSE ablation study |

Environment variables can override each script's default hyperparameters. For example, run HCL + PDSE with a specific random seed, query count, and test period:

```bash
  bash experiments/scripts/base_hcl_pdse.sh gen_apigraph_drebin
```

## 4. Outputs

During execution, model checkpoints are saved in `models/`. Result CSV files and logs are saved in `experiments/results/`, unless a script specifies another result subdirectory. These generated artifacts are excluded from version control by `.gitignore`.
