"""Generate separate APIGraph budget comparisons for monthly HCL/PDSE F1."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import pandas as pd


# Configuration: edit these values, then run this file directly.
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
START_MONTH = "2013-01"
END_MONTH = "2018-12"
OUTPUT_SVG_PATHS = {
    budget: SCRIPT_DIR / f"hcl_pdse_apigraph_budget{budget}_f1.svg"
    for budget in (25, 50, 100, 200)
}

RESULT_PATHS = {
    25: {
        "HCL": Path(
            "results/hcl/25/07.25-14.20.41/"
            "gen_apigraph_cnt25_001_warm_lr0.003_sgd_step_0.95_e250_adam_"
            "wlr0.00015_we100_test_2013-01_2018-12_cnt25.csv"
        ),
        "PDSE": Path(
            "results/hcl_pdse_combined/25/07.30-15.34.57/"
            "gen_apigraph_hcl_pdse_combined_cnt25_001_seed1_"
            "test_2013-01_2018-12.csv"
        ),
    },
    50: {
        "HCL": Path(
            "results/hcl/50/07.22-10.54.56/"
            "gen_apigraph_cnt50_001_warm_lr0.003_sgd_step_0.95_e250_adam_"
            "wlr0.00015_we100_test_2013-01_2018-12_cnt50.csv"
        ),
        "PDSE": Path(
            "results/hcl_pdse_combined/50/08.03-18.38.54/"
            "gen_apigraph_hcl_pdse_combined_cnt50_001_seed1_"
            "test_2013-01_2018-12.csv"
        ),
    },
    100: {
        "HCL": Path(
            "results/hcl/100/07.22-11.28.58/"
            "gen_apigraph_cnt100_001_warm_lr0.003_sgd_step_0.95_e250_adam_"
            "wlr0.00015_we100_test_2013-01_2018-12_cnt100.csv"
        ),
        "PDSE": Path(
            "results/hcl_pdse_combined/100/08.03-18.47.15/"
            "gen_apigraph_hcl_pdse_combined_cnt100_001_seed1_"
            "test_2013-01_2018-12.csv"
        ),
    },
    200: {
        "HCL": Path(
            "results/hcl/200/07.22-16.19.15/"
            "gen_apigraph_cnt200_001_warm_lr0.003_sgd_step_0.95_e250_adam_"
            "wlr0.00015_we100_test_2013-01_2018-12_cnt200.csv"
        ),
        "PDSE": Path(
            "results/hcl_pdse_combined/200/08.03-18.47.53/"
            "gen_apigraph_hcl_pdse_combined_cnt200_001_seed1_"
            "test_2013-01_2018-12.csv"
        ),
    },
}

MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else EXPERIMENTS_DIR / path


def read_monthly_f1(path: Path) -> pd.Series:
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")

    frame = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    columns = {str(column).strip().lower(): column for column in frame.columns}
    month_column = next(
        (columns[name] for name in ("date", "month") if name in columns), None
    )
    f1_column = next(
        (columns[name] for name in ("f1", "f1_score", "f1 score") if name in columns),
        None,
    )
    if month_column is None or f1_column is None:
        raise ValueError(f"{path} must contain date/month and F1 columns")

    months = frame[month_column].astype(str).str.strip()
    if not months.str.match(MONTH_PATTERN).all():
        raise ValueError(f"All months in {path} must use YYYY-MM format")
    if months.duplicated().any():
        raise ValueError(f"Duplicate months found in {path}")

    scores = pd.to_numeric(frame[f1_column], errors="coerce")
    if scores.isna().any() or ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"F1 values in {path} must be numeric and between 0 and 1")

    index = pd.to_datetime(months, format="%Y-%m")
    series = pd.Series(scores.to_numpy(dtype=float), index=index).sort_index()
    start = pd.to_datetime(START_MONTH, format="%Y-%m")
    end = pd.to_datetime(END_MONTH, format="%Y-%m")
    return series[(series.index >= start) & (series.index <= end)]


def plot_budget(axis: plt.Axes, hcl: pd.Series, pdse: pd.Series) -> None:
    months = hcl.index.union(pdse.index).sort_values()
    if months.empty:
        raise ValueError("No data remains in the configured month range")

    axis.plot(
        months,
        hcl.reindex(months),
        color="#2878B5",
        linewidth=1.9,
        marker="o",
        markersize=2.8,
        label="HCL",
    )
    axis.plot(
        months,
        pdse.reindex(months),
        color="#D95319",
        linewidth=1.9,
        marker="s",
        markersize=2.8,
        label="PDSE",
    )
    axis.set_ylim(0.7, 1.0)
    axis.set_xlim(months.min(), months.max())
    tick_months = months[(months.month == 6) | (months.month == 12)]
    first_month = pd.to_datetime(START_MONTH, format="%Y-%m")
    tick_months = tick_months.insert(0, first_month).unique().sort_values()
    axis.xaxis.set_major_locator(FixedLocator(mdates.date2num(tick_months.to_pydatetime())))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.tick_params(axis="x", labelrotation=45, labelsize=11)
    axis.tick_params(axis="y", labelsize=12)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, linestyle="--", alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", frameon=False, fontsize=12)


def main() -> int:
    try:
        for budget, paths in RESULT_PATHS.items():
            hcl = read_monthly_f1(resolve_path(paths["HCL"]))
            pdse = read_monthly_f1(resolve_path(paths["PDSE"]))
            figure, axis = plt.subplots(figsize=(11, 5.8), dpi=120)
            plot_budget(axis, hcl, pdse)
            axis.set_ylabel("F1 Score", fontsize=14)
            figure.tight_layout()
            figure.savefig(OUTPUT_SVG_PATHS[budget], format="svg", bbox_inches="tight")
            plt.close(figure)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    for budget, output_path in OUTPUT_SVG_PATHS.items():
        print(f"Budget {budget}: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
