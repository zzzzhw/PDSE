"""Generate the APIGraph monthly HCL/PDSE FNR comparison as an SVG."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


# FNR chart configuration: edit these values, then run this file directly.
# Relative input paths are resolved from the repository's experiments folder.
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent

HCL_CSV_PATH = Path(
    "results/hcl/200/07.22-16.19.15/"
    "gen_apigraph_cnt200_001_warm_lr0.003_sgd_step_0.95_e250_adam_"
    "wlr0.00015_we100_test_2013-01_2018-12_cnt200.csv"
)
PDSE_CSV_PATH = Path(
    "results/hcl_pdse_combined/200/08.03-18.47.53/"
    "gen_apigraph_hcl_pdse_combined_cnt200_001_seed1_"
    "test_2013-01_2018-12.csv"
)
OUTPUT_SVG_PATH = SCRIPT_DIR / "hcl_pdse_apigraph_fnr.svg"
START_MONTH = "2013-01"
END_MONTH = "2018-12"
CHART_TITLE = "HCL vs. PDSE FNR on APIGraph"


MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def resolve_input_path(path: Path) -> Path:
    """Resolve a configured relative path from the experiments directory."""
    return path if path.is_absolute() else EXPERIMENTS_DIR / path


def read_monthly_fnr(path: Path) -> pd.Series:
    """Read and validate monthly FNR data from a comma- or tab-separated file."""
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")

    try:
        frame = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc

    columns = {str(column).strip().lower(): column for column in frame.columns}
    month_column = next(
        (columns[name] for name in ("date", "month") if name in columns), None
    )
    fnr_column = next(
        (columns[name] for name in ("fnr", "fnr_score", "fnr score") if name in columns),
        None,
    )
    if month_column is None or fnr_column is None:
        available = ", ".join(map(str, frame.columns)) or "<none>"
        raise ValueError(
            f"{path} must contain a date/month column and an FNR column; "
            f"found: {available}"
        )

    months = frame[month_column].astype(str).str.strip()
    invalid_months = months[~months.str.match(MONTH_PATTERN)]
    if not invalid_months.empty:
        examples = ", ".join(invalid_months.head(3))
        raise ValueError(f"Invalid month in {path} (expected YYYY-MM): {examples}")

    scores = pd.to_numeric(frame[fnr_column], errors="coerce")
    invalid_scores = frame.loc[scores.isna(), fnr_column]
    if not invalid_scores.empty:
        examples = ", ".join(map(str, invalid_scores.head(3)))
        raise ValueError(f"Non-numeric FNR value in {path}: {examples}")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"FNR values in {path} must be between 0 and 1")
    if months.duplicated().any():
        duplicates = ", ".join(months[months.duplicated(keep=False)].unique())
        raise ValueError(f"Duplicate months in {path}: {duplicates}")

    index = pd.to_datetime(months, format="%Y-%m")
    return pd.Series(scores.to_numpy(dtype=float), index=index, name="FNR").sort_index()


def plot_fnr(
    hcl: pd.Series,
    pdse: pd.Series,
    output: Path,
    title: str,
    start_month: str | None,
    end_month: str | None,
) -> None:
    """Render the two monthly series to an SVG file."""
    start = pd.to_datetime(start_month, format="%Y-%m") if start_month else None
    end = pd.to_datetime(end_month, format="%Y-%m") if end_month else None
    if start is not None and end is not None and start > end:
        raise ValueError("START_MONTH must not be later than END_MONTH")

    if start is not None:
        hcl, pdse = hcl[hcl.index >= start], pdse[pdse.index >= start]
    if end is not None:
        hcl, pdse = hcl[hcl.index <= end], pdse[pdse.index <= end]
    if hcl.empty and pdse.empty:
        raise ValueError("No data remains in the configured month range")

    all_months = hcl.index.union(pdse.index).sort_values()
    hcl = hcl.reindex(all_months)
    pdse = pdse.reindex(all_months)

    figure, axis = plt.subplots(figsize=(13, 6.5), dpi=120)
    axis.plot(
        all_months,
        hcl,
        color="#2878B5",
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label="HCL",
    )
    axis.plot(
        all_months,
        pdse,
        color="#D95319",
        linewidth=2.0,
        marker="s",
        markersize=3.5,
        label="PDSE",
    )

    axis.set_title(title, fontsize=15, pad=12)
    axis.set_xlabel("Month", fontsize=12)
    axis.set_ylabel("FNR", fontsize=12)
    axis.set_ylim(0.0, 0.3)
    if len(all_months) <= 18:
        tick_months = range(1, 13)
    elif len(all_months) <= 36:
        tick_months = range(1, 13, 3)
    else:
        tick_months = range(1, 13, 6)
    axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=tick_months))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.set_xlim(all_months.min(), all_months.max())
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, linestyle="--", alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="best", frameon=False, fontsize=11)
    figure.autofmt_xdate(rotation=45, ha="right")
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    hcl_path = resolve_input_path(HCL_CSV_PATH)
    pdse_path = resolve_input_path(PDSE_CSV_PATH)
    try:
        hcl = read_monthly_fnr(hcl_path)
        pdse = read_monthly_fnr(pdse_path)
        plot_fnr(
            hcl,
            pdse,
            OUTPUT_SVG_PATH,
            CHART_TITLE,
            START_MONTH,
            END_MONTH,
        )
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"HCL input:  {hcl_path}")
    print(f"PDSE input: {pdse_path}")
    print(f"SVG written to: {OUTPUT_SVG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
