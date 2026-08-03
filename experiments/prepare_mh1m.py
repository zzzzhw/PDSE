"""Split the processed MH-1M feature matrix by VirusTotal analysis month."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOURCE_NAME = "amex-1M-[intents-permissions-opcodes-apicalls].npz"
DATE_COLUMN = "VT_LAST_ANALYSIS_DATE"
LABEL_COLUMN = "CLASS"


def _metadata_key(source, primary, fallback=None):
    if primary in source.files:
        return primary
    if fallback is not None and fallback in source.files:
        return fallback
    available = ", ".join(source.files)
    raise KeyError(f"Missing {primary!r} in source NPZ; available keys: {available}")


def _normalize_timestamps(values):
    timestamps = pd.to_datetime(
        pd.Series(values, dtype="object"), errors="coerce", utc=True
    )
    invalid = np.flatnonzero(timestamps.isna().to_numpy())
    if invalid.size:
        preview = ", ".join(map(str, invalid[:10]))
        raise ValueError(
            f"{DATE_COLUMN} contains {invalid.size} invalid timestamps; "
            f"first invalid row(s): {preview}"
        )
    months = timestamps.dt.strftime("%Y-%m").to_numpy(dtype=str)
    # Datetime64 carries no timezone; values remain normalized to UTC.
    timestamps_utc = timestamps.dt.tz_localize(None).to_numpy(dtype="datetime64[s]")
    return months, timestamps_utc


def _normalize_labels(values):
    labels_numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if labels_numeric.isna().any():
        raise ValueError(f"{LABEL_COLUMN} contains a missing or non-numeric label")
    labels = labels_numeric.to_numpy(dtype=np.int8)
    unique = np.unique(labels)
    if not np.array_equal(unique, np.array([0, 1], dtype=np.int8)):
        raise ValueError(
            f"{LABEL_COLUMN} must contain exactly binary labels 0 and 1, got {unique}"
        )
    return labels


def _write_month(path, features, labels, indices, column_names, sha256,
                 timestamps, chunk_size):
    sample_count = indices.size
    feature_count = features.shape[1]
    combined = np.empty((sample_count, feature_count + 1), dtype=np.int8)
    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        source_indices = indices[start:end]
        combined[start:end, :-1] = features[source_indices]
    combined[:, -1] = labels[indices]

    temporary = path.with_suffix(".tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            data=combined,
            column_names=column_names,
            sha256=np.asarray(sha256[indices], dtype=str),
            last_analysis_time=timestamps[indices],
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return combined[:, -1].copy()


def prepare_mh1m(source_path, output_dir, force=False, chunk_size=4096):
    """Create one NPZ per observed analysis month and return its manifest."""
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    existing = sorted(output_dir.glob("????-??_selected.npz"))
    if existing and not force:
        raise FileExistsError(
            f"Monthly files already exist (for example {existing[0]}). "
            "Use --force to replace them."
        )

    print(f"Reading metadata from {source_path}", flush=True)
    with np.load(source_path, allow_pickle=True) as source:
        metadata_columns_key = _metadata_key(
            source, "metadata_columns", "metadata_feature_names"
        )
        column_names_key = _metadata_key(source, "column_names", "feature_names")
        metadata_columns = [str(value) for value in source[metadata_columns_key]]
        try:
            date_index = metadata_columns.index(DATE_COLUMN)
            label_index = metadata_columns.index(LABEL_COLUMN)
        except ValueError as exc:
            raise ValueError(
                f"Source metadata must contain {DATE_COLUMN} and {LABEL_COLUMN}"
            ) from exc

        metadata = source["metadata"]
        months, timestamps = _normalize_timestamps(metadata[:, date_index])
        labels = _normalize_labels(metadata[:, label_index])
        del metadata

        feature_names = np.asarray(source[column_names_key], dtype=str)
        output_column_names = np.concatenate(
            (feature_names, np.asarray([LABEL_COLUMN], dtype=str))
        )
        sha256 = source["sha256"]
        print("Decompressing the full feature matrix once", flush=True)
        features = source["data"]

    if features.ndim != 2:
        raise ValueError(f"Source data must be two-dimensional, got {features.shape}")
    if features.dtype != np.int8:
        raise ValueError(f"Source data must use int8 features, got {features.dtype}")
    sample_count = features.shape[0]
    if not (labels.size == months.size == timestamps.size == sha256.size == sample_count):
        raise ValueError("Features, labels, timestamps, and SHA256 arrays are misaligned")
    if feature_names.size != features.shape[1]:
        raise ValueError("Feature names do not align with source feature columns")

    order = np.argsort(months, kind="stable")
    sorted_months = months[order]
    observed_months, starts, counts = np.unique(
        sorted_months, return_index=True, return_counts=True
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    split_summaries = {}
    for month, start, count in zip(observed_months, starts, counts):
        indices = order[start:start + count]
        output_path = output_dir / f"{month}_selected.npz"
        print(f"Writing {output_path.name}: {count} samples", flush=True)
        written_labels = _write_month(
            output_path,
            features,
            labels,
            indices,
            output_column_names,
            sha256,
            timestamps,
            chunk_size,
        )
        malware = int(written_labels.sum())
        split_summaries[str(month)] = {
            "file": output_path.name,
            "samples": int(count),
            "features_including_label": int(features.shape[1] + 1),
            "benign": int(count - malware),
            "malware": malware,
        }
        del written_labels

    full_range = pd.period_range(observed_months[0], observed_months[-1], freq="M")
    missing_months = sorted(
        set(map(str, full_range)) - set(map(str, observed_months))
    )
    manifest = {
        "source": str(source_path),
        "timestamp_field": DATE_COLUMN,
        "timestamp_semantics": "VirusTotal last analysis time, normalized to UTC",
        "label_field": LABEL_COLUMN,
        "label_position": "last column of the data array",
        "samples": int(sample_count),
        "source_features": int(features.shape[1]),
        "output_columns": int(features.shape[1] + 1),
        "benign": int((labels == 0).sum()),
        "malware": int((labels == 1).sum()),
        "first_month": str(observed_months[0]),
        "last_month": str(observed_months[-1]),
        "observed_month_count": int(observed_months.size),
        "missing_months": missing_months,
        "splits": split_summaries,
    }
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".tmp.json")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=f"data/mh1m/data/processed/{DEFAULT_SOURCE_NAME}",
    )
    parser.add_argument("--output-dir", default="data/mh1m_monthly")
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = prepare_mh1m(
        args.source,
        args.output_dir,
        force=args.force,
        chunk_size=args.chunk_size,
    )
    print(
        f"Created {manifest['observed_month_count']} monthly files with "
        f"{manifest['samples']} samples in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
