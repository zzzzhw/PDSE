"""Prepare the raw BODMAS release for the repository's monthly loader."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


BENIGN_FAMILY = "benign"


def normalize_families(series):
    """Normalize BODMAS family names while making benign rows explicit."""
    normalized = series.astype("string").str.strip().str.lower()
    normalized = normalized.where(normalized.notna() & normalized.ne(""), BENIGN_FAMILY)
    return normalized.to_numpy(dtype=str)


def build_family_mapping(families):
    malicious_families = sorted(set(families) - {BENIGN_FAMILY})
    return {family: index for index, family in enumerate(malicious_families, start=1)}


def previous_month(month):
    return str(pd.Period(month, freq="M") - 1)


def validate_source(X, y_binary, metadata, families, chunk_size=4096):
    if X.ndim != 2:
        raise ValueError(f"BODMAS X must be two-dimensional, got {X.shape}")
    if y_binary.shape != (X.shape[0],) or len(metadata) != X.shape[0]:
        raise ValueError("BODMAS features, labels, and metadata must have equal row counts")
    if set(np.unique(y_binary)) != {0, 1}:
        raise ValueError("BODMAS y must contain exactly the binary labels 0 and 1")

    family_is_benign = families == BENIGN_FAMILY
    if not np.array_equal(y_binary == 0, family_is_benign):
        raise ValueError("BODMAS binary labels do not agree with missing family metadata")

    for start in range(0, X.shape[0], chunk_size):
        if not np.isfinite(X[start:start + chunk_size]).all():
            raise ValueError(f"BODMAS X contains a non-finite value near row {start}")


def prepare_split(X, y_binary, families, indices, family_mapping, scaler):
    """Return one split in the malware-first order expected by data.py."""
    indices = np.asarray(indices, dtype=np.int64)
    malicious = indices[y_binary[indices] == 1]
    benign = indices[y_binary[indices] == 0]
    ordered = np.concatenate((malicious, benign))

    X_split = scaler.transform(X[ordered]).astype(np.float32, copy=False)
    y_family = np.zeros(ordered.shape[0], dtype=np.int32)
    y_family[:malicious.shape[0]] = np.fromiter(
        (family_mapping[families[index]] for index in malicious),
        dtype=np.int32,
        count=malicious.shape[0],
    )
    malicious_family_names = families[malicious]
    return X_split, y_family, malicious_family_names, ordered


def write_npz(path, X, y, y_mal_family):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        X_train=X,
        y_train=y,
        y_mal_family=y_mal_family,
    )
    os.replace(temporary, path)


def sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_bodmas(source_dir, output_dir, test_start="2019-10", test_end="2020-09",
                   force=False):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    source_npz = source_dir / "bodmas.npz"
    metadata_path = source_dir / "bodmas_metadata.csv"
    if not source_npz.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("bodmas.npz and bodmas_metadata.csv are required")

    metadata = pd.read_csv(metadata_path)
    timestamps = pd.to_datetime(
        metadata["timestamp"], format="mixed", errors="coerce", utc=True
    )
    if timestamps.isna().any():
        raise ValueError("BODMAS metadata contains invalid timestamps")
    months = timestamps.dt.strftime("%Y-%m").to_numpy(dtype=str)
    families = normalize_families(metadata["family"])

    with np.load(source_npz, allow_pickle=False) as source:
        X = source["X"]
        y_binary = source["y"]
    validate_source(X, y_binary, metadata, families)

    test_months = [str(month) for month in pd.period_range(test_start, test_end, freq="M")]
    train_indices = np.flatnonzero(months < test_start)
    test_indices_by_month = {
        month: np.flatnonzero(months == month) for month in test_months
    }
    covered_count = train_indices.shape[0] + sum(
        indices.shape[0] for indices in test_indices_by_month.values()
    )
    if covered_count != X.shape[0]:
        raise ValueError(
            "The configured train/test window does not cover every BODMAS sample: "
            f"covered={covered_count}, total={X.shape[0]}"
        )

    family_mapping = build_family_mapping(families)
    scaler = MinMaxScaler()
    scaler.fit(X[train_indices])

    train_start = min(months.tolist())
    train_end = previous_month(test_start)
    train_name = f"{train_start}to{train_end}_selected.npz"
    output_paths = [output_dir / train_name]
    output_paths.extend(output_dir / f"{month}_selected.npz" for month in test_months)
    existing = [path for path in output_paths if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"Prepared files already exist (for example {existing[0]}). Use --force to replace them."
        )

    X_train, y_train, train_mal_families, ordered_train = prepare_split(
        X, y_binary, families, train_indices, family_mapping, scaler
    )
    write_npz(output_dir / train_name, X_train, y_train, train_mal_families)

    split_summaries = {
        "train": {
            "file": train_name,
            "start": train_start,
            "end": train_end,
            "samples": int(ordered_train.shape[0]),
            "benign": int((y_binary[ordered_train] == 0).sum()),
            "malware": int((y_binary[ordered_train] == 1).sum()),
            "families": int(np.unique(train_mal_families).shape[0]),
        }
    }
    del X_train

    for month, indices in test_indices_by_month.items():
        X_month, y_month, month_mal_families, ordered_month = prepare_split(
            X, y_binary, families, indices, family_mapping, scaler
        )
        filename = f"{month}_selected.npz"
        write_npz(output_dir / filename, X_month, y_month, month_mal_families)
        split_summaries[month] = {
            "file": filename,
            "samples": int(ordered_month.shape[0]),
            "benign": int((y_binary[ordered_month] == 0).sum()),
            "malware": int((y_binary[ordered_month] == 1).sum()),
            "families": int(np.unique(month_mal_families).shape[0]),
        }
        del X_month

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = output_dir / "family_mapping.json"
    mapping_path.write_text(
        json.dumps(family_mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "source": {
            "npz": str(source_npz),
            "npz_sha256": sha256(source_npz),
            "metadata": str(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "samples": int(X.shape[0]),
            "features": int(X.shape[1]),
            "benign": int((y_binary == 0).sum()),
            "malware": int((y_binary == 1).sum()),
            "normalized_malware_families": len(family_mapping),
        },
        "protocol": {
            "train_rule": f"timestamp < {test_start}",
            "test_start": test_start,
            "test_end": test_end,
            "scaling": "MinMaxScaler fitted on training rows only",
            "family_normalization": "strip and lowercase; missing family is benign",
        },
        "splits": split_summaries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/bodmas")
    parser.add_argument("--output-dir", default="data/bodmas_monthly")
    parser.add_argument("--test-start", default="2019-10")
    parser.add_argument("--test-end", default="2020-09")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = prepare_bodmas(
        args.source_dir,
        args.output_dir,
        test_start=args.test_start,
        test_end=args.test_end,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
