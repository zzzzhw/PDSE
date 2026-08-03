"""Prepare the public KronoDroid real-device data for temporal experiments."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "c6ec342167bc449967a802824d068900ac8120c5"
SOURCE_REPOSITORY = "https://github.com/aleguma/kronodroid"
SOURCE_PAPER = "https://doi.org/10.1016/j.cose.2021.102399"
SOURCE_FILES = {
    "legitimate": {
        "archive": "real_legitimate_v1.zip",
        "member": "real_legitimate_v1.csv",
        "sha256": "21f6d507321856eefa6c40a31f484d3e532b5dea23144e3c2a9470161bd5f782",
    },
    "malware": {
        "archive": "real_malware_v1.zip",
        "member": "real_malware_v1.csv",
        "sha256": "03aa36a9c3aa3430523fcf87042659562e5af3219688562087107817095cb8cd",
    },
}
TRAIN_START = "2008-01"
TRAIN_END = "2010-12"
TEST_START = "2011-01"
TEST_END = "2014-12"
TIMESTAMP_COLUMN = "HighestModDate"
INTRINSIC_METADATA = [
    "CFileSize",
    "UFileSize",
    "FilesInsideAPK",
    "Activities",
    "NrIntServices",
    "NrIntServicesActions",
    "NrIntActivities",
    "NrIntActivitiesActions",
    "NrIntReceivers",
    "NrIntReceiversActions",
    "TotalIntentFilters",
    "NrServices",
]


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_url(archive):
    return (
        "https://raw.githubusercontent.com/aleguma/kronodroid/"
        f"{SOURCE_COMMIT}/real_device/{archive}"
    )


def acquire_sources(raw_dir):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = {}
    for label, spec in SOURCE_FILES.items():
        archive = raw_dir / spec["archive"]
        if not archive.exists():
            print(f"Downloading {archive.name}", flush=True)
            request = urllib.request.Request(
                source_url(spec["archive"]), headers={"User-Agent": "CDA-OE"}
            )
            temporary = archive.with_suffix(".tmp.zip")
            with urllib.request.urlopen(request, timeout=120) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
            os.replace(temporary, archive)
        actual_hash = file_sha256(archive)
        if actual_hash != spec["sha256"]:
            raise ValueError(
                f"SHA256 mismatch for {archive}: {actual_hash} != {spec['sha256']}"
            )

        csv_path = raw_dir / spec["member"]
        if not csv_path.exists():
            with zipfile.ZipFile(archive) as zipped:
                with zipped.open(spec["member"]) as source:
                    with csv_path.open("wb") as output:
                        shutil.copyfileobj(source, output)
        csv_paths[label] = csv_path
    return csv_paths


def load_and_clean(legitimate_path, malware_path):
    legitimate = pd.read_csv(legitimate_path, low_memory=False)
    malware = pd.read_csv(malware_path, low_memory=False)
    if legitimate.shape != (36_755, 484) or malware.shape != (41_382, 484):
        raise ValueError(
            "Unexpected KronoDroid real-device shapes: "
            f"legitimate={legitimate.shape}, malware={malware.shape}"
        )
    if list(legitimate.columns) != list(malware.columns):
        raise ValueError("KronoDroid real-device CSV schemas do not match")

    frame = pd.concat([legitimate, malware], ignore_index=True)
    if set(frame["Malware"].unique()) != {0, 1}:
        raise ValueError("KronoDroid Malware labels must be binary")
    conflicts = sorted(
        frame.groupby("sha256")["Malware"].nunique().loc[lambda x: x > 1].index
    )
    frame = frame.loc[~frame["sha256"].isin(conflicts)].copy()
    frame["sample_date"] = pd.to_datetime(
        frame[TIMESTAMP_COLUMN], format="%m/%d/%Y", errors="coerce"
    )
    frame["month"] = frame["sample_date"].dt.to_period("M").astype(str)
    return frame.reset_index(drop=True), conflicts


def feature_columns(frame):
    columns = list(frame.columns)
    if columns[2] != "execve" or columns[290] != "nr_syscalls":
        raise ValueError("Unexpected KronoDroid dynamic feature schema")
    if columns[291] != "ACCEPT_HANDOVER" or columns[463] != "total_perm":
        raise ValueError("Unexpected KronoDroid static feature schema")
    selected = columns[2:464] + INTRINSIC_METADATA
    leakage = {
        "Package",
        "Malware",
        "sha256",
        "EarliestModDate",
        "HighestModDate",
        "TimesSubmitted",
        "NrContactedIps",
        "Scanners",
        "Detection_Ratio",
        "MalFamily",
    }
    overlap = leakage.intersection(selected)
    if overlap:
        raise ValueError(f"Label-leaking features selected: {sorted(overlap)}")
    return selected


def fit_transform_features(frame, selected_columns, train_mask):
    numeric = frame[selected_columns].apply(pd.to_numeric, errors="coerce")
    missing_cells = int(numeric.isna().sum().sum())
    negative_cells = int((numeric < 0).sum().sum())
    numeric = numeric.fillna(0).clip(lower=0)
    logged = np.log1p(numeric.to_numpy(dtype=np.float64))

    train = logged[train_mask]
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    retained = scale > 1e-8
    if not retained.any():
        raise ValueError("No non-constant KronoDroid training features remain")
    transformed = (logged[:, retained] - mean[retained]) / scale[retained]
    if not np.isfinite(transformed).all():
        raise ValueError("KronoDroid preprocessing produced non-finite values")
    retained_names = [
        name for name, keep in zip(selected_columns, retained) if keep
    ]
    return (
        transformed.astype(np.float32),
        retained_names,
        mean[retained],
        scale[retained],
        missing_cells,
        negative_cells,
    )


def build_family_mapping(frame):
    families = (
        frame.loc[frame["Malware"].eq(1), "MalFamily"]
        .fillna("unknown")
        .astype(str)
    )
    return {
        family: index
        for index, family in enumerate(sorted(families.unique()), start=1)
    }


def ordered_split_arrays(frame, features, indices, family_to_index):
    indices = np.asarray(indices, dtype=np.int64)
    subset = frame.iloc[indices]
    malware_local = np.flatnonzero(subset["Malware"].to_numpy() == 1)
    benign_local = np.flatnonzero(subset["Malware"].to_numpy() == 0)
    order = np.concatenate((malware_local, benign_local))
    ordered_frame = subset.iloc[order]
    ordered_features = features[indices][order]

    malware_families = (
        ordered_frame.loc[ordered_frame["Malware"].eq(1), "MalFamily"]
        .fillna("unknown")
        .astype(str)
        .to_numpy(dtype=str)
    )
    malware_labels = np.fromiter(
        (family_to_index[family] for family in malware_families),
        dtype=np.int64,
        count=malware_families.size,
    )
    labels = np.concatenate(
        (malware_labels, np.zeros(benign_local.size, dtype=np.int64))
    )
    return ordered_features, labels, malware_families, ordered_frame


def save_split(path, frame, features, indices, family_to_index):
    X, y, malware_families, ordered_frame = ordered_split_arrays(
        frame, features, indices, family_to_index
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        X_train=X,
        y_train=y,
        y_mal_family=malware_families,
    )
    os.replace(temporary, path)
    malware_count = int((y != 0).sum())
    return {
        "file": path.name,
        "samples": int(y.size),
        "benign": int(y.size - malware_count),
        "malware": malware_count,
        "families": int(np.unique(malware_families).size),
        "first_sha256": str(ordered_frame["sha256"].iloc[0]) if y.size else None,
    }


def write_monthly_counts(path, splits):
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["month", "total", "benign", "malware", "families"])
        for month, summary in sorted(splits.items()):
            writer.writerow(
                [
                    month,
                    summary["samples"],
                    summary["benign"],
                    summary["malware"],
                    summary["families"],
                ]
            )
    os.replace(temporary, path)


def prepare_kronodroid(raw_dir, output_dir, force=False):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    expected = [output_dir / f"{TRAIN_START}to{TRAIN_END}_selected.npz"]
    expected.extend(
        output_dir / f"{month}_selected.npz"
        for month in map(
            str, pd.period_range(TEST_START, TEST_END, freq="M")
        )
    )
    existing = [path for path in expected if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"Prepared file already exists: {existing[0]}; use --force to replace"
        )

    sources = acquire_sources(raw_dir)
    frame, conflicts = load_and_clean(sources["legitimate"], sources["malware"])
    train_mask = frame["month"].between(TRAIN_START, TRAIN_END).to_numpy()
    test_mask = frame["month"].between(TEST_START, TEST_END).to_numpy()
    selected = feature_columns(frame)
    features, retained, mean, scale, missing_cells, negative_cells = (
        fit_transform_features(frame, selected, train_mask)
    )
    family_to_index = build_family_mapping(frame.loc[train_mask | test_mask])

    output_dir.mkdir(parents=True, exist_ok=True)
    train_name = f"{TRAIN_START}to{TRAIN_END}"
    train_indices = np.flatnonzero(train_mask)
    train_summary = save_split(
        output_dir / f"{train_name}_selected.npz",
        frame,
        features,
        train_indices,
        family_to_index,
    )

    monthly_summaries = {}
    for month in map(str, pd.period_range(TEST_START, TEST_END, freq="M")):
        indices = np.flatnonzero(frame["month"].eq(month).to_numpy())
        monthly_summaries[month] = save_split(
            output_dir / f"{month}_selected.npz",
            frame,
            features,
            indices,
            family_to_index,
        )

    manifest = {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_paper": SOURCE_PAPER,
        "source_files": SOURCE_FILES,
        "timestamp": TIMESTAMP_COLUMN,
        "train_window": [TRAIN_START, TRAIN_END],
        "test_window": [TEST_START, TEST_END],
        "conflicting_sha256_removed": conflicts,
        "raw_feature_count": len(selected),
        "retained_feature_count": len(retained),
        "retained_features": retained,
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "missing_feature_cells_filled_with_zero": missing_cells,
        "negative_feature_cells_clipped_to_zero": negative_cells,
        "family_to_index": family_to_index,
        "train": train_summary,
        "test_total": {
            "samples": int(sum(item["samples"] for item in monthly_summaries.values())),
            "benign": int(sum(item["benign"] for item in monthly_summaries.values())),
            "malware": int(sum(item["malware"] for item in monthly_summaries.values())),
        },
        "test_months": monthly_summaries,
    }
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".tmp.json")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    write_monthly_counts(output_dir / "monthly_counts.csv", monthly_summaries)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "kronodroid_raw" / "real_device",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "kronodroid_real_2008_2014",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = prepare_kronodroid(args.raw_dir, args.output_dir, force=args.force)
    print(json.dumps({"train": manifest["train"], "test": manifest["test_total"]}, indent=2))


if __name__ == "__main__":
    main()
