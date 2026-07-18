"""Download and prepare the public KronoDroid real-device temporal dataset."""

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = 'c6ec342167bc449967a802824d068900ac8120c5'
SOURCE_REPOSITORY = 'https://github.com/aleguma/kronodroid'
SOURCE_PAPER = 'https://doi.org/10.1016/j.cose.2021.102399'

SOURCE_FILES = {
    'legitimate': {
        'archive': 'real_legitimate_v1.zip',
        'member': 'real_legitimate_v1.csv',
        'sha256': '21f6d507321856eefa6c40a31f484d3e532b5dea23144e3c2a9470161bd5f782',
    },
    'malware': {
        'archive': 'real_malware_v1.zip',
        'member': 'real_malware_v1.csv',
        'sha256': '03aa36a9c3aa3430523fcf87042659562e5af3219688562087107817095cb8cd',
    },
}

TRAIN_START = '2011-01'
TRAIN_END = '2011-12'
TEST_START = '2012-01'
TEST_END = '2012-12'
INTRINSIC_METADATA = [
    'CFileSize',
    'UFileSize',
    'FilesInsideAPK',
    'Activities',
    'NrIntServices',
    'NrIntServicesActions',
    'NrIntActivities',
    'NrIntActivitiesActions',
    'NrIntReceivers',
    'NrIntReceiversActions',
    'TotalIntentFilters',
    'NrServices',
]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_url(archive):
    return (
        f'https://raw.githubusercontent.com/aleguma/kronodroid/'
        f'{SOURCE_COMMIT}/real_device/{archive}'
    )


def download_verified(url, destination, expected_sha256, attempts=5):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = file_sha256(destination)
        if actual != expected_sha256:
            raise RuntimeError(
                f'Existing archive checksum mismatch: {destination} ({actual})'
            )
        return

    partial = destination.with_suffix(destination.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'CDA-OE'})
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with partial.open('wb') as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            actual = file_sha256(partial)
            if actual != expected_sha256:
                raise RuntimeError(
                    f'Downloaded archive checksum mismatch: {partial} ({actual})'
                )
            partial.replace(destination)
            return
        except Exception as error:
            last_error = error
            partial.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(attempt * 2)
    raise RuntimeError(f'Unable to download {url}') from last_error


def extract_member(archive, member, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        match = next((name for name in names if Path(name).name == member), None)
        if match is None:
            raise RuntimeError(f'{member} is not present in {archive}')
        with zipped.open(match) as source, destination.open('wb') as output:
            shutil.copyfileobj(source, output)


def acquire_sources(raw_dir):
    csv_paths = {}
    for label, spec in SOURCE_FILES.items():
        archive = raw_dir / spec['archive']
        download_verified(source_url(spec['archive']), archive, spec['sha256'])
        csv_path = raw_dir / spec['member']
        extract_member(archive, spec['member'], csv_path)
        csv_paths[label] = csv_path
    return csv_paths


def feature_columns(frame):
    columns = list(frame.columns)
    if columns[2] != 'execve' or columns[463] != 'total_perm':
        raise RuntimeError('Unexpected KronoDroid feature schema')
    selected = columns[2:464] + INTRINSIC_METADATA
    forbidden = {
        'Package', 'Malware', 'sha256', 'EarliestModDate', 'HighestModDate',
        'TimesSubmitted', 'NrContactedIps', 'Scanners', 'Detection_Ratio',
        'MalFamily',
    }
    overlap = forbidden.intersection(selected)
    if overlap:
        raise RuntimeError(f'Label-leaking columns selected: {sorted(overlap)}')
    return selected


def clean_frame(legitimate_path, malware_path):
    legitimate = pd.read_csv(legitimate_path, low_memory=False)
    malware = pd.read_csv(malware_path, low_memory=False)
    if list(legitimate.columns) != list(malware.columns):
        raise RuntimeError('KronoDroid real-device CSV schemas do not match')

    frame = pd.concat([legitimate, malware], ignore_index=True)
    if frame.shape != (78137, 484):
        raise RuntimeError(f'Unexpected KronoDroid shape: {frame.shape}')
    if set(frame['Malware'].unique()) != {0, 1}:
        raise RuntimeError('KronoDroid Malware labels are not binary')

    label_counts = frame.groupby('sha256')['Malware'].nunique()
    conflicting_hashes = set(label_counts[label_counts > 1].index)
    frame = frame[~frame['sha256'].isin(conflicting_hashes)].copy()
    frame['sample_date'] = pd.to_datetime(
        frame['HighestModDate'], errors='coerce'
    )
    frame['month'] = frame['sample_date'].dt.to_period('M').astype(str)
    return frame.reset_index(drop=True), sorted(conflicting_hashes)


def fit_transform_features(frame, selected_columns, train_mask):
    numeric = frame[selected_columns].apply(pd.to_numeric, errors='coerce')
    negative_cells = int((numeric < 0).sum().sum())
    numeric = numeric.fillna(0).clip(lower=0)
    logged = np.log1p(numeric.to_numpy(dtype=np.float64))

    train = logged[train_mask]
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    keep = scale > 1e-8
    if not keep.any():
        raise RuntimeError('No non-constant KronoDroid features remain')
    transformed = (logged[:, keep] - mean[keep]) / scale[keep]
    if not np.isfinite(transformed).all():
        raise RuntimeError('KronoDroid preprocessing produced non-finite values')
    return (
        transformed.astype(np.float32),
        [name for name, retained in zip(selected_columns, keep) if retained],
        mean[keep],
        scale[keep],
        negative_cells,
    )


def family_mapping(frame):
    families = frame.loc[frame['Malware'].eq(1), 'MalFamily'].fillna('unknown')
    return {family: index + 1 for index, family in enumerate(sorted(families.unique()))}


def ordered_split_arrays(frame, features, indices, family_to_index):
    subset = frame.iloc[indices]
    malware_local = np.flatnonzero(subset['Malware'].to_numpy() == 1)
    benign_local = np.flatnonzero(subset['Malware'].to_numpy() == 0)
    order = np.concatenate([malware_local, benign_local])
    ordered_frame = subset.iloc[order]
    ordered_features = features[indices][order]

    malware_families = ordered_frame.loc[
        ordered_frame['Malware'].eq(1), 'MalFamily'
    ].fillna('unknown').astype(str).to_numpy()
    y_malware = np.asarray(
        [family_to_index[family] for family in malware_families],
        dtype=np.int64,
    )
    y_benign = np.zeros(len(benign_local), dtype=np.int64)
    labels = np.concatenate([y_malware, y_benign])
    return ordered_features, labels, malware_families


def save_split(path, frame, features, mask, family_to_index):
    indices = np.flatnonzero(mask)
    X, y, malware_families = ordered_split_arrays(
        frame, features, indices, family_to_index
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X_train=X,
        y_train=y,
        y_mal_family=malware_families,
    )
    return {
        'samples': int(len(y)),
        'benign': int((y == 0).sum()),
        'malware': int((y != 0).sum()),
        'families': int(len(np.unique(malware_families))),
    }


def prepare_dataset(raw_dir, output_dir):
    paths = acquire_sources(raw_dir)
    frame, conflicting_hashes = clean_frame(
        paths['legitimate'], paths['malware']
    )
    selected = feature_columns(frame)
    train_mask = frame['month'].between(TRAIN_START, TRAIN_END).to_numpy()
    test_mask = frame['month'].between(TEST_START, TEST_END).to_numpy()
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError('KronoDroid temporal window is empty')

    features, retained, mean, scale, negative_cells = fit_transform_features(
        frame, selected, train_mask
    )
    family_to_index = family_mapping(frame[train_mask | test_mask])

    splits = {}
    train_name = f'{TRAIN_START}to{TRAIN_END}'
    splits[train_name] = save_split(
        output_dir / f'{train_name}_selected.npz',
        frame,
        features,
        train_mask,
        family_to_index,
    )
    for period in pd.period_range(TEST_START, TEST_END, freq='M'):
        month = str(period)
        month_mask = frame['month'].eq(month).to_numpy()
        stats = save_split(
            output_dir / f'{month}_selected.npz',
            frame,
            features,
            month_mask,
            family_to_index,
        )
        if not stats['benign'] or not stats['malware']:
            raise RuntimeError(f'{month} does not contain both classes')
        splits[month] = stats

    manifest = {
        'source_repository': SOURCE_REPOSITORY,
        'source_commit': SOURCE_COMMIT,
        'source_paper': SOURCE_PAPER,
        'source_archives': {
            key: {'name': spec['archive'], 'sha256': spec['sha256']}
            for key, spec in SOURCE_FILES.items()
        },
        'timestamp': 'HighestModDate',
        'train_window': [TRAIN_START, TRAIN_END],
        'test_window': [TEST_START, TEST_END],
        'conflicting_sha256_removed': conflicting_hashes,
        'excluded_label_leakage': [
            'Scanners', 'Detection_Ratio', 'Malware', 'MalFamily'
        ],
        'excluded_non_intrinsic_metadata': ['TimesSubmitted', 'NrContactedIps'],
        'negative_cells_replaced_with_zero': negative_cells,
        'raw_feature_count': len(selected),
        'retained_feature_count': len(retained),
        'retained_features': retained,
        'scaler_mean': mean.tolist(),
        'scaler_scale': scale.tolist(),
        'family_to_index': family_to_index,
        'splits': splits,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'manifest.json').open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
    print(json.dumps(splits, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw-dir', type=Path, default=REPO_ROOT / 'data' / 'kronodroid_raw'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=REPO_ROOT / 'data' / 'kronodroid_real_hybrid',
    )
    args = parser.parse_args()
    prepare_dataset(args.raw_dir, args.output_dir)


if __name__ == '__main__':
    main()
