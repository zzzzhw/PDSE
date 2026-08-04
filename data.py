"""
data.py
~~~~~~~

Functions for loading data.

"""

import numpy as np
import os
import zipfile


DATASET_DIRECTORIES = {
    'bodmas': 'bodmas_monthly',
}


def resolve_dataset_directory(data_name):
    return DATASET_DIRECTORIES.get(data_name, data_name)


def is_empty_month_dataset(data_name, month, folder='data/'):
    """Check the row count from an NPZ header without loading its feature matrix."""
    data_directory = {
        'bodmas': 'bodmas_monthly',
        'mh1m': 'mh1m_monthly',
    }.get(data_name, data_name)
    saved_data_file = os.path.join(
        folder, data_directory, f'{month}_selected.npz'
    )
    with zipfile.ZipFile(saved_data_file) as archive:
        names = set(archive.namelist())
        array_name = next(
            (name for name in ('X_train.npy', 'data.npy') if name in names),
            None,
        )
        if array_name is None:
            raise KeyError(
                f'{saved_data_file} contains neither X_train nor data arrays'
            )
        with archive.open(array_name) as stream:
            version = np.lib.format.read_magic(stream)
            shape, _, _ = np.lib.format._read_array_header(stream, version)
    return shape[0] == 0


def load_range_dataset_w_benign(args, data_name, start_month, end_month, folder='data/'):
    if start_month != end_month:
        dataset_name = f'{start_month}to{end_month}'
    else:
        dataset_name = f'{start_month}'
    data_directory = resolve_dataset_directory(data_name)
    saved_data_file = os.path.join(folder, data_directory, f'{dataset_name}_selected.npz')
    if data_name == 'bodmas' and not os.path.exists(saved_data_file):
        raise FileNotFoundError(
            f'{saved_data_file} does not exist. Prepare BODMAS with: '
            'python experiments/prepare_bodmas.py'
        )
    data = np.load(saved_data_file, allow_pickle=True)
    X_train, y_train = data['X_train'], data['y_train']
    y_mal_family = data['y_mal_family']
    if args.classifier == 'res':
        X_train = X_train[:,0:1156]
    return X_train, y_train, y_mal_family
