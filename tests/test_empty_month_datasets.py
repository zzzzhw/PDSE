import numpy as np

from data import is_empty_month_dataset


def test_detects_empty_hcl_style_month_without_loading_features(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    np.savez_compressed(
        dataset / "2023-01_selected.npz",
        X_train=np.empty((0, 3), dtype=np.int8),
        y_train=np.empty(0, dtype=np.int8),
        y_mal_family=np.empty(0, dtype=str),
    )

    assert is_empty_month_dataset(
        "dataset", "2023-01", folder=str(tmp_path)
    )


def test_detects_empty_mh1m_style_month_and_accepts_populated_month(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    np.savez_compressed(
        dataset / "2023-01_selected.npz",
        data=np.empty((0, 3), dtype=np.int8),
    )
    np.savez_compressed(
        dataset / "2023-02_selected.npz",
        data=np.ones((1, 3), dtype=np.int8),
    )

    assert is_empty_month_dataset(
        "dataset", "2023-01", folder=str(tmp_path)
    )
    assert not is_empty_month_dataset(
        "dataset", "2023-02", folder=str(tmp_path)
    )
