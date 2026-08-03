import numpy as np
import pandas as pd

from experiments.prepare_kronodroid import fit_transform_features
from experiments.prepare_kronodroid import ordered_split_arrays


def test_preprocessing_fits_only_training_rows():
    frame = pd.DataFrame(
        {
            "a": [0, 1, 2, 100],
            "constant": [1, 1, 1, 10],
        }
    )
    train_mask = np.array([True, True, True, False])

    transformed, retained, _, _, missing, negative = fit_transform_features(
        frame, ["a", "constant"], train_mask
    )

    assert retained == ["a"]
    assert transformed.shape == (4, 1)
    assert abs(float(transformed[:3].mean())) < 1e-6
    assert missing == 0
    assert negative == 0


def test_split_places_malware_before_benign():
    frame = pd.DataFrame(
        {
            "Malware": [0, 1, 0, 1],
            "MalFamily": [None, "alpha", None, "beta"],
            "sha256": ["a", "b", "c", "d"],
        }
    )
    features = np.arange(8, dtype=np.float32).reshape(4, 2)

    X, y, families, ordered = ordered_split_arrays(
        frame, features, np.arange(4), {"alpha": 1, "beta": 2}
    )

    np.testing.assert_array_equal(y, [1, 2, 0, 0])
    np.testing.assert_array_equal(families, ["alpha", "beta"])
    assert families.dtype.kind == "U"
    np.testing.assert_array_equal(X, features[[1, 3, 0, 2]])
    assert ordered["sha256"].tolist() == ["b", "d", "a", "c"]
