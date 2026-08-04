import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from experiments.prepare_bodmas import BENIGN_FAMILY
from experiments.prepare_bodmas import build_family_mapping
from experiments.prepare_bodmas import normalize_families
from experiments.prepare_bodmas import prepare_split
from experiments.prepare_bodmas import validate_source


class BODMASPreparationTest(unittest.TestCase):
    def test_normalizes_case_and_missing_benign_family(self):
        families = normalize_families(pd.Series([None, "GandCrab", " gandcrab ", "Trojan"]))

        np.testing.assert_array_equal(
            families,
            [BENIGN_FAMILY, "gandcrab", "gandcrab", "trojan"],
        )
        self.assertEqual(build_family_mapping(families), {"gandcrab": 1, "trojan": 2})

    def test_split_is_malware_first_and_uses_family_labels(self):
        X = np.array([[0.0], [10.0], [5.0], [2.0]], dtype=np.float32)
        y_binary = np.array([0, 1, 1, 0], dtype=np.int32)
        families = np.array([BENIGN_FAMILY, "b", "a", BENIGN_FAMILY])
        mapping = {"a": 1, "b": 2}
        scaler = MinMaxScaler().fit(X)

        X_split, y_family, malicious_families, ordered = prepare_split(
            X, y_binary, families, np.arange(4), mapping, scaler
        )

        np.testing.assert_array_equal(ordered, [1, 2, 0, 3])
        np.testing.assert_array_equal(y_family, [2, 1, 0, 0])
        np.testing.assert_array_equal(malicious_families, ["b", "a"])
        np.testing.assert_allclose(X_split[:, 0], [1.0, 0.5, 0.0, 0.2])

    def test_rejects_binary_family_mismatch(self):
        X = np.ones((2, 2), dtype=np.float32)
        y_binary = np.array([0, 1], dtype=np.int32)
        metadata = pd.DataFrame({"family": [None, None]})
        families = np.array([BENIGN_FAMILY, BENIGN_FAMILY])

        with self.assertRaisesRegex(ValueError, "do not agree"):
            validate_source(X, y_binary, metadata, families)


if __name__ == "__main__":
    unittest.main()
