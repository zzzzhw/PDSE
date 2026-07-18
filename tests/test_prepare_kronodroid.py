import unittest

import numpy as np
import pandas as pd

from experiments.prepare_kronodroid import (
    fit_transform_features,
    ordered_split_arrays,
)


class KronoDroidPreparationTest(unittest.TestCase):
    def test_preprocessing_fits_only_training_rows(self):
        frame = pd.DataFrame({
            'a': [0, 1, 2, 100],
            'b': [3, 3, 3, 99],
            'constant': [1, 1, 1, 10],
        })
        train_mask = np.array([True, True, True, False])
        transformed, retained, _, _, negative_cells = fit_transform_features(
            frame, ['a', 'b', 'constant'], train_mask
        )
        self.assertEqual(retained, ['a'])
        self.assertEqual(transformed.shape, (4, 1))
        self.assertAlmostEqual(float(transformed[:3].mean()), 0.0, places=6)
        self.assertEqual(negative_cells, 0)

    def test_split_places_malware_before_benign(self):
        frame = pd.DataFrame({
            'Malware': [0, 1, 0, 1],
            'MalFamily': [None, 'alpha', None, 'beta'],
        })
        features = np.arange(8, dtype=np.float32).reshape(4, 2)
        X, y, families = ordered_split_arrays(
            frame,
            features,
            np.arange(4),
            {'alpha': 1, 'beta': 2},
        )
        np.testing.assert_array_equal(y, np.array([1, 2, 0, 0]))
        np.testing.assert_array_equal(families, np.array(['alpha', 'beta']))
        np.testing.assert_array_equal(X[:2], features[[1, 3]])


if __name__ == '__main__':
    unittest.main()
