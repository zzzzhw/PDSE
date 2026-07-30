import unittest

import numpy as np

from pdse import build_exposure_indices


class PDSETripletConstructionTest(unittest.TestCase):
    def test_uses_nearest_valid_history_for_positive_and_negative(self):
        y_family = np.array([1, 1, 0, 0, 1])
        y_binary = np.array([1, 1, 0, 0, 1])
        timestamps = np.array([
            '2020-01',
            '2020-03',
            '2020-02',
            '2020-04',
            '2020-05',
        ])

        indices = build_exposure_indices(
            y_family,
            y_binary,
            exposure_count=1,
            timestamps=timestamps,
            seed=3,
        ).reshape(-1, 3)

        np.testing.assert_array_equal(indices, [[4, 1, 3]])

    def test_builds_one_triplet_for_every_unbalanced_current_sample(self):
        y_family = np.array([0, 1, 2, 0, 0, 1])
        y_binary = np.array([0, 1, 1, 0, 0, 1])
        timestamps = np.array([
            '2020-01',
            '2020-01',
            '2020-02',
            '2020-03',
            '2020-03',
            '2020-03',
        ])

        triplets = build_exposure_indices(
            y_family,
            y_binary,
            exposure_count=3,
            timestamps=timestamps,
            seed=5,
        ).reshape(-1, 3)

        self.assertEqual(triplets.shape, (3, 3))
        self.assertEqual(set(triplets[:, 0]), {3, 4, 5})

    def test_random_tie_breaking_never_uses_a_more_distant_candidate(self):
        y_family = np.array([1, 1, 1, 0, 1])
        y_binary = np.array([1, 1, 1, 0, 1])
        timestamps = np.array([
            '2020-01',
            '2020-04',
            '2020-06',
            '2020-05',
            '2020-05',
        ])

        for seed in range(10):
            triplet = build_exposure_indices(
                y_family,
                y_binary,
                exposure_count=1,
                timestamps=timestamps,
                seed=seed,
            ).reshape(-1, 3)[0]
            self.assertIn(triplet[1], {1, 2})
            self.assertEqual(triplet[2], 3)

    def test_rejects_misaligned_timestamps(self):
        with self.assertRaisesRegex(ValueError, 'one value per training sample'):
            build_exposure_indices(
                np.array([0, 1, 1]),
                np.array([0, 1, 1]),
                exposure_count=1,
                timestamps=np.array(['2020-01', '2020-02']),
            )


if __name__ == '__main__':
    unittest.main()
