import unittest

import numpy as np

from common import get_model_stats


class CommonMetricsTest(unittest.TestCase):
    def test_no_positive_predictions_produce_zero_precision_and_f1(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])

        tpr, tnr, fpr, fnr, accuracy, precision, f1 = get_model_stats(
            y_true, y_pred
        )

        self.assertEqual(tpr, 0.0)
        self.assertEqual(tnr, 1.0)
        self.assertEqual(fpr, 0.0)
        self.assertEqual(fnr, 1.0)
        self.assertEqual(accuracy, 0.5)
        self.assertEqual(precision, 0.0)
        self.assertEqual(f1, 0.0)

    def test_single_class_input_still_returns_binary_metrics(self):
        metrics = get_model_stats(np.zeros(3), np.zeros(3))

        self.assertTrue(np.isfinite(metrics).all())
        self.assertEqual(metrics[1], 1.0)


if __name__ == "__main__":
    unittest.main()
