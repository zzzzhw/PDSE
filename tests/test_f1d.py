import tempfile
import unittest
from pathlib import Path

from utils import append_f1d_results
from utils import delayed_month


class F1DTest(unittest.TestCase):
    def test_delayed_month_is_relative_to_test_start(self):
        self.assertIsNone(delayed_month("2020-03", "2019-10", 6))
        self.assertEqual(delayed_month("2020-04", "2019-10", 6), "2019-10")
        self.assertEqual(delayed_month("2020-09", "2019-10", 6), "2020-03")

    def test_results_are_attached_to_the_evaluation_month(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.csv"
            result.write_text(
                "date\tF1\tF1-D\n"
                "2019-09\t0.9\n"
                "2019-10\t0.8\n"
                "2020-04\t0.7\n",
                encoding="utf-8",
            )

            append_f1d_results(result, {"2020-04": 0.65})

            self.assertEqual(
                result.read_text(encoding="utf-8").splitlines(),
                [
                    "date\tF1\tF1-D",
                    "2019-09\t0.9\tnan",
                    "2019-10\t0.8\tnan",
                    "2020-04\t0.7\t0.6500",
                ],
            )


if __name__ == "__main__":
    unittest.main()
