import unittest

from data import resolve_dataset_directory


class DatasetAliasTest(unittest.TestCase):
    def test_bodmas_resolves_to_prepared_monthly_directory(self):
        self.assertEqual(resolve_dataset_directory("bodmas"), "bodmas_monthly")

    def test_other_dataset_names_are_unchanged(self):
        self.assertEqual(resolve_dataset_directory("gen_apigraph_drebin"), "gen_apigraph_drebin")


if __name__ == "__main__":
    unittest.main()
