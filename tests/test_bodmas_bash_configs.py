import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DatasetBashConfigTest(unittest.TestCase):
    def _read_script(self, name):
        return (ROOT / "experiments" / "scripts" / name).read_text(encoding="utf-8")

    def _assert_switchable_dataset_config(self, script):
        self.assertIn("DATA=${1:-${DATA:-bodmas}}", script)
        self.assertIn("gen_apigraph_drebin)", script)
        self.assertIn("DATA_TAG=gen_apigraph", script)
        self.assertIn("TRAIN_START=2012-01", script)
        self.assertIn("TRAIN_END=2012-12", script)
        self.assertIn("TEST_START=2013-01", script)
        self.assertIn("TEST_END=2018-12", script)
        self.assertIn("bodmas)", script)
        self.assertIn("DATA_TAG=bodmas", script)
        self.assertIn("TRAIN_START=2007-01", script)
        self.assertIn("TRAIN_END=2019-09", script)
        self.assertIn("TEST_START=2019-10", script)
        self.assertIn("TEST_END=2020-09", script)
        self.assertIn('echo "Unsupported dataset: ${DATA}"', script)

    def test_hcl_script_switches_dataset_protocol_and_paths(self):
        script = self._read_script("base_hcl.sh")

        self._assert_switchable_dataset_config(script)
        self.assertIn("MODEL_DIR=models/${RUN_TAG}/${DATA}/${CNT}", script)
        self.assertIn("RUN_NAME=${DATA_TAG}_${RUN_TAG}_cnt${CNT}_${SEQ}_seed${SEED}", script)

    def test_hcl_pdse_script_switches_dataset_protocol_and_paths(self):
        script = self._read_script("base_hcl_pdse.sh")

        self._assert_switchable_dataset_config(script)
        self.assertIn("MODEL_DIR=models/hcl_pdse_combined/${DATA}/${CNT}", script)
        self.assertIn(
            "RUN_NAME=${DATA_TAG}_hcl_pdse_combined_cnt${CNT}_${SEQ}_seed${SEED}",
            script,
        )


if __name__ == "__main__":
    unittest.main()
