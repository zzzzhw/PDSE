import sys
import unittest
from unittest.mock import patch

from ids import CADEIDS
from ids import HCLIDS
from pdse import CADEPDSE
from pdse import HCLPDSE
from utils import parse_args


class PDSECLITest(unittest.TestCase):
    def _parse(self, *arguments):
        with patch.object(sys, 'argv', ['base.py', *arguments]):
            return parse_args()

    def test_pdse_flags_use_pdse_namespace(self):
        args = self._parse(
            '--pdse',
            '--pdse-lambda', '0.02',
            '--pdse-hcl-update-mode', 'separate',
        )

        self.assertTrue(args.pdse)
        self.assertEqual(args.pdse_lambda, 0.02)
        self.assertEqual(args.pdse_hcl_update_mode, 'separate')
        self.assertFalse(hasattr(args, 'ids'))

    def test_legacy_ids_flags_map_to_pdse_namespace(self):
        args = self._parse(
            '--ids',
            '--ids-lambda', '0.03',
            '--ids-cade-update-mode', 'separate',
        )

        self.assertTrue(args.pdse)
        self.assertEqual(args.pdse_lambda, 0.03)
        self.assertEqual(args.pdse_cade_update_mode, 'separate')

    def test_legacy_class_names_alias_pdse_classes(self):
        self.assertIs(HCLIDS, HCLPDSE)
        self.assertIs(CADEIDS, CADEPDSE)


if __name__ == '__main__':
    unittest.main()
