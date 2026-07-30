"""Compatibility imports for the former IDS method name.

New code should import the PDSE implementation from :mod:`pdse`.
"""

from pdse import CADEPDSE
from pdse import HCLPDSE
from pdse import add_weight_diff
from pdse import average_weight_diff
from pdse import build_exposure_indices
from pdse import build_exposure_loader
from pdse import normalized_weight_diff


HCLIDS = HCLPDSE
CADEIDS = CADEPDSE

__all__ = [
    'CADEIDS',
    'CADEPDSE',
    'HCLIDS',
    'HCLPDSE',
    'add_weight_diff',
    'average_weight_diff',
    'build_exposure_indices',
    'build_exposure_loader',
    'normalized_weight_diff',
]
