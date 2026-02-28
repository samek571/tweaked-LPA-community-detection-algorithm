"""LPKit public package API.

This module intentionally re-exports the stable entry points used by:
- the CLI (`lpkit ...`)
- tests (RAM baseline and streaming pipeline)
- external callers (`from lpkit import stream_lpa`)

Anything not exported == internal == possibly changing over time
"""

from .label_propagation import label_propagation, LPAResult
from .generate_graph import generate_large_graph
from .stream import (
    symmetrize_and_sort,
    init_labels_memmap,
    split_sorted_sym_to_blocks,
    stream_multi_sweep_parallel_blocks,
)
from .api import stream_lpa

__all__ = [
    "label_propagation",
    "LPAResult",
    "generate_large_graph",
    "symmetrize_and_sort",
    "init_labels_memmap",
    "split_sorted_sym_to_blocks",
    "stream_multi_sweep_parallel_blocks",
    "stream_lpa",
]
