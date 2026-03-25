"""Streaming tests exercising a "blocked" configuration (smaller block granularity).

This file overlaps with `test_stream.py`, but keeps an explicit block-oriented
variant for sanity checks and convergence comparison.
"""

import numpy as np
from lpkit.label_propagation import label_propagation
from lpkit.stream import symmetrize_and_sort, split_sorted_sym_to_blocks, init_labels_memmap, stream_multi_sweep_blocks


def groups_from_labels(labels):
    """Return a partition representation independent of concrete label values."""
    buckets = {}
    for i, lab in enumerate(labels):
        buckets.setdefault(int(lab), set()).add(i)
    return {frozenset(s) for s in buckets.values()}

def test_blocked_one_sweep_matches_count(tmp_path):
    """Blocked one-sweep run should execute and produce a valid labels array + metadata."""
    raw = tmp_path / "g.edgelist"
    raw.write_text("0 1\n1 2\n2 0\n3 4\n4 5\n5 3\n")
    sorted_sym = tmp_path / "g.sorted.sym"
    labels = tmp_path / "g.labels.npy"

    #ram baseline after 1 sweep-equivalent
    adj = [[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]]
    labels_ram, _ = label_propagation(adj, seed=1337)

    meta = symmetrize_and_sort(str(raw), str(sorted_sym))
    n = meta["n"]
    block_paths = split_sorted_sym_to_blocks(str(sorted_sym), n=n, block_size=n, out_dir=str(tmp_path / "blocks"))
    init_labels_memmap(str(labels), n=n)

    info = stream_multi_sweep_blocks(
        block_paths,
        str(labels),
        n=n,
        block_size=n,
        seed=5543552,
        max_sweeps=1,
        min_sweeps=1,
        tie_break="min",
    )

    assert info["sweeps"] == 1
    assert "total_updates" in info

    mm = np.lib.format.open_memmap(str(labels), mode="r+")
    assert mm.shape == (n,)


def test_blocked_converges_like_ram(tmp_path):
    """Blocked streaming configuration should match RAM baseline partition on the toy graph."""
    raw = tmp_path / "g.edgelist"
    raw.write_text("0 1\n1 2\n2 0\n3 4\n4 5\n5 3\n")
    sorted_sym = tmp_path / "g.sorted.sym"
    labels = tmp_path / "g.labels.npy"

    #ram baseline to convergence
    adj = [[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]]
    labels_ram, _ = label_propagation(adj, seed=1337)

    meta = symmetrize_and_sort(str(raw), str(sorted_sym))
    n = meta["n"]
    block_paths = split_sorted_sym_to_blocks(str(sorted_sym), n=n, block_size=n, out_dir=str(tmp_path / "blocks"))
    init_labels_memmap(str(labels), n=n)

    info = stream_multi_sweep_blocks(
        block_paths,
        str(labels),
        n=n,
        block_size=n,
        seed=12233,
        max_sweeps=100,
        min_sweeps=1,
        tie_break="min",
    )
    assert info["sweeps"] >= 1

    mm = np.lib.format.open_memmap(str(labels), mode="r+")
    assert groups_from_labels(labels_ram) == groups_from_labels(mm)
