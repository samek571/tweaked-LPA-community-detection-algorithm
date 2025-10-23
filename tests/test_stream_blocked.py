import numpy as np
from lpkit.label_propagation import label_propagation
from lpkit import symmetrize_and_sort, build_block_index, init_labels_memmap, sweep_labels_inplace_blocked, stream_multi_sweep


#partition into frozenset vertex-ids
def groups_from_labels(labels):
    buckets = {}
    for i, lab in enumerate(labels):
        lab = int(lab)
        buckets.setdefault(lab, set()).add(i)
    return {frozenset(s) for s in buckets.values()}

def test_blocked_one_sweep_matches_count(tmp_path):
    raw = tmp_path / "g.edgelist"
    raw.write_text("0 1\n1 2\n2 0\n3 4\n4 5\n5 3\n")
    sorted_sym = tmp_path / "g.sorted.sym"
    blk = tmp_path / "g.blockidx.npy"
    labels = tmp_path / "g.labels.npy"

    #ram baseline after 1 sweep-equivalent
    adj = [[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]]
    labels_ram, _ = label_propagation(adj, seed=1337)

    meta = symmetrize_and_sort(str(raw), str(sorted_sym))
    n = meta["n"]
    build_block_index(str(sorted_sym), n=n, block_size=n, index_path=str(blk))
    init_labels_memmap(str(labels), n=n)
    sweep_labels_inplace_blocked(str(sorted_sym), str(blk), str(labels), n=n, block_size=n, seed=5543552, tie_break="min")
    mm = np.lib.format.open_memmap(str(labels), mode="r+")

    assert len(set(labels_ram)) == len(set(mm))

def test_blocked_converges_like_ram(tmp_path):
    raw = tmp_path / "g.edgelist"
    raw.write_text("0 1\n1 2\n2 0\n3 4\n4 5\n5 3\n")
    sorted_sym = tmp_path / "g.sorted.sym"
    blk = tmp_path / "g.blockidx.npy"
    labels = tmp_path / "g.labels.npy"

    #ram baseline to convergence
    adj = [[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]]
    labels_ram, _ = label_propagation(adj, seed=1337)

    meta = symmetrize_and_sort(str(raw), str(sorted_sym))
    n = meta["n"]
    build_block_index(str(sorted_sym), n=n, block_size=n, index_path=str(blk))
    init_labels_memmap(str(labels), n=n)

    fin = stream_multi_sweep(str(sorted_sym), str(blk), str(labels), n=n, block_size=n, seed=12233, max_sweeps=100, tie_break="min")
    assert fin["sweeps"] >= 1

    mm = np.lib.format.open_memmap(str(labels), mode="r+")
    assert groups_from_labels(labels_ram) == groups_from_labels(mm)
