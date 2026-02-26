import numpy as np
from lpkit.label_propagation import label_propagation
from lpkit.stream import symmetrize_and_sort, split_sorted_sym_to_blocks, init_labels_memmap, stream_multi_sweep_parallel_blocks

#partition into frozenset vertex-ids
def groups_from_labels(labels):
    buckets = {}
    for i, lab in enumerate(labels):
        buckets.setdefault(int(lab), set()).add(i)
    return {frozenset(s) for s in buckets.values()}

def test_streaming(tmp_path: str):
    raw = tmp_path / "g.edgelist"
    # two K_3 as undirected edges
    raw.write_text("0 1\n1 2\n2 0\n3 4\n4 5\n5 3\n")
    sorted_sym = tmp_path / "g.sorted.sym"
    labels = tmp_path / "g.labels.npy"

    meta = symmetrize_and_sort(str(raw), str(sorted_sym))
    n = meta["n"]

    block_paths = split_sorted_sym_to_blocks(str(sorted_sym), n=n, block_size=3, out_dir=str(tmp_path / "blocks"))
    init_labels_memmap(str(labels), n=n)

    info = stream_multi_sweep_parallel_blocks(
        block_paths,
        str(labels),
        n=n,
        block_size=3,
        seed=7,
        max_sweeps=1,
        min_sweeps=1,
        tie_break="min",
        workers=1,
    )

    assert info["sweeps"] == 1
    assert isinstance(info.get("total_updates"), int)

    mm = np.lib.format.open_memmap(str(labels), mode="r")
    assert mm.shape == (n,)



#this is the reason we did the first commit basically, just to prove correctness of the following
#that off ram and in ram is doing the same step and are correct
def test_streaming_algo_doing_the_same_as_non_RAM_algo(tmp_path):
    raw = tmp_path / "g.txt"
    raw.write_text("0 1\n1 2\n2 0\n3 4\n4 5\n5 3\n")
    adj = [[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]]
    labels_ram, _ = label_propagation(adj, seed=1337)

    sorted_sym = tmp_path / "g.sorted.sym"
    labels = tmp_path / "g.labels.npy"

    meta = symmetrize_and_sort(str(raw), str(sorted_sym))
    n = meta["n"]

    block_paths = split_sorted_sym_to_blocks(str(sorted_sym), n=n, block_size=n, out_dir=str(tmp_path / "blocks"))
    init_labels_memmap(str(labels), n=n)

    info = stream_multi_sweep_parallel_blocks(
        block_paths,
        str(labels),
        n=n,
        block_size=n,
        seed=1337,
        max_sweeps=100,
        min_sweeps=1,
        tie_break="min",
        workers=1,
    )
    assert info["sweeps"] >= 1

    labels_stream = np.lib.format.open_memmap(str(labels), mode="r+")
    assert groups_from_labels(labels_ram) == groups_from_labels(labels_stream)
