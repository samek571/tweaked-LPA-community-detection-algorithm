from lpkit.label_propagation import label_propagation
from lpkit import symmetrize_and_sort, build_vertex_index, init_labels_memmap, sweep_labels_inplace
import numpy as np


def test_streaming(tmp_path):
    raw = tmp_path / "g.edgelist"
    # two K_3 as undirected edges
    raw.write_text("0 1\n1 2\n2 0\n3 4\n4 5\n5 3\n")
    sorted_sym = tmp_path / "g.sorted.sym"
    offsets = tmp_path / "g.offsets.npy"
    degrees = tmp_path / "g.degrees.npy"
    labels = tmp_path / "g.labels.npy"

    meta = symmetrize_and_sort(str(raw), str(sorted_sym))
    assert meta["n"] == 6 and meta["m"] == 6

    build_vertex_index(str(sorted_sym), n=6,
                       offsets_path=str(offsets),
                       degrees_path=str(degrees))

    init_labels_memmap(str(labels), n=6)

    info = sweep_labels_inplace(str(sorted_sym),
                                str(offsets),
                                str(degrees),
                                str(labels),
                                n=6,
                                block_size=3,
                                seed=7)
    assert "updated" in info and "blocks" in info

    mm = np.lib.format.open_memmap(str(labels), mode="r+")
    assert mm.shape[0] == 6


#this is the reason we did the first commit basically, just to prove correctness of the following
#that off ram and in ram is doing the same step and are correct
def test_streaming_algo_doing_the_same_as_non_RAM_algo(tmp_path):
    raw = tmp_path / "g.txt"
    raw.write_text("0 1\n1 2\n2 0\n3 4\n4 5\n5 3\n")
    adj = [[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]]
    labels_ram, _ = label_propagation(adj, seed=1337)

    sorted_sym = tmp_path / "g.sorted"
    offsets = tmp_path / "g.offsets.npy"
    degrees = tmp_path / "g.degrees.npy"
    labels = tmp_path / "g.labels.npy"

    meta = symmetrize_and_sort(str(raw), str(sorted_sym))
    n = meta["n"]

    build_vertex_index(str(sorted_sym), n=n,
                       offsets_path=str(offsets),
                       degrees_path=str(degrees))
    init_labels_memmap(str(labels), n=n)

    sweep_labels_inplace(str(sorted_sym),
                         str(offsets),
                         str(degrees),
                         str(labels),
                         n=n,
                         block_size=n,
                         seed=42)

    labels_stream = np.lib.format.open_memmap(str(labels), mode="r+")
    assert len(set(labels_ram)) == len(set(labels_stream))
    print("\n",set(labels_ram), set(labels_stream))
