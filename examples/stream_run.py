from pathlib import Path
import numpy as np
from lpkit.stream import (
    symmetrize_and_sort,
    split_sorted_sym_to_blocks,
    init_labels_memmap,
    stream_multi_sweep_blocks)


BASE = Path(__file__).resolve().parent
RAW         = BASE / "simple.edgelist"        #input edge list
SORTED      = BASE / "simple.sorted.sym"      #symetric + sorted by source
LABELS      = BASE / "simple.labels.npy"      #labels (memmap file)
BLOCKS_DIR  = BASE / "simple.blocks"          #blocks

BASE.mkdir(parents=True, exist_ok=True)
if not RAW.exists():
    RAW.write_text(
        "0 1\n1 2\n2 0\n" #clique1
        "3 4\n4 5\n5 3\n" #clique2
    )

#symmetrize + sort by source vertex
meta = symmetrize_and_sort(str(RAW), str(SORTED))
n = meta["n"]
print("sym+sort:", meta)   #{'n': 6, 'm': 6} for 2clieuqs

block_paths = split_sorted_sym_to_blocks(str(SORTED), n=n, block_size=n, out_dir=str(BLOCKS_DIR))
init_labels_memmap(str(LABELS), n=n)

info = stream_multi_sweep_blocks(
    block_paths,
    str(LABELS),
    n=n,
    block_size=n,
    seed=42,
    max_sweeps=100,
    min_sweeps=1,
    tie_break="min",
)
print("stream:", info)

mm = np.lib.format.open_memmap(str(LABELS), mode="r+")
print("labels:", list(mm))
print("unique labels:", len(set(map(int, mm))))