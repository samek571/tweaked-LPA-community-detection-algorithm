from pathlib import Path
import numpy as np
from lpkit import (
    symmetrize_and_sort,
    build_block_index,
    init_labels_memmap,
    sweep_labels_inplace_blocked,
    stream_multi_sweep)

BASE = Path(__file__).resolve().parent
RAW      = BASE / "simple.edgelist"        #input edge list
SORTED   = BASE / "simple.sorted.sym"      #symetric + sorted by source
BLOCKIDX = BASE / "simple.blockidx.npy"    #idx of blocks
LABELS   = BASE / "simple.labels.npy"      #labels (memmap file)

# create tiny input if missing
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

block_size = max(1, n)  #full graph as one block for parity
build_block_index(str(SORTED), n=n, block_size=block_size, index_path=str(BLOCKIDX))

init_labels_memmap(str(LABELS), n=n)

#single blocked sweep
info = sweep_labels_inplace_blocked(str(SORTED), str(BLOCKIDX), str(LABELS), n=n, block_size=block_size, seed=42, tie_break="min")
print("one sweep:", info)

#multisweep until convergence (optional)
fin = stream_multi_sweep(str(SORTED), str(BLOCKIDX), str(LABELS), n=n, block_size=block_size, seed=123, max_sweeps=10, tie_break="min")
print("multi-sweep:", fin)

mm = np.lib.format.open_memmap(str(LABELS), mode="r+")
print("labels:", list(mm))
print("unique:", len(set(mm)))
