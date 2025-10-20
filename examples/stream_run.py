from pathlib import Path
import numpy as np
from lpkit import (
    symmetrize_and_sort,
    build_vertex_index,
    init_labels_memmap,
    sweep_labels_inplace)

BASE = Path(__file__).resolve().parent
RAW      = BASE / "simple.edgelist"        #input edge list
SORTED   = BASE / "simple.sorted.sym"      #symetric + sorted by source
OFFSETS  = BASE / "simple.offsets.npy"     #byte offsets per vertex (memmap file)
DEGREES  = BASE / "simple.degrees.npy"     #degree per vertex (memmap file)
LABELS   = BASE / "simple.labels.npy"      #labels (memmap file)

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

#on-disk per-vertex index (offsets + deg)
build_vertex_index(str(SORTED), n=n,
                   offsets_path=str(OFFSETS),
                   degrees_path=str(DEGREES))

#on disk labels[i] = i
init_labels_memmap(str(LABELS), n=n)

#1 async sweep (rng order of u + updates)
#if graph is small lets use 1block, otherwise chunk
info = sweep_labels_inplace(str(SORTED),
                            str(OFFSETS),
                            str(DEGREES),
                            str(LABELS),
                            n=n,
                            block_size=n,   # one big block
                            seed=42)
print("sweep:", info)

#final labels and community cnt
mm = np.lib.format.open_memmap(str(LABELS), mode="r+")
print("labels:", list(mm))
print("unique labels:", len(set(mm)))
