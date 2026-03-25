from pathlib import Path
import time

from lpkit.generate_graph import generate_large_graph
from lpkit.stream import (
    symmetrize_and_sort,
    split_sorted_sym_to_blocks,
    init_labels_memmap,
    stream_multi_sweep_blocks,
)

BASE    = Path(__file__).resolve().parent
RAW     = BASE / "large_random.edgelist"
SORTED  = BASE / "large_random.sorted.sym"
LABELS  = BASE / "large_random.labels.npy"
BLOCKS_DIR = BASE / "large_random.blocks"

generate_large_graph(str(RAW), n=100_000, m=200_000, topology="random", seed=123)

meta = symmetrize_and_sort(str(RAW), str(SORTED))
n = meta["n"]

block_size = 50_000
block_paths = split_sorted_sym_to_blocks(str(SORTED), n=n, block_size=block_size, out_dir=str(BLOCKS_DIR))
init_labels_memmap(str(LABELS), n=n)

t0 = time.time()
info = stream_multi_sweep_blocks(
    block_paths,
    str(LABELS),
    n=n,
    block_size=block_size,
    seed=42,
    max_sweeps=10,
    min_sweeps=1,
    tie_break="min",
)
dt = time.time() - t0

print("Streaming LPA finished:", info)
print(f"Time to run: {dt:.2f}s")
