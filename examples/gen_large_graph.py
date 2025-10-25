from lpkit.generate_graph import generate_large_graph
from lpkit import symmetrize_and_sort, build_block_index, init_labels_memmap, stream_multi_sweep
from pathlib import Path
import time


BASE    = Path(__file__).resolve().parent
RAW     = BASE / "large_random.edgelist"
SORTED  = BASE / "large_random.sorted.sym"
IDX     = BASE / "large_random.blockidx.npy"
LABELS  = BASE / "large_random.labels.npy"

generate_large_graph(str(RAW), n=100_000, m=200_000, topology="random", seed=123)

meta = symmetrize_and_sort(str(RAW), str(SORTED))
build_block_index(str(SORTED), n=meta["n"], block_size=50_000, index_path=str(IDX))
init_labels_memmap(str(LABELS), n=meta["n"])

t0 = time.time()
info = stream_multi_sweep(str(SORTED), str(IDX), str(LABELS), n=meta["n"], block_size=50_000, seed=42, max_sweeps=10)
dt = time.time() - t0

print("Streaming LPA finished:", info)
print(f"Time to run: {dt:.2f}s")
