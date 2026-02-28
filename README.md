## 1. Overview

`lpkit` implements scalable Label Propagation Algorithm (LPA) for community detection on large graphs.

It supports:

- In-memory LPA for small/medium graphs.
- Disk-backed streaming LPA for graphs too large to fit into RAM.
- Execution under strict memory limits (systemd-run -p MemoryMax=...).

The streaming pipeline is designed for graphs with:
- millions of vertices
- hundreds of millions to billions of edges
- limited availability of RAM size

## 2. Conceptual Pipeline
1. raw edgelist
2. symmetrize + sort
3. split into blocks
4. initialize labels (memmap)
5. multi-sweep label propagation
6. labels.npy (final communities)

## 3. Detailed Pipeline Description
###   3.1 Input Format

Input must be a 2-column whitespace-separated edgelist:
u v
u v
...

Comment lines beginning with # are ignored.
Graph may be directed; streaming mode symmetrizes internally.

### 3.2 Symmetrization + Sorting

Function: `symmetrize_and_sort(in_path, out_path)`

Purpose:
- Convert to undirected (ensure (u,v) and (v,u)).
- Sort edges by source vertex.
- Normalize vertex IDs if necessary.
- Determine n.

Time Complexity:
- $\Theta(m)$ to read
- $\Theta(m log m)$ external sort (dominant)
- I/O heavy

Disk Impact:

- Produces a large .sorted.sym file.
- For m edges, size ≈ 2m edges stored.

### 3.3 Block Splitting

Function: `split_sorted_sym_to_blocks(...)`

Purpose:
- Split sorted symmetric edges into contiguous blocks of size block_size.
- Each block is stored separately on disk.
  - Limits memory footprint during sweep.
  - Enables processing huge graphs under tight RAM.

Block Size Tradeoff: 
- Small block_size → more blocks → higher I/O overhead.
- Large block_size → fewer blocks → higher RAM peak.

Recommended:
- millions for large graphs
- tens of thousands for medium graphs

### 3.4 Label Initialization

Function: `init_labels_memmap(path, n)`

Creates a NumPy memmap array:

- size: n
- dtype: uint64
- initial labels = vertex ID

Memory usage:
- $\Theta(n)$
- Stored on disk
- Paged automatically by OS

### 3.5 Multi-Sweep Streaming LPA

Function: `stream_multi_sweep_parallel_blocks(...)`

For each sweep:
```py
for block in blocks:
    load block
    update labels in-place
```

Termination:
- Stop if updated_this_sweep <= epsilon * n
- Or if max_sweeps reached

### 3.6 Design Guarantees

The streaming pipeline maintains the following invariants:

- Labels are stored in a contiguous `uint64` NumPy memmap of size `n`.
- The full adjacency is never loaded into RAM in streaming mode.
- At any time, only a single block of edges is processed in memory.
- Peak RAM usage is approximately:

  O(block_size) + O(label working window)

- Disk is used as the primary storage medium for adjacency.
- If execution completes successfully (i.e., not killed by OOM), the output is a valid LPA partition under the chosen update order and tie-breaking rule.

#### Determinism Conditions

Reproducible results require:

- fixed `seed`
- `tie_break="min"`
- `workers=1`
- fixed `block_size`
Changing block size or parallelism may change update order and lead to different local optima.

## 4. Algorithmic Properties
### 4.1 Time Complexity

Let:
- n = vertices
- m = edges
- S = number of sweeps

Total runtime decomposes into:

- $\Theta(m)$ input scan / symmetrization
- $\Theta(m log m)$ external sort (dominant preprocessing cost, heavily I/O-bound)
- $\Theta(m)$ block splitting
- $\Theta(S * m)$ streaming label propagation sweeps

Overall cost is: $\Theta(m log m)$ + $\Theta(S * m)$

#### Practical interpretation

- The first execution is usually dominated by sorting and block materialization.
- If `S` is small (typical range: 10–50 sweeps), runtime after preprocessing is close to linear in `m`.
- If `S` is large, repeated full sweeps dominate.

#### Worst case / problematic behavior

In pathological cases (e.g., unstable near-ties or slow-moving labels), convergence may require many sweeps, so runtime trends toward $\Theta(S * m)$ with large $S$.
This is why `max_sweeps` is an important operational guardrail.



### 4.2 Space Complexity

#### RAM

- Labels memmap metadata / access window: `Θ(n)` logical state (OS-paged)
- One in-memory edge block: `Θ(block_size)`
- Temporary buffers and bookkeeping: implementation-dependent, but small relative to edge storage

Peak resident memory is therefore controlled primarily by `block_size` and the OS page cache behavior.

#### Disk

Streaming mode may materialize:

- raw edgelist
- sorted symmetric edgelist (`.sorted.sym`)
- block files (`.blocks/`)
- labels memmap (`labels.npy`)

For very large graphs, total disk footprint can exceed **3–5×** the raw edgelist size.  
This is expected due to symmetrization, sorting, and block materialization.

## 5. Memory-Constrained Execution

Run under memory cap:
```py
systemd-run --user --scope -p MemoryMax=512M 
lpkit stream ...
```

Behavior regimes:
1. Working set fits → same runtime.
2. Slightly exceeds → paging → slower.
3. Strongly exceeds → OOM kill.

If process completes, result is valid.
If killed, memory cap too tight.

## 6. Recommended Usage Patterns
### Small graph (<1M edges)
``` lpkit stream --block-size n ```

or use in-memory `label_propagation`

### Medium graph (1M–100M edges)
`--block-size 100k–5M`

### Very large graph (>100M edges)
```
--block-size millions
low initial max_sweeps
increase gradually
```


## 7. Known Bottlenecks

1. External sort phase
2. Very small block sizes
3. Extremely high-degree vertices
4. Tight memory caps causing thrash
5. Very strict convergence threshold

## 8. Failure Modes & Diagnostics

### OOM Kill under MemoryMax

If the process exits with return codes like `-9`, `-15`, `137`, or `143`, it was likely killed by the cgroup memory limit.

Typical fixes:
- Increase `MemoryMax`
- Increase `block_size` only if you are overhead-bound (fewer blocks)
- Reduce graph size for debugging
- Run without a cap first to establish a baseline
---

### Disk Full During Preprocessing

Streaming mode produces:

- `.sorted.sym`
- `blocks/` directory
- `labels.npy`

Disk usage can exceed 3–5× raw graph size.
Check disk space:
`df -h`

---

### Converged = 0 at max_sweeps

Example output:

    [sweep 100/100] updated=5819
    [STREAM(bs=50000)] n=36692 communities=2710 sweeps=100 converged=0 time=0.585s

Meaning:
- The algorithm hit `max_sweeps`.
- The convergence threshold was not satisfied.

Convergence condition:
`updated_this_sweep <= epsilon * n`

To allow convergence:
- Increase `--max-sweeps`
- Relax epsilon (if configurable)

---

### Extremely Small block_size

Very small block sizes cause:
- Excessive block count
- Heavy filesystem overhead
- Slower overall runtime


## 9. Test Coverage Summary
- `test_basic.py`: correctness on small graphs.
- `test_stream.py`: streaming equivalence to in-memory.
- `test_compare_hdd_vs_ram.py`: streaming vs RAM on larger graphs.
- `test_perf_memory_budget_trend.py`: behavior under MemoryMax.

Performance tests are gated behind:
` LPKIT_RUN_PERF=1`

## 10. Set-up

### Install and venv setup
`python3 -m venv .venv` \
`source .venv/bin/activate`
- if `python3 -m venv gives an error`, install it: `sudo apt install python3-venv`

`pip install -e .` (registers lpkit)

### Run Pre-Made Tests
`PYTHONPATH=src pytest -q` or simply `pytest -q` (thanks to existing `.ini` file)

### Example run
```sh
lpkit stream --in path/to/big_graph.edgelist \
             --out labels.npy \
             --seed 1337 \
             --max-sweeps 50 \
             --block-size 5000
```
Output:
```[STREAM(bs=50000)] n=36692 communities=2710 sweeps=100 converged=0 time=0.585s```

### refer to manual
`lpkit --help`\
`lpkit ram --help`\
`lpkit stream --help`

## 11. Programmatic API Usage (One-Line Wrapper)

The streaming pipeline can also be used directly from Python:
For library usage, `lpkit` can be used from Python through a high-level wrapper:

```python
from lpkit import stream_lpa

info = stream_lpa(
    "graph.edgelist",
    "labels.npy",
    block_size=100_000,
    max_sweeps=50,
    seed=1337,
    workers=1,
    tie_break="min",
)

print(info)
```

## 12. Programmatic API Usage (Low-Level Pipeline)

If you need full control over intermediate artifacts:

```python
from lpkit.stream import (
    symmetrize_and_sort,
    split_sorted_sym_to_blocks,
    init_labels_memmap,
    stream_multi_sweep_parallel_blocks,
)

raw = "graph.edgelist"
sorted_sym = "graph.sorted.sym"
labels = "graph.labels.npy"
blocks_dir = "graph.blocks"

meta = symmetrize_and_sort(raw, sorted_sym)
n = meta["n"]

block_paths = split_sorted_sym_to_blocks(
    sorted_sym,
    n=n,
    block_size=100_000,
    out_dir=blocks_dir,
)

init_labels_memmap(labels, n=n)

info = stream_multi_sweep_parallel_blocks(
    block_paths,
    labels,
    n=n,
    block_size=100_000,
    seed=1337,
    max_sweeps=50,
    min_sweeps=1,
    tie_break="min",
    workers=1,
)

print(info)
```


## Disclaimer
1. No windows support, dev is doing linux. Some things might now work such as external sort or newest python internal methods might not be compatible with older versions of windows...
2. This is my Semestral project at MFF UK, under supervision of David Hartman.