"""Disk-backed / streaming pipeline for Label Propagation.

This module contains the core building blocks used by both the CLI and the
high-level `stream_lpa(...)` API:

1. scan + symmetrize + external sort of edgelist
2. optional index construction for byte-seeking (legacy path / tooling)
3. splitting a sorted symmetric edgelist into binary per-block files
4. multi-sweep streaming label propagation over those blocks

Important invariants
--------------------
- Streaming mode never loads the full adjacency into RAM.
- Labels are stored in a NumPy `.npy` memmap and mutated in place.
- Block size is the main throughput vs. RAM tradeoff knob.
- Determinism requires fixed seed, fixed block size, `tie_break='min'` (or `max`),
- Block processing is intentionally sequential.
"""

import os
import shutil
import subprocess
import numpy as np
import struct
from random import Random
from pathlib import Path
from typing import Iterator, Optional, Tuple, Dict

try:
    from lpkit import _cprop
    HAS_CPROP = True
except Exception: #cytho acceleration is optional
    HAS_CPROP = False


def _iter_edges(path: str, ignore_line_its_comment: str = "#") -> Iterator[Tuple[int, int]]:
    """Yield `(u, v)` edges from a text edgelist.

    Parameters
    ----------
    path:
        Edgelist file (`u v` per line).
    ignore_line_its_comment:
        Prefix used to skip comment lines.
    """
    with open(path, "r") as f:
        for discrete_line in f:
            discrete_line = discrete_line.strip()
            if not discrete_line or discrete_line.startswith(ignore_line_its_comment):
                continue
            a, b = discrete_line.split()
            yield int(a), int(b)


def scan_edgelist(path: str, ignore_line_its_comment: str = "#") -> Tuple[int, int]:
    """Single-pass scan returning `(n, m)` for a text edgelist.

    Returns
    -------
    (n, m)
        `n` is inferred as `max_vertex_id + 1`, `m` is the number of parsed input
        lines (i.e. edges before symmetrization).

    Notes
    -----
    This uses O(1) extra RAM (besides parsing buffers) and O(m) time.
    """
    n, m = 0, 0 #label arr len, m: num of lines aka undir edges in file
    for u, v in _iter_edges(path, ignore_line_its_comment):
        if u < 0 or v < 0:
            raise ValueError(f"Vertex ids: {u} {v} are NEGATIVE!!!!")
        n = max(n, u + 1, v + 1)
        m += 1
    return n, m


def symmetrize_and_sort(
        in_path: str,
        out_path: str,
        *,
        ignore_line_its_comment: str = "#",
        tmp_dir: Optional[str] = None,
    ) -> Dict[str, int]:
    """Create a sorted symmetric edgelist using external `sort`.

    The output contains both `(u, v)` and `(v, u)` for every input edge. The file
    is sorted by source vertex `u` (and then by `v`) so later stages can stream by
    source-vertex blocks.

    Notes
    -----
    - No deduplication is performed.
    - Self loops are preserved.
    - This is usually the dominant preprocessing stage for large graphs.
    """

    #external sort which doesnt work on ram like mergesort or others
    if shutil.which("sort") is None:
        raise RuntimeError("External 'sort' not found. Fix by installing core-utils or presort file manually (ha ha)")

    if tmp_dir is None:
        tmp_dir = os.path.dirname(os.path.abspath(out_path)) or "."

    n, m = scan_edgelist(in_path, ignore_line_its_comment=ignore_line_its_comment)
    tmp_unsym = os.path.join(tmp_dir, os.path.basename(out_path) + ".unsym.tmp")

    #symm unsorted file built from one input streamed
    with open(tmp_unsym, "w") as w:
        for u, v in _iter_edges(in_path, ignore_line_its_comment):
            if u == v: #keeping selfloops as hypergraph is what we operate on
                pass
            w.write(f"{u} {v}\n")
            w.write(f"{v} {u}\n")

    #console external optimized sort, -numeric, primary (u) secondary (v), inputfile, outpath
    # `LC_ALL=C` makes the external sort deterministic bytewise on the same input.
    cmd = ["sort", "-n", "-k1,1", "-k2,2", tmp_unsym, "-o", out_path]
    env = os.environ.copy()
    env["LC_ALL"] = "C" #bytewise deterministic sorting ()
    subprocess.run(cmd, check=True, env=env)

    os.remove(tmp_unsym)
    return {"n": n, "m": m}

# idea is to create 2 memory maps on disk that we iteratively work with
# great seeking at neighbors, runs in O(n) disk and O(1) in RAM
def build_vertex_index(
        sorted_sym_path: str,
        *,
        n: int,
        offsets_path: str, #indexed gives start of u
        degrees_path: str, #number of lines for u == out deg in graph
    ) -> None:
    """Build vertex-level byte-offset and degree memmaps for a sorted edgelist.

    This is a legacy helper path used by older experiments. The newer streaming path
    primarily uses `split_sorted_sym_to_blocks(...)`, but the index is still useful
    for debugging and auxiliary scripts.

    `offsets[u]` points to the first byte position where source vertex `u` appears in
    the sorted symmetric text file. `degrees[u]` stores the number of outgoing lines
    for `u` in that file.
    """
    offsets = np.lib.format.open_memmap(offsets_path, mode="w+", dtype=np.uint64, shape=(n,))
    degrees = np.lib.format.open_memmap(degrees_path, mode="w+", dtype=np.uint64, shape=(n,))
    offsets[:] = 0
    degrees[:] = 0

    with open(sorted_sym_path, "rb") as f:
        prev_u = -1
        while True:
            pos = f.tell()
            line = f.readline()
            if not line: break

            splitted_line = line.split()
            if not splitted_line: continue
            u = int(splitted_line[0])

            #filling holes
            if u > prev_u + 1:
                offsets[prev_u + 1 : u] = pos

            #first occur
            if u != prev_u:
                offsets[u] = pos
                prev_u = u
            degrees[u] += 1

        #end of file...
        eof = f.tell()
        if prev_u < n - 1:
            offsets[prev_u + 1 : n] = eof

    #flush to disk
    offsets.flush()
    degrees.flush()


def init_labels_memmap(path: str, *, n: int, dtype=np.uint64) -> np.memmap:
    """Create label storage as a `.npy` memmap initialized to identity labels."""
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(n,))
    mm[:] = np.arange(n, dtype=dtype)
    return mm

def open_labels_memmap(path: str) -> np.memmap:
    """Open an existing labels memmap for read/write updates."""
    return np.lib.format.open_memmap(path, mode="r+")


def build_block_index(
        sorted_path: str,
        *,
        n: int,
        block_size: int,
        index_path: str,
    ) -> None:
    """Build a byte-offset index for source-vertex blocks in a sorted edgelist.

    The saved array has length `num_blocks + 1`, where each entry marks the byte
    offset where a new source-vertex block starts in the text file. This is a legacy
    helper retained for compatibility; the current fast path uses binary split block
    files generated by `split_sorted_sym_to_blocks(...)`.

    [b*block_size, (b+1)*block_size) lie within [offsets[b], offsets[b+1]).
    """
    num_blocks = (n + block_size - 1) // block_size
    offsets = np.zeros(num_blocks + 1, dtype=np.uint64)

    with open(sorted_path, "rb") as f:
        offsets[0] = f.tell() #start of block 0
        last_pos = offsets[0]
        current_block = 0

        while True:
            line_start = last_pos #position at the start of the line
            line = f.readline()
            last_pos = f.tell() #position after reading the line

            if not line: break

            tmp = line.split()
            if not tmp: continue

            u = int(tmp[0])
            b = u // block_size
            #entering new block b, its offset is marked as start
            while b > current_block:
                current_block += 1
                if current_block <= num_blocks:
                    offsets[current_block] = line_start

        #endof file
        offsets[-1] = last_pos

    np.save(index_path, offsets)


def split_sorted_sym_to_blocks(
        sorted_sym_path: str,
        *,
        n: int,
        block_size: int,
        out_dir: str | None = None,
    ) -> list[str]:
    """Split a sorted symmetric text edgelist into binary per-block files.

    Output block `b` contains little-endian `uint32` pairs `(u, v)` where
    `u // block_size == b`. Empty trailing blocks are created as empty files so the
    returned block list always has a stable length (`ceil(n / block_size)`).

    Returns
    -------
    list[str]
        Paths to block files in block index order.
    """
    base = Path(sorted_sym_path)
    if out_dir is None:
        out_dir = base.with_suffix(".blocks")
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    num_blocks = (n + block_size - 1) // block_size
    block_paths = [out_dir_path / f"block_{b:05d}.bin" for b in range(num_blocks)]

    current_block = 0
    fout = open(block_paths[current_block], "wb")

    with open(sorted_sym_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = line.split()
            u = int(a)
            v = int(b)
            target_block = u // block_size

            #closer to the block where u belongs
            while target_block > current_block:
                fout.close()
                current_block += 1
                fout = open(block_paths[current_block], "wb")

            #(u,v) as two uint32 little-endian
            fout.write(struct.pack("<II", u, v))

    fout.close()

    for b in range(current_block + 1, num_blocks):
        block_paths[b].touch() #in case we never reach them, new gets made

    return [str(p) for p in block_paths]


def stream_multi_sweep_blocks(
        block_paths: list[str],
        labels_path: str,
        *,
        n: int,
        block_size: int | None = None,
        seed: int = 0,
        max_sweeps: int = 50,
        min_sweeps: int = 1,
        tie_break: str = "random",
    ) -> Dict[str, int]:
    """Run streaming LPA sweeps over pre-split block files.

    Execution model (per sweep)
    ---------------------------
    1. Snapshot current labels into RAM (`snapshot`).
    2. Process blocks in a shuffled order sequentially
    3. For each block, compute label updates using the snapshot.
    4. Apply updates to the labels memmap and flush.

    Convergence
    -----------
    The current heuristic marks convergence when:

        updated_this_sweep <= 1e-4 * n

    and `s >= min_sweeps`.

    Notes
    -----
    - Different `block_size`, `workers`, and tie-breaking can change update order and
      lead to different local optima (valid but not identical partitions).
    - singlethread sequential block processing to keep the sweep semantics deterministic
    """
    if block_size is None:
        block_size = 5000

    labels = open_labels_memmap(labels_path)
    num_blocks = len(block_paths)
    total_updates, done= 0, False

    def tb_code():
        """Map tie-break mode to the integer codes expected by the Cython kernel."""
        if tie_break == "min": return 1
        if tie_break == "max": return 2
        return 0
    TB = tb_code()

    def process_block(block_idx: int, sweep_seed: int) -> Dict[int, int]:
        """Compute updates for a single block using the current sweep snapshot.

        This returns a sparse mapping `{vertex: new_label}`. The actual memmap writes
        are applied by the caller so update accounting stays centralized.

        if Cython kernel (_cprop.lpa_block) is available, we use it for like massive improvements
        python fallback: block file is already sorted so we maintain only label frequency map for inspected vertex
        hence we avoid storing all neighbor vertex ids in ram, only labels will suffice improving memory complexity
        (time complexity stays the same as we need to check trough all of them)
        """
        path = block_paths[block_idx]
        data = np.memmap(path, mode="r", dtype=np.uint32)
        if data.size == 0:
            return {}

        edges = data.reshape(-1, 2)

        # fast path: Cython kernel, no Python preprocessing (consumes raw edges + snapshot directly)
        if HAS_CPROP:
            return _cprop.lpa_block(edges, snapshot, TB, sweep_seed + block_idx)


        updates: Dict[int, int] = {}
        current_u: int | None = None
        cnts: Dict[int, int] = {}

        def _finalize_vertex(u: int, counts: Dict[int, int]) -> None:
            """Choose the plurality label for one vertex and stage an update if needed."""
            if not counts:
                return

            max_cnt = max(cnts.values())
            if tie_break == "min":
                new_lab = min([lab for lab, c in cnts.items() if c == max_cnt])
            elif tie_break == "max":
                new_lab = max([lab for lab, c in cnts.items() if c == max_cnt])
            else:
                r = Random(sweep_seed + u)
                cands = [lab for lab, c in cnts.items() if c == max_cnt]
                new_lab = r.choice(cands)

            if new_lab != int(snapshot[u]):
                updates[u] = new_lab


        for u_raw, v_raw in edges:
            u = int(u_raw)
            v = int(v_raw)

            if current_u is None:
                current_u = u

            if u != current_u:
                _finalize_vertex(current_u, cnts)
                cnts.clear()
                current_u = u

            lv = int(snapshot[v])
            cnts[lv] = cnts.get(lv, 0) + 1

        if current_u is not None:
            _finalize_vertex(current_u, cnts)

        return updates


    for s in range(1, max_sweeps + 1):
        print(f"\r[sweep {s}/{max_sweeps}] running...", end="", flush=True)

        snapshot = np.asarray(labels, dtype=np.int64) # decouples read labels (this sweep) from writes
        rng = Random(seed + s)
        block_order = list(range(num_blocks))
        rng.shuffle(block_order)
        updated_this = 0
        for b in block_order:
            block_updates = process_block(b, seed + s * 1000003)
            for u, new_lab in block_updates.items():
                labels[u] = new_lab
                updated_this += 1

        labels.flush()
        total_updates += updated_this
        print(f"\r[sweep {s}/{max_sweeps}] updated={updated_this}      ", end="", flush=True)

        if updated_this <= 1e-4*n and s >= min_sweeps:
            done = True
            sweeps = s
            break
    else:
        sweeps = max_sweeps

    return {
        "sweeps": sweeps,
        "converged": int(done),
        "total_updates": int(total_updates),
        "n": int(n),
        "block_size": int(block_size) if block_size is not None else -1,
    }

def is_global_stable(sorted_sym_path, snapshot):
    """Check global LPA stability by scanning a sorted symmetric edgelist.

    This is an expensive verification helper used for diagnostics / experiments,
    not in the main fast path.
    """
    with open(sorted_sym_path, "r") as f:
        last_u = None
        neighbor_labels = []

        for line in f:
            u, v = map(int, line.split())

            if last_u is None:
                last_u = u

            if u != last_u:
                #evaluate stability for last_u
                if not is_vertex_stable(last_u, neighbor_labels, snapshot):
                    return False
                last_u = u
                neighbor_labels = []

            neighbor_labels.append(snapshot[v])

        if last_u is not None:
            if not is_vertex_stable(last_u, neighbor_labels, snapshot):
                return False

    return True


def is_vertex_stable(u, neigh_labels, snapshot):
    """Return whether vertex `u` already holds a maximal-frequency neighbor label."""
    #max degr is small == python sufficies
    cnt = {}
    for L in neigh_labels:
        cnt[L] = cnt.get(L, 0) + 1

    if not cnt:
        return True

    max_count = max(cnt.values())
    my_label = snapshot[u]
    my_count = cnt.get(my_label, 0)
    return my_count >= max_count
