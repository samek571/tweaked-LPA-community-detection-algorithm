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
from bisect import bisect_left
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
    """Run streaming LPA sweeps over pre-split source-sorted block files.

    Execution model (per sweep)
    ---------------------------
    1. Ask the oracle for a random permutation of vertices.
    2. For each vertex `u` in that order:
       - locate the first block that may contain `u`,
       - search inside the block for the first occurrence of source `u`,
       - scan the contiguous run of edges with source `u`,
       - if the run reaches the end of the block, continue seamlessly into following blocks,
       - count neighbor labels on the fly,
       - assign the winning plurality label to `u` immediately.

    This is an asynchronous random-order update scheme.

    Convergence
    -----------
    The current heuristic marks convergence when:

        updated_this_sweep <= 1e-4 * n

    and `s >= min_sweeps`.

    Notes
    -----
    - Only block files are materialized in RAM; full adjacency is never built.
    - Temporary per-vertex memory is the label-frequency dictionary for the currently
      processed vertex.
    - High-degree vertices may still be expensive because their source run can span
      many edges and possibly multiple consecutive blocks.
    """
    if block_size is None:
        block_size = 5000

    labels = open_labels_memmap(labels_path)
    num_blocks = len(block_paths)
    total_updates, done= 0, False

    # Load block memmaps once and store source-range metadata.
    block_edges = []
    block_min_u = []
    block_max_u = []

    for path in block_paths:
        data = np.memmap(path, mode="r", dtype=np.uint32)
        if data.size == 0:
            edges = data.reshape(0, 2)
            block_edges.append(edges)
            block_min_u.append(None)
            block_max_u.append(None)
            continue

        edges = data.reshape(-1, 2)
        block_edges.append(edges)
        block_min_u.append(int(edges[0, 0]))
        block_max_u.append(int(edges[-1, 0]))

    # Monotone helper for locating the first candidate block for a source vertex
    block_max_u_search = [(-1 if x is None else x) for x in block_max_u]
    block_srcs = [
        (None if edges.shape[0] == 0 else edges[:, 0])
        for edges in block_edges
    ]

    # Lazy cache: once we locate the first occurrence of source u, reuse it in later sweeps.
    cached_block = np.full(n, -1, dtype=np.int64)
    cached_lo = np.full(n, -1, dtype=np.int64)

    def process_vertex(u: int, sweep_seed: int) -> int | None:
        """Locate source u across block files, count neighbor labels on the fly,
        and return the winning new label or None if no change is needed.
        """
        counts: Dict[int, int] = {}

        # Fast path: reuse previously discovered start location for u.
        if cached_block[u] != -1:
            b = int(cached_block[u])
            lo = int(cached_lo[u])
        else:
            # Find the first block whose max source is >= u.
            b = bisect_left(block_max_u_search, u)
            if b >= num_blocks:
                return None

            lo = -1
            while b < num_blocks:
                if block_min_u[b] is None:
                    b += 1
                    continue

                # If the current block starts after u, then no later block can contain u.
                if block_min_u[b] > u:
                    return None

                srcs = block_srcs[b]
                pos = int(np.searchsorted(srcs, u, side="left"))

                if pos < len(srcs) and int(srcs[pos]) == u:
                    lo = pos
                    cached_block[u] = b
                    cached_lo[u] = pos
                    break

                b += 1

            if lo == -1:
                return None

        found_any = False

        while b < num_blocks:
            if block_min_u[b] is None:
                b += 1
                lo = 0
                continue

            # If block source range is already beyond u, the run is finished globally.
            if block_min_u[b] > u:
                break

            edges = block_edges[b]
            srcs = block_srcs[b]

            # First block starts at cached/searched lo; continuation blocks start at 0.
            if not found_any:
                start = lo
                if start >= len(srcs) or int(srcs[start]) != u:
                    break
            else:
                start = 0
                if len(srcs) == 0 or int(srcs[0]) != u:
                    break

            found_any = True

            if HAS_CPROP:
                i = int(_cprop.accumulate_source_run(edges, labels, u, start, counts))
            else:
                i = start
                while i < len(srcs) and int(srcs[i]) == u:
                    v = int(edges[i, 1])
                    lv = int(labels[v])   # live labels => asynchronous semantics
                    counts[lv] = counts.get(lv, 0) + 1
                    i += 1

            # If source changed inside this block, u is finished.
            if i < len(srcs):
                break

            # Otherwise the run may continue into the next consecutive block.
            b += 1
            lo = 0

        if not found_any or not counts:
            return None

        max_cnt = max(counts.values())

        if tie_break == "min":
            new_lab = min(lab for lab, c in counts.items() if c == max_cnt)
        elif tie_break == "max":
            new_lab = max(lab for lab, c in counts.items() if c == max_cnt)
        else:
            r = Random(sweep_seed + u)
            cands = [lab for lab, c in counts.items() if c == max_cnt]
            new_lab = r.choice(cands)

        if new_lab != int(labels[u]):
            return int(new_lab)

        return None


    for s in range(1, max_sweeps + 1):
        print(f"\r[sweep {s}/{max_sweeps}] running...", end="", flush=True)

        rng = Random(seed + s)
        vertex_order = list(range(n))
        rng.shuffle(vertex_order)
        updated_this = 0

        for u in vertex_order:
            new_lab = process_vertex(u, seed + s * 1000003)
            if new_lab is not None:
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
