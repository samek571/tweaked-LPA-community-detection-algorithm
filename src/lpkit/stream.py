import os
import shutil
import subprocess
import numpy as np
import struct
from random import Random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Optional, Tuple, Dict, List

try:
    from lpkit import _cprop
    HAS_CPROP = True
except Exception:
    HAS_CPROP = False


def _iter_edges(path: str, ignore_line_its_comment: str = "#") -> Iterator[Tuple[int, int]]:
    #yields edges (u, v) line by line without prior or posterior knowledge, each line is discrete
    with open(path, "r") as f:
        for discrete_line in f:
            discrete_line = discrete_line.strip()
            if not discrete_line or discrete_line.startswith(ignore_line_its_comment):
                continue
            a, b = discrete_line.split()
            yield int(a), int(b)


def scan_edgelist(path: str, ignore_line_its_comment: str = "#") -> Tuple[int, int]:
    #one streaming pass, O(1) RAM as we keep only nm
    #(and O(E) time)
    n, m = 0, 0 #label arr len, m: num of lines aka undir edges in file
    for u, v in _iter_edges(path, ignore_line_its_comment):
        if u < 0 or v < 0:
            raise ValueError(f"Vertex ids: {u} {v} are NEGATIVE!!!!")
        n = max(n, u + 1, v + 1)
        m += 1
    return n, m


#makes uv and vu + sort by u then by v (makes O(1) seeking to u without the need of adj list in RAM)
#guarantees consequitveness
#no deduplication (we are dealing with hypergraphs)
def symmetrize_and_sort(
        in_path: str,
        out_path: str,
        *,
        ignore_line_its_comment: str = "#",
        tmp_dir: Optional[str] = None,
    ) -> Dict[str, int]:
    
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

    #single pass over sorted n symmetric file
    #there is no O(n) RAM arrays we write directly to memmaps
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
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(n,))
    mm[:] = np.arange(n, dtype=dtype)
    return mm

def open_labels_memmap(path: str) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="r+")


#one streaming pass, offset = file byte pos
#we create array64 with lenght = numblocks+1 where
#[b*block_size, (b+1)*block_size) lie within [offsets[b], offsets[b+1]).
def build_block_index(
        sorted_path: str,
        *,
        n: int,
        block_size: int,
        index_path: str,
    ) -> None:

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


#one pass over sorted symmetric edgelist, gets splitted into per-block binary files, prep for cython/C idk yet
#returns a list of paths to bin files, little endian unit32 pairs where u in [b*block_size, (b+1)*block_size).
def split_sorted_sym_to_blocks(
        sorted_sym_path: str,
        *,
        n: int,
        block_size: int,
        out_dir: str | None = None,
    ) -> list[str]:

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

#in one sweep -
# - take a snapshot of labels into RAM
# - process blocks in random order (optionally in parallel)
# - accumulate updates per block
# - apply updates back into the memmap
def stream_multi_sweep_parallel_blocks(
        block_paths: list[str],
        labels_path: str,
        *,
        n: int,
        block_size: int | None = None,
        seed: int = 0,
        max_sweeps: int = 50,
        min_sweeps: int = 1,
        tie_break: str = "random",
        workers: int | None = None, #if 1 it is sequential
    ) -> Dict[str, int]:

    if block_size is None:
        block_size = 5000

    labels = open_labels_memmap(labels_path)
    num_blocks = len(block_paths)
    total_updates, done= 0, False

    if workers is None or workers <= 0:
        workers = os.cpu_count() or 4

    #Cython comes in handy
    def tb_code():
        if tie_break == "min": return 1
        if tie_break == "max": return 2
        return 0

    TB = tb_code()

    #one binary block file is loeaded and built u neighbor lists, then label is computed all u in block using snapshot
    #if Cython kernel (_cprop.lpa_block) is available, we use it for like massive improvements
    def process_block(block_idx: int, sweep_seed: int) -> Dict[int, int]:
        path = block_paths[block_idx]
        data = np.memmap(path, mode="r+", dtype=np.uint32)
        if data.size == 0:
            return {}

        edges = data.reshape(-1, 2)

        # fast path: Cython kernel, no Python preprocessing
        if HAS_CPROP:
            return _cprop.lpa_block(edges, snapshot, TB, sweep_seed + block_idx)

        # python fallback
        per_u_neighbors: Dict[int, List[int]] = {}
        for u, v in edges:
            per_u_neighbors.setdefault(u, []).append(v)

        updates: Dict[int, int] = {}
        verts = list(per_u_neighbors.keys())
        rng_loc = Random(sweep_seed + block_idx)
        rng_loc.shuffle(verts)

        for u in verts:
            neighs = per_u_neighbors[u]
            if not neighs:
                continue

            cnts: Dict[int, int] = {}
            for v in neighs:
                lv = int(snapshot[v])
                cnts[lv] = cnts.get(lv, 0) + 1

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

        return updates


    for s in range(1, max_sweeps + 1):
        print(f"\r[sweep {s}/{max_sweeps}] running...", end="", flush=True)

        snapshot = np.asarray(labels, dtype=np.int64)
        rng = Random(seed + s)
        block_order = list(range(num_blocks))
        rng.shuffle(block_order)
        updated_this = 0
        if workers == 1: #sequential
            for b in block_order:
                block_updates = process_block(b, seed + s * 1000003)
                for u, new_lab in block_updates.items():
                    labels[u] = new_lab
                    updated_this += 1
        else: #parallel
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [
                    ex.submit(process_block, b, seed + s * 1000003)
                    for b in block_order
                ]
                done_blocks = 0
                for fut in futures:
                    block_updates = fut.result()
                    for u, new_lab in block_updates.items():
                        labels[u] = new_lab
                        updated_this += 1
                    done_blocks += 1
                    if workers <= 16:
                        print(
                            f"\r[sweep {s}/{max_sweeps}] running... block {done_blocks}/{num_blocks}",
                            end="",
                            flush=True,
                        )

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
        "parallel": int(workers > 1),
        "workers": int(workers),
    }

def is_global_stable(sorted_sym_path, snapshot):
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


#count frequencies
def is_vertex_stable(u, neigh_labels, snapshot):
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
