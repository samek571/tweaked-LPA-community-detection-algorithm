import os
import shutil
import subprocess
from typing import Iterator, Optional, Tuple, Dict
import numpy as np
from random import Random


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

# async rng sweep
# for each rng vertex u compute neighbor label histogram using actual labels and update u after
def sweep_labels_inplace(
        sorted_sym_path: str,
        offsets_path: str,
        degrees_path: str,
        labels_path: str,
        *,
        n: int,
        block_size: int,
        seed: Optional[int] = 0,
    ) -> Dict[str, int]:

    labels = open_labels_memmap(labels_path)
    offsets = np.lib.format.open_memmap(offsets_path, mode="r")
    degrees = np.lib.format.open_memmap(degrees_path, mode="r")

    rng, updated = Random(seed), 0
    num_blocks = (n + block_size - 1) // block_size
    block_order = list(range(num_blocks))
    rng.shuffle(block_order)
    with open(sorted_sym_path, "rb") as f: #lazy loading by os, not everything is in RAM
        for b in block_order:

            low = b * block_size
            high = min(n, (b + 1) * block_size)
            vertices = list(range(low, high))
            rng.shuffle(vertices)

            #async updates
            for u in vertices:
                deg_u = int(degrees[u])
                if deg_u == 0: continue

                f.seek(int(offsets[u])) #move to that vertex offset
                #recompute counts from newer labels
                cnts: Dict[int, int] = {}
                for _ in range(deg_u):
                    line = f.readline() #this acrually is in ram
                    if not line: break

                    lab_v = int(labels[int(line.split()[-1])])
                    cnts[lab_v] = cnts.get(lab_v, 0) + 1

                if not cnts: continue

                max_cnt = max(cnts.values())
                new_lab = rng.choice([lab for lab, c in cnts.items() if c == max_cnt])
                if int(labels[u]) != new_lab:
                    labels[u] = new_lab
                    updated += 1

    return {"updated": updated, "blocks": int(num_blocks)}

#______________________ improvement for hdd fast seek

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


#one asynchronous LPA sweep, no per-vertex seek
def sweep_labels_inplace_blocked(
        sorted_sym_path: str,
        block_index_path: str,
        labels_path: str,
        *,
        n: int | None = None,
        block_size: int | None = None,
        seed: int = 0,
        tie_break: str = "random",
    ) -> Dict[str, int]:

    labels = open_labels_memmap(labels_path)
    offsets = np.load(block_index_path) #len = num_blocks+1
    num_blocks = offsets.size - 1

    rng = Random(seed)
    updated = 0
    block_order = list(range(num_blocks)) #rng of block processing
    rng.shuffle(block_order)

    #main sweep
    with open(sorted_sym_path, "rb") as f:
        for b in block_order:

            #sanity check of offset consistency
            if int(offsets[b]) >= int(offsets[b + 1]): continue

            #sequential scan of this blocks edge slice
            f.seek(int(offsets[b]))
            per_u_neighbors: Dict[int, list[int]] = {}
            block_end = int(offsets[b + 1])
            while f.tell() < block_end:
                line = f.readline()
                if not line: break

                sp = line.split()
                if not sp: continue

                u = int(sp[0]); v = int(sp[1])
                per_u_neighbors.setdefault(u, []).append(v) #adj list of verts in this block

            #rng vertex ordering within block
            verts = list(per_u_neighbors.keys())
            rng.shuffle(verts)

            #async label updates
            for u in verts:
                cnts: Dict[int, int] = {}
                for v in per_u_neighbors[u]:
                    lv = int(labels[v])
                    cnts[lv] = cnts.get(lv, 0) + 1
                if not cnts: continue

                max_cnt = max(cnts.values())
                cands = [lab for lab, c in cnts.items() if c == max_cnt]

                #deterministic testing or rng for tie break
                if tie_break == "min": new_lab = min(cands)
                elif tie_break == "max": new_lab = max(cands)
                else: new_lab = rng.choice(cands)

                #if changed: update label
                if int(labels[u]) != new_lab:
                    labels[u] = new_lab
                    updated += 1

            #dfree memory used by this block b and move on
            per_u_neighbors.clear()

    return {"updated": updated, "blocks": int(num_blocks)}


#Run multiple blocked sweeps until convergence or max_sweeps
def stream_multi_sweep(
        sorted_sym_path: str,
        block_index_path: str,
        labels_path: str,
        *,
        n: int | None = None,
        block_size: int | None = None,
        seed: int = 0,
        max_sweeps: int = 50,
        min_sweeps: int = 1,
        tie_break: str = "random",
    ) -> Dict[str, int]:

    total_updates = 0
    done = False
    for s in range(1, max_sweeps + 1):
        info = sweep_labels_inplace_blocked(
            sorted_sym_path,
            block_index_path,
            labels_path,
            n=n,
            block_size=block_size,
            seed=seed + s, #diff rng across sweeps
            tie_break=tie_break,
        )

        total_updates += info["updated"]
        if info["updated"] == 0 and s >= min_sweeps:
            done = True
            sweeps = s
            break
    else:
        sweeps = max_sweeps

    return {
        "sweeps": sweeps,
        "converged": int(done),
        "total_updates": total_updates,
        "n": int(n) if n is not None else -1,
        "block_size": int(block_size) if block_size is not None else -1,
    }