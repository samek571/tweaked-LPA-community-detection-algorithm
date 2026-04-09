# LPKit Developer Specification

## 1. Purpose of this document

This document describes the **developer-facing specification** of LPKit.

It is not a quickstart or user tutorial.
Its goal is to answer the questions a reviewer, supervisor, or future maintainer would otherwise have to answer by reading the code:

- what exact variant of Label Propagation Algorithm (LPA) is implemented,
- how the streaming pipeline works,
- what each major method is responsible for,
- what invariants hold during execution,
- what correctness means in this project,
- how the program behaves asymptotically and operationally,
- where implementation choices differ from the classical randomized presentation of LPA.

---

## 2. Project scope

`lpkit` implements community detection by **label propagation** in two forms:

- **in-memory** implementation for small and medium graphs,
- **streaming disk-backed** implementation for graphs too large to fit into memory (main purpose of this project)

The streaming path is the primary contribution of the project. It replaces full adjacency residency in RAM with:
1. preprocessing of the raw edgelist,
2. a source-sorted symmetric edge layout on disk,
3. block materialization for bounded working-set size,
4. lightweight per-block source-range metadata,
5. memmap-backed label storage,
6. random-order vertex sweeps driven by an oracle permutation.

The project does not introduce a new community objective. It implements a scalable execution model for LPA.

---

## 3. Classical Label Propagation Algorithm

The classical Label Propagation Algorithm works as follows:

1. Assign each vertex a unique initial label.
2. Repeatedly visit vertices in some order.
3. For each vertex, inspect the labels of its neighbors.
4. Replace the current label by the most frequent neighboring label.
5. Repeat until no vertex wishes to change, or until an equivalent local stopping condition is satisfied.

In the widely cited asynchronous form of LPA:

- every node starts with label equal to its own id,
- the node update order is randomized each iteration,
- ties among equally frequent labels are broken randomly,
- updates are applied immediately, so later vertices in the same iteration may see already-updated labels.

This implementation preserves the same **local update semantics**:
- Vertex adopts a label that is maximally represented in its neighborhood, but changes the **execution schedule** in streaming mode.

---

## 4. The core implementation difference from textbook LPA

Textbook LPA usually describes an iteration over a **random permutation of vertices**.

A naive streaming implementation cannot do that directly on huge graphs, because random vertex access normally assumes an in-memory adjacency structure. LPKit preserves the randomized vertex schedule while keeping adjacency on disk:

1. symmetrize the graph,
2. sort edges by source vertex,
3. split the sorted stream into blocks,
4. for each sweep, draw a random permutation of vertices,
5. for each vertex `u` in that order:
    - locate the first block that may contain source `u`,
    - search inside the block for the first edge of `u`,
    - scan the contiguous run of edges with source `u`,
    - if the run reaches the end of the block, continue into the next block,
    - count neighbor labels on the fly,
    - update `u` immediately.

So the project preserves the classical **randomized asynchronous vertex schedule**, but changes the storage model:

- **from in-memory adjacency**
  to
- **source-sorted edge blocks on disk with lightweight metadata**

for scalability under limited RAM.

---

## 5. Bijection between LPA concepts and LPKit methods

This section maps the mathematical description of LPA to the actual implementation.

### 5.1 Unique initial labels

Classical concept:
- every vertex `v` starts with `L(v) = v`

LPKit method:
- `init_labels_memmap(path, n)`

Responsibility:
- allocate a contiguous `uint64` memmap of size `n`
- initialize labels so that label of vertex `i` is `i`

---

### 5.2 Choosing the vertex processing order

Classical concept:
- pick a random order of vertices in each iteration

LPKit methods:
- `symmetrize_and_sort(in_path, out_path)`
- `split_sorted_sym_to_blocks(...)`
- `stream_multi_sweep_blocks(...)`

Responsibilities:
- convert the graph into an undirected/symmetric form,
- produce a source-sorted edge stream,
- split it into bounded-size blocks,
- build per-block source-range metadata,
- generate a fresh random permutation of vertices each sweep,
- process vertices in that permuted order by locating their adjacency runs across blocks.

Unlike the older purely block-sequential design, vertices are now explicitly permuted at sweep level.

---

### 5.3 Neighborhood majority-label update

Classical concept:
- for a vertex `v`, select the most frequent label among `N(v)`

LPKit implementations:
- `label_propagation(...)` in RAM mode
- the update kernel inside `stream_multi_sweep_blocks(...)` in streaming mode

Responsibility:
- read the labels of neighboring vertices,
- count label frequencies,
- choose the winning label,
- update the current vertex label

Yielding semantic core of LPA

---

### 5.4 Tie-breaking

Classical concept:
- random choice among equally frequent maximal labels

LPKit default/testing behavior:
- deterministic tie-breaking, typically `tie_break="min"`

Reason:
- reproducibility,
- stable test behavior,
- easier comparison between RAM and streaming runs

This is an intentional divergence from the original random-tie description. Doesn't hurt anybody.

---

### 5.5 Termination

Classical concept:
- stop when every vertex already carries a maximal neighborhood label

LPKit streaming behavior:
- stop when the number of updates in a sweep is sufficiently small relative to graph size,
- or stop when `max_sweeps` is reached

Operational condition:
- `updated_this_sweep <= epsilon * n`
- or stop by sweep budget

Practical convergence substitute suitable for large-scale streaming.

---

## 6. Streaming execution pipeline

A call to `lpkit stream ...` or `stream_lpa(...)` is a pipeline, not a single primitive.

### 6.1 Input ingestion
Input is a 2-column whitespace-separated edgelist:

```text
u1 v1
u2 v2
...
```
The input may represent a directed graph. Streaming mode symmetrizes it internally.

Questions resolved in this stage:
- what vertices exist,
- what edges exist,
- what the maximum vertex id is,
- what the logical graph size is.

### 6.2 Symmetrization and sorting

Method:
- `symmetrize_and_sort(in_path, out_path)`

Responsibilities:
- ensure undirected behavior by creating both directions of every logical edge,
- sort edges by source vertex id,
- determine `n`,
- produce `.sorted.sym`

Why it exists:
- later streaming sweeps require fast location of each vertex's contiguous source run,
- sorted source order groups neighbors coherently,
- the raw edgelist is not suitable for random-order vertex processing without preprocessing.

#### About the sorting algorithm

For the purposes of LPKit, the only semantically relevant property of the sorting step is:

> the resulting file is sorted by source vertex in ascending order.

The exact external/disk sort algorithm is **not important** to the propagation logic, as long as this invariant holds
- LPKit does not rely on sort stability, relies on the fact that edges of the same source vertex are grouped together,
- after preprocessing, the propagation stage works by scanning blocks and reading/writing labels through a memmap, not by reasoning about the internal mechanics of the sort.

Once that invariant is established, the rest of the algorithm depends on:
- random-order vertex processing over source-sorted block storage,
- memmap-backed label access,
- local majority-label computation,
- sweep-level stopping logic.

---

### 6.3 Block splitting

Method:
- `split_sorted_sym_to_blocks(...)`

Responsibilities:
- split the source-sorted symmetric edge stream into block files,
- preserve the global source-sorted order across the concatenation of blocks,
- keep the working set bounded so that only a small portion of the graph must be resident at once.

Why it exists:
- limits RAM usage,
- avoids full adjacency residency,
- provides the storage substrate over which random-order vertex processing can still be implemented.

---

### 6.4 Block metadata

Inside the propagation routine, LPKit derives lightweight metadata for each block:

- minimum source vertex in the block,
- maximum source vertex in the block.

This metadata is used to locate the first candidate block for a requested vertex `u`.

The metadata is intentionally lightweight:
- it avoids indexing every edge globally,
- it does not change the external storage format,
- it allows the implementation to preserve the existing block-based architecture.

---

### 6.5 Multi-sweep propagation

Method:
- `stream_multi_sweep_blocks(...)`

Responsibilities:
- draw a fresh random permutation of vertices each sweep,
- for each vertex `u` in that order:
    - locate the first block that may contain source `u`,
    - search inside the block for the first occurrence of `u`,
    - scan the contiguous run of edges with source `u`,
    - continue into following blocks if the run spans block boundaries,
    - count neighbor labels on the fly,
    - choose the winning plurality label,
    - update `u` immediately,
- count changes,
- decide whether another sweep is needed.

This is an asynchronous random-order streaming implementation built on top of block files.

---

### 6.6 Multi-sweep propagation

Method:
- `stream_multi_sweep_blocks(...)`

Responsibilities:
- draw a fresh random permutation of vertices each sweep,
- for each vertex `u` in that order:
    - locate the first block that may contain `u`,
    - search inside the block for `u`,
    - scan the contiguous run of edges with source `u`,
    - continue into following blocks if the run spans block boundaries,
    - count neighbor labels on the fly,
    - choose the winning plurality label,
    - update `u` immediately,
- count changes,
- decide whether another sweep is needed.

This is an asynchronous random-order streaming implementation built on top of block files.

---

## 7. How vertices are actually selected for processing

## 7. How vertices are actually selected for processing

This is one of the most important implementation questions.

### 7.1 Classical LPA

In the classical asynchronous description, vertices are processed in a random order in each iteration.

### 7.2 LPKit streaming mode

In LPKit streaming mode, vertices are also processed in an explicit random order in each sweep.

The implementation does **not** process blocks as the scheduling unit anymore. Instead:

1. the graph is stored as a source-sorted symmetric edge stream split into blocks,
2. each sweep samples a random permutation of all vertices,
3. for each vertex `u` in that order:
    - the block metadata is used to locate the first candidate block,
    - the block is searched for the first occurrence of source `u`,
    - the source run is scanned,
    - if needed, scanning continues seamlessly into following blocks until the source changes.

So the effective schedule is:

```text
for each sweep:
    permute vertices randomly
    for u in permutation:
        locate block containing u
        scan adjacency run of u
        count neighbor labels
        update u immediately
```

### 7.3 Why this matters
This explains several observed properties:
- update order can influence which local optimum is reached,
- tie-breaking remains important for reproducibility,
- the implementation keeps storage block-based while making scheduling vertex-based,
- wall-clock time may suffer from weaker locality than purely sequential block scans.

--- 

## 8. Correctness model

“Correctness” for LPA cannot mean “there is only one correct partition.” That is not how the algorithm works.

### 8.1 What correctness means here

In this project, correctness means:
- unique initial labels are assigned correctly,
- neighborhood-majority updates are computed correctly,
- labels are updated asynchronously during a sweep,
- stopping conditions are applied consistently,
- output labels define a valid partition of the graph,
- streaming mode preserves the semantic spirit of asynchronous random-order LPA while using a block-based disk storage discipline.

### 8.2 What correctness does not mean

Correctness does **not** imply:
- the output is unique,
- the partition is globally optimal,
- the streaming result must match a RAM run exactly on every graph,
- the same graph under different block sizes must produce identical partitions.

Different execution orders can legitimately produce different local equilibria.

## 9. Determinism and reproducibility

Reproducibility in this project requires:

- fixed `seed`,
- deterministic tie-breaking, typically `tie_break="min"`,
- identical source-sorted symmetric file,
- identical block split and block ordering.
- identical input graph.

This is not an implementation error. It is a structural consequence of asynchronous label propagation.

## 10. The role of `stream_lpa(...)`

`stream_lpa(...)` is the high-level programmatic integration point.

It is not a separate algorithm. It is a **coordinator** over the lower-level streaming pipeline.

Logical call graph:
`
stream_lpa(...)
    -> symmetrize_and_sort(...)
    -> split_sorted_sym_to_blocks(...)
    -> init_labels_memmap(...)
    -> stream_multi_sweep_blocks(...)
`
Responsibilities of the wrapper:
- validate arguments,
- manage temporary or explicit work directories,
- create and organize intermediate artifacts,
- run the full block-based random-order streaming pipeline,
- return metadata needed by callers.

For Python integration, this is the intended one-line public API.

## 11. Runtime analysis

Let:
- `n` = number of vertices
- `m` = number of edges
- `S` = number of sweeps

### 11.1 Preprocessing

- input scan / symmetrization: `Θ(m)`
- sort: `Θ(m log m)`
- block splitting: `Θ(m)`

The sort is typically the dominant preprocessing cost and is heavily I/O-bound.

### 11.2 Propagation

Each sweep touches the graph through the block stream once, so the logical propagation cost per sweep is:

- `Θ(m)`

Thus the total propagation cost is:

- `Θ(S · m)`

### 11.3 Whole-pipeline runtime

Total runtime is: `Θ(m log m) + Θ(S · m)`

Interpretation:
- first execution is often dominated by sort and block materialization,
- later sweeps are dominated by repeated block lookups and adjacency-run scans.
- if `S` is modest, runtime after preprocessing is near-linear in `m`,
- if `S` is large, repeated sweeps dominate.

### 11.4 Worst-case behavior

Worst-case runtime is driven by large `S`.

This can happen when:
- many vertices repeatedly face unstable near-ties,
- the graph mixes communities weakly,
- the chosen execution order slows consolidation,
- max_sweeps is large and epsilon is strict.

In pathological settings, the propagation part can trend toward many full passes over the graph.

## 12. Space analysis

### 12.1 RAM

In streaming mode, the intended working set is bounded by:
- labels memmap access window,
- one or a few currently loaded blocks,
- lightweight per-block source-range metadata,
- one temporary counting dictionary for the currently processed vertex.

For a vertex `u`, temporary counting memory is:

- `O(number of distinct neighbor labels of u)`

not `O(deg(u))` in the common case.

In the worst case, if every neighbor of `u` has a different label, this still becomes `O(deg(u))`.

### 12.2 Disk

Streaming mode may materialize:
- raw edgelist,
- sorted symmetric edgelist,
- block directory,
- labels memmap.

Disk usage may exceed several multiples of the raw graph size. This is expected because the project deliberately trades disk storage for reduced RAM usage.

## 13. Analysis of problematic subparts

This section answers the practical “where can this become slow or problematic?” question.

### 13.1 Sorting

Sorting is expensive because:

- it touches all edges,
- it is `Θ(m log m)`,
- it is I/O-heavy for large datasets.

However, once sorting is complete, it is no longer central to the propagation logic. The algorithmic behavior of LPKit after preprocessing depends on the **sorted layout**, not on how the sort was internally implemented.

### 13.2 Random vertex order vs block locality

The current implementation preserves a random vertex processing order, but storage is still block-based.

This means:
- the logical propagation work stays linear in `m`,
- but wall-clock time may be hurt by jumping between distant source runs,
- blocks may be re-opened or revisited multiple times within the same sweep,
- SSD-backed execution is much more suitable than HDD-backed execution.

This is the main practical cost of combining random-order scheduling with block-based storage.

### 13.3 Cross-block source runs

A high-degree vertex may have its source run span multiple consecutive blocks.

The implementation must therefore:
- detect the first block containing the vertex,
- continue scanning into following blocks while the source remains unchanged.

This is required for correctness and is not an edge case for hub-heavy graphs.

### 13.4 High-degree vertices

Vertices with huge neighborhoods are expensive because their update step must count many neighboring labels.

Even though the implementation no longer stores full neighbor lists explicitly, the counting dictionary may still grow to:

- `O(number of distinct neighbor labels)`

and in the worst case this is still `O(deg(u))`.

### 13.5 Strict convergence threshold

If epsilon is very small, the algorithm may continue sweeping for a long time even when the partition is already nearly stable.

This explains runs where `converged=0` even after many sweeps.

### 13.6 Tight MemoryMax

Under cgroup memory limits:

- if the working set fits -> runtime may remain almost unchanged,
- if the working set slightly exceeds -> paging and slowdown occur,
- if the working set significantly exceeds -> the process may be killed.

This means “less memory” does not always mean “same run, just slower.” Below a threshold the run simply terminates.

## 14. Interpretation of output metadata

Example style of output:

`[STREAM(bs=50000)] n=36692 communities=2710 sweeps=100 converged=0 time=0.585s`

Meaning of fields:
- `n` -> number of vertices
- `communities` -> number of distinct final labels
- `sweeps` -> number of completed sweeps
- `converged` -> whether the stopping criterion was satisfied before budget exhaustion
- `time` -> measured execution time reported by the streaming pipeline
- `bs` -> block size used when the current block-based streaming mode is active

**What converged=0 means**
- sweep budget was exhausted,
- epsilon-style convergence threshold was not met.

The produced labels still represent a valid propagated state under the chosen budget.


## 15. What the tests are actually verifying

The test suite should be interpreted as follows.

### 15.1 `test_basic.py`

Checks correctness of the in-memory baseline on tiny graphs.

### 15.2 `test_stream.py`

Checks that the streaming pipeline runs and matches the RAM baseline on small deterministic toy graphs, typically by comparing partitions rather than raw label values.

### 15.3 `test_compare_hdd_vs_ram.py`

Checks behavioral consistency between streaming and RAM modes on larger graphs, with intentionally looser assertions because LPA is order-sensitive.

### 15.4 `test_perf_memory_budget_trend.py`

Checks execution under real cgroup memory limits and validates monotone degradation assumptions under constrained memory budgets.

### 15.5 `test_api_wrapper.py`

Checks the public one-call Python API and its artifact semantics.

The tests do not claim that all runs of LPA must produce one unique partition. They validate the implementation under the appropriate notion of correctness for LPA.

## 16. Questions a reviewer is likely to ask

### Is this still LPA if vertices are not processed in random order?

Yes. The local update rule is still LPA. What changed is the scheduling strategy necessary for streaming-scale execution.

### Why not explicitly store random permutations on disk?

Because the main engineering goal is sequential disk access and bounded RAM use. Random scheduling would undermine both.

### Why is the sorting algorithm itself not central to the algorithm description?

Because the semantic requirement is only that edges become source-sorted. The propagation stage depends on that ordering invariant, not on the internal choice of sort implementation.

### Is streaming mode synchronous?

No. It is asynchronous in effect because updated labels can influence later vertices in the same sweep.

### Can two valid runs disagree?

Yes. LPA is update-order sensitive and tie-sensitive.

### Why is there a RAM implementation?

It serves as:
- correctness oracle for tiny graphs,
- simpler reference implementation

### Why might `max_sweeps` stop a run before convergence?

Because the project uses an operational stopping surrogate suitable for large-scale runs, and sweep budgets are needed to bound runtime.

## 17. Practical maintainer guidance

A maintainer can safely assume:
- streaming mode is built around sequential block traversal,
- labels are memmap-backed and persist across sweeps,
- deterministic output requires deterministic settings,
- preprocessing layout invariants are more important than sort internals,
- output partitions may differ across block sizes and still be valid.

A maintainer should **not** assume:
- one unique partition exists,
- exact equality with RAM mode on all graphs,
- exact equality across different streaming schedules.