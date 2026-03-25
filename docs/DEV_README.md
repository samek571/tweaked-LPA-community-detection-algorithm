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
2. sequential disk layout,
3. block-wise propagation,
4. memmap-backed label storage.

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

Streaming LPKit cannot do that efficiently on huge graphs because random vertex access would require random access to adjacency lists stored on disk. Instead, the streaming implementation uses a **deterministic edge-ordered schedule**:

1. symmetrize the graph,
2. sort edges by source vertex,
3. split sorted edges into sequential blocks,
4. process those blocks in order, sweep after sweep.

So the project preserves the update rule, but replaces:

- **random node schedule**
with
- **deterministic block-stream schedule**

for engineering reasons, changeing for scalability and disk locality.

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
- pick an order of vertices, often random in each iteration

LPKit methods:
- `symmetrize_and_sort(in_path, out_path)`
- `split_sorted_sym_to_blocks(...)`

Responsibilities:
- convert the graph into an undirected/symmetric form,
- produce a sorted edge stream,
- materialize a sequence of on-disk edge blocks

The sorted stream defines the effective processing order.
Vertices are not explicitly permuted.
They are processed implicitly through the order in which their incident edges appear in the sorted edge stream.

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
- later streaming sweeps require local adjacency access in sequential disk order,
- sorted source order groups neighbors coherently,
- the raw edgelist is not suitable for efficient block streaming.

#### About the sorting algorithm

For the purposes of LPKit, the only semantically relevant property of the sorting step is:

> the resulting file is sorted by source vertex in ascending order.

The exact external/disk sort algorithm is **not important** to the propagation logic, as long as this invariant holds
- LPKit does not rely on sort stability, relies on the fact that edges of the same source vertex are grouped together,
- after preprocessing, the propagation stage works by scanning blocks and reading/writing labels through a memmap, not by reasoning about the internal mechanics of the sort.

Once that invariant is established, the rest of the algorithm depends on:
- sequential block traversal,
- memmap-backed label access,
- local majority-label computation,
- sweep-level stopping logic.

---

### 6.3 Block splitting

Method:
- `split_sorted_sym_to_blocks(...)`

Responsibilities:
- split the sorted symmetric edge stream into contiguous on-disk blocks,
- return the ordered list of those block paths.

Why it exists:
- bounds the in-memory working set,
- enables one-block-at-a-time processing,
- makes execution feasible under MemoryMax constraints.

`block_size` is the key tradeoff parameter:
- smaller blocks -> more files and more filesystem overhead,
- larger blocks -> higher resident memory pressure and weaker tolerance to tight RAM caps.

---

### 6.4 Label-state initialization
Method:
- `init_labels_memmap(path, n)`

Responsibilities:
- create the disk-backed labels file,
- store one label per vertex,
- initialize with identity labels.

Why memmap is used:
- labels can be much smaller than the edge set but still large,
- they must remain persistent across sweeps,
- they should not require full RAM residency.

This stage is the concrete implementation of `L(v) = v`.

---

### 6.5 Multi-sweep propagation

Method:
- `stream_multi_sweep_blocks(...)`

Responsibilities:
- iterate over the block list repeatedly,
- load block data,
- read current labels,
- compute new labels,
- write updates immediately,
- count changes,
- decide whether another sweep is needed.

---

## 7. How vertices are actually selected for processing

This is one of the most important implementation questions.

### 7.1 Classical LPA

In the classical asynchronous description, vertices are processed in a random order in each iteration.

### 7.2 LPKit streaming mode

In LPKit streaming mode, vertices are not selected by an explicit vertex permutation. They are selected **implicitly** by the sorted edge order.

The effective schedule is:

```text
for each sweep:
    for each block in block_paths:
        process vertices represented by edges in that block
```
Because edges are sorted by source vertex id, the effective order is approximately ascending source-vertex order, block by block.

That means:
- the implementation is deterministic under fixed seed, fixed block size, fixed worker count, and fixed tie-breaking;
- the implementation is still asynchronous, because a label updated earlier in a sweep can influence later updates within that same sweep;
- the implementation is not “random-order LPA.”

### 7.3 Why this matters
This explains several observed properties:
- changing block_size can change the final partition,
- changing worker count can change outcomes,
- deterministic settings are required for reproducible comparisons,
- streaming mode may reach a different valid local equilibrium than a random-order RAM implementation.

## 8. Correctness model

“Correctness” for LPA cannot mean “there is only one correct partition.” That is not how the algorithm works.

### 8.1 What correctness means here

In this project, correctness means:
- unique initial labels are assigned correctly,
- neighborhood-majority updates are computed correctly,
- labels are updated asynchronously during a sweep,
- stopping conditions are applied consistently,
- output labels define a valid partition of the graph,
- streaming mode preserves the semantic spirit of LPA while using a different scheduling discipline.

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
- fixed `block_size`,
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
- run the full streaming pipeline,
- return metadata needed by callers.
- For Python integration, this is the intended one-line public API.

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
- one in-memory block of edges: approximately `Θ(block_size)`
- memmap access window over labels
- small temporary counting buffers

Peak resident memory therefore depends primarily on:
- `block_size`,
- OS page-cache behavior,
- degree distribution,
- implementation overhead in local counting.

### 12.2 Disk

Streaming mode may materialize:
- raw edgelist,
- sorted symmetric edgelist,
- block directory,
- labels memmap.

Disk usage may exceed several multiples of the raw graph size.

## 13. Analysis of problematic subparts

This section answers the practical “where can this become slow or problematic?” question.

### 13.1 Sorting

Sorting is expensive because:

- it touches all edges,
- it is `Θ(m log m)`,
- it is I/O-heavy for large datasets.

However, once sorting is complete, it is no longer central to the propagation logic. The algorithmic behavior of LPKit after preprocessing depends on the **sorted layout**, not on how the sort was internally implemented.

### 13.2 Very small block sizes

Too-small blocks cause:

- too many files,
- frequent block transitions,
- more filesystem overhead,
- weaker throughput.

This increases wall-clock time without improving algorithmic correctness.

### 13.3 Very large block sizes

Very large blocks:

- reduce file-count overhead,
- but increase resident memory pressure,
- and make MemoryMax failures more likely.

### 13.4 High-degree vertices

Vertices with huge neighborhoods are expensive because their update step must count many neighboring labels.

These vertices can dominate local runtime even if the rest of the graph is sparse.

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

- `bs` -> block size used in this run
- `n` -> number of vertices
- `communities` -> number of distinct final labels
- `sweeps` -> number of completed sweeps
- `converged` -> whether the stopping criterion was satisfied before budget exhaustion
- `time` -> measured execution time reported by the streaming pipeline

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