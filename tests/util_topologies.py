"""Helpers for generating small structured graphs used in topology tests."""

import random
from itertools import product

#deg4 per vertex, grid like graph
def write_grid_graph(path, side_len: int, seed: int = 0):
    """Write a bidirectional 2D grid graph edgelist.

    Returns
    -------
    tuple[int, int]
        (number_of_vertices, number_of_undirected_edges)
    """

    random.seed(seed)
    n = side_len **2
    edges = []

    for x, y in product(range(side_len), repeat=2):
        u = x * side_len + y

        if y + 1 < side_len:
            v = x * side_len + (y + 1)
            edges.append((u, v))

        if x + 1 < side_len:
            v = (x + 1) * side_len + y
            edges.append((u, v))

    with open(path, "w") as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")
            f.write(f"{v} {u}\n")

    return n, len(edges) #num of undir edges


def write_clusters_graph(
        path,
        k_clusters: int,
        cluster_size: int,
        p_in: float = 0.8,
        p_out: float = 0.06,
        seed: int = 0,
):
    """Write a clustered Erdős–Rényi-like graph as a bidirectional edgelist.

    - Intra-cluster edges appear with probability `p_in`.
    - Inter-cluster edges appear with probability `p_out`.

    Returns
    -------
    tuple[int, int]
        (number_of_vertices, number_of_undirected_edges)
    """

    random.seed(seed)
    n = k_clusters * cluster_size
    edges = []
    #precompute cluster boundaries
    cluster_of = lambda v: v // cluster_size

    for u in range(n):
        for v in range(u + 1, n):
            same_cluster = cluster_of(u) == cluster_of(v)
            p = p_in if same_cluster else p_out
            if random.random() < p:
                edges.append((u, v))

    with open(path, "w") as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")
            f.write(f"{v} {u}\n")

    return n, len(edges)
