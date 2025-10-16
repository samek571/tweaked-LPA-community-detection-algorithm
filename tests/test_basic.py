from lpkit.label_propagation import label_propagation

def num_communities(labels):
    return len(set(labels))

def test_disconnected_components_2_P3():
    # 0-1-2 and 3-4-5 (two components of P_3)
    neighbors = [
        [1],
        [0,2],
        [1],
        [4],
        [3,5],
        [4],
    ]
    labels, info = label_propagation(neighbors, seed=1, min_sweeps=1)
    assert info["n"] == 6
    assert info["m"] == 4
    assert num_communities(labels) == 2

def test_star_graph_single_community():
    k = 7
    neighbors = [[] for _ in range(k+1)]
    for leaf in range(1, k+1):
        neighbors[0].append(leaf)
        neighbors[leaf].append(0)
    labels, info = label_propagation(neighbors, seed=1337)
    assert num_communities(labels) == 1

def test_reproducibility_given_arbitrary_seed():
    neighbors = [
        [1],
        [0,2],
        [1,3],
        [2]
    ]
    l1, _ = label_propagation(neighbors, seed=27)
    l2, _ = label_propagation(neighbors, seed=27)
    assert l1 == l2

def test_isolated_vertices():
    n = 5
    neighbors = [[] for _ in range(n)]
    labels, info = label_propagation(neighbors, seed=7)
    assert len(set(labels)) == n
    assert info["m"] == 0