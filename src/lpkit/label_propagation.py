"""In-memory reference Label Propagation Algorithm (LPA).

This implementation is intentionally simple and readable. It serves two roles:
- a baseline for correctness tests against the streaming implementation
- a usable RAM-mode algorithm for small/medium graphs

The graph is expected to be represented as an undirected adjacency list
(`u->v` and `v->u` both present).
"""

from collections import defaultdict
from random import Random
from typing import List, Sequence, Dict, Optional, Any, NamedTuple

Label = int
Adjacency_list = Sequence[Sequence[int]]

#just a custom DS
class LPAResult(NamedTuple):
    """Return type for in-memory LPA.

    Attributes
    ----------
    labels:
        Final label assignment per vertex index.
    info:
        Metadata describing convergence and run parameters.
    """
    labels: List[int]
    info: Dict[str, Any]


#async label propagation as Raghavan–Albert–Kumara (Phys. Rev. E 76, 036106) suggested (see paper)
#time complexity: O(|E| * number_of_sweeps)
#space complexity: O(|E| + |V|)
def label_propagation(
        neighbors: Adjacency_list, #a->b and b->a should exist
        *,
        seed: Optional[int] = 0,
        max_sweeps: int = 100, #hard upper limit 
        min_sweeps: int = 1, #min num of sweeps before stoping is enabled
        verify_each_sweep: bool = True,
        shuffle_each_sweep: bool = True
    ) -> LPAResult:
    """Run in-memory asynchronous label propagation.

    Parameters
    ----------
    neighbors:
        Adjacency list of an undirected graph.
    seed:
        RNG seed controlling vertex-order shuffles and random tie-breaking.
    max_sweeps / min_sweeps:
        Hard cap and lower bound on the number of sweeps.
    verify_each_sweep:
        If True, performs an explicit stability verification pass after each
        sweep. This adds work but gives a stricter convergence signal.
    shuffle_each_sweep:
        If True, updates vertices in a new random order every sweep.

    Returns
    -------
    LPAResult
        `(labels, info)` where `labels` is a list and `info` is metadata.
    """
    n = len(neighbors)
    labels = list(range(n))
    rng = Random(seed) #deterministic tie breaking, local rng
    order = list(range(n)) #vertex update hierarchy

    '''picking best label for v'''
    def _best_label_helper(v: int) -> Label:
        """Return the most frequent neighbor label for `v` (randomly breaking ties)."""
        cnts: Dict[Label, int] = defaultdict(int)

        for u in neighbors[v]:
            cnts[labels[u]] += 1

        if not cnts:
            return labels[v]

        _max_val = max(cnts.values())
        return rng.choice([lab for lab, c in cnts.items() if c == _max_val]) #randonly break ties

    '''propagation loop'''
    sweeps, changed_any = 0, True
    while sweeps < max_sweeps and (changed_any or sweeps < min_sweeps):
        sweeps, changed_any = sweeps+1, False

        if shuffle_each_sweep:
            rng.shuffle(order)

        #for each vertex we update label to most frequent neigbor label
        for v in order:
            new_lab = _best_label_helper(v)
            if new_lab != labels[v]:
                labels[v] = new_lab
                changed_any = True

        #convergence check
        if verify_each_sweep:
            #extra cvg verif pass. sweep can make no changes and
            #still not be globally stable under some update orders
            stable = True
            for v in range(n):
                cnts: Dict[Label, int] = defaultdict(int)
                for u in neighbors[v]:
                    cnts[labels[u]] += 1
                if cnts and cnts.get(labels[v], 0) < max(cnts.values()):
                    stable = False
                    break

            if stable and sweeps >= min_sweeps:
                changed_any = False

    m2 = sum(len(neighbors[v]) for v in range(n))
    info = {
        "sweeps": sweeps,
        "converged": int(not changed_any), # we will need this later to do some analysis
        #it couldhabe stayed T/F but its better for sorting and running prebuilt libs
        "n": n,
        "m": m2 // 2,  #undirected graph doesnt need to keep A->B and B->A
        "seed": seed if seed is not None else -1,
    }

    return LPAResult(labels, info)

