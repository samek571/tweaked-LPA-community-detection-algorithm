import os
import random
from typing import Literal


def generate_large_graph(
        out_path: str,
        *,
        n: int,
        m: int,
        # TODO other topos for future work possibly, just so i dont have to refactor
        topology: Literal["random", "grid", "clusters"] = "random",
        seed: int = 0,
        self_loops: bool = False,
    ) -> None:

    rng = random.Random(seed)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w") as f:
        if topology == "random":
            for _ in range(m):
                u = rng.randrange(n)
                v = rng.randrange(n)
                if not self_loops and u == v: continue
                f.write(f"{u} {v}\n")

        #sqrt(n) x sqrt(n) 2D grid
        elif topology == "grid":
            side = int(n ** 0.5)
            for i in range(side):
                for j in range(side):
                    u = i*side +j
                    if j+1 < side:
                        f.write(f"{u} {u+1}\n")
                    if i+1 < side:
                        f.write(f"{u} {u+side}\n")

        elif topology == "clusters":
            #k clusters of size c each
            #dense inside, sparse otherwise
            k = max(1, int(n**0.5))
            c = n // k
            for cluster in range(k):
                start = cluster*c
                end = min(start+c, n)

                #some internal clique
                for i in range(start, end):
                    for j in range(i+1, end):
                        f.write(f"{i} {j}\n")

                #sparse links
                if cluster < k - 1:
                    for _ in range(2):
                        u = rng.randrange(start, end)
                        v = rng.randrange(end, min(end+c, n))
                        f.write(f"{u} {v}\n")
        else:
            raise ValueError(f"Unknown topology: {topology}")

    print(f"Enscribed {out_path} with up to {m} edges among {n} vertices")
