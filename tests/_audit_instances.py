import random
import igraph as ig

def random_cover(rng: random.Random, n: int, k: int, max_labels: int) -> list[list[int]]:
    """A valid cover: contiguous labels 0..k-1, each vertex 1..max_labels labels."""
    memberships = [sorted(rng.sample(range(k), rng.randint(1, max_labels))) for _ in range(n)]
    for label in range(k):
        if not any(label in row for row in memberships):
            memberships[rng.randrange(n)] = sorted(set(memberships[rng.randrange(n)]) | {label})
    return memberships


def random_instance(seed: int) -> tuple[ig.Graph, list[list[int]]]:
    rng = random.Random(seed)
    n = rng.randint(6, 14)
    p = rng.choice([0.2, 0.4, 0.7])
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    graph = ig.Graph(n=n, edges=edges)
    k = rng.randint(2, 4)
    return graph, random_cover(rng, n, k, min(k, rng.choice([1, 2, 3])))


def many_label_instance(seed: int, communities: int = 300) -> tuple[ig.Graph, list[list[int]]]:
    """More communities than vertices can see: most candidate labels have zero support at every vertex.

    300 communities is above the 256 below which ``rb._candidate_labels`` lists every label, so this exercises the
    sparse branch (own, neighbour and ``cap + |labels| + 1`` lightest labels) as well.
    """
    rng = random.Random(seed)
    n = communities + 10
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < 0.03]
    memberships = []
    for v in range(n):
        labels = {v % communities} | {rng.randrange(communities) for _ in range(rng.choice([0, 0, 1, 2]))}
        memberships.append(sorted(labels))
    return ig.Graph(n=n, edges=edges), memberships
