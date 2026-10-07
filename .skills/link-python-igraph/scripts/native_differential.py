"""Seeded Python-level differential for Graph.community_leiden.

Runs a deterministic battery of small instances through the *installed*
python-igraph binding and writes one JSON line per instance (membership rows
and quality, or the error). Run it once per environment -- for example a
baseline build and a development build, or two snapshot libraries selected
with DYLD_LIBRARY_PATH / LD_LIBRARY_PATH -- and compare the two files.

The battery covers unweighted, integer and real edge weights; unit, integer
(including zero) and real node weights; isolated vertices; negative, zero and
positive resolutions; isolation on and off; local-only and multilevel runs;
negative, zero and positive iteration budgets; singleton and random starts;
disjoint (M = 1) and overlapping caps up to n. It does not draw directed
graphs, signed weights or count limits: use leiden_equivalence.c for those.

Every instance runs in a forked child with a wall-clock limit, so a
non-terminating call is recorded as a timeout instead of hanging the battery
(igraph's interruption hook is polled too rarely to stop small-graph loops).

Usage::

    python native_differential.py run --count 6000 --seed 0 --out base.jsonl
    DYLD_LIBRARY_PATH=/path/to/dev/lib python native_differential.py run \\
        --count 6000 --seed 0 --out dev.jsonl
    python native_differential.py compare base.jsonl dev.jsonl --seed 0

The instance generator is frozen: a given (count, seed) always produces the
same battery, so files from different sessions stay comparable.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import select
import signal
import sys


def make_instance(rng: random.Random, index: int) -> dict:
    n = rng.randint(2, 28)
    kind = rng.choice(["er", "er", "er", "star", "path", "complete", "two_blocks"])
    edges: set[tuple[int, int]] = set()
    if kind == "er":
        p = rng.choice([0.08, 0.15, 0.3, 0.6])
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
    elif kind == "star":
        for v in range(1, n):
            edges.add((0, v))
    elif kind == "path":
        for v in range(1, n):
            edges.add((v - 1, v))
    elif kind == "complete":
        n = min(n, 9)
        edges = {(u, v) for u in range(n) for v in range(u + 1, n)}
    else:
        half = max(1, n // 2)
        for u in range(n):
            for v in range(u + 1, n):
                same = (u < half) == (v < half)
                if rng.random() < (0.6 if same else 0.05):
                    edges.add((u, v))
    # Isolated vertices appended after the structure.
    n += rng.choice([0, 0, 0, 1, 3])
    if not edges:
        edges.add((0, 1))
    edges_list = sorted(edges)
    m = len(edges_list)
    weight_kind = rng.choice(["none"] * 6 + ["int", "real"])
    if weight_kind == "int":
        weights = [float(rng.randint(1, 3)) for _ in range(m)]
    elif weight_kind == "real":
        weights = [round(rng.uniform(0.1, 2.0), 6) for _ in range(m)]
    else:
        weights = None
    node_kind = rng.choice(["none"] * 6 + ["int", "real"])
    if node_kind == "int":
        node_weights = [float(rng.randint(0, 3)) for _ in range(n)]
    elif node_kind == "real":
        node_weights = [round(rng.uniform(0.0, 2.0), 6) for _ in range(n)]
    else:
        node_weights = None
    M = rng.choice([1, 2, 2, 3, 3, 4, n])
    resolution = rng.choice([-0.5, -0.05, 0.0, 0.01, 0.05, 0.1, 0.3, 1.0])
    start = rng.choice(["singleton", "singleton", "random"])
    initial = None
    if start == "random":
        k = rng.randint(1, n)
        if M == 1:
            labels = [rng.randrange(k) for _ in range(n)]
            remap = {c: i for i, c in enumerate(sorted(set(labels)))}
            initial = [remap[c] for c in labels]
        else:
            rows = []
            for _ in range(n):
                size = rng.randint(1, min(M, k))
                rows.append(sorted(rng.sample(range(k), size)))
            used = sorted({c for row in rows for c in row})
            remap = {c: i for i, c in enumerate(used)}
            initial = [[remap[c] for c in row] for row in rows]
    return {
        "index": index,
        "kind": kind,
        "n": n,
        "edges": edges_list,
        "weights": weights,
        "node_weights": node_weights,
        "max_memberships": M,
        "resolution": resolution,
        "beta": rng.choice([0.0, 0.01, 0.1]),
        "allow_isolation": rng.random() < 0.5,
        "local_move_only": rng.random() < 0.5,
        "n_iterations": rng.choice([-1, -1, -1, 0, 1, 3]),
        "initial": initial,
        "rng_seed": rng.randrange(2**31),
    }


def run_instance(inst: dict) -> dict:
    import igraph as ig

    graph = ig.Graph(n=inst["n"], edges=inst["edges"])
    ig.set_random_number_generator(random.Random(inst["rng_seed"]))
    kwargs = dict(
        objective_function="CPM",
        weights=inst["weights"],
        resolution=inst["resolution"],
        beta=inst["beta"],
        max_memberships=inst["max_memberships"],
        initial_membership=inst["initial"],
        n_iterations=inst["n_iterations"],
        allow_isolation=inst["allow_isolation"],
        local_move_only=inst["local_move_only"],
        node_weights=inst["node_weights"],
    )
    try:
        result = graph.community_leiden(**kwargs)
    except Exception as exc:  # the error class and message are compared too
        return {"index": inst["index"], "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if inst["max_memberships"] == 1:
        rows = [[int(c)] for c in result.membership]
    else:
        rows = [sorted(int(c) for c in r) for r in result.membership]
    quality = result._params.get("quality") if hasattr(result, "_params") else None
    if quality is not None and not math.isfinite(quality):
        quality = repr(quality)
    return {"index": inst["index"], "status": "ok", "rows": rows, "quality": quality}


def run_isolated(inst: dict, timeout: float) -> dict:
    """Runs one instance in a forked child; a timeout becomes a status."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            payload = json.dumps(run_instance(inst)).encode()
        except BaseException as exc:  # noqa: BLE001 - report anything
            payload = json.dumps({"index": inst["index"], "status": "error",
                                  "error": f"harness: {type(exc).__name__}: {exc}"}).encode()
        with os.fdopen(write_fd, "wb") as fh:
            fh.write(payload)
        os._exit(0)
    os.close(write_fd)
    chunks = []
    deadline_ok = True
    with os.fdopen(read_fd, "rb") as fh:
        while True:
            ready, _, _ = select.select([fh], [], [], timeout)
            if not ready:
                deadline_ok = False
                break
            chunk = os.read(fh.fileno(), 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
    if not deadline_ok:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        return {"index": inst["index"], "status": "error", "error": f"timeout after {timeout:g} s"}
    _, status = os.waitpid(pid, 0)
    if not chunks:
        return {"index": inst["index"], "status": "error", "error": f"child exit status {status}"}
    return json.loads(b"".join(chunks))


def canonical(rows: list[list[int]]) -> list[tuple[int, ...]]:
    bodies: dict[int, list[int]] = {}
    for v, labels in enumerate(rows):
        for c in labels:
            bodies.setdefault(c, []).append(v)
    return sorted(tuple(b) for b in bodies.values())


def instance_class(inst: dict) -> str:
    budget = inst["n_iterations"]
    return "/".join((
        "M=1" if inst["max_memberships"] == 1 else "M>1",
        "local" if inst["local_move_only"] else "multilevel",
        "budget<0" if budget < 0 else ("budget=0" if budget == 0 else "budget>0"),
    ))


def classify(ra: dict, rb: dict) -> str:
    if ra["status"] != rb["status"] or ra.get("error") != rb.get("error"):
        return "status"
    if ra["status"] == "error" or (ra["rows"] == rb["rows"] and ra["quality"] == rb["quality"]):
        return "exact"
    if ra["rows"] == rb["rows"]:
        return "quality_only"
    if canonical(ra["rows"]) == canonical(rb["rows"]):
        return "renamed"
    return "different"


def compare(path_a: str, path_b: str, seed: int | None) -> int:
    a = {r["index"]: r for r in map(json.loads, open(path_a))}
    b = {r["index"]: r for r in map(json.loads, open(path_b))}
    instances = None
    if seed is not None:
        rng = random.Random(seed)
        instances = [make_instance(rng, i) for i in range(max(a) + 1)]
    kinds: collections.Counter[str] = collections.Counter()
    by_class: dict[str, collections.Counter[str]] = {}
    examples = []
    for i in sorted(a):
        kind = classify(a[i], b[i])
        kinds[kind] += 1
        if kind != "exact":
            if instances is not None:
                by_class.setdefault(instance_class(instances[i]), collections.Counter())[kind] += 1
            if len(examples) < 20:
                examples.append((i, kind, a[i].get("error") or a[i].get("quality"),
                                 b[i].get("error") or b[i].get("quality")))
    print(json.dumps({"instances": len(a), "kinds": dict(kinds),
                      "non_exact_by_class": {k: dict(v) for k, v in sorted(by_class.items())}},
                     indent=1))
    for example in examples:
        print(example)
    return 0 if kinds["exact"] == len(a) else 1


def identity() -> dict:
    import igraph as ig
    return {"python": sys.executable, "igraph": ig.__version__,
            "igraph_c": getattr(ig._igraph, "__igraph_version__", None), "path": ig.__file__}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a battery and write JSON lines")
    run.add_argument("--count", type=int, default=2000)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", required=True)
    run.add_argument("--timeout", type=float, default=20.0, help="seconds per instance")
    cmp_ = sub.add_parser("compare", help="compare two battery files")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    cmp_.add_argument("--seed", type=int, help="battery seed, to classify by instance class")
    dump = sub.add_parser("dump", help="print one instance as JSON")
    dump.add_argument("index", type=int)
    dump.add_argument("--seed", type=int, default=0)
    sub.add_parser("identity", help="print the igraph build this interpreter loads")
    args = ap.parse_args()

    if args.command == "compare":
        return compare(args.a, args.b, args.seed)
    if args.command == "identity":
        print(json.dumps(identity(), indent=1))
        return 0
    if args.command == "dump":
        rng = random.Random(args.seed)
        print(json.dumps([make_instance(rng, i) for i in range(args.index + 1)][-1]))
        return 0
    rng = random.Random(args.seed)
    instances = [make_instance(rng, i) for i in range(args.count)]
    print(json.dumps(identity()), file=sys.stderr)
    with open(args.out, "w") as fh:
        for inst in instances:
            fh.write(json.dumps(run_isolated(inst, args.timeout)) + "\n")
            fh.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
