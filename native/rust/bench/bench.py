"""The Rust core against the C core: the same calls, interleaved, timed.

Each case runs detect() through both extensions in alternation, keeps the
fastest of --reps runs of each, and checks that the results are identical.
Ratio = C time / Rust time (above 1: Rust is faster). The Rust core is
hedonic._core, the C core hedonic._core_c built beside it (HEDONIC_CORE unset).

    python native/rust/bench/bench.py [--reps 7] [--filter NAME] [--quick]
"""

import argparse
import os
import random
import statistics
import time
from array import array

import igraph

if os.environ.get("HEDONIC_CORE"):
    raise SystemExit("bench.py times hedonic._core against hedonic._core_c: unset HEDONIC_CORE")
from hedonic import _core as rs

try:
    import hedonic._core_c as c
except ModuleNotFoundError:
    raise SystemExit("bench.py needs the C core as hedonic._core_c, built beside hedonic._core") from None


def sbm(n, blocks, p_in, p_out, seed):
    random.seed(seed)
    size = n // blocks
    pref = [[p_in if i == j else p_out for j in range(blocks)] for i in range(blocks)]
    return igraph.Graph.SBM(pref_matrix=pref, block_sizes=[size] * blocks)


def cases(quick):
    random.seed(1)
    ba50 = igraph.Graph.Barabasi(50_000, 3)
    sbm20 = sbm(20_000, 40, 0.01, 0.0002, 2)
    er = igraph.Graph.Erdos_Renyi(n=100_000, m=500_000)
    random.seed(3)
    wts = array("d", [random.uniform(0.1, 2.0) for _ in range(sbm20.ecount())])
    density = lambda g: 2 * g.ecount() / (g.vcount() * (g.vcount() - 1))
    out = [
        ("zachary x200 partition", igraph.Graph.Famous("Zachary"), 1, 78 / 561, None, 200),
        ("zachary x200 cover M=3", igraph.Graph.Famous("Zachary"), 3, 0.1, None, 200),
        ("BA 50k partition", ba50, 1, density(ba50), None, 1),
        ("BA 50k cover M=3", ba50, 3, 4 * density(ba50), None, 1),
        ("SBM 20k partition", sbm20, 1, density(sbm20), None, 1),
        ("SBM 20k cover M=3", sbm20, 3, 4 * density(sbm20), None, 1),
        ("SBM 20k weighted partition", sbm20, 1, density(sbm20), wts, 1),
        ("SBM 20k weighted cover M=2", sbm20, 2, 4 * density(sbm20), wts, 1),
        ("ER 100k/500k partition", er, 1, density(er), None, 1),
        ("ER 100k/500k cover M=2", er, 2, 2 * density(er), None, 1),
    ]
    if not quick:
        ba300 = igraph.Graph.Barabasi(300_000, 5)
        out += [
            ("BA 300k partition", ba300, 1, density(ba300), None, 1),
            ("BA 300k cover M=4", ba300, 4, 4 * density(ba300), None, 1),
            ("SBM 20k cover M=-1 (cap n)", sbm20, sbm20.vcount(), 4 * density(sbm20), None, 1),
        ]
    return out


def run(module, g, cap, gamma, w, times):
    out = None
    for t in range(times):
        out = module.detect(g, random.Random(t), gamma, max_memberships=cap, edge_weights=w)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--filter", default="")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    ratios = []
    print(f"{'case':32s} {'C ms':>9s} {'Rust ms':>9s} {'C/Rust':>7s}")
    for name, g, cap, gamma, w, times in cases(a.quick):
        if a.filter not in name:
            continue
        tc, tr = [], []
        for _ in range(a.reps):
            s = time.perf_counter(); oc = run(c, g, cap, gamma, w, times); tc.append(time.perf_counter() - s)
            s = time.perf_counter(); orr = run(rs, g, cap, gamma, w, times); tr.append(time.perf_counter() - s)
        if repr(oc) != repr(orr):
            raise SystemExit(f"{name}: results differ")
        ratio = min(tc) / min(tr)
        ratios.append(ratio)
        print(f"{name:32s} {min(tc)*1e3:9.2f} {min(tr)*1e3:9.2f} {ratio:7.2f}", flush=True)
    print(f"{'geometric mean':32s} {'':9s} {'':9s} {statistics.geometric_mean(ratios):7.2f}")


if __name__ == "__main__":
    main()
