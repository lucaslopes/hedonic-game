"""Bit-level fingerprint of the installed hedonic detector on a fixed grid of calls.

    python bitcheck.py [--covers] [--records FILE]

Prints one line per grid, ``partitions <calls> <sha256>`` (and with --covers
``covers <calls> <sha256>``), over the memberships, the quality's exact bits
(``float.hex``) and the equilibrium slack of every call, or the error message
of a call that is refused. Two builds that print the same line agree bit for
bit on every call of that grid. --records writes one JSON line per call, so
``diff`` of two record files names the first call that differs.

The partition grid is 26 graphs (Zachary, Petersen and 24 seeded planted
partitions of 60-200 vertices) x 6 resolutions x 2 isolation policies x 6
option sets (none, edge weights, node weights, at most 5 communities, exactly
3, a random start in 4 labels) x 3 seeds = 5616 calls. Do not change it: its
hash is recorded in SKILL.md. Add a new grid instead.

--covers (hedonic 2 only) runs the same graphs at max_memberships 2 and 4.
Runs in seconds; it tests the installed hedonic and the core it loads: the
package's hedonic._core, or with HEDONIC_CORE=<name> hedonic._core_<name>
(check the module, backend, path and source digest it prints). hedonic 2.0's
C core and 2.1's Rust core print the same lines.
"""

import argparse
import hashlib
import json
import random
import sys

import igraph as ig

import hedonic
from hedonic import Game


def graphs():
    out = [ig.Graph.Famous("Zachary"), ig.Graph.Famous("Petersen")]
    random.seed(7)
    for _ in range(24):
        n = random.choice([60, 120, 200])
        k = random.choice([2, 3, 5])
        edges = [(a, b) for a in range(n) for b in range(a + 1, n)
                 if random.random() < (0.25 if a % k == b % k else 0.02)]
        out.append(ig.Graph(n=n, edges=edges))
    return out


def grid(name, max_memberships, records):
    h = hashlib.sha256()
    calls = 0
    for gi, g in enumerate(graphs()):
        g = Game(g.simplify())
        if g.ecount() == 0:
            continue
        w = [random.uniform(0.5, 2) for _ in range(g.ecount())]
        nw = [random.uniform(0.5, 2) for _ in range(g.vcount())]
        for res in (None, 0.0, 0.05, 0.5, 1.0, -0.01):
            for iso in (False, True):
                for kw in ({}, {"edge_weights": w}, {"node_weights": nw}, {"max_total_communities": 5},
                           {"n_communities": 3}, {"max_communities": 4}):
                    if max_memberships is not None:
                        kw = {**kw, "max_memberships": max_memberships}
                    for seed in (0, 1, 2):
                        try:
                            p = g.community_hedonic(resolution=res, allow_isolation=iso, seed=seed, **kw)
                            members = p.membership
                            if max_memberships is not None:
                                members = [tuple(row) for row in members]
                            out = (tuple(members), p._hedonic_native_quality.hex(),
                                   repr(p._hedonic_provenance["equilibrium_slack"]))
                        except ValueError as e:
                            out = ("error", str(e))
                        # The partition grid's hash predates max_memberships in kw: keep its key as it was.
                        key = (gi, res, iso, sorted(k for k in kw if k != "max_memberships"), seed)
                        if max_memberships is not None:
                            key += (max_memberships,)
                        h.update(repr(key + (out,)).encode())
                        calls += 1
                        if records is not None:
                            records.write(json.dumps({"grid": name, "key": repr(key), "out": repr(out)}) + "\n")
    return calls, h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--covers", action="store_true", help="also the cover grid (hedonic 2)")
    parser.add_argument("--records", help="write one JSON line per call to this file")
    args = parser.parse_args(argv)
    from hedonic import _core
    print(f"# hedonic {hedonic.__version__}, core {_core.__spec__.name} (backend {getattr(_core, 'backend', 'c')}) "
          f"from {_core.__file__}", file=sys.stderr)
    print(f"# core source sha256 {getattr(_core, 'source_sha256', '?')}", file=sys.stderr)
    records = open(args.records, "w") if args.records else None
    try:
        print("partitions", *grid("partitions", None, records))
        if args.covers:
            for m in (2, 4):
                print(f"covers-M{m}", *grid(f"covers-M{m}", m, records))
    finally:
        if records is not None:
            records.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
