"""Bad arguments through the public API raise the same errors on both cores.

A table of bad inputs to Game.community_hedonic, Game.stable_intervals and
Game.regrets runs once in a process on the C core and once in a process on
the Rust core (HEDONIC_CORE=rust); every row must end the same way: the same
exception type and message, or (for the rows that are fine after all) the
same result. Whatever Python checks before the core is the same code on both;
the rows are there for what reaches the core.

    python -m pytest -q rust/tests/test_error_parity.py
"""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("igraph")
pytest.importorskip("numpy")
pytest.importorskip("hedonic_core_rs")

TABLE = r'''
import json, math, random, sys
from array import array
import igraph, numpy as np
from hedonic import Game, _core

nan, inf = float("nan"), float("inf")
g = Game(igraph.Graph.Famous("Zachary"))
n, m = g.vcount(), g.ecount()
rows = [[v % 3] for v in range(n)]
cover = [[v % 3, (v + 1) % 3] for v in range(n)]

class Odd(random.Random):
    def getrandbits(self, k):
        raise ZeroDivisionError("no bits")

detect = [
    {"resolution": nan}, {"resolution": inf}, {"resolution": -0.5}, {"resolution": "x"},
    {"resolution": 2.0}, {"max_memberships": 0}, {"max_memberships": -2}, {"max_memberships": 1.5},
    {"max_memberships": "a"}, {"max_memberships": True}, {"max_memberships": 2**70},
    {"initial_membership": [0] * (n - 1)}, {"initial_membership": [-1] + [0] * (n - 1)},
    {"initial_membership": [0.5] * n}, {"initial_membership": ["a"] * n},
    {"initial_membership": [[0], []] + [[0]] * (n - 2), "max_memberships": 2},
    {"initial_membership": [[0, 1, 2]] * n, "max_memberships": 2},
    {"initial_membership": [2**70] * n}, {"initial_membership": [None] * n},
    {"initial_membership": 5},
    {"edge_weights": [1.0] * (m - 1)}, {"edge_weights": [-1.0] * m}, {"edge_weights": [nan] * m},
    {"edge_weights": [inf] * m}, {"edge_weights": [0.0] * m}, {"edge_weights": "missing"},
    {"edge_weights": ["a"] * m}, {"edge_weights": [1e308] * m},
    {"node_weights": [1.0] * (n - 1)}, {"node_weights": [-1.0] * n}, {"node_weights": [nan] * n},
    {"node_weights": [0.0] * n}, {"node_weights": [inf] * n},
    {"seed": "x"}, {"seed": -1}, {"seed": 1.5},
    {"max_total_communities": 0}, {"max_total_communities": -1}, {"max_total_communities": 1.5},
    {"n_communities": 0}, {"n_communities": n + 1}, {"n_communities": 5, "max_total_communities": 2},
    {"max_communities": 0}, {"max_communities": n + 5}, {"max_communities": 3, "initial_membership": [0] * n},
    {"allow_isolation": "maybe"}, {"allow_isolation": None},
    {"max_memberships": 2, "max_total_communities": 1}, {"max_memberships": 2, "n_communities": 0},
    {"max_memberships": -1, "edge_weights": [nan] * m},
    {"unknown_option": 1}, {"positional": (1,)},
]
audit = [
    {"memberships": rows[:-1]}, {"memberships": [[-1]] + rows[1:]}, {"memberships": [[]] + rows[1:]},
    {"memberships": [[0.5]] * n}, {"memberships": [["a"]] * n}, {"memberships": 7},
    {"memberships": cover, "max_memberships": 1}, {"memberships": cover, "max_memberships": 0},
    {"memberships": cover, "max_memberships": -3}, {"memberships": cover, "max_memberships": 1.5},
    {"memberships": rows, "edge_weights": [nan] * m}, {"memberships": rows, "edge_weights": [-1.0] * m},
    {"memberships": rows, "edge_weights": [1.0] * (m + 1)}, {"memberships": rows, "edge_weights": "missing"},
    {"memberships": rows, "atol": -1.0}, {"memberships": rows, "rtol": nan}, {"memberships": rows, "atol": "x"},
    {"memberships": rows, "atol": None}, {"memberships": rows, "rtol": 2**2000},
    {"memberships": rows, "vertices": [n]}, {"memberships": rows, "vertices": [-1]},
    {"memberships": rows, "vertices": [0.5]}, {"memberships": rows, "vertices": "all"},
    {"memberships": rows, "allow_isolation": "maybe"},
    {"memberships": [[2**70]] * n},
]
grids = [[-0.1], [1.5], [nan], [inf], [[0.1, 0.2]], [], ["a"], 0.5, None, np.zeros((2, 2))]

def outcome(fn, *args, **kw):
    try:
        out = fn(*args, **kw)
    except BaseException as e:
        return [type(e).__module__ + "." + type(e).__qualname__, str(e)]
    if hasattr(out, "membership"):
        out = list(out.membership)
    elif hasattr(out, "regret"):
        out = [np.asarray(out.regret).tolist(), np.asarray(out.tolerance).tolist()]
    return ["ok", repr(out)]

def detect_call(kw):
    kw = dict(kw)
    pos = kw.pop("positional", ())
    if "seed" not in kw:
        kw["seed"] = 1
    return outcome(g.community_hedonic, *pos, **kw)

result = {}
for i, kw in enumerate(detect):
    result[f"community_hedonic {kw!r}"[:200] + f" #{i}"] = detect_call(kw)
for i, kw in enumerate(audit):
    kw = dict(kw)
    mem = kw.pop("memberships")
    kw.setdefault("max_memberships", 2)
    result[f"stable_intervals {i}"] = outcome(g.stable_intervals, mem, **kw)
    result[f"regrets {i}"] = outcome(g.regrets, [0.0, 0.1, 0.5], mem, **kw)
for i, grid in enumerate(grids):
    result[f"regrets grid {i}"] = outcome(g.regrets, grid, cover, max_memberships=2)
result["directed"] = outcome(Game(igraph.Graph(n=3, edges=[(0, 1)], directed=True)).community_hedonic, seed=1)
result["self loop"] = outcome(Game(igraph.Graph(n=3, edges=[(0, 0), (0, 1)])).community_hedonic, seed=1)
result["empty"] = outcome(Game(igraph.Graph(n=0)).community_hedonic, seed=1)
result["no edges cover"] = outcome(Game(igraph.Graph(n=3)).community_hedonic, max_memberships=2, seed=1)
result["no edges audit"] = outcome(Game(igraph.Graph(n=3)).stable_intervals, [[0], [1], [2]])
result["generator fails"] = outcome(_core.detect, g, Odd(0), 0.1)
json.dump(result, sys.stdout)
'''


def run(core):
    env = dict(os.environ)
    env.pop("HEDONIC_CORE", None)
    if core == "rust":
        env["HEDONIC_CORE"] = "rust"
    got = subprocess.run([sys.executable, "-c", TABLE], env=env, capture_output=True, text=True, timeout=600)
    assert got.returncode == 0, got.stderr[-2000:]
    return json.loads(got.stdout)


def test_errors_match_c_core():
    env = dict(os.environ)
    env.pop("HEDONIC_CORE", None)
    probe = subprocess.run([sys.executable, "-c", "from hedonic import _core; print(getattr(_core, 'backend', 'c'))"],
                           env=env, capture_output=True, text=True)
    if probe.returncode != 0 or probe.stdout.strip() != "c":
        pytest.skip("needs the C core built as hedonic._core")
    c, rust = run("c"), run("rust")
    assert c.keys() == rust.keys()
    errors = sum(1 for v in c.values() if v and v[0] != "ok")
    assert errors > 60  # the table reaches the errors it is about
    differ = {k: (c[k], rust[k]) for k in c if c[k] != rust[k]}
    assert not differ, json.dumps(differ, indent=1)[:4000]
