"""How many times each overlap-benchmark method runs: the one replication table of the calibration and the benchmark.

A method's ``kind`` says how the implementation the benchmark calls behaves under a seed, read from the adapters and
checked by running them twice (not taken from the papers' claims):

- ``deterministic``: one run per setting is the result. The benchmark runs it once per pair, at the lowest seed of its
  plan (seed 0 in the paper); more seeds would repeat the same cover.
- ``seeded``: the seed reaches the program, so a seed list is a real replication: one run per seed.
- ``replicas``: the program is stochastic but the runner's seed does not reach it, so repeated runs are independent
  replicas that a seed number does not reproduce: one run per seed too (the seed only names the replica), and the
  calibration averages ``--replicates`` runs per setting.

``runs`` is the number of benchmark runs the paper's ten-seed plan makes. ``calibration_kind``, where given, is the kind
under the calibration's own settings when they differ from the benchmark's (no registered method now). ``replication = "all"`` (``--replication
all``) runs every method at every seed, for an audit of determinism; it changes no unit's identity, so the two modes
share their records.
"""

from __future__ import annotations

from typing import Any, Iterable

KINDS = ("deterministic", "seeded", "replicas")
MODES = ("auto", "all")
DEFAULT_MODE = "auto"
UNCLASSIFIED = "seeded"  # a method nobody classified runs at every seed: never drop work on a guess

# Keyed by the benchmark's method keys (``quickstart.BY_KEY``); the calibration uses the baselines' rows.
REPLICATION: dict[str, dict[str, Any]] = {
    "hedonic": {"kind": "seeded", "runs": 10,
                "note": "the seed sets the vertex visiting order and the tie-breaking of local moving"},
    "clique_percolation": {"kind": "deterministic", "runs": 1,
                           "note": "the adapter ignores the seed; two runs on Amazon/top5000 gave identical covers"},
    "angel": {"kind": "seeded", "runs": 10,
              "note": "its label propagation draws from the module-level random; the worker seeds it from the request, so "
                      "a seed reproduces a run (unseeded, two runs on one graph gave 38 and 39 communities)"},
    "ncgame": {"kind": "deterministic", "runs": 1,
               "note": "the authors describe no stochastic process; two runs on Amazon/top5000 gave identical covers"},
    "codeseg": {"kind": "deterministic", "runs": 1,
                "note": "no seed reaches the program; the benchmark and the calibration run it at one thread, where two "
                        "runs on Amazon/top5000 gave identical covers. On several threads its runs differ (2,073 and 2,081 "
                        "communities at all cores), so a run with method_params threads > 1 is a replica, not a repeat"},
    "ego_splitting": {"kind": "seeded", "runs": 10,
                      "note": "the Louvain step of the persona graph takes the seed as random_state"},
    "bigclam": {"kind": "deterministic", "runs": 1,
                "note": "no seed reaches the SNAP program, but repeated runs gave identical covers at one thread (all 95 "
                        "replicated settings of baseline-paper scored identically) and at the benchmark's thread count "
                        "(two runs on Amazon/top5000; ten earlier runs on DBLP and Amazon/top5000)"},
    "fox": {"kind": "deterministic", "runs": 1,
            "note": "no seed is passed; with queue size 1 repeated runs gave identical covers on one thread and at the "
                    "benchmark's thread count (two runs on Amazon/top5000; ten earlier runs on three top5000 pairs)"},
    "neo_kmeans": {"kind": "deterministic", "runs": 1,
                   "note": "the adapter ignores the seed, and repeated runs gave identical covers (two runs on "
                           "Amazon/top5000 at K = 5,348); one run per setting"},
    "disjoint_oracle": {"kind": "deterministic", "runs": 1,
                        "note": "a supervised control built from the reference cover; it reads no seed (two runs on "
                                "Amazon/top5000 gave identical covers)"},
}
RUNNER_ALIASES = {"hedonic_local": "hedonic"}  # the runner's name of a method whose key differs


def check_mode(value: str) -> str:
    """``auto`` or ``all``, else a ValueError that names both."""
    mode = str(value).strip().lower()
    if mode not in MODES:
        raise ValueError(f"replication must be {' or '.join(MODES)}, not {value!r}")
    return mode


def kind(method: str) -> str:
    """The method's replication kind (an unclassified method counts as seeded: it runs at every seed)."""
    info = REPLICATION.get(RUNNER_ALIASES.get(method, method))
    return info["kind"] if info else UNCLASSIFIED


def calibration_kind(method: str) -> str:
    """The kind under the baseline calibration's settings (``calibration_kind``, else the benchmark's ``kind``)."""
    info = REPLICATION.get(RUNNER_ALIASES.get(method, method)) or {}
    return info.get("calibration_kind") or kind(method)


def runs_once(method: str, mode: str = DEFAULT_MODE) -> bool:
    """Whether the plan runs ``method`` at one seed only (deterministic, under ``auto``)."""
    return check_mode(mode) == "auto" and kind(method) == "deterministic"


def planned_seeds(method: str, seeds: Iterable[int], mode: str = DEFAULT_MODE) -> list[int]:
    """The seeds the benchmark runs ``method`` at: every seed, or the lowest one for a deterministic method."""
    seeds = sorted(dict.fromkeys(int(seed) for seed in seeds))
    return seeds[:1] if seeds and runs_once(method, mode) else seeds


def describe(method: str, seeds: Iterable[int], mode: str = DEFAULT_MODE) -> str:
    """One phrase for a plan line: how many runs ``method`` gets and why."""
    planned = planned_seeds(method, seeds, mode)
    if len(planned) == 1:
        return f"1 run (seed {planned[0]}, {kind(method)})"
    return f"{len(planned)} runs ({kind(method)})"
