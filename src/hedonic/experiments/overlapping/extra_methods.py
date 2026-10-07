"""Opt-in methods layered on top of ``codeseg_reproduction`` at run time.

The public registry (and its 25-method DBLP default set) is left unchanged; a
caller that wants these methods calls :func:`install` before invoking the
runner, which then loads, bounds, scores, times, and records them exactly like
the registered methods.

ego_splitting  Epasto, Lattanzi & Paes Leme (KDD 2017) persona-graph splitting
               followed by Louvain, run by the pinned CDlib worker
               (``cdlib_worker.py``; the same isolated environment as ANGEL).
"""

from __future__ import annotations

from hedonic.experiments.overlapping import codeseg_reproduction as cr
from hedonic.experiments.overlapping import methods as om

EXTRA = {
    "ego_splitting": cr.MethodSpec(
        "ego_splitting",
        "ego-network splitting / overlapping",
        "EgoSplitting persona graph + python-louvain in the pinned CDlib worker",
        "undirected, unweighted NetworkX conversion",
        {"resolution": 1.0},
    ),
}
_INSTALLED = False


def _run_extra(name, graph, ground_truth, seed, parameters):
    if name == "ego_splitting":
        cover = om._run_cdlib_worker(
            cr._undirected_graph(graph), method="ego_splitting",
            parameters={"resolution": float(parameters.get("resolution", 1.0))},
            seed=int(seed), timeout_seconds=float(parameters.get("timeout_seconds", 600.0)),
        )
        return cover, {"implementation": EXTRA[name].paper_implementation,
                       "resolution": float(parameters.get("resolution", 1.0))}
    raise ValueError(name)


def install() -> None:
    """Register the extra methods with the runner (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    cr.METHOD_SPECS.update(EXTRA)
    cr.EXTENDED_METHODS = cr.EXTENDED_METHODS + tuple(n for n in EXTRA if n not in cr.EXTENDED_METHODS)
    run, params, preflight = cr._run_method, cr._method_parameters, cr._method_preflight

    def run_method(name, graph, ground_truth, seed, parameters):
        if name in EXTRA:
            return _run_extra(name, graph, ground_truth, seed, parameters)
        return run(name, graph, ground_truth, seed, parameters)

    def method_parameters(args, runtime_manifest=None):
        values = params(args, runtime_manifest=runtime_manifest)
        for name, spec in EXTRA.items():
            values.setdefault(name, dict(spec.parameters))
        return values

    def method_preflight(name, parameters):
        if name in EXTRA:
            return {"status": "ready", "reason": None, "spec": EXTRA[name].__dict__}
        return preflight(name, parameters)

    cr._run_method, cr._method_parameters, cr._method_preflight = run_method, method_parameters, method_preflight
    _INSTALLED = True
