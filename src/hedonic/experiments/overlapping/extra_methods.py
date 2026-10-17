"""Opt-in methods layered on top of ``codeseg_reproduction`` at run time.

The public registry (and its 25-method DBLP default set) is left unchanged; a
caller that wants these methods calls :func:`install` before invoking the
runner, which then loads, bounds, scores, times (``timing.py``), and records them exactly like
the registered methods.

ego_splitting  Epasto, Lattanzi & Paes Leme (KDD 2017) persona-graph splitting
               followed by Louvain, run by the pinned CDlib worker
               (``cdlib_worker.py``; the same isolated environment as ANGEL).

Controls (``CONTROLS``) are supervised reference points, not detectors: they read the scored reference cover the
runner passes to every method. They add no parameters to the runner's configuration digest, so installing them leaves
the digest of every other method's record unchanged.

disjoint_oracle  the ground-truth disjoint oracle (``disjoint_oracle.py``): the best disjoint answer of a family of
                 projections of the reference cover, a ceiling for overlap-blind answers.
"""

from __future__ import annotations

from hedonic.experiments.overlapping import codeseg_reproduction as cr
from hedonic.experiments.overlapping import methods as om
from hedonic.experiments import timing

EXTRA = {
    "ego_splitting": cr.MethodSpec(
        "ego_splitting",
        "ego-network splitting / overlapping",
        "EgoSplitting persona graph + python-louvain in the pinned CDlib worker",
        "undirected, unweighted NetworkX conversion",
        {"resolution": 1.0},
    ),
}
CONTROLS = {
    "disjoint_oracle": cr.MethodSpec(
        "disjoint_oracle",
        "supervised control / disjoint",
        "reads the scored reference cover; best of three disjoint projections improved by single-vertex moves",
        "ignores the graph's edges (reference cover only)",
        {},
    ),
}
_INSTALLED = False
_SAVED: tuple | None = None  # what install() replaced, for uninstall()


def _run_extra(name, graph, ground_truth, seed, parameters):
    if name == "ego_splitting":
        cover = om._run_cdlib_worker(
            cr._undirected_graph(graph), method="ego_splitting",
            parameters={"resolution": float(parameters.get("resolution", 1.0))},
            seed=int(seed), timeout_seconds=float(parameters.get("timeout_seconds", 600.0)),
        )
        return cover, {"implementation": EXTRA[name].paper_implementation,
                       "resolution": float(parameters.get("resolution", 1.0))}
    if name == "disjoint_oracle":
        from hedonic.experiments.overlapping.disjoint_oracle import oracle_partition

        with timing.active().measure("oracle_partition"):  # same clock as every detector: the computation only
            cover, report = oracle_partition(ground_truth)
        return cover, {"implementation": CONTROLS[name].paper_implementation, "supervised": True,
                       "max_memberships": 1, "oracle": report}
    raise ValueError(name)


def install() -> None:
    """Register the extra methods with the runner (idempotent; :func:`uninstall` undoes it)."""
    global _INSTALLED, _SAVED
    if _INSTALLED:
        return
    added = tuple(name for name in (*EXTRA, *CONTROLS) if name not in cr.METHOD_SPECS)
    _SAVED = (added, cr.EXTENDED_METHODS, cr._run_method, cr._method_parameters, cr._method_preflight)
    cr.METHOD_SPECS.update(EXTRA)
    cr.METHOD_SPECS.update(CONTROLS)
    cr.EXTENDED_METHODS = cr.EXTENDED_METHODS + tuple(
        n for n in (*EXTRA, *CONTROLS) if n not in cr.EXTENDED_METHODS)
    run, params, preflight = cr._run_method, cr._method_parameters, cr._method_preflight

    def run_method(name, graph, ground_truth, seed, parameters):
        if name in EXTRA or name in CONTROLS:
            return _run_extra(name, graph, ground_truth, seed, parameters)
        return run(name, graph, ground_truth, seed, parameters)

    def method_parameters(args, runtime_manifest=None):
        values = params(args, runtime_manifest=runtime_manifest)
        for name, spec in EXTRA.items():
            values.setdefault(name, dict(spec.parameters))
        return values

    def method_preflight(name, parameters):
        if name in EXTRA or name in CONTROLS:
            return {"status": "ready", "reason": None, "spec": {**EXTRA, **CONTROLS}[name].__dict__}
        return preflight(name, parameters)

    cr._run_method, cr._method_parameters, cr._method_preflight = run_method, method_parameters, method_preflight
    _INSTALLED = True


def uninstall() -> None:
    """Restore the runner's registry as it was before :func:`install` (a no-op when nothing is installed)."""
    global _INSTALLED, _SAVED
    if not _INSTALLED or _SAVED is None:
        return
    added, cr.EXTENDED_METHODS, cr._run_method, cr._method_parameters, cr._method_preflight = _SAVED
    for name in added:
        cr.METHOD_SPECS.pop(name, None)
    _INSTALLED, _SAVED = False, None
