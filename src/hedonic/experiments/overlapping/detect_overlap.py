"""The overlap half of ``hedonic detect`` (hedonic 2): SNAP pairs, LFR-derived overlap, the baselines, covers.

Imported by :mod:`hedonic.experiments.detect` when it exists. It registers:

- source ``snap``: a SNAP ``network:cover`` pair from the local archive (``download=yes`` fetches a missing pair
  from snap.stanford.edu), on the subgraph its cover induces (``induce=covered``, the paper's protocol) or the
  whole graph, optionally bounded to ``max_nodes`` vertices; ground truth: the cover;
- source ``lfr_overlap``: the LFR-derived controlled-overlap generator of the controlled study (a disjoint LFR base
  plus seeded secondary memberships; not canonical overlapping LFR); ground truth: its cover;
- ``max_memberships`` on the Hedonic method (above 1 it returns a cover);
- the eight overlapping baselines of the benchmark, each through the benchmark's own dispatcher
  (``codeseg_reproduction._run_method``) at its screened setting, with every setting a parameter;
- cover scores (F1, ONMI, Omega, ...) for any combination of partitions and covers.

A baseline whose runtime is not prepared is listed as unavailable with the ``hedonic setup`` command that prepares
it: a detection never downloads or compiles anything by itself.
"""

from __future__ import annotations

import os
from typing import Any

d: Any = None  # the core module (hedonic.experiments.detect), set by register()

DBLP_VERTICES = 317_080  # the screened community counts were tuned on full DBLP; smaller graphs scale them


# --------------------------------------------------------------------------- sources
def _pair(raw: str) -> str:
    from hedonic.experiments.overlapping import snap

    network, cover = snap.split_pair(raw)
    return f"{network}:{cover}"


def _snap(values: dict, seed: int) -> d.Network:
    from hedonic.experiments.overlapping import snap
    from hedonic.experiments.overlapping.ground_truth_data import covered_induced_dataset

    network, variant = snap.split_pair(values["pair"])
    if snap.pair_files(network, variant) is None and not values["download"]:
        raise ValueError(f"{network}:{snap.cover_variant_label(variant)} is not on disk (`hedonic show covers`); "
                         "add download=yes to fetch it from snap.stanford.edu, or `hedonic setup --pairs ...`")
    files = snap.ensure_pair(network, variant, download=bool(values["download"]))
    ds = snap.load_snap_dataset(network, cover_variant=variant, data_root=files.root, allow_catalog=False)
    if ds.graph.is_directed():
        ds = snap.common_undirected_analysis_dataset(ds)
    whole = (ds.graph.vcount(), ds.graph.ecount())
    if values["induce"] == "covered":
        ds = covered_induced_dataset(ds)
    if values["max_nodes"]:
        ds = snap.bounded_induced_dataset(ds, values["max_nodes"])
    label = f"{network}:{snap.cover_variant_label(variant)}"
    g = ds.graph
    return d.Network(d._game(g), [sorted(c) for c in ds.cover], "cover",
                     f"SNAP {label} · {values['induce']} graph · {g.vcount():,} of {whole[0]:,} vertices · "
                     f"{g.ecount():,} edges · {len(ds.cover):,} communities",
                     {"pair": label, "induce": values["induce"], "max_nodes": values["max_nodes"],
                      "whole_graph": {"vertices": whole[0], "edges": whole[1]}})


def _lfr_overlap(values: dict, seed: int) -> d.Network:
    from hedonic.experiments.overlapping.controlled_overlap import generate_controlled_cover

    seed = values["seed"] if values["seed"] is not None else seed
    made = generate_controlled_cover(
        n=values["n"], mu=values["mu"], overlap_fraction=values["overlap_fraction"],
        memberships_per_overlapping_vertex=values["memberships"],
        secondary_edge_probability=values["secondary_edge_probability"], seed=seed,
        average_degree=values["average_degree"], min_community=values["min_community"],
        max_community=values["max_community"])
    return d.Network(d._game(made.graph), [sorted(c) for c in made.cover], "cover",
                     f"LFR-derived controlled overlap · n {values['n']} · mu {values['mu']:g} · "
                     f"{len(made.overlapping_vertices)} overlapping vertices · {len(made.cover)} communities · "
                     f"seed {seed}", {"metadata": made.metadata})


def register(core) -> None:
    """Add the overlap sources, methods and cover scoring to the core's registries (called once by the core)."""
    global d
    if d is core:
        return
    d = core
    d.TRUTH_KINDS.add("cover")
    _register_sources()
    d.METHODS["hedonic"].params.append(d.Param(
        "max_memberships", "int", 1, "labels per vertex: 1 gives a partition, more a cover (-1: no practical cap)",
        low=-1))
    _register_baselines()


def _register_sources() -> None:
    d.register_source(d.Source(
        "snap", "A SNAP network with ground truth", "network:cover pairs (dblp, amazon, youtube, livejournal, "
        "orkut, friendster, wikipedia; covers all, top5000); from the local archive, or downloaded with download=yes",
        [d.Param("pair", "str", "dblp:top5000", "network:cover, e.g. amazon:top5000 or dblp:all", always=True),
         d.Param("induce", "choice", "covered", "covered: the subgraph the cover induces (the paper's); full: the "
                 "whole graph", choices=("covered", "full")),
         d.Param("max_nodes", "int", None, "bound the graph to about this many vertices around the communities",
                 low=10),
         d.Param("download", "bool", False, "fetch a missing pair from snap.stanford.edu")],
        _snap, overlap=True))
    d.register_source(d.Source(
        "lfr_overlap", "LFR with overlapping communities", "a disjoint LFR base, then a fraction of vertices joins "
        "extra communities with reinforcing edges (the controlled-overlap study's generator)",
        [d.Param("n", "int", 200, "vertices", low=20),
         d.Param("mu", "float", 0.2, "LFR mixing of the base graph", low=0, high=1),
         d.Param("overlap_fraction", "float", 0.2, "fraction of vertices with extra memberships", low=0, high=1),
         d.Param("memberships", "int", 2, "communities of each overlapping vertex", low=2),
         d.Param("secondary_edge_probability", "float", 0.2, "edge probability into each extra community",
                 low=0, high=1),
         d.Param("average_degree", "int", 5, "average degree of the base", low=1),
         d.Param("min_community", "int", 20, "smallest base community", low=2),
         d.Param("max_community", "int", 50, "largest base community", low=2),
         d.Param("seed", "int", None, "graph seed (default: the run's --seed)", low=0)],
        _lfr_overlap, overlap=True,
        available=lambda: None if d._has("networkx") else "needs networkx (pip install networkx)"))


# --------------------------------------------------------------------------- the overlapping baselines
def _kind(value: Any) -> str:
    return "bool" if isinstance(value, bool) else "int" if isinstance(value, int) else \
        "float" if isinstance(value, float) else "str"


def _readiness() -> dict[str, dict]:
    from hedonic.experiments.config import codeseg_cache_dir
    from hedonic.experiments.overlapping import info

    try:
        return {row["key"]: row for row in info.method_status(codeseg_cache_dir())}
    except Exception:  # a broken manifest must not take the whole command down
        return {}


def _available(key: str):
    def check() -> str | None:
        row = _readiness().get(key)
        if row is None or row["status"] == "ready":
            return None
        return f"{row['detail']} (hedonic setup --methods {key})" if row["status"] == "installable" else row["detail"]
    return check


def _baseline(m):
    from hedonic.experiments.overlapping import quickstart as qs

    def run(network: d.Network, values: dict, seed: int) -> d.Result:
        import os

        from hedonic.experiments.config import codeseg_cache_dir
        from hedonic.experiments.overlapping import codeseg_reproduction as cr
        from hedonic.experiments.overlapping import extra_methods

        tools = codeseg_cache_dir()
        os.environ["HEDONIC_CODESEG_CACHE"] = str(tools)
        extra_methods.install()
        params = {k: v for k, v in values.items() if v is not None}
        if m.key == "codeseg":
            params["binary"] = str(qs.codeseg_binary(tools))
        truth = network.truth if network.truth_kind == "cover" else \
            d.partition_cover(network.truth) if network.truth else []
        cover, metadata = cr._run_method(m.runner, network.graph, truth, seed, params)
        return d.Result(cover=[sorted(int(v) for v in c) for c in cover],
                        info={k: v for k, v in (metadata or {}).items() if isinstance(v, (int, float, str, bool))})
    return run


def _register_baselines() -> None:
    from hedonic.experiments.overlapping import quickstart as qs

    for m in qs.METHODS:
        if m.key == "hedonic":
            continue
        params = [d.Param(k, _kind(v), v, f"{m.label} setting (screened on full DBLP)"
                          + (", scaled with the graph when not given" if k in m.scaled else ""))
                  for k, v in m.screened.items()]
        if m.threads:
            params.append(d.Param("threads", "int", min(8, os.cpu_count() or 1), "threads", low=1))
        method = d.Method(m.key, m.label, f"overlapping baseline ({m.runner}); the benchmark's runner and setting",
                          params, _scale(m, _baseline(m)), overlap=True, available=_available(m.key))
        d.register_method(method)


def _scale(m, run):
    """Community-count settings were tuned on full DBLP: shrink them with the graph unless the user set them."""
    def scaled(network: d.Network, values: dict, seed: int) -> d.Result:
        values = dict(values)
        fraction = min(1.0, network.graph.vcount() / DBLP_VERTICES)
        for key in m.scaled:
            if key in values and values[key] == m.screened.get(key):
                values[key] = max(2, int(round(values[key] * fraction)))
        return run(network, values, seed)
    return scaled



# --------------------------------------------------------------------------- scoring covers
COVER_SCORES = ("f1", "matching_f1", "node_micro_f1", "jaccard", "omega")


def score_cover(network: d.Network, result: d.Result) -> dict[str, Any]:
    """Cover agreement when either side is a cover (a partition is read as a cover of its blocks)."""
    from hedonic.metrics import evaluate_cover, overlapping_normalized_mutual_information_lfk

    n = network.graph.vcount()
    truth = network.truth if network.truth_kind == "cover" else d.partition_cover(network.truth)
    found = result.cover if result.cover is not None else d.partition_cover(result.membership)
    scores = evaluate_cover(found, truth, n, compute_omega=True, omega_sample_size=100_000, omega_seed=0)
    out = {k: float(scores[k]) for k in COVER_SCORES if isinstance(scores.get(k), (int, float))}
    out["onmi"] = float(overlapping_normalized_mutual_information_lfk(found, truth, n_vertices=n))
    return out
