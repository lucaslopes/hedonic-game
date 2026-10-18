"""``hedonic detect``: one network, one method, one seed, in this terminal.

    hedonic detect                                    a wizard (in a terminal): network, method, seed, then yes/no
    hedonic detect --graph sbm:k=4,size=50,p_in=0.3,difficulty=0.3 --method hedonic:resolution=0.05 --seed 0
    hedonic detect --graph file:path=edges.txt,truth=labels.txt --method leiden
    hedonic detect --list                             every network source and method with its parameters

A network comes from a *source* (a generator with parameters and a seed, a
famous graph, or a local file) and a partition or cover from a *method* with
its parameters. Both are written ``key`` or ``key:name=value,name=value``;
every parameter not given keeps its default, and the run's ``--seed`` seeds
the generator (unless the source's own ``seed`` is given) and the method. When
the source has ground truth, the result is scored against it. The result
(the call, the communities and the scores) is saved as JSON in
``<data root>/Detect/<name>`` unless ``--no-save``.

Sources and methods are registries (:func:`register_source`,
:func:`register_method`). hedonic 1 registers the disjoint ones here; the
overlap ones (SNAP pairs, LFR-derived overlap, the overlapping baselines,
covers) are registered by ``hedonic.experiments.overlapping.detect_overlap``
when that module exists (hedonic 2). This file is the same in both lines.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import os
import shlex
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hedonic.experiments.overlapping import tui
from hedonic.experiments.overlapping.tui import style

PROG = "hedonic detect"


# --------------------------------------------------------------------------- parameters
@dataclass
class Param:
    """One parameter of a source or a method.

    ``kind`` is int, float, str, bool, path, or choice (with ``choices``). A default of None means "not set"
    (the callee's own default); ``auto`` lists the strings accepted besides a value (e.g. ``density``).
    """

    name: str
    kind: str
    default: Any
    help: str
    choices: tuple[str, ...] = ()
    auto: tuple[str, ...] = ()
    low: float | None = None
    high: float | None = None
    aliases: tuple[str, ...] = ()
    required: bool = False
    always: bool = False  # always written in the "same run" spec, even at its default (it names the network)

    def parse(self, raw: Any) -> Any:
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in ("", "none", "default")):
            if self.required:
                raise ValueError(f"{self.name} is required")
            return None if self.kind != "bool" else bool(self.default)
        if isinstance(raw, str):
            raw = raw.strip()
            if raw.lower() in self.auto:
                return raw.lower()
        if self.kind == "bool":
            if isinstance(raw, bool):
                return raw
            value = str(raw).lower()
            if value in ("1", "true", "yes", "y", "on"):
                return True
            if value in ("0", "false", "no", "n", "off"):
                return False
            raise ValueError(f"{self.name} takes yes or no, not {raw!r}")
        if self.kind == "choice":
            if str(raw) not in self.choices:
                raise ValueError(f"{self.name} is one of {', '.join(self.choices)}, not {raw!r}")
            return str(raw)
        if self.kind == "path":
            path = os.path.expanduser(str(raw))
            if self.required and not os.path.isfile(path):
                raise ValueError(f"{self.name}: no such file {path}")
            return path
        if self.kind == "str":
            return str(raw)
        try:
            value = int(raw) if self.kind == "int" else float(raw)
        except (TypeError, ValueError):
            extra = f" or {', '.join(self.auto)}" if self.auto else ""
            raise ValueError(f"{self.name} takes {'an integer' if self.kind == 'int' else 'a number'}{extra}, "
                             f"not {raw!r}") from None
        if self.kind == "float" and not math.isfinite(value):
            raise ValueError(f"{self.name} must be finite, not {raw!r}")
        if self.low is not None and value < self.low:
            raise ValueError(f"{self.name} must be at least {self.low:g}, not {raw}")
        if self.high is not None and value > self.high:
            raise ValueError(f"{self.name} must be at most {self.high:g}, not {raw}")
        return value

    def show(self, value: Any) -> str:
        if value is None:
            return "default"
        if isinstance(value, bool):
            return "yes" if value else "no"
        return f"{value:g}" if isinstance(value, float) else str(value)


@dataclass
class Network:
    """A graph ready for detection, its ground truth (if any) and how it was made."""

    graph: Any  # hedonic.Game
    truth: list | None = None  # a membership (one label per vertex) or a cover (vertex lists)
    truth_kind: str | None = None  # "partition", "cover" or None
    describe: str = ""
    info: dict = field(default_factory=dict)


@dataclass
class Result:
    """A method's answer: a membership (partition) or a cover (vertex lists), plus what the call reports."""

    membership: list[int] | None = None
    cover: list[list[int]] | None = None
    info: dict = field(default_factory=dict)


@dataclass
class Source:
    key: str
    label: str
    help: str
    params: list[Param]
    build: Callable[[dict, int], Network]
    overlap: bool = False
    available: Callable[[], str | None] = lambda: None  # None when usable, else why not


@dataclass
class Method:
    key: str
    label: str
    help: str
    params: list[Param]
    run: Callable[[Network, dict, int], Result]
    overlap: bool = False
    available: Callable[[], str | None] = lambda: None
    needs_truth: bool = False  # starts from the ground truth (a supervised control)


SOURCES: dict[str, Source] = {}
METHODS: dict[str, Method] = {}


def register_source(source: Source) -> None:
    SOURCES[source.key] = source


def register_method(method: Method) -> None:
    METHODS[method.key] = method


def resolve(params: Sequence[Param], given: dict[str, Any], what: str) -> dict[str, Any]:
    """Parse ``given`` (name -> raw value) against ``params``; unknown names are an error."""
    by_name = {p.name: p for p in params}
    for p in params:
        for alias in p.aliases:
            by_name.setdefault(alias, p)
    out = {p.name: p.default for p in params}
    for name, raw in given.items():
        if name not in by_name:
            names = ", ".join(p.name for p in params) or "none"
            raise ValueError(f"{what} has no parameter {name!r} (its parameters: {names})")
        out[by_name[name].name] = by_name[name].parse(raw)
    return out


def parse_spec(text: str, registry: dict, what: str) -> tuple[Any, dict[str, str]]:
    """``key`` or ``key:name=value,name=value`` -> (registry entry, raw values)."""
    key, _, rest = str(text).strip().partition(":")
    key = key.strip().lower().replace("-", "_")
    if key not in registry:
        raise ValueError(f"unknown {what} {key!r}; choose from {', '.join(registry)} (`{PROG} --list`)")
    given: dict[str, str] = {}
    for part in _split_assignments(rest):
        name, eq, value = part.partition("=")
        if not eq or not name.strip():
            raise ValueError(f"{what} {key}: write parameters as name=value, not {part!r}")
        if name.strip() in given:
            raise ValueError(f"{what} {key}: {name.strip()} is given twice")
        given[name.strip()] = value.strip()
    return registry[key], given


def _split_assignments(rest: str) -> list[str]:
    """Split ``a=1,b=x:y,path=/p,q`` on the commas that start a new ``name=``."""
    parts: list[str] = []
    for piece in rest.split(","):
        if not piece.strip():
            continue
        if parts and "=" not in piece:
            parts[-1] += "," + piece  # a comma inside a value (e.g. a path)
        else:
            parts.append(piece.strip())
    return parts


def spec_text(key: str, params: Sequence[Param], values: dict[str, Any]) -> str:
    """The shortest spec that gives ``values``: only parameters that differ from their default."""
    parts = []
    for p in params:
        value = values.get(p.name)
        if value != p.default or p.always:
            shown = p.show(value) if value is not None else "default"
            parts.append(f"{p.name}={shown}")
    return key + (":" + ",".join(parts) if parts else "")


# --------------------------------------------------------------------------- built-in sources (both lines)
def _game(graph):
    from hedonic import Game

    return graph if isinstance(graph, Game) else Game(graph)


def _sbm(values: dict, seed: int) -> Network:
    from hedonic.experiments.disjoint import sbm_sweep

    k, size = values["k"], values["size"]
    seed = values["seed"] if values["seed"] is not None else seed
    game = sbm_sweep.generate_graph(k, size, values["p_in"], values["difficulty"], seed)
    truth = sbm_sweep.get_ground_truth(k, size)
    q = values["p_in"] * values["difficulty"]
    return Network(game, truth, "partition",
                   f"planted-partition SBM · {k} x {size} vertices · p_in {values['p_in']:g} · p_out {q:g} · seed {seed}",
                   {"generator": sbm_sweep.SBM_GENERATOR_ID, "seed": seed,
                    "graph_sha256": sbm_sweep.graph_fingerprint(game)})


def _lfr(values: dict, seed: int) -> Network:
    import networkx as nx

    import igraph as ig

    seed = values["seed"] if values["seed"] is not None else seed
    kwargs = {"tau1": values["tau1"], "tau2": values["tau2"], "mu": values["mu"],
              "average_degree": values["average_degree"], "min_community": values["min_community"],
              "max_community": values["max_community"], "max_iters": values["max_iters"], "seed": seed}
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        graph = nx.LFR_benchmark_graph(values["n"], **kwargs)
    except nx.ExceededMaxIterations as exc:
        raise ValueError(f"LFR could not build this graph ({exc}); try another seed, a smaller mu, or wider "
                         "community sizes") from None
    communities = {frozenset(graph.nodes[v]["community"]) for v in graph}
    label = {v: i for i, body in enumerate(sorted(communities, key=min)) for v in body}
    graph.remove_edges_from(nx.selfloop_edges(graph))
    g = ig.Graph(n=graph.number_of_nodes(), edges=list(graph.edges()))
    truth = [label[v] for v in range(graph.number_of_nodes())]
    return Network(_game(g), truth, "partition",
                   f"LFR benchmark (disjoint) · n {values['n']} · mu {values['mu']:g} · {len(communities)} "
                   f"communities · seed {seed}", {"generator": f"networkx {nx.__version__} LFR_benchmark_graph",
                                                  "seed": seed})


FAMOUS = ("Zachary", "Petersen", "Krackhardt_Kite", "Coxeter", "Dodecahedron", "Frucht", "Heawood", "Tutte",
          "Meredith", "Zachary_karate")


def _famous(values: dict, seed: int) -> Network:
    import igraph as ig

    name = values["name"]
    if name == "Zachary_karate":  # the karate club with the club each member joined after the split
        truth = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        return Network(_game(ig.Graph.Famous("Zachary")), truth, "partition",
                       "Zachary's karate club (34 members, the two clubs after the split as ground truth)")
    try:
        g = ig.Graph.Famous(name)
    except Exception as exc:  # igraph's message names the graphs it knows
        raise ValueError(f"igraph has no famous graph {name!r}: {exc}") from None
    return Network(_game(g.simplify()), None, None, f"igraph famous graph {name} ({g.vcount()} vertices)")


def read_graph(path: str) -> tuple[Any, list[str] | None, dict]:
    """A local network: GraphML, GML, Pajek, an igraph pickle, or an edge list (text, optionally .gz).

    Returns (undirected simple igraph.Graph, the original vertex names or None, what was done to it).
    """
    import igraph as ig

    p = Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"no such file: {p}")
    suffix = "".join(p.suffixes[-2:]).lower() if p.suffix == ".gz" else p.suffix.lower()
    info: dict[str, Any] = {"path": str(p.resolve()), "bytes": p.stat().st_size}
    names = None
    if suffix in (".graphml", ".gml", ".net", ".pajek"):
        g = ig.Graph.Read(str(p), format={".graphml": "graphml", ".gml": "gml"}.get(suffix, "pajek"))
        if "name" in g.vs.attributes():
            names = [str(x) for x in g.vs["name"]]
        elif "id" in g.vs.attributes():
            names = [str(x) for x in g.vs["id"]]
    elif suffix in (".pkl", ".pickle"):
        g = ig.Graph.Read_Pickle(str(p))  # only open pickles you made: unpickling runs code
        if not isinstance(g, ig.Graph):
            raise ValueError(f"{p} holds a {type(g).__name__}, not an igraph graph")
        if "name" in g.vs.attributes():
            names = [str(x) for x in g.vs["name"]]
    else:  # an edge list: two vertex names per line, '#' or '%' comments, any separator
        index: dict[str, int] = {}
        edges = []
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                line = line.strip()
                if not line or line[0] in "#%":
                    continue
                fields = line.replace(",", " ").replace(";", " ").split()
                if len(fields) < 2:
                    raise ValueError(f"{p}:{number}: an edge line needs two vertices, got {line[:80]!r}")
                u, v = (index.setdefault(x, len(index)) for x in fields[:2])
                edges.append((u, v))
        names = list(index)
        g = ig.Graph(n=len(names), edges=edges)
    info["read"] = {"vertices": g.vcount(), "edges": g.ecount(), "directed": g.is_directed()}
    if g.is_directed():
        g = g.as_undirected(mode="collapse")
    g = g.simplify(multiple=True, loops=True)
    info["analysed"] = {"vertices": g.vcount(), "edges": g.ecount(), "policy": "undirected, simple (no loops)"}
    return g, names, info


def read_truth(path: str, names: list[str] | None, n: int, form: str = "auto") -> tuple[list, str]:
    """Ground truth for a file graph: ``labels`` (vertex label per line) or ``communities`` (one per line).

    Vertices are named as in the graph file (its names; plain indices when it has none). Returns
    (membership or cover, "partition" or "cover"). ``auto`` reads labels when every line has two fields and
    no vertex repeats, else communities.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"no such ground-truth file: {p}")
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8", errors="replace") as handle:
        rows = [line.replace(",", " ").replace(";", " ").split() for line in handle
                if line.strip() and line.strip()[0] not in "#%"]
    index = {name: i for i, name in enumerate(names)} if names is not None else {str(i): i for i in range(n)}

    def vertex(token: str) -> int | None:
        return index.get(token)

    if form == "auto":
        firsts = [r[0] for r in rows]
        form = "labels" if rows and all(len(r) == 2 for r in rows) and len(set(firsts)) == len(firsts) else "communities"
    if form == "labels":
        membership: list = [None] * n
        labels: dict[str, int] = {}
        for r in rows:
            v = vertex(r[0])
            if v is not None:
                membership[v] = labels.setdefault(r[1], len(labels))
        missing = sum(m is None for m in membership)
        if missing:
            raise ValueError(f"{p}: {missing} of {n} vertices have no label (a partition labels every vertex; "
                             "write one community per line for a cover)")
        return membership, "partition"
    cover = []
    for r in rows:
        body = sorted({v for v in (vertex(t) for t in r) if v is not None})
        if body:
            cover.append(body)
    if not cover:
        raise ValueError(f"{p}: no community names a vertex of the graph")
    counts = [0] * n
    for body in cover:
        for v in body:
            counts[v] += 1
    if all(c == 1 for c in counts):  # every vertex in exactly one community: a partition
        membership = [0] * n
        for i, body in enumerate(cover):
            for v in body:
                membership[v] = i
        return membership, "partition"
    return cover, "cover"


def _file(values: dict, seed: int) -> Network:
    g, names, info = read_graph(values["path"])
    truth, kind = (None, None)
    if values["truth"]:
        truth, kind = read_truth(values["truth"], names, g.vcount(), values["truth_format"])
        if kind == "cover" and "cover" not in TRUTH_KINDS:
            raise ValueError("this ground truth is a cover (some vertices are in several communities or in none); "
                             "covers are scored by hedonic 2")
        info["truth"] = {"path": str(Path(values["truth"]).expanduser().resolve()), "kind": kind}
    game = _game(g)
    return Network(game, truth, kind, f"{Path(values['path']).name} · {g.vcount():,} vertices · {g.ecount():,} edges"
                   + (f" · ground truth: {kind}" if kind else ""), info | ({"names": names} if names else {}))


TRUTH_KINDS = {"partition"}  # the overlap extension adds "cover"

register_source(Source(
    "sbm", "Planted-partition SBM", "k equal communities; edge probability p_in inside, p_in x difficulty "
    "between (the disjoint paper's generator; seeded, the same graph everywhere)",
    [Param("k", "int", 4, "number of communities", low=1, aliases=("n_communities",)),
     Param("size", "int", 50, "vertices per community", low=1, aliases=("community_size",)),
     Param("p_in", "float", 0.3, "edge probability inside a community", low=0, high=1),
     Param("difficulty", "float", 0.3, "between/inside probability ratio λ = p_out/p_in", low=0, high=1),
     Param("seed", "int", None, "graph seed (default: the run's --seed)", low=0)],
    _sbm))
register_source(Source(
    "lfr", "LFR benchmark (disjoint)", "power-law degrees and community sizes, mixing mu (networkx "
    "LFR_benchmark_graph; ground truth: its communities)",
    [Param("n", "int", 250, "vertices", low=10),
     Param("tau1", "float", 3.0, "degree power-law exponent (> 1)", low=1.0001),
     Param("tau2", "float", 1.5, "community-size power-law exponent (> 1)", low=1.0001),
     Param("mu", "float", 0.1, "fraction of each vertex's edges leaving its community", low=0, high=1),
     Param("average_degree", "float", 5, "average degree", low=1),
     Param("min_community", "int", 20, "smallest community", low=1),
     Param("max_community", "int", None, "largest community (default: n)", low=1),
     Param("max_iters", "int", 500, "generator attempts before giving up", low=1),
     Param("seed", "int", None, "graph seed (default: the run's --seed)", low=0)],
    _lfr, available=lambda: None if _has("networkx") else "needs networkx (pip install networkx)"))
register_source(Source(
    "famous", "A famous small graph", "igraph's named graphs; Zachary_karate has the two clubs as ground truth",
    [Param("name", "choice", "Zachary_karate", "which graph", choices=FAMOUS)], _famous))
register_source(Source(
    "file", "A network file", "an edge list (.txt/.csv, .gz), GraphML, GML, Pajek or an igraph pickle; "
    "optional ground truth (labels or one community per line)",
    [Param("path", "path", None, "the network file", required=True),
     Param("truth", "path", None, "ground-truth file (optional)"),
     Param("truth_format", "choice", "auto", "auto, labels (vertex label per line) or communities (one per line)",
           choices=("auto", "labels", "communities"))],
    _file))


def _has(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


# --------------------------------------------------------------------------- built-in methods (both lines)
def _resolution(network: Network, value) -> float:
    return network.graph.density() if value in (None, "density") else float(value)


def _start(network: Network, values: dict, seed: int):
    """The initial partition a method starts from: singletons, the ground truth, or a noisy ground truth."""
    start = values.get("start", "singletons")
    if start == "singletons":
        return None
    if network.truth is None or network.truth_kind != "partition":
        raise ValueError(f"start={start} needs a source with a ground-truth partition")
    if start == "truth":
        return list(network.truth)
    from hedonic.experiments.disjoint import sbm_sweep

    return sbm_sweep.get_initial_membership(list(network.truth), noise=values["noise"], seed=seed)


START = [Param("start", "choice", "singletons", "start partition: singletons, truth, or noisy (truth with label "
               "swaps)", choices=("singletons", "truth", "noisy")),
         Param("noise", "float", 0.5, "with start=noisy: fraction of vertices whose labels are swapped", low=0, high=1)]


def _hedonic(network: Network, values: dict, seed: int) -> Result:
    kwargs = {k: values[k] for k in ("allow_isolation", "max_communities", "n_communities",
                                     "max_total_communities") if values.get(k) is not None}
    if values.get("max_memberships") not in (None, 1):
        kwargs["max_memberships"] = values["max_memberships"]
    result = network.graph.community_hedonic(_start(network, values, seed), resolution=_resolution(network,
                                             values["resolution"]), seed=seed, **kwargs)
    provenance = getattr(result, "_hedonic_provenance", {}) or {}
    info = {"resolution": _resolution(network, values["resolution"]),
            "quality": getattr(result, "_hedonic_native_quality", None),
            "equilibrium_slack": provenance.get("equilibrium_slack")}
    import igraph as ig

    if isinstance(result, ig.VertexCover):  # max_memberships > 1: vertex lists, one per community
        return Result(cover=[sorted(c) for c in result if len(c)], info=info)
    return Result(membership=list(result.membership), info=info)


def _leiden(network: Network, values: dict, seed: int) -> Result:
    from hedonic.experiments.leiden import leiden

    part = leiden(network.graph, resolution=_resolution(network, values["resolution"]), beta=values["beta"],
                  n_iterations=values["n_iterations"], initial_membership=_start(network, values, seed), seed=seed)
    return Result(membership=list(part.membership), info={"resolution": _resolution(network, values["resolution"])})


def _igraph_method(call: Callable[[Any, dict], Any]):
    def run(network: Network, values: dict, seed: int) -> Result:
        from hedonic.experiments.leiden import seeded_igraph_rng

        with seeded_igraph_rng(seed):
            part = call(network.graph, values)
        return Result(membership=list(part.membership))
    return run


def _spectral(g, values):
    k = values["clusters"]
    return g.community_leading_eigenvector(clusters=k)


def _onepass(network: Network, values: dict, seed: int) -> Result:
    from hedonic.experiments.disjoint import sbm_sweep

    start = _start(network, values, seed) or list(range(network.graph.vcount()))
    return Result(membership=list(sbm_sweep.community_onepass_improvement(network.graph, start).membership))


def _mirror(network: Network, values: dict, seed: int) -> Result:
    return Result(membership=list(_start(network, values, seed) or range(network.graph.vcount())))


RESOLUTION = Param("resolution", "float", "density", "CPM resolution γ (or density: the graph's density)",
                   auto=("density",), aliases=("gamma",))

register_method(Method(
    "hedonic", "Hedonic", "the hedonic game: local moving to an equilibrium (this package's C core)",
    [RESOLUTION,
     Param("allow_isolation", "bool", False, "a vertex may leave for a new empty community"),
     Param("max_communities", "int", None, "random start with labels in [0, K) (initialization only)", low=1),
     Param("max_total_communities", "int", None, "at most K occupied communities", low=1),
     Param("n_communities", "int", None, "exactly K occupied communities", low=1),
     *START],
    _hedonic))
register_method(Method(
    "leiden", "Leiden (CPM)", "full CPM Leiden: stock igraph community_leiden",
    [RESOLUTION, Param("beta", "float", 0.01, "refinement randomness", low=0),
     Param("n_iterations", "int", -1, "iterations; negative: until nothing changes"), *START],
    _leiden))
register_method(Method(
    "louvain", "Louvain (modularity)", "igraph community_multilevel",
    [Param("resolution", "float", 1.0, "modularity resolution", low=0)],
    _igraph_method(lambda g, v: g.community_multilevel(resolution=v["resolution"]))))
register_method(Method(
    "spectral", "Spectral (leading eigenvector)", "igraph community_leading_eigenvector",
    [Param("clusters", "int", None, "number of communities (default: as many as it finds)", low=1)],
    _igraph_method(_spectral)))
register_method(Method(
    "label_propagation", "Label propagation", "igraph community_label_propagation", [],
    _igraph_method(lambda g, v: g.community_label_propagation())))
register_method(Method(
    "infomap", "Infomap", "igraph community_infomap",
    [Param("trials", "int", 10, "independent runs, the best kept", low=1)],
    _igraph_method(lambda g, v: g.community_infomap(trials=v["trials"]))))
register_method(Method(
    "onepass", "One pass", "every vertex moves once to its most common neighbour label (the disjoint paper's "
    "baseline; from a noisy ground truth by default)",
    [Param("start", "choice", "noisy", START[0].help, choices=START[0].choices), START[1]],
    _onepass, needs_truth=True))
register_method(Method(
    "mirror", "Mirror", "returns its start unchanged (the disjoint paper's control)",
    [Param("start", "choice", "noisy", START[0].help, choices=START[0].choices), START[1]],
    _mirror, needs_truth=True))

def _load_overlap():
    """hedonic 2: SNAP pairs, LFR-derived overlap, the overlapping baselines and covers (absent in hedonic 1)."""
    import importlib.util

    if importlib.util.find_spec("hedonic.experiments.overlapping.detect_overlap") is None:
        return None
    from hedonic.experiments.overlapping import detect_overlap

    detect_overlap.register(sys.modules[__name__])
    return detect_overlap


_overlap = _load_overlap()


# --------------------------------------------------------------------------- scoring
def partition_cover(membership: Sequence[int]) -> list[list[int]]:
    """The blocks of a membership as vertex lists (in order of first appearance)."""
    blocks: dict[int, list[int]] = {}
    for v, label in enumerate(membership):
        blocks.setdefault(label, []).append(v)
    return list(blocks.values())


def score(network: Network, result: Result) -> dict[str, Any]:
    """Size of the answer, and agreement with the ground truth when the source has one."""
    g = network.graph
    out: dict[str, Any] = {}
    if result.membership is not None:
        sizes = sorted((result.membership.count(c) for c in set(result.membership)), reverse=True)
        out.update(communities=len(sizes), largest=sizes[0] if sizes else 0,
                   singletons=sum(s == 1 for s in sizes))
        try:
            out["modularity"] = float(g.modularity(result.membership))
        except Exception:
            pass
        try:
            from hedonic.experiments.disjoint import sbm_sweep

            out["robustness"] = sbm_sweep.robustness(g, list(result.membership))
        except Exception:
            pass
    else:
        cover = result.cover or []
        counts = [0] * g.vcount()
        for body in cover:
            for v in body:
                counts[v] += 1
        out.update(communities=len(cover), largest=max((len(b) for b in cover), default=0),
                   overlapping_vertices=sum(c > 1 for c in counts), uncovered_vertices=sum(c == 0 for c in counts))
    if network.truth is None:
        return out
    if network.truth_kind == "partition" and result.membership is not None:
        import igraph as ig

        from hedonic.experiments.disjoint import sbm_sweep

        out["ari"] = float(ig.compare_communities(result.membership, network.truth, method="adjusted_rand"))
        out.update(sbm_sweep.partition_agreement(result.membership, network.truth))
        return out
    if _overlap is not None:
        out.update(_overlap.score_cover(network, result))
    return out


SCORE_LABELS = {"ari": "ARI", "nmi": "NMI", "overlap": "overlap", "overlap_normalized": "overlap (0 = chance)",
                "vi": "VI", "f1": "F1 (symmetric)", "matching_f1": "F1 (matching)",
                "node_micro_f1": "F1 (node micro)", "jaccard": "Jaccard", "onmi": "ONMI (LFK)", "omega": "Omega",
                "communities": "communities", "largest": "largest", "singletons": "singletons",
                "modularity": "modularity", "robustness": "robust vertices", "overlapping_vertices": "overlapping",
                "uncovered_vertices": "uncovered"}


# --------------------------------------------------------------------------- running
@dataclass
class Plan:
    source: Source
    source_values: dict
    method: Method
    method_values: dict
    seed: int
    output: str | None  # folder; None: not saved

    def argv(self) -> list[str]:
        out = ["detect", "--graph", spec_text(self.source.key, self.source.params, self.source_values),
               "--method", spec_text(self.method.key, self.method.params, self.method_values),
               "--seed", str(self.seed)]
        if self.output is None:
            out.append("--no-save")
        return out

    def command(self) -> str:
        return "hedonic " + " ".join(shlex.quote(a) for a in self.argv())


def default_output(plan_or_keys) -> str:
    from hedonic.experiments import config

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(getattr(config, "DATA_ROOT", Path("~/Databases/Hedonic"))).expanduser()
    return str(root / "Detect" / f"{plan_or_keys}-{stamp}")


def check(plan: Plan) -> None:
    for what, entry in (("network source", plan.source), ("method", plan.method)):
        reason = entry.available()
        if reason:
            raise ValueError(f"{what} {entry.key} is not available here: {reason}")


def execute(plan: Plan, say=print) -> dict[str, Any]:
    """Build the network, run the method, score; returns the record (also saved when ``plan.output``)."""
    from hedonic.experiments.runtime import runtime_versions

    check(plan)
    t0 = time.perf_counter()
    network = plan.source.build(plan.source_values, plan.seed)
    built = time.perf_counter() - t0
    say(style(f"  network  {network.describe}", "dim"))
    if plan.method.needs_truth and plan.method_values.get("start") != "singletons" and network.truth is None:
        raise ValueError(f"{plan.method.label} starts from the ground truth, and {plan.source.label} has none "
                         "(choose start=singletons)")
    say(style(f"  running  {plan.method.label} ...", "dim"))
    t1 = time.perf_counter()
    result = plan.method.run(network, plan.method_values, plan.seed)
    seconds = time.perf_counter() - t1
    scores = score(network, result)
    record = {
        "command": plan.command(),
        "source": {"key": plan.source.key, "values": plan.source_values, "describe": network.describe,
                   "info": {k: v for k, v in network.info.items() if k != "names"},
                   "vertices": network.graph.vcount(), "edges": network.graph.ecount(),
                   "truth": network.truth_kind},
        "method": {"key": plan.method.key, "values": plan.method_values, "info": result.info},
        "seed": plan.seed, "seconds": {"network": built, "detection": seconds}, "scores": scores,
        "runtime": runtime_versions(), "created": dt.datetime.now().isoformat(timespec="seconds"),
    }
    if plan.output:
        folder = Path(plan.output).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        names = network.info.get("names")
        answer = ({"membership": result.membership} if result.membership is not None else {"cover": result.cover})
        if names:
            answer["vertex_names"] = names
        (folder / "result.json").write_text(json.dumps(record | {"answer": answer}, indent=1, default=str) + "\n")
        record["saved"] = str(folder / "result.json")
    return record


def render(record: dict) -> str:
    s = record["scores"]
    lines = [style(f"  {record['method']['key']} on {record['source']['key']}", "bold") +
             f"  ·  {record['source']['vertices']:,} vertices · {record['source']['edges']:,} edges · "
             f"seed {record['seed']} · {record['seconds']['detection']:.2f} s"]
    width = max(len(SCORE_LABELS.get(k, k)) for k in s) if s else 0
    for key, value in s.items():
        shown = f"{value:.4f}" if isinstance(value, float) else f"{value:,}" if isinstance(value, int) else str(value)
        lines.append(f"    {SCORE_LABELS.get(key, key):<{width}}  {shown}")
    if record["source"]["truth"] is None:
        lines.append(style("    (no ground truth: no agreement scores)", "dim"))
    if record.get("saved"):
        lines.append(style(f"  saved    {record['saved']}", "dim"))
    lines.append(style(f"  same run: {record['command']}", "dim"))
    return "\n".join(lines)


# --------------------------------------------------------------------------- listing and the wizard
def render_list() -> str:
    lines = []
    for title, registry in (("network sources (--graph)", SOURCES), ("methods (--method)", METHODS)):
        lines.append(style(f"\n  {title}", "bold"))
        for entry in registry.values():
            reason = entry.available()
            tag = style(f"  [unavailable: {reason}]", "yellow") if reason else ""
            lines.append(f"    {style(entry.key, 'cyan')}  {entry.label}: {entry.help}{tag}")
            for p in entry.params:
                extra = f" ({'|'.join(p.choices)})" if p.choices else ""
                lines.append(style(f"        {p.name}={p.show(p.default)}  {p.help}{extra}", "dim"))
    lines.append(style(f"\n  e.g. {PROG} --graph sbm:k=3,size=40,p_in=0.4,difficulty=0.2 --method hedonic --seed 1",
                       "dim"))
    return "\n".join(lines)


def _ask_params(title: str, params: Sequence[Param], values: dict) -> dict:
    values = dict(values)
    for p in params:
        if p.kind == "choice":
            index = p.choices.index(values[p.name]) if values[p.name] in p.choices else 0
            values[p.name] = p.choices[tui.select(f"{title} · {p.name}: {p.help}", list(p.choices), index)]
        elif p.kind == "bool":
            values[p.name] = tui.select(f"{title} · {p.name}: {p.help}", ["no", "yes"], int(bool(values[p.name]))) == 1
        else:
            def check(raw, p=p):
                try:
                    p.parse(raw)
                except ValueError as exc:
                    return str(exc)
                return None
            raw = tui.text(f"{title} · {p.name} ({p.help})", p.show(values[p.name]) if values[p.name] not in
                           (None, "") else ("" if p.kind == "path" else "default"), check)
            values[p.name] = p.parse(raw)
    return values


def wizard(plan: Plan | None = None) -> Plan | None:
    """Ask for the network, the method, the seed and the folder; None when cancelled or answered no."""
    print(style("\n  hedonic detect · one network, one method, one seed\n", "bold", "magenta"))
    sources = list(SOURCES.values())
    i = tui.select("Network", [(s.label, s.available() and f"unavailable: {s.available()}" or s.help)
                               for s in sources])
    source = sources[i]
    if source.available():
        print(style(f"  {source.label} is not available here: {source.available()}", "yellow"))
        return None
    source_values = {p.name: p.default for p in source.params}
    if any(p.required for p in source.params) or tui.select(f"{source.label}: parameters", [("Keep the defaults", ", ".join(
            f"{p.name}={p.show(p.default)}" for p in source.params) or "none"), ("Set them", "one by one")]) == 1:
        source_values = _ask_params(source.key, source.params, source_values)
    methods = list(METHODS.values())
    i = tui.select("Method", [(m.label, m.available() and f"unavailable: {m.available()}" or m.help) for m in methods])
    method = methods[i]
    if method.available():
        print(style(f"  {method.label} is not available here: {method.available()}", "yellow"))
        return None
    method_values = {p.name: p.default for p in method.params}
    if method.params and tui.select(f"{method.label}: parameters", [("Keep the defaults", ", ".join(
            f"{p.name}={p.show(p.default)}" for p in method.params)), ("Set them", "one by one")]) == 1:
        method_values = _ask_params(method.key, method.params, method_values)
    seed = int(tui.text("Seed", "0", lambda v: None if v.isdigit() else "a non-negative integer"))
    save = tui.select("Save the result?", [("Yes", "JSON in the data folder's Detect/"), ("No", "only print it")]) == 0
    plan = Plan(source, source_values, method, method_values, seed,
                default_output(f"{source.key}-{method.key}") if save else None)
    print(style(f"\n  same run without the wizard:\n    {plan.command()}", "dim"))
    if plan.output:
        print(style(f"  saving to {plan.output}", "dim"))
    return plan if tui.confirm("Run it?") else None


# --------------------------------------------------------------------------- command line
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[1].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=f"`{PROG} --list` lists every source and method with its parameters.")
    p.add_argument("--graph", "--network", dest="graph", metavar="SOURCE[:name=value,...]",
                   help="where the network comes from (sbm, lfr, famous, file, ...)")
    p.add_argument("--method", metavar="METHOD[:name=value,...]", default="hedonic", help="the detector (default hedonic)")
    p.add_argument("--seed", type=int, default=0, help="seeds the generator and the method (default 0)")
    p.add_argument("--output", metavar="DIR", help="save result.json here (default <data root>/Detect/<name>)")
    p.add_argument("--no-save", action="store_true", help="only print the result")
    p.add_argument("--json", action="store_true", help="print the record as JSON")
    p.add_argument("--list", action="store_true", help="list the sources and methods with their parameters")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the command; build and run nothing")
    p.add_argument("-i", "--interactive", action="store_true", help="the wizard (default without --graph in a terminal)")
    return p


def plan_from_args(args) -> Plan:
    source, given = parse_spec(args.graph, SOURCES, "network source")
    source_values = resolve(source.params, given, f"source {source.key}")
    for p in source.params:
        if p.required and source_values[p.name] in (None, ""):
            raise ValueError(f"source {source.key} needs {p.name}=... ({p.help})")
    method, given = parse_spec(args.method, METHODS, "method")
    method_values = resolve(method.params, given, f"method {method.key}")
    if args.seed < 0:
        raise ValueError("--seed must be a non-negative integer")
    output = None if args.no_save else (args.output or default_output(f"{source.key}-{method.key}"))
    return Plan(source, source_values, method, method_values, args.seed, output)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.list:
        print(render_list())
        return 0
    try:
        if args.graph is None:
            if not (args.interactive or tui.interactive()):
                parser.print_usage(sys.stderr)
                print(f"{PROG}: error: --graph is required without a terminal (`{PROG} --list`)", file=sys.stderr)
                return 2
            plan = wizard()
            if plan is None:
                print(style("  not run", "yellow"))
                tui.note_cancelled()
                return 0
        else:
            plan = plan_from_args(args)
            check(plan)
        if args.dry_run:
            print(f"  network  {spec_text(plan.source.key, plan.source.params, plan.source_values)}")
            print(f"  method   {spec_text(plan.method.key, plan.method.params, plan.method_values)}")
            print(f"  seed     {plan.seed}" + (f" · saving to {plan.output}" if plan.output else " · not saved"))
            print(style(f"  {plan.command()}", "dim"))
            return 0
        record = execute(plan, say=(lambda line: print(line, file=sys.stderr)) if args.json else print)
    except tui.Cancelled:  # q, Esc or Quit: a choice, not a failure
        print(style("\n  not run", "yellow"))
        tui.note_cancelled()
        return 0
    except KeyboardInterrupt:
        print(style("\n  interrupted", "yellow"), file=sys.stderr)
        return 130
    except (ValueError, OSError) as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(record, indent=1, default=str))
    else:
        print(render(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
