"""Named method adapters for reproducible overlapping-community benchmarks.

The registry exposes the complete literature vocabulary used by the DBLP
SNAP reproduction. Native methods and maintained optional baselines run
directly here; methods whose external runtime is owned by the CoDeSEG runner
are delegated lazily to that dispatcher. Availability is always recorded and
missing implementations fail closed rather than being silently substituted.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import inspect
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import igraph as ig
from hedonic import Game
from hedonic.experiments.leiden import seeded_igraph_rng
from hedonic.experiments import timing
from hedonic.experiments.overlapping import cdlib_env
from hedonic.experiments.overlapping.cdlib_env import MethodUnavailable
from hedonic.experiments.overlapping.metrics import partition_to_cover_lists
from hedonic.experiments.overlapping.snap import canonicalize_cover
from hedonic.utils import sample_uniform_ints

# ``MethodUnavailable`` (a method's implementation cannot run) is one class for
# the whole runner: the CDlib environment, these adapters and the SNAP runner.


# One public vocabulary is shared by the SNAP/CoDeSEG runner and the older
# resumable benchmark.  Keeping the names here makes ``resolve_methods`` a
# useful library entry point instead of requiring callers to know which CLI
# owns a detector.  The implementations that are maintained in
# ``codeseg_reproduction`` are exposed below through a lazy adapter; this file
# remains free of an import cycle at module import time.
LITERATURE_METHODS: tuple[str, ...] = (
    "codeseg",
    "slpa",
    "bigclam",
    "ncgame",
    "fox",
    "louvain",
    "der",
    "leiden",
    "flpa",
    "hedonic_local",
    "angel",
    "infomap",
    "demon",
    "clique_percolation",
    "link_clustering",
    "neo_kmeans",
    "nise",
    "sse",
    "qoce",
    "svi",
    "essc",
)

METHOD_ALIASES: dict[str, str] = {
    "co-deseg": "codeseg",
    "bigclam": "bigclam",
    "big-clam": "bigclam",
    "nc-game": "ncgame",
    "neo-k-means": "neo_kmeans",
    "neo_k_means": "neo_kmeans",
    "link-communities": "link_clustering",
    "link_communities": "link_clustering",
    "link communities": "link_clustering",
    "hedonic": "hedonic_local",
    "hedonic-local": "hedonic_local",
    "community-hedonic": "community_hedonic",
    "hedonic-multiphase": "hedonic_multiphase",
    "hedonic-multiphase-x10": "hedonic_multiphase_x10",
    "hedonic-multiphase-x100": "hedonic_multiphase_x100",
}

# Hedonic methods of 1.x that ran overlapping multilevel (Leiden refinement and
# aggregation over covers after local moving). hedonic 2.0 has local
# moving only, so they are not registered; naming one is an error that says so.
REMOVED_METHODS = frozenset(
    {"community_hedonic", "hedonic_multiphase", "hedonic_multiphase_x10", "hedonic_multiphase_x100"}
)
REMOVED_REASON = (
    "the multi-phase (multilevel) Hedonic variants are not available in hedonic 2.0, "
    "which runs local moving only; use hedonic_local"
)


def removed_method_error(names: Sequence[str]) -> ValueError | None:
    """The error for a request that names a 1.x-only Hedonic variant, or None."""
    removed = [name for name in names if name in REMOVED_METHODS]
    if not removed:
        return None
    return ValueError(f"{', '.join(removed)}: {REMOVED_REASON}")

DISPATCHER_METHODS = frozenset(
    name
    for name in LITERATURE_METHODS
    if name not in {
        "hedonic_local",
        "clique_percolation",
        "demon",
        "infomap",
        "angel",
        "link_clustering",
    }
)
GROUND_TRUTH_REQUIRED_METHODS = frozenset(
    {
        "codeseg",
        "bigclam",
        "ncgame",
        "neo_kmeans",
        "nise",
        "sse",
        "qoce",
        "svi",
    }
)


def canonical_method_name(name: str) -> str:
    """Normalize a literature/CLI method name to the public registry key."""
    value = str(name).strip()
    if not value:
        return value
    lowered = value.lower()
    return METHOD_ALIASES.get(lowered, lowered)


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


EXTERNAL_DISTRIBUTIONS = {
    "clique_percolation": "networkx",
    "demon": "demon",
    # Baselines from Yang--Leskovec (ICDM 2012).  These are deliberately
    # optional: no implementation is bundled with hedonic and availability is
    # reported rather than silently substituting another detector.
    "agmfit": "agmfit",
    "link_clustering": "cdlib",
    "mmsb": "mmsb",
    "infomap": "infomap",
    "angel": "cdlib",
}
EXTERNAL_IMPLEMENTATIONS = {
    "clique_percolation": (
        "networkx.algorithms.community.kclique",
        "k_clique_communities",
    ),
    "demon": ("demon.alg.Demon", "Demon"),
    "agmfit": ("agmfit", "AGMfit"),
    "link_clustering": ("cdlib.algorithms", "hierarchical_link_community"),
    "mmsb": ("mmsb", "MMSB"),
    "infomap": ("infomap", "Infomap"),
    "angel": ("cdlib.algorithms", "angel"),
}


def method_dependency_identity(name: str) -> dict[str, Any] | None:
    if name in DISPATCHER_METHODS:
        return {
            "schema_version": 1,
            "dispatcher": "hedonic.experiments.overlapping.codeseg_reproduction._run_method",
            "method": name,
            "implementation_belongs_to_distribution": False,
        }
    distribution = EXTERNAL_DISTRIBUTIONS.get(name)
    if distribution is None:
        return None
    if name in {"link_clustering", "angel"}:
        # CDlib's maintained API is named ``hierarchical_link_community``
        # and lives in the isolated project used by SLPA.  Do not import it
        # here: it pins its own python-igraph, which must not replace the
        # igraph of this process.
        try:
            receipt = cdlib_env.link_clustering_environment_receipt()
        except Exception:
            receipt = {}
        files = receipt.get("files", {}) if isinstance(receipt, dict) else {}
        worker_name = "link_clustering_worker" if name == "link_clustering" else "cdlib_worker.py"
        script = "link_clustering_worker.py" if name == "link_clustering" else "cdlib_worker.py"
        try:
            worker_path = cdlib_env.worker_script(script)
        except FileNotFoundError:
            worker_path = Path(script)
        worker = files.get(worker_name, {})
        if not worker:
            worker = {"path": str(worker_path), "sha256": _sha256_file(worker_path)}
        isolated_environment = dict(receipt)
        isolated_files = dict(files)
        isolated_files[worker_name] = worker
        isolated_environment["files"] = isolated_files
        isolated_environment["worker"] = str(worker_path)
        isolated_environment["api"] = (
            "cdlib.algorithms.hierarchical_link_community"
            if name == "link_clustering"
            else "cdlib.algorithms.angel"
        )
        return {
            "schema_version": 1,
            "distribution": distribution,
            "version": "0.4.0" if receipt else None,
            "implementation_module": "cdlib.algorithms",
            "implementation_attribute": (
                "hierarchical_link_community" if name == "link_clustering" else "angel"
            ),
            "implementation_path": f"isolated/{worker_name}",
            "implementation_sha256": worker.get("sha256"),
            "implementation_belongs_to_distribution": False,
            "isolated_environment_ready": bool(receipt.get("ready")) and bool(worker.get("sha256")) if receipt else False,
            "isolated_environment": isolated_environment,
        }
    try:
        installed = importlib.metadata.distribution(distribution)
        version = installed.version
    except importlib.metadata.PackageNotFoundError:
        return {
            "schema_version": 1,
            "distribution": distribution,
            "version": None,
            "implementation_module": EXTERNAL_IMPLEMENTATIONS[name][0],
            "implementation_attribute": EXTERNAL_IMPLEMENTATIONS[name][1],
            "implementation_path": None,
            "implementation_sha256": None,
            "implementation_belongs_to_distribution": False,
        }
    try:
        module_name, attribute = EXTERNAL_IMPLEMENTATIONS[name]
        module = importlib.import_module(module_name)
        getattr(module, attribute)
        source_name = inspect.getsourcefile(module)
        if source_name is None:
            raise OSError("implementation source file is unavailable")
        source_path = Path(source_name).resolve()
        distribution_root = Path(installed.locate_file("")).resolve()
        try:
            relative_path = source_path.relative_to(distribution_root).as_posix()
        except ValueError:
            relative_path = None
        installed_files = installed.files or []
        belongs_to_distribution = any(
            Path(installed.locate_file(item)).resolve() == source_path
            for item in installed_files
        )
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except (ImportError, AttributeError, OSError):
        return {
            "schema_version": 1,
            "distribution": distribution,
            "version": version,
            "implementation_module": EXTERNAL_IMPLEMENTATIONS[name][0],
            "implementation_attribute": EXTERNAL_IMPLEMENTATIONS[name][1],
            "implementation_path": None,
            "implementation_sha256": None,
            "implementation_belongs_to_distribution": False,
        }
    return {
        "schema_version": 1,
        "distribution": distribution,
        "version": version,
        "implementation_module": module_name,
        "implementation_attribute": attribute,
        "implementation_path": relative_path,
        "implementation_sha256": source_sha256,
        "implementation_belongs_to_distribution": belongs_to_distribution,
    }


@dataclass(frozen=True)
class DetectorOutput:
    """Detector cover plus the labelled per-vertex memberships it came from.

    ``community_hedonic`` returns one row of labels per vertex
    (``Game.memberships``), in which two labels may have the same members.
    Keeping those rows beside the normalized cover lets the benchmark record
    exactly the state the detector reached (the cover audit's ``labelled``
    profile), without changing the cover contract of the external baselines.
    """

    cover: list[list[int]]
    final_memberships: list[list[int]] | None = None


def seeded_initial_membership(
    n_vertices: int, n_communities: int, seed: int
) -> list[int]:
    """Return reproducible contiguous disjoint labels for a warm start."""
    k = max(1, int(n_communities))
    if k == 1:
        return [0] * int(n_vertices)
    raw = sample_uniform_ints(int(n_vertices), k - 1, int(seed)).tolist()
    unique = sorted(set(raw))
    if len(unique) == k and unique[0] == 0 and unique[-1] == k - 1:
        return [int(label) for label in raw]
    remap = {old: new for new, old in enumerate(unique)}
    return [remap[int(label)] for label in raw]


@dataclass(frozen=True)
class MethodAdapter:
    """Metadata and callable for one normalized-cover detector."""

    name: str
    family: str
    implementation: str
    parameters: dict[str, Any]
    scalability: str
    output_kind: str
    install_requirement: str | None
    runner: Callable[[ig.Graph, int, float, int, dict[str, Any]], DetectorOutput | list[list[int]]]
    density_resolution_multiplier: float | None = None


def normalize_cover(
    cover: Sequence[Sequence[int]], n_vertices: int
) -> tuple[list[list[int]], dict[str, int]]:
    """Validate and normalize arbitrary detector output to ``list[list[int]]``.

    Bad values are counted in the accompanying metadata.  A detector producing
    only invalid/empty communities is rejected by the benchmark runner rather
    than being evaluated as a misleading empty prediction.
    """
    normalized: list[list[int]] = []
    invalid_members = 0
    duplicate_members = 0
    empty_communities = 0
    for community in cover:
        members: list[int] = []
        seen: set[int] = set()
        for member in community:
            try:
                vertex = int(member)
            except (TypeError, ValueError):
                invalid_members += 1
                continue
            if vertex < 0 or vertex >= n_vertices:
                invalid_members += 1
                continue
            if vertex in seen:
                duplicate_members += 1
                continue
            seen.add(vertex)
            members.append(vertex)
        if members:
            normalized.append(members)
        else:
            empty_communities += 1
    canonical, canonicalization = canonicalize_cover(
        normalized, n_vertices=n_vertices, minimum_size=1
    )
    return canonical, {
        "invalid_members_dropped": invalid_members,
        "duplicate_members_removed": duplicate_members
        + canonicalization["duplicate_members_removed"],
        "duplicate_communities_removed": canonicalization[
            "duplicate_communities_removed"
        ],
        "empty_communities_dropped": empty_communities,
        "canonical_community_count": len(canonical),
    }


def _adapter_cap(parameters: dict[str, Any], k: int) -> int:
    """Cap of a Hedonic adapter: 1 for the explicit disjoint control, else the usual overlap cap."""
    if parameters.get("disjoint_control"):
        return 1
    return int(parameters.get("max_memberships", max(2, k)))


def hedonic_resolution(parameters: dict[str, Any], resolution: float) -> float:
    """The resolution a Hedonic adapter runs at.

    ``absolute_resolution`` fixes it; otherwise ``resolution_multiplier``
    scales the requested one (the graph density by default), capped at 1.
    """
    if "absolute_resolution" in parameters:
        return float(parameters["absolute_resolution"])
    return min(1.0, float(parameters.get("resolution_multiplier", 1.0)) * resolution)


def _hedonic_local(
    graph: ig.Graph, k: int, resolution: float, seed: int, parameters: dict[str, Any]
) -> DetectorOutput:
    """Hedonic local moving (``Game.community_hedonic``) to an equilibrium, seeded by ``seed``.

    The detector clock (``timing``) covers the ``community_hedonic`` call alone: building the ``Game`` from the graph
    and reading the cover and the labelled memberships out of the result come before and after it.
    """
    game = Game(graph)
    options = dict(
        resolution=hedonic_resolution(parameters, resolution),
        max_memberships=_adapter_cap(parameters, k),
        allow_isolation=bool(parameters.get("allow_isolation", True)),
        seed=int(seed),
    )
    initial = parameters.get("_initial_membership")
    with timing.active().measure("Game.community_hedonic"):
        result = game.community_hedonic(initial, **options)
    return DetectorOutput(
        cover=partition_to_cover_lists(result),
        final_memberships=_final_memberships(game),
    )


def _final_memberships(game: Game) -> list[list[int]]:
    """The labelled per-vertex rows of the last ``community_hedonic`` call (``Game.memberships``)."""
    return [[int(label) for label in labels] for labels in (game.memberships or [])]


def _cover_from_memberships(
    memberships: Any, n_vertices: int
) -> list[list[int]] | None:
    if not isinstance(memberships, list) or len(memberships) != n_vertices:
        return None
    communities: dict[int, list[int]] = {}
    try:
        for vertex, labels in enumerate(memberships):
            row = [int(label) for label in labels]
            if not row or len(row) != len(set(row)) or min(row) < 0:
                return None
            for label in row:
                communities.setdefault(label, []).append(vertex)
        cover, _ = canonicalize_cover(
            list(communities.values()),
            n_vertices=n_vertices,
            minimum_size=1,
        )
        return cover
    except (TypeError, ValueError):
        return None


def _networkx_graph(graph: ig.Graph):
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - availability handles this
        raise MethodUnavailable("networkx is not installed") from exc
    # Clique percolation and DEMON operate on ordinary undirected simple graphs.
    # The conversion is recorded in the method metadata by the benchmark.
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.vcount()))
    nx_graph.add_edges_from(graph.get_edgelist())
    return nx_graph


def k_clique_communities(graph: ig.Graph, k: int, *, block_cost: float = 3e7) -> list[list[int]]:
    """Clique-percolation communities (Palla et al. 2005): the covers ``networkx.community.k_clique_communities`` returns.

    A community is the union of the maximal cliques of size at least ``k`` that are connected through chains of cliques
    sharing at least ``k - 1`` vertices. NetworkX finds the maximal cliques in seconds but then compares clique pairs in
    Python, which on a hub-heavy graph (YouTube) ran past an hour. Here igraph lists the maximal cliques (C) and the
    overlaps come from a sparse product of the clique-vertex incidence matrix with its transpose, taken in row blocks of
    bounded cost; components are merged block by block, so memory stays small. The result is the same cover, with the
    communities ordered by their first clique and each community's vertices sorted.
    """
    import numpy as np
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    if k < 2:
        raise ValueError(f"k={k}, k must be greater than 1")
    cliques = graph.maximal_cliques(min=k)
    if not cliques:
        return []
    count = len(cliques)
    lengths = np.fromiter((len(c) for c in cliques), dtype=np.int64, count=count)
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    indices = np.fromiter((v for c in cliques for v in c), dtype=np.int32, count=int(indptr[-1]))
    incidence = sparse.csr_matrix((np.ones(len(indices), dtype=np.int32), indices, indptr), shape=(count, graph.vcount()))
    by_vertex = incidence.T.tocsr()
    # row cost of the product: for each vertex of clique i, the cliques holding it (the work of that row of A @ A.T)
    degree = np.diff(by_vertex.indptr).astype(np.float64)
    row_cost = np.add.reduceat(degree[indices], indptr[:-1]) if len(indices) else np.zeros(count)
    label = np.arange(count, dtype=np.int64)
    start = 0
    while start < count:
        stop, spent = start, 0.0
        while stop < count and (stop == start or spent + row_cost[stop] <= block_cost):
            spent += row_cost[stop]
            stop += 1
        overlap = (incidence[start:stop] @ by_vertex).tocoo()
        keep = (overlap.data >= k - 1) & (overlap.col > overlap.row + start)
        if keep.any():
            left, right = label[overlap.row[keep] + start], label[overlap.col[keep]]
            differ = left != right
            if differ.any():
                pairs = np.unique(np.stack([left[differ], right[differ]]), axis=1)
                graph_of_labels = sparse.coo_matrix((np.ones(pairs.shape[1], dtype=np.int8), (pairs[0], pairs[1])),
                                                    shape=(count, count))
                _, merged = connected_components(graph_of_labels, directed=False)
                label = merged[label]
        start = stop
    order: dict[int, int] = {}
    members: list[set[int]] = []
    for index, clique in enumerate(cliques):
        slot = order.setdefault(int(label[index]), len(order))
        if slot == len(members):
            members.append(set())
        members[slot].update(clique)
    return [sorted(int(v) for v in community) for community in members]


def _clique_percolation(
    graph: ig.Graph, _k: int, _resolution: float, _seed: int, parameters: dict[str, Any]
) -> list[list[int]]:
    """Clique-percolation baseline (an overlapping node cover); see :func:`k_clique_communities`.

    NumPy and SciPy are imported before the detector clock starts, so the first call of a process does not pay for them.
    """
    clique_size = int(parameters.get("clique_size", 3))
    max_communities = int(parameters.get("max_communities", 50_000))
    import numpy  # noqa: F401
    from scipy import sparse  # noqa: F401
    from scipy.sparse.csgraph import connected_components  # noqa: F401

    with timing.active().measure("k_clique_communities"):
        cover = k_clique_communities(graph, clique_size)
    return cover[:max_communities]


def _demon(
    graph: ig.Graph, _k: int, _resolution: float, seed: int, parameters: dict[str, Any]
) -> list[list[int]]:
    try:
        from demon.alg.Demon import Demon
    except ImportError as exc:  # pragma: no cover - availability handles this
        raise MethodUnavailable("demon is not installed") from exc
    random.seed(seed)
    # The maintained external DEMON package emits a progress bar and timing
    # line. Keep benchmark logs deterministic and compact while retaining the
    # package's implementation rather than reproducing the algorithm here.
    # The NetworkX graph is the detector's input, built before the clock starts; only ``execute`` is timed.
    detector = Demon(
        graph=_networkx_graph(graph),
        epsilon=float(parameters.get("epsilon", 0.25)),
        min_community_size=int(parameters.get("min_community_size", 2)),
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        with timing.active().measure("demon.Demon.execute"):
            result = detector.execute()
    return [list(map(int, community)) for community in result]


def _agmfit(
    graph: ig.Graph, _k: int, _resolution: float, _seed: int, _parameters: dict[str, Any]
) -> list[list[int]]:
    """Run an optional AGMfit implementation when one is installed.

    The Yang--Leskovec implementation is not part of the Python standard
    scientific stack and has no stable package/API.  We therefore require an
    explicit ``agmfit.AGMfit`` provider and fail closed when it is absent.
    This prevents a CliquePerc or Leiden result from being mislabeled as AGMfit.
    """
    try:
        from agmfit import AGMfit
    except (ImportError, AttributeError) as exc:  # pragma: no cover - optional
        raise MethodUnavailable(
            "AGMfit implementation is not installed; install a compatible "
            "agmfit.AGMfit provider to enable this adapter"
        ) from exc
    raise MethodUnavailable(
        "AGMfit provider was discovered but its API is not registered; "
        "configure an explicit adapter before running it"
    )


def _link_clustering(
    graph: ig.Graph, _k: int, _resolution: float, _seed: int, parameters: dict[str, Any]
) -> list[list[int]]:
    """Run CDlib's isolated Ahn--Bagrow--Lehmann Link Communities adapter."""
    cover, _metadata = cdlib_env.run_link_clustering(
        graph, timeout_seconds=float(parameters.get("timeout_seconds", 90.0))
    )
    return cover


def _mmsb(
    graph: ig.Graph, _k: int, _resolution: float, _seed: int, _parameters: dict[str, Any]
) -> list[list[int]]:
    """Run an optional MMSB provider; never substitute a different model."""
    try:
        from mmsb import MMSB
    except (ImportError, AttributeError) as exc:  # pragma: no cover - optional
        raise MethodUnavailable(
            "MMSB implementation is not installed; install a compatible mmsb.MMSB provider"
        ) from exc
    raise MethodUnavailable(
        "MMSB provider was discovered but its API is not registered; "
        "configure an explicit adapter before running it"
    )


def _run_cdlib_worker(
    graph: ig.Graph,
    *,
    method: str,
    parameters: dict[str, Any],
    seed: int,
    timeout_seconds: float = 90.0,
) -> list[list[int]]:
    """Run one CDlib method without importing its conflicting igraph package (see ``cdlib_env``)."""
    return cdlib_env.run_cdlib_method(
        graph, method=method, parameters=parameters, seed=seed, timeout_seconds=timeout_seconds
    )


def _infomap(
    graph: ig.Graph, _k: int, _resolution: float, seed: int, parameters: dict[str, Any]
) -> list[list[int]]:
    """Run the optional Infomap Python binding and return node communities."""
    try:
        from infomap import Infomap
    except (ImportError, AttributeError):
        # python-igraph ships the same open-source Infomap implementation as a
        # graph method.  Use it when the standalone binding is absent, while
        # recording the fallback explicitly rather than claiming the Python
        # binding was installed.  It draws from igraph's generator: seed it.
        try:
            with seeded_igraph_rng(int(seed)):
                with timing.active().measure("igraph.community_infomap"):
                    result = graph.community_infomap()
        except Exception as exc:  # pragma: no cover - native build dependent
            raise MethodUnavailable(
                "Infomap is unavailable: neither the Python binding nor the igraph implementation could run"
            ) from exc
        return partition_to_cover_lists(result)
    options = str(parameters.get("options", "--two-level --silent"))
    try:
        infomap = Infomap(options)
        for source, target in graph.get_edgelist():
            infomap.add_link(int(source), int(target))
        with timing.active().measure("infomap.Infomap.run"):
            infomap.run()
        communities: dict[int, list[int]] = {}
        for node in infomap.tree:
            leaf_marker = getattr(node, "is_leaf", False)
            is_leaf = leaf_marker() if callable(leaf_marker) else bool(leaf_marker)
            if is_leaf:
                module_marker = getattr(node, "module_id", None)
                module_id = module_marker() if callable(module_marker) else module_marker
                node_marker = getattr(node, "node_id", None)
                node_id = node_marker() if callable(node_marker) else node_marker
                if module_id is None or node_id is None:
                    continue
                communities.setdefault(int(module_id), []).append(int(node_id))
        return list(communities.values())
    except Exception as exc:  # pragma: no cover - optional API/version drift
        raise MethodUnavailable(
            f"installed Infomap binding could not be executed by this adapter: {exc}"
        ) from exc


def _angel(
    graph: ig.Graph, _k: int, _resolution: float, seed: int, parameters: dict[str, Any]
) -> list[list[int]]:
    cover = _run_cdlib_worker(
        graph,
        method="angel",
        parameters={
            "threshold": float(parameters.get("threshold", 0.6)),
            "min_community_size": int(parameters.get("min_community_size", 3)),
        },
        seed=seed,
        timeout_seconds=float(parameters.get("timeout_seconds", 90.0)),
    )
    if not cover:
        raise MethodUnavailable(
            "ANGEL produced no valid communities for this input window"
        )
    return cover


def _codeseg_proxy_runner(method_name: str) -> Callable[
    [ig.Graph, int, float, int, dict[str, Any]], list[list[int]]
]:
    """Build a lazy adapter for a method implemented by the SNAP runner.

    The CoDeSEG reproduction already owns the external-runtime details for
    these methods.  Reusing its dispatcher keeps one implementation per
    method while making the general ``run_method(METHODS[name], ...)`` API
    complete.  Ground truth is passed explicitly by the benchmark because
    several upstream command wrappers need it to construct their input files.
    """

    def runner(
        graph: ig.Graph,
        _max_memberships: int,
        _resolution: float,
        seed: int,
        parameters: dict[str, Any],
    ) -> list[list[int]]:
        ground_truth = parameters.get("_ground_truth")
        if ground_truth is None and method_name not in GROUND_TRUTH_REQUIRED_METHODS:
            ground_truth = []
        if not isinstance(ground_truth, Sequence):
            raise MethodUnavailable(
                f"{method_name} requires ground_truth=... when called through run_method"
            )
        try:
            from hedonic.experiments.overlapping.codeseg_reproduction import (
                _run_method as run_codeseg_method,
            )
        except ImportError as exc:  # pragma: no cover - package installation issue
            raise MethodUnavailable(
                f"the {method_name} reproduction dispatcher is unavailable"
            ) from exc
        forwarded = {
            key: value
            for key, value in parameters.items()
            if not str(key).startswith("_")
        }
        cover, _metadata = run_codeseg_method(
            method_name,
            graph,
            ground_truth,
            int(seed),
            forwarded,
        )
        return cover

    return runner


METHODS: dict[str, MethodAdapter] = {
    "hedonic_local": MethodAdapter(
        name="hedonic_local",
        family="hedonic local-moving",
        implementation="hedonic.Game.community_hedonic (hedonic C core)",
        parameters={
            "allow_isolation": True,
            "rng": "community_hedonic(seed=seed)",
        },
        scalability="Native C core; local moving to an equilibrium.",
        output_kind="VertexCover converted directly to node cover",
        install_requirement=None,
        runner=_hedonic_local,
    ),
    "clique_percolation": MethodAdapter(
        name="clique_percolation",
        family="clique-based / clique percolation",
        implementation="networkx.algorithms.community.k_clique_communities",
        parameters={"clique_size": 3, "max_communities": 50_000},
        scalability="Can grow exponentially with clique density; protected by timeout.",
        output_kind="Direct overlapping node cover from clique components",
        install_requirement='pip install "hedonic[experiments]" (networkx)',
        runner=_clique_percolation,
    ),
    "demon": MethodAdapter(
        name="demon",
        family="local expansion / DEMON",
        implementation="demon.Demon external package",
        parameters={"epsilon": 0.25, "min_community_size": 2},
        scalability="Local expansion can be costly on high-degree social graphs.",
        output_kind="Direct overlapping node cover returned by the DEMON package",
        install_requirement='pip install "hedonic[experiments]" (demon)',
        runner=_demon,
    ),
    "agmfit": MethodAdapter(
        name="agmfit",
        family="generative affiliation / AGMfit",
        implementation="agmfit.AGMfit (Yang–Leskovec ICDM 2012)",
        parameters={
            "article": "Yang & Leskovec, Community-Affiliation Graph Model for Overlapping Network Community Detection (ICDM 2012), §VI",
            "selection": "AGMfit's detector is evaluated on ground-truth-derived subnetworks",
        },
        scalability="Optional external implementation; fail-closed when unavailable.",
        output_kind="Overlapping node cover from AGMfit",
        install_requirement="Install a compatible external AGMfit implementation separately; no package is bundled.",
        runner=_agmfit,
    ),
    "link_clustering": MethodAdapter(
        name="link_clustering",
        family="edge-based / Link Clustering",
        implementation="cdlib.algorithms.hierarchical_link_community (isolated worker)",
        parameters={
            "article": "Yang & Leskovec (ICDM 2012), §VI baselines",
            "cdlib_api": "hierarchical_link_community",
            "output_conversion": "edge communities projected to unique node sets",
        },
        scalability="Pinned CDlib worker; edge dendrogram cut is selected by partition density.",
        output_kind="Overlapping node cover from edge communities",
        install_requirement="Run hedonic-exp codeseg-setup --all (the locked CDlib environment); CDlib is never imported in the hedonic process.",
        runner=_link_clustering,
    ),
    "mmsb": MethodAdapter(
        name="mmsb",
        family="mixed-membership stochastic block model",
        implementation="mmsb.MMSB (external provider)",
        parameters={
            "article": "Yang & Leskovec (ICDM 2012), §VI baselines",
        },
        scalability="Optional external implementation; fail-closed when unavailable.",
        output_kind="Overlapping node cover from mixed memberships",
        install_requirement="Install and configure a compatible MMSB implementation separately; no package is bundled.",
        runner=_mmsb,
    ),
    "infomap": MethodAdapter(
        name="infomap",
        family="flow-based / Infomap",
        implementation="igraph.Graph.community_infomap (standalone infomap binding when installed)",
        parameters={
            "options": "--two-level --silent",
            "article": "Yang & Leskovec (ICDM 2012), §VI; results omitted as non-competitive",
        },
        scalability="python-igraph's Infomap implementation, with the standalone binding preferred when installed.",
        output_kind="Overlapping node cover (when hierarchical modules overlap by node)",
        install_requirement="python-igraph includes Infomap; install the standalone binding only when its exact CLI/API is required.",
        runner=_infomap,
    ),
    "angel": MethodAdapter(
        name="angel",
        family="ego-network / ANGEL",
        implementation="cdlib.algorithms.angel (isolated CDlib 0.4.0 worker)",
        parameters={"threshold": 0.6, "min_community_size": 3},
        scalability="Pinned CDlib worker; bounded by the benchmark timeout.",
        output_kind="Overlapping node cover returned by ANGEL",
        install_requirement="Run hedonic-exp codeseg-setup --all (the locked CDlib environment).",
        runner=_angel,
    ),
}


def _register_literature_proxy(
    name: str,
    *,
    family: str,
    implementation: str,
    install_requirement: str,
    scalability: str = "Uses the corresponding CoDeSEG/SNAP reproduction adapter.",
    output_kind: str = "Normalized node cover",
    parameters: dict[str, Any] | None = None,
) -> None:
    """Register one dispatcher-backed method without duplicating its code."""
    if name in METHODS:
        return
    METHODS[name] = MethodAdapter(
        name=name,
        family=family,
        implementation=implementation,
        parameters=dict(parameters or {}),
        scalability=scalability,
        output_kind=output_kind,
        install_requirement=install_requirement,
        runner=_codeseg_proxy_runner(name),
    )


_register_literature_proxy(
    "codeseg",
    family="structural-entropy game / overlapping",
    implementation="SELGroup/CoDeSEG C++ executable",
    install_requirement="Run codeseg-setup --all --build-native or provide HEDONIC_CODESEG_BIN.",
)
_register_literature_proxy(
    "slpa",
    family="label propagation / overlapping",
    implementation="isolated CDlib SLPA worker",
    install_requirement="Run codeseg-setup --all to materialize the locked CDlib worker.",
    parameters={"iterations": 21, "threshold": 0.01},
)
_register_literature_proxy(
    "bigclam",
    family="nonnegative matrix factorization / overlapping",
    implementation="SNAP BigCLAM executable",
    install_requirement="Provide --bigclam-bin or HEDONIC_BIGCLAM_BIN.",
    parameters={"communities": 25_000},
)
_register_literature_proxy(
    "ncgame",
    family="non-cooperative game / overlapping",
    implementation="NcGame reference command",
    install_requirement="Provide an NcGame command template.",
)
_register_literature_proxy(
    "fox",
    family="triangle heuristic / overlapping",
    implementation="LazyFox/FOX executable",
    install_requirement="Provide --fox-bin or HEDONIC_FOX_BIN.",
)
_register_literature_proxy(
    "louvain",
    family="modularity / disjoint control",
    implementation="igraph.Graph.community_multilevel",
    install_requirement="python-igraph",
)
_register_literature_proxy(
    "der",
    family="diffusion entropy reduction / disjoint control",
    implementation="isolated CDlib DER worker",
    install_requirement="Run codeseg-setup --all to materialize the locked DER worker.",
)
_register_literature_proxy(
    "leiden",
    family="modularity / disjoint control",
    implementation="igraph.Graph.community_leiden",
    install_requirement="python-igraph",
)
_register_literature_proxy(
    "flpa",
    family="fast label propagation / disjoint control",
    implementation="CoDeSEG FLPA adapter",
    install_requirement="CoDeSEG reproduction checkout",
)
_register_literature_proxy(
    "neo_kmeans",
    family="graph k-means / overlapping",
    implementation="NEO-K-Means Graph Clustering executable",
    install_requirement="Provide --neo-bin or the configured NEO runtime.",
    parameters={"clusters": 64, "alpha": 0.2, "beta": 0.0, "sigma": 0.0},
)
_register_literature_proxy(
    "nise",
    family="seed-set expansion / overlapping",
    implementation="official NISE MATLAB/Octave source",
    install_requirement="Provide --nise-source and an Octave runtime.",
    parameters={"communities": 64},
)
_register_literature_proxy(
    "sse",
    family="seed-set expansion / overlapping",
    implementation="official SSE/NISE MATLAB/Octave source",
    install_requirement="Provide --nise-source and an Octave runtime.",
    parameters={"communities": 64},
)
_register_literature_proxy(
    "qoce",
    family="quadratic optimization clique expansion / overlapping",
    implementation="PanShi2016/QOCE MATLAB/Octave source",
    install_requirement="Provide --qoce-source, clique finder, and Octave.",
)
_register_literature_proxy(
    "svi",
    family="stochastic variational inference / overlapping",
    implementation="premgopalan/svinet executable",
    install_requirement="Provide --svi-bin or HEDONIC_SVI_BIN.",
    parameters={"communities": 64, "max_iterations": 10},
)
_register_literature_proxy(
    "essc",
    family="statistical significance extraction / overlapping",
    implementation="jdwilson4/ESSC R package",
    install_requirement="Provide ESSC R library and Rscript.",
)


DEFAULT_METHODS: tuple[str, ...] = tuple(
    name for name in LITERATURE_METHODS if name in METHODS
)


def effective_resolution(
    adapter: MethodAdapter, graph: ig.Graph, requested_resolution: float
) -> float:
    """Return the detector resolution, including a fixed density rule.

    An adapter with ``density_resolution_multiplier`` ignores the requested
    resolution and runs at ``min(density * multiplier, 1)`` (no registered
    adapter has one in 2.0; the 1.x multi-phase variants did).
    """
    multiplier = adapter.density_resolution_multiplier
    if multiplier is None:
        return requested_resolution
    return min(graph.density() * multiplier, 1.0)


def method_availability() -> dict[str, dict[str, Any]]:
    """Report every adapter and optional dependency without running a method."""
    result: dict[str, dict[str, Any]] = {}
    for name, adapter in METHODS.items():
        available = True
        reason = None
        dependency = method_dependency_identity(name)
        if name in {"agmfit", "mmsb"}:
            # No stable Python API for these historical implementations is
            # bundled or registered.  Keep them explicitly unavailable even
            # if an unrelated distribution happens to expose the same module
            # name; silently guessing an API would invalidate the benchmark.
            available = False
            reason = (
                f"{name} adapter has no registered compatible implementation; "
                "install/configure an explicit provider before running"
            )
        if name in {"link_clustering", "angel"}:
            available = bool(
                isinstance(dependency, dict)
                and dependency.get("version") == "0.4.0"
                and dependency.get("isolated_environment_ready") is True
                and dependency.get("implementation_sha256")
            )
            if not available:
                reason = (
                    "isolated CDlib 0.4.0 worker is unavailable; "
                    "run `hedonic-exp codeseg-setup --all` before enabling this adapter"
                )
        if available and name in EXTERNAL_DISTRIBUTIONS and name not in {"link_clustering", "angel", "infomap"} and (
            not isinstance(dependency, dict)
            or dependency.get("version") is None
            or dependency.get("implementation_belongs_to_distribution") is not True
            or not dependency.get("implementation_sha256")
        ):
            available = False
            reason = (
                f"{EXTERNAL_DISTRIBUTIONS[name]} exact implementation is unavailable"
            )
        if name == "infomap":
            native_available = hasattr(ig.Graph, "community_infomap")
            binding_available = bool(
                isinstance(dependency, dict)
                and dependency.get("version") is not None
                and dependency.get("implementation_belongs_to_distribution") is True
            )
            available = native_available or binding_available
            if not available:
                reason = "neither python-igraph's community_infomap nor the standalone Infomap binding is available"
        if name in DISPATCHER_METHODS:
            # Reuse the authoritative runtime checks from the SNAP runner.
            # This keeps ``--list-methods`` and the library registry honest
            # about binaries, Octave sources, and isolated workers without
            # importing those optional environments at module import time.
            try:
                from hedonic.experiments.overlapping.codeseg_reproduction import (
                    _method_preflight,
                )

                preflight = _method_preflight(name, dict(adapter.parameters))
                available = preflight.get("status") in {"ready", "configured"}
                reason = preflight.get("reason")
            except Exception as exc:
                available = False
                reason = f"dispatcher preflight failed: {exc}"
        result[name] = {
            "available": available,
            "reason": reason,
            "family": adapter.family,
            "implementation": adapter.implementation,
            "parameters": adapter.parameters,
            "scalability": adapter.scalability,
            "output_kind": adapter.output_kind,
            "install_requirement": adapter.install_requirement,
            "dependency": dependency,
        }
    return result


def resolve_methods(names: str | Sequence[str] | None) -> list[MethodAdapter]:
    """Resolve a comma-list (or sequence) and reject unknown names early."""
    if names is None:
        selected = list(DEFAULT_METHODS)
    elif isinstance(names, str):
        if names.strip().lower() in {"all", "literature"}:
            selected = list(DEFAULT_METHODS)
        else:
            selected = [canonical_method_name(part) for part in names.split(",") if part.strip()]
    else:
        selected = [canonical_method_name(str(name)) for name in names if str(name).strip()]
        if len(selected) == 1 and selected[0] in {"all", "literature"}:
            selected = list(DEFAULT_METHODS)
    if not selected:
        raise ValueError("At least one benchmark method is required")
    removed = removed_method_error(selected)
    if removed is not None:
        raise removed
    unknown = [name for name in selected if name not in METHODS]
    if unknown:
        raise ValueError(
            f"Unknown method(s): {', '.join(unknown)}; choose from {', '.join(METHODS)}"
        )
    return [METHODS[name] for name in selected]


def run_method(
    adapter: MethodAdapter,
    graph: ig.Graph,
    *,
    max_memberships: int,
    resolution: float,
    seed: int,
    parameters: dict[str, Any] | None = None,
    initial_membership: list[int] | list[list[int]] | None = None,
    ground_truth: Sequence[Sequence[int]] | None = None,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Run an adapter and return a valid normalized cover with provenance.

    ``ground_truth`` is forwarded only to dispatcher-backed implementations
    that need it to construct their native input files or initialization.
    Detectors themselves never receive the evaluation score, so supplying a
    ground truth does not change metric computation or post-hoc scoring.
    """
    params = {**adapter.parameters, **(parameters or {})}
    runner_params = {**params}
    if initial_membership is not None:
        runner_params["_initial_membership"] = initial_membership
    if ground_truth is not None:
        runner_params["_ground_truth"] = ground_truth
    detector_resolution = effective_resolution(adapter, graph, resolution)
    # The detector clock (``timing``) is the run's own when this is called by the benchmark runner, else a fresh one.
    # Normalising and verifying the cover below happen after it stopped.
    outer = timing.current()
    with (contextlib.nullcontext(outer) if outer is not None else timing.detector_run()) as clock:
        detector_output = adapter.runner(
            graph, max_memberships, detector_resolution, seed, runner_params
        )
    if not clock.used:
        raise RuntimeError(f"{adapter.name} never started the detector clock (see timing.py)")
    # Record the resolution the detector actually used: the Hedonic adapters
    # replace the density rule by an explicit absolute resolution, or scale it.
    if adapter.runner is _hedonic_local:
        detector_resolution = hedonic_resolution(params, detector_resolution)
    elif "absolute_resolution" in params:
        detector_resolution = float(params["absolute_resolution"])
    if isinstance(detector_output, DetectorOutput):
        raw_cover = detector_output.cover
        final_memberships = detector_output.final_memberships
    else:
        raw_cover = detector_output
        final_memberships = None
    cover, normalization = normalize_cover(raw_cover, graph.vcount())
    if not cover:
        raise RuntimeError(f"{adapter.name} produced no valid communities")
    if adapter.name.startswith("hedonic_"):
        exact_final_cover = _cover_from_memberships(
            final_memberships, graph.vcount()
        )
        if exact_final_cover is None:
            raise RuntimeError(
                f"{adapter.name} did not expose valid exact final memberships"
            )
        if exact_final_cover != cover:
            raise RuntimeError(
                f"{adapter.name} final memberships disagree with its scoring cover"
            )
    return cover, {
        "runtime_seconds": clock.seconds,
        "timing_scope": timing.TIMING_SCOPE,
        "method": adapter.name,
        "family": adapter.family,
        "implementation": adapter.implementation,
        "parameters": params,
        "requested_resolution": resolution,
        "effective_resolution": detector_resolution,
        "seed": seed,
        "initial_membership_supplied": initial_membership is not None,
        "final_memberships": final_memberships,
        "normalization": normalization,
        "dependency": method_dependency_identity(adapter.name),
        "directed_input_converted_to_undirected": (
            graph.is_directed() and adapter.name in {"clique_percolation", "demon"}
        ),
    }


def run_method_by_name(
    name: str,
    graph: ig.Graph,
    *,
    max_memberships: int,
    resolution: float,
    seed: int,
    parameters: dict[str, Any] | None = None,
    initial_membership: list[int] | list[list[int]] | None = None,
    ground_truth: Sequence[Sequence[int]] | None = None,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Run one registered literature method by its public name.

    This is the convenience facade for notebooks and scripts: callers do not
    need to index ``METHODS`` or know which implementation module owns the
    adapter.  Hyphenated/spaced aliases accepted by :func:`resolve_methods`
    are accepted here as well.
    """
    adapters = resolve_methods(name)
    if len(adapters) != 1:
        raise ValueError("run_method_by_name accepts exactly one method name")
    return run_method(
        adapters[0],
        graph,
        max_memberships=max_memberships,
        resolution=resolution,
        seed=seed,
        parameters=parameters,
        initial_membership=initial_membership,
        ground_truth=ground_truth,
    )
