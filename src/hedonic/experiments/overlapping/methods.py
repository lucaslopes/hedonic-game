"""Method adapters for reproducible overlapping-community benchmarks.

Only the two hedonic methods live in the core detector binding.  Independent
baselines are delegated to maintained experiment-only libraries, so their
availability and installation requirements can be recorded rather than hidden.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import inspect
import io
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import igraph as ig
from hedonic import Game
from hedonic.experiments.overlapping.metrics import partition_to_cover_lists
from hedonic.experiments.overlapping.snap import canonicalize_cover
from hedonic.utils import sample_uniform_ints


class MethodUnavailable(RuntimeError):
    """A method's optional implementation is not installed."""


EXTERNAL_DISTRIBUTIONS = {"cpm": "networkx", "demon": "demon"}
EXTERNAL_IMPLEMENTATIONS = {
    "cpm": (
        "networkx.algorithms.community.kclique",
        "k_clique_communities",
    ),
    "demon": ("demon.alg.Demon", "Demon"),
}


def method_dependency_identity(name: str) -> dict[str, Any] | None:
    distribution = EXTERNAL_DISTRIBUTIONS.get(name)
    if distribution is None:
        return None
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
    """Detector cover plus optional raw per-vertex memberships.

    Native overlapping Leiden returns a nested membership vector.  Keeping it
    beside the normalized cover lets the benchmark record exactly what the
    native call returned, without changing the public cover adapter contract
    used by external baselines.
    """

    cover: list[list[int]]
    pre_cleanup_memberships: list[list[int]] | None = None
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


def _hedonic_local(
    graph: ig.Graph, k: int, resolution: float, seed: int, parameters: dict[str, Any]
) -> DetectorOutput:
    # lucas-igraph's community_leiden binding does not expose a ``seed``
    # argument.  Set igraph's Python RNG explicitly so a benchmark seed is a
    # real reproducibility control rather than metadata only. Benchmark calls
    # run in an isolated detector process when timeouts are enabled.
    ig.set_random_number_generator(random.Random(seed))
    result = Game(graph).community_hedonic(
        resolution=resolution,
        max_memberships=max(2, k),
        local_move_only=True,
        n_iterations=-1,
        allow_isolation=bool(parameters.get("allow_isolation", True)),
        initial_membership=parameters.get("_initial_membership"),
        seed=seed,
        ensure_equilibrium=bool(parameters.get("ensure_equilibrium", True)),
    )
    return DetectorOutput(
        cover=partition_to_cover_lists(result),
        pre_cleanup_memberships=_pre_cleanup_memberships(result),
        final_memberships=_final_memberships(result),
    )


def _hedonic_multiphase(
    graph: ig.Graph, k: int, resolution: float, seed: int, parameters: dict[str, Any]
) -> DetectorOutput:
    ig.set_random_number_generator(random.Random(seed))
    result = Game(graph).community_hedonic(
        resolution=resolution,
        max_memberships=max(2, k),
        local_move_only=False,
        n_iterations=-1,
        allow_isolation=bool(parameters.get("allow_isolation", True)),
        initial_membership=parameters.get("_initial_membership"),
        seed=seed,
        ensure_equilibrium=bool(parameters.get("ensure_equilibrium", True)),
    )
    return DetectorOutput(
        cover=partition_to_cover_lists(result),
        pre_cleanup_memberships=_pre_cleanup_memberships(result),
        final_memberships=_final_memberships(result),
    )


def _pre_cleanup_memberships(result) -> list[list[int]] | None:
    """Extract the preserved first native call before equilibrium cleanup."""
    preserved = getattr(result, "_hedonic_raw_memberships", None)
    if preserved is None:
        return None
    return [list(map(int, labels)) for labels in preserved]


def _final_memberships(result) -> list[list[int]]:
    """Extract the exact per-vertex rows returned by the final native call."""
    membership = getattr(result, "membership", None)
    if membership is None:
        return []
    if membership and isinstance(membership[0], (list, tuple)):
        return [list(map(int, labels)) for labels in membership]
    return [[int(label)] for label in membership]


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
    # CPM and DEMON operate on ordinary undirected simple graphs.
    # The conversion is recorded in the method metadata by the benchmark.
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.vcount()))
    nx_graph.add_edges_from(graph.get_edgelist())
    return nx_graph


def _cpm(
    graph: ig.Graph, _k: int, _resolution: float, _seed: int, parameters: dict[str, Any]
) -> list[list[int]]:
    """NetworkX clique-percolation baseline (an overlapping node cover)."""
    try:
        from networkx.algorithms.community.kclique import k_clique_communities
    except ImportError as exc:  # pragma: no cover - availability handles this
        raise MethodUnavailable("networkx is not installed") from exc
    clique_size = int(parameters.get("clique_size", 3))
    max_communities = int(parameters.get("max_communities", 50_000))
    communities: list[list[int]] = []
    for community in k_clique_communities(_networkx_graph(graph), clique_size):
        communities.append(sorted(int(vertex) for vertex in community))
        if len(communities) >= max_communities:
            break
    return communities


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
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = Demon(
            graph=_networkx_graph(graph),
            epsilon=float(parameters.get("epsilon", 0.25)),
            min_community_size=int(parameters.get("min_community_size", 2)),
        ).execute()
    return [list(map(int, community)) for community in result]


METHODS: dict[str, MethodAdapter] = {
    "hedonic_local": MethodAdapter(
        name="hedonic_local",
        family="hedonic local-moving",
        implementation="hedonic.Game.community_hedonic (lucas-igraph)",
        parameters={
            "local_move_only": True,
            "n_iterations": -1,
            "allow_isolation": True,
            "ensure_equilibrium": True,
            "igraph_rng": "random.Random(seed)",
        },
        scalability="Native igraph binding; standard profile bounds graph size.",
        output_kind="VertexCover converted directly to node cover",
        install_requirement=None,
        runner=_hedonic_local,
    ),
    "hedonic_multiphase": MethodAdapter(
        name="hedonic_multiphase",
        family="hedonic / Leiden multi-phase",
        implementation="hedonic.Game.community_hedonic (lucas-igraph)",
        parameters={
            "local_move_only": False,
            "n_iterations": -1,
            "allow_isolation": True,
            "ensure_equilibrium": True,
            "igraph_rng": "random.Random(seed)",
            "resolution_rule": "min(graph.density() * 1, 1)",
        },
        scalability="Native igraph binding; refinement/aggregation can be slower.",
        output_kind="VertexCover converted directly to node cover",
        install_requirement=None,
        runner=_hedonic_multiphase,
        density_resolution_multiplier=1.0,
    ),
    "hedonic_multiphase_x10": MethodAdapter(
        name="hedonic_multiphase_x10",
        family="hedonic / Leiden multi-phase",
        implementation="hedonic.Game.community_hedonic (lucas-igraph)",
        parameters={
            "local_move_only": False,
            "n_iterations": -1,
            "allow_isolation": True,
            "ensure_equilibrium": True,
            "igraph_rng": "random.Random(seed)",
            "resolution_rule": "min(graph.density() * 10, 1)",
        },
        scalability="Native igraph binding; refinement/aggregation can be slower.",
        output_kind="VertexCover converted directly to node cover",
        install_requirement=None,
        runner=_hedonic_multiphase,
        density_resolution_multiplier=10.0,
    ),
    "hedonic_multiphase_x100": MethodAdapter(
        name="hedonic_multiphase_x100",
        family="hedonic / Leiden multi-phase",
        implementation="hedonic.Game.community_hedonic (lucas-igraph)",
        parameters={
            "local_move_only": False,
            "n_iterations": -1,
            "allow_isolation": True,
            "ensure_equilibrium": True,
            "igraph_rng": "random.Random(seed)",
            "resolution_rule": "min(graph.density() * 100, 1)",
        },
        scalability="Native igraph binding; refinement/aggregation can be slower.",
        output_kind="VertexCover converted directly to node cover",
        install_requirement=None,
        runner=_hedonic_multiphase,
        density_resolution_multiplier=100.0,
    ),
    "cpm": MethodAdapter(
        name="cpm",
        family="clique-based / clique percolation",
        implementation="networkx.algorithms.community.k_clique_communities",
        parameters={"clique_size": 3, "max_communities": 50_000},
        scalability="Can grow exponentially with clique density; protected by timeout.",
        output_kind="Direct overlapping node cover from clique components",
        install_requirement='pip install "hedonic[experiments]" (networkx)',
        runner=_cpm,
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
}


DEFAULT_METHODS: tuple[str, ...] = tuple(METHODS)


def effective_resolution(
    adapter: MethodAdapter, graph: ig.Graph, requested_resolution: float
) -> float:
    """Return the detector resolution, including fixed hedonic density rules.

    The three paper hedonic variants deliberately ignore a user resolution
    sweep: their identities are the density multipliers 1, 10, and 100.  The
    resulting value is used for detection, cache keys, metrics, and records.
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
        if name in EXTERNAL_DISTRIBUTIONS and (
            not isinstance(dependency, dict)
            or dependency.get("version") is None
            or dependency.get("implementation_belongs_to_distribution") is not True
            or not dependency.get("implementation_sha256")
        ):
            available = False
            reason = (
                f"{EXTERNAL_DISTRIBUTIONS[name]} exact implementation is unavailable"
            )
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
        selected = [part.strip() for part in names.split(",") if part.strip()]
    else:
        selected = [str(name).strip() for name in names if str(name).strip()]
    if not selected:
        raise ValueError("At least one benchmark method is required")
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
) -> tuple[list[list[int]], dict[str, Any]]:
    """Run an adapter and return a valid normalized cover with provenance."""
    params = {**adapter.parameters, **(parameters or {})}
    started = time.monotonic()
    runner_params = {**params}
    if initial_membership is not None:
        runner_params["_initial_membership"] = initial_membership
    detector_output = adapter.runner(
        graph, max_memberships, resolution, seed, runner_params
    )
    if isinstance(detector_output, DetectorOutput):
        raw_cover = detector_output.cover
        pre_cleanup_memberships = detector_output.pre_cleanup_memberships
        final_memberships = detector_output.final_memberships
    else:
        raw_cover = detector_output
        pre_cleanup_memberships = None
        final_memberships = None
    cover, normalization = normalize_cover(raw_cover, graph.vcount())
    if not cover:
        raise RuntimeError(f"{adapter.name} produced no valid communities")
    if adapter.name.startswith("hedonic_"):
        if pre_cleanup_memberships is None:
            raise RuntimeError(
                f"{adapter.name} did not preserve exact pre-cleanup memberships"
            )
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
        "runtime_seconds": time.monotonic() - started,
        "method": adapter.name,
        "family": adapter.family,
        "implementation": adapter.implementation,
        "parameters": params,
        "seed": seed,
        "initial_membership_supplied": initial_membership is not None,
        "pre_cleanup_memberships": pre_cleanup_memberships,
        "final_memberships": final_memberships,
        "ensure_equilibrium": bool(params.get("ensure_equilibrium", False)),
        "normalization": normalization,
        "dependency": method_dependency_identity(adapter.name),
        "directed_input_converted_to_undirected": (
            graph.is_directed() and adapter.name in {"cpm", "demon"}
        ),
    }
