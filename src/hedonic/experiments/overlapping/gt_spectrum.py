"""SNAP ground-truth robustness spectrum (``hedonic-exp overlapping-gt-spectrum``).

A study of the *supplied* overlapping ground-truth covers of the SNAP
networks. It shares the detector worker and the artifact stores of
:mod:`ground_truth_robustness` and adds:

* cache **discovery** of graph/cover pairs (present, missing, unsupported);
  a missing pair is downloaded (``snap.ensure_pair``) when provisioning is on:
  ``--provision``, the TOML key ``provision``, or by default for a run that
  names its pairs explicitly (``pairs``/``--pairs``, as every queue step does);
  ``--no-provision`` never downloads;
* a **resolution-spectrum audit** of every supplied cover: the fraction of
  vertices whose best-response regret is within the numerical tolerance, per
  action policy (fixed-label and open-label), with mean/max positive regret,
  profitable-vertex fraction and Nash status;
* multi-seed **local-moving** runs of ``Game.community_hedonic`` started from
  the same complete, canonical ground-truth cover (to an equilibrium), each
  final cover independently audited and scored against its exact start with the
  shared accuracy vector;
* resumable, identity-bound records, coverage accounting and figures;
* optional **explicit detector settings** (TOML ``detector_settings``): per
  pair, a list of (gamma, cap) settings (``phase`` may only be ``local``: the
  multilevel phase is not available in hedonic 2.0) that replaces the
  resolution grid x GT-multiplicity cap.  When a setting's cap is below the
  supplied cover's largest multiplicity the start is the deterministic
  **cap projection** of the GT (:func:`project_memberships`); returns are scored
  against the full reference and against that projected start.  Without the
  key every record, identity and summary is unchanged.

The plotted "fraction in equilibrium" is a vertex-level diagnostic under the
stated action policy, cap and tolerance; a cover is a Nash equilibrium only when
that fraction equals one.  Everything is descriptive of supplied SNAP metadata
and the analysed graph transform, not of latent truth or global optima.

Identity. A stored record is reused only when it was written by the same code:
the SHA-256 of the package source files the study runs (located through their
modules, keyed by their path inside the package), the hedonic version and the
SHA-256 of the compiled core (:func:`tracked_digest`). Packaged configs run
unlocked: hedonic 2.0 has no protocol lock file, and the manifest says so.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import statistics
import time
import tomllib
from pathlib import Path
from typing import Any, Callable, Iterable

import hedonic
from hedonic.experiments import runtime
from hedonic.experiments.config import (
    DEFAULT_NETWORKS_DIR,
    OVERLAPPING_ARTIFACTS_DIR,
    RUNS_ROOT,
    expand_path,
)
from hedonic.experiments.resources import config_path as resolve_config
from hedonic.experiments.overlapping import ground_truth_data as gtd
from hedonic.experiments.overlapping import ground_truth_robustness as v3
from hedonic.experiments.overlapping import capacity
from hedonic.experiments.overlapping import robustness as rb
from hedonic.experiments.overlapping import snap
from hedonic.experiments.overlapping import stability as stab
from hedonic.experiments.overlapping.execution import run_in_subprocess
from hedonic.experiments.overlapping.metrics import (
    _cover_sets,
    evaluate_cover,
    omega_index,
    overlapping_normalized_mutual_information_lfk,
)

SCHEMA_VERSION = 1
PROTOCOL_NAME = "overlapping-gt-spectrum-v1"
ANALYSIS_PIPELINE = "bounded_then_common_undirected_simple_then_covered_induced_v1"
DEFAULT_CONFIG = "overlapping-gt-spectrum"
# The registered 131-point audit grid (0, 0.01, ..., 1 and 33 log-spaced points from 1e-4 to 1, shared points once),
# equal point for point to the canonical grid of the 1.x protocol lock.
AUDIT_GRID = "0:1:101+geom:1e-4:1:33"
DEFAULT_OUTPUT_DIR = OVERLAPPING_ARTIFACTS_DIR / "gt_spectrum_v1"
LOCK_NOTE = ("packaged configs run unlocked in hedonic 2.0: a record's identity is the digest of the package source "
             "files the study runs, the hedonic version and the SHA-256 of the compiled core")
COVER_VARIANTS = ("top5000", "all")
ACTION_POLICIES = ("fixed_labels", "open_labels")
DATASET_ORDER = ("amazon", "dblp", "livejournal", "youtube", "wikipedia", "orkut", "friendster")
FINAL_STATUSES = {"completed", "completed_non_equilibrium"}
RESOURCE_STATUSES = {"timeout", "failed", "invalid_worker_payload", "invalid_cover", "unsupported_cleanup"}
HEADLINE_METRICS = ("f1", "jaccard", "matching_f1", "node_micro_f1", "omega")
# The modules whose source defines this study's results (with the library's own modules, see tracked_file_hashes).
SOURCE_MODULES = (
    "hedonic.experiments.overlapping.gt_spectrum",
    "hedonic.experiments.overlapping.gt_spectrum_report",
    "hedonic.experiments.overlapping.ground_truth_robustness",
    "hedonic.experiments.overlapping.ground_truth_data",
    "hedonic.experiments.overlapping.robustness",
    "hedonic.experiments.overlapping.stability",
    "hedonic.experiments.overlapping.execution",
    "hedonic.experiments.overlapping.snap",
    "hedonic.experiments.overlapping.metrics",
)


# --------------------------------------------------------------------------- small helpers
def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _unique_tmp(path: Path) -> Path:
    """A temp name no other worker can share (content-addressed artifacts are written concurrently)."""
    import uuid

    return path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_tmp(path)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_gzip_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_tmp(path)
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"))
    temporary.replace(path)


def _persist_cover(output_dir: Path, cover: list[list[int]]) -> str:
    canonical = rb.canonicalize_cover(cover)
    digest = rb.cover_hash(canonical)
    if v3._load_cover_artifact(output_dir, digest) != canonical:
        _atomic_gzip_json(v3._cover_artifact_path(output_dir, digest), canonical)
    return digest


def _persist_memberships(output_dir: Path, memberships: list[list[int]], n_vertices: int) -> str:
    digest = v3._raw_cover_hash(rb.vertex_memberships_to_cover(memberships), n_vertices)
    if v3._load_membership_artifact(output_dir, digest) is None:
        _atomic_gzip_json(v3._membership_artifact_path(output_dir, digest), memberships)
    return digest


def _gamma_key(gamma: float) -> str:
    return f"{float(gamma):.10g}"


def _parse_grid(value: str | Iterable[float]) -> list[float]:
    """Resolution grid in [0, 1].

    Accepts ``START:STOP:COUNT`` (linear), ``geom:START:STOP:COUNT`` (log-spaced,
    both ends > 0), a comma list, a TOML list, or any of these joined with ``+``
    (for example ``0+geom:1e-4:1:9`` = 0 and a half-decade grid up to 1).
    """
    grid: list[float] = []
    if isinstance(value, str):
        for part in value.split("+"):
            part = part.strip()
            if part.startswith("geom:"):
                fields = part.split(":")
                if len(fields) != 4:
                    raise ValueError("geometric grids are geom:START:STOP:COUNT")
                lo, hi, count = float(fields[1]), float(fields[2]), int(fields[3])
                if lo <= 0 or hi <= 0 or count < 2:
                    raise ValueError("geometric grids need START, STOP > 0 and COUNT >= 2")
                grid.extend(math.exp(math.log(lo) + i * (math.log(hi) - math.log(lo)) / (count - 1)) for i in range(count))
            else:
                grid.extend(v3._parse_profile(part))
    else:
        grid = [float(v) for v in value]
    if not grid or any(not 0.0 <= g <= 1.0 for g in grid):
        raise ValueError("resolution grids must contain values in [0, 1]")
    unique = sorted({float(_gamma_key(g)) for g in grid})
    return unique


def _runtime_identity() -> dict[str, Any]:
    """The runtime part of a record's identity: the hedonic version and the compiled core (no paths)."""
    return {
        **runtime.build_identity(),
        "algorithm_identity": runtime.ALGORITHM_IDENTITY,
    }


def canonical_digest(digest: str | None) -> str | None:
    """The digest a code version's records are filed under. hedonic 2.0 keeps no equivalence file, so it is the digest
    itself (the name stays for the queue, which compares stored digests through it)."""
    return digest


def _package_root() -> Path:
    return Path(hedonic.__file__).resolve().parent


def _package_key(path: Path) -> str:
    """``hedonic/experiments/overlapping/gt_spectrum.py``: a source file's path inside the installed package."""
    return path.resolve().relative_to(_package_root().parent).as_posix()


def tracked_file_hashes() -> dict[str, str | None]:
    """SHA-256 of every source file the study's results depend on, keyed by its path inside the package.

    The study's modules (``SOURCE_MODULES``, located through the import system, never through a repository) and the
    library's own modules (``hedonic/*.py``: Game, metrics; ``hedonic/_game/*.py``). No absolute path is a key.
    """
    hashes: dict[str, str | None] = {}
    for module in SOURCE_MODULES:
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin:
            hashes[module] = None
            continue
        hashes[_package_key(Path(spec.origin))] = _sha256_file(Path(spec.origin))
    root = _package_root()
    for path in sorted([*root.glob("*.py"), *(root / "_game").glob("*.py")]):
        hashes[_package_key(path)] = _sha256_file(path)
    return dict(sorted(hashes.items()))


def _implementation_digest(tracked: dict[str, str | None], runtime_identity: dict[str, Any]) -> str:
    return _json_hash({"files": tracked, **runtime_identity})


def tracked_digest() -> str:
    """The code identity a stored record is compared with: the package source digest, the hedonic version and the core
    SHA-256 (the queue view uses it to tell finished work from work whose code has changed since)."""
    return _implementation_digest(tracked_file_hashes(), _runtime_identity())


# --------------------------------------------------------------------------- discovery
def _probe(spec: snap.SnapDatasetSpec, base: Path, variant: str, *, allow_pickles: bool) -> dict[str, Any]:
    """Locate one graph/cover pair with the loader's own precedence (pickles, then raw text)."""
    if not base.is_dir():
        return {"found": False, "missing": ["dataset directory"], "base": str(base)}
    attempts = []
    if allow_pickles:
        attempts.append(("trusted_archive_pickle", spec.graph_files, spec.cover_files.get(variant, ())))
    attempts.append(("streamed_raw_gzip", spec.raw_edge_files, spec.raw_cover_files.get(variant, ())))
    for kind, graph_names, cover_names in attempts:
        graph = snap._first_existing(base, graph_names)
        cover = snap._first_existing(base, cover_names)
        if graph is not None and cover is not None:
            return {
                "found": True,
                "source_kind": kind,
                "base": str(base),
                "graph_file": graph.relative_to(base).as_posix(),
                "cover_file": cover.relative_to(base).as_posix(),
                "bytes": graph.stat().st_size + cover.stat().st_size,
            }
    have_graph = snap._first_existing(base, (*spec.graph_files, *spec.raw_edge_files)) is not None
    have_cover = snap._first_existing(
        base, (*spec.cover_files.get(variant, ()), *spec.raw_cover_files.get(variant, ()))
    ) is not None
    missing = [name for name, ok in (("graph", have_graph), ("cover", have_cover)) if not ok]
    return {"found": False, "missing": missing or ["a matching graph/cover file pair"], "base": str(base)}


def discover(
    data_root: str | Path,
    snap_cache_dir: str | Path | None = None,
    *,
    datasets: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Report every catalogued (dataset, cover variant) as present, missing or unsupported.

    Only inspects the file system: nothing is loaded, hashed or downloaded.  The
    two cover variants are kept distinct, and a network lacking a variant
    (Wikipedia has only ``all``) is ``unsupported`` for it, not ``missing``.
    """
    root = Path(data_root).expanduser()
    cache_root = Path(snap_cache_dir).expanduser() / "raw" if snap_cache_dir else None
    wanted = list(datasets) if datasets is not None else list(DATASET_ORDER)
    rows: list[dict[str, Any]] = []
    for name in wanted:
        spec = snap.SPECS[name]
        for variant in COVER_VARIANTS:
            row: dict[str, Any] = {"dataset": name, "cover": variant, "directed_source": spec.directed}
            if variant not in spec.raw_cover_files:
                row.update(
                    status="unsupported",
                    reason=f"{name} has no supplied {snap.cover_variant_label(variant)!r} cover "
                           f"(available: {', '.join(snap.cover_variant_label(v) for v in spec.raw_cover_files)})",
                )
                rows.append(row)
                continue
            probe = _probe(spec, snap._dataset_dir(root, spec), variant, allow_pickles=True)
            root_kind = "networks_dir"
            if not probe["found"] and cache_root is not None:
                cache_probe = _probe(spec, cache_root / name, variant, allow_pickles=False)
                if cache_probe["found"]:
                    probe, root_kind = cache_probe, "hedonic_cache"
            if probe["found"]:
                row.update(status="present", root_kind=root_kind, **{k: v for k, v in probe.items() if k != "found"})
            else:
                row.update(
                    status="missing",
                    reason="missing " + " and ".join(probe["missing"]) + f" under {probe['base']}",
                    base=probe["base"],
                )
            rows.append(row)
    return rows


def select_cohort(
    rows: list[dict[str, Any]], datasets: list[str] | str, covers: list[str] | str,
    pairs: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Mark each row ``selected`` / ``eligible``, and whether an absence is a *gap* the user asked for.

    ``pairs`` (explicit ``dataset/cover`` list) overrides the dataset and cover lists.  A pair is a gap only
    if the user named it: a named dataset that is missing, a named pair that is missing or cannot exist.
    ``auto`` never turns absence into a gap.
    """
    explicit_datasets = datasets != "auto"
    explicit_covers = covers != "auto"
    for row in rows:
        if pairs is not None:
            row["selected"] = (row["dataset"], row["cover"]) in pairs
            named = row["selected"]
        else:
            in_datasets = datasets == "auto" or row["dataset"] in datasets
            in_covers = covers == "auto" or row["cover"] in covers
            row["selected"] = bool(in_datasets and in_covers)
            named = row["selected"] and explicit_datasets
        row["explicitly_requested"] = bool(
            row["selected"]
            and (
                (row["status"] == "missing" and named)
                or (row["status"] == "unsupported" and named and (pairs is not None or explicit_covers))
            )
        )
        row["eligible"] = bool(row["selected"] and row["status"] == "present")
    return rows


def render_cohort(rows: list[dict[str, Any]]) -> str:
    """Plain-text discovery report shared by the CLI, dry-run and the TUI."""
    lines = ["  graph/cover pairs in the local SNAP cache", ""]
    lines.append(f"  {'dataset':<12} {'cover':<8} {'status':<12} {'selected':<9} detail")
    for row in rows:
        detail = (
            f"{row.get('graph_file')} + {row.get('cover_file')} ({row.get('source_kind')}, "
            f"{row.get('bytes', 0) / 1e6:.0f} MB)"
            if row["status"] == "present"
            else row.get("reason", "")
        )
        chosen = "yes" if row.get("eligible") else ("gap" if row.get("explicitly_requested") else "–")
        cover_label = snap.cover_variant_label(row["cover"])
        lines.append(f"  {row['dataset']:<12} {cover_label:<8} {row['status']:<12} {chosen:<9} {detail}")
    counts = {s: sum(r["status"] == s for r in rows) for s in ("present", "missing", "unsupported")}
    lines.append("")
    lines.append(
        f"  {counts['present']} present · {counts['missing']} missing · {counts['unsupported']} unsupported; "
        f"{sum(bool(r.get('eligible')) for r in rows)} eligible for this study "
        "(a missing pair is downloaded only when provisioning is on: --provision, or explicitly named pairs)"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- options
def _section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("overlapping_gt_spectrum", {})
    return section if isinstance(section, dict) else {}


def named_config(value: str) -> Path:
    """A config path, or a name: ``full`` -> the shipped ``configs/overlapping-gt-spectrum-full.toml``.

    A path that exists as given wins; otherwise the shipped configs are searched (``resources.config_path``), first
    for ``overlapping-gt-spectrum-NAME`` and then for the name itself.
    """
    value = str(value)
    given = Path(value).expanduser()
    if given.exists():
        return given
    if "/" not in value and not value.endswith(".toml") and not value.startswith("overlapping-gt-spectrum"):
        shipped = resolve_config(f"overlapping-gt-spectrum-{value}")
        if shipped.exists():
            return shipped
    return resolve_config(value)


def resolve_options(args: argparse.Namespace) -> dict[str, Any]:
    """Priority: CLI flag > environment > TOML > defaults."""
    configured = args.config or os.getenv("HEDONIC_GT_SPECTRUM_CONFIG")
    config_path = named_config(configured or DEFAULT_CONFIG)
    if configured and not config_path.is_file():
        raise ValueError(f"no spectrum config {configured!r} (neither a file nor a shipped config)")
    config: dict[str, Any] = {}
    config_sha = None
    if config_path.is_file():
        raw = config_path.read_bytes()
        config = tomllib.loads(raw.decode("utf-8"))
        config_sha = hashlib.sha256(raw).hexdigest()
    section = _section(config)
    paths = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}

    def pick(cli_value, key: str, default):
        return cli_value if cli_value is not None else section.get(key, default)

    smoke = bool(args.profile == "smoke")

    def csv_or_auto(value, default) -> list[str] | str:
        raw = value if value is not None else default
        if isinstance(raw, list):
            raw = ",".join(map(str, raw))
        raw = str(raw).strip().lower()
        return "auto" if raw in ("auto", "all-available", "") else v3._parse_csv(raw, str)

    if smoke:
        datasets: list[str] | str = ["amazon", "dblp"] if args.datasets is None else csv_or_auto(args.datasets, "auto")
        covers: list[str] | str = ["top5000"] if args.cover is None else csv_or_auto(args.cover, "auto")
        seeds = v3._parse_range(args.seeds if args.seeds is not None else "0-1")
        det_grid = _parse_grid(args.resolutions if args.resolutions is not None else "0,0.5,1")
        audit_grid = _parse_grid(args.audit_resolutions if args.audit_resolutions is not None else "0:1:11")
        max_nodes = None
    else:
        datasets = csv_or_auto(args.datasets, section.get("datasets", "auto"))
        covers = csv_or_auto(args.cover, section.get("covers", "auto"))
        seeds = v3._parse_range(args.seeds if args.seeds is not None else str(section.get("seeds", "0-4")))
        det_grid = _parse_grid(
            args.resolutions if args.resolutions is not None else section.get("detector_resolutions", "0+geom:1e-4:1:9")
        )
        audit_grid = _parse_grid(
            args.audit_resolutions if args.audit_resolutions is not None else section.get("audit_resolutions", AUDIT_GRID)
        )
        max_nodes = int(pick(args.max_nodes, "max_nodes", 3000) or 0) or None
    if isinstance(covers, list):
        covers = [snap.canonical_cover_variant(cover) for cover in covers]
    for name in datasets if isinstance(datasets, list) else []:
        if name not in snap.SPECS:
            raise ValueError(f"unknown dataset {name!r}; choose from {', '.join(DATASET_ORDER)} or auto")
    for variant in covers if isinstance(covers, list) else []:
        if variant not in COVER_VARIANTS:
            labels = ", ".join(snap.cover_variant_label(v) for v in COVER_VARIANTS)
            raise ValueError(f"unknown cover {snap.cover_variant_label(variant)!r}; choose from {labels} or auto")
    raw_pairs = args.pairs if args.pairs is not None else section.get("pairs")
    pairs = None
    if raw_pairs:
        items = raw_pairs if isinstance(raw_pairs, list) else v3._parse_csv(raw_pairs, str)
        pairs = []
        for item in items:
            name, _, variant = str(item).partition("/")
            variant = snap.canonical_cover_variant(variant)
            if name not in snap.SPECS or variant not in COVER_VARIANTS:
                raise ValueError(f"invalid pair {item!r}; use dataset/cover, e.g. amazon/top5k")
            pairs.append((name, variant))
    policies = v3._parse_csv(
        args.isolation_policies, str, default=section.get("isolation_policies", list(ACTION_POLICIES))
    )
    if not policies or any(p not in ACTION_POLICIES for p in policies) or len(set(policies)) != len(policies):
        raise ValueError(f"isolation policies must be distinct values from {', '.join(ACTION_POLICIES)}")
    data_root = args.data_root or os.getenv("HEDONIC_NETWORKS_DIR") or section.get("data_root") \
        or paths.get("networks_dir") or DEFAULT_NETWORKS_DIR
    output_dir = args.output_dir or section.get("output_dir")
    if not output_dir and section.get("name"):  # a named run: <user-config output_dir>/<name>
        from hedonic.experiments.overlapping import userconfig

        output_dir = str(expand_path(userconfig.defaults().get("output_dir") or RUNS_ROOT) / str(section["name"]))
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    snap_cache = args.snap_cache_dir or section.get("snap_cache_dir") or "~/.cache/hedonic/snap"
    omega = section.get("omega", True) if args.omega is None else bool(args.omega)
    atol = float(pick(args.robustness_atol, "robustness_atol", 1e-10))
    rtol = float(pick(args.robustness_rtol, "robustness_rtol", 1e-9))
    timeout = float(pick(args.timeout_per_run, "timeout_per_run", 120.0 if smoke else 3600.0))
    if not timeout > 0 or not atol >= 0 or not rtol >= 0:
        raise ValueError("timeout must be positive and tolerances non-negative")
    policy = args.uncovered_policy or section.get("uncovered_policy", "covered-induced")
    if policy not in ("covered-induced", "singleton"):
        raise ValueError("uncovered policy must be covered-induced or singleton")
    detector_settings = parse_detector_settings(section.get("detector_settings"))
    provision = args.provision
    if provision is None:  # TOML key, else on for a run that names its pairs (every queue step does), off otherwise
        provision = bool(section.get("provision", pairs is not None and not smoke))
    return {
        "schema_version": SCHEMA_VERSION,
        "detector_settings": detector_settings,
        "config_path": str(config_path) if config_path.is_file() else None,
        "config_sha256": config_sha,
        "smoke": smoke,
        "datasets": datasets,
        "covers": covers,
        "pairs": pairs,
        "data_root": str(expand_path(data_root)),
        "snap_cache_dir": str(expand_path(snap_cache)),
        "output_dir": expand_path(output_dir),
        "output_dir_explicit": args.output_dir is not None,
        "policy": policy,
        "max_nodes": max_nodes,
        "detector_resolutions": det_grid,
        "audit_resolutions": audit_grid,
        "isolation_policies": policies,
        "seeds": seeds,
        "detector_seeds": seeds,
        "omega": bool(omega),
        "omega_sample_size": int(
            args.omega_sample_size if args.omega_sample_size is not None else section.get("omega_sample_size", 100_000)
        ),
        "timeout_seconds": timeout,
        "atol": atol,
        "rtol": rtol,
        "dense": bool(args.dense_oracle),
        "discover": bool(args.discover),
        "dry_run": bool(args.dry_run),
        "inspect": bool(args.inspect),
        "audit_only": bool(args.audit_only),
        "rescore_only": bool(args.rescore_only),
        "replot": bool(args.replot),
        "resume": not bool(args.force),
        "retry_failed": bool(args.retry_failed),
        "provision": bool(provision) and not smoke,
        "max_download_bytes": int(float(args.max_download_gb) * 1e9) if args.max_download_gb else None,
        "locked": False,
        "run_name": section.get("name"),
        "json": bool(args.json),
        "workers": max(1, int(args.workers if args.workers is not None else section.get("workers", 1 if smoke else 4))),
    }


DETECTOR_PHASES = v3.PHASES  # ("local",): the 1.x multilevel phase is not available in hedonic 2.0
CAP_PROJECTION_RULE = "keep_cap_labels_by_neighbours_in_label_then_larger_community_then_index_v1"


def parse_detector_settings(raw: Any) -> list[dict[str, Any]] | None:
    """Validate the optional TOML ``detector_settings`` list; ``None`` keeps the grid study."""
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("detector_settings must be a non-empty list of tables")
    out, seen = [], set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each detector_settings entry must be a table")
        unknown = set(item) - {"pair", "phase", "gamma", "cap", "label"}
        if unknown:
            raise ValueError(f"unknown detector_settings keys {sorted(unknown)}")
        pair = str(item.get("pair", ""))
        name, _, variant = pair.partition("/")
        if name not in snap.SPECS or variant not in COVER_VARIANTS:
            raise ValueError(f"invalid detector_settings pair {pair!r}; use dataset/cover, e.g. dblp/all")
        phase = v3.require_local_phase(str(item.get("phase", "local")))
        gamma, cap = float(item["gamma"]), item["cap"]
        if not math.isfinite(gamma) or isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError("detector_settings needs a finite gamma and an integer cap >= 1")
        label = str(item.get("label") or f"{phase}-g{gamma:.6g}-M{cap}")
        if (pair, label) in seen:
            raise ValueError(f"duplicate detector_settings label {label!r} for {pair}")
        seen.add((pair, label))
        out.append({"pair": pair, "label": label, "phase": phase, "gamma": gamma, "cap": int(cap)})
    return out


def job_settings(options: dict[str, Any], dataset: str, cover: str) -> list[dict[str, Any]]:
    """The detector settings of one pair: the resolution grid (local, GT cap), or the explicit list."""
    explicit = options.get("detector_settings")
    if explicit is None:
        return [{"label": None, "phase": "local", "gamma": float(g), "cap": None}
                for g in options["detector_resolutions"]]
    return [s for s in explicit if s["pair"] == f"{dataset}/{cover}"]


def conditions_per_job(options: dict[str, Any], dataset: str, cover: str) -> int:
    return len(options["isolation_policies"]) * len(job_settings(options, dataset, cover)) * len(options["detector_seeds"])


def project_memberships(graph, memberships: list[list[int]], cap: int) -> tuple[list[list[int]], dict[str, Any]]:
    """Deterministic cap projection of a (canonical) GT membership list.

    A vertex with more than ``cap`` labels keeps the ``cap`` labels in which it has the most neighbours (its
    strongest attachments in the reference), ties broken by the larger community, then by the smaller canonical
    label.  Vertices within the cap are untouched, every vertex keeps at least one label, and the result is
    re-canonicalised (communities emptied or made identical by the projection are dropped or merged).
    """
    cap = int(cap)
    sizes: dict[int, int] = {}
    for labels in memberships:
        for label in labels:
            sizes[label] = sizes.get(label, 0) + 1
    projected, truncated, removed = [], 0, 0
    for vertex, labels in enumerate(memberships):
        if len(labels) <= cap:
            projected.append(sorted(labels))
            continue
        own = set(labels)
        counts = dict.fromkeys(own, 0)
        for neighbour in graph.neighbors(vertex):
            for label in memberships[neighbour]:
                if label in own:
                    counts[label] += 1
        keep = sorted(own, key=lambda label: (-counts[label], -sizes[label], label))[:cap]
        projected.append(sorted(keep))
        truncated += 1
        removed += len(labels) - cap
    n = len(memberships)
    cover = rb.canonicalize_cover(rb.vertex_memberships_to_cover(projected), n)
    canonical = rb.cover_to_vertex_memberships(cover, n, require_covered=True)
    incidences = sum(map(len, memberships))
    return canonical, {
        "rule": CAP_PROJECTION_RULE,
        "cap": cap,
        "reference_max_memberships": max(map(len, memberships)) if memberships else 0,
        "truncated_vertices": truncated,
        "truncated_vertex_fraction": truncated / n if n else 0.0,
        "memberships_removed": removed,
        "reference_incidences": incidences,
        "projected_incidences": sum(map(len, canonical)),
        "removed_incidence_fraction": removed / incidences if incidences else 0.0,
        "reference_communities": len(sizes),
        "projected_communities": len(cover),
        "projected_cover_hash": rb.cover_hash(cover, n),
    }


def scientific_grid(options: dict[str, Any], jobs: list[tuple[str, str]]) -> dict[str, Any]:
    """Every scientific axis of the study, hashed into the lock and the manifest."""
    grid = _scientific_grid(options, jobs)
    if options.get("detector_settings") is not None:  # absent key: the grid (and its hash) is unchanged
        grid["detector_settings"] = [dict(s) for s in options["detector_settings"]]
        grid["detector_resolutions"] = []
        grid["start"] = "canonical_supplied_cover_cap_projected_when_cap_below_multiplicity"
        grid["cap_rule"] = "explicit per setting"
        grid["cap_projection_rule"] = CAP_PROJECTION_RULE
        grid["detector"] = {**grid["detector"], "phase": "local (every setting)"}
    return grid


def _scientific_grid(options: dict[str, Any], jobs: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "jobs": [list(job) for job in jobs],
        "completion_policy": options["policy"],
        "analysis_graph_pipeline": ANALYSIS_PIPELINE,
        "max_nodes": options["max_nodes"],
        "cap_rule": "max(2, supplied_cover_max_memberships_per_node)",
        "action_policies": list(options["isolation_policies"]),
        "audit_resolutions": [float(g) for g in options["audit_resolutions"]],
        "detector_resolutions": [float(g) for g in options["detector_resolutions"]],
        "detector_seeds": list(options["detector_seeds"]),
        "start": "complete_canonical_supplied_cover_exact",
        "detector": {
            "method": "Game.community_hedonic",
            "phase": "local",
            "stopping": "local moving to an equilibrium",
            "allow_isolation": {"fixed_labels": False, "open_labels": True},
        },
        "robustness_atol": float(options["atol"]),
        "robustness_rtol": float(options["rtol"]),
        "dense_oracle": bool(options["dense"]),
        "omega": bool(options["omega"]),
        "omega_sample_size": int(options["omega_sample_size"]),
        "timeout_seconds": float(options["timeout_seconds"]),
        "equilibrium_fraction_definition": "fraction of vertices with best-response regret <= atol + rtol*max(1,|u|)",
        "nash_definition": "every vertex within tolerance under the stated action policy and cap",
    }


def _json_hash(value: Any) -> str:
    return v3._json_hash(value)


def implementation_identity(options: dict[str, Any]) -> dict[str, Any]:
    """What must be identical for a stored record to be reused (excludes the grid, so seeds can be extended).

    ``tracked_files_sha256`` is :func:`tracked_digest` (source files, hedonic version, core SHA-256); ``runtime`` is
    the path-free runtime identity; ``runtime_versions`` (with paths) is provenance only and never compared.
    """
    tracked = tracked_file_hashes()
    runtime_identity = _runtime_identity()
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "tracked_files": tracked,
        "tracked_files_sha256": _implementation_digest(tracked, runtime_identity),
        "runtime": runtime_identity,
        "runtime_versions": runtime.runtime_versions(),
        "config_sha256": options.get("config_sha256"),
        "locked": False,
        "lock": LOCK_NOTE,
    }


def _compact_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """The part stored in every record and compared on resume."""
    return {
        "schema_version": identity["schema_version"],
        "protocol_name": identity["protocol_name"],
        "tracked_files_sha256": identity["tracked_files_sha256"],
        "runtime": identity["runtime"],
        "analysis_graph_pipeline": ANALYSIS_PIPELINE,
    }


# --------------------------------------------------------------------------- spectrum audit
def spectrum_audit(
    graph,
    memberships: list[list[int]],
    *,
    cap: int,
    allow_isolation: bool,
    gammas: Iterable[float],
    atol: float,
    rtol: float,
    dense: bool = False,
    cover: stab.PreparedCover | None = None,
    density: float | None = None,
    workers: int = 1,
    progress=None,
) -> list[dict[str, Any]]:
    """Per-resolution vertex stability of a cover under one action policy.

    ``cover`` (``stability.prepare``) and the graph ``density`` may be passed in when they are already known: a pair's
    cover is prepared once and serves every resolution and both policies. ``workers`` forked processes split the
    vertices; ``progress(done, gamma)`` follows the resolutions.

    For each gamma the exact unit-l2 best response (prefix optimisation) of every vertex is evaluated by the C audit; a
    vertex is *stable* when its regret is at most ``atol + rtol * max(1, |utility|)`` (the audit tolerance). The
    candidate set is exact for every gamma in [0, 1], so ``dense`` (the 1.x all-labels oracle) changes no number; it
    stays a recorded option.
    """
    if cover is None:
        cover = stab.prepare(graph, memberships)
    density = graph.density() if density is None else density
    rows = stab.spectrum_rows(cover, cap, [float(g) for g in gammas], atol=atol, rtol=rtol,
                            allow_isolation=allow_isolation, workers=workers, progress=progress)
    return [{"gamma": row["gamma"], "gamma_over_density": row["gamma"] / density if density > 0 else None,
             **{key: value for key, value in row.items() if key != "gamma"}} for row in rows]


# --------------------------------------------------------------------------- jobs
class Start:
    """A detector start: memberships, canonical cover and hash, and the cap projection (``None`` = exact GT)."""

    def __init__(self, memberships, cover, digest: str, projection: dict[str, Any] | None):
        self.memberships, self.cover, self.hash, self.projection = memberships, cover, digest, projection

    @property
    def is_ground_truth(self) -> bool:
        return self.projection is None


class Job:
    """One prepared (dataset, cover) pair: graph, canonical GT, cap and provenance."""

    def __init__(self, dataset: str, cover: str, prepared, cohort_row: dict[str, Any] | None):
        self.dataset = dataset
        self.cover = cover
        self.prepared = prepared
        self.cohort_row = cohort_row or {}
        graph = prepared.dataset.graph
        self.graph = graph
        self.gt_cover = rb.canonicalize_cover(prepared.dataset.cover, graph.vcount())
        self.memberships = rb.cover_to_vertex_memberships(self.gt_cover, graph.vcount(), require_covered=True)
        self.cap = max(2, max(len(labels) for labels in self.memberships))
        self.gt_hash = rb.cover_hash(self.gt_cover, graph.vcount())
        self.density = graph.density()
        self._starts: dict[int, Start] = {}

    def start(self, cap: int | None) -> "Start":
        """The detector start under ``cap``: the exact GT (``None`` or cap >= multiplicity) or its projection."""
        if cap is None or int(cap) >= max(map(len, self.memberships)):
            return Start(self.memberships, self.gt_cover, self.gt_hash, None)
        if int(cap) not in self._starts:
            memberships, projection = project_memberships(self.graph, self.memberships, int(cap))
            cover = rb.vertex_memberships_to_cover(memberships)
            cover = rb.canonicalize_cover(cover, self.graph.vcount())
            self._starts[int(cap)] = Start(memberships, cover, projection["projected_cover_hash"], projection)
        return self._starts[int(cap)]

    def __getstate__(self):
        return {**self.__dict__, "_starts": {}}

    @property
    def key(self) -> str:
        return f"{self.dataset}-{self.cover}"

    def provenance(self) -> dict[str, Any]:
        report = self.prepared.dataset.report
        completion = dict(report.get("ground_truth_completion", {}))
        original_ids = completion.pop("original_vertex_ids", None)
        return {
            "dataset": self.dataset,
            "cover": self.cover,
            "completion_policy": self.prepared.policy,
            "analysis_graph_pipeline": ANALYSIS_PIPELINE,
            "graph_sha256": self.prepared.graph_identity,
            "ground_truth_cover_sha256": self.prepared.ground_truth_identity,
            "canonical_cover_hash": self.gt_hash,
            "n_vertices": self.graph.vcount(),
            "n_edges": self.graph.ecount(),
            "density": self.density,
            "ground_truth_community_count": len(self.gt_cover),
            "ground_truth_max_memberships": max(map(len, self.memberships)),
            "cap": self.cap,
            "covered_vertices": self.graph.vcount(),
            "source_covered_vertices": len(self.prepared.source_covered_vertices),
            "synthetic_memberships_added": self.prepared.completion_count,
            "source_artifacts": report.get("source_artifacts"),
            "source_artifact_fingerprint_sha256": report.get("source_artifact_fingerprint_sha256"),
            "source_graph": report.get("source_graph"),
            "analysis_graph": report.get("analysis_graph"),
            "completion": completion,
            "retained_original_vertex_ids_sha256": _json_hash(original_ids) if original_ids is not None else None,
            "cohort": {k: v for k, v in self.cohort_row.items() if k in ("root_kind", "source_kind", "graph_file", "cover_file")},
        }


def load_job(options: dict[str, Any], row: dict[str, Any]) -> Job:
    """Load and prepare one eligible pair; downloads happen only under ``--provision``."""
    name, variant = row["dataset"], row["cover"]
    if options["smoke"]:
        raw = snap.smoke_dataset(name, cover_variant=variant)
    elif row.get("root_kind") == "hedonic_cache":
        raw = snap._load_from_raw(snap.SPECS[name], Path(row["base"]), variant)
    else:
        raw = snap.load_snap_dataset(
            name,
            cover_variant=variant,
            data_root=options["data_root"],
            allow_catalog=bool(options["provision"]),
            max_download_bytes=options["max_download_bytes"],
        )
    prepared = gtd.prepare_dataset(raw, policy=options["policy"], max_nodes=options["max_nodes"])
    return Job(name, variant, prepared, row)


def _smoke_rows(options: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name in ("amazon", "dblp"):
        rows.append(
            {"dataset": name, "cover": "top5000", "status": "present", "root_kind": "built_in_smoke",
             "source_kind": "deterministic_smoke_fixture", "graph_file": "(built-in)", "cover_file": "(built-in)",
             "bytes": 0}
        )
    return rows


# --------------------------------------------------------------------------- conditions
def condition_payload(job: Job, options: dict[str, Any], identity: dict[str, Any], policy: str, gamma: float, seed: int,
                      setting: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = _grid_payload(job, options, identity, policy, gamma, seed)
    if setting is not None and setting.get("label") is not None:  # explicit settings: phase, cap and start in the key
        start = job.start(setting["cap"])
        payload.update({
            "setting": setting["label"], "phase": setting["phase"], "max_memberships": int(setting["cap"]),
            "start": "exact_canonical_ground_truth" if start.is_ground_truth else "cap_projected_canonical_ground_truth",
            "initial_cover_identity": start.hash,
            "cap_projection_rule": None if start.is_ground_truth else CAP_PROJECTION_RULE,
        })
    return payload


def _grid_payload(job: Job, options: dict[str, Any], identity: dict[str, Any], policy: str, gamma: float, seed: int) -> dict[str, Any]:
    return {
        "study": PROTOCOL_NAME,
        "schema_version": SCHEMA_VERSION,
        "dataset": job.dataset,
        "cover": job.cover,
        "completion_policy": job.prepared.policy,
        "analysis_graph_pipeline": ANALYSIS_PIPELINE,
        "max_nodes": options["max_nodes"],
        "graph_identity": job.prepared.graph_identity,
        "ground_truth_identity": job.prepared.ground_truth_identity,
        "initial_cover_identity": job.gt_hash,
        "start": "exact_canonical_ground_truth",
        "phase": "local",
        "action_policy": policy,
        "allow_isolation": policy == "open_labels",
        "max_memberships": job.cap,
        "gamma": float(gamma),
        "seed": int(seed),
        "robustness_atol": options["atol"],
        "robustness_rtol": options["rtol"],
        "dense_oracle": options["dense"],
        "metric_sampling": {"omega": options["omega"], "omega_sample_size": options["omega_sample_size"]},
        "timeout_seconds": options["timeout_seconds"],
        "implementation": _compact_identity(identity),
    }


def _numeric(mapping: dict[str, Any]) -> dict[str, float]:
    return {
        k: float(v) for k, v in (mapping or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
    }


def _augment_record(record: dict[str, Any], job: Job, policy: str, gamma: float,
                    setting: dict[str, Any] | None = None) -> None:
    """Add the spectrum-specific fields (paired deltas, unchanged flag) to a scored record."""
    initial, final = _numeric(record.get("initial_metrics", {})), _numeric(record.get("final_metrics", {}))
    if setting is not None and setting.get("label") is not None:
        start = job.start(setting["cap"])
        record.update({
            "setting": setting["label"], "phase": setting["phase"], "max_memberships": int(setting["cap"]),
            "start_cover_hash": start.hash, "cap_projection": start.projection,
            "returned_start_unchanged": (
                record.get("final_cover_hash") is not None and record.get("final_cover_hash") == start.hash),
        })
    record.update(
        {
            "study": PROTOCOL_NAME,
            "row_type": "detector_run",
            "start": (record.get("condition") or {}).get("start", "exact_canonical_ground_truth"),
            "action_policy": policy,
            "gamma": float(gamma),
            "gamma_over_density": float(gamma) / job.density if job.density > 0 else None,
            "paired_metric_deltas": {k: final[k] - initial[k] for k in sorted(final) if k in initial},
            "returned_ground_truth_unchanged": (
                record.get("final_cover_hash") is not None and record.get("final_cover_hash") == job.gt_hash
            ),
        }
    )


def _base_metrics(job: "Job", options: dict[str, Any], identity: dict[str, Any], cover: list[list[int]],
                  cover_digest: str, reference: "Start | None" = None, onmi: bool = False) -> dict[str, Any]:
    """The seed-independent accuracy vector of ``cover`` against the supplied GT, cached on disk.

    ``evaluate_cover(..., compute_omega=False)`` is the expensive part (quadratic-ish in the community count);
    it is deterministic in (graph, GT, cover, implementation), so identical returned covers — across seeds,
    action spaces, resumes and rescoring — share one evaluation.
    """
    ref_cover, ref_hash = (job.gt_cover, job.gt_hash) if reference is None else (reference.cover, reference.hash)
    key_payload = {"graph": job.prepared.graph_identity, "gt": ref_hash, "cover": cover_digest,
                   "implementation": identity["tracked_files_sha256"], "singleton_mode": "all"}
    if onmi:  # explicit-settings runs only: the grid study's cache keys are unchanged
        key_payload["onmi"] = "lfk"
    key = _json_hash(key_payload)[:32]
    path = options["output_dir"] / "metric_cache" / f"{key}.json"
    cached = v3._read_json(path)
    if cached is not None:
        return cached
    base = evaluate_cover(cover, ref_cover, job.graph.vcount(), compute_omega=False)
    if onmi:
        base["onmi"] = float(overlapping_normalized_mutual_information_lfk(cover, ref_cover, n_vertices=job.graph.vcount()))
    _atomic_json(path, base)
    return base


def _with_omega(base: dict[str, Any], job: "Job", cover: list[list[int]], options: dict[str, Any], seed: int) -> dict[str, Any]:
    """Attach the sampled Omega exactly as ``evaluate_cover(compute_omega=True)`` does."""
    if not options["omega"]:
        return dict(base)
    omega = omega_index(_cover_sets(cover, "all"), _cover_sets(job.gt_cover, "all"), job.graph.vcount(),
                        sample_size=options["omega_sample_size"], seed=seed)
    return {**base, "omega": omega, "omega_method": "sampled_pairwise",
            "omega_sample_size": int(options["omega_sample_size"]), "omega_seed": int(seed)}


def _score_record(
    record: dict[str, Any], path: Path, job: "Job", options: dict[str, Any], identity: dict[str, Any], policy: str,
    seed: int, gamma: float, final_memberships: list[list[int]], raw_memberships: list[list[int]],
    audit_cache: dict[tuple[Any, ...], dict[str, Any]], rescored: bool = False, setting: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate, audit (under the run's own action policy) and score one returned cover.

    Mirrors the v3 record fields that this study uses.  Two identities let it skip work without changing any
    number: the start *is* the ground truth, so the start-versus-final transition equals the final-versus-GT
    vector (without Omega); and only the action space being run is audited (the v3 scorer audited both).
    """
    graph, n = job.graph, job.graph.vcount()
    explicit = setting is not None and setting.get("label") is not None
    cap = int(setting["cap"]) if explicit else job.cap
    start = job.start(cap if explicit else None)
    try:
        raw = v3._normalize_raw_memberships(final_memberships, n, cap)
        raw_cover = rb.vertex_memberships_to_cover(raw)
        final_cover = rb.canonicalize_cover(raw_cover, n)
        canonical_memberships = rb.cover_to_vertex_memberships(final_cover, n, require_covered=True)
        raw_digest = v3._raw_cover_hash(raw_cover, n)
        cover_digest = rb.cover_hash(final_cover, n)
        changed = raw_digest != cover_digest
        duplicates = len(raw_cover) - len({tuple(sorted(c)) for c in raw_cover})
    except Exception as exc:  # noqa: BLE001
        record.update(status="invalid_cover" if not rescored else "rescore_invalid_cover",
                      error=f"{type(exc).__name__}: {exc}", runtime_applicable=False)
        _augment_record(record, job, policy, gamma, setting)
        _atomic_json(path, record)
        return record
    output_dir = options["output_dir"]
    _persist_cover(output_dir, final_cover)
    raw_membership_digest = _persist_memberships(output_dir, raw, n)
    pre_digest = _persist_memberships(output_dir, v3._normalize_raw_memberships(raw_memberships, n, cap), n)
    started = time.monotonic()
    base = _base_metrics(job, options, identity, final_cover, cover_digest, onmi=explicit)
    final_metrics = _with_omega(base, job, final_cover, options, seed)
    if explicit:
        _persist_cover(output_dir, start.cover)
    initial_base = _base_metrics(job, options, identity, start.cover, start.hash, onmi=explicit)
    initial_metrics = _with_omega(initial_base, job, start.cover, options, seed)
    if start.is_ground_truth:
        transition_metrics = {**base, "omega": None, "omega_method": None, "omega_sample_size": None, "omega_seed": None}
    else:  # final versus the projected start it began from (Omega omitted)
        transition_metrics = _base_metrics(job, options, identity, final_cover, cover_digest, reference=start, onmi=True)
    metrics_runtime = time.monotonic() - started
    allow_isolation = policy == "open_labels"
    audit_started = time.monotonic()

    def audit(memberships: list[list[int]], digest: str) -> dict[str, Any]:
        key = (job.prepared.graph_identity, digest, int(cap), float(gamma), allow_isolation,
               float(options["atol"]), float(options["rtol"]), bool(options["dense"]))
        if key not in audit_cache:
            audit_cache[key] = stab.audit_cover(
                graph, memberships, max_memberships=cap, allow_isolation=allow_isolation, gamma=float(gamma),
                interval=(0.0, 1.0), atol=options["atol"], rtol=options["rtol"])
        return audit_cache[key]

    selected = audit(raw, raw_membership_digest)
    canonical_selected = audit(canonical_memberships, cover_digest) if changed else selected
    audit_runtime = time.monotonic() - audit_started
    phi_initial = rb.fractional_phi(graph, start.memberships, float(gamma))
    phi_final = rb.fractional_phi(graph, raw, float(gamma))
    equilibrium = bool(selected["is_local_equilibrium_at_resolution"])
    record.update({
        "protocol_identity": _compact_identity(identity),
        "status": "completed" if equilibrium else "completed_non_equilibrium",
        "equilibrium_status": "verified" if equilibrium else "stationary_not_equilibrium",
        "runtime_applicable": equilibrium,
        "final_cover_hash": cover_digest, "raw_cover_hash": raw_digest,
        "raw_membership_hash": raw_membership_digest, "final_membership_hash": raw_membership_digest,
        "pre_cleanup_membership_hash": pre_digest,
        "normalization_source": "raw_memberships", "normalization_changed": changed,
        "duplicate_community_count": duplicates,
        "final_metrics": final_metrics, "initial_metrics": initial_metrics, "transition_metrics": transition_metrics,
        "transition_metrics_note": (
            "the start is the ground truth, so start-vs-final equals final-vs-GT (Omega omitted)" if start.is_ground_truth
            else "final versus the cap-projected start (Omega omitted); initial_metrics = projected start versus GT"),
        "distance": v3._initialization_distance(initial_metrics, final_metrics, transition_metrics),
        "cover_change": v3._cover_change_diagnostics(start.memberships, raw),
        "robustness": {"audited_policy": policy, "selected_policy": selected, "canonical_selected_policy": canonical_selected},
        "fractional_phi_initial": phi_initial, "fractional_phi_final": phi_final,
        "fractional_phi_delta": phi_final - phi_initial,
        "metrics_runtime_seconds": metrics_runtime, "robustness_audit_runtime_seconds": audit_runtime,
        "scoring_runtime_seconds": time.monotonic() - started,
    })
    if rescored:
        record["rescored_at"] = time.time()
    _augment_record(record, job, policy, gamma, setting)
    _atomic_json(path, record)
    return record


def run_condition(
    job: Job, options: dict[str, Any], identity: dict[str, Any], policy: str, gamma: float, seed: int,
    audit_cache: dict[tuple[Any, ...], dict[str, Any]], setting: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run (or reuse) one detector condition (exact-GT local moving, or an explicit setting) and score it."""
    explicit = setting is not None and setting.get("label") is not None
    cap = int(setting["cap"]) if explicit else job.cap
    phase = setting["phase"] if explicit else "local"
    start = job.start(cap if explicit else None)
    payload = condition_payload(job, options, identity, policy, gamma, seed, setting)
    key = _json_hash(payload)[:24]
    output_dir: Path = options["output_dir"]
    path = v3._condition_path(output_dir, job.key, key)
    existing = v3._read_json(path)
    expected_identity = _json_hash(payload)
    if (
        options["resume"]
        and existing is not None
        and existing.get("condition_identity") == expected_identity
        and existing.get("protocol_identity") == _compact_identity(identity)
    ):
        status = existing.get("status")
        if status in FINAL_STATUSES and v3._resume_artifacts_match(
            output_dir, existing, n_vertices=job.graph.vcount(), cap=cap
        ):
            return existing
        if status in RESOURCE_STATUSES and not options["retry_failed"]:
            return existing  # a preserved resource outcome is retried only under --retry-failed
    outcome = run_in_subprocess(
        v3._detector_worker,
        job.graph,
        start.memberships,
        cap,
        float(gamma),
        phase,
        policy == "open_labels",
        int(seed),
        timeout_seconds=options["timeout_seconds"],
        packet_dir=output_dir / "worker_packets",
    )
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "condition_key": key,
        "condition_identity": expected_identity,
        "condition": payload,
        "protocol_identity": _compact_identity(identity),
        "dataset": job.dataset,
        "cover": job.cover,
        "policy": job.prepared.policy,
        "source_covered_vertices": len(job.prepared.source_covered_vertices),
        "synthetic_completion_count": job.prepared.completion_count,
        "initial_cover_hash": start.hash,
        "ground_truth_cover_hash": job.gt_hash,
        "detector_runtime_seconds": outcome.runtime_seconds,
        "detector_peak_rss_bytes": outcome.peak_rss_bytes,
        "status": outcome.status,
        "error": outcome.error,
        "runtime_applicable": outcome.status == "ok",
    }
    if outcome.status != "ok" or not isinstance(outcome.payload, dict):
        record["status"] = "failed" if outcome.status not in ("timeout",) else "timeout"
        record["failure_status"] = outcome.status
        _augment_record(record, job, policy, gamma, setting)
        _atomic_json(path, record)
        return record
    raw, final = outcome.payload.get("raw_memberships"), outcome.payload.get("final_memberships")
    if not isinstance(raw, list) or not isinstance(final, list):
        record.update(status="invalid_worker_payload", error="worker returned no memberships", runtime_applicable=False)
        _augment_record(record, job, policy, gamma, setting)
        _atomic_json(path, record)
        return record
    return _score_record(record, path, job, options, identity, policy, seed, float(gamma), final, raw, audit_cache,
                         setting=setting)


# --------------------------------------------------------------------------- flat rows / summaries
def flatten(record: dict[str, Any]) -> dict[str, Any]:
    """One flat, unambiguously named row per detector condition."""
    condition = record.get("condition", {})
    final = _numeric(record.get("final_metrics", {}))
    initial = _numeric(record.get("initial_metrics", {}))
    audit = (record.get("robustness") or {}).get("selected_policy") or {}
    row: dict[str, Any] = {
        "row_type": record.get("row_type", "detector_run"),
        "dataset": record.get("dataset"),
        "cover": record.get("cover"),
        "action_policy": record.get("action_policy") or condition.get("action_policy"),
        "gamma": record.get("gamma", condition.get("gamma")),
        "gamma_over_density": record.get("gamma_over_density"),
        "seed": condition.get("seed"),
        "status": record.get("status"),
        "equilibrium_status": record.get("equilibrium_status"),
        "max_memberships": condition.get("max_memberships"),
        "condition_key": record.get("condition_key"),
        "graph_identity": condition.get("graph_identity"),
        "ground_truth_identity": condition.get("ground_truth_identity"),
        "initial_cover_hash": record.get("initial_cover_hash"),
        "final_cover_hash": record.get("final_cover_hash"),
        "final_membership_hash": record.get("final_membership_hash"),
        "returned_ground_truth_unchanged": record.get("returned_ground_truth_unchanged"),
        "detector_runtime_seconds": record.get("detector_runtime_seconds"),
        "runtime_applicable": record.get("runtime_applicable"),
        "detector_peak_rss_bytes": record.get("detector_peak_rss_bytes"),
        "audit_final_stable_fraction": audit.get("stable_fraction_at_resolution"),
        "audit_final_profitable_fraction": (
            audit["profitable_vertex_count_at_resolution"] / audit["n_vertices_scored"]
            if audit.get("n_vertices_scored") else None
        ),
        "audit_final_mean_positive_regret": audit.get("mean_positive_regret_at_resolution"),
        "audit_final_max_positive_regret": audit.get("max_positive_regret_at_resolution"),
        "audit_final_is_nash_equilibrium": audit.get("is_local_equilibrium_at_resolution"),
        "changed_vertex_fraction": (record.get("cover_change") or {}).get("changed_vertex_fraction"),
        "memberships_added": (record.get("cover_change") or {}).get("memberships_added"),
        "memberships_removed": (record.get("cover_change") or {}).get("memberships_removed"),
        "cover_distance_initial_to_final": (record.get("distance") or {}).get("initial_to_final_distance"),
        "fractional_phi_initial": record.get("fractional_phi_initial"),
        "fractional_phi_final": record.get("fractional_phi_final"),
        "fractional_phi_delta": record.get("fractional_phi_delta"),
        "error": (record.get("error") or "")[:300] or None,
    }
    if record.get("setting") is not None:  # explicit settings only (grid rows are unchanged)
        projection = record.get("cap_projection") or {}
        row.update({
            "setting": record["setting"], "phase": record.get("phase"), "start": record.get("start"),
            "start_cover_hash": record.get("start_cover_hash"),
            "returned_start_unchanged": record.get("returned_start_unchanged"),
            "start_truncated_vertex_fraction": projection.get("truncated_vertex_fraction", 0.0),
            "start_removed_incidence_fraction": projection.get("removed_incidence_fraction", 0.0),
        })
        if (record.get("start") or "") != "exact_canonical_ground_truth":
            for key, value in sorted(_numeric(record.get("transition_metrics", {})).items()):
                row[f"vs_start_{key}"] = value
    for key, value in sorted(final.items()):
        row[f"final_{key}"] = value
    for key, value in sorted(initial.items()):
        row[f"gt_{key}"] = value
    for key, value in sorted((record.get("paired_metric_deltas") or {}).items()):
        row[f"delta_{key}"] = value
    return row


def summarise(rows: list[dict[str, Any]], expected: dict[tuple[str, str, str, float], int]) -> list[dict[str, Any]]:
    """Per (dataset, cover, policy, gamma): denominators first, then seed means over returned covers."""
    groups: dict[tuple[str, str, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        if row["row_type"] != "detector_run":
            continue
        group = (row["dataset"], row["cover"], row["action_policy"], float(row["gamma"]))
        if row.get("setting") is not None:
            group += (row["setting"],)
        groups.setdefault(group, []).append(row)
    out = []
    for key in sorted(expected, key=lambda k: (DATASET_ORDER.index(k[0]), k[1], k[2], k[3], *k[4:])):
        members = groups.get(key, [])
        returned = [r for r in members if r["status"] in FINAL_STATUSES]
        verified = [r for r in returned if r["status"] == "completed"]
        summary: dict[str, Any] = {
            "dataset": key[0], "cover": key[1], "action_policy": key[2], "gamma": key[3],
            **({"setting": key[4]} if len(key) > 4 else {}),
            "expected_runs": expected[key],
            "returned_covers": len(returned),
            "verified_equilibria": len(verified),
            "non_equilibrium_returns": len(returned) - len(verified),
            "timeouts": sum(r["status"] == "timeout" for r in members),
            "failures": sum(r["status"] in RESOURCE_STATUSES - {"timeout"} for r in members),
            "missing": expected[key] - len(members),
            "unchanged_returns": sum(bool(r.get("returned_ground_truth_unchanged")) for r in returned),
        }
        prefixes = ("delta_", "final_", "vs_start_") if len(key) > 4 else ("delta_", "final_")
        extra = ("detector_peak_rss_bytes",) if len(key) > 4 else ()
        fields = sorted({k for r in returned for k in r if k.startswith(prefixes) or k in extra or k in (
            "detector_runtime_seconds", "changed_vertex_fraction", "cover_distance_initial_to_final",
            "audit_final_stable_fraction", "audit_final_max_positive_regret", "fractional_phi_delta")})
        for field in fields:
            values = [float(r[field]) for r in returned if isinstance(r.get(field), (int, float))]
            if values:
                summary[f"{field}_mean"] = statistics.fmean(values)
                summary[f"{field}_min"] = min(values)
                summary[f"{field}_max"] = max(values)
        for metric in HEADLINE_METRICS:
            values = [float(r[f"final_{metric}"]) for r in verified if isinstance(r.get(f"final_{metric}"), (int, float))]
            if values:
                summary[f"final_{metric}_verified_mean"] = statistics.fmean(values)
        out.append(summary)
    return out


def coverage(options: dict[str, Any], cohort: list[dict[str, Any]], jobs: list[Job], rows: list[dict[str, Any]]) -> dict[str, Any]:
    detector_rows = [r for r in rows if r["row_type"] == "detector_run"]
    per_status: dict[str, int] = {}
    for r in detector_rows:
        per_status[r["status"]] = per_status.get(r["status"], 0) + 1
    expected = sum(conditions_per_job(options, job.dataset, job.cover) for job in jobs)
    gaps = [r for r in cohort if r.get("explicitly_requested") and not r.get("eligible")]
    all_accounted = len(detector_rows) == expected
    return {
        "study": PROTOCOL_NAME,
        "expected_detector_conditions": expected,
        "recorded_detector_conditions": len(detector_rows),
        "status_counts": dict(sorted(per_status.items())),
        "verified_equilibria": per_status.get("completed", 0),
        "non_equilibrium_returns": per_status.get("completed_non_equilibrium", 0),
        "timeouts": per_status.get("timeout", 0),
        "failures_and_unsupported": sum(v for k, v in per_status.items() if k in RESOURCE_STATUSES - {"timeout"}),
        "missing_conditions": expected - len(detector_rows),
        "all_conditions_accounted_for": all_accounted,
        "requested_but_unavailable_pairs": [f"{r['dataset']}/{r['cover']} ({r['status']})" for r in gaps],
        "jobs": [job.key for job in jobs],
        "complete": bool(all_accounted and not gaps and jobs),
        "audit_only": bool(options["audit_only"]),
        "note": "returned covers are all scored; only status 'completed' has a verified zero-regret audit under the "
                "action policy. Timeouts and failures carry no imputed metrics.",
    }


# --------------------------------------------------------------------------- estimate / inspect
def _priors(options: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per dataset: median local-moving runtime and max peak RSS of this run's own stored records, if any."""
    runtimes: dict[str, list[float]] = {}
    rss: dict[str, int] = {}
    for path in (Path(options["output_dir"]) / "runs").glob("*/*.json"):
        record = v3._read_json(path)
        if not record or record.get("status") not in FINAL_STATUSES:
            continue
        name = str(record.get("dataset"))
        if record.get("detector_runtime_seconds"):
            runtimes.setdefault(name, []).append(float(record["detector_runtime_seconds"]))
        if record.get("detector_peak_rss_bytes"):
            rss[name] = max(rss.get(name, 0), int(record["detector_peak_rss_bytes"]))
    return {name: {"median_seconds": statistics.median(times), "max_peak_rss_bytes": rss.get(name), "runs": len(times)}
            for name, times in runtimes.items()}


def estimate_runtime(options: dict[str, Any], jobs_count: int, datasets: list[str] | None = None) -> dict[str, Any]:
    """Seconds per local-moving run from the records already in the output folder, else an explicit unknown."""
    priors = _priors(options)
    conditions = jobs_count * len(options["isolation_policies"]) * len(options["detector_resolutions"]) * len(options["detector_seeds"])
    if options.get("detector_settings") is not None:
        conditions = len(options["detector_settings"]) * len(options["isolation_policies"]) * len(options["detector_seeds"])
    chosen = [priors[d] for d in (datasets or priors) if d in priors]
    if not chosen:
        return {"conditions": conditions, "per_run_seconds": None, "total_seconds": None, "max_peak_rss_bytes": None,
                "basis": "no earlier local-moving records found; runtime and peak RSS are measured live and stored per run"}
    per_run = statistics.median([c["median_seconds"] for c in chosen])
    peak = max((c["max_peak_rss_bytes"] or 0) for c in chosen) or None
    return {"conditions": conditions, "per_run_seconds": per_run, "total_seconds": conditions * per_run,
            "max_peak_rss_bytes": peak,
            "basis": f"median over {sum(c['runs'] for c in chosen)} local-moving runs already in the output folder; "
                     "excludes graph loading, scoring and the audit"}


def inspect_pairs(options: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load each eligible pair (no download) and report prepared sizes, cap and audit workload."""
    out = []
    for row in [r for r in rows if r.get("eligible")]:
        started = time.monotonic()
        if row.get("root_kind") == "provisioned_download":  # a dry run never downloads
            out.append({"dataset": row["dataset"], "cover": row["cover"], "status": "not_downloaded",
                        "error": "missing locally; downloaded when the study runs"})
            continue
        try:
            job = load_job(options, row)
        except Exception as exc:  # noqa: BLE001
            out.append({"dataset": row["dataset"], "cover": row["cover"], "status": "load_failed",
                        "error": f"{type(exc).__name__}: {exc}"})
            continue
        provenance = job.provenance()
        vertices = job.graph.vcount()
        try:  # what the memory free now allows this pair, before anything is built
            plan = job_capacity(job, options, say=None)
            capacity_note = {"audit_workers": plan["audit_workers"], "detector_workers": plan["detector_workers"],
                             "lines": plan["lines"]}
        except capacity.CapacityError as exc:
            capacity_note = {"refused": str(exc)}
        out.append({
            "capacity": capacity_note,
            "dataset": job.dataset, "cover": job.cover, "status": "ready",
            "n_vertices": vertices, "n_edges": job.graph.ecount(),
            "communities": len(job.gt_cover), "cap": job.cap,
            "covered_vertices": provenance["covered_vertices"],
            "source_covered_vertices": provenance["source_covered_vertices"],
            "audit_best_responses": vertices * len(options["audit_resolutions"]) * len(options["isolation_policies"]),
            "detector_runs": conditions_per_job(options, job.dataset, job.cover),
            "settings": [
                {"label": s["label"], "phase": s["phase"], "gamma": s["gamma"], "cap": s["cap"],
                 "truncated_vertex_fraction": (job.start(s["cap"]).projection or {}).get("truncated_vertex_fraction", 0.0)}
                for s in job_settings(options, job.dataset, job.cover) if s["label"] is not None],
            "load_seconds": time.monotonic() - started,
        })
    return out


def describe(options: dict[str, Any], pairs: int) -> str:
    """One line for viewers: what this run covers."""
    size = ("built-in smoke fixture" if options["smoke"] else
            f"{options['max_nodes']:,}-vertex bounded" if options["max_nodes"] else "full-graph")
    if options.get("detector_settings") is not None:
        return (f"{pairs} graph/cover pairs · {size} covered-induced graph · {', '.join(options['isolation_policies'])} · "
                f"{len(options['detector_settings'])} explicit detector settings × {len(options['detector_seeds'])} seeds "
                "(GT start, cap-projected where needed)")
    return (f"{pairs} graph/cover pairs · {size} covered-induced graph · {', '.join(options['isolation_policies'])} · "
            f"{len(options['detector_resolutions'])} resolutions × {len(options['detector_seeds'])} seeds (local moving from the exact GT)")


def render_plan(options: dict[str, Any], rows: list[dict[str, Any]], inspected: list[dict[str, Any]] | None = None) -> str:
    jobs = [r for r in rows if r.get("eligible")]
    estimate = estimate_runtime(options, len(jobs), sorted({r["dataset"] for r in jobs}))
    size = f"{options['max_nodes']:,}-vertex bounded" if options["max_nodes"] else "full-graph"
    lines = [render_cohort(rows), ""]
    lines.append(
        f"  plan: {len(jobs)} graph/cover pairs · {size} covered-induced analysis graph ({options['policy']}) · "
        f"policies {', '.join(options['isolation_policies'])}"
    )
    if options.get("detector_settings") is None:
        lines.append(
            f"        audit grid {len(options['audit_resolutions'])} resolutions in [{options['audit_resolutions'][0]:g}, "
            f"{options['audit_resolutions'][-1]:g}] · detector grid {len(options['detector_resolutions'])} resolutions × "
            f"{len(options['detector_seeds'])} seeds (local moving to an equilibrium, exact GT start)"
        )
    else:
        lines.append(
            f"        audit grid {len(options['audit_resolutions'])} resolutions (GT cap) + the projected start at each "
            f"setting · {len(options['detector_settings'])} explicit settings × {len(options['detector_seeds'])} seeds "
            f"(local moving; GT start, cap-projected by {CAP_PROJECTION_RULE})")
        for s in options["detector_settings"]:
            lines.append(f"          {s['pair']:<14} {s['label']:<16} phase={s['phase']:<10} gamma={s['gamma']:.10g} cap={s['cap']}")
    lines.append(f"        {estimate['conditions']} detector conditions · tolerance atol={options['atol']:g} rtol={options['rtol']:g}"
                 f" · timeout {options['timeout_seconds']:g} s per run · workers planned from free memory before each pair")
    if estimate.get("max_peak_rss_bytes"):
        lines.append(f"        memory: detector worker peak RSS up to ~{estimate['max_peak_rss_bytes'] / 1e9:.1f} GB in the "
                     "stored records; each run records its own peak RSS")
    if estimate["total_seconds"] is not None:
        minutes = estimate["total_seconds"] / 60
        lines.append(f"        estimated detector time {'<1' if minutes < 1 else f'~{minutes:.0f}'} min serial "
                     f"({estimate['basis']})")
    else:
        lines.append(f"        runtime: {estimate['basis']}")
    if not options["max_nodes"]:
        lines.append("        warning: full graphs; the pure-Python oracle keeps O(edges) state and is impractical for "
                     "millions of edges — sizes below")
    if inspected:
        lines += ["", f"  {'pair':<22} {'vertices':>9} {'edges':>10} {'comms':>7} {'cap':>4} {'load s':>7}"]
        for item in inspected:
            if item["status"] != "ready":
                lines.append(f"  {item['dataset']}/{item['cover']:<12} {item['status']}: {item.get('error')}")
            else:
                lines.append(f"  {item['dataset'] + '/' + item['cover']:<22} {item['n_vertices']:>9,} {item['n_edges']:>10,} "
                             f"{item['communities']:>7,} {item['cap']:>4} {item['load_seconds']:>7.1f}")
                note = item.get("capacity") or {}
                if "refused" in note:
                    lines.append(f"      memory: REFUSED now · {note['refused']}")
                elif note:
                    lines.append(f"      memory: audit {note['audit_workers']} worker(s), "
                                 f"detector runs {note['detector_workers']} worker(s) of {options['workers']} requested "
                                 "(from the memory free now)")
                for s in item.get("settings", []):
                    lines.append(f"      {s['label']:<16} cap {s['cap']:>3}: start truncates "
                                 f"{100 * s['truncated_vertex_fraction']:.2f}% of vertices")
    downloads = [f"{r['dataset']}/{r['cover']}" for r in rows if r.get("root_kind") == "provisioned_download"]
    if downloads:
        lines.append(f"        provisioning: {', '.join(downloads)} missing locally, downloaded from SNAP when the study runs")
    lines += ["", f"  output {options['output_dir']}", "  (--inspect loads the pairs to report exact sizes and caps; "
              "a dry run downloads nothing)"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- study
def _assert_output(options: dict[str, Any]) -> None:
    v3._assert_output_safe(options["output_dir"], options["data_root"])


def provision_pairs(options: dict[str, Any], cohort: list[dict[str, Any]], *, download: bool = True,
                    say: Callable[[str], None] | None = print) -> list[dict[str, Any]]:
    """Fetch the selected pairs that are missing locally (``snap.ensure_pair``), when provisioning is on.

    Each downloaded pair's row becomes ``present`` with ``root_kind = "provisioned_download"``; a failed download
    keeps the row ``missing`` with ``provision_error`` (the pair is then a recorded gap, never silently dropped).
    With ``download=False`` (dry runs) nothing is fetched: the rows are marked as pairs the run would download.
    """
    if not options["provision"]:
        return cohort
    for row in cohort:
        if not (row.get("selected") and row.get("status") == "missing"):
            continue
        if not download:
            row.update(root_kind="provisioned_download", eligible=True, will_download=True)
            continue
        try:
            files = snap.ensure_pair(row["dataset"], row["cover"], options["data_root"], download=True,
                                     max_download_bytes=options["max_download_bytes"],
                                     say=(lambda line: say(f"  {line}")) if say is not None else None)
        except (snap.SnapLoadError, OSError) as exc:
            row["provision_error"] = f"{type(exc).__name__}: {exc}"
            if say is not None:
                say(f"  [provision failed] {row['dataset']}/{row['cover']}: {exc}")
            continue
        row.pop("reason", None)
        row.update(status="present", root_kind="provisioned_download", eligible=True, base=str(files.folder),
                   source_kind="downloaded_official_snap" if files.downloaded else "streamed_raw_gzip",
                   graph_file=files.graph.name, cover_file=files.cover_file.name,
                   bytes=files.graph.stat().st_size + files.cover_file.stat().st_size,
                   downloaded=[Path(path).name for path in files.downloaded])
    return cohort


def run_study(options: dict[str, Any], progress: Callable[[dict[str, Any]], None] | None = None) -> int:
    emit = progress or (lambda event: None)
    _assert_output(options)
    discovered = _smoke_rows(options) if options["smoke"] else discover(
        options["data_root"], options["snap_cache_dir"]
    )
    cohort = select_cohort(discovered, options["datasets"], options["covers"], options["pairs"])
    planning = bool(options["discover"] or options["dry_run"])
    provision_pairs(options, cohort, download=not planning)
    eligible = [r for r in cohort if r["eligible"]]
    if options["pairs"]:  # explicit pairs run in the order they are listed, not in the catalogue's
        listed = {tuple(pair): place for place, pair in enumerate(options["pairs"])}
        eligible.sort(key=lambda r: listed[(r["dataset"], r["cover"])])
    if options["discover"]:
        print(render_cohort(cohort))
        if options["json"]:
            print(json.dumps(cohort, indent=2, default=str))
        return 0
    if options["dry_run"]:
        inspected = inspect_pairs(options, cohort) if options["inspect"] else None
        print(render_plan(options, cohort, inspected))
        if options["json"]:
            print(json.dumps({"cohort": cohort, "inspected": inspected,
                              "estimate": estimate_runtime(options, len(eligible))}, indent=2, default=str))
        return 0
    if not eligible:
        print(render_cohort(cohort))
        print("\n  no eligible graph/cover pair: nothing to run (download missing pairs with --provision "
              "or point --data-root at a SNAP archive folder)")
        return 1
    identity = implementation_identity(options)
    jobs_pairs = [(r["dataset"], r["cover"]) for r in eligible]
    grid = scientific_grid(options, jobs_pairs)
    output_dir: Path = options["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _lock = v3._acquire_output_lock(output_dir)  # noqa: F841 - held until return
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL_NAME, "started_at": time.time(),
        "identity": identity, "effective_grid": grid, "effective_grid_sha256": _json_hash(grid),
        "locked": False,
        "lock": LOCK_NOTE,
        "options": {k: str(v) if isinstance(v, Path) else v for k, v in options.items()},
        "cohort": cohort,
        "environment": identity["runtime_versions"],
    }
    _atomic_json(output_dir / "plan.json", manifest)
    from hedonic.experiments.overlapping import gt_spectrum_report as report

    audit_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    jobs: list[Job] = []
    job_provenance: list[dict[str, Any]] = []
    plan = [
        {"job": f"{r['dataset']}-{r['cover']}", "dataset": r["dataset"], "cover": r["cover"], "policy": p,
         "gamma": g, "seed": s}
        | ({"setting": st["label"]} if st["label"] is not None else {})
        for r in eligible for p in options["isolation_policies"]
        for st in job_settings(options, r["dataset"], r["cover"]) for g in (st["gamma"],)
        for s in options["detector_seeds"]
    ]
    emit({"type": "plan", "conditions": plan, "jobs": [f"{r['dataset']}/{r['cover']}" for r in eligible]})
    done = 0
    load_failures: list[dict[str, Any]] = []
    for row in eligible:
        label = f"{row['dataset']}/{snap.cover_variant_label(row['cover'])}"
        emit({"type": "job_start", "job": label})
        try:
            job = load_job(options, row)
            capacity_plan = job_capacity(job, options)  # refuses the pair, before building anything, if it cannot fit
        except Exception as exc:  # noqa: BLE001 - recorded, never hidden
            refused = isinstance(exc, capacity.CapacityError)
            failure = {"dataset": row["dataset"], "cover": row["cover"],
                       "status": "capacity_refused" if refused else "load_failed",
                       "error": f"{type(exc).__name__}: {exc}"}
            load_failures.append(failure)
            print(f"[{failure['status']}] {label}: {exc}")
            done += conditions_per_job(options, row["dataset"], row["cover"])
            emit({"type": "job_failed", "job": label, "error": failure["error"], "skipped": done})
            continue
        jobs.append(job)
        provenance = job.provenance()
        job_provenance.append(provenance)
        _atomic_json(output_dir / "jobs" / f"{job.key}.json", provenance)
        _persist_cover(output_dir, job.gt_cover)
        executor = None
        try:
            audit_rows.extend(_audit_job(job, options, identity, emit))
            executor = _make_executor(job, options, identity, capacity_plan["detector_workers"])
            if options.get("detector_settings") is not None:
                audit_rows.extend(_audit_settings(job, options, identity, emit))
            reference = _with_omega(_base_metrics(job, options, identity, job.gt_cover, job.gt_hash), job, job.gt_cover,
                                    options, 0)
            result_rows.append({
                "row_type": "ground_truth_reference", "dataset": job.dataset, "cover": job.cover,
                "status": "reference_ground_truth", "runtime_applicable": False, "detector_runtime_seconds": None,
                "note": "untouched supplied cover; not a detector result", "initial_cover_hash": job.gt_hash,
                **{f"final_{k}": v for k, v in _numeric(reference).items()}})
            if options["audit_only"]:
                continue
            tasks = [(policy, setting["gamma"], seed, setting) for policy in options["isolation_policies"]
                     for setting in job_settings(options, job.dataset, job.cover) for seed in options["detector_seeds"]]
            records = _run_tasks(job, options, identity, tasks, executor, emit, done, capacity_plan["detector_workers"])
            done += len(tasks)
            result_rows.extend(flatten(r) for r in records if r is not None)
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        print(f"[{label}] {len([r for r in result_rows if r.get('dataset') == job.dataset and r.get('cover') == job.cover and r['row_type'] == 'detector_run'])} conditions recorded")
    _finalise(options, cohort, jobs, job_provenance, audit_rows, result_rows, manifest, load_failures, report)
    emit({"type": "finished"})
    return 0


_WORKER: dict[str, Any] = {}


def _pool_init(job, options, identity) -> None:
    _WORKER.update(job=job, options=options, identity=identity, audit_cache={})


def _pool_condition(policy: str, gamma: float, seed: int, rescore: bool, setting: dict[str, Any] | None = None):
    job, options, identity, cache = (_WORKER[k] for k in ("job", "options", "identity", "audit_cache"))
    if rescore:
        return _rescore_one(job, options, identity, policy, gamma, seed, cache, setting)
    return run_condition(job, options, identity, policy, gamma, seed, cache, setting)


GIB = 1024 ** 3


def job_capacity(job: Job, options: dict[str, Any], say=print) -> dict[str, Any]:
    """What this pair's detector runs need, against the memory free now: the workers they may use.

    The audit needs no plan: the C audit's prepared cover is O(n + m + incidences), a fraction of the loaded graph,
    and forked audit workers share it. A detector worker holds a copy of the graph, the cap's native workspace and the
    prepared covers of the returns it audits. ``capacity.CapacityError`` is raised, before anything is built, when not
    even one detector worker fits; the pair is then recorded as refused (with the figures).
    """
    graph, n = job.graph, job.graph.vcount()
    edges, incidences = graph.ecount(), sum(map(len, job.memberships))
    graph_copy = n * 200 + edges * 100 + incidences * 100  # what each spawned worker holds of the job
    requested = max(1, min(int(options["workers"]), os.cpu_count() or 1))
    lines: list[str] = []
    detector_workers = requested
    if not options["audit_only"]:
        per_detector = 2 * graph_copy + n * job.cap * 8 + 0.5 * GIB  # copy, native n x cap workspace, audited covers
        detector = capacity.plan_workers(capacity.Need(f"{job.key}: detector runs", 0, per_detector), requested)
        detector_workers = detector.workers
        lines.append(f"[capacity] {detector.line()}")
    for line in lines:
        if say is not None:
            say(line, flush=True)
    return {"audit_workers": requested, "detector_workers": detector_workers, "lines": lines}


def _make_executor(job: Job, options: dict[str, Any], identity: dict[str, Any], workers: int | None = None):
    """A spawn-based process pool for this job, or ``None`` for in-process execution (one worker)."""
    workers = max(1, min(int(workers if workers is not None else options["workers"]), os.cpu_count() or 1))
    if workers == 1:
        return None
    import concurrent.futures
    import multiprocessing

    return concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_pool_init, initargs=(job, options, identity))


def _run_tasks(job: Job, options, identity, tasks: list[tuple[str, float, int, dict[str, Any]]], executor, emit,
               done_before: int, workers: int | None = None
               ) -> list[dict[str, Any] | None]:
    """Run the (policy, gamma, seed) conditions of one job, in order in-process or concurrently in a pool."""
    results: list[dict[str, Any] | None] = [None] * len(tasks)
    rescore = bool(options["rescore_only"])

    def tagged(setting) -> dict[str, Any]:
        return {"setting": setting["label"]} if setting["label"] is not None else {}

    def announce(index: int) -> None:
        policy, gamma, seed, setting = tasks[index]
        emit({"type": "condition_start", "job": job.key, "policy": policy, "gamma": gamma, "seed": seed,
              **tagged(setting)})

    def finish(index: int, record, started: float, done: int) -> None:
        policy, gamma, seed, setting = tasks[index]
        results[index] = record
        emit({"type": "condition", "index": done, "job": job.key, "policy": policy, "gamma": gamma, "seed": seed,
              "status": None if record is None else record.get("status"), "seconds": time.monotonic() - started,
              "f1": None if record is None else (record.get("final_metrics") or {}).get("f1"), **tagged(setting)})

    if executor is None:
        cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        for index, (policy, gamma, seed, setting) in enumerate(tasks):
            announce(index)
            started = time.monotonic()
            record = (_rescore_one(job, options, identity, policy, gamma, seed, cache, setting) if rescore
                      else run_condition(job, options, identity, policy, gamma, seed, cache, setting))
            finish(index, record, started, done_before + index + 1)
        return results
    import concurrent.futures

    workers = max(1, min(int(workers if workers is not None else options["workers"]), os.cpu_count() or 1))
    in_flight: dict[Any, tuple[int, float]] = {}
    next_index, completed = 0, 0
    while next_index < len(tasks) or in_flight:
        while next_index < len(tasks) and len(in_flight) < workers:
            announce(next_index)
            policy, gamma, seed, setting = tasks[next_index]
            in_flight[executor.submit(_pool_condition, policy, gamma, seed, rescore, setting)] = (next_index, time.monotonic())
            next_index += 1
        finished, _ = concurrent.futures.wait(in_flight, return_when=concurrent.futures.FIRST_COMPLETED)
        for future in finished:
            index, started = in_flight.pop(future)
            completed += 1
            finish(index, future.result(), started, done_before + completed)
    return results


def _audit_cached(cached: dict[str, Any] | None, audit_identity: str) -> bool:
    """A stored audit answers this one when its identity (input, grid and code) equals it."""
    return bool(cached) and cached.get("audit_identity") == audit_identity


def _audit_job(job: Job, options: dict[str, Any], identity: dict[str, Any], emit) -> list[dict[str, Any]]:
    """Spectrum audit of the untouched GT, cached per exact input and grid (forked workers split the vertices)."""
    output_dir: Path = options["output_dir"]
    path = output_dir / "audit" / f"{job.key}.json"
    identity_payload = {
        "graph": job.prepared.graph_identity, "ground_truth": job.prepared.ground_truth_identity,
        "cap": job.cap, "policies": list(options["isolation_policies"]),
        "grid": [float(g) for g in options["audit_resolutions"]], "atol": options["atol"], "rtol": options["rtol"],
        "dense": options["dense"], "implementation": _compact_identity(identity),
    }
    audit_identity = _json_hash(identity_payload)
    cached = v3._read_json(path)
    if options["resume"] and _audit_cached(cached, audit_identity):
        return cached["rows"]
    started = time.monotonic()
    display_job = f"{job.dataset}/{snap.cover_variant_label(job.cover)}"
    emit({"type": "audit_start", "job": display_job})
    gammas = [float(g) for g in options["audit_resolutions"]]
    total = len(gammas) * len(options["isolation_policies"])
    workers = max(1, min(int(options["workers"]), os.cpu_count() or 1))
    emit({"type": "stage", "job": display_job, "text": "preparing the audit"})
    cover = stab.prepare(job.graph, job.memberships)  # once per pair, for every resolution, policy and worker
    rows = []
    for index, policy in enumerate(options["isolation_policies"]):
        def progress(done: int, gamma: float, before: int = index * len(gammas), policy: str = policy) -> None:
            emit({"type": "audit_progress", "job": display_job, "policy": policy, "gamma": gamma,
                  "done": before + done, "total": total, "elapsed": time.monotonic() - started})

        items = spectrum_audit(job.graph, job.memberships, cap=job.cap, allow_isolation=policy == "open_labels",
                               gammas=gammas, atol=options["atol"], rtol=options["rtol"], dense=options["dense"],
                               cover=cover, density=job.density, workers=workers, progress=progress)
        rows.extend({"row_type": "ground_truth_audit", "dataset": job.dataset, "cover": job.cover,
                     "density": job.density, "n_vertices": job.graph.vcount(), **item}
                    for item in sorted(items, key=lambda item: item["gamma"]))
    _atomic_json(path, {"audit_identity": audit_identity, "identity": identity_payload, "rows": rows,
                        "audit_runtime_seconds": time.monotonic() - started})
    return rows


def _audit_settings(job: Job, options: dict[str, Any], identity: dict[str, Any], emit) -> list[dict[str, Any]]:
    """Audit the detector start (exact or cap-projected GT) at every explicit (gamma, cap) and action policy.

    This is the supplied reference judged in exactly the benchmark's game: same resolution and cap.  Cached on disk
    per exact input like the grid audit.
    """
    output_dir: Path = options["output_dir"]
    rows: list[dict[str, Any]] = []
    for setting in job_settings(options, job.dataset, job.cover):
        start = job.start(setting["cap"])
        for policy in options["isolation_policies"]:
            identity_payload = {
                "graph": job.prepared.graph_identity, "start": start.hash, "cap": setting["cap"],
                "gamma": setting["gamma"], "policy": policy, "atol": options["atol"], "rtol": options["rtol"],
                "dense": options["dense"], "implementation": _compact_identity(identity),
            }
            audit_identity = _json_hash(identity_payload)
            path = output_dir / "audit" / f"{job.key}-setting-{audit_identity[:16]}.json"
            cached = v3._read_json(path)
            if options["resume"] and _audit_cached(cached, audit_identity):
                rows.extend(cached["rows"])
                continue
            emit({"type": "audit_start", "job": f"{job.dataset}/{snap.cover_variant_label(job.cover)}",
                  "setting": setting["label"]})
            started = time.monotonic()
            items = spectrum_audit(job.graph, start.memberships, cap=int(setting["cap"]),
                                   allow_isolation=policy == "open_labels", gammas=[float(setting["gamma"])],
                                   atol=options["atol"], rtol=options["rtol"], dense=options["dense"])
            projection = start.projection or {}
            found = [{"row_type": "start_setting_audit", "dataset": job.dataset, "cover": job.cover,
                      "density": job.density, "n_vertices": job.graph.vcount(), "setting": setting["label"],
                      "phase": setting["phase"], "start": "exact_canonical_ground_truth" if start.is_ground_truth
                      else "cap_projected_canonical_ground_truth", "start_cover_hash": start.hash,
                      "start_truncated_vertex_fraction": projection.get("truncated_vertex_fraction", 0.0),
                      "start_removed_incidence_fraction": projection.get("removed_incidence_fraction", 0.0),
                      **item} for item in items]
            _atomic_json(path, {"audit_identity": audit_identity, "identity": identity_payload, "rows": found,
                                "cap_projection": start.projection, "audit_runtime_seconds": time.monotonic() - started})
            rows.extend(found)
    return rows


def _rescore_one(job: Job, options, identity, policy, gamma, seed, audit_cache, setting=None) -> dict[str, Any] | None:
    """Detector-free re-score of a persisted condition from its stored covers."""
    payload = condition_payload(job, options, identity, policy, gamma, seed, setting)
    key = _json_hash(payload)[:24]
    output_dir: Path = options["output_dir"]
    path = v3._condition_path(output_dir, job.key, key)
    existing = v3._read_json(path)
    if existing is None or existing.get("condition_identity") != _json_hash(payload):
        return None
    if existing.get("status") not in FINAL_STATUSES:
        return existing
    final = v3._load_membership_artifact(output_dir, str(existing.get("final_membership_hash")))
    pre = v3._load_membership_artifact(output_dir, str(existing.get("pre_cleanup_membership_hash")))
    if final is None or pre is None:
        return existing
    return _score_record(existing, path, job, options, identity, policy, seed, float(gamma), final, pre, audit_cache,
                         rescored=True, setting=setting)


def _finalise(options, cohort, jobs, job_provenance, audit_rows, result_rows, manifest, load_failures, report) -> None:
    output_dir: Path = options["output_dir"]
    expected = {
        (job.dataset, job.cover, p, float(st["gamma"]), *([st["label"]] if st["label"] is not None else [])):
            len(options["detector_seeds"])
        for job in jobs for p in options["isolation_policies"] for st in job_settings(options, job.dataset, job.cover)
    }
    summary_rows = summarise(result_rows, expected)
    coverage_report = coverage(options, cohort, jobs, result_rows)
    coverage_report["load_failures"] = load_failures
    if load_failures:
        coverage_report["complete"] = False
    # A source file edited while the study ran would leave records from two implementations in one ledger.
    drift = sorted(k for k, v in tracked_file_hashes().items() if manifest["identity"]["tracked_files"].get(k) != v)
    coverage_report["implementation_drift_during_run"] = drift
    if drift:
        coverage_report["complete"] = False
        coverage_report["note"] += " IMPLEMENTATION DRIFT: " + ", ".join(drift) + " changed during the run; rerun."
        print(f"  WARNING: source files changed during the run: {', '.join(drift)} — ledger is not clean")
    v3._append_jsonl(output_dir / "results.jsonl", result_rows)
    with gzip.open(output_dir / "results.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        import csv as _csv

        keys = {k for r in result_rows for k in r}
        fields = [k for k in LEAD_COLUMNS if k in keys] + sorted(keys - set(LEAD_COLUMNS))
        writer = _csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in result_rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()})
    setting_audit = [r for r in audit_rows if r.get("row_type") == "start_setting_audit"]
    audit_rows = [r for r in audit_rows if r.get("row_type") != "start_setting_audit"]
    _write_ordered_csv(output_dir / "spectrum_audit.csv", audit_rows)
    if setting_audit:
        _write_ordered_csv(output_dir / "start_setting_audit.csv", setting_audit)
    _write_ordered_csv(output_dir / "condition_summary.csv", summary_rows)
    _write_ordered_csv(output_dir / "coverage_report.csv", summary_rows_coverage(summary_rows))
    _atomic_json(output_dir / "coverage_report.json", coverage_report)
    _atomic_json(output_dir / "cohort.json", {"cohort": cohort, "load_failures": load_failures})
    plots = report.write_all(output_dir, audit_rows, result_rows, summary_rows, job_provenance, coverage_report, options)
    manifest.update(
        {"finished_at": time.time(), "jobs": job_provenance, "coverage": coverage_report,
         "artifacts": {"plots": plots, "results_jsonl": "results.jsonl", "results_csv_gz": "results.csv.gz",
                       "condition_summary": "condition_summary.csv", "spectrum_audit": "spectrum_audit.csv",
                       "coverage_report": "coverage_report.json"}}
    )
    _atomic_json(output_dir / "manifest.json", manifest)
    print(f"\n  wrote {output_dir}")
    print(f"  {coverage_report['recorded_detector_conditions']}/{coverage_report['expected_detector_conditions']} "
          f"conditions · {coverage_report['verified_equilibria']} verified equilibria · "
          f"{coverage_report['non_equilibrium_returns']} non-equilibrium returns · "
          f"{coverage_report['timeouts']} timeouts · {coverage_report['failures_and_unsupported']} failures · "
          f"complete={coverage_report['complete']}")


LEAD_COLUMNS = ("row_type", "dataset", "cover", "setting", "phase", "action_policy", "gamma", "gamma_over_density",
                "seed", "status",
                "expected_runs", "returned_covers", "verified_equilibria", "non_equilibrium_returns", "timeouts",
                "failures", "missing", "unchanged_returns")


def _write_ordered_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """CSV with identifying columns first (v3's writer sorts all columns alphabetically)."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = {k for row in rows for k in row}
    fields = [k for k in LEAD_COLUMNS if k in keys] + sorted(keys - set(LEAD_COLUMNS))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    temporary.replace(path)


def summary_rows_coverage(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = ("dataset", "cover", "setting", "action_policy", "gamma", "expected_runs", "returned_covers", "verified_equilibria",
            "non_equilibrium_returns", "timeouts", "failures", "missing", "unchanged_returns")
    return [{k: row.get(k) for k in keep if k != "setting" or "setting" in row} for row in summary_rows]


def replot(options: dict[str, Any]) -> int:
    """Rebuild summaries and figures from the stored ledger (no graph loading, no detection)."""
    from hedonic.experiments.overlapping import gt_spectrum_report as report

    output_dir: Path = options["output_dir"]
    manifest = v3._read_json(output_dir / "manifest.json")
    if manifest is None:
        raise ValueError(f"no manifest.json in {output_dir}; run the study first")
    result_rows = [json.loads(line) for line in (output_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    audit_rows = []
    for path in sorted((output_dir / "audit").glob("*.json")):
        audit_rows.extend(r for r in (v3._read_json(path) or {}).get("rows", [])
                          if r.get("row_type") != "start_setting_audit")
    grid = manifest["effective_grid"]
    settings_options = {"detector_settings": grid.get("detector_settings"),
                        "detector_resolutions": grid["detector_resolutions"]}
    expected = {
        (job[0], job[1], p, float(st["gamma"]), *([st["label"]] if st["label"] is not None else [])):
            len(grid["detector_seeds"])
        for job in grid["jobs"] for p in grid["action_policies"] for st in job_settings(settings_options, job[0], job[1])
    }
    summary_rows = summarise(result_rows, expected)
    _write_ordered_csv(output_dir / "condition_summary.csv", summary_rows)
    coverage_report = manifest.get("coverage", {})
    plots = report.write_all(output_dir, audit_rows, result_rows, summary_rows, manifest.get("jobs", []),
                             coverage_report, {**options, "isolation_policies": grid["action_policies"],
                                               "atol": grid["robustness_atol"], "rtol": grid["robustness_rtol"]})
    print(f"  wrote {len(plots)} figure/table files under {output_dir}")
    return 0


# --------------------------------------------------------------------------- CLI
def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        add_help=add_help,
        prog="hedonic-exp overlapping-gt-spectrum",
        description=(
            "SNAP ground-truth robustness spectrum: for every graph/overlapping-cover pair in the local SNAP cache, "
            "audit the supplied cover across resolution (fraction of vertices with no profitable unilateral action, "
            "fixed- and open-label) and run seeded local moving from the exact ground-truth cover, auditing and scoring "
            "every returned cover. Packaged configs run unlocked (the identity is the package source digest, the "
            "hedonic version and the core SHA-256)."
        ),
    )
    p.add_argument("--config", help="TOML file, or the name of a shipped config: full -> overlapping-gt-spectrum-full.toml "
                                    "(default: the shipped overlapping-gt-spectrum.toml)")
    p.add_argument("--profile", choices=("smoke", "study"), default="study",
                   help="smoke: deterministic built-in graphs, no data (default: study)")
    p.add_argument("--datasets", help="comma list or 'auto' = every network with an available pair (default auto)")
    p.add_argument("--cover", help="comma list of top5k,all or 'auto' = every available variant (default auto)")
    p.add_argument("--pairs", help="explicit dataset/cover list, e.g. amazon/top5k,dblp/all (overrides --datasets/--cover)")
    p.add_argument("--data-root", help="folder with SNAP archives (default HEDONIC_NETWORKS_DIR)")
    p.add_argument("--snap-cache-dir", help="hedonic download cache also inspected (default ~/.cache/hedonic/snap)")
    p.add_argument("--output-dir", help="artifact folder (default: the config's output_dir, else "
                                        "<output_dir>/artifacts/overlapping/gt_spectrum_v1)")
    p.add_argument("--uncovered-policy", choices=("covered-induced", "singleton"),
                   help="partial-cover policy; singleton is a synthetic-completion sensitivity, never GT")
    p.add_argument("--max-nodes", type=int, help="bounded induced-subgraph size; 0 = full graph (default 3000)")
    p.add_argument("--audit-resolutions", help="audit grid: START:STOP:COUNT, geom:START:STOP:COUNT, a list, or joined with + (default 0:1:101+geom:1e-4:1:33)")
    p.add_argument("--resolutions", help="detector grid, same syntax (default 0+geom:1e-4:1:9)")
    p.add_argument("--isolation-policies", help="fixed_labels,open_labels (default both)")
    p.add_argument("--seeds", help="detector seeds, e.g. 0-4 (default 0-4)")
    p.add_argument("--timeout-per-run", type=float, help="hard wall-clock seconds per detector run (default 3600)")
    omega = p.add_mutually_exclusive_group()
    omega.add_argument("--omega", dest="omega", action="store_true", default=None)
    omega.add_argument("--no-omega", dest="omega", action="store_false")
    p.add_argument("--omega-sample-size", type=int, help="sampled Omega pairs (default 100000)")
    p.add_argument("--robustness-atol", type=float)
    p.add_argument("--robustness-rtol", type=float)
    p.add_argument("--dense-oracle", action="store_true", help="dense candidate sets (validation only)")
    p.add_argument("--workers", type=int,
                   help="processes scoring conditions/audits concurrently (default 4; 1 = in-process). "
                        "Results do not depend on it; each worker holds one graph and its scoring state")
    p.add_argument("--discover", action="store_true", help="list present/missing/unsupported pairs and exit")
    p.add_argument("--dry-run", action="store_true", help="report cohort, plan, tolerances and estimates; run nothing")
    p.add_argument("--inspect", action="store_true", help="with --dry-run: load pairs to report exact sizes and caps")
    p.add_argument("--audit-only", action="store_true", help="spectrum audit of the supplied covers, no detector")
    p.add_argument("--force", action="store_true", help="recompute even compatible stored records")
    p.add_argument("--resume", action="store_true", help="reuse compatible stored records (default)")
    p.add_argument("--retry-failed", action="store_true", help="re-run stored timeout/failed conditions")
    p.add_argument("--rescore-only", action="store_true", help="re-audit and re-score stored covers; no detector")
    p.add_argument("--replot", action="store_true", help="rebuild summaries and figures from the stored ledger")
    p.add_argument("--provision", action=argparse.BooleanOptionalAction, default=None,
                   help="download the selected pairs that are missing locally (official SNAP files, into --data-root). "
                        "Default: the TOML key provision, else on when the run names its pairs (pairs / --pairs, as "
                        "every queue step does) and off for auto discovery; --no-provision never downloads")
    p.add_argument("--max-download-gb", type=float, help="abort a provisioning download larger than this")
    p.add_argument("--allow-unlocked", action="store_true", help=argparse.SUPPRESS)  # 1.x flag; every run is unlocked
    p.add_argument("--json", action="store_true", help="also print machine-readable output")
    return p


# --------------------------------------------------------------------------- hedonic front door
FRONT_FLAGS = {"--detach": 0, "--foreground": 0, "-y": 0, "--yes": 0, "-i": 0, "--interactive": 0, "--name": 1}


def _strip_front(argv: list[str]) -> list[str]:
    out, skip = [], 0
    for token in argv:
        if skip:
            skip -= 1
            continue
        flag = token.split("=", 1)[0]
        if flag in FRONT_FLAGS:
            skip = 0 if "=" in token else FRONT_FLAGS[flag]
            continue
        out.append(token)
    return out


def _pretty(argv: list[str]) -> str:
    import shlex

    words, groups = [shlex.quote(a) for a in argv], []
    for word in words:
        if word.startswith("--") or not groups:
            groups.append([word])
        else:
            groups[-1].append(word)
    return " \\\n      ".join(" ".join(g) for g in ["hedonic run spectrum".split(), *groups] if g) if groups else \
        "hedonic run spectrum"


def wizard(defaults: dict[str, Any], rows: list[dict[str, Any]], default_output: str) -> list[str]:
    """Arrow-key wizard (``hedonic run spectrum`` in a terminal). Returns the study's flag list."""
    from hedonic.experiments.overlapping import tui
    from hedonic.experiments.overlapping.tui import style

    eligible = [r for r in rows if r["status"] == "present"]
    print(style("\n  hedonic · SNAP ground-truth robustness spectrum\n", "bold", "magenta"))
    print(render_cohort(select_cohort(rows, "auto", "auto")))
    print()
    if not eligible:
        print(style("  no graph/cover pair is available locally; point --data-root at a SNAP archive folder "
                    "(or name pairs with --pairs to download them).", "yellow"))
        raise tui.Cancelled
    registered = f"{len(eligible)} pairs · {defaults['max_nodes'] or 'full':,} vertices · both action spaces · " \
                 f"seeds {defaults['seeds'][0]}-{defaults['seeds'][-1]}" if defaults["max_nodes"] else "full graphs"
    choice = tui.select("What would you like to do?", [
        ("Run the registered study", registered),
        ("Audit only", "resolution spectrum of the supplied covers, no detector runs"),
        ("Customize", "pairs, size, seeds, action spaces, grids, output"),
        ("Dry run", "show the plan and estimates, run nothing"),
        ("Quit", "")])
    if choice == 4:
        raise tui.Cancelled
    if choice == 3:
        return ["--dry-run", "--inspect"]
    if choice == 1:
        return ["--audit-only"]
    if choice == 0:
        return []
    argv: list[str] = []
    while True:
        argv = []
        picks = tui.multiselect(
            "Graph/cover pairs to include",
            [(f"{r['dataset']}/{r['cover']}", f"{r['bytes'] / 1e6:.0f} MB on disk") for r in eligible],
            list(range(len(eligible))))
        chosen = [eligible[i] for i in picks]
        if len(chosen) != len(eligible):
            argv += ["--pairs", ",".join(f"{r['dataset']}/{r['cover']}" for r in chosen)]
        sizes = [3000, 1000, 10000, 0, -1]
        i = tui.select("Graph size (deterministic induced subgraph around the labelled communities)",
                       [("3,000 vertices", "registered; seconds per run"), ("1,000 vertices", "quick"),
                        ("10,000 vertices", ""), ("Full graph", "the pure-Python oracle keeps O(edges) state: hours or more"),
                        ("Custom…", "")], 0)
        size = sizes[i] if sizes[i] >= 0 else int(tui.text("Number of vertices", "3000",
                                                            lambda v: None if v.isdigit() else "enter an integer"))
        if size != 3000:
            argv += ["--max-nodes", str(size)]
        seeds = [("5 seeds", "0-4"), ("1 seed", "0"), ("3 seeds", "0-2"), ("10 seeds", "0-9"), ("Custom…", "")]
        i = tui.select("Detector seeds (restarts on the same graph)", [(a, b) for a, b in seeds], 0)
        spec = seeds[i][1] or tui.text("Seeds (e.g. 0-4 or 0,3,7)", "0-4", lambda v: _validate(v3._parse_range, v))
        if spec != "0-4":
            argv += ["--seeds", spec]
        pol = tui.multiselect("Action spaces to audit and run", [
            ("Fixed labels", "no new community may be created"), ("Open labels", "a fresh singleton community is allowed")],
            [0, 1])
        if len(pol) == 1:
            argv += ["--isolation-policies", ACTION_POLICIES[pol[0]]]
        grids = [("Registered", "detector γ: 0 and 1e-4…1 (10 points); audit: 131 points"),
                 ("Coarse", "detector 0, 1e-3, 1e-2, 0.1, 1; audit 0:1:21"), ("Custom…", "")]
        i = tui.select("Resolution grids", grids, 0)
        if i == 1:
            argv += ["--resolutions", "0,1e-3,1e-2,0.1,1", "--audit-resolutions", "0:1:21+geom:1e-4:1:9"]
        elif i == 2:
            argv += ["--resolutions", tui.text("Detector grid", "0+geom:1e-4:1:9", lambda v: _validate(_parse_grid, v)),
                     "--audit-resolutions", tui.text("Audit grid", "0:1:101+geom:1e-4:1:33",
                                                    lambda v: _validate(_parse_grid, v))]
        if tui.select("Omega index", [("Compute it (sampled)", "100,000 vertex pairs"),
                                      ("Skip it", "faster")], 0) == 1:
            argv.append("--no-omega")
        output = tui.text("Save results in", default_output)
        if output != default_output:
            argv += ["--output-dir", output]
        print()
        print(style("  Summary", "bold"))
        print("  pairs    " + (f"all {len(chosen)} available" if len(chosen) == len(eligible) else
                                ", ".join(f"{r['dataset']}/{r['cover']}" for r in chosen)))
        print(f"  size     {'full graph' if size == 0 else f'{size:,} vertices'} · seeds {spec} · "
              f"{' + '.join(ACTION_POLICIES[j] for j in pol)}")
        print(style("  same run without the wizard:", "dim"))
        print(style("    " + _pretty(argv), "dim"))
        if size == 0:
            print(style("  warning: full graphs are impractical for the LiveJournal/Wikipedia-scale networks", "yellow"))
        answer = tui.select("Start?", ["Run", "Change settings", "Quit"])
        if answer == 0:
            return argv
        if answer == 2:
            raise tui.Cancelled


def _validate(fn, value):
    try:
        fn(value)
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    return None


def front_main(argv: list[str] | None = None) -> int:
    """``hedonic run spectrum``: wizard or flags, durable tmux/process run with a live viewer."""
    import sys

    from hedonic.experiments.overlapping import quickstart as qs
    from hedonic.experiments.overlapping import runmanager, tui, userconfig
    from hedonic.experiments.overlapping.tui import style

    argv = list(sys.argv[1:] if argv is None else argv)
    front = argparse.ArgumentParser(prog="hedonic run spectrum", parents=[build_parser(add_help=False)],
                                    description="SNAP ground-truth robustness spectrum (durable). Without arguments, "
                                                "in a terminal, a wizard starts.")
    front.add_argument("-i", "--interactive", action="store_true", help="open the wizard (default when no arguments)")
    front.add_argument("-y", "--yes", action="store_true", help="run with the registered defaults, never prompt")
    front.add_argument("--name", help="run name (default spectrum-YYYYmmdd-HHMMSS)")
    front.add_argument("--detach", action="store_true", help="start in the background and return")
    front.add_argument("--foreground", action="store_true", help="run in this terminal (not recoverable)")
    try:
        args = front.parse_args(argv)
    except SystemExit as exc:  # --help / a bad flag: argparse already printed; return the code
        return int(exc.code or 0)
    study = _strip_front(argv)
    machine_flags: list[str] = []
    try:
        if args.name:
            qs.parse_name(args.name)
        machine = userconfig.defaults()
        if args.data_root is None and machine.get("network_root"):
            machine_flags += ["--data-root", str(machine["network_root"])]
        if args.snap_cache_dir is None and machine.get("cache_dir"):
            machine_flags += ["--snap-cache-dir", str(Path(str(machine["cache_dir"])).expanduser() / "snap")]
        study += machine_flags
        options = resolve_options(build_parser().parse_args(study))
    except ValueError as exc:
        print(f"hedonic run spectrum: error: {exc}\n(see `hedonic run spectrum --help`)", file=sys.stderr)
        return 2
    direct = options["discover"] or options["dry_run"] or options["replot"] or args.foreground
    try:
        if (args.interactive or (not argv and tui.interactive())) and not args.yes:
            rows = discover(options["data_root"], options["snap_cache_dir"])
            wizard_name = args.name or runmanager.new_name("spectrum")
            args.name = wizard_name
            picked = wizard(options, rows, str(expand_path(machine.get("output_dir") or RUNS_ROOT) / wizard_name))
            study = picked + machine_flags
            options = resolve_options(build_parser().parse_args(study))
            direct = options["discover"] or options["dry_run"]
        if direct:
            return run_study(options) if not options["replot"] else replot(options)
        explicit_output = args.output_dir is not None or ("--output-dir" in study)
        name = args.name or options.get("run_name") or runmanager.new_name("spectrum")
        base = machine.get("output_dir") or RUNS_ROOT
        output = str(options["output_dir"]) if explicit_output else str(expand_path(base) / name)
        entry = runmanager.launch_spectrum(study, name, output)
        where = f"tmux session {entry['session']}" if entry["backend"] == "tmux" else f"background process (log {entry['log']})"
        out = sys.stderr if options["json"] else sys.stdout
        print(style(f"  started {entry['name']}", "green", "bold") + f" in a {where}", file=out)
        print(style(f"  it survives closing this terminal · reopen: hedonic run attach {entry['name']} · "
                    f"stop: hedonic run stop {entry['name']}", "dim"), file=out)
        if options["json"] and not args.detach:
            return runmanager.wait_and_print_json(entry)
        if args.detach or not sys.stdout.isatty():
            return 0
        time.sleep(0.5)
        return runmanager.view(entry)
    except (tui.Cancelled, KeyboardInterrupt):
        print(style("\n  cancelled", "yellow"))
        return 130


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        options = resolve_options(args)
        if options["replot"]:
            return replot(options)
        return run_study(options)
    except ValueError as exc:
        print(f"overlapping-gt-spectrum: error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
