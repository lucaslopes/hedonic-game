"""Full-DBLP F1 vs resolution for multi-phase overlapping community_hedonic.

Sweeps resolution γ over [0, 1], runs::

    community_hedonic(
        n_iterations=-1,
        only_local_moving=False,
        allow_isolation=True,
        max_memberships=K,
    )

where ``K`` is the number of ground-truth communities with more than one node.
Each γ is repeated over multiple seeds (seed-dependent random initial membership)
so F1 can be summarized with a confidence interval.

**Persistence / resume**

Each ``(resolution, seed)`` run is written immediately under::

    <output_dir>/runs/res_<γ>_seed_<s>.json

including the predicted **cover** (community lists), quality, wallclock, seed,
algorithm flags, and metrics. Re-running the same ``--output_dir`` **resumes**
by default (skips completed cells). Covers stay on disk so later metrics
(e.g. a new ARI-like score) can be computed via ``--rescore-only`` without
re-running detection.

CLI::

    hedonic-exp overlapping-resolution --smoke --output_dir /tmp/res-f1
    hedonic-exp overlapping-resolution --config configs/hedonic.toml \\
        --resolutions 0:1:11 --seeds 0-4
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import igraph as ig
import numpy as np

from hedonic import Game
from hedonic.experiments import config as exp_config
from hedonic.experiments.config import DBLP_DIR, OUTPUT_DIR
from hedonic.experiments.overlapping.dblp_full import load_dblp
from hedonic.experiments.overlapping.metrics import (
    cover_quality,
    evaluate_cover,
    partition_to_cover_lists,
    quality_overlapping_cpm,
    symmetric_best_match_metrics,
)
from hedonic.utils import sample_uniform_ints

# Fixed algorithm flags (not CLI knobs) — match the experiment contract.
N_ITERATIONS = -1
ONLY_LOCAL_MOVING = False
ALLOW_ISOLATION = True

DEFAULT_OUTPUT_DIR = Path("overlapping_resolution_f1_results")
DEFAULT_RESOLUTIONS = "0:1:11"  # linspace 0..1 inclusive, 11 points
DEFAULT_SEEDS = "0-4"  # five seeds for CI
DEFAULT_CI_LEVEL = 0.95
DEFAULT_SINGLETON_MODE = "size_ge_2"
DEFAULT_OMEGA_SAMPLE_SIZE = 100_000
RUNS_SUBDIR = "runs"


def log(msg: str, t0: float | None = None) -> None:
    elapsed = f"  [{time.time() - t0:.1f}s]" if t0 else ""
    print(f"{msg}{elapsed}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_gt_communities_gt1(gt: Sequence[Sequence[int]]) -> int:
    """Count ground-truth communities with more than one node."""
    return sum(1 for c in gt if len(c) > 1)


def filter_gt_communities_gt1(
    gt: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Keep only GT communities with more than one node."""
    return [list(c) for c in gt if len(c) > 1]


def resolve_max_memberships(
    max_memberships: int | None,
    gt: Sequence[Sequence[int]],
) -> int:
    """Explicit K, else # GT communities with size > 1 (at least 1)."""
    if max_memberships is not None:
        return max(1, int(max_memberships))
    return max(1, count_gt_communities_gt1(gt))


def parse_resolutions(spec: str) -> list[float]:
    """Parse resolution grid.

    Formats
    -------
    * ``start:stop:n`` — ``np.linspace(start, stop, n)`` (inclusive ends)
    * comma list — ``0,0.25,0.5,0.75,1``
    """
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("empty resolution spec")
    if ":" in spec and "," not in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"resolution range must be start:stop:n, got {spec!r}"
            )
        start, stop, n = float(parts[0]), float(parts[1]), int(parts[2])
        if n < 1:
            raise ValueError("resolution count n must be >= 1")
        return [float(x) for x in np.linspace(start, stop, n)]
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"no resolutions parsed from {spec!r}")
    return vals


def parse_seeds(spec: str) -> list[int]:
    """Parse seed list: ``0-4`` → 0..4, or comma list ``0,1,7``."""
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("empty seed spec")
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        start, stop = int(a.strip()), int(b.strip())
        if stop < start:
            raise ValueError(f"seed range end < start: {spec!r}")
        return list(range(start, stop + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def seeded_initial_membership(
    n_vertices: int,
    n_communities: int,
    seed: int,
) -> list[int]:
    """Seed-dependent random disjoint labels in ``[0, K)``, labels contiguous.

    Used as overlapping init (flat vector expanded to singleton lists per
    vertex) so multi-seed F1 CIs are not identical duplicates when the
    Leiden binding does not take an RNG seed.
    """
    k = max(1, int(n_communities))
    if k == 1:
        return [0] * n_vertices
    raw = sample_uniform_ints(n_vertices, k - 1, seed).tolist()
    uniq = sorted(set(raw))
    if len(uniq) == k and uniq[0] == 0 and uniq[-1] == k - 1:
        return [int(x) for x in raw]
    remap = {old: new for new, old in enumerate(uniq)}
    return [remap[int(x)] for x in raw]


def mean_ci(
    values: Sequence[float],
    *,
    confidence: float = DEFAULT_CI_LEVEL,
) -> dict[str, float]:
    """Mean and two-sided CI half-width over seed replicates."""
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_half": float("nan"),
            "n": 0,
            "confidence": float(confidence),
        }
    mean = float(np.mean(arr))
    if n == 1:
        return {
            "mean": mean,
            "std": 0.0,
            "se": 0.0,
            "ci_low": mean,
            "ci_high": mean,
            "ci_half": 0.0,
            "n": 1,
            "confidence": float(confidence),
        }
    std = float(np.std(arr, ddof=1))
    se = std / float(np.sqrt(n))
    try:
        from scipy import stats

        tcrit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    except Exception:  # pragma: no cover
        tcrit = 1.96
    half = tcrit * se
    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "ci_half": half,
        "n": n,
        "confidence": float(confidence),
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Run cache (per γ × seed) — covers + metadata for resume / re-score
# ---------------------------------------------------------------------------


def run_cache_key(resolution: float, seed: int) -> str:
    """Stable id for one (γ, seed) cell."""
    return f"res_{float(resolution):.10f}_seed_{int(seed)}"


def run_cache_path(runs_dir: Path | str, resolution: float, seed: int) -> Path:
    return Path(runs_dir) / f"{run_cache_key(resolution, seed)}.json"


def atomic_write_json(path: Path | str, payload: dict[str, Any]) -> Path:
    """Write JSON atomically (temp file + replace) so crashes don't corrupt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def is_run_complete(record: dict[str, Any] | None) -> bool:
    """True if a cached record is usable (has cover + complete status)."""
    if not record:
        return False
    if record.get("status") == "failed":
        return False
    cover = record.get("cover")
    if not isinstance(cover, list):
        return False
    status = record.get("status")
    if status in ("complete", "ok"):
        return True
    # Legacy / partial writes: cover + a metric field is enough to resume
    if status is None:
        return "f1" in record or isinstance(record.get("metrics"), dict)
    return False


def load_run_file(path: Path | str) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_run_file(record: dict[str, Any], path: Path | str) -> Path:
    return atomic_write_json(path, record)


def load_completed_runs(runs_dir: Path | str) -> dict[tuple[float, int], dict[str, Any]]:
    """Load all complete run records keyed by ``(resolution, seed)``."""
    runs_dir = Path(runs_dir)
    out: dict[tuple[float, int], dict[str, Any]] = {}
    if not runs_dir.is_dir():
        return out
    for path in sorted(runs_dir.glob("res_*.json")):
        rec = load_run_file(path)
        if not is_run_complete(rec):
            continue
        assert rec is not None
        key = (float(rec["resolution"]), int(rec["seed"]))
        out[key] = rec
    return out


def metrics_from_cover(
    cover: Sequence[Sequence[int]],
    gt: Sequence[Sequence[int]],
    n_vertices: int,
    *,
    compute_omega: bool = False,
    singleton_mode: str = "all",
    omega_sample_size: int = DEFAULT_OMEGA_SAMPLE_SIZE,
    omega_seed: int = 0,
) -> dict[str, Any]:
    """Score a cached cover vs GT without invoking community detection."""
    if singleton_mode not in ("all", "size_ge_2"):
        raise ValueError("singleton_mode must be 'all' or 'size_ge_2'")
    return evaluate_cover(
        [list(c) for c in cover],
        [list(c) for c in gt],
        int(n_vertices),
        compute_omega=compute_omega,
        singleton_mode=singleton_mode,
        omega_sample_size=omega_sample_size,
        omega_seed=omega_seed,
    )


def singleton_modes(singleton_mode: str) -> tuple[str, ...]:
    """Expand the CLI singleton selector into concrete evaluation modes."""
    if singleton_mode == "both":
        return ("all", "size_ge_2")
    if singleton_mode in ("all", "size_ge_2"):
        return (singleton_mode,)
    raise ValueError("singleton_mode must be 'all', 'size_ge_2', or 'both'")


def rescore_record(
    record: dict[str, Any],
    gt: Sequence[Sequence[int]],
    *,
    n_vertices: int | None = None,
    compute_omega: bool = False,
    singleton_mode: str = DEFAULT_SINGLETON_MODE,
    omega_sample_size: int = DEFAULT_OMEGA_SAMPLE_SIZE,
    omega_seed: int = 0,
) -> dict[str, Any]:
    """Recompute metrics on a cached run **without** re-running detection.

    Updates ``metrics`` / convenience ``f1``/… fields in a **copy**; cover and
    wallclock are left unchanged. Use this when adding a new accuracy metric
    later (compute from ``record["cover"]``).
    """
    rec = dict(record)
    cover = rec.get("cover")
    if not isinstance(cover, list):
        raise ValueError("record has no cover; cannot rescore")
    n = int(n_vertices if n_vertices is not None else rec.get("n_vertices") or 0)
    if n <= 0:
        # Infer n from cover membership
        n = max((max(c) for c in cover if c), default=-1) + 1
    # Preserve the historical resolution score exactly: all predicted
    # communities versus GT communities of size >= 2.  This intentionally
    # differs from the new consistent singleton modes and keeps old plots and
    # cache consumers stable.
    legacy = symmetric_best_match_metrics(
        cover,
        filter_gt_communities_gt1(gt),
        singleton_mode="all",
    )
    by_mode = {
        mode: metrics_from_cover(
            cover,
            gt,
            n,
            compute_omega=compute_omega,
            singleton_mode=mode,
            omega_sample_size=omega_sample_size,
            omega_seed=omega_seed,
        )
        for mode in singleton_modes(singleton_mode)
    }
    old_metrics = rec.get("metrics")
    rec["metrics"] = dict(old_metrics) if isinstance(old_metrics, dict) else {}
    rec["metrics"].update(
        {
            "f1": float(legacy["f1"]),
            "symmetric_best_match_f1": float(legacy["f1"]),
            "jaccard": float(legacy["jaccard"]),
            "symmetric_best_match_jaccard": float(legacy["jaccard"]),
            "precision": float(legacy["precision"]),
            "recall": float(legacy["recall"]),
            "omega": None,
            "n_predicted_comms": int(legacy["n_predicted_comms"]),
            "n_gt_comms": int(legacy["n_gt_comms"]),
            "by_singleton_mode": by_mode,
        }
    )
    rec["metrics_by_singleton_mode"] = by_mode
    primary_mode = "size_ge_2" if "size_ge_2" in by_mode else "all"
    primary = by_mode[primary_mode]
    rec["metrics"]["omega"] = primary.get("omega")
    rec["omega"] = primary.get("omega")
    rec["sampled_omega"] = primary.get("omega")
    # Convenient unsuffixed aliases for the recommended metrics. Historical
    # aliases above are excluded so f1/jaccard semantics do not change.
    legacy_names = {
        "f1",
        "jaccard",
        "precision",
        "recall",
        "omega",
        "n_predicted_comms",
        "n_gt_comms",
        "symmetric_best_match_f1",
        "symmetric_best_match_jaccard",
    }
    for key, value in primary.items():
        if key not in legacy_names:
            rec["metrics"][key] = value
            rec[key] = value
    for mode, mode_metrics in by_mode.items():
        for key, value in mode_metrics.items():
            rec[f"{key}_{mode}"] = value
    # Convenience aliases (plot / aggregate path)
    rec["f1"] = rec["metrics"]["f1"]
    rec["symmetric_best_match_f1"] = rec["metrics"]["f1"]
    rec["jaccard"] = rec["metrics"]["jaccard"]
    rec["precision"] = rec["metrics"]["precision"]
    rec["recall"] = rec["metrics"]["recall"]
    rec["n_predicted_comms"] = rec["metrics"]["n_predicted_comms"]
    rec["n_gt_comms"] = rec["metrics"]["n_gt_comms"]
    rec["singleton_mode"] = singleton_mode
    rec["omega_enabled"] = bool(compute_omega)
    rec["omega_method"] = "sampled_pairwise" if compute_omega else None
    rec["omega_sample_size"] = int(omega_sample_size) if compute_omega else None
    rec["omega_seed"] = int(omega_seed) if compute_omega else None
    rec["rescored_at"] = utc_now_iso()
    return rec


# ---------------------------------------------------------------------------
# Single run + sweep
# ---------------------------------------------------------------------------


def run_one(
    game: Game,
    gt: Sequence[Sequence[int]],
    *,
    resolution: float,
    seed: int,
    max_memberships: int,
    n_iterations: int = N_ITERATIONS,
    only_local_moving: bool = ONLY_LOCAL_MOVING,
    allow_isolation: bool = ALLOW_ISOLATION,
    singleton_mode: str = DEFAULT_SINGLETON_MODE,
    compute_omega: bool = False,
    omega_sample_size: int = DEFAULT_OMEGA_SAMPLE_SIZE,
    omega_seed: int = 0,
) -> dict[str, Any]:
    """One (resolution, seed) detection + full metadata + cover for cache."""
    n = game.vcount()
    m = int(game.ecount())
    k = max(1, int(max_memberships))
    init = seeded_initial_membership(n, k, seed)
    t0 = time.perf_counter()
    cover_obj = game.community_hedonic(
        resolution=float(resolution),
        n_iterations=int(n_iterations),
        only_local_moving=bool(only_local_moving),
        allow_isolation=bool(allow_isolation),
        max_memberships=k,
        initial_membership=init,
        seed=int(seed),
    )
    elapsed = time.perf_counter() - t0
    cover = partition_to_cover_lists(cover_obj)
    q = cover_quality(cover_obj)
    if q is None:
        q = quality_overlapping_cpm(game, cover, float(resolution))
    record = {
        "status": "complete",
        "resolution": float(resolution),
        "seed": int(seed),
        "max_memberships": k,
        "n_iterations": int(n_iterations),
        "only_local_moving": bool(only_local_moving),
        "allow_isolation": bool(allow_isolation),
        "wallclock_s": float(elapsed),
        "quality": float(q) if q is not None else None,
        "n_vertices": n,
        "n_edges": m,
        # Cache for later metrics (ARI, …) without re-running detection
        "cover": [[int(v) for v in comm] for comm in cover],
        "initial_membership": [int(x) for x in init],
        "completed_at": utc_now_iso(),
        "from_cache": False,
    }
    return rescore_record(
        record,
        gt,
        n_vertices=n,
        compute_omega=compute_omega,
        singleton_mode=singleton_mode,
        omega_sample_size=omega_sample_size,
        omega_seed=omega_seed,
    )


def aggregate_runs(
    runs: Sequence[dict[str, Any]],
    *,
    confidence: float = DEFAULT_CI_LEVEL,
) -> list[dict[str, Any]]:
    """Group by resolution; retain legacy F1 CI and aggregate new metrics."""
    by_res: dict[float, list[dict[str, Any]]] = {}
    for r in runs:
        by_res.setdefault(float(r["resolution"]), []).append(r)

    out: list[dict[str, Any]] = []
    for res in sorted(by_res):
        group = by_res[res]
        f1s = [float(g["f1"]) for g in group]
        stats = mean_ci(f1s, confidence=confidence)
        qualities = [
            float(g["quality"])
            for g in group
            if g.get("quality") is not None
        ]
        row: dict[str, Any] = {
                "resolution": res,
                "f1_mean": stats["mean"],
                "f1_std": stats["std"],
                "f1_se": stats["se"],
                "f1_ci_low": stats["ci_low"],
                "f1_ci_high": stats["ci_high"],
                "f1_ci_half": stats["ci_half"],
                "f1_samples": f1s,
                "seeds": [int(g["seed"]) for g in group],
                "n_seeds": stats["n"],
                "confidence": stats["confidence"],
                "jaccard_mean": float(
                    np.mean([float(g["jaccard"]) for g in group])
                ),
                "wallclock_s_mean": float(np.mean([
                    float(g.get("wallclock_s", 0.0)) for g in group
                ])),
                "quality_mean": (
                    float(np.mean(qualities)) if qualities else None
                ),
                "max_memberships": int(group[0].get("max_memberships", 1)),
                "n_iterations": int(group[0].get("n_iterations", N_ITERATIONS)),
                "only_local_moving": bool(
                    group[0].get("only_local_moving", ONLY_LOCAL_MOVING)
                ),
                "allow_isolation": bool(
                    group[0].get("allow_isolation", ALLOW_ISOLATION)
                ),
            }

        mode_names = sorted(
            {
                mode
                for run in group
                for mode in (
                    run.get("metrics_by_singleton_mode", {}).keys()
                    if isinstance(run.get("metrics_by_singleton_mode"), dict)
                    else ()
                )
            }
        )
        mode_summaries: dict[str, dict[str, Any]] = {}
        for mode in mode_names:
            metric_dicts = [
                run["metrics_by_singleton_mode"][mode]
                for run in group
                if isinstance(run.get("metrics_by_singleton_mode"), dict)
                and isinstance(run["metrics_by_singleton_mode"].get(mode), dict)
            ]
            keys = sorted({key for metrics in metric_dicts for key in metrics})
            summary: dict[str, Any] = {}
            for key in keys:
                values = [
                    float(metrics[key])
                    for metrics in metric_dicts
                    if isinstance(metrics.get(key), (int, float))
                    and not isinstance(metrics.get(key), bool)
                    and metrics.get(key) is not None
                ]
                if not values:
                    continue
                summary[f"{key}_mean"] = float(np.mean(values))
                row[f"{key}_{mode}_mean"] = summary[f"{key}_mean"]
                if key.endswith("f1"):
                    score_stats = mean_ci(values, confidence=confidence)
                    summary[f"{key}_ci_low"] = score_stats["ci_low"]
                    summary[f"{key}_ci_high"] = score_stats["ci_high"]
                    summary[f"{key}_samples"] = values
            mode_summaries[mode] = summary
        row["metrics_by_singleton_mode"] = mode_summaries
        out.append(row)
    return out


@dataclass
class ResolutionF1Result:
    """Full experiment payload (serializable)."""

    runs: list[dict[str, Any]] = field(default_factory=list)
    aggregated: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_covers: bool = False) -> dict[str, Any]:
        """Serialize. Summary JSON omits covers by default (they live in runs/)."""
        runs_out: list[dict[str, Any]] = []
        for r in self.runs:
            if include_covers:
                runs_out.append(r)
            else:
                slim = {k: v for k, v in r.items() if k not in ("cover",)}
                # Keep a pointer so covers are discoverable
                slim["cover_cached"] = bool(r.get("cover") is not None)
                runs_out.append(slim)
        return {
            "meta": self.meta,
            "runs": runs_out,
            "aggregated": self.aggregated,
        }


def run_resolution_f1_experiment(
    game: Game,
    gt: Sequence[Sequence[int]],
    *,
    resolutions: Sequence[float],
    seeds: Sequence[int],
    max_memberships: int | None = None,
    confidence: float = DEFAULT_CI_LEVEL,
    cache_dir: Path | str | None = None,
    resume: bool = True,
    force: bool = False,
    rescore_only: bool = False,
    singleton_mode: str = DEFAULT_SINGLETON_MODE,
    compute_omega: bool = False,
    omega_sample_size: int = DEFAULT_OMEGA_SAMPLE_SIZE,
    omega_seed: int = 0,
) -> ResolutionF1Result:
    """Sweep γ × seeds; optionally resume from / write to ``cache_dir/runs``.

    Parameters
    ----------
    cache_dir :
        Experiment output root. Per-cell files go to ``cache_dir/runs/``.
        If None, no disk cache (in-memory only; no resume).
    resume :
        Skip cells that already have a complete cached run (default True).
    force :
        Ignore cache and re-run every cell (overwrites run files).
    rescore_only :
        Do not call ``community_hedonic``; only load cached covers and
        recompute metrics vs ``gt``. Fails if a required cell is missing.
    """
    singleton_modes(singleton_mode)
    if omega_sample_size <= 0:
        raise ValueError("omega_sample_size must be positive")
    k = resolve_max_memberships(max_memberships, gt)
    resolutions = [float(r) for r in resolutions]
    seeds = [int(s) for s in seeds]
    n_res = len(resolutions)
    n_seeds = len(seeds)

    runs_dir: Path | None = None
    cached: dict[tuple[float, int], dict[str, Any]] = {}
    if cache_dir is not None:
        runs_dir = Path(cache_dir) / RUNS_SUBDIR
        runs_dir.mkdir(parents=True, exist_ok=True)
        if resume or rescore_only:
            cached = load_completed_runs(runs_dir)

    log("=" * 55)
    log("  Overlapping resolution × F1 (full multi-phase)")
    log("=" * 55)
    log(f"  n_vertices      : {game.vcount():,}")
    log(f"  n_edges         : {game.ecount():,}")
    log(f"  n_gt_gt1        : {count_gt_communities_gt1(gt)}")
    log(f"  max_memberships : {k}")
    log(f"  n_iterations    : {N_ITERATIONS}")
    log(f"  only_local_moving: {ONLY_LOCAL_MOVING}")
    log(f"  allow_isolation : {ALLOW_ISOLATION}")
    log(f"  resolutions     : {resolutions}")
    log(f"  seeds           : {seeds}")
    log(f"  CI level        : {confidence}")
    log(f"  cache_dir       : {cache_dir}")
    log(f"  resume          : {resume and not force}")
    log(f"  force           : {force}")
    log(f"  rescore_only    : {rescore_only}")
    log(f"  singleton_mode  : {singleton_mode}")
    log(f"  sampled Omega   : {compute_omega}")
    if compute_omega:
        log(f"  Omega samples   : {omega_sample_size:,} (seed={omega_seed})")
    log(f"  cached complete : {len(cached)}")
    log("=" * 55)

    runs: list[dict[str, Any]] = []
    n_skipped = 0
    n_ran = 0
    n_rescored = 0
    total = n_res * n_seeds
    done = 0
    t_all = time.time()

    for res in resolutions:
        for seed in seeds:
            done += 1
            key = (float(res), int(seed))
            path = (
                run_cache_path(runs_dir, res, seed) if runs_dir is not None else None
            )

            use_cache = (
                (not force)
                and (resume or rescore_only)
                and key in cached
                and is_run_complete(cached[key])
            )

            if rescore_only and not use_cache:
                raise FileNotFoundError(
                    f"rescore-only: missing cached run for γ={res} seed={seed}"
                    + (f" (expected {path})" if path else "")
                )

            if use_cache:
                rec = rescore_record(
                    cached[key],
                    gt,
                    n_vertices=game.vcount(),
                    compute_omega=compute_omega,
                    singleton_mode=singleton_mode,
                    omega_sample_size=omega_sample_size,
                    omega_seed=omega_seed,
                )
                rec["from_cache"] = True
                if path is not None:
                    # Persist updated metrics (cover unchanged)
                    save_run_file(rec, path)
                n_skipped += 1
                n_rescored += 1
                log(
                    f"\n[{done}/{total}] γ={float(res):.6g}  seed={seed}  "
                    f"SKIP (cache)  F1={rec['f1']:.4f}"
                )
            else:
                log(
                    f"\n[{done}/{total}] γ={float(res):.6g}  seed={seed}  K={k} …"
                )
                t0 = time.time()
                rec = run_one(
                    game,
                    gt,
                    resolution=float(res),
                    seed=int(seed),
                    max_memberships=k,
                    singleton_mode=singleton_mode,
                    compute_omega=compute_omega,
                    omega_sample_size=omega_sample_size,
                    omega_seed=omega_seed,
                )
                n_ran += 1
                if path is not None:
                    save_run_file(rec, path)
                    log(f"  cached → {path.name}")
                log(
                    f"  F1={rec['f1']:.4f}  Jaccard={rec['jaccard']:.4f}  "
                    f"Q={rec.get('quality')}  "
                    f"comms={rec['n_predicted_comms']}  "
                    f"wall={rec['wallclock_s']:.3f}s",
                    t0,
                )

            runs.append(rec)

    aggregated = aggregate_runs(runs, confidence=confidence)
    meta = {
        "n_iterations": N_ITERATIONS,
        "only_local_moving": ONLY_LOCAL_MOVING,
        "allow_isolation": ALLOW_ISOLATION,
        "max_memberships": k,
        "max_memberships_rule": "n_gt_communities_size_gt_1",
        "n_gt_communities_raw": len(gt),
        "n_gt_communities_gt1": count_gt_communities_gt1(gt),
        "resolutions": resolutions,
        "seeds": seeds,
        "confidence": float(confidence),
        "n_runs": len(runs),
        "n_vertices": int(game.vcount()),
        "n_edges": int(game.ecount()),
        "wallclock_total_s": float(time.time() - t_all),
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "runs_dir": str(runs_dir) if runs_dir is not None else None,
        "resume": bool(resume and not force),
        "force": bool(force),
        "rescore_only": bool(rescore_only),
        "singleton_mode": singleton_mode,
        "singleton_modes_reported": list(singleton_modes(singleton_mode)),
        "omega_enabled": bool(compute_omega),
        "omega_sample_size": int(omega_sample_size) if compute_omega else None,
        "omega_seed": int(omega_seed) if compute_omega else None,
        "n_ran": n_ran,
        "n_skipped_cache": n_skipped,
        "n_rescored": n_rescored,
        "covers_cached": True,
    }
    log(
        f"\n[done] {len(runs)} cells  ran={n_ran}  "
        f"cache_hits={n_skipped}  in {meta['wallclock_total_s']:.1f}s"
    )
    for row in aggregated:
        log(
            f"  γ={row['resolution']:.4g}  "
            f"F1={row['f1_mean']:.4f}  "
            f"CI95=[{row['f1_ci_low']:.4f}, {row['f1_ci_high']:.4f}]  "
            f"n={row['n_seeds']}"
        )
    return ResolutionF1Result(runs=runs, aggregated=aggregated, meta=meta)


# ---------------------------------------------------------------------------
# Smoke graph (no DBLP)
# ---------------------------------------------------------------------------


def build_smoke_instance(
    *,
    n_blocks: int = 3,
    block_size: int = 8,
    p_in: float = 0.45,
    p_out: float = 0.05,
    seed: int = 0,
) -> tuple[Game, list[list[int]]]:
    """Tiny SBM + planted block GT for CI / smoke (no DBLP)."""
    block_sizes = [block_size] * n_blocks
    pref = [
        [p_in if i == j else p_out for j in range(n_blocks)]
        for i in range(n_blocks)
    ]
    g = ig.Graph.SBM(pref, block_sizes, directed=False)
    g.simplify()
    gt: list[list[int]] = []
    start = 0
    for sz in block_sizes:
        gt.append(list(range(start, start + sz)))
        start += sz
    # One singleton GT community (must be ignored for K)
    gt.append([0])
    return Game(g), gt


# ---------------------------------------------------------------------------
# I/O + plot
# ---------------------------------------------------------------------------


def save_results(
    result: ResolutionF1Result,
    path: Path | str,
    *,
    include_covers: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, result.to_dict(include_covers=include_covers))
    return path


def load_results(path: Path | str) -> ResolutionF1Result:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ResolutionF1Result(
        runs=list(data.get("runs", [])),
        aggregated=list(data.get("aggregated", [])),
        meta=dict(data.get("meta", {})),
    )


def plot_resolution_f1(
    result: ResolutionF1Result | dict[str, Any] | Sequence[dict[str, Any]],
    output_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Line plot: resolution (x) vs F1 (y) with CI error bars / band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(result, ResolutionF1Result):
        rows = result.aggregated
        meta = result.meta
    elif isinstance(result, dict) and "aggregated" in result:
        rows = result["aggregated"]
        meta = result.get("meta", {})
    else:
        rows = list(result)
        meta = {}

    rows = sorted(rows, key=lambda r: float(r["resolution"]))
    xs = [float(r["resolution"]) for r in rows]
    ys = [float(r["f1_mean"]) for r in rows]
    yerr_lo = [
        max(0.0, ys[i] - float(rows[i]["f1_ci_low"])) for i in range(len(rows))
    ]
    yerr_hi = [
        max(0.0, float(rows[i]["f1_ci_high"]) - ys[i]) for i in range(len(rows))
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xs, ys, "o-", color="C0", linewidth=2, markersize=6, label="F1 mean")
    ax.fill_between(
        xs,
        [ys[i] - yerr_lo[i] for i in range(len(xs))],
        [ys[i] + yerr_hi[i] for i in range(len(xs))],
        color="C0",
        alpha=0.2,
        label="CI band",
    )
    ax.errorbar(
        xs,
        ys,
        yerr=[yerr_lo, yerr_hi],
        fmt="none",
        ecolor="C0",
        elinewidth=1.2,
        capsize=3,
    )
    ax.set_xlabel("Resolution γ")
    ax.set_ylabel("F1 (vs ground truth)")
    conf = meta.get("confidence", DEFAULT_CI_LEVEL)
    k = meta.get("max_memberships", "?")
    ax.set_title(
        title
        or (
            f"Overlapping multi-phase hedonic: F1 vs γ "
            f"(K={k}, {conf:.0%} CI over seeds)"
        )
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hedonic-exp overlapping-resolution",
        description=(
            "Full-DBLP (or --smoke) F1 vs resolution for multi-phase "
            "community_hedonic: n_iterations=-1, only_local_moving=False, "
            "allow_isolation=True, max_memberships=#GT communities with "
            "size>1. Reports legacy symmetric best-match F1 plus one-to-one, "
            "node-membership, size-weighted, and diagnostic metrics. "
            "Singleton handling is selectable with --singleton-mode; sampled "
            "Omega is opt-in. Multi-seed F1 confidence intervals + line plot. "
            "Per-(γ,seed) covers are cached under <output_dir>/runs/ for "
            "resume and later re-scoring without re-running detection. "
            "Paths may come from a TOML config (default: configs/hedonic.toml)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "TOML config path (paths + optional [overlapping_resolution]). "
            "Also: HEDONIC_CONFIG or configs/hedonic.toml (cwd search)"
        ),
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="DBLP root (pkl/raw); default from config/env/DBLP_DIR",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory for results JSON, plot, and runs/ cache; "
            "default from config/env or local folder"
        ),
    )
    parser.add_argument(
        "--resolutions",
        default=None,
        help=(
            "Resolution grid: start:stop:n (linspace) or comma list. "
            f"Default {DEFAULT_RESOLUTIONS} (γ from 0 to 1 inclusive)."
        ),
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help=f"Seed list: a-b or comma list (default {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--max_memberships",
        type=int,
        default=None,
        help=(
            "Cap on communities per vertex. Default: number of ground-truth "
            "communities with more than one node."
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Two-sided CI level for F1 over seeds (e.g. 0.95)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny synthetic SBM only (no DBLP); CI-friendly",
    )
    parser.add_argument(
        "--smoke-blocks",
        type=int,
        default=3,
        help="Planted blocks for --smoke SBM",
    )
    parser.add_argument(
        "--smoke-block-size",
        type=int,
        default=8,
        help="Nodes per block for --smoke SBM",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip completed runs in <output_dir>/runs/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all cells and overwrite cached run files",
    )
    parser.add_argument(
        "--rescore-only",
        action="store_true",
        help=(
            "Only recompute metrics from cached covers under runs/ "
            "(no community_hedonic). Fails if any cell is missing."
        ),
    )
    parser.add_argument(
        "--singleton-mode",
        choices=("all", "size_ge_2", "both"),
        default=None,
        help=(
            "Apply singleton handling consistently to predicted and GT covers. "
            "'both' stores both evaluations; legacy f1 remains unchanged."
        ),
    )
    parser.add_argument(
        "--omega",
        action="store_true",
        help=(
            "Compute the scalable sampled pairwise Omega approximation while "
            "scoring cached covers (disabled by default)"
        ),
    )
    parser.add_argument(
        "--omega-sample-size",
        type=int,
        default=DEFAULT_OMEGA_SAMPLE_SIZE,
        help="Number of uniformly sampled vertex pairs for --omega",
    )
    parser.add_argument(
        "--omega-seed",
        type=int,
        default=0,
        help="RNG seed for reproducible sampled Omega",
    )
    parser.add_argument(
        "--include-covers-in-summary",
        action="store_true",
        help="Embed full covers in resolution_f1.json (default: only in runs/)",
    )
    parser.add_argument(
        "--plot-format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Figure format",
    )
    parser.add_argument(
        "--results-name",
        default="resolution_f1.json",
        help="Results filename under --output_dir",
    )
    parser.add_argument(
        "--plot-name",
        default=None,
        help="Plot filename (default: resolution_f1.<format>)",
    )
    args = parser.parse_args(argv)

    # Resolve paths: CLI > env > TOML section > defaults
    try:
        resolved = exp_config.resolve_experiment_paths(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            experiment_section="overlapping_resolution",
            search_cwd=True,
        )
    except FileNotFoundError as exc:
        log(f"[error] {exc}")
        return 2

    section = resolved["section"]
    data_dir = resolved["data_dir"]
    # If user did not pass --output_dir and TOML/env only set global OUTPUT_DIR
    # to the shared experiments root, fall back to local default folder name
    # unless the section explicitly set output_dir.
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    elif section.get("output_dir"):
        out_dir = Path(str(section["output_dir"])).expanduser()
    elif args.smoke:
        out_dir = DEFAULT_OUTPUT_DIR
    else:
        # Prefer HEDONIC_OUTPUT_DIR / [paths].output_dir / default folder
        out_dir = Path(resolved["output_dir"])
        if out_dir == OUTPUT_DIR and str(OUTPUT_DIR) == str(
            exp_config.DEFAULT_OUTPUT_DIR
        ):
            # Keep a dedicated subfolder under the global artifacts root
            out_dir = OUTPUT_DIR / "overlapping_resolution_f1"
        elif out_dir == exp_config.DEFAULT_OUTPUT_DIR:
            out_dir = DEFAULT_OUTPUT_DIR

    res_spec = (
        args.resolutions
        if args.resolutions is not None
        else section.get("resolutions") or DEFAULT_RESOLUTIONS
    )
    seeds_spec = (
        args.seeds if args.seeds is not None else section.get("seeds") or DEFAULT_SEEDS
    )
    confidence = (
        float(args.confidence)
        if args.confidence is not None
        else float(section.get("confidence", DEFAULT_CI_LEVEL))
    )

    try:
        resolutions = parse_resolutions(str(res_spec))
        seeds = parse_seeds(str(seeds_spec))
    except ValueError as exc:
        log(f"[error] {exc}")
        return 2

    if not (0.0 < confidence < 1.0):
        log("[error] --confidence must be in (0, 1)")
        return 2
    if args.omega_sample_size <= 0:
        log("[error] --omega-sample-size must be positive")
        return 2
    if args.rescore_only and args.force:
        log("[error] --rescore-only cannot be combined with --force")
        return 2

    singleton_mode = str(
        args.singleton_mode
        if args.singleton_mode is not None
        else section.get("singleton_mode", DEFAULT_SINGLETON_MODE)
    )
    try:
        singleton_modes(singleton_mode)
    except ValueError as exc:
        log(f"[error] {exc}")
        return 2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        if args.resolutions is None and str(res_spec) == DEFAULT_RESOLUTIONS:
            resolutions = parse_resolutions("0,0.5,1")
        if args.seeds is None and str(seeds_spec) == DEFAULT_SEEDS:
            seeds = parse_seeds("0,1")
        log(
            f"[smoke] synthetic SBM  blocks={args.smoke_blocks}  "
            f"block_size={args.smoke_block_size}"
        )
        game, gt = build_smoke_instance(
            n_blocks=args.smoke_blocks,
            block_size=args.smoke_block_size,
            seed=0,
        )
    else:
        if args.rescore_only:
            # Still need GT for metrics; load graph+GT (detection skipped)
            g, gt, _node_map = load_dblp(data_dir)
            game = Game(g)
        else:
            g, gt, _node_map = load_dblp(data_dir)
            game = Game(g)

    if resolved["config_path"]:
        log(f"[config] {resolved['config_path']}")

    result = run_resolution_f1_experiment(
        game,
        gt,
        resolutions=resolutions,
        seeds=seeds,
        max_memberships=args.max_memberships,
        confidence=confidence,
        cache_dir=out_dir,
        resume=not args.no_resume,
        force=bool(args.force),
        rescore_only=bool(args.rescore_only),
        singleton_mode=singleton_mode,
        compute_omega=bool(args.omega),
        omega_sample_size=int(args.omega_sample_size),
        omega_seed=int(args.omega_seed),
    )
    result.meta["smoke"] = bool(args.smoke)
    result.meta["data_dir"] = None if args.smoke else str(data_dir)
    result.meta["output_dir"] = str(out_dir)
    result.meta["resolutions_spec"] = str(res_spec)
    result.meta["seeds_spec"] = str(seeds_spec)
    result.meta["config_path"] = (
        str(resolved["config_path"]) if resolved["config_path"] else None
    )

    results_path = out_dir / args.results_name
    save_results(
        result,
        results_path,
        include_covers=bool(args.include_covers_in_summary),
    )
    log(f"\n[done] Results → {results_path}")
    log(f"[done] Run cache → {out_dir / RUNS_SUBDIR}")

    plot_name = args.plot_name or f"resolution_f1.{args.plot_format}"
    plot_path = out_dir / plot_name
    plot_resolution_f1(result, plot_path)
    log(f"[done] Plot    → {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
