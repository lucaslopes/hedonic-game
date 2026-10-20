"""Shared helpers of the ground-truth-seeded studies: the detector worker and the artifact store.

The ground-truth robustness spectrum (:mod:`gt_spectrum`) and the start-quality curve (:mod:`start_quality`)
start ``Game.community_hedonic`` from a ground-truth cover and keep its exact labelled return. This module holds
what they share: the detector worker (a top-level function, so a spawned worker can pickle it), the validation and
label compaction of returned memberships, the content-addressed cover and membership artifacts, the resume check of
a cached return, the small parsers, and the output lock.

The locked v3 experiment of the 1.x series (its CLI, protocol lock and figures) is not part of hedonic 2.0.
"""

from __future__ import annotations

import csv
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from hedonic import Game
from hedonic.experiments.overlapping.robustness import (
    canonicalize_cover,
    cover_hash,
    vertex_memberships_to_cover,
)


PHASES = ("local",)
"""Detector phases hedonic 2.0 runs. The 1.x overlapping multilevel phase is not available."""


def require_local_phase(phase: str) -> str:
    """Return ``phase`` when it is local moving; refuse anything else with a clear error."""
    if str(phase) != "local":
        raise ValueError(
            f"phase {phase!r} is not available in hedonic 2.0: community_hedonic runs hedonic local moving "
            "only (the overlapping multilevel phase was removed); use phase = \"local\""
        )
    return "local"


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None



def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_gzip_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"), sort_keys=True)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _read_gzip_json(path: Path) -> list[list[int]] | None:
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, list) else None


def _parse_csv(value: str | None, cast, *, default: Iterable[Any] = ()) -> list[Any]:
    if value is None:
        return list(default)
    result: list[Any] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if raw:
            result.append(cast(raw))
    if not result:
        raise ValueError("expected at least one comma-separated value")
    return result


def _parse_range(value: str | None, *, default: Iterable[int] = ()) -> list[int]:
    if value is None:
        return list(default)
    result: list[int] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "-" in raw and raw.count("-") == 1:
            first, last = raw.split("-", 1)
            start, stop = int(first), int(last)
            step = 1 if stop >= start else -1
            result.extend(range(start, stop + step, step))
        else:
            result.append(int(raw))
    if not result:
        raise ValueError("expected at least one integer seed")
    return result


def _parse_profile(value: str) -> list[float]:
    """Parse ``start:stop:count`` or comma-separated resolution values."""
    if ":" in value:
        fields = value.split(":")
        if len(fields) != 3:
            raise ValueError("profile must be start:stop:count")
        start, stop, count = float(fields[0]), float(fields[1]), int(fields[2])
        if count < 2:
            return [start]
        step = (stop - start) / (count - 1)
        return [start + index * step for index in range(count)]
    return _parse_csv(value, float)


def _cover_artifact_path(output_dir: Path, digest: str) -> Path:
    return output_dir / "covers" / f"{digest}.json.gz"


def _persist_cover(output_dir: Path, cover: list[list[int]]) -> str:
    canonical = canonicalize_cover(cover)
    digest = cover_hash(canonical)
    path = _cover_artifact_path(output_dir, digest)
    if _load_cover_artifact(output_dir, digest) != canonical:
        _atomic_gzip_json(path, canonical)
    return digest


def _load_cover_artifact(output_dir: Path, digest: str) -> list[list[int]] | None:
    cover = _read_gzip_json(_cover_artifact_path(output_dir, digest))
    if cover is None:
        return None
    try:
        return cover if cover_hash(cover) == digest else None
    except (TypeError, ValueError):
        return None


def _membership_artifact_path(output_dir: Path, digest: str) -> Path:
    return output_dir / "raw_memberships" / f"{digest}.json.gz"


def _raw_cover_hash(cover: list[list[int]], n_vertices: int | None = None) -> str:
    """Hash a cover while preserving duplicate communities as a multiset."""
    communities: list[tuple[int, ...]] = []
    for community in cover:
        members = tuple(sorted({int(member) for member in community}))
        if n_vertices is not None and any(
            member < 0 or member >= n_vertices for member in members
        ):
            raise ValueError("cover contains a vertex outside the graph")
        if members:
            communities.append(members)
    payload = [list(members) for members in sorted(communities)]
    return _json_hash(payload)


def _persist_memberships(
    output_dir: Path,
    memberships: list[list[int]],
    n_vertices: int,
) -> str:
    """Persist raw native memberships without globally merging communities."""
    raw_cover = vertex_memberships_to_cover(memberships)
    digest = _raw_cover_hash(raw_cover, n_vertices)
    path = _membership_artifact_path(output_dir, digest)
    if _load_membership_artifact(output_dir, digest) is None:
        _atomic_gzip_json(path, memberships)
    return digest


def _load_membership_artifact(
    output_dir: Path,
    digest: str,
) -> list[list[int]] | None:
    memberships = _read_gzip_json(_membership_artifact_path(output_dir, digest))
    if memberships is None:
        return None
    try:
        raw_cover = vertex_memberships_to_cover(memberships)
        actual = _raw_cover_hash(raw_cover, len(memberships))
    except (TypeError, ValueError):
        return None
    return memberships if actual == digest else None


def _normalize_raw_memberships(
    memberships: Any,
    n_vertices: int,
    cap: int,
) -> list[list[int]]:
    """Validate raw native rows and compact labels without merging communities."""
    if not isinstance(memberships, list) or len(memberships) != n_vertices:
        raise ValueError("raw memberships must contain one row per graph vertex")
    label_map: dict[int, int] = {}
    normalized: list[list[int]] = []
    for labels in memberships:
        if not isinstance(labels, (list, tuple)) or not labels:
            raise ValueError("every raw membership row must be non-empty")
        unique = sorted({int(label) for label in labels})
        if unique[0] < 0:
            raise ValueError("raw membership labels must be non-negative")
        if len(unique) > cap:
            raise ValueError("detector returned a cover above the configured cap")
        row: list[int] = []
        for label in unique:
            if label not in label_map:
                label_map[label] = len(label_map)
            row.append(label_map[label])
        normalized.append(row)
    if not label_map:
        raise ValueError("raw memberships contain no community labels")
    return normalized


def _resume_artifacts_match(
    output_dir: Path,
    record: dict[str, Any],
    *,
    n_vertices: int,
    cap: int,
) -> bool:
    """Verify every exact-state artifact before accepting a cached return."""
    final_membership_hash = record.get("final_membership_hash")
    pre_cleanup_hash = record.get("pre_cleanup_membership_hash")
    final_cover_hash = record.get("final_cover_hash")
    if not all(
        isinstance(value, str) and value
        for value in (final_membership_hash, pre_cleanup_hash, final_cover_hash)
    ):
        return False
    final_memberships = _load_membership_artifact(
        output_dir, str(final_membership_hash)
    )
    pre_cleanup_memberships = _load_membership_artifact(
        output_dir, str(pre_cleanup_hash)
    )
    final_cover = _load_cover_artifact(output_dir, str(final_cover_hash))
    if (
        final_memberships is None
        or pre_cleanup_memberships is None
        or final_cover is None
    ):
        return False
    try:
        normalized = _normalize_raw_memberships(final_memberships, n_vertices, cap)
        _normalize_raw_memberships(pre_cleanup_memberships, n_vertices, cap)
        projected = canonicalize_cover(
            vertex_memberships_to_cover(normalized), n_vertices
        )
    except (TypeError, ValueError):
        return False
    return (
        projected == final_cover
        and cover_hash(projected, n_vertices) == final_cover_hash
        and record.get("raw_membership_hash") == final_membership_hash
    )


def _detector_worker(
    graph,
    initial_membership: list[list[int]],
    max_memberships: int,
    gamma: float,
    phase: str,
    allow_isolation: bool,
    seed: int,
) -> dict[str, Any]:
    """Run local moving from ``initial_membership`` and return the exact labelled memberships.

    Top-level so a spawned worker can pickle it. hedonic 2.0 has no cleanup pass after local moving, so the raw
    (pre-cleanup) and final memberships are the same labelled rows (``game.memberships``).
    """
    require_local_phase(phase)
    game = Game(graph)
    game.community_hedonic(
        initial_membership,
        max_memberships=int(max_memberships),
        resolution=float(gamma),
        allow_isolation=bool(allow_isolation),
        seed=int(seed),
    )
    memberships = [[int(label) for label in labels] for labels in (game.memberships or [])]
    return {"raw_memberships": memberships, "final_memberships": [list(row) for row in memberships]}


def _condition_path(output_dir: Path, job_key: str, condition_key: str) -> Path:
    return output_dir / "runs" / job_key / f"{condition_key}.json"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _initialization_distance(
    initial_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
    transition_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    transition_metrics = transition_metrics or {}
    return {
        "initial_node_micro_f1": initial_metrics.get("node_micro_f1"),
        "final_node_micro_f1": final_metrics.get("node_micro_f1"),
        "distance_to_ground_truth": (
            None
            if final_metrics.get("node_micro_f1") is None
            else 1.0 - float(final_metrics["node_micro_f1"])
        ),
        "initial_to_final_node_micro_f1": transition_metrics.get("node_micro_f1"),
        "initial_to_final_distance": (
            None
            if transition_metrics.get("node_micro_f1") is None
            else 1.0 - float(transition_metrics["node_micro_f1"])
        ),
        "membership_count_delta": (
            final_metrics.get("predicted_average_memberships_per_vertex", 0.0)
            - initial_metrics.get("predicted_average_memberships_per_vertex", 0.0)
        ),
    }


def _cover_change_diagnostics(
    initial_memberships: list[list[int]],
    final_memberships: list[list[int]],
) -> dict[str, Any]:
    """Summarize how a returned cover moved from its warm start."""
    if len(initial_memberships) != len(final_memberships):
        raise ValueError("initial and final memberships have different lengths")
    initial_sets = [set(labels) for labels in initial_memberships]
    final_sets = [set(labels) for labels in final_memberships]
    changed = [first != second for first, second in zip(initial_sets, final_sets)]
    added = sum(len(second - first) for first, second in zip(initial_sets, final_sets))
    removed = sum(len(first - second) for first, second in zip(initial_sets, final_sets))
    return {
        "changed_vertex_fraction": sum(changed) / len(changed) if changed else 0.0,
        "changed_vertex_count": int(sum(changed)),
        "memberships_added": int(added),
        "memberships_removed": int(removed),
        "initial_membership_count": int(sum(map(len, initial_sets))),
        "final_membership_count": int(sum(map(len, final_sets))),
    }


def _assert_output_safe(output_dir: Path, data_root: str | Path) -> None:
    """Keep experiment artifacts out of the raw network data."""
    output = Path(output_dir).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    if output == data or data in output.parents:
        raise ValueError(
            f"refusing to write experiment artifacts inside raw network data: {output}"
        )


class _OutputLock:
    def __init__(self, stream) -> None:
        self.stream = stream

    def close(self) -> None:
        if self.stream.closed:
            return
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()

    def __del__(self) -> None:
        self.close()


def _acquire_output_lock(output_dir: Path) -> _OutputLock:
    """Hold a nonblocking process lock for the complete output mutation."""
    path = output_dir / ".ground_truth_robustness.lock"
    stream = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RuntimeError(
            f"another ground-truth robustness process owns {output_dir}"
        ) from exc
    stream.seek(0)
    stream.truncate()
    stream.write(f"pid={os.getpid()}\n")
    stream.flush()
    return _OutputLock(stream)

