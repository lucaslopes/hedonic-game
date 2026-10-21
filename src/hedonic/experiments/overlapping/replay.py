"""Do a run's Hedonic records reproduce with the loaded core? (``hedonic run check --replay N``)

Records name the core by its results class (``runtime.CORE_CLASSES``): another core with the same three fingerprints
keeps every record of the class, whatever its source. The fingerprints are small grids; this check replays the run's
own work on the loaded core, as the disjoint store does before it adopts rows of another source
(``store.check_adoption``):

* **Hedonic units** of a benchmark (``runs/<net>/hedonic_local.json`` with the answer ``--save-covers`` keeps:
  ``cover_store``). The analysis graph is loaded as the audit loads it and checked against the record
  (``dataset_report.graph_sha256``, else n and m); the detector runs again through the runner's own
  ``codeseg_reproduction._run_method`` at the recorded parameters, cap and seed; the cover must equal the saved cover
  and the labelled rows the saved labels, bit for bit.
* **Cover audits** (``cover_audit.jsonl`` ledgers, as ``hedonic run audit`` writes them). A record keeps aggregates over
  every vertex, so a sample cannot be compared with it; instead a sample of the audited cover's vertices (the one of
  largest degree among them) is audited again with the loaded core and with the pure-Python reference
  (``stability.prepare(engine="python")``), whose intervals every core of the class returns bit for bit, and their
  interval ends must agree bit for bit. The cover on disk must be the one audited (its file's SHA-256).

The cheapest items are taken first (smallest graph, then fastest detector run). A replay that passes is appended to
``<root>/adopted.json`` (``adopted``: when, what was replayed, the sources that wrote it and the loaded core's source
digest, identity and fingerprints); nothing else is written, and ``--dry-run`` only lists what would be replayed.
On SNAP graphs a replay loads each graph once, which takes seconds to minutes per pair: it is a check to run on
purpose after a change of core, never on every resume.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Iterable

RUNNER = "hedonic_local"
MANIFEST_NAME = "adopted.json"
SCHEMA = 1
DEFAULT_VERTICES = 32

GraphLoader = Callable[[str, str, str], Any]  # (network, cover variant, induce) -> igraph.Graph


class Skip(Exception):
    """An item that cannot be replayed here (said why); it neither passes nor fails."""


# --------------------------------------------------------------------------- what a root holds
def find_units(root: Path) -> list[dict[str, Any]]:
    """The completed Hedonic benchmark units under ``root`` (any depth), cheapest first."""
    units = []
    for path in sorted(Path(root).rglob(f"{RUNNER}.json")):
        if path.parent.parent.name != "runs":
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict) or record.get("status") != "completed" or record.get("method") != RUNNER:
            continue
        report = record.get("dataset_report") or {}
        units.append({"kind": "unit", "path": path, "unit_dir": path.parents[2], "record": record,
                      "key": str(path.relative_to(root)), "n": int(report.get("n") or 0),
                      "seconds": float(record.get("detection_seconds") or 0.0), "source": record_source(record)})
    return sorted(units, key=lambda u: (u["n"], u["seconds"], u["key"]))


def find_audits(root: Path) -> list[dict[str, Any]]:
    """The completed cover audits of every ``cover_audit.jsonl`` ledger under ``root``, cheapest first."""
    from hedonic.experiments.overlapping import cover_audit as ca

    audits = []
    for ledger in sorted(Path(root).rglob(ca.RESULTS_NAME)):
        for record in ca.read_records(ledger).values():
            if record.get("status") != "completed":
                continue
            audits.append({"kind": "audit", "ledger": ledger, "record": record,
                           "key": f"{ledger.parent.relative_to(root)}::{record['key']}",
                           "n": int(record.get("n_vertices") or 0), "seconds": float(record.get("audit_seconds") or 0.0),
                           "source": None})
    return sorted(audits, key=lambda a: (a["n"], a["seconds"], a["key"]))


def record_source(record: dict[str, Any]) -> str | None:
    """The source digest of the core that wrote a record (its real one where the record kept it)."""
    versions = record.get("runtime_versions") or {}
    return versions.get("hedonic_core_built_source_sha256") or versions.get("hedonic_core_source_sha256")


# --------------------------------------------------------------------------- one item
def _load(loader: GraphLoader, graphs: dict, network: str, variant: str, induce: str, report: dict[str, Any]):
    from hedonic.experiments.overlapping import cover_store

    key = (network, variant, induce)
    if key not in graphs:
        graphs.clear()  # one graph at a time: a covered LiveJournal is GBs
        graph = loader(network, variant, induce)
        graphs[key] = (graph, cover_store.graph_sha256(graph))
    graph, sha = graphs[key]
    if report.get("graph_sha256") and report["graph_sha256"] != sha:
        raise Skip(f"the loaded {network} graph is not the one the record ran on (SHA-256 {sha[:12]}..., the record's "
                   f"{report['graph_sha256'][:12]}...)")
    if report.get("n") is not None and (report.get("n"), report.get("m")) != (graph.vcount(), graph.ecount()):
        raise Skip(f"the loaded {network} graph has n={graph.vcount()}, m={graph.ecount()}; the record's has "
                   f"n={report.get('n')}, m={report.get('m')}")
    return graph


def _induce(unit_dir: Path) -> str:
    return "covered" if unit_dir.parent.name.endswith("-covered") else "full"


def _read_cover(path: Path) -> list[list[int]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def replay_unit(item: dict[str, Any], loader: GraphLoader, graphs: dict) -> dict[str, Any]:
    """Run a Hedonic unit's detector call again and compare its answer with the saved one."""
    from hedonic.experiments import timing
    from hedonic.experiments.overlapping import codeseg_reproduction as cr
    from hedonic.experiments.overlapping import cover_store

    record, unit_dir = item["record"], item["unit_dir"]
    detector = record.get("detector") or {}
    saved_cover = unit_dir / record["cover_path"] if record.get("cover_path") else None
    labels = cover_store.read_labels(record, unit_dir) if cover_store.has_labels(record) else None
    if (saved_cover is None or not saved_cover.is_file()) and labels is None:
        raise Skip("the unit kept no answer (it ran without --save-covers)")
    if detector.get("max_memberships") is None or detector.get("parameters") is None:
        raise Skip("the record does not name the detector's cap and parameters")
    report = record.get("dataset_report") or {}
    graph = _load(loader, graphs, str(record.get("dataset")), str(report.get("cover_variant") or "all"),
                  _induce(unit_dir), report)
    parameters = {key: value for key, value in detector["parameters"].items() if not str(key).startswith("_")}
    parameters["max_memberships"] = int(detector["max_memberships"])  # the cap it played (an unlimited cap is n)
    seed = int(detector.get("seed", record.get("seed", 0)))
    with timing.detector_run():
        cover, metadata = cr._run_method(RUNNER, graph, [], seed, parameters)
    differences = []
    if saved_cover is not None and saved_cover.is_file():
        if [sorted(int(v) for v in community) for community in cover] != _read_cover(saved_cover):
            differences.append("cover")
    if labels is not None and metadata.get("final_memberships") != labels:
        differences.append("labelled rows")
    if detector.get("effective_resolution") is not None and \
            float(metadata["effective_resolution"]).hex() != float(detector["effective_resolution"]).hex():
        differences.append("resolution")
    return {"key": item["key"], "kind": "unit", "status": "differs" if differences else "same",
            "differences": differences, "source": item["source"], "n": graph.vcount(),
            "compared": [what for what, kept in (("cover", saved_cover is not None and saved_cover.is_file()),
                                                 ("labelled rows", labels is not None)) if kept]}


def _audit_sample(graph, n_vertices: int, count: int, key: str) -> list[int]:
    rng = random.Random(key)
    picked = set(rng.sample(range(n_vertices), min(count, n_vertices)))
    if picked and n_vertices:
        degrees = graph.degree()
        picked.discard(min(picked))
        picked.add(max(range(n_vertices), key=degrees.__getitem__))
    return sorted(picked)


def replay_audit(item: dict[str, Any], loader: GraphLoader, graphs: dict, vertices: int) -> dict[str, Any]:
    """Audit a sample of an audited cover's vertices again, with the loaded core and the Python reference."""
    import numpy as np

    from hedonic.experiments.overlapping import cover_audit as ca
    from hedonic.experiments.overlapping import cover_store
    from hedonic.experiments.overlapping import stability as st

    record = item["record"]
    path = Path(record.get("source") or "") / str(record.get("record_file") or "")
    if not record.get("record_file") or not path.is_file():
        raise Skip(f"the benchmark record it audited is not on disk ({path})")
    unit = json.loads(path.read_text(encoding="utf-8"))
    unit_dir = path.parents[2]
    report = {**(unit.get("dataset_report") or {}), "graph_sha256": record.get("graph_sha256")}
    network = ca.qs.split_target(record["network"])[0]  # NETWORK[:COVER] -> the dataset it loads
    graph = _load(loader, graphs, network, str(record.get("cover_variant") or "all"), _induce(unit_dir), report)
    n = graph.vcount()
    if record["profile"] == "cover":
        file = unit_dir / str(unit.get("cover_path") or "")
        if not unit.get("cover_path") or not file.is_file():
            raise Skip("the audited cover is not on disk")
        if record.get("cover_file_sha256") and hashlib.sha256(file.read_bytes()).hexdigest() != record["cover_file_sha256"]:
            raise Skip("the cover on disk is not the one audited (its unit ran again)")
        memberships, _ = ca.cover_profile(ca.read_cover(file), n)
    else:
        labels_file = cover_store.labels_path(unit)
        if labels_file and record.get("labels_file_sha256") and \
                hashlib.sha256((unit_dir / labels_file).read_bytes()).hexdigest() != record["labels_file_sha256"]:
            raise Skip("the labelled rows on disk are not the ones audited (their unit ran again)")
        rows = cover_store.read_labels(unit, unit_dir)
        if not rows:
            raise Skip("the unit keeps no labelled rows")
        memberships, _ = ca.labelled_profile(rows, n)
    sample = np.asarray(_audit_sample(graph, n, vertices, record["key"]), dtype=np.int64)
    core, reference = st.prepare(graph, memberships, engine="c"), st.prepare(graph, memberships, engine="python")
    differences = []
    for policy, allow in ca.POLICIES:
        kwargs = dict(atol=float(record["atol"]), rtol=float(record["rtol"]))
        mine = core.intervals(sample, int(record["cap"]), allow, **kwargs)
        theirs = reference.intervals(sample, int(record["cap"]), allow, **kwargs)
        bits = [[(float(lo).hex(), float(hi).hex()) for lo, hi in row] for row in mine]
        if bits != [[(float(lo).hex(), float(hi).hex()) for lo, hi in row] for row in theirs]:
            differences.append(policy)
    return {"key": item["key"], "kind": "audit", "status": "differs" if differences else "same",
            "differences": differences, "source": item["source"], "n": n, "vertices": len(sample)}


# --------------------------------------------------------------------------- the check
def core_identity() -> dict[str, Any]:
    """The loaded core as an adoption names it."""
    from hedonic.experiments import runtime

    return {"hedonic": runtime.hedonic_version(), "hedonic_core_sha256": runtime.core_sha256(),
            "hedonic_core_built_source_sha256": runtime.core_built_source_sha256(),
            "hedonic_core_source_sha256": runtime.core_source_sha256(),
            "hedonic_core_fingerprints": list(runtime.core_fingerprints())}


def read_manifest(root: Path) -> dict[str, Any]:
    try:
        data = json.loads((Path(root) / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": SCHEMA, "adopted": []}
    return data if isinstance(data, dict) else {"schema": SCHEMA, "adopted": []}


def record_adoption(root: Path, results: list[dict[str, Any]]) -> Path:
    """Append a passed replay to ``<root>/adopted.json`` (written atomically; earlier entries are kept)."""
    manifest = read_manifest(root)
    sources: dict[str, int] = {}
    for result in results:
        if result["status"] == "same":
            sources[result.get("source") or "unknown"] = sources.get(result.get("source") or "unknown", 0) + 1
    manifest.setdefault("adopted", []).append({
        "checked": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "core": core_identity(),
        "sources": dict(sorted(sources.items())),
        "replayed": [{key: result[key] for key in ("kind", "key", "n", "vertices") if key in result}
                     for result in results if result["status"] == "same"],
    })
    manifest["schema"] = SCHEMA
    path = Path(root) / MANIFEST_NAME
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _default_loader(network_root: Path) -> GraphLoader:
    from hedonic.experiments.overlapping import cover_audit as ca

    return lambda network, variant, induce: ca.load_graph(network, variant, network_root, induce)


def check(root: str | Path, count: int, *, vertices: int = DEFAULT_VERTICES, network_root: str | Path | None = None,
          load_graph: GraphLoader | None = None, dry_run: bool = False, record: bool = True,
          emit: Callable[[str], Any] = print) -> int:
    """Replay ``count`` Hedonic units and ``count`` cover audits of ``root``. 0 when every replayed item reproduced
    (and at least one was replayed), 1 otherwise."""
    from hedonic.experiments.config import NETWORKS_DIR, expand_path

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        emit(f"  no run folder at {root}")
        return 1
    units, audits = find_units(root), find_audits(root)
    picked = units[:max(0, count)] + audits[:max(0, count)]
    emit(f"  replay   {root} · {len(units)} Hedonic units, {len(audits)} cover audits; replaying "
         f"{min(count, len(units))} + {min(count, len(audits))} (cheapest first)")
    if dry_run:
        for item in picked:
            emit(f"    {item['kind']:<5} {item['key']} · n={item['n']:,}"
                 + (f" · written by core {item['source'][:12]}..." if item.get("source") else ""))
        written = ", ".join(f"{source[:12]}... ({count})" if source != "unknown" else f"unknown ({count})"
                            for source, count in sources_of(units).items())
        emit(f"  units written by core sources: {written or 'none'}")
        emit("  dry run: nothing loaded, run or written")
        return 0
    loader = load_graph or _default_loader(expand_path(network_root or NETWORKS_DIR).resolve())
    graphs: dict = {}
    results = []
    for item in picked:
        try:
            result = (replay_unit(item, loader, graphs) if item["kind"] == "unit"
                      else replay_audit(item, loader, graphs, vertices))
        except Skip as why:
            result = {"key": item["key"], "kind": item["kind"], "status": "skipped", "reason": str(why)}
        results.append(result)
        mark = {"same": "✓", "differs": "✗", "skipped": "·"}[result["status"]]
        detail = {"same": "identical", "skipped": f"skipped: {result.get('reason')}",
                  "differs": f"DIFFERS: {', '.join(result.get('differences', []))}"}[result["status"]]
        emit(f"    {mark} {item['kind']:<5} {item['key']} · {detail}")
    same = sum(r["status"] == "same" for r in results)
    differs = sum(r["status"] == "differs" for r in results)
    emit(f"  replay   {same} identical, {differs} different, {len(results) - same - differs} skipped")
    if differs or not same:
        return 1
    if record:
        emit(f"  recorded in {record_adoption(root, results)}")
    return 0


def sources_of(items: Iterable[dict[str, Any]]) -> dict[str, int]:
    """How many of ``items`` each core source wrote (``unknown`` where a record did not keep it)."""
    out: dict[str, int] = {}
    for item in items:
        out[item.get("source") or "unknown"] = out.get(item.get("source") or "unknown", 0) + 1
    return out
