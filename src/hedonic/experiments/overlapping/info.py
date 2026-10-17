"""``hedonic show``: what can be run, what is on disk, and what the scores mean.

    hedonic show methods              Hedonic and the eight baselines: ready / can be installed / missing tools
    hedonic show networks [--remote]  the seven SNAP networks: size, on-disk copy, download size
    hedonic show metrics              the accuracy metrics reported by `hedonic run exp`
    hedonic show covers               graph/overlapping-cover pairs in the local SNAP cache (no download)
    hedonic show table                analyse the local pairs: a CSV of network and induced-cover statistics,
                                      plus LaTeX table rows and figure panels next to it
    hedonic show figure               regenerate only the figure panels from that CSV
    hedonic show disjoint             the disjoint presets, grids and estimates

These commands use local files; they do not download datasets or build runtimes (that is `hedonic setup`).
``--remote`` only sends HTTP HEAD requests to snap.stanford.edu for archive
sizes. Add ``--json`` for machine-readable output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from hedonic.experiments.config import NETWORKS_DIR, codeseg_cache_dir, expand_path
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping.tui import style

# Tools each method's runtime needs, checked with shutil.which (see codeseg_setup).
REQUIREMENTS = {
    "hedonic": [], "clique_percolation": ["python:networkx"],
    "codeseg": ["c++"],
    "fox": ["git", "cmake", "openmp-c++"],
    "bigclam": ["git", "make", "c++"],
    "neo_kmeans": ["make", "cc"],
    "ncgame": ["git"],
    "angel": ["uv"], "ego_splitting": ["uv"],
}
METRICS = [
    ("f1", "F1 sym.", "symmetric best-match F1",
     "Half the mean, over reference communities, of the best F1 with any detected community, plus half the same "
     "mean over detected communities. The headline score of CoDeSEG/QOCE; penalises both missed and spurious "
     "communities, but many small well-matched communities can score well."),
    ("matching_f1", "F1 match", "one-to-one matching F1",
     "Communities are paired one-to-one (maximum-weight matching on F1). Precision averages over all detected "
     "communities and recall over all reference communities, unmatched ones scoring zero; the score is their "
     "harmonic mean. Rewards getting the number of communities right."),
    ("node_micro_f1", "F1 micro", "vertex-membership micro F1",
     "After the one-to-one alignment, every (vertex, community) assignment is a label; micro F1 counts correct "
     "assignments over all vertices. Measures how many memberships are recovered."),
    ("size_weighted_community_f1", "F1 size-w.", "size-weighted best-match F1",
     "Symmetric best-match F1 in which each community's best match is weighted by its size, so large communities "
     "count more than small ones."),
    ("onmi", "ONMI", "LFK overlapping normalised mutual information",
     "Lancichinetti–Fortunato–Kertész overlapping NMI (the OvpNMI definition used by CoDeSEG), computed from "
     "positive community intersections over the union of covered vertices. 1 = identical covers."),
    ("omega", "Omega", "Omega index (sampled)",
     "Chance-corrected agreement on how many communities each vertex pair shares, estimated from 100,000 "
     "uniformly sampled pairs (seed fixed). Near 0 on sparse covers of large graphs, where almost no sampled pair "
     "shares a community; read it with care there."),
]
PROTOCOL = ("Scoring protocol (all metrics): the SNAP `all` cover, canonicalised and restricted to communities of at "
            "least 3 vertices; detected communities are restricted to vertices covered by the reference (the "
            "CoDeSEG paper filter). Higher is better; all lie in [0, 1] (Omega can be slightly negative).")


def _tool(name: str) -> bool:
    if name.startswith("python:"):
        return importlib.util.find_spec(name.split(":", 1)[1]) is not None
    if name == "openmp-c++":
        from hedonic.experiments.overlapping.codeseg_setup import _fox_openmp_compiler
        return _fox_openmp_compiler(None) is not None or sys.platform != "darwin" and shutil.which("g++") is not None
    if name == "c++":
        return qs._compiler() is not None
    if name == "cc":
        return any(shutil.which(c) for c in ("cc", "gcc", "clang"))
    return shutil.which(name) is not None


def method_status(tools: Path) -> list[dict]:
    """Each catalogue method's readiness, from the ``hedonic setup`` folder ``tools`` (``config.codeseg_cache_dir``)."""
    try:
        manifest = json.loads((tools / "setup_manifest.json").read_text()).get("methods", {})
    except (OSError, ValueError):
        manifest = {}
    rows = []
    for m in qs.METHODS:
        missing = [t for t in REQUIREMENTS[m.key] if not _tool(t)]
        if m.key == "hedonic":
            status, detail = "ready", "built in (the hedonic C core)"
        elif m.key == "clique_percolation":
            status, detail = ("ready", "NetworkX clique percolation") if not missing else ("missing", "pip install networkx")
        elif m.key == "codeseg":
            binary = tools / "quickstart" / f"CoDeSEG-{qs.CODESEG_COMMIT[:12]}"
            if (manifest.get("codeseg") or {}).get("status") == "ready":
                status, detail = "ready", "prepared by hedonic setup"
            elif binary.is_file():
                status, detail = "ready", f"built from the pinned source ({binary.name})"
            else:
                status, detail = ("installable", "source download + compile on first use") if not missing else \
                    ("missing", "needs a C++ compiler")
        else:
            entry = manifest.get(m.setup) or {}
            if entry.get("status") == "ready":
                status, detail = "ready", "prepared by hedonic setup"
            elif not missing:
                status, detail = "installable", f"hedonic setup --methods {m.key} (or automatically on first use)"
            else:
                status, detail = "missing", "needs " + ", ".join(missing)
        rows.append({"key": m.key, "label": m.label, "status": status, "detail": detail,
                     "requires": REQUIREMENTS[m.key], "missing": missing, "screened_setting": m.screened,
                     "full_dblp": {"f1": m.f1, "onmi": m.onmi, "detection_seconds": m.seconds}})
    return rows


def _size(paths) -> int:
    return sum(p.stat().st_size for p in paths if p.is_file())


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)


def _remote_size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as r:
            return int(r.headers.get("Content-Length") or 0) or None
    except Exception:  # noqa: BLE001
        return None


def network_status(cache: Path, network_root: str | None, remote: bool) -> list[dict]:
    from hedonic.experiments.overlapping import snap

    root = expand_path(network_root) if network_root else Path(NETWORKS_DIR)
    rows = []
    for key, (label, n, m) in qs.NETWORKS.items():
        spec = snap.SPECS[key]
        local = snap._dataset_dir(root, spec)
        local_files = [local / f for f in (*spec.graph_files, *spec.raw_edge_files,
                                           *spec.cover_files.get("all", ()), *spec.raw_cover_files.get("all", ()))]
        cached = cache / "snap" / "raw" / key
        cached_files = [cached / spec.raw_edge_files[0], cached / spec.raw_cover_files["all"][0]]
        has_local = any(p.is_file() for p in local_files[: len(spec.graph_files) + len(spec.raw_edge_files)])
        has_cached = all(p.is_file() for p in cached_files)
        if has_local:
            where, disk, status = str(local), _size(p for p in local.rglob("*") if p.is_file()), "local archive"
        elif has_cached:
            where, disk, status = str(cached), _size(cached_files), "downloaded"
        else:
            where, disk, status = "", 0, "not downloaded"
        urls = snap.SNAP_DOWNLOAD_URLS.get(key, {})
        download = None
        if remote and urls:
            sizes = [_remote_size(urls["graph"]), _remote_size(urls["all"])]
            download = sum(sizes) if all(sizes) else None
        rows.append({"key": key, "label": label, "vertices": n, "edges": m, "avg_degree": 2 * m / n,
                     "status": status, "location": where, "disk_bytes": disk, "download_bytes": download,
                     "urls": {"graph": urls.get("graph"), "cover": urls.get("all")}})
    return rows


# --------------------------------------------------------------------------- rendering
def _columns() -> int:
    return shutil.get_terminal_size((160, 24)).columns  # a pipe or file: no width to respect


def _table(headers, rows, align, drop=()):
    """Box table. On a narrow terminal, columns listed in ``drop`` (in order) are left out until it fits."""
    keep = list(range(len(headers)))

    def build(keep):
        cols = [[str(r[i]) for r in rows] for i in keep]
        widths = [max(len(headers[i]), *(qs._visible(c) for c in col)) for i, col in zip(keep, cols)]
        return widths

    while sum(build(keep)) + 3 * len(keep) + 1 + 2 > _columns() and any(d in keep for d in drop):
        keep.remove(next(d for d in drop if d in keep))
    widths = build(keep)

    def line(cells):
        out = []
        for i, w in zip(keep, widths):
            c = cells[i]
            pad = " " * (w - qs._visible(str(c)))
            out.append(f" {c}{pad} " if align[i] == "l" else f" {pad}{c} ")
        return "  │" + "│".join(out) + "│"
    bar = lambda l, m_, r_: "  " + l + m_.join("─" * (w + 2) for w in widths) + r_  # noqa: E731
    return "\n".join([bar("┌", "┬", "┐"), line(headers), bar("├", "┼", "┤"), *map(line, rows), bar("└", "┴", "┘")])


def render_methods(rows: list[dict]) -> str:
    colour = {"ready": "green", "installable": "cyan", "missing": "red"}
    body = [[r["key"], r["label"], style(f"● {r['status']}", colour[r["status"]]),
             f"{r['full_dblp']['f1']:.3f}", f"{r['full_dblp']['onmi']:.3f}", f"{r['full_dblp']['detection_seconds']:.0f}",
             r["detail"]] for r in rows]
    out = [style("  Methods", "bold") + style("  (passed the staged screening; scores = screened setting on full "
                                               "com-DBLP, M1 Max)", "dim"),
           _table(["key", "method", "status", "F1", "ONMI", "time [s]", "details"], body, "lllrrrl", drop=(6, 1, 5, 4))]
    for r in rows:
        if r["status"] == "missing":
            out.append(style(f"  ✗ {r['key']}: {r['detail']}", "red"))
    counts = {s: sum(r["status"] == s for r in rows) for s in colour}
    out.append(style(f"  {counts['ready']} ready · {counts['installable']} installed automatically on first use · "
                     f"{counts['missing']} missing a system tool", "dim"))
    return "\n".join(out)


def render_networks(rows: list[dict], remote: bool) -> str:
    colour = {"local archive": "green", "downloaded": "green", "not downloaded": "yellow"}
    body = [[r["key"], r["label"], f"{r['vertices']:,}", f"{r['edges']:,}", f"{r['avg_degree']:.1f}",
             style(f"● {r['status']}", colour[r["status"]]), _human(r["disk_bytes"]) if r["disk_bytes"] else "–",
             (_human(r["download_bytes"]) if r["download_bytes"] else "?") if remote else "–"] for r in rows]
    out = [style("  SNAP networks with ground-truth communities", "bold"),
           _table(["key", "network", "vertices", "edges", "avg deg.", "status", "on disk", "download"], body, "llrrrlrr", drop=(4, 7, 6))]
    where = sorted({r["location"] for r in rows if r["location"]})
    if len(where) > 1 and os.path.commonpath(where) not in ("", "/"):
        out.append(style(f"  on disk under {os.path.commonpath(where)}", "dim"))
    elif where:
        out.append(style("  on disk: " + "; ".join(where), "dim"))
    if not remote:
        out.append(style("  missing networks are downloaded on first use; add --remote to query archive sizes", "dim"))
    return "\n".join(out)


def render_metrics() -> str:
    lines = [style("  Accuracy metrics reported by `hedonic run exp`", "bold")]
    for key, short, name, text in METRICS:
        lines.append(f"\n  {style(short, 'cyan', 'bold')}  {name}  " + style(f"(record key: {key})", "dim"))
        words, row = text.split(), "    "
        for w in words:
            if len(row) + len(w) > 96:
                lines.append(row)
                row = "    "
            row += w + " "
        lines.append(row.rstrip())
    lines.append("\n  " + style(PROTOCOL, "dim"))
    lines.append(style("  Also reported per run: detection time (detector only, excludes loading and scoring).", "dim"))
    return "\n".join(lines)


def _cover_labels(rows: list[dict]) -> list[dict]:
    """Expose concise cover labels while retaining the canonical SNAP identifier."""
    from hedonic.experiments.overlapping import snap

    result = []
    for row in rows:
        item = dict(row)
        if item.get("cover"):
            item["cover_id"] = item["cover"]
            item["cover"] = snap.cover_variant_label(item["cover"])
        result.append(item)
    return result


def _display_pairs(pairs: list[str]) -> list[str]:
    from hedonic.experiments.overlapping import snap

    result = []
    for pair in pairs:
        dataset, separator, cover = pair.partition(":")
        result.append(f"{dataset}:{snap.cover_variant_label(cover)}" if separator else pair)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hedonic show", description="Show the methods, the local data and the metrics.")
    p.add_argument("what", choices=("methods", "networks", "metrics", "covers", "table", "figure", "disjoint"), nargs="?",
                   default="methods")
    p.add_argument("--cache-dir", default=None, help="downloads and built runtimes (default: your `hedonic config` "
                                                      "cache_dir, else ~/.cache/hedonic)")
    p.add_argument("--network-root", default=None, help="folder with local SNAP archives (default: your `hedonic "
                                                        "config` network_root, else HEDONIC_NETWORKS_DIR)")
    p.add_argument("--analysis-cache", default=None,
                   help="directory for the generated CSV and source manifest (default: <network-root>/.hedonic-cache)")
    p.add_argument("--csv", default=None, help="CSV output path (default: <analysis-cache>/snap_table1.csv)")
    p.add_argument("--tex-output", default=None,
                   help="generated LaTeX table rows (default: <analysis-cache>/snap_networks_table_rows.tex)")
    p.add_argument("--figure-dir", default=None,
                   help="directory for Figure 1 A/B/C TeX panels (default: next to the table fragment)")
    p.add_argument("--refresh", action="store_true", help="re-analyse local files even when the CSV cache is current")
    p.add_argument("--remote", action="store_true", help="query snap.stanford.edu for archive sizes (HEAD only)")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    from hedonic.experiments.overlapping import userconfig

    machine = userconfig.defaults()
    cache = expand_path(a.cache_dir or machine.get("cache_dir") or "~/.cache/hedonic")
    a.network_root = a.network_root or machine.get("network_root")
    if a.what == "disjoint":
        from hedonic.experiments.disjoint import front

        if a.json:
            print(json.dumps({name: {**front.resolve(front.build_parser().parse_args(["--preset", name])),
                                     **front.plan(front.resolve(front.build_parser().parse_args(["--preset", name])))}
                              for name in front.PRESETS}, indent=2))
        else:
            print(front.render_presets())
        return 0
    if a.what == "methods":
        rows = method_status(codeseg_cache_dir(a.cache_dir))
        print(json.dumps(rows, indent=2) if a.json else render_methods(rows))
    elif a.what == "covers":
        from hedonic.experiments.overlapping import gt_spectrum

        root = a.network_root or str(NETWORKS_DIR)
        rows = gt_spectrum.select_cohort(
            gt_spectrum.discover(root, cache / "snap", datasets=list(qs.NETWORKS)), "auto", "auto"
        )
        print(json.dumps(_cover_labels(rows), indent=2, default=str) if a.json else gt_spectrum.render_cohort(rows))
    elif a.what in ("table", "figure"):
        from hedonic.experiments.overlapping import snap_table

        root = expand_path(a.network_root or machine.get("network_root") or str(NETWORKS_DIR))
        analysis_cache = expand_path(a.analysis_cache) if a.analysis_cache else root / ".hedonic-cache"
        rows_tex = expand_path(a.tex_output) if a.tex_output else analysis_cache / "snap_networks_table_rows.tex"
        counts_tex = rows_tex.with_name("snap_networks_table_counts.tex")
        figure_dir = expand_path(a.figure_dir) if a.figure_dir else rows_tex.parent
        figure_a_tex = figure_dir / "snap_networks_figure_a.tex"
        progress = None if a.json else lambda message: print(message, file=sys.stderr, flush=True)
        rows, refreshed, missing = snap_table.run(
            root, cache_dir=analysis_cache, rows_tex=rows_tex, counts_tex=counts_tex,
            figure_a_tex=figure_a_tex, figure_only=a.what == "figure",
            refresh=a.refresh, csv_path=expand_path(a.csv) if a.csv else None,
            progress=progress,
        )
        csv_path = expand_path(a.csv) if a.csv else analysis_cache / "snap_table1.csv"
        figure_paths = [str(figure_dir / f"snap_networks_figure_{letter}.tex") for letter in "abc"]
        if a.json:
            print(json.dumps({"network_root": str(root), "rows": _cover_labels(rows), "missing_pairs": _display_pairs(missing),
                              "cache_reused": not refreshed, "csv": str(csv_path),
                              "latex_rows": str(rows_tex), "figure_panels": figure_paths},
                             indent=2, default=str))
        else:
            if a.what == "table":
                print(snap_table.render(rows) if rows else "  No local network/cover pairs found.")
                print(f"\n  Analysed {sum(r['row_type'] == 'network' for r in rows)} networks and "
                      f"{sum(r['row_type'] == 'cover' for r in rows)} network/cover pairs"
                      f" ({'reused current cache' if not refreshed else 'saved fresh analysis'}).")
                print(f"  CSV: {csv_path}")
                print(f"  Table LaTeX: {rows_tex}")
            else:
                status = "reused current analysis CSV" if not refreshed else "refreshed the analysis CSV from local files"
                print(f"  Generated Figure 1 panels A/B/C ({status}).")
                print(f"  CSV: {csv_path}")
            print("  Figure panels: " + ", ".join(figure_paths))
            if missing:
                print(f"  Skipped absent pairs: {', '.join(_display_pairs(missing))}")
    elif a.what == "networks":
        rows = network_status(cache, a.network_root, a.remote)
        print(json.dumps(rows, indent=2) if a.json else render_networks(rows, a.remote))
    else:
        print(json.dumps([dict(zip(("key", "short", "name", "definition"), m)) for m in METRICS] + [{"protocol": PROTOCOL}],
                         indent=2) if a.json else render_metrics())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
