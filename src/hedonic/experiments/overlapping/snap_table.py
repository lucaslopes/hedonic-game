"""Build the SNAP network/cover table from files already present on disk.

The CSV is the durable analysis record.  A small manifest fingerprints the
source files by size and modification time so repeated calls can reuse it
without rescanning large SNAP archives.  This module never provisions data.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hedonic.experiments.overlapping import quickstart, snap

SCHEMA_VERSION = 1
CSV_FIELDS = (
    "row_type", "dataset", "network", "cover", "communities", "vertices",
    "edges", "avg_degree", "density", "covered_fraction", "cover_vertices_outside",
    "network_vertices",
    "network_edges", "directed", "source_kind", "source_graph", "source_cover",
)
# The six SNAP ``com-*`` networks with ground-truth communities. Wiki-topcats (also hosted by SNAP) is a directed
# hyperlink graph with category lists, not one of them, and is not part of Table 1.
NETWORK_ORDER = (
    "dblp", "amazon", "youtube", "friendster", "livejournal", "orkut",
)
NETWORK_SYMBOLS = {
    "dblp": "D", "amazon": "A", "youtube": "Y", "livejournal": "L",
    "friendster": "F", "orkut": "O",
}
# Orkut's top5000 induced-graph counts were measured separately because loading
# the raw graph requires substantial memory. Keep the cover in the descriptive
# table and Figure 1 without rescanning the 117-million-edge source graph.
CURATED_COVER_ROWS = {
    ("orkut", "top5000"): {
        "communities": 5_000, "vertices": 731_514, "edges": 21_992_510,
        "network_vertices": 3_072_441, "network_edges": 117_185_083,
        "directed": False,
        "source_kind": "reported_snap_stats",
        "source_graph": "SNAP com-orkut.ungraph.txt.gz",
        "source_cover": "SNAP com-orkut.top5000.cmty.txt.gz",
    },
}
ARTICLE_COVERS = {
    "dblp": frozenset({"all", "top5000"}),
    "amazon": frozenset({"all", "top5000"}),
    "youtube": frozenset({"all", "top5000"}),
    "livejournal": frozenset({"top5000"}),
    "friendster": frozenset({"top5000"}),
}
DATASET_SEMANTICS = {
    "amazon": (
        "Product",
        "Co-purchase",
        "Product categories",
    ),
    "dblp": (
        "Author",
        "Co-authorship",
        "Venue authors",
    ),
    "youtube": (
        "User",
        "Friendship",
        "User groups",
    ),
    "livejournal": (
        "User",
        "Friendship",
        "User groups",
    ),
    "orkut": (
        "User",
        "Friendship",
        "User groups",
    ),
    "friendster": (
        "User",
        "Friendship",
        "User groups",
    ),
}


def _source_pair(root: Path, name: str, cover: str) -> dict[str, Any] | None:
    spec = snap.SPECS[name]
    base = snap._dataset_dir(root, spec)
    edge_path = snap._first_existing(base, spec.raw_edge_files)
    raw_cover = snap._first_existing(base, spec.raw_cover_files.get(cover, ()))
    if edge_path is not None and raw_cover is not None:
        return {"base": base, "graph": edge_path, "cover": raw_cover, "kind": "raw_gzip"}
    graph_pickle = snap._first_existing(base, spec.graph_files)
    cover_pickle = snap._first_existing(base, spec.cover_files.get(cover, ()))
    if graph_pickle is not None and cover_pickle is not None:
        return {"base": base, "graph": graph_pickle, "cover": cover_pickle, "kind": "igraph_pickle"}
    return None


def _signature(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[Path] = set()
    result = []
    for pair in pairs:
        for role in ("graph", "cover"):
            path = pair[role].resolve()
            if path in seen:
                continue
            seen.add(path)
            stat = path.stat()
            result.append({"path": str(path), "size_bytes": stat.st_size,
                           "mtime_ns": stat.st_mtime_ns})
    return result


def _same_sources(saved: Any, current: list[dict[str, Any]]) -> bool:
    """Compare the source manifest without making network display order significant."""
    try:
        return sorted(saved, key=lambda item: item["path"]) == sorted(
            current, key=lambda item: item["path"]
        )
    except (KeyError, TypeError):
        return False


def _cover_bitmap(path: Path) -> tuple[bytearray, int]:
    """Read a raw SNAP cover into a compact vertex-ID bitmap and count lines."""
    bits = bytearray()
    communities = 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            communities += 1
            try:
                members = np.fromstring(line, sep=" ", dtype=np.int64)
            except ValueError as exc:
                raise ValueError(f"invalid community at {path}:{line_number}") from exc
            if not members.size:
                continue
            if int(members.min()) < 0:
                raise ValueError(f"negative vertex ID at {path}:{line_number}")
            max_byte = int(members.max()) // 8
            if max_byte >= len(bits):
                bits.extend(b"\0" * (max_byte + 1 - len(bits)))
            view = np.frombuffer(bits, dtype=np.uint8)
            byte_ids = members >> 3
            masks = (1 << (members & 7)).astype(np.uint8)
            np.bitwise_or.at(view, byte_ids, masks)
            del view
    return bits, communities


def _raw_graph_stats(graph_path: Path, cover_bits: dict[str, bytearray]) -> dict[str, Any]:
    """Count graph vertices/edges and each induced cover edge count in one pass."""
    if not cover_bits:
        raise ValueError("no locally available cover files")
    names = list(cover_bits)
    node_bits = bytearray()
    edge_counts = {name: 0 for name in names}
    n_vertices = 0
    n_edges = 0
    chunk_size = 8 * 1024 * 1024
    with gzip.open(graph_path, "rb") as stream:
        # SNAP edge files put comments before the data block.
        while True:
            line = stream.readline()
            if not line or not line.startswith(b"#"):
                break
        carry = line if line else b""
        while True:
            block = stream.read(chunk_size)
            if not block:
                data = carry
                carry = b""
            else:
                data = carry + block
                boundary = data.rfind(b"\n")
                if boundary < 0:
                    carry = data
                    continue
                carry = data[boundary + 1:]
                data = data[:boundary + 1]
            if data:
                values = np.fromstring(data.decode("ascii"), sep=" ", dtype=np.int64)
                if values.size % 2:
                    raise ValueError(f"malformed edge list near end of {graph_path}")
                if values.size:
                    if int(values.min()) < 0:
                        raise ValueError(f"negative vertex ID in {graph_path}")
                    source, target = values[0::2], values[1::2]
                    n_edges += int(source.size)
                    nodes = np.unique(values)
                    max_byte = int(nodes[-1]) // 8
                    if max_byte >= len(node_bits):
                        node_bits.extend(b"\0" * (max_byte + 1 - len(node_bits)))
                    node_view = np.frombuffer(node_bits, dtype=np.uint8)
                    byte_ids = nodes >> 3
                    masks = (1 << (nodes & 7)).astype(np.uint8)
                    before = node_view[byte_ids].copy()
                    n_vertices += int(np.count_nonzero((before | masks) != before))
                    np.bitwise_or.at(node_view, byte_ids, masks)
                    del node_view

                    for name in names:
                        bits = cover_bits[name]
                        if max_byte >= len(bits):
                            bits.extend(b"\0" * (max_byte + 1 - len(bits)))
                        view = np.frombuffer(bits, dtype=np.uint8)
                        src_hit = (view[source >> 3] & (1 << (source & 7))) != 0
                        dst_hit = (view[target >> 3] & (1 << (target & 7))) != 0
                        edge_counts[name] += int(np.count_nonzero(src_hit & dst_hit))
                        del view
            if not block:
                break
    covered_counts = {}
    outside_counts = {}
    node_view = np.frombuffer(node_bits, dtype=np.uint8)
    for name, bits in cover_bits.items():
        cover_view = np.frombuffer(bits, dtype=np.uint8)
        shared_length = min(len(node_view), len(cover_view))
        inside = sum(int(byte).bit_count() for byte in
                     np.bitwise_and(node_view[:shared_length], cover_view[:shared_length]))
        total = sum(int(byte).bit_count() for byte in cover_view)
        covered_counts[name] = inside
        outside_counts[name] = total - inside
    return {"vertices": n_vertices, "edges": n_edges, "induced_edges": edge_counts,
            "covered_vertices": covered_counts, "outside_vertices": outside_counts}


def _stats(graph, cover: list[list[int]]) -> tuple[int, int, float, float]:
    covered = {int(v) for community in cover for v in community}
    n = len(covered)
    induced = graph.induced_subgraph(sorted(covered))
    m = induced.ecount()
    degree = 2 * m / n if n else 0.0
    density = 2 * m / (n * (n - 1)) if n > 1 else 0.0
    return n, m, degree, density


def _row(row_type: str, name: str, label: str, cover: str, communities: int,
         vertices: int, edges: int, network_vertices: int, network_edges: int,
         directed: bool, source_kind: str, graph_source: str, cover_source: str,
         outside_vertices: int = 0) -> dict[str, Any]:
    degree = 2 * edges / vertices if vertices else 0.0
    density = 2 * edges / (vertices * (vertices - 1)) if vertices > 1 else 0.0
    return {
        "row_type": row_type, "dataset": name, "network": label, "cover": cover,
        "communities": communities, "vertices": vertices, "edges": edges,
        "avg_degree": degree, "density": density,
        "covered_fraction": vertices / network_vertices if network_vertices else 0.0,
        "cover_vertices_outside": outside_vertices,
        "network_vertices": network_vertices, "network_edges": network_edges,
        "directed": directed, "source_kind": source_kind,
        "source_graph": graph_source, "source_cover": cover_source,
    }


def _add_curated_cover_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    present = {(row["dataset"], row["cover"]) for row in rows if row["row_type"] == "cover"}
    for (name, variant), values in CURATED_COVER_ROWS.items():
        if (name, variant) in present:
            continue
        label = quickstart.NETWORKS[name][0].split(" (")[0]
        rows.append(_row("cover", name, label, variant,
                         values["communities"], values["vertices"], values["edges"],
                         values["network_vertices"], values["network_edges"],
                         values["directed"], values["source_kind"], values["source_graph"],
                         values["source_cover"]))
    return rows


def _analyze_raw(name: str, label: str, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first = pairs[0]
    spec = snap.SPECS[name]
    covers: dict[str, bytearray] = {}
    community_counts: dict[str, int] = {}
    for pair in pairs:
        bits, count = _cover_bitmap(pair["cover"])
        covers[pair["variant"]] = bits
        community_counts[pair["variant"]] = count
    graph_stats = _raw_graph_stats(first["graph"], covers)
    n, m = graph_stats["vertices"], graph_stats["edges"]
    rows = [_row("network", name, label, "", 0, n, m, n, m,
                 spec.directed, "raw_gzip", str(first["graph"].resolve()), "")]
    for pair in pairs:
        variant = pair["variant"]
        covered = graph_stats["covered_vertices"][variant]
        induced = graph_stats["induced_edges"][variant]
        rows.append(_row("cover", name, label, variant, community_counts[variant],
                         covered, induced, n, m, spec.directed, "raw_gzip",
                         str(first["graph"].resolve()), str(pair["cover"].resolve()),
                         graph_stats["outside_vertices"][variant]))
    return rows


def _analyze_pickles(name: str, label: str, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spec = snap.SPECS[name]
    data = [(pair, snap._load_from_pickles(spec, pair["base"], pair["variant"])) for pair in pairs]
    data = [(pair, dataset) for pair, dataset in data if dataset is not None]
    if not data:
        return []
    graph = data[0][1].graph
    n, m = graph.vcount(), graph.ecount()
    rows = [_row("network", name, label, "", 0, n, m, n, m,
                 graph.is_directed(), "igraph_pickle", data[0][1].report["graph_path"], "")]
    for pair, dataset in data:
        covered, induced, degree, density = _stats(graph, dataset.cover)
        # Keep the common row schema and use the same induced-subgraph convention.
        row = _row("cover", name, label, pair["variant"], len(dataset.cover), covered,
                   induced, n, m, graph.is_directed(), "igraph_pickle",
                   dataset.report["graph_path"], dataset.report["ground_truth_path"])
        row["avg_degree"], row["density"] = degree, density
        rows.append(row)
    return rows


def analyze(root: Path, progress: Callable[[str], None] | None = None
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    pairs_for_signature: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in NETWORK_ORDER:
        label = quickstart.NETWORKS[name][0].split(" (")[0]
        spec = snap.SPECS[name]
        base = snap._dataset_dir(root, spec)
        local_pairs = []
        for variant in ("all", "top5000"):
            if (name, variant) in CURATED_COVER_ROWS:
                continue
            pair = _source_pair(root, name, variant)
            if pair is None:
                missing.append(f"{name}:{variant}")
                continue
            pair["variant"] = variant
            local_pairs.append(pair)
            pairs_for_signature.append(pair)
        if not local_pairs:
            continue
        if progress:
            progress(f"  {label}: scanning {len(local_pairs)} local cover file(s) and the graph")
        raw_pairs = [p for p in local_pairs if p["kind"] == "raw_gzip"]
        if raw_pairs:
            rows.extend(_analyze_raw(name, label, raw_pairs))
        pickle_pairs = [p for p in local_pairs if p["kind"] == "igraph_pickle"]
        if pickle_pairs:
            rows.extend(_analyze_pickles(name, label, pickle_pairs))
    return _add_curated_cover_rows(rows), _signature(pairs_for_signature), missing


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        result = []
        for row in csv.DictReader(stream):
            for key in ("communities", "vertices", "edges", "network_vertices", "network_edges"):
                row[key] = int(row[key])
            for key in ("avg_degree", "density", "covered_fraction"):
                row[key] = float(row[key])
            row["cover_vertices_outside"] = int(row["cover_vertices_outside"])
            row["directed"] = row["directed"].lower() == "true"
            result.append(row)
        return result


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _canonicalize_source_paths(rows: list[dict[str, Any]], root: Path) -> bool:
    """Upgrade older CSV rows that stored source basenames instead of full paths."""
    changed = False
    for row in rows:
        spec = snap.SPECS[row["dataset"]]
        base = snap._dataset_dir(root, spec)
        cover = row["cover"]
        for key in ("source_graph", "source_cover"):
            value = row[key]
            if not value or Path(value).is_absolute():
                continue
            candidates = ((*spec.raw_edge_files, *spec.graph_files) if key == "source_graph" else
                          (*spec.raw_cover_files.get(cover, ()), *spec.cover_files.get(cover, ()))
                          if cover else ())
            candidate = next((base / relative for relative in candidates
                              if Path(relative).name == Path(value).name and (base / relative).is_file()), None)
            if candidate is not None:
                row[key] = str(candidate.resolve())
                changed = True
    return changed


def _thousands(value: int) -> str:
    return f"{value:,}".replace(",", "\\,")


def _density_tex(value: float) -> str:
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / (10 ** exponent)
    return rf"${mantissa:.2g}\times10^{{{exponent}}}$"


def _tex_rows(rows: list[dict[str, Any]]) -> tuple[str, str, int, int]:
    order = {name: index for index, name in enumerate(NETWORK_ORDER)}
    rows = sorted(rows, key=lambda row: order[row["dataset"]])
    pairs = [r for r in rows if r["row_type"] == "cover"]
    networks = [r for r in rows if r["row_type"] == "network"]
    lines = [
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lcrrrrlll@{}}",
        r"\toprule",
        r"Network / cover & In paper & vertices $n(H)$ & edges $m(H)$ & $\bar d(H)$ & $\rho(H)$ & "
        r"\shortstack[l]{Node\\represents} & \shortstack[l]{Edge\\represents} & "
        r"\shortstack[l]{Ground-truth\\community represents}\\",
        r"\midrule",
    ]
    previous_network = None
    for row in rows:
        if row["row_type"] == "network":
            if previous_network is not None:
                lines.append(r"\cmidrule(lr){1-9}")
            previous_network = row["dataset"]
            symbol = NETWORK_SYMBOLS[row["dataset"]]
            name = rf"\textbf{{{row['network']}}} ($G_{{\mathrm {symbol}}}$)"
            marker = ""
        else:
            cover_label = snap.cover_variant_label(row["cover"])
            label = rf"\quad \texttt{{{cover_label}}} ({_thousands(row['communities'])})"
            name = label
            marker = (r"\PaperCheckmark" if row["cover"] in ARTICLE_COVERS.get(row["dataset"], ())
                      else r"$\times$")
        vertices = _thousands(row["vertices"])
        if row["row_type"] == "cover":
            vertices += rf" ({100 * row['covered_fraction']:.1f}\%)"
            semantics = ("", "", "")
        else:
            semantics = DATASET_SEMANTICS[row["dataset"]]
        lines.append(rf"{name} & {marker} & {vertices} & {_thousands(row['edges'])} & "
                     rf"{row['avg_degree']:.1f} & {_density_tex(row['density'])} & "
                     rf"{semantics[0]} & {semantics[1]} & {semantics[2]}\\")
    lines.extend([r"\bottomrule", r"\end{tabular}}"])
    counts = (rf"\def\SnapDatasetCount{{{len(networks)}}}" + "\n" +
              rf"\def\SnapPairCount{{{len(pairs)}}}" + "\n")
    return "\n".join(lines) + "\n", counts, len(networks), len(pairs)


def _hatch(out: list[str], color: str, x0: float, end: float, bottom: float, top: float, *,
           forward: bool = True, backward: bool = False, spacing: float | None = None) -> None:
    """Thin coloured 45-degree strokes clipped to a bar: "/" for the all cover, "/" plus "\\" (crosses) for top5000.

    Drawn as lines rather than a TeX pattern, so the colour follows the network in every PDF viewer and the
    strokes have the same weight and spacing in all three panels.
    """
    height = top - bottom
    spacing = spacing or 0.105
    out.append(r"\begin{scope}")
    out.append(rf"\clip ({x0:.3f},{bottom:.3f}) rectangle ({end:.3f},{top:.3f});")
    first = math.floor(-height / spacing) - 1
    last = math.ceil((end - x0) / spacing) + 1
    for index in range(first, last + 1):
        start = x0 + index * spacing
        if forward:
            out.append(rf"\path[draw={color}!90,line width=0.012cm] ({start:.3f},{bottom:.3f})--({start + height:.3f},{top:.3f});")
        if backward:
            out.append(rf"\path[draw={color}!90,line width=0.012cm] ({start + height:.3f},{bottom:.3f})--({start:.3f},{top:.3f});")
    out.append(r"\end{scope}")


def _figure_panel(rows: list[dict[str, Any]], *, metric: str, low: int, high: int,
                  ticks: list[int], axis_label: str,
                  separate_tracks: bool = False) -> str:
    palette = {
        # Matplotlib's qualitative tab20 hues, assigned per the manuscript's
        # requested network mapping.
        "dblp": (31, 119, 180), "amazon": (255, 127, 14),
        "youtube": (214, 39, 40), "livejournal": (23, 190, 207),
        "friendster": (44, 160, 44), "orkut": (148, 103, 189),
    }
    networks = {row["dataset"]: row for row in rows if row["row_type"] == "network"}
    covers: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["row_type"] == "cover":
            covers.setdefault(row["dataset"], {})[row["cover"]] = row
    x0, x1 = 1.50, 5.60

    def x_of(value: float) -> float:
        bounded = max(10**low, value)
        return x0 + (x1 - x0) * (math.log10(bounded) - low) / (high - low)

    out = []
    for name, rgb in palette.items():
        if name in networks:
            out.append(rf"\definecolor{{Net{name.title()}}}{{RGB}}{{{rgb[0]},{rgb[1]},{rgb[2]}}}")
    out.extend([
        r"\begin{tikzpicture}[scale=0.973,transform shape,x=1cm,y=1cm,inner sep=0pt,outer sep=0pt]",
        r"\path[fill=white] (0.013,0.028) rectangle (5.814,4.064);",
    ])
    tick_x = [x0 + (x1 - x0) * (power - low) / (high - low) for power in ticks]
    for tx in tick_x:
        out.append(rf"\path[draw=gray!25,line width=0.014cm] ({tx:.3f},0.98)--({tx:.3f},3.62);")
    display_order = [name for name in NETWORK_ORDER if name in networks]
    row_y = [3.40 - i * (2.25 / max(1, len(display_order) - 1)) for i in range(len(display_order))]
    for i, name in enumerate(display_order):
        row = networks[name]
        color = f"Net{name.title()}"
        y = row_y[i]
        xa, xb = y - 0.10, y + 0.10
        full_x = x_of(float(row[metric]))
        available = covers.get(name, {})
        selected = ARTICLE_COVERS.get(name, frozenset())
        if selected and selected == frozenset(available):
            label = rf"\textbf{{{row['network']}}}"
        elif selected == frozenset({"top5000"}):
            label = rf"\textit{{{row['network']}}}"
        else:
            label = row["network"]
        out.append(rf"\node[anchor=east] at (1.38,{y:.3f}) {{{label}}};")
        if separate_tracks:
            # Densities are independent ratios, so show aligned tracks instead
            # of making their bar lengths look like nested set sizes.
            tracks = [(row, y + 0.105, None)]
            if "all" in available:
                tracks.append((available["all"], y, "all"))
            if "top5000" in available:
                tracks.append((available["top5000"], y - 0.105, "top5000"))
            for item, center, pattern in tracks:
                end = x_of(float(item[metric]))
                if end <= x0:
                    end = x0 + 0.025
                top, bottom = center + 0.035, center - 0.035
                out.append(rf"\path[fill={color}!12,draw={color}!65,line width=0.012cm] ({x0:.3f},{bottom:.3f}) rectangle ({end:.3f},{top:.3f});")
                if pattern is not None:
                    _hatch(out, color, x0, end, bottom, top, backward=item.get("cover") == "top5000")
                    out.append(rf"\path[draw={color}!65,line width=0.012cm] ({x0:.3f},{bottom:.3f}) rectangle ({end:.3f},{top:.3f});")
        else:
            out.append(rf"\path[fill={color}!12] ({x0:.3f},{xa:.3f}) rectangle ({full_x:.3f},{xb:.3f});")
            for variant in ("all", "top5000"):
                cover = available.get(variant)
                if cover is None:
                    continue
                end = x_of(float(cover[metric]))
                if end <= x0:
                    end = x0 + 0.025
                # Each cover replaces the hatch beneath its interval. Thus the
                # all-only portion has one hatch, while top5000 has both cover
                # hatches and visibly reads as a subset of all.
                out.append(rf"\path[fill=white] ({x0:.3f},{xa:.3f}) rectangle ({end:.3f},{xb:.3f});")
                out.append(rf"\path[fill={color}!12] ({x0:.3f},{xa:.3f}) rectangle ({end:.3f},{xb:.3f});")
                _hatch(out, color, x0, end, xa, xb, backward=variant == "top5000")
            out.append(rf"\path[draw={color}!65,line width=0.016cm] ({x0:.3f},{xa:.3f}) rectangle ({full_x:.3f},{xb:.3f});")
    out.append(rf"\path[draw=black,line width=0.021cm] ({x0:.3f},0.98)--({x1:.3f},0.98);")
    out.append(rf"\path[draw=black,line width=0.021cm] ({x0:.3f},0.98)--({x0:.3f},3.62);")
    for tx, power in zip(tick_x, ticks):
        out.append(rf"\path[draw=black,line width=0.021cm] ({tx:.3f},0.98)--({tx:.3f},0.90);")
        out.append(rf"\node[anchor=north] at ({tx:.3f},0.86) {{$10^{{{power}}}$}};")
    out.append(rf"\node[anchor=north] at (3.55,0.43) {{{axis_label}}};")
    out.append(r"\end{tikzpicture}")
    return "\n".join(out) + "\n"


def write_figures(rows: list[dict[str, Any]], figure_a_path: Path,
                  figure_b_path: Path | None = None,
                  figure_c_path: Path | None = None) -> None:
    """Write the three TeX panels used to assemble Figure 1."""
    figure_b_path = figure_b_path or figure_a_path.with_name("snap_networks_figure_b.tex")
    figure_c_path = figure_c_path or figure_a_path.with_name("snap_networks_figure_c.tex")
    figure_a_path.parent.mkdir(parents=True, exist_ok=True)
    figure_a_path.write_text(_figure_panel(
        rows, metric="vertices", low=4, high=8, ticks=[4, 5, 6, 7, 8],
        axis_label="number of vertices (log scale)"), encoding="utf-8")
    figure_b_path.write_text(_figure_panel(
        rows, metric="edges", low=4, high=10, ticks=[4, 6, 8, 10],
        axis_label="number of edges (log scale)"), encoding="utf-8")
    figure_c_path.write_text(_figure_panel(
        rows, metric="density", low=-7, high=-3, ticks=[-7, -6, -5, -4, -3],
        axis_label="edge density (log scale)", separate_tracks=True), encoding="utf-8")


def write_latex(rows: list[dict[str, Any]], rows_path: Path, counts_path: Path,
                figure_a_path: Path, figure_b_path: Path | None = None,
                figure_c_path: Path | None = None) -> tuple[int, int]:
    table, counts, networks, pairs = _tex_rows(rows)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(table, encoding="utf-8")
    counts_path.write_text(counts, encoding="utf-8")
    write_figures(rows, figure_a_path, figure_b_path, figure_c_path)
    return networks, pairs


def render(rows: list[dict[str, Any]]) -> str:
    order = {name: index for index, name in enumerate(NETWORK_ORDER)}
    rows = sorted(rows, key=lambda row: order[row["dataset"]])
    headers = ("Network / cover", "In paper", "vertices n", "edges m", "mean degree", "density")
    display = []
    for row in rows:
        if row["row_type"] == "network":
            name, marker = row["network"], ""
        else:
            name = f"  {snap.cover_variant_label(row['cover'])} ({row['communities']:,} communities)"
            marker = "✓" if row["cover"] in ARTICLE_COVERS.get(row["dataset"], ()) else "×"
        vertices = f"{row['vertices']:,}"
        if row["row_type"] == "cover":
            vertices += f" ({100 * row['covered_fraction']:.1f}%)"
        display.append((name, marker, vertices, f"{row['edges']:,}", f"{row['avg_degree']:.1f}",
                        f"{row['density']:.3g}"))
    widths = [max(len(h), *(len(row[i]) for row in display)) for i, h in enumerate(headers)]
    lines = ["  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("  " + "  ".join("-" * w for w in widths))
    for row in display:
        lines.append("  " + "  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
    lines.extend(["", "  ✓ included in this article's cover summary; × deferred to future work due to computational cost."])
    return "\n".join(lines)


def run(root: Path, *, cache_dir: Path, rows_tex: Path, counts_tex: Path,
        figure_a_tex: Path | None = None, refresh: bool = False,
        csv_path: Path | None = None, progress: Callable[[str], None] | None = None,
        figure_only: bool = False,
        ) -> tuple[list[dict[str, Any]], bool, list[str]]:
    csv_path = csv_path or cache_dir / "snap_table1.csv"
    manifest_path = csv_path.with_suffix(".json")
    # Discover source signatures without opening the archives.  A changed file
    # invalidates the persisted analysis; no remote catalogue is consulted.
    source_pairs = []
    missing = []
    for name in NETWORK_ORDER:
        for variant in ("all", "top5000"):
            if (name, variant) in CURATED_COVER_ROWS:
                continue
            pair = _source_pair(root, name, variant)
            if pair is None:
                missing.append(f"{name}:{variant}")
            else:
                source_pairs.append(pair)
    signatures = _signature(source_pairs)
    cached = None
    if not refresh and csv_path.is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") == SCHEMA_VERSION and _same_sources(
                manifest.get("sources"), signatures
            ):
                cached = _load_csv(csv_path)
        except (OSError, ValueError, KeyError):
            cached = None
    if cached is None:
        rows, signatures, missing = analyze(root, progress=progress)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _save_csv(csv_path, rows)
        manifest_path.write_text(json.dumps({"schema": SCHEMA_VERSION, "sources": signatures,
                                             "rows": len(rows)}, indent=2) + "\n", encoding="utf-8")
        refreshed = True
    else:
        rows, refreshed = _add_curated_cover_rows(cached), False
        if _canonicalize_source_paths(rows, root):
            _save_csv(csv_path, rows)
    figure_a_tex = figure_a_tex or rows_tex.with_name("snap_networks_figure_a.tex")
    if figure_only and not refreshed:
        write_figures(rows, figure_a_tex)
    else:
        # If a figure request discovers stale data, keep the table and figure
        # fragments synchronized with the same freshly analyzed CSV.
        write_latex(rows, rows_tex, counts_tex, figure_a_tex)
    return rows, refreshed, missing
