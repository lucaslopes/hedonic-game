"""``hedonic run exp``: a configurable, one-command overlapping-community benchmark.

Run with no arguments in a terminal to open an interactive wizard (arrow keys,
space, enter); its first choice runs the default experiment. Any flag, ``--yes``,
or a non-terminal session runs directly. Examples::

    hedonic run exp                                  # wizard (the quick default run is the first choice, then a yes)
    hedonic run exp --yes                            # default: DBLP, 5,000 vertices, seed 0, Hedonic + CoDeSEG
    hedonic run exp --networks dblp,amazon --methods all --seeds 0-2 --nodes 20000
    hedonic run exp --networks youtube --nodes 0 --methods hedonic,fox,clique_percolation   # clique_percolation is CliquePerc

Networks are the seven SNAP ground-truth networks; methods are Hedonic and the
eight baselines that passed the overlap paper's staged screening.
``--preset screened`` (default) uses each method's most accurate setting within
ten minutes on full DBLP; ``--preset paper`` uses the registry (paper)
defaults. Detection and scoring go through ``hedonic-exp overlapping-codeseg``
unchanged, so every record has the same loader, bound, paper filter, and
metrics as the benchmark.

Everything is fetched lazily and cached under ``--cache-dir`` (default
``~/.cache/hedonic``): SNAP archives from snap.stanford.edu, CoDeSEG's C++
source pinned to the reproduced commit (compiled with the system compiler; the
repository has no licence file, so it is not bundled), and the other external
runtimes through ``hedonic setup``. A method whose runtime cannot be
prepared is reported as unavailable with the reason; the others still run.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from hedonic.experiments.config import RUNS_ROOT, codeseg_cache_dir, expand_path
from hedonic.experiments.overlapping import replication, snap, tui
from hedonic.experiments.overlapping.tui import style

# --------------------------------------------------------------------------- catalogue
class _Targets(dict):
    """The network catalogue. It also answers for a *target* ``network:cover`` (``dblp:top5k``): that network's figures
    under a label that names the cover, so every lookup by a plan's network works whether or not the plan has covers."""

    def __missing__(self, key):
        base, _, cover = str(key).partition(":")
        if cover and dict.__contains__(self, base):
            label, vertices, edges = dict.__getitem__(self, base)
            return (f"{label} · {snap.cover_variant_label(cover)}", vertices, edges)
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


NETWORKS = _Targets({  # name: (label, vertices, edges) — SNAP ground-truth networks
    "dblp": ("DBLP", 317_080, 1_049_866),
    "amazon": ("Amazon", 334_863, 925_872),
    "youtube": ("YouTube", 1_134_890, 2_987_624),
    "livejournal": ("LiveJournal", 3_997_962, 34_681_189),
    "friendster": ("Friendster", 65_608_366, 1_806_067_135),
    "orkut": ("Orkut", 3_072_441, 117_185_083),
    "wikipedia": ("Wikipedia (topcats)", 1_791_489, 28_511_807),
})
DBLP_EDGES = NETWORKS["dblp"][2]


@dataclass(frozen=True)
class Method:
    key: str            # CLI name
    runner: str         # overlapping-codeseg method name
    label: str
    screened: dict      # Stage-3 setting (most accurate within 10 min on full DBLP)
    seconds: float      # Stage-3 detection time on full DBLP (M1 Max) — ETA basis
    f1: float           # Stage-3 score on full DBLP (hint only)
    onmi: float
    setup: str | None = None      # `hedonic setup --methods` name that provides the runtime
    threads: bool = False         # accepts a "threads" parameter
    scaled: tuple = ()            # community-count parameters scaled with graph size


METHODS = [
    # gamma = 7000 x density(full DBLP) = 0.1462, given as an absolute CPM resolution so that it transfers to
    # subgraphs and other networks (a density multiple would grow with the subgraph's higher density).
    Method("hedonic", "hedonic_local", "Hedonic", {"max_memberships": 64, "absolute_resolution": 0.1462}, 40, .425, .519),
    # one thread (the runner's default), where its runs are deterministic: see replication.REPLICATION
    Method("codeseg", "codeseg", "CoDeSEG", {"iterations": 20, "tau": 0.5}, 6, .406, .518),
    Method("fox", "fox", "FOX (LazyFox)", {"wcc_threshold": 0.01, "queue_size": 1}, 184, .428, .525, "fox", True),
    Method("clique_percolation", "clique_percolation", "CliquePerc (clique percolation)", {"clique_size": 4}, 9, .419, .524, "clique_percolation"),
    Method("angel", "angel", "ANGEL", {"threshold": 0.6, "min_community_size": 3}, 58, .403, .518, "angel"),
    Method("bigclam", "bigclam", "BigCLAM", {"communities": 25000}, 582, .395, .472, "bigclam", True, ("communities",)),
    Method("ego_splitting", "ego_splitting", "Ego-Splitting", {"resolution": 1.0}, 96, .333, .500, "angel"),
    Method("ncgame", "ncgame", "NcGame", {}, 15, .326, .484, "ncgame"),
    Method("neo_kmeans", "neo_kmeans", "NEO-K-Means", {"clusters": 13477, "alpha": 10, "beta": 0.0001}, 475, .334, .398,
           "neo_kmeans", False, ("clusters",)),
]
# Supervised controls: reference points that read the ground truth, never part of "all" or of the screened
# catalogue. disjoint_oracle answers the best disjoint partition it can build from the reference cover (the ceiling of
# an overlap-blind answer); its F1 hint is measured on full DBLP (deterministic, no parameters).
CONTROLS = [
    Method("disjoint_oracle", "disjoint_oracle", "Disjoint oracle (GT)", {}, 12, .840, float("nan")),
]
CONTROL_KEYS = [m.key for m in CONTROLS]
BY_KEY = {m.key: m for m in METHODS + CONTROLS}
ALIASES = {"hedonic_local": "hedonic", "hoc": "hedonic", "hoc_local": "hedonic",
           "lazyfox": "fox", "neo-k-means": "neo_kmeans", "ego-splitting": "ego_splitting"}
COLUMNS = (("f1", "F1 sym."), ("matching_f1", "F1 match"), ("node_micro_f1", "F1 micro"),
           ("size_weighted_community_f1", "F1 size-w."), ("onmi", "ONMI"), ("omega", "Omega"))
CODESEG_COMMIT = "d8ba74f6f0eac11ff8a92dbb5aa4fb1780193c1c"
CODESEG_RAW = f"https://raw.githubusercontent.com/SELGroup/CoDeSEG/{CODESEG_COMMIT}/code_c%2B%2B/CoDeSEG/"
CODESEG_SOURCES = ("main.cpp", "lib/CoDeSEG.cpp", "lib/DynamicArray.cpp", "lib/Utility.cpp")
CODESEG_HEADERS = ("lib/CoDeSEG.h", "lib/DynamicArray.h", "lib/CxxOpts.h", "lib/Utility.h", "lib/ThreadPool.h")


@dataclass
class Config:
    networks: list[str] = field(default_factory=lambda: ["dblp"])
    methods: list[str] = field(default_factory=lambda: ["hedonic", "codeseg"])
    nodes: int = 5000                      # 0 = full graph
    seeds: list[int] = field(default_factory=lambda: [0])
    preset: str = "screened"
    threads: int = os.cpu_count() or 1
    timeout: float = 600.0
    limit: float | None = None              # a queue's time limit for one detector run (min with timeout; not identity)
    output_dir: str = str(RUNS_ROOT / "exp")  # a durable run writes to OUTPUT_DIR/NAME
    cache_dir: str = "~/.cache/hedonic"
    network_root: str | None = None
    omega: bool = True
    json: bool = False
    order: str = "network,seed,method"      # loop nesting, outermost first
    save_covers: bool = False               # keep every detector cover (covers/<net>/<method>.json.gz)
    isolate: bool = False                   # each run in a fresh child process: per-run peak RSS
    profile: str | None = None              # named config this run came from (informational)
    # Per-network method settings over the preset: {method key: {network: {param: value}}}.
    method_params: dict = field(default_factory=dict)
    tuned: str = ""                         # calibration file (hoc_tuned.json) laid over the HOC rows of method_params
    robustness: bool = False                # audit every saved cover (robustness/cover_audit.jsonl): needs covers, full graphs
    induce: str = "full"                    # "covered": every method sees only the subgraph induced by the cover's vertices
    replication: str = "auto"               # "auto": a deterministic method runs once (lowest seed); "all": every seed
    shared_folder: bool = False             # output_dir is the run's folder itself, shared with other runs (never /NAME)

    def command(self) -> str:
        allowed = {"output_dir", "cache_dir", "network_root", "json", "shared_folder", "limit"} | (
            {"tuned", "method_params"} if self.tuned else set())
        if self.profile in PROFILES and PROFILES[self.profile].differences(self) <= allowed:
            parts = ["hedonic run exp", "--config", self.profile]
            if self.tuned:
                parts += ["--tuned", self.tuned]
            if self.output_dir != PROFILES[self.profile].output_dir:
                parts += ["--output-dir", self.output_dir]
            if self.shared_folder:
                parts.append("--shared-folder")
            if self.limit is not None:
                parts += ["--limit-seconds", f"{self.limit:g}"]
            if self.cache_dir != "~/.cache/hedonic":
                parts += ["--cache-dir", self.cache_dir]
            if self.network_root:
                parts += ["--network-root", self.network_root]
            return " ".join(shlex.quote(p) if " " in p and p != "hedonic run exp" else p for p in parts)
        public_targets = []
        for target in self.networks:
            base, cover = split_target(target)
            public_targets.append(f"{base}:{snap.cover_variant_label(cover)}" if cover else base)
        parts = ["hedonic run exp", "--networks", ",".join(public_targets), "--methods", ",".join(self.methods),
                 "--nodes", str(self.nodes), "--seeds", ",".join(map(str, self.seeds)), "--preset", self.preset,
                 "--threads", str(self.threads), "--timeout", f"{self.timeout:g}", "--output-dir", self.output_dir]
        if self.cache_dir != "~/.cache/hedonic":
            parts += ["--cache-dir", self.cache_dir]
        if self.network_root:
            parts += ["--network-root", self.network_root]
        if not self.omega:
            parts.append("--no-omega")
        if self.order != "network,seed,method":
            parts += ["--order", self.order]
        if self.save_covers:
            parts.append("--save-covers")
        if self.isolate:
            parts.append("--isolate")
        if self.tuned:
            parts += ["--tuned", self.tuned]
        if self.robustness:
            parts.append("--robustness")
        if self.induce != "full":
            parts += ["--induce", self.induce]
        if self.replication != "auto":
            parts += ["--replication", self.replication]
        if self.shared_folder:
            parts.append("--shared-folder")
        if self.limit is not None:
            parts += ["--limit-seconds", f"{self.limit:g}"]
        return " ".join(shlex.quote(p) if " " in p and p != "hedonic run exp" else p for p in parts)

    def pretty_command(self) -> str:
        """The same command broken at option boundaries, for narrow terminals (shell line continuations)."""
        words = self.command().split(" --")
        return " \\\n      --".join([words[0], *words[1:]])

    def differences(self, other: "Config") -> set[str]:
        return {k for k in self.__dict__ if k != "profile" and getattr(self, k) != getattr(other, k)}


# Named configurations (``--config NAME``). "paper" is the paper's reproducible
# experiment: the two smallest SNAP networks (DBLP, Amazon) as full graphs, all
# nine screened methods at their screened settings, seeds 0-9, run seed -> method
# -> network (seed 0: method 1 on DBLP then Amazon; method 2 on both; ...; then
# seed 1, ...), one run at a time. YouTube is excluded (ANGEL did not finish on
# full YouTube within an hour). The per-run timeout is one hour. Both paper profiles keep every cover and
# audit it for robustness right after its detector run (the third axis of Chapter 6, next to accuracy and time).
PROFILES: dict[str, Config] = {}
PROFILES["default"] = Config()
PROFILES["paper"] = Config(networks=["dblp", "amazon"],
                           methods=["hedonic", "codeseg", "fox", "clique_percolation", "angel", "bigclam",
                                    "ego_splitting", "ncgame", "neo_kmeans"],
                           nodes=0, seeds=list(range(10)), preset="screened", timeout=3600.0,
                           order="seed,method,network", output_dir=str(RUNS_ROOT / "paper"), profile="paper",
                           save_covers=True, robustness=True)
# "paper-hoc" reruns only the Hedonic rows of the whole-graph profile: DBLP and Amazon full graphs, seeds 0-9, at the
# whole-graph settings the first calibration's bisection search selected per network (HOC_TUNED, frozen data; a run
# with --tuned takes a newer calibration instead). Covers are kept so `hedonic run check` can rerun a sample and
# compare them exactly.
HOC_TUNED = {
    "hedonic": {"dblp": {"max_memberships": 32, "absolute_resolution": 1 / 7},
                  "amazon": {"max_memberships": 4, "absolute_resolution": 1 / 723}},
}
PROFILES["paper-hoc"] = Config(networks=["dblp", "amazon"], methods=["hedonic"],
                               nodes=0, seeds=list(range(10)), preset="screened", timeout=3600.0,
                               order="seed,method,network", output_dir=str(RUNS_ROOT / "paper-hoc"), profile="paper-hoc",
                               save_covers=True, robustness=True, method_params=HOC_TUNED)


HOC_METHODS = ("hedonic",)


class TunedMissing(ValueError):
    """The calibration file does not exist yet, or its calibration has not finished."""


def load_tuned_file(path: str | Path) -> dict:
    """The calibrated settings of ``hedonic run calibrate`` or ``calibrate-baselines``: {method: {network: params}}.

    A HOC row is {max_memberships, absolute_resolution}; a baseline's row is the ``method_params`` the runner reads for it
    (``baseline_tuned.json`` writes exactly those). Only a finished calibration is accepted, so a run never starts on half
    of a calibration. A missing or unfinished file raises :class:`TunedMissing`, which a queue treats as "not calibrated
    yet" rather than as an error.
    """
    file = expand_path(path)
    if not file.is_file():
        raise TunedMissing(f"no calibration file {file}; run `hedonic run calibrate` or `calibrate-baselines` first")
    try:
        data = json.loads(file.read_text())
    except ValueError as exc:
        raise ValueError(f"{file} is not valid JSON: {exc}") from None
    if not data.get("complete"):
        raise TunedMissing(f"the calibration in {file} has not finished")
    out: dict = {}
    for method, networks in (data.get("settings") or {}).items():
        method = ALIASES.get(method, method)  # files written before the rename call the method hoc_local
        if method == "hoc_multilevel":  # a 1.x file may still carry rows of the removed multilevel phase: not a method
            continue
        if method not in BY_KEY:
            raise ValueError(f"{file}: unknown method {method!r}")
        for network, values in networks.items():
            if method in HOC_METHODS:
                out.setdefault(method, {})[network] = {"max_memberships": int(values["max_memberships"]),
                                                       "absolute_resolution": float(values["absolute_resolution"])}
            elif values:  # a baseline: the parameters the runner reads, as the calibration wrote them
                out.setdefault(method, {})[network] = dict(values)
    return out


def tuned_files(value: str | Path | Sequence[str | Path]) -> list[str]:
    """The calibration files named by ``--tuned``: one path, a comma list, or several ``--tuned`` flags."""
    parts = [value] if isinstance(value, (str, Path)) else list(value)
    return [piece.strip() for part in parts for piece in str(part).split(",") if piece.strip()]


def apply_tuned(cfg: "Config", path: str | Path | Sequence[str | Path]) -> None:
    """Lay one or more calibrations over ``cfg.method_params`` (for the methods and networks they cover).

    Files are applied in the order given; a later file overrides an earlier one for the same method and network."""
    files = tuned_files(path)
    merged = {method: {net: dict(values) for net, values in nets.items()}
              for method, nets in (cfg.method_params or {}).items()}
    for file in files:
        for method, nets in load_tuned_file(file).items():
            if method in cfg.methods:
                for network, values in nets.items():
                    merged.setdefault(method, {}).setdefault(network, {}).update(values)
    cfg.method_params = merged
    cfg.tuned = ",".join(str(expand_path(file).resolve()) for file in files)


# "paper-oracle": the supervised disjoint control of Chapter 6 on the paper's networks. The oracle is deterministic,
# so one seed is the whole result; its cover is kept and audited like the detectors' covers.
PROFILES["paper-oracle"] = Config(networks=["dblp", "amazon"], methods=["disjoint_oracle"], nodes=0, seeds=[0],
                                  preset="screened", timeout=3600.0, order="seed,method,network",
                                  output_dir=str(RUNS_ROOT / "paper-oracle"), profile="paper-oracle", save_covers=True,
                                  robustness=True)
PROFILE_HELP = {"default": "DBLP, 5,000-vertex subgraph, seed 0, Hedonic + CoDeSEG (a minute)",
                "paper": "paper reproduction: DBLP, Amazon (full graphs) x all 9 methods x seeds 0-9, "
                         "order seed -> method -> network (many hours)",
                "paper-hoc": "the paper's Hedonic rows at their tuned per-network settings, DBLP and Amazon "
                             "(full graphs), seeds 0-9, covers kept",
                "paper-oracle": "supervised control: the best disjoint answer built from the ground truth, DBLP and "
                                "Amazon (full graphs), deterministic (seed 0), cover kept and audited"}
ORDER_KEYS = ("network", "seed", "method")


# --------------------------------------------------------------------------- parsing
def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    try:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                lo, hi = part.split("-", 1)
                if int(hi) < int(lo):
                    raise ValueError
                seeds.extend(range(int(lo), int(hi) + 1))
            else:
                seeds.append(int(part))
    except ValueError:
        raise ValueError(f"invalid seeds {value!r}; use e.g. 0, 0-4 or 0,3,7") from None
    if not seeds:
        raise ValueError("no seeds given")
    if min(seeds) < 0:
        raise ValueError(f"invalid seeds {value!r}; seeds are non-negative integers")
    return sorted(dict.fromkeys(seeds))


def seeds_text(seeds: list[int]) -> str:
    """``seed 0`` or ``seeds 0,1,2``."""
    return ("seed " if len(seeds) == 1 else "seeds ") + ",".join(map(str, seeds))


def parse_name(value: str) -> str:
    """Run names become directory and tmux session names, so keep them plain."""
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError(f"invalid run name {value!r}; use letters, digits, '-' and '_' (at most 64, no dots or spaces)")
    return value


def parse_choice(value: str, universe: list[str], aliases: dict, what: str, extra: "tuple[str, ...] | list[str]" = ()) -> list[str]:
    """Names from ``universe`` (``all`` = every one of them) or from ``extra``, which ``all`` leaves out."""
    if value.strip().lower() == "all":
        return list(universe)
    chosen = []
    for part in value.split(","):
        key = aliases.get(part.strip().lower(), part.strip().lower())
        if key in REMOVED_METHODS:
            raise ValueError(f"{what} {part!r} is not available in hedonic 2.0: {REMOVED_METHODS[key]}")
        if key not in universe and key not in extra:
            raise ValueError(f"unknown {what} {part!r}; choose from: {', '.join([*universe, *extra])} (or all)")
        chosen.append(key)
    return list(dict.fromkeys(chosen))


# 1.x method names that 2.0 refuses with the reason (never a silent fallback).
REMOVED_METHODS = {name: "the multilevel phase was removed; Hedonic runs local moving only (method hedonic)"
                   for name in ("hoc_multilevel", "hedonic_multilevel", "hedonic_multiphase", "hedonic_multiphase_x10",
                                "hedonic_multiphase_x100", "multilevel")}
NETWORK_ALIASES = {"wiki": "wikipedia", "lj": "livejournal"}


def split_target(target: str) -> tuple[str, str | None]:
    """``dblp:top5k`` -> ("dblp", SNAP's stable ``top5000`` identifier); plain names have no cover."""
    base, _, cover = str(target).partition(":")
    return base, (snap.canonical_cover_variant(cover) if cover else None)


def parse_targets(value: str) -> list[str]:
    """Networks, each optionally with the ground-truth cover to score against: ``dblp,amazon:top5k``.

    A name without a cover keeps the benchmark's default (the ``all`` cover, named nowhere). ``all`` is every network.
    A cover must be one the network has (``snap.SPECS``): Wikipedia only has its categories, the others ``all`` and ``top5k``.
    """
    if value.strip().lower() == "all":
        return list(NETWORKS)
    from hedonic.experiments.overlapping import snap

    chosen = []
    for part in value.split(","):
        base, cover = split_target(part.strip().lower())
        base = parse_choice(NETWORK_ALIASES.get(base, base), list(NETWORKS), {}, "network")[0]
        if cover is not None and cover not in snap.SPECS[base].cover_files:
            available = ", ".join(snap.cover_variant_label(item) for item in snap.SPECS[base].cover_files)
            raise ValueError(f"{NETWORKS[base][0]} has no cover {snap.cover_variant_label(cover)!r}; its covers: {available}")
        chosen.append(base if cover is None else f"{base}:{cover}")
    return list(dict.fromkeys(chosen))


def expand_covers(networks: list[str], covers: "str | list[str]") -> list[str]:
    """Cross networks with covers, network by network; ``top5k`` normalizes to SNAP's stable ID."""
    names = [c.strip() for c in (covers.split(",") if isinstance(covers, str) else covers) if c.strip()]
    if not names:
        raise ValueError("covers is empty")
    return parse_targets(",".join(n if ":" in n else f"{n}:{c}" for n in networks for c in names))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hedonic run exp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Overlapping community detection benchmark on SNAP ground-truth networks.\n"
                    "Without arguments (in a terminal) an interactive wizard starts.",
        epilog="networks: " + ", ".join(NETWORKS) + "\nmethods:  " + ", ".join(BY_KEY) +
               "\n\nexamples:\n  hedonic run exp --yes\n  hedonic run exp --networks dblp,amazon --methods all --seeds 0-2\n"
               "  hedonic run exp --networks youtube --nodes 0 --methods hedonic,fox\n  hedonic run exp --list")
    p.add_argument("-i", "--interactive", action="store_true", help="open the wizard (default when no arguments)")
    p.add_argument("-y", "--yes", action="store_true", help="run with defaults/flags, never prompt")
    p.add_argument("--networks", "--datasets", default=None, help="comma list or 'all' (default: dblp)")
    p.add_argument("--methods", default=None, help="comma list or 'all' (default: hedonic,codeseg)")
    p.add_argument("--covers", default=None, metavar="LIST",
                   help="ground-truth covers to score against, e.g. all,top5k: every network is run once per cover, network "
                        "by network (a network can also name its own, as in --networks dblp:all,dblp:top5k,amazon)")
    p.add_argument("--induce", choices=("full", "covered"), default=None,
                   help="covered: every method detects on the subgraph induced by the vertices of the cover (default full: "
                        "the whole graph, answers filtered to the cover's vertices when scoring)")
    p.add_argument("--nodes", type=int, default=None, help="vertices of the induced subgraph; 0 = full graph (default 5000)")
    p.add_argument("--full", action="store_true", help="shorthand for --nodes 0")
    p.add_argument("--seeds", default=None,
                   help="e.g. 0 | 0-4 | 0,3,7 (default 0); a deterministic method runs only at the lowest (see --replication)")
    once = ", ".join(m.label.split(" (")[0] for m in METHODS + CONTROLS if replication.kind(m.key) == "deterministic")
    p.add_argument("--replication", choices=replication.MODES, default=None,
                   help=f"auto (default): a deterministic method ({once}) runs once per network, at the lowest seed, since "
                        "more seeds repeat its cover; the others run at every seed. all: every method at every seed (an "
                        "audit of determinism). Both modes share their records")
    p.add_argument("--preset", choices=("screened", "registry", "paper"), default=None,
                   help="screened = most accurate setting within 10 min on full DBLP; registry = registry/paper "
                        "defaults ('paper' is accepted as an old spelling)")
    p.add_argument("--threads", type=int, default=None, help="threads for methods that support them (default: all cores)")
    p.add_argument("--timeout", type=float, default=None, help="per-run detection timeout in seconds (default 600)")
    p.add_argument("--limit-seconds", type=float, default=None, metavar="S",
                   help="a queue's time limit for one detector run: lowers --timeout without changing any record's "
                        "identity; a run it stops is recorded as a timeout and reruns under a larger limit or none")
    p.add_argument("--output-dir", default=None, help="where run records and the summary are saved (default ~/Databases/Hedonic/Overlap/exp; a durable run adds /NAME)")
    p.add_argument("--cache-dir", default=None, help="downloads and built runtimes (default ~/.cache/hedonic)")
    p.add_argument("--network-root", default=None, help="optional folder with local SNAP archives")
    p.add_argument("--no-omega", action="store_true", help="skip the sampled Omega index (faster on big graphs)")
    p.add_argument("--json", action="store_true", help="also print the summary as JSON")
    p.add_argument("--list", action="store_true", help="list networks and methods, then exit")
    p.add_argument("-c", "--config", default=None,
                   help="named config (" + ", ".join(PROFILES) + ") or a .json/.toml file; flags given too override it")
    p.add_argument("--order", default=None,
                   help="loop nesting, outermost first, e.g. seed,method,network (default network,seed,method); "
                        "a config file sets it with its `order` key")
    p.add_argument("--save-covers", action="store_true", help="keep every detector cover next to its record")
    p.add_argument("--shared-folder", action="store_true",
                   help="--output-dir is the run's folder itself, shared by several runs (each with its own --name and "
                        "--methods), instead of OUTPUT_DIR/NAME: a unit another run finished there is reused")
    p.add_argument("--isolate", action="store_true",
                   help="run each detector run in a fresh child process, so its record's peak RSS is its own")
    p.add_argument("--robustness", action=argparse.BooleanOptionalAction, default=None,
                   help="audit every saved cover for robustness (exact stable-vertex fraction over resolution) right "
                        "after its detector run, into OUTPUT_DIR/NAME/robustness; implies --save-covers, full graphs only")
    p.add_argument("--tuned", action="append", default=None, metavar="FILE[,FILE]",
                   help="calibrated settings laid over the config's method_params: hoc_tuned.json of `hedonic run calibrate` "
                        "(HOC rows) and baseline_tuned.json of `hedonic run calibrate-baselines` (the baselines); "
                        "repeat the flag or give a comma list, a later file overrides an earlier one")
    p.add_argument("--dry-run", action="store_true", help="print the run plan and estimated time, then exit")
    p.add_argument("--scan", action="store_true",
                   help="read-only: check which units of the plan already have a certified record (identity, final status, "
                        "cover, SHA-256 seal) in the run's folder and which still have work, then exit; needs --name")
    p.add_argument("--save-config", default=None, metavar="PATH", help="write the resolved config to a .json file and exit")
    p.add_argument("--list-configs", action="store_true", help="list the named configs, then exit")
    p.add_argument("--confirm", action="store_true", help=argparse.SUPPRESS)  # bare `hedonic run`: ask before starting
    p.add_argument("--name", default=None, help="run name (default exp-YYYYmmdd-HHMMSS); records go to OUTPUT_DIR/NAME")
    p.add_argument("--detach", action="store_true", help="start in the background and return (reattach: hedonic run attach)")
    p.add_argument("--foreground", action="store_true",
                   help="run in this terminal instead of a detached tmux session (not recoverable)")
    return p


def load_config(name_or_path: str | None) -> Config:
    """A named profile, or a .json/.toml file holding Config fields (seeds may be '0-9')."""
    if not name_or_path:
        return Config(**{**PROFILES["default"].__dict__})
    if name_or_path in PROFILES:
        return Config(**{**PROFILES[name_or_path].__dict__, "networks": list(PROFILES[name_or_path].networks),
                         "methods": list(PROFILES[name_or_path].methods), "seeds": list(PROFILES[name_or_path].seeds),
                         "profile": name_or_path})
    from hedonic.experiments.overlapping import userconfig

    user_profiles = userconfig.profiles()
    if name_or_path in user_profiles:
        cfg = _config_from_mapping(user_profiles[name_or_path], f"[profiles.{name_or_path}] in {userconfig.path()}")
        cfg.profile = name_or_path
        return cfg
    from hedonic.experiments import resources

    path = resources.config_path(name_or_path)  # a file as given, else a config shipped with the package
    if not path.is_file():
        names = [*PROFILES, *user_profiles]
        shipped = [p.stem for p in resources.shipped_configs() if p.suffix == ".toml" and "hedonic_run" in p.read_text()]
        raise ValueError(f"unknown config {name_or_path!r}: not one of {', '.join(names)}, not a shipped config "
                         f"({', '.join(shipped) or 'none'}) and not a file")
    if path.suffix == ".toml":
        import tomllib
        data = tomllib.loads(path.read_text())
        data = data.get("hedonic_run", data)
    else:
        data = json.loads(path.read_text())
    return _config_from_mapping(data, str(path))


def _config_from_mapping(data: dict, source: str) -> Config:
    """A Config from a mapping of its fields; a 'config' key names the base profile."""
    base = data.get("config")
    cfg = load_config(base) if base else Config(**{**PROFILES["default"].__dict__, "profile": None})
    data = {k: v for k, v in data.items() if k != "config"}
    covers = data.pop("covers", None)  # not a field: it expands `networks` into network:cover targets
    path = source
    unknown = set(data) - set(Config().__dict__)
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {', '.join(sorted(unknown))}")
    for key, value in data.items():
        if key == "seeds" and isinstance(value, str):
            value = parse_seeds(value)
        if key in ("networks", "methods") and isinstance(value, str):
            value = value.split(",")
        if key == "order":  # "seed,method,network" or ["seed", "method", "network"], outermost loop first
            value = parse_order(value, f"order in {path}")
        if key == "replication":
            try:
                value = replication.check_mode(value)
            except ValueError as exc:
                raise ValueError(f"{exc} (in {path})") from None
        setattr(cfg, key, value)
    if cfg.induce not in ("full", "covered"):
        raise ValueError(f"induce in {path} must be full or covered; got {cfg.induce!r}")
    cfg.networks = parse_targets(",".join(cfg.networks))
    if covers is not None:
        cfg.networks = expand_covers(cfg.networks, covers)
    cfg.methods = parse_choice(",".join(cfg.methods), [m.key for m in METHODS], ALIASES, "method", CONTROL_KEYS)
    return cfg


def parse_order(value: "str | list[str] | tuple[str, ...]", source: str = "--order") -> str:
    """The loop nesting, outermost first: network, seed and method once each, as ``seed,method,network`` or as a list."""
    keys = [str(k).strip() for k in (value if isinstance(value, (list, tuple)) else str(value).split(","))]
    if sorted(keys) != sorted(ORDER_KEYS):
        raise ValueError(f"{source} must name network, seed and method once each, outermost loop first "
                         f"(for example seed,method,network); got {','.join(keys) or 'nothing'}")
    return ",".join(keys)


def config_from_args(a: argparse.Namespace) -> Config:
    from hedonic.experiments.overlapping import userconfig

    cfg = load_config(a.config)
    # Machine settings from ~/.config/hedonic/config.toml. One rule everywhere: flag > environment > file > default,
    # so an environment variable that names a folder wins over the file (as the home screen and the loaders read it).
    overridden = {"network_root": "HEDONIC_NETWORKS_DIR", "output_dir": "HEDONIC_OUTPUT_DIR"}
    for key, value in userconfig.defaults().items():
        if key in userconfig.MACHINE_KEYS and not os.environ.get(overridden.get(key, "")):
            setattr(cfg, key, value)
    if a.networks is not None:
        cfg.networks = parse_targets(a.networks)
    if a.covers is not None:
        cfg.networks = expand_covers(cfg.networks, a.covers)
    if a.induce is not None:
        cfg.induce = a.induce
    if a.methods is not None:
        cfg.methods = parse_choice(a.methods, [m.key for m in METHODS], ALIASES, "method", CONTROL_KEYS)
    if a.full:
        cfg.nodes = 0
    elif a.nodes is not None:
        if a.nodes < 0:
            raise ValueError("--nodes must be 0 (full graph) or a positive number of vertices")
        cfg.nodes = a.nodes
    if a.seeds is not None:
        cfg.seeds = parse_seeds(a.seeds)
    if a.replication is not None:
        cfg.replication = a.replication
    for key in ("preset", "timeout", "output_dir", "cache_dir", "network_root"):
        if getattr(a, key) is not None:
            setattr(cfg, key, getattr(a, key))
    if not cfg.timeout > 0:
        raise ValueError("--timeout must be a positive number of seconds")
    if a.limit_seconds is not None:
        if not a.limit_seconds > 0:
            raise ValueError("--limit-seconds must be a positive number of seconds")
        cfg.limit = float(a.limit_seconds)
    if cfg.preset == "paper":
        cfg.preset = "registry"
    if a.threads is not None:
        if a.threads < 1:
            raise ValueError("--threads must be at least 1")
        cfg.threads = a.threads
    if a.order is not None:
        cfg.order = parse_order(a.order)
    if a.no_omega:
        cfg.omega = False
    if a.save_covers:
        cfg.save_covers = True
    if a.isolate:
        cfg.isolate = True
    if a.shared_folder:
        cfg.shared_folder = True
    if a.robustness is not None:
        cfg.robustness = a.robustness
    if cfg.robustness:
        if cfg.nodes:
            raise ValueError("--robustness audits the complete graphs: use --full (or --nodes 0)")
        cfg.save_covers = True  # the audit reads the saved cover
    if a.tuned:
        apply_tuned(cfg, a.tuned)
    cfg.json = cfg.json or a.json
    return cfg


# --------------------------------------------------------------------------- wizard
class SwitchToSpectrum(Exception):
    """The wizard's start menu chose the ground-truth robustness spectrum instead of the benchmark."""


def _check(fn, value):
    try:
        fn(value)
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    return None


def _positive_seconds(value: str) -> str | None:
    try:
        ok = float(value) > 0
    except ValueError:
        ok = False
    return None if ok else "enter a positive number of seconds"


def confirm_start(question: str = "Start it?") -> bool:
    """The wizards' last question before a run starts (:func:`tui.confirm`)."""
    return tui.confirm(question)


def wizard(cfg: Config) -> Config:
    print(style("\n  hedonic · overlapping community benchmark\n", "bold", "magenta"))
    default = (f"{NETWORKS[cfg.networks[0]][0]} · {cfg.nodes:,} vertices · seed {cfg.seeds[0]} · "
               + " + ".join(BY_KEY[m].label for m in cfg.methods))
    start = tui.select("What would you like to do?",
                       [("Run the default experiment", default),
                        ("Reproduce the paper experiment", PROFILE_HELP["paper"]),
                        ("Customize the experiment", "step by step"),
                        ("Ground-truth robustness spectrum", "how stable are the supplied SNAP covers across resolution?"),
                        ("Quit", "")])
    if start == 4:
        raise tui.Cancelled
    if start == 3:
        raise SwitchToSpectrum
    if start == 0:  # the quick show-off run: say what it is and ask once
        print(render_plan(cfg).split("\n\n")[0])
        if confirm_start():
            return cfg
        return wizard(cfg)
    if start == 1:
        paper = load_config("paper")
        print(render_plan(paper))
        choice = tui.select("Start the paper reproduction?",
                            [("Run", "start it now"), ("Back", "to the first menu"), ("Quit", "nothing starts")],
                            default=1)
        if choice == 0:
            return paper
        if choice == 2:
            raise tui.Cancelled
        return wizard(cfg)
    while True:
        names = list(NETWORKS)
        idx = tui.multiselect("Networks (SNAP, with ground truth)",
                              [(NETWORKS[n][0], f"{NETWORKS[n][1]:,} vertices · {NETWORKS[n][2]:,} edges") for n in names],
                              [names.index(n) for n in cfg.networks])
        cfg.networks = [names[i] for i in idx]
        idx = tui.multiselect("Methods (passed the staged screening)",
                              [(m.label, f"full DBLP: F1 {m.f1:.3f} · ONMI {m.onmi:.3f} · ~{m.seconds:.0f} s") for m in METHODS],
                              [i for i, m in enumerate(METHODS) if m.key in cfg.methods])
        cfg.methods = [METHODS[i].key for i in idx] + [k for k in cfg.methods if k in CONTROL_KEYS]
        sizes = [1000, 5000, 20000, 100000, 0, -1]
        labels = [("1,000 vertices", "seconds"), ("5,000 vertices", "default"), ("20,000 vertices", ""),
                  ("100,000 vertices", ""), ("Full graph", "slow on the large networks"), ("Custom…", "")]
        i = tui.select("Graph size (deterministic induced subgraph built around the labelled communities)", labels,
                       sizes.index(cfg.nodes) if cfg.nodes in sizes else 1)
        cfg.nodes = sizes[i] if sizes[i] >= 0 else int(tui.text(
            "Number of vertices", "5000",
            lambda v: None if v.isdigit() and int(v) > 0 else "enter a positive whole number (Full graph is its own choice)"))
        seed_opts = [("1 seed", "0"), ("3 seeds", "0-2"), ("5 seeds", "0-4"), ("10 seeds", "0-9"), ("Custom…", "")]
        i = tui.select("Seeds", seed_opts, 0)
        spec = seed_opts[i][1] or tui.text("Seeds (e.g. 0-4 or 0,3,7)", "0", lambda v: _check(parse_seeds, v))
        cfg.seeds = parse_seeds(spec)
        i = tui.select("Method settings", [("Screened", "each method's most accurate setting within 10 min on full DBLP"),
                                           ("Registry defaults", "the defaults of the original implementations")],
                       0 if cfg.preset == "screened" else 1)
        cfg.preset = ("screened", "registry")[i]
        i = tui.select("Omega index", [("Compute it (sampled)", "100,000 vertex pairs; default"),
                                       ("Skip it", "faster on big graphs, same as --no-omega")], 0 if cfg.omega else 1)
        cfg.omega = i == 0
        cfg.threads = int(tui.text("Threads for multi-threaded methods", str(cfg.threads),
                                   lambda v: None if v.isdigit() and int(v) > 0 else "enter a positive integer"))
        cfg.timeout = float(tui.text("Per-run timeout [s]", f"{cfg.timeout:g}", _positive_seconds))
        cfg.output_dir = tui.text("Save runs and results in", cfg.output_dir)
        cfg.cache_dir = tui.text("Cache for downloads and built tools", cfg.cache_dir)
        runs = len(build_plan(cfg, {}))
        size = "full graph" if not cfg.nodes else f"{cfg.nodes:,} vertices"
        print()
        print(style("  Summary", "bold"))
        print(f"  networks  {', '.join(NETWORKS[n][0] for n in cfg.networks)}")
        print(f"  methods   {', '.join(BY_KEY[m].label for m in cfg.methods)}")
        print(f"  size      {size} · {seeds_text(cfg.seeds)} · {runs} runs · preset {cfg.preset}"
              + ("" if cfg.omega else " · no Omega"))
        estimate = sum(expected_seconds(m, cfg, net) for net, _, m in build_plan(cfg, {}))
        print(f"  estimate  ~{fmt_duration(estimate)} on an M1-Max-class machine (rough; refined live while running)")
        print(f"  saving to {expand_path(cfg.output_dir).resolve()}/<run name>")
        print(style("  same run without the wizard:", "dim"))
        print(style("    " + cfg.pretty_command(), "dim"))
        if "friendster" in cfg.networks and not cfg.nodes:
            print(style("  warning: full Friendster has 1.8 billion edges (tens of GB to download and load)", "yellow"))
        choice = tui.select("Start?", [("Run", "start it now"), ("Change settings", "ask again, keeping these"),
                                       ("Quit", "nothing starts")], default=1)
        if choice == 0:
            return cfg
        if choice == 2:
            raise tui.Cancelled


# --------------------------------------------------------------------------- provisioning
def _compiler() -> str | None:
    for name in (os.environ.get("CXX"), "c++", "clang++", "g++"):
        if name and shutil.which(name):
            return shutil.which(name)
    return None


def codeseg_binary(tools: Path) -> Path:
    """Where ensure_codeseg keeps the compiled CoDeSEG binary, for the CoDeSEG folder ``tools`` of the cache."""
    return tools / "quickstart" / f"CoDeSEG-{CODESEG_COMMIT[:12]}"


def ensure_codeseg(cache: Path, say) -> tuple[Path | None, str | None]:
    """Build CoDeSEG once from its pinned source; return (binary, None) or (None, reason)."""
    binary = codeseg_binary(cache)
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary, None
    cxx = _compiler()
    if cxx is None:
        return None, "no C++ compiler found (set CXX or install clang/g++)"
    root = cache / "quickstart" / f"CoDeSEG-{CODESEG_COMMIT[:12]}-src"
    missing = [f for f in CODESEG_SOURCES + CODESEG_HEADERS if not (root / f).is_file()]
    if missing:
        say(f"downloading CoDeSEG source ({CODESEG_COMMIT[:12]})")
        try:
            for name in missing:
                with urllib.request.urlopen(CODESEG_RAW + name, timeout=60) as response:
                    payload = response.read()
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes(payload)
        except Exception as exc:  # noqa: BLE001
            return None, f"could not download CoDeSEG source: {exc}"
    say(f"compiling CoDeSEG with {Path(cxx).name}")
    binary.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([cxx, "-std=c++17", "-O3", "-pthread", f"-I{root / 'lib'}",
                             *(str(root / f) for f in CODESEG_SOURCES), "-o", str(binary)],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None, "CoDeSEG failed to compile: " + " | ".join((result.stderr or result.stdout).strip().splitlines()[-3:])
    return binary, None


def ensure_runtimes(methods: list[str], cache: Path, say) -> dict[str, str]:
    """Prepare external runtimes through `hedonic setup`; return {method: reason} for unavailable ones."""
    from hedonic.experiments.overlapping import codeseg_setup

    manifest_path = codeseg_setup._manifest_default_path(cache)  # where setup writes it (HEDONIC_CODESEG_MANIFEST, else here)

    def statuses() -> dict:
        try:
            return json.loads(manifest_path.read_text()).get("methods", {})
        except (OSError, ValueError):
            return {}

    needed = sorted({BY_KEY[m].setup for m in methods if BY_KEY[m].setup})
    todo = [s for s in needed if (statuses().get(s) or {}).get("status") != "ready"]
    if todo:
        say(f"preparing runtimes: {', '.join(todo)} (first use only; this can take a few minutes)")
        # ``cache`` is the runtime folder itself; setup's ``--cache-dir DIR`` would mean DIR/codeseg, so it is named
        # through the environment, which setup takes as the folder (the manifest went to cache/codeseg/ before)
        saved = os.environ.get("HEDONIC_CODESEG_CACHE")
        os.environ["HEDONIC_CODESEG_CACHE"] = str(cache)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                codeseg_setup.setup_main(["--datasets", "dblp", "--methods", ",".join(todo), "--build-native",
                                          "--skip-data"])
            except SystemExit:
                pass
            finally:
                if saved is None:
                    os.environ.pop("HEDONIC_CODESEG_CACHE", None)
                else:
                    os.environ["HEDONIC_CODESEG_CACHE"] = saved
    status = statuses()
    missing = {}
    for m in methods:
        need = BY_KEY[m].setup
        if need and (status.get(need) or {}).get("status") != "ready":
            missing[m] = str((status.get(need) or {}).get("reason") or f"{need} runtime could not be prepared")
    return missing


# --------------------------------------------------------------------------- running
def method_params(m: Method, cfg: Config, network: str) -> dict:
    params = dict(m.screened) if cfg.preset == "screened" else {}  # "registry": the runner's defaults
    fraction = min(1.0, cfg.nodes / NETWORKS[network][1]) if cfg.nodes else 1.0
    for key in m.scaled:  # community counts tuned on full DBLP: scale with the subgraph
        if key in params:
            params[key] = max(2, int(round(params[key] * fraction)))
    if m.threads:
        params["threads"] = cfg.threads
    per_network = cfg.method_params.get(m.key) or {}
    params.update(per_network.get(split_target(network)[0]) or {})   # a network's settings apply to all its covers ...
    params.update(per_network.get(network) or {})                    # ... and a canonical network:cover target may override them
    return params


# Detection times measured on full Amazon/YouTube in saved records (M1 Max; overnight queue and
# snap-v4 baselines, 2026-09-26/27). They override the linear-in-edges extrapolation from full
# DBLP, which underestimates super-linear methods (HOC-local: 1,484 s measured on YouTube vs
# ~115 s extrapolated). ANGEL on full YouTube ran for over an hour without finishing.
MEASURED_FULL = {("hedonic", "amazon"): 355, ("disjoint_oracle", "amazon"): 40, ("hedonic", "youtube"): 1484,
                 ("codeseg", "amazon"): 4, ("codeseg", "youtube"): 8,
                 ("fox", "amazon"): 12, ("fox", "youtube"): 141,
                 ("angel", "amazon"): 57, ("angel", "youtube"): float("inf")}


def expected_seconds(m: Method, cfg: Config, network: str) -> float:
    n, e = NETWORKS[network][1:]
    # whole-graph figures: the plan of a covered-induced run is smaller than this, so its estimate is an upper bound
    if not cfg.nodes and (m.key, split_target(network)[0]) in MEASURED_FULL:
        return 2.0 + min(run_timeout(cfg), MEASURED_FULL[m.key, split_target(network)[0]])
    edges = e * min(1.0, cfg.nodes / n) if cfg.nodes else e
    return 2.0 + min(run_timeout(cfg), m.seconds * edges / DBLP_EDGES)


def run_timeout(cfg: Config) -> float:
    """The longest one detector run may take: the run's timeout, lowered by a queue's limit."""
    return cfg.timeout if cfg.limit is None else min(cfg.timeout, cfg.limit)


def graph_text(cfg: Config) -> str:
    """What graph the methods see, for plan and result headings."""
    size = "full graph" if not cfg.nodes else f"{cfg.nodes:,}-vertex subgraph"
    return size + (", covered vertices only" if cfg.induce == "covered" else "")


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rest = divmod(seconds, 3600)
    return f"{h}:{rest // 60:02d}:{rest % 60:02d}" if h else f"{rest // 60}:{rest % 60:02d}"


def fmt_duration(seconds: float) -> str:
    """A length of time that cannot be read as a clock time: ``45 s``, ``12 min``, ``19 h 42 min``, ``3 d 4 h``."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours} h {minutes:02d} min" if minutes else f"{hours} h"
    days, hours = divmod(hours, 24)
    return f"{days} d {hours} h" if hours else f"{days} d"


class Progress:
    """A live status line: spinner, current run, its elapsed time, total elapsed, calibrated ETA."""

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, expected: list[float]):
        self.expected, self.actual, self.started = expected, [], time.time()
        self.current, self.run_started, self._stop = "", time.time(), threading.Event()
        self.live = sys.stdout.isatty()
        self._out = sys.stdout
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def eta(self) -> float:
        done = len(self.actual)
        factor = (sum(self.actual) / max(1e-9, sum(self.expected[:done]))) if done else 1.0
        remaining = sum(self.expected[done:]) * factor - (time.time() - self.run_started if self.current else 0)
        return max(0.0, remaining)

    def _line(self, frame: str) -> str:
        return (f"  {style(frame, 'cyan')} {self.current}  {style(fmt_time(time.time() - self.run_started), 'bold')}"
                + style(f"  · total {fmt_time(time.time() - self.started)} · ETA ~{fmt_time(self.eta())}", "dim"))

    def _loop(self):
        i = 0
        while not self._stop.wait(0.12):
            if self.live and self.current:
                self._out.write("\r\033[K" + self._line(self.FRAMES[i % len(self.FRAMES)]))
                self._out.flush()
            i += 1

    def start(self, text: str):
        self.current, self.run_started = text, time.time()
        if not self.live:
            self._out.write(f"  … {text} (ETA ~{fmt_time(self.eta())})\n")
            self._out.flush()

    def finish(self, text: str):
        self.actual.append(time.time() - self.run_started)
        self.current = ""
        if self.live:
            self._out.write("\r\033[K")
        self._out.write(text + "\n")
        self._out.flush()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        if self.live:
            self._out.write("\r\033[K")
            self._out.flush()


def codeseg_tools(cfg: Config) -> Path:
    """The external methods' folder, as ``hedonic setup`` resolves it (``config.codeseg_cache_dir``): a cache_dir set
    on this run (``--cache-dir`` or a config file) gives ``cache_dir/codeseg``; otherwise ``HEDONIC_CODESEG_CACHE``,
    else the user config's ``cache_dir/codeseg``, else ``~/.cache/hedonic/codeseg``."""
    from hedonic.experiments.overlapping import userconfig

    inherited = {Config().cache_dir, userconfig.defaults().get("cache_dir")}
    return codeseg_cache_dir(None if cfg.cache_dir in inherited else cfg.cache_dir)


def prepare(cfg: Config, say) -> tuple[Path | None, dict[str, str]]:
    """Provision runtimes; return (CoDeSEG binary or None, {method: unavailable reason})."""
    tools = codeseg_tools(cfg)
    os.environ["HEDONIC_CODESEG_CACHE"] = str(tools)
    from hedonic.experiments.overlapping import extra_methods

    extra_methods.install()
    unavailable: dict[str, str] = {}
    binary = None
    if "codeseg" in cfg.methods:
        binary, reason = ensure_codeseg(tools, say)
        if reason:
            unavailable["codeseg"] = reason
    unavailable.update(ensure_runtimes([m for m in cfg.methods if m != "codeseg"], tools, say))
    for m, reason in unavailable.items():
        say(style(f"{BY_KEY[m].label} unavailable: {reason[:160]}", "yellow"))
    return binary, unavailable


def planned_seeds(cfg: Config, method: str) -> list[int]:
    """The seeds ``method`` runs at in this run: all of them, or the lowest for a deterministic method
    (``replication.planned_seeds`` under ``cfg.replication``)."""
    return replication.planned_seeds(method, cfg.seeds, cfg.replication)


def build_plan(cfg: Config, unavailable: dict[str, str]) -> list[tuple[str, int, Method]]:
    """The (network, seed, method) runs, nested in ``cfg.order`` (outermost first).

    A method runs at the seeds of ``planned_seeds``: under ``replication = "auto"`` a deterministic method runs once per
    network, at the lowest seed, because more seeds would repeat its cover. The plan only leaves units out; a unit's
    arguments, identity and folder never depend on the replication, so both modes share their records.
    """
    import itertools

    axes = {"network": cfg.networks, "seed": cfg.seeds, "method": [m for m in cfg.methods if m not in unavailable]}
    seeds = {m: set(planned_seeds(cfg, m)) for m in axes["method"]}
    names = cfg.order.split(",")
    plan = []
    for combo in itertools.product(*(axes[n] for n in names)):
        point = dict(zip(names, combo))
        if point["seed"] in seeds[point["method"]]:
            plan.append((point["network"], point["seed"], BY_KEY[point["method"]]))
    return plan


def replication_text(cfg: Config) -> str:
    """How many runs each method gets, for plan and result headings ('' when that is one per seed for all)."""
    if len(cfg.seeds) < 2:
        return ""
    once = [BY_KEY[m].label.split(" (")[0] for m in cfg.methods if len(planned_seeds(cfg, m)) == 1]
    if cfg.replication == "all":
        return "replication all: every method at every seed"
    if not once:
        return ""
    return f"deterministic, one run at seed {min(cfg.seeds)}: {', '.join(once)}"


def run_folder(cfg: Config, name: str) -> Path:
    """The folder a run named ``name`` writes into: OUTPUT_DIR/NAME, or OUTPUT_DIR itself when it already ends with the
    name or the run shares it (``--shared-folder``)."""
    folder = expand_path(cfg.output_dir)
    return folder if cfg.shared_folder or str(cfg.output_dir).endswith(name) else folder / name


def scan_command(cfg: Config, name: str | None) -> int:
    """``--scan``: print what the run's folder already holds for every unit of the plan. Reads only; seals nothing."""
    if not name:
        print(style("  --scan needs --name: a run's records live in OUTPUT_DIR/NAME", "yellow"))
        return 2
    cfg.output_dir = str(run_folder(cfg, name))
    plan = build_plan(cfg, {})
    binary = codeseg_binary(codeseg_tools(cfg)) if "codeseg" in cfg.methods else None
    if binary is not None and not binary.is_file():
        binary = None
    stage = None
    if cfg.robustness:
        from hedonic.experiments.overlapping import cover_audit as ca

        class _Ledger:  # the audit ledger as the stage reads it, without opening a stage
            records = ca.read_records(Path(cfg.output_dir) / ca.STAGE_DIR / ca.RESULTS_NAME)

            def audited(self, net, seed, method):
                return all(key in self.records for key in ca.unit_keys(net, seed, method))

        stage = _Ledger()
    scan = scan_plan(cfg, plan, binary, stage=stage)
    counts = summarize_scan(scan)
    print(style(f"  {cfg.output_dir}", "dim"))
    width = max(len(BY_KEY[key].label) for key in cfg.methods)
    for key in cfg.methods:
        for net in cfg.networks:
            row = [e for e in scan if e["method"] == key and e["network"] == net]
            if not row:
                continue
            marks = "".join("✓" if e["done"] else "a" if e["audit_pending"] else "!" if e["state"] == "corrupt" else "·"
                            for e in sorted(row, key=lambda e: e["seed"]))
            print(f"  {BY_KEY[key].label:<{width}}  {NETWORKS[net][0]:<10} {marks}  "
                  f"{sum(e['done'] for e in row)}/{len(row)} certified")
    print(style("  ✓ certified   a audit still to do   ! corrupt (would be moved aside)   · to run", "dim"))
    for e in scan:
        if e["state"] in ("corrupt", "stale") or e.get("warning"):
            print(style(f"  ! {NETWORKS[e['network']][0]} · seed {e['seed']} · {BY_KEY[e['method']].label}: "
                        f"{e.get('warning') or e['reason']}", "yellow"))
    print(f"\n  {counts.get('done', 0)} of {counts['units']} units are certified; {counts['units'] - counts.get('done', 0)} remain")
    return 0


def render_plan(cfg: Config) -> str:
    plan = build_plan(cfg, {})
    total = sum(expected_seconds(m, cfg, net) for net, _, m in plan)
    import textwrap

    width = max(60, shutil.get_terminal_size((100, 24)).columns - 4)

    def wrap(text: str) -> str:
        return textwrap.fill(text, width, initial_indent="  ", subsequent_indent="  ")

    lines = [style(f"  {len(plan)} runs", "bold") + f" · order {cfg.order.replace(',', ' → ')}",
             wrap(describe(cfg)),
             *([wrap(replication_text(cfg))] if replication_text(cfg) else []),
             style(wrap(environment_line()), "dim"),
             style(wrap(f"estimated time ~{fmt_time(total)} on an M1-Max-class machine: a rough guide from the screening "
                        f"and saved timings, refined live while running. Downloads and first-use builds are not "
                        f"included. Timeout {cfg.timeout:g} s per run."), "dim"),
             style(f"  output {expand_path(cfg.output_dir)}", "dim"), ""]
    if cfg.robustness:
        lines.insert(-2, style(wrap("robustness: each saved cover is audited after its run (seconds to minutes per cover; a "
                                    "seed-0 trial of all nine methods on both networks took about 24 minutes of audits); "
                                    "the estimate above is detection only. Ledger: robustness/cover_audit.jsonl."), "dim"))
    for key in cfg.methods:  # per-network settings, when the config sets them, are part of the experiment
        per_network = cfg.method_params.get(key) or {}
        if per_network:
            parts = []
            for net in cfg.networks:
                params = method_params(BY_KEY[key], cfg, net)
                shown = ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())
                parts.append(f"{NETWORKS[net][0]}: {shown}")
            lines.append(f"  settings  {BY_KEY[key].label:<16} " + " · ".join(parts))
    if any(cfg.method_params.get(key) for key in cfg.methods):
        lines.append("")
    for i, (net, seed, m) in enumerate(plan[:12], 1):
        lines.append(f"  {i:>4}. seed {seed} · {m.label:<26} · {NETWORKS[net][0]:<12} ~{fmt_time(expected_seconds(m, cfg, net))}")
    if len(plan) > 12:
        lines.append(style(f"   …  {len(plan) - 12} more; last: seed {plan[-1][1]} · {plan[-1][2].label} · "
                           f"{NETWORKS[plan[-1][0]][0]}", "dim"))
    lines.append(style(f"\n  start it with: {cfg.command()}", "dim"))
    return "\n".join(lines)


def _run_runner_quietly(argv: list[str]) -> None:
    """The runner in an isolated child (``Config.isolate``), output discarded like the in-process path."""
    from hedonic.experiments.overlapping import codeseg_reproduction as cr

    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            cr.main(argv)
        except SystemExit:
            pass


# A record's config digest covers the runner's default parameters for every method, not only the method that ran.
# The 1.0.0.6 runs were stamped with ego_splitting's default {"resolution": 1.0}, which the runner no longer
# carries (ego_splitting is an opt-in extra method). It is passed again so the digests of those records still
# reproduce; no HOC method reads it, so the detector and its results are unchanged.
DIGEST_DEFAULTS = {"ego_splitting": {"resolution": 1.0}}


def unit_argv(cfg: Config, net: str, seed: int, m: Method, binary: Path | None) -> tuple[list[str], Path]:
    """The runner's argv for one detector run, and that run's output folder. The queue's skip check builds the same argv."""
    cache = expand_path(cfg.cache_dir)
    base, cover = split_target(net)  # the dataset the runner loads, and the ground-truth cover it scores against
    tag = base if cover is None else f"{base}-{cover}"
    out = (expand_path(cfg.output_dir) / f"{tag}-{'full' if not cfg.nodes else f'n{cfg.nodes}'}"
           f"{'-covered' if cfg.induce == 'covered' else ''}" / f"seed{seed}")
    argv = ["--datasets", base, "--methods", m.runner, "--seed", str(seed), "--timeout", f"{cfg.timeout:g}",
            "--resume", "--auto-download", "--snap-cache-dir", str(cache / "snap"), "--output-dir", str(out),
            "--method-params", json.dumps({**DIGEST_DEFAULTS, m.runner: method_params(m, cfg, net)})]
    if cover is not None:
        argv += ["--cover", cover]
    if cfg.induce == "covered":
        argv += ["--induce", "covered"]
    if cfg.nodes:
        argv += ["--max-nodes", str(cfg.nodes)]
    if cfg.network_root:
        argv += ["--network-root", str(expand_path(cfg.network_root))]
    if binary is not None:
        argv += ["--codeseg-bin", str(binary)]
    if not cfg.omega:
        argv.append("--no-omega")
    if cfg.save_covers:
        argv.append("--save-covers")
    if cfg.limit is not None:  # policy, not identity: the runner keeps it out of the config digest
        argv += ["--limit-seconds", f"{cfg.limit:g}"]
    return argv, out


def unit_digest(cfg: Config, net: str, seed: int, m: Method, binary: Path | None) -> str | None:
    """The config digest the runner would stamp on this unit's record now (its identity, with no detector work)."""
    from hedonic.experiments.overlapping import codeseg_reproduction as cr, extra_methods, threadio

    extra_methods.install()  # the runner only knows ego_splitting and the controls once they are registered
    argv, _ = unit_argv(cfg, net, seed, m, binary)
    buffer = io.StringIO()
    try:
        with threadio.redirected(stdout=buffer):  # the view certifies units on a thread of its own
            cr.main([*argv, "--identity-only"])
        return json.loads(buffer.getvalue().strip().splitlines()[-1]).get("config_digest")
    except (SystemExit, Exception):  # noqa: BLE001 - an identity that cannot be read is not reused; the runner decides
        return None


def unit_check(cfg: Config, net: str, seed: int, m: Method, binary: Path | None, *, adopt: bool = False,
               digest: str | None = None):
    """Certify one unit from the files on disk (``unit_integrity.Check``): present, final, same settings, hashes sealed.

    ``adopt`` also seals a valid record that has no seal yet (it writes the seal; the plain check writes nothing).
    A digest that cannot be computed makes every unit ``stale``, so the runner decides.
    """
    from hedonic.experiments.overlapping import unit_integrity as ui

    _, out = unit_argv(cfg, net, seed, m, binary)
    digest = digest or unit_digest(cfg, net, seed, m, binary)
    if digest is None:  # without an identity nothing can be certified; no record at all is simply missing
        if not ui.record_path(out, split_target(net)[0], m.runner).is_file():
            return ui.Check("missing", "no record")
        return ui.Check("stale", "the unit's identity could not be computed")
    return (ui.adopt if adopt else ui.check)(out, split_target(net)[0], m.runner, digest, cfg.save_covers, cfg.limit)


def unit_record(cfg: Config, net: str, seed: int, m: Method) -> dict:
    """The record a certified unit already holds, tagged like the one ``run_one`` returns."""
    _, out = unit_argv(cfg, net, seed, m, None)
    rec = json.loads((out / "runs" / split_target(net)[0] / f"{m.runner}.json").read_text())
    rec.update(network=net, method_key=m.key, seed=seed)
    return rec


def quarantine_unit(cfg: Config, net: str, seed: int, m: Method, reason: str) -> list[str]:
    """Move a corrupt unit's files aside (never deleted), so the unit runs again."""
    from hedonic.experiments.overlapping import unit_integrity as ui

    _, out = unit_argv(cfg, net, seed, m, None)
    return ui.quarantine(out, split_target(net)[0], m.runner, reason)


def scan_plan(cfg: Config, plan: list[tuple[str, int, Method]], binary: Path | None, *, adopt: bool = False,
              stage=None, on_unit=None) -> list[dict]:
    """What the disk already holds for every unit of the plan, in plan order, before anything runs.

    Each entry: ``state`` (see ``unit_integrity``), ``reason``, ``done`` (a certified record that needs nothing more),
    ``audit_pending`` (certified, but ``stage`` has no audit of its cover yet). ``adopt`` seals valid unsealed records.
    """
    out = []
    for i, (net, seed, m) in enumerate(plan):
        result = unit_check(cfg, net, seed, m, binary, adopt=adopt)
        audited = stage is None or result.record is None or stage.audited(net, seed, m.key)
        entry = {"network": net, "seed": seed, "method": m.key, "state": result.state, "reason": result.reason,
                 "done": result.ok and audited, "audit_pending": result.ok and not audited}
        if stage is not None and result.ok and audited:  # the audit must describe the cover that is on disk now
            from hedonic.experiments.overlapping import cover_audit as ca
            from hedonic.experiments.overlapping import unit_integrity as ui

            audit = stage.records.get(ca.key_of(net, seed, m.key, "cover")) or {}
            seal = ui.read_seal(unit_argv(cfg, net, seed, m, None)[1], split_target(net)[0], m.runner) or {}
            if audit.get("cover_file_sha256") and seal.get("cover_sha256") and audit["cover_file_sha256"] != seal["cover_sha256"]:
                entry["warning"] = "its robustness audit was made on a different cover file than the one on disk"
        out.append(entry)
        if on_unit is not None:
            on_unit(i, entry)
    return out


def summarize_scan(entries: list[dict]) -> dict:
    """Counts of a scan: units done, and why each of the others still has work."""
    counts: dict[str, int] = {}
    for e in entries:
        key = "done" if e["done"] else "audit_pending" if e["audit_pending"] else e["state"]
        counts[key] = counts.get(key, 0) + 1
    return {"units": len(entries), **counts}


def run_one(cfg: Config, net: str, seed: int, m: Method, binary: Path | None) -> dict:
    """One detector run through the unchanged runner (resumable: a finished record is reused)."""
    from hedonic.experiments.overlapping import codeseg_reproduction as cr

    argv, out = unit_argv(cfg, net, seed, m, binary)
    buffer = io.StringIO()
    if cfg.isolate:  # a forked child: its process peak RSS (recorded by the runner) belongs to this run alone
        import multiprocessing as mp

        child = mp.get_context("fork").Process(target=_run_runner_quietly, args=(argv,))
        child.start()
        child.join()
        if child.exitcode:
            buffer.write(f"isolated run exited with code {child.exitcode}\n")
    else:
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                cr.main(argv)
        except SystemExit:
            pass
        except Exception as exc:  # noqa: BLE001 - keep going with the other runs
            buffer.write(f"{type(exc).__name__}: {exc}\n")
    dataset = split_target(net)[0]
    path = out / "runs" / dataset / f"{m.runner}.json"
    if path.is_file():
        rec = json.loads(path.read_text())
        from hedonic.experiments.overlapping import unit_integrity as ui

        if ui.final(rec, cfg.limit) and ui.check(out, dataset, m.runner, None, cfg.save_covers, cfg.limit).state in ("unsealed", "corrupt"):
            ui.seal(out, dataset, m.runner, by="run")  # certify the bytes the runner just wrote (a reused, sealed record keeps its seal)
    else:
        tail = [line for line in buffer.getvalue().strip().splitlines() if line.strip()]
        rec = {"status": "failed", "reason": tail[-1] if tail else "no record written"}
    rec.update(network=net, method_key=m.key, seed=seed)
    return rec


def open_stage(cfg: Config):
    """The robustness stage of a run (``cover_audit.Stage``), or None when the run does not audit its covers."""
    if not cfg.robustness:
        return None
    from hedonic.experiments.overlapping import cover_audit

    return cover_audit.Stage(cfg.output_dir, cfg.networks, cfg.methods, cfg.seeds, cfg.network_root, cfg.induce,
                             cfg.replication)


def attach_robustness(stage, net: str, seed: int, m: Method, rec: dict) -> bool:
    """Audit one unit's saved cover and put its robustness on ``rec``. True when the audit failed.

    A failed audit never stops the benchmark: the reason goes on the record, the unit stays unaudited in the ledger,
    and a rerun of the same run tries it again.
    """
    try:
        measure = stage.audit(net, seed, m.key)
        failed = False
    except Exception as exc:  # noqa: BLE001 - the audit of one cover must not stop the benchmark
        measure, failed = None, True
        rec["robustness_error"] = f"{type(exc).__name__}: {exc}"
    rec["robustness"] = None if measure is None else measure["log"]
    rec["robustness_linear"] = None if measure is None else measure["linear"]
    return failed


def result_line(tag: str, rec: dict) -> str:
    mm = rec.get("metrics") or {}
    if rec.get("status") == "completed":
        note = f"F1 {mm.get('f1', 0):.3f} · ONMI {mm.get('onmi', 0):.3f} · {rec.get('detection_seconds') or 0:.1f} s"
        if isinstance(rec.get("robustness"), (int, float)):
            note += f" · robust {rec['robustness']:.3f}"
        return f"  {style('✓', 'green')} {tag}  {style(note, 'dim')}"
    why = f"{rec.get('status')}: {str(rec.get('reason'))[:110]}"
    return f"  {style('✗', 'red')} {tag}  {style(why, 'yellow')}"


def finish(cfg: Config, records: list[dict], unavailable: dict[str, str]) -> list[dict]:
    for m in unavailable:
        for net in cfg.networks:
            records.append({"network": net, "method_key": m, "status": "unavailable", "reason": unavailable[m]})
    summary = summarise(records, cfg)
    output = expand_path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps({"command": cfg.command(), "config": cfg.__dict__,
                                                     "environment": environment(), "results": summary},
                                                    indent=2, default=str))
    return summary


def environment() -> dict:
    """Software provenance stored with every run (summary.json, progress.json)."""
    import platform

    from hedonic.experiments.runtime import ALGORITHM_IDENTITY, runtime_versions

    try:  # the loaded native builds and their SHA-256 (a rebuild is a different runtime)
        from hedonic.experiments.disjoint.store import runtime_identity

        native = runtime_identity()
    except Exception:  # noqa: BLE001 - provenance only
        native = {}
    versions = runtime_versions()
    return {"hedonic": versions.get("hedonic"), "hedonic_core_sha256": versions.get("hedonic_core_sha256"),
            "python_igraph": versions.get("python_igraph"), "igraph_c": versions.get("igraph_c"),
            "igraph_extension_sha256": native.get("igraph_extension_sha256"),
            "igraph_extension_path": versions.get("igraph_extension_path"),
            "algorithm_identity": ALGORITHM_IDENTITY,
            "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "executable": sys.executable}


def environment_line() -> str:
    env = environment()
    return (f"hedonic {env['hedonic']} · python-igraph {env['python_igraph']} · Python {env['python']} · "
            f"{env['machine']}")


def describe(cfg: Config) -> str:
    size = graph_text(cfg)
    return (f"{', '.join(NETWORKS[n][0] for n in cfg.networks)} · {size} · {seeds_text(cfg.seeds)}"
            f" · preset {cfg.preset} · {cfg.threads} threads" + (" · + robustness" if cfg.robustness else "")
            + (" · replication all" if cfg.replication == "all" else ""))


# --------------------------------------------------------------------------- results
def summarise(records: list[dict], cfg: Config) -> list[dict]:
    rows = []
    for net in cfg.networks:
        for key in cfg.methods:
            recs = [r for r in records if r["network"] == net and r["method_key"] == key]
            done = [r for r in recs if r.get("status") == "completed"]
            planned = planned_seeds(cfg, key)
            row = {"network": net, "method": key, "label": BY_KEY[key].label, "completed": len(done), "runs": len(recs),
                   "replication": replication.kind(key), "planned_runs": len(planned),
                   "status": "completed" if done and len(done) == len(recs) else
                             (recs[-1].get("status") if recs else "not run"),
                   "reason": None if done else (recs[-1].get("reason") if recs else None)}
            for metric in [k for k, _ in COLUMNS] + ["detection_seconds", "robustness"]:
                vals = [(r.get("metrics") or {}).get(metric) if metric not in ("detection_seconds", "robustness")
                        else r.get(metric) for r in done]
                vals = [float(v) for v in vals if isinstance(v, (int, float))]
                row[metric] = statistics.fmean(vals) if vals else None
                row[metric + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else None
            row["sd_note"] = "n/a: one planned run" if len(planned) == 1 else None  # no spread to report, not a missing one
            rows.append(row)
    return rows


def _visible(text: str) -> int:
    import re
    return len(re.sub(r"\033\[[0-9;]*m", "", text))


def render(rows: list[dict], cfg: Config, columns: int | None = None) -> str:
    """The result tables. Multi-seed cells show mean ± sd; if that is wider than the terminal, mean only."""
    columns = columns or shutil.get_terminal_size((200, 24)).columns  # a pipe or log file: no width to respect
    full = _render_tables(rows, cfg, True)
    if len(cfg.seeds) > 1 and max(_visible(line) for line in full.split("\n")) > columns:
        return _render_tables(rows, cfg, False)
    return full


ROBUSTNESS_COLUMN = ("robustness", "Robust.")  # log-averaged stable-vertex share of the saved cover, open labels


def table_columns(cfg: Config) -> tuple[tuple[str, str], ...]:
    """The metric columns of a run's tables: the cover metrics, then robustness when the run audits its covers."""
    return COLUMNS + ((ROBUSTNESS_COLUMN,) if cfg.robustness else ())


def _render_tables(rows: list[dict], cfg: Config, show_sd: bool) -> str:
    out = []
    multi = len(cfg.seeds) > 1
    many = multi and show_sd
    columns_ = table_columns(cfg)
    headers = ["Method", *(label for _, label in columns_), "time [s]", "runs"]
    for net in cfg.networks:
        block = [r for r in rows if r["network"] == net]
        size = graph_text(cfg)
        seeds = (f"mean ± sd over {len(cfg.seeds)} seeds" if many else
                 f"mean over {len(cfg.seeds)} seeds (± sd in summary.json)" if multi else f"seed {cfg.seeds[0]}")
        if multi and any(r.get("planned_runs") == 1 for r in block):
            seeds += f"; a deterministic method: its one run (seed {min(cfg.seeds)})"
        out.append("\n" + style(f"  {NETWORKS[net][0]} · {size} · {seeds}", "bold"))
        best = {k: max((r[k] for r in block if r[k] is not None), default=None) for k, _ in columns_}
        fastest = min((r["detection_seconds"] for r in block if r["detection_seconds"] is not None), default=None)
        body = []
        for r in block:
            if r["completed"] == 0:
                body.append((r, None))
                continue
            cells = [r["label"]]
            for k, _ in columns_:
                v, sd = r[k], r[k + "_sd"]
                text = "–" if v is None else f"{v:.3f}" + (f" ± {sd:.3f}" if many and sd is not None else "")
                cells.append(style(text, "green", "bold") if best[k] is not None and v is not None
                             and abs(v - best[k]) < 5e-4 else text)
            t = r["detection_seconds"]
            tt = "–" if t is None else f"{t:.1f}"
            cells.append(style(tt, "cyan", "bold") if t is not None and t == fastest else tt)
            cells.append(f"{r['completed']}/{r['runs']}")
            body.append((r, cells))
        widths = [len(h) for h in headers]
        for _, cells in body:
            if cells:
                widths = [max(w, _visible(c)) for w, c in zip(widths, cells)]
        inner = sum(widths) + 3 * (len(widths) - 1)

        def row_line(cells):
            parts = []
            for i, (c, w) in enumerate(zip(cells, widths)):
                pad = " " * (w - _visible(c))
                parts.append(f" {c}{pad} " if i == 0 else f" {pad}{c} ")
            return "  │" + "│".join(parts) + "│"

        bar = lambda l, m_, r_: "  " + l + m_.join("─" * (w + 2) for w in widths) + r_  # noqa: E731
        out += [bar("┌", "┬", "┐"), row_line(headers), bar("├", "┼", "┤")]
        for r, cells in body:
            if cells is None:
                note = f"{r['label']}  —  {r['status']}: {str(r['reason'] or '')}"[:inner]
                out.append("  │ " + style(note, "yellow") + " " * (inner - len(note)) + " │")
            else:
                out.append(row_line(cells))
        out.append(bar("└", "┴", "┘"))
    out.append(style("  best per column in green, fastest in cyan · time = detection only", "dim"))
    out.append(style("  F1 sym. = symmetric best-match F1 · ONMI = LFK overlapping NMI · all metrics: hedonic show metrics",
                     "dim"))
    if multi and any(r.get("planned_runs") == 1 for r in rows):
        out.append(style("  runs = completed / planned: a deterministic method runs once (more seeds repeat its cover), "
                         "so it has no ± sd", "dim"))
    if cfg.robustness:
        out.append(style("  Robust. = share of vertices with no profitable move, averaged over log10(resolution) from "
                         "1e-4 to 1 (saved cover, open labels)", "dim"))
    return "\n".join(out)


# --------------------------------------------------------------------------- entry point
def list_catalogue() -> str:
    lines = [style("networks", "bold")]
    lines += [f"  {k:<12} {v[0]:<20} {v[1]:>12,} vertices {v[2]:>15,} edges" for k, v in NETWORKS.items()]
    lines += ["", style("methods", "bold") + style("  (screened setting on full DBLP: F1 / ONMI / detection)", "dim")]
    lines += [f"  {m.key:<15} {m.label:<26} {m.f1:.3f} / {m.onmi:.3f} / ~{m.seconds:.0f} s" for m in METHODS]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        from hedonic.experiments.overlapping import info, userconfig

        cache = expand_path(args.cache_dir or userconfig.defaults().get("cache_dir") or Config().cache_dir)
        network_root = args.network_root or userconfig.defaults().get("network_root")
        print(info.render_networks(info.network_status(cache, network_root, False), False))
        print()
        print(info.render_methods(info.method_status(codeseg_cache_dir(args.cache_dir))))
        return 0
    if args.list_configs:
        from hedonic.experiments.overlapping import userconfig

        for name, text in PROFILE_HELP.items():
            print(f"  {style(name, 'cyan', 'bold'):<20} {text}")
            print(style(f"      hedonic run exp --config {name}", "dim"))
        for name in userconfig.profiles():
            print(f"  {style(name, 'cyan', 'bold'):<20} your config ({userconfig.path()})")
            print(style(f"      hedonic run exp --config {name}", "dim"))
        fav = userconfig.favourite()
        print(style(f"\n  bare `hedonic run` starts: {fav or 'the wizard'}  (hedonic config set run NAME)", "dim"))
        return 0
    if args.preset == "paper":  # an old spelling that reads like the paper's experiment, which it is not
        print(style("  note: --preset sets the methods' settings, and 'paper' is the old spelling of 'registry'; the "
                    "paper's experiment is `hedonic run paper` (= --config paper)", "yellow"), file=sys.stderr)
    try:
        cfg = config_from_args(args)
        if args.name is not None:
            parse_name(args.name)
    except ValueError as exc:  # one line, not the whole usage block
        parser.exit(2, f"hedonic run exp: error: {exc}\n(see `hedonic run exp --help`)\n")
    if args.save_config:
        data = {k: v for k, v in cfg.__dict__.items() if k not in ("profile", "json") and v is not None}
        target = expand_path(args.save_config)
        if target.suffix == ".toml":  # the extension decides the format, so the file loads back
            from hedonic.experiments.overlapping.userconfig import _value

            target.write_text("# hedonic run exp settings — start with: hedonic run exp --config " + target.name + "\n"
                              + "\n".join(f"{k} = {_value(v)}" for k, v in data.items()) + "\n")
        else:
            target.write_text(json.dumps(data, indent=2))
        print(f"  wrote {expand_path(args.save_config)} — run it with: hedonic run exp --config {args.save_config}")
        return 0
    if args.dry_run:
        print(render_plan(cfg))
        return 0
    if args.scan:
        return scan_command(cfg, args.name)
    if args.confirm:
        print(render_plan(cfg).split("\n\n")[0])
        if not args.yes:
            if not tui.interactive():  # never start a long run without an explicit yes
                print(style("  not started: no terminal to confirm in; add --yes to start", "yellow"))
                return 0
            try:
                answer = input(style("\n  start it? [Y/n] ", "bold")).strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer not in ("", "y", "yes"):
                print(style("  not started", "yellow"))
                return 0
        argv = [a for a in argv if a != "--confirm"] or ["--yes"]
    if not argv and not tui.interactive():  # nothing typed and nobody to ask: never start on our own
        print(render_plan(cfg).split("\n\n")[0])
        print(style("  not started: no terminal to confirm in; add --yes (or the run's flags) to start", "yellow"))
        return 0
    wants_wizard = args.interactive or (not argv and tui.interactive())
    try:
        if wants_wizard and not args.yes:
            cfg = wizard(cfg)
        from hedonic.experiments.overlapping import runmanager

        if args.foreground:  # the worker in this terminal, registered like a detached run
            code = runmanager.run_foreground(cfg, args.name)
            summary = expand_path(cfg.output_dir) / "summary.json"
            if cfg.json and summary.is_file():
                print(summary.read_text())
            return code

        entry = runmanager.launch(cfg, args.name)
        where = (f"tmux session {entry['session']}" if entry["backend"] == "tmux"
                 else f"background process (log {entry['log']})")
        info_out = sys.stderr if cfg.json else sys.stdout  # keep stdout clean for --json
        print(style(f"  started {entry['name']}", "green", "bold") + f" in a {where}", file=info_out)
        print(style(f"  it survives closing this terminal · reopen: hedonic run attach {entry['name']} · "
                    f"stop: hedonic run stop {entry['name']}", "dim"), file=info_out)
        if cfg.json and not args.detach:  # scripts: wait for the run, then print the summary as JSON
            return runmanager.wait_and_print_json(entry)
        if args.detach or not sys.stdout.isatty():
            return 0
        time.sleep(0.5)
        return runmanager.view(entry)
    except SwitchToSpectrum:
        from hedonic.experiments.overlapping import gt_spectrum

        return gt_spectrum.front_main(["--interactive"])
    except tui.Cancelled:  # q, Esc or Quit: a choice, not a failure
        print(style("\n  nothing started", "yellow"))
        tui.note_cancelled()
        return 0
    except KeyboardInterrupt:
        print(style("\n  cancelled", "yellow"))
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
