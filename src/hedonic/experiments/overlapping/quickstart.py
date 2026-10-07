"""``hedonic run exp``: a configurable, one-command overlapping-community benchmark.

Run with no arguments in a terminal to open an interactive wizard (arrow keys,
space, enter); its first choice runs the default experiment. Any flag, ``--yes``,
or a non-terminal session runs directly. Examples::

    hedonic run exp                                  # wizard (default run is one keypress away)
    hedonic run exp --yes                            # default: DBLP, 5,000 vertices, seed 0, HOC-local + CoDeSEG
    hedonic run exp --networks dblp,amazon --methods all --seeds 0-2 --nodes 20000
    hedonic run exp --networks youtube --nodes 0 --methods hoc_local,fox,cpm   # full graph

Networks are the seven SNAP ground-truth networks; methods are the ten that
passed the staged screening (manuscript appendix "Staged screening").
``--preset screened`` (default) uses each method's most accurate setting within
ten minutes on full DBLP; ``--preset paper`` uses the registry (paper)
defaults. Detection and scoring go through ``hedonic-exp overlapping-codeseg``
unchanged, so every record has the same loader, bound, paper filter, and
metrics as the benchmark.

Everything is fetched lazily and cached under ``--cache-dir`` (default
``~/.cache/hedonic``): SNAP archives from snap.stanford.edu, CoDeSEG's C++
source pinned to the reproduced commit (compiled with the system compiler; the
repository has no licence file, so it is not bundled), and the other external
runtimes through ``hedonic-exp codeseg-setup``. A method whose runtime cannot be
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
from dataclasses import dataclass, field
from pathlib import Path

from hedonic.experiments.config import expand_path
from hedonic.experiments.overlapping import tui
from hedonic.experiments.overlapping.tui import style

# --------------------------------------------------------------------------- catalogue
NETWORKS = {  # name: (label, vertices, edges) — SNAP ground-truth networks
    "dblp": ("DBLP", 317_080, 1_049_866),
    "amazon": ("Amazon", 334_863, 925_872),
    "youtube": ("YouTube", 1_134_890, 2_987_624),
    "wikipedia": ("Wikipedia (topcats)", 1_791_489, 28_511_807),
    "livejournal": ("LiveJournal", 3_997_962, 34_681_189),
    "orkut": ("Orkut", 3_072_441, 117_185_083),
    "friendster": ("Friendster", 65_608_366, 1_806_067_135),
}
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
    setup: str | None = None      # codeseg-setup method that provides the runtime
    threads: bool = False         # accepts a "threads" parameter
    scaled: tuple = ()            # community-count parameters scaled with graph size


METHODS = [
    # gamma = 7000 x density(full DBLP) = 0.1462, given as an absolute CPM resolution so that it transfers to
    # subgraphs and other networks (a density multiple would grow with the subgraph's higher density).
    Method("hoc_local", "hedonic_local", "HOC-local", {"max_memberships": 64, "absolute_resolution": 0.1462}, 40, .425, .519),
    Method("hoc_multilevel", "hedonic_multiphase", "HOC-multilevel", {}, 63, .171, .392),
    Method("codeseg", "codeseg", "CoDeSEG", {"iterations": 20, "tau": 0.5}, 6, .406, .518, threads=True),
    Method("fox", "fox", "FOX (LazyFox)", {"wcc_threshold": 0.01, "queue_size": 1}, 184, .428, .525, "fox", True),
    Method("cpm", "cpm", "CPM (clique percolation)", {"clique_size": 4}, 9, .419, .524, "cpm"),
    Method("angel", "angel", "ANGEL", {"threshold": 0.6, "min_community_size": 3}, 58, .403, .518, "angel"),
    Method("bigclam", "bigclam", "BigCLAM", {"communities": 25000}, 582, .395, .472, "bigclam", True, ("communities",)),
    Method("ego_splitting", "ego_splitting", "Ego-Splitting", {"resolution": 1.0}, 96, .333, .500, "angel"),
    Method("ncgame", "ncgame", "NcGame", {}, 15, .326, .484, "ncgame"),
    Method("neo_kmeans", "neo_kmeans", "NEO-K-Means", {"clusters": 13477, "alpha": 10, "beta": 0.0001}, 475, .334, .398,
           "neo_kmeans", False, ("clusters",)),
]
BY_KEY = {m.key: m for m in METHODS}
ALIASES = {"hedonic_local": "hoc_local", "hoc": "hoc_local", "hedonic_multiphase": "hoc_multilevel",
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
    methods: list[str] = field(default_factory=lambda: ["hoc_local", "codeseg"])
    nodes: int = 5000                      # 0 = full graph
    seeds: list[int] = field(default_factory=lambda: [0])
    preset: str = "screened"
    threads: int = os.cpu_count() or 1
    timeout: float = 600.0
    output_dir: str = "hedonic-exp-output"
    cache_dir: str = "~/.cache/hedonic"
    network_root: str | None = None
    omega: bool = True
    json: bool = False
    order: str = "network,seed,method"      # loop nesting, outermost first
    save_covers: bool = False               # keep every detector cover (covers/<net>/<method>.json.gz)
    profile: str | None = None              # named config this run came from (informational)

    def command(self) -> str:
        if self.profile in PROFILES and PROFILES[self.profile].differences(self) <= {"output_dir", "cache_dir",
                                                                                      "network_root", "json"}:
            parts = ["hedonic run exp", "--config", self.profile]
            if self.output_dir != PROFILES[self.profile].output_dir:
                parts += ["--output-dir", self.output_dir]
            if self.cache_dir != "~/.cache/hedonic":
                parts += ["--cache-dir", self.cache_dir]
            if self.network_root:
                parts += ["--network-root", self.network_root]
            return " ".join(shlex.quote(p) if " " in p and p != "hedonic run exp" else p for p in parts)
        parts = ["hedonic run exp", "--networks", ",".join(self.networks), "--methods", ",".join(self.methods),
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
        return " ".join(shlex.quote(p) if " " in p and p != "hedonic run exp" else p for p in parts)

    def pretty_command(self) -> str:
        """The same command broken at option boundaries, for narrow terminals (shell line continuations)."""
        words = self.command().split(" --")
        return " \\\n      --".join([words[0], *words[1:]])

    def differences(self, other: "Config") -> set[str]:
        return {k for k in self.__dict__ if k != "profile" and getattr(self, k) != getattr(other, k)}


# Named configurations (``--config NAME``). "paper" is the paper's reproducible
# experiment: the two smallest SNAP networks (DBLP, Amazon) as full graphs, all
# ten screened methods at their screened settings, seeds 0-9, run seed -> method
# -> network (seed 0: method 1 on DBLP then Amazon; method 2 on both; ...; then
# seed 1, ...), one run at a time. YouTube is excluded (ANGEL did not finish on
# full YouTube within an hour). The per-run timeout is one hour.
PROFILES: dict[str, Config] = {}
PROFILES["default"] = Config()
PROFILES["paper"] = Config(networks=["dblp", "amazon"],
                           methods=["hoc_local", "hoc_multilevel", "codeseg", "fox", "cpm", "angel", "bigclam",
                                    "ego_splitting", "ncgame", "neo_kmeans"],
                           nodes=0, seeds=list(range(10)), preset="screened", timeout=3600.0,
                           order="seed,method,network", output_dir="hedonic-paper-reproduction", profile="paper")
PROFILE_HELP = {"default": "DBLP, 5,000-vertex subgraph, seed 0, HOC-local + CoDeSEG (a minute)",
                "paper": "paper reproduction: DBLP, Amazon (full graphs) x all 10 methods x seeds 0-9, "
                         "order seed -> method -> network (many hours)"}
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


def parse_name(value: str) -> str:
    """Run names become directory and tmux session names, so keep them plain."""
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError(f"invalid run name {value!r}; use letters, digits, '-' and '_' (at most 64, no dots or spaces)")
    return value


def parse_choice(value: str, universe: list[str], aliases: dict, what: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(universe)
    chosen = []
    for part in value.split(","):
        key = aliases.get(part.strip().lower(), part.strip().lower())
        if key not in universe:
            raise ValueError(f"unknown {what} {part!r}; choose from: {', '.join(universe)} (or all)")
        chosen.append(key)
    return list(dict.fromkeys(chosen))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hedonic run exp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Overlapping community detection benchmark on SNAP ground-truth networks.\n"
                    "Without arguments (in a terminal) an interactive wizard starts.",
        epilog="networks: " + ", ".join(NETWORKS) + "\nmethods:  " + ", ".join(BY_KEY) +
               "\n\nexamples:\n  hedonic run exp --yes\n  hedonic run exp --networks dblp,amazon --methods all --seeds 0-2\n"
               "  hedonic run exp --networks youtube --nodes 0 --methods hoc_local,fox\n  hedonic run exp --list")
    p.add_argument("-i", "--interactive", action="store_true", help="open the wizard (default when no arguments)")
    p.add_argument("-y", "--yes", action="store_true", help="run with defaults/flags, never prompt")
    p.add_argument("--networks", "--datasets", default=None, help="comma list or 'all' (default: dblp)")
    p.add_argument("--methods", default=None, help="comma list or 'all' (default: hoc_local,codeseg)")
    p.add_argument("--nodes", type=int, default=None, help="vertices of the induced subgraph; 0 = full graph (default 5000)")
    p.add_argument("--full", action="store_true", help="shorthand for --nodes 0")
    p.add_argument("--seeds", default=None, help="e.g. 0 | 0-4 | 0,3,7 (default 0)")
    p.add_argument("--preset", choices=("screened", "registry", "paper"), default=None,
                   help="screened = most accurate setting within 10 min on full DBLP; registry = registry/paper "
                        "defaults ('paper' is accepted as an old spelling)")
    p.add_argument("--threads", type=int, default=None, help="threads for methods that support them (default: all cores)")
    p.add_argument("--timeout", type=float, default=None, help="per-run detection timeout in seconds (default 600)")
    p.add_argument("--output-dir", default=None, help="where run records and the summary are saved (default hedonic-exp-output)")
    p.add_argument("--cache-dir", default=None, help="downloads and built runtimes (default ~/.cache/hedonic)")
    p.add_argument("--network-root", default=None, help="optional folder with local SNAP archives")
    p.add_argument("--no-omega", action="store_true", help="skip the sampled Omega index (faster on big graphs)")
    p.add_argument("--json", action="store_true", help="also print the summary as JSON")
    p.add_argument("--list", action="store_true", help="list networks and methods, then exit")
    p.add_argument("-c", "--config", default=None,
                   help="named config (" + ", ".join(PROFILES) + ") or a .json/.toml file; flags given too override it")
    p.add_argument("--order", default=None,
                   help="loop nesting, outermost first, e.g. seed,method,network (default network,seed,method)")
    p.add_argument("--save-covers", action="store_true", help="keep every detector cover next to its record")
    p.add_argument("--dry-run", action="store_true", help="print the run plan and estimated time, then exit")
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
    path = expand_path(name_or_path)
    if not path.is_file():
        names = [*PROFILES, *user_profiles]
        raise ValueError(f"unknown config {name_or_path!r}: not one of {', '.join(names)} and not a file")
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
    path = source
    unknown = set(data) - set(Config().__dict__)
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {', '.join(sorted(unknown))}")
    for key, value in data.items():
        if key == "seeds" and isinstance(value, str):
            value = parse_seeds(value)
        if key in ("networks", "methods") and isinstance(value, str):
            value = value.split(",")
        setattr(cfg, key, value)
    cfg.networks = parse_choice(",".join(cfg.networks), list(NETWORKS), {"wiki": "wikipedia", "lj": "livejournal"}, "network")
    cfg.methods = parse_choice(",".join(cfg.methods), list(BY_KEY), ALIASES, "method")
    return cfg


def parse_order(value: str) -> str:
    keys = [k.strip() for k in value.split(",")]
    if sorted(keys) != sorted(ORDER_KEYS):
        raise ValueError("--order must be a permutation of network,seed,method")
    return ",".join(keys)


def config_from_args(a: argparse.Namespace) -> Config:
    from hedonic.experiments.overlapping import userconfig

    cfg = load_config(a.config)
    for key, value in userconfig.defaults().items():  # machine settings from ~/.config/hedonic/config.toml
        if key in userconfig.MACHINE_KEYS:
            setattr(cfg, key, value)
    if a.networks is not None:
        cfg.networks = parse_choice(a.networks, list(NETWORKS), {"wiki": "wikipedia", "lj": "livejournal"}, "network")
    if a.methods is not None:
        cfg.methods = parse_choice(a.methods, list(BY_KEY), ALIASES, "method")
    if a.full:
        cfg.nodes = 0
    elif a.nodes is not None:
        if a.nodes < 0:
            raise ValueError("--nodes must be 0 (full graph) or a positive number of vertices")
        cfg.nodes = a.nodes
    if a.seeds is not None:
        cfg.seeds = parse_seeds(a.seeds)
    for key in ("preset", "timeout", "output_dir", "cache_dir", "network_root"):
        if getattr(a, key) is not None:
            setattr(cfg, key, getattr(a, key))
    if not cfg.timeout > 0:
        raise ValueError("--timeout must be a positive number of seconds")
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
    if start == 0:
        return cfg
    if start == 1:
        paper = load_config("paper")
        print(render_plan(paper))
        if tui.select("Start the paper reproduction?", ["Run", "Back", "Quit"]) == 0:
            return paper
        return wizard(cfg)
    while True:
        names = list(NETWORKS)
        idx = tui.multiselect("Networks (SNAP, with ground truth)",
                              [(NETWORKS[n][0], f"{NETWORKS[n][1]:,} vertices · {NETWORKS[n][2]:,} edges") for n in names],
                              [names.index(n) for n in cfg.networks])
        cfg.networks = [names[i] for i in idx]
        idx = tui.multiselect("Methods (passed the staged screening)",
                              [(m.label, f"full DBLP: F1 {m.f1:.3f} · ONMI {m.onmi:.3f} · ~{m.seconds:.0f} s") for m in METHODS],
                              [list(BY_KEY).index(k) for k in cfg.methods])
        cfg.methods = [METHODS[i].key for i in idx]
        sizes = [1000, 5000, 20000, 100000, 0, -1]
        labels = [("1,000 vertices", "seconds"), ("5,000 vertices", "default"), ("20,000 vertices", ""),
                  ("100,000 vertices", ""), ("Full graph", "slow on the large networks"), ("Custom…", "")]
        i = tui.select("Graph size (deterministic induced subgraph built around the labelled communities)", labels,
                       sizes.index(cfg.nodes) if cfg.nodes in sizes else 1)
        cfg.nodes = sizes[i] if sizes[i] >= 0 else int(tui.text("Number of vertices", "5000",
                                                                  lambda v: None if v.isdigit() else "enter an integer"))
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
        cfg.timeout = float(tui.text("Per-run timeout [s]", f"{cfg.timeout:g}", lambda v: _check(float, v)))
        cfg.output_dir = tui.text("Save runs and results in", cfg.output_dir)
        cfg.cache_dir = tui.text("Cache for downloads and built tools", cfg.cache_dir)
        runs = len(cfg.networks) * len(cfg.seeds) * len(cfg.methods)
        size = "full graph" if not cfg.nodes else f"{cfg.nodes:,} vertices"
        print()
        print(style("  Summary", "bold"))
        print(f"  networks  {', '.join(NETWORKS[n][0] for n in cfg.networks)}")
        print(f"  methods   {', '.join(BY_KEY[m].label for m in cfg.methods)}")
        print(f"  size      {size} · seeds {','.join(map(str, cfg.seeds))} · {runs} runs · preset {cfg.preset}"
              + ("" if cfg.omega else " · no Omega"))
        estimate = sum(expected_seconds(m, cfg, net) for net, _, m in build_plan(cfg, {}))
        print(f"  estimate  ~{fmt_time(estimate)} on an M1-Max-class machine (rough; refined live while running)")
        print(f"  saving to {expand_path(cfg.output_dir).resolve()}/<run name>")
        print(style("  same run without the wizard:", "dim"))
        print(style("    " + cfg.pretty_command(), "dim"))
        if "friendster" in cfg.networks and not cfg.nodes:
            print(style("  warning: full Friendster has 1.8 billion edges (tens of GB to download and load)", "yellow"))
        choice = tui.select("Start?", ["Run", "Change settings", "Quit"])
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


def ensure_codeseg(cache: Path, say) -> tuple[Path | None, str | None]:
    """Build CoDeSEG once from its pinned source; return (binary, None) or (None, reason)."""
    binary = cache / "quickstart" / f"CoDeSEG-{CODESEG_COMMIT[:12]}"
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
    """Prepare external runtimes through codeseg-setup; return {method: reason} for unavailable ones."""
    from hedonic.experiments.overlapping import codeseg_setup

    manifest_path = cache / "setup_manifest.json"

    def statuses() -> dict:
        try:
            return json.loads(manifest_path.read_text()).get("methods", {})
        except (OSError, ValueError):
            return {}

    needed = sorted({BY_KEY[m].setup for m in methods if BY_KEY[m].setup})
    todo = [s for s in needed if (statuses().get(s) or {}).get("status") != "ready"]
    if todo:
        say(f"preparing runtimes: {', '.join(todo)} (first use only; this can take a few minutes)")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                codeseg_setup.setup_main(["--datasets", "dblp", "--methods", ",".join(todo), "--build-native",
                                          "--skip-data", "--cache-dir", str(cache)])
            except SystemExit:
                pass
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
    return params


# Detection times measured on full Amazon/YouTube in saved records (M1 Max; overnight queue and
# snap-v4 baselines, 2026-09-26/27). They override the linear-in-edges extrapolation from full
# DBLP, which underestimates super-linear methods (HOC-local: 1,484 s measured on YouTube vs
# ~115 s extrapolated). ANGEL on full YouTube ran for over an hour without finishing.
MEASURED_FULL = {("hoc_local", "amazon"): 355, ("hoc_local", "youtube"): 1484,
                 ("codeseg", "amazon"): 4, ("codeseg", "youtube"): 8,
                 ("fox", "amazon"): 12, ("fox", "youtube"): 141,
                 ("angel", "amazon"): 57, ("angel", "youtube"): float("inf")}


def expected_seconds(m: Method, cfg: Config, network: str) -> float:
    n, e = NETWORKS[network][1:]
    if not cfg.nodes and (m.key, network) in MEASURED_FULL:
        return 2.0 + min(cfg.timeout, MEASURED_FULL[m.key, network])
    edges = e * min(1.0, cfg.nodes / n) if cfg.nodes else e
    return 2.0 + min(cfg.timeout, m.seconds * edges / DBLP_EDGES)


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rest = divmod(seconds, 3600)
    return f"{h}:{rest // 60:02d}:{rest % 60:02d}" if h else f"{rest // 60}:{rest % 60:02d}"


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


def prepare(cfg: Config, say) -> tuple[Path | None, dict[str, str]]:
    """Provision runtimes; return (CoDeSEG binary or None, {method: unavailable reason})."""
    tools = expand_path(cfg.cache_dir) / "codeseg"
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


def build_plan(cfg: Config, unavailable: dict[str, str]) -> list[tuple[str, int, Method]]:
    """All (network, seed, method) runs, nested in ``cfg.order`` (outermost first)."""
    import itertools

    axes = {"network": cfg.networks, "seed": cfg.seeds, "method": [m for m in cfg.methods if m not in unavailable]}
    names = cfg.order.split(",")
    plan = []
    for combo in itertools.product(*(axes[n] for n in names)):
        point = dict(zip(names, combo))
        plan.append((point["network"], point["seed"], BY_KEY[point["method"]]))
    return plan


def render_plan(cfg: Config) -> str:
    plan = build_plan(cfg, {})
    total = sum(expected_seconds(m, cfg, net) for net, _, m in plan)
    import textwrap

    width = max(60, shutil.get_terminal_size((100, 24)).columns - 4)

    def wrap(text: str) -> str:
        return textwrap.fill(text, width, initial_indent="  ", subsequent_indent="  ")

    lines = [style(f"  {len(plan)} runs", "bold") + f" · order {cfg.order.replace(',', ' → ')}",
             wrap(describe(cfg)),
             style(wrap(environment_line()), "dim"),
             style(wrap(f"estimated time ~{fmt_time(total)} on an M1-Max-class machine: a rough guide from the screening "
                        f"and saved timings, refined live while running. Downloads and first-use builds are not "
                        f"included. Timeout {cfg.timeout:g} s per run."), "dim"),
             style(f"  output {expand_path(cfg.output_dir)}", "dim"), ""]
    for i, (net, seed, m) in enumerate(plan[:12], 1):
        lines.append(f"  {i:>4}. seed {seed} · {m.label:<26} · {NETWORKS[net][0]:<12} ~{fmt_time(expected_seconds(m, cfg, net))}")
    if len(plan) > 12:
        lines.append(style(f"   …  {len(plan) - 12} more; last: seed {plan[-1][1]} · {plan[-1][2].label} · "
                           f"{NETWORKS[plan[-1][0]][0]}", "dim"))
    lines.append(style(f"\n  start it with: {cfg.command()}", "dim"))
    return "\n".join(lines)


def run_one(cfg: Config, net: str, seed: int, m: Method, binary: Path | None) -> dict:
    """One detector run through the unchanged runner (resumable: a finished record is reused)."""
    from hedonic.experiments.overlapping import codeseg_reproduction as cr

    cache = expand_path(cfg.cache_dir)
    out = expand_path(cfg.output_dir) / f"{net}-{'full' if not cfg.nodes else f'n{cfg.nodes}'}" / f"seed{seed}"
    argv = ["--datasets", net, "--methods", m.runner, "--seed", str(seed), "--timeout", f"{cfg.timeout:g}",
            "--resume", "--auto-download", "--snap-cache-dir", str(cache / "snap"), "--output-dir", str(out),
            "--method-params", json.dumps({m.runner: method_params(m, cfg, net)})]
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
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            cr.main(argv)
    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001 - keep going with the other runs
        buffer.write(f"{type(exc).__name__}: {exc}\n")
    path = out / "runs" / net / f"{m.runner}.json"
    if path.is_file():
        rec = json.loads(path.read_text())
    else:
        tail = [line for line in buffer.getvalue().strip().splitlines() if line.strip()]
        rec = {"status": "failed", "reason": tail[-1] if tail else "no record written"}
    rec.update(network=net, method_key=m.key, seed=seed)
    return rec


def result_line(tag: str, rec: dict) -> str:
    mm = rec.get("metrics") or {}
    if rec.get("status") == "completed":
        note = f"F1 {mm.get('f1', 0):.3f} · ONMI {mm.get('onmi', 0):.3f} · {rec.get('detection_seconds') or 0:.1f} s"
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


def run(cfg: Config) -> int:
    """Foreground execution in this terminal (``--foreground``)."""
    def say(text: str) -> None:
        print(style("  setup ", "magenta") + text, flush=True)

    binary, unavailable = prepare(cfg, say)
    plan = build_plan(cfg, unavailable)
    print(style(f"\n  {len(plan)} runs", "bold") + f" · {describe(cfg)}", flush=True)
    print(style(f"  {environment_line()}\n", "dim"), flush=True)
    records: list[dict] = []
    with Progress([expected_seconds(m, cfg, net) for net, _, m in plan]) as progress:
        for i, (net, seed, m) in enumerate(plan, 1):
            tag = f"[{i}/{len(plan)}] {NETWORKS[net][0]} · seed {seed} · {m.label}"
            progress.start(tag)
            rec = run_one(cfg, net, seed, m, binary)
            records.append(rec)
            progress.finish(result_line(tag, rec))
    summary = finish(cfg, records, unavailable)
    print(render(summary, cfg))
    output = expand_path(cfg.output_dir)
    print(style(f"\n  records   {output}", "dim"))
    print(style(f"  summary   {output / 'summary.json'}", "dim"))
    print(style(f"  rerun     {cfg.command()}", "dim"))
    if cfg.json:
        print(json.dumps(summary, indent=2, default=str))
    return 0 if all(r.get("status") == "completed" for r in records) else 1


def environment() -> dict:
    """Software provenance stored with every run (summary.json, progress.json)."""
    import importlib.metadata as md
    import platform

    def version(dist: str) -> str | None:
        try:
            return md.version(dist)
        except md.PackageNotFoundError:
            return None

    try:
        from hedonic.Game import HEDONIC_ALGORITHM_IDENTITY as identity
    except ImportError:
        identity = None
    import igraph

    return {"hedonic": version("hedonic"), "lucas_igraph": version("lucas-igraph"),
            "igraph_module_version": getattr(igraph, "__version__", None), "algorithm_identity": identity,
            "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "executable": sys.executable}


def environment_line() -> str:
    env = environment()
    return (f"hedonic {env['hedonic']} · lucas-igraph {env['lucas_igraph']} · Python {env['python']} · "
            f"{env['machine']}")


def describe(cfg: Config) -> str:
    size = "full graph" if not cfg.nodes else f"{cfg.nodes:,}-vertex subgraph"
    return (f"{', '.join(NETWORKS[n][0] for n in cfg.networks)} · {size} · seeds {','.join(map(str, cfg.seeds))}"
            f" · preset {cfg.preset} · {cfg.threads} threads")


# --------------------------------------------------------------------------- results
def summarise(records: list[dict], cfg: Config) -> list[dict]:
    rows = []
    for net in cfg.networks:
        for key in cfg.methods:
            recs = [r for r in records if r["network"] == net and r["method_key"] == key]
            done = [r for r in recs if r.get("status") == "completed"]
            row = {"network": net, "method": key, "label": BY_KEY[key].label, "completed": len(done), "runs": len(recs),
                   "status": "completed" if done and len(done) == len(recs) else
                             (recs[-1].get("status") if recs else "not run"),
                   "reason": None if done else (recs[-1].get("reason") if recs else None)}
            for metric in [k for k, _ in COLUMNS] + ["detection_seconds"]:
                vals = [(r.get("metrics") or {}).get(metric) if metric != "detection_seconds" else r.get(metric)
                        for r in done]
                vals = [float(v) for v in vals if isinstance(v, (int, float))]
                row[metric] = statistics.fmean(vals) if vals else None
                row[metric + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else None
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


def _render_tables(rows: list[dict], cfg: Config, show_sd: bool) -> str:
    out = []
    multi = len(cfg.seeds) > 1
    many = multi and show_sd
    headers = ["Method", *(label for _, label in COLUMNS), "time [s]", "runs"]
    for net in cfg.networks:
        block = [r for r in rows if r["network"] == net]
        size = "full graph" if not cfg.nodes else f"{cfg.nodes:,}-vertex subgraph"
        seeds = (f"mean ± sd over {len(cfg.seeds)} seeds" if many else
                 f"mean over {len(cfg.seeds)} seeds (± sd in summary.json)" if multi else f"seed {cfg.seeds[0]}")
        out.append("\n" + style(f"  {NETWORKS[net][0]} · {size} · {seeds}", "bold"))
        best = {k: max((r[k] for r in block if r[k] is not None), default=None) for k, _ in COLUMNS}
        fastest = min((r["detection_seconds"] for r in block if r["detection_seconds"] is not None), default=None)
        body = []
        for r in block:
            if r["completed"] == 0:
                body.append((r, None))
                continue
            cells = [r["label"]]
            for k, _ in COLUMNS:
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
        print(info.render_methods(info.method_status(cache)))
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
    try:
        cfg = config_from_args(args)
        if args.name:
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
    wants_wizard = args.interactive or (not argv and tui.interactive())
    try:
        if wants_wizard and not args.yes:
            cfg = wizard(cfg)
        if args.foreground:
            return run(cfg)
        from hedonic.experiments.overlapping import runmanager

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
    except (tui.Cancelled, KeyboardInterrupt):
        print(style("\n  cancelled", "yellow"))
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
