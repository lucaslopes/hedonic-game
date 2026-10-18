"""Per-user settings for ``hedonic run`` (``~/.config/hedonic/config.toml``).

Example::

    [defaults]          # machine settings applied to every run, unless a flag overrides them
    disjoint_output_root = "/data/hedonic/disjoint"   # parent folder of `hedonic run disjoint` runs and queues
    threads = 10

Manage it with ``hedonic config`` (show), ``hedonic config set KEY VALUE``,
``hedonic config unset KEY``, ``hedonic config path``; ``hedonic config example``
prints the commented template that ships with the package. The file location can
be overridden with ``HEDONIC_CONFIG``.
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path

MACHINE_KEYS = ("threads",)
# Machine paths read by one front end (the disjoint benchmark and its queues).
OTHER_DEFAULT_KEYS = ("disjoint_output_root",)
DEFAULT_KEYS = (*MACHINE_KEYS, *OTHER_DEFAULT_KEYS)  # everything [defaults] accepts


def path() -> Path:
    configured = os.environ.get("HEDONIC_CONFIG")
    if configured:
        return Path(configured).expanduser()
    base = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser()
    return base / "hedonic" / "config.toml"


def load() -> dict:
    p = path()
    if not p.is_file():
        return {}
    try:
        return tomllib.loads(p.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f"invalid config file {p}: {exc}") from exc


def defaults() -> dict:
    return dict(load().get("defaults", {}))


# --------------------------------------------------------------------------- writing (small TOML subset)
def _value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    return json.dumps(str(v))


def save(data: dict) -> Path:
    lines = ["# hedonic user configuration — see `hedonic config --help`", ""]
    if data.get("defaults"):
        lines += ["[defaults]"] + [f"{k} = {_value(v)}" for k, v in data["defaults"].items()]
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    return p


def _parse(raw: str):
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def _validate(name: str, value) -> None:
    """Reject a setting that would only fail later, when a run starts."""
    if name == "threads" and not (isinstance(value, int) and not isinstance(value, bool) and value >= 1):
        raise SystemExit(f"threads must be a positive integer, not {value!r}")
    if name != "threads" and not isinstance(value, str):
        raise SystemExit(f"{name} must be a path; quote it if it looks like something else "
                         f"(hedonic config set {name} \"...\")")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    data = load()
    if not argv or argv[0] == "show":
        p = path()
        print(f"# {p}" + ("" if p.is_file() else "  (not created yet — `hedonic config set disjoint_output_root PATH`)"))
        if p.is_file():
            print(p.read_text().rstrip())
        return 0
    if argv[0] == "path":
        print(path())
        return 0
    if argv[0] == "example":
        from hedonic.experiments.resources import CONFIG_DIR

        print((CONFIG_DIR / "hedonic-run.example.toml").read_text().rstrip())
        return 0
    if argv[0] in ("set", "unset") and len(argv) >= 2:
        key = argv[1]
        section, _, name = key.rpartition(".")
        if section and section != "defaults":
            raise SystemExit(f"unknown section in {key!r}; hedonic 1 keeps machine settings only: "
                             f"{', '.join(DEFAULT_KEYS)} (or defaults.KEY)")
        target = data.setdefault("defaults", {})
        if argv[0] == "set":
            if len(argv) < 3:
                raise SystemExit("usage: hedonic config set KEY VALUE")
            if name not in DEFAULT_KEYS:
                raise SystemExit(f"[defaults] accepts {', '.join(DEFAULT_KEYS)}")
            value = _parse(" ".join(argv[2:]))
            _validate(name, value)
            target[name] = value
        elif name not in target:
            print(f"  {key} is not set; nothing to remove")
            return 0
        else:
            target.pop(name)
        print(f"  {'set' if argv[0] == 'set' else 'removed'} {key} in {save(data)}")
        return 0
    raise SystemExit("usage: hedonic config [show | path | example | set KEY VALUE | unset KEY]")


if __name__ == "__main__":
    raise SystemExit(main())
