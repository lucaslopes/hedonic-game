"""Per-user settings for ``hedonic run`` (``~/.config/hedonic/config.toml``).

Example::

    # what a bare `hedonic run` starts (a named config or a config-file path)
    run = "paper"

    [defaults]          # machine settings applied to every run, unless a flag overrides them
    output_dir = "/data/hedonic/runs"
    cache_dir = "~/.cache/hedonic"
    network_root = "/data/snap"
    threads = 10

    [profiles.quick]    # your own named configs (any `hedonic run exp` setting)
    networks = ["dblp", "amazon"]
    methods = ["hoc_local", "codeseg", "fox"]
    nodes = 20000
    seeds = "0-4"

Manage it with ``hedonic config`` (show), ``hedonic config set KEY VALUE``,
``hedonic config unset KEY``, ``hedonic config path``. The file location can be
overridden with ``HEDONIC_CONFIG``.
"""

from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path

MACHINE_KEYS = ("output_dir", "cache_dir", "network_root", "threads")
TOP_KEYS = ("run",)


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


def profiles() -> dict:
    return dict(load().get("profiles", {}))


def favourite() -> str | None:
    return load().get("run")


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
    for key in TOP_KEYS:
        if key in data:
            lines.append(f"{key} = {_value(data[key])}")
    if data.get("defaults"):
        lines += ["", "[defaults]"] + [f"{k} = {_value(v)}" for k, v in data["defaults"].items()]
    for name, table in (data.get("profiles") or {}).items():
        lines += ["", f"[profiles.{name}]"] + [f"{k} = {_value(v)}" for k, v in table.items()]
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    return p


def _parse(raw: str):
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def _validate(section: str, name: str, value) -> None:
    """Reject a setting that would only fail later, when a run starts."""
    if section == "defaults":
        if name == "threads" and not (isinstance(value, int) and not isinstance(value, bool) and value >= 1):
            raise SystemExit(f"threads must be a positive integer, not {value!r}")
        if name != "threads" and not isinstance(value, str):
            raise SystemExit(f"{name} must be a path; quote it if it looks like something else "
                             f"(hedonic config set {name} \"...\")")
    elif section.startswith("profiles."):
        from hedonic.experiments.overlapping import quickstart as qs

        fields = set(qs.Config().__dict__) - {"profile"} | {"config"}
        if name not in fields:
            raise SystemExit(f"unknown run setting {name!r}; choose from: {', '.join(sorted(fields))}")
    elif name == "run":
        from hedonic.experiments.overlapping import quickstart as qs

        known = {*qs.PROFILES, *load().get("profiles", {})}
        if not isinstance(value, str) or (value not in known and not Path(value).expanduser().is_file()):
            print(f"  note: {value!r} is not a known config or file yet (known: {', '.join(sorted(known))})",
                  file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    data = load()
    if not argv or argv[0] == "show":
        p = path()
        print(f"# {p}" + ("" if p.is_file() else "  (not created yet — `hedonic config set output_dir PATH`)"))
        if p.is_file():
            print(p.read_text().rstrip())
        return 0
    if argv[0] == "path":
        print(path())
        return 0
    if argv[0] in ("set", "unset") and len(argv) >= 2:
        key = argv[1]
        section, _, name = key.rpartition(".")
        if not section:
            section = "" if key in TOP_KEYS else "defaults"
        if section and section != "defaults" and not section.startswith("profiles."):
            raise SystemExit(f"unknown section in {key!r}; use run, defaults.KEY, or profiles.NAME.KEY")
        target = data
        for part in [s for s in section.split(".") if s]:
            target = target.setdefault(part, {})
        if argv[0] == "set":
            if len(argv) < 3:
                raise SystemExit("usage: hedonic config set KEY VALUE")
            if section == "defaults" and name not in MACHINE_KEYS:
                raise SystemExit(f"[defaults] accepts {', '.join(MACHINE_KEYS)}; experiment settings go in profiles.NAME")
            value = _parse(" ".join(argv[2:]))
            _validate(section, name, value)
            target[name] = value
        elif name not in target:
            print(f"  {key} is not set; nothing to remove")
            return 0
        else:
            target.pop(name)
        print(f"  {'set' if argv[0] == 'set' else 'removed'} {key} in {save(data)}")
        return 0
    raise SystemExit("usage: hedonic config [show | path | set KEY VALUE | unset KEY]")


if __name__ == "__main__":
    raise SystemExit(main())
