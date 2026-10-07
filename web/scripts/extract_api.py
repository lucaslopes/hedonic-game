#!/usr/bin/env python3
"""Extract the public Python API of ``hedonic`` into JSON for the docs site.

The script parses source files with :mod:`ast` and never imports the package,
so it runs with a plain Python >= 3.9 and without lucas-igraph installed. The
output is deterministic: no timestamps, stable ordering, sorted keys.

Usage::

    python3 web/scripts/extract_api.py --output web/src/generated/api.json

Environment:
    GITHUB_REPOSITORY / GITHUB_SHA  set by GitHub Actions; used to pin source
                                    links to the exact public commit.
    HEDONIC_DOCS_REF                override the git ref used in source links.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PATHS = {
    "init": "src/hedonic/__init__.py",
    "game": "src/hedonic/Game.py",
    "utils": "src/hedonic/utils.py",
    "cli": "src/hedonic/experiments/CLI.py",
    "sbm_sweep": "src/hedonic/experiments/disjoint/sbm_sweep.py",
    "reproduce": "src/hedonic/experiments/disjoint/reproduce.py",
    "data_loader": "src/hedonic/experiments/disjoint/data_loader.py",
    "pyproject": "pyproject.toml",
}

SECTION_NAMES = (
    "Parameters",
    "Returns",
    "Raises",
    "Notes",
    "Examples",
    "See Also",
    "Environment",
)


def read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def parse(rel: str) -> ast.Module:
    return ast.parse(read(rel), filename=rel)


# ---------------------------------------------------------------------------
# Docstrings (numpydoc subset)
# ---------------------------------------------------------------------------


def split_sections(doc: str) -> tuple[str, dict[str, str]]:
    """Split a cleaned docstring into its free text and numpydoc sections."""
    lines = doc.splitlines()
    free: list[str] = []
    sections: dict[str, list[str]] = {}
    current: list[str] = free
    i = 0
    while i < len(lines):
        line = lines[i]
        underline = lines[i + 1] if i + 1 < len(lines) else ""
        title = line.strip()
        if title in SECTION_NAMES and re.fullmatch(r"-{3,}", underline.strip() or "x"):
            current = sections.setdefault(title, [])
            i += 2
            continue
        current.append(line)
        i += 1
    return "\n".join(free).strip(), {k: "\n".join(v).strip() for k, v in sections.items()}


def parse_parameter_section(text: str) -> list[dict[str, str]]:
    """Parse ``name : type`` entries followed by indented descriptions."""
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    description: list[str] = []

    def flush() -> None:
        if current is not None:
            current["description"] = inspect.cleandoc("\n".join(description)).strip()
            entries.append(current)

    for raw in text.splitlines():
        header = re.match(r"^([A-Za-z_][\w, ]*?)\s*:\s*(.*)$", raw)
        if header and not raw.startswith((" ", "\t")):
            flush()
            names = [name.strip() for name in header.group(1).split(",") if name.strip()]
            current = {"names": ", ".join(names), "type": header.group(2).strip()}
            description = []
        elif current is not None:
            description.append(raw)
    flush()
    return entries


def docstring_of(node: ast.AST) -> dict | None:
    raw = ast.get_docstring(node, clean=True)
    if not raw:
        return None
    free, sections = split_sections(raw)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", free) if p.strip()]
    summary = paragraphs[0] if paragraphs else ""
    result: dict = {
        "summary": summary,
        "description": "\n\n".join(paragraphs[1:]),
        "sections": sections,
    }
    if "Parameters" in sections:
        result["parameters"] = parse_parameter_section(sections["Parameters"])
    return result


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


def unparse(node: ast.AST | None) -> str | None:
    return None if node is None else ast.unparse(node)


def signature_of(fn: ast.FunctionDef, *, drop_self: bool) -> tuple[str, list[dict]]:
    args = fn.args
    params: list[dict] = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for index, (arg, default) in enumerate(zip(positional, defaults)):
        kind = "positional-only" if index < len(args.posonlyargs) else "positional-or-keyword"
        params.append({"name": arg.arg, "kind": kind, "annotation": unparse(arg.annotation), "default": unparse(default)})
    if args.vararg:
        params.append({"name": args.vararg.arg, "kind": "var-positional", "annotation": unparse(args.vararg.annotation), "default": None})
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append({"name": arg.arg, "kind": "keyword-only", "annotation": unparse(arg.annotation), "default": unparse(default)})
    if args.kwarg:
        params.append({"name": args.kwarg.arg, "kind": "var-keyword", "annotation": unparse(args.kwarg.annotation), "default": None})
    if drop_self and params and params[0]["name"] in {"self", "cls"}:
        params = params[1:]

    parts: list[str] = []
    star_emitted = False
    for param in params:
        text = param["name"]
        if param["kind"] == "var-positional":
            text = "*" + text
            star_emitted = True
        elif param["kind"] == "var-keyword":
            text = "**" + text
        elif param["kind"] == "keyword-only" and not star_emitted:
            parts.append("*")
            star_emitted = True
        if param["annotation"]:
            text += f": {param['annotation']}"
        if param["default"] is not None:
            text += f" = {param['default']}" if param["annotation"] else f"={param['default']}"
        parts.append(text)
    returns = unparse(fn.returns)
    signature = f"{fn.name}({', '.join(parts)})" + (f" -> {returns}" if returns else "")
    return signature, params


def decorator_names(fn: ast.FunctionDef) -> list[str]:
    return [ast.unparse(d) for d in fn.decorator_list]


def function_entry(fn: ast.FunctionDef, *, path: str, owner: str | None) -> dict:
    signature, params = signature_of(fn, drop_self=owner is not None)
    doc = docstring_of(fn)
    described = {entry["names"]: entry for entry in (doc or {}).get("parameters", [])}
    for param in params:
        for names, entry in described.items():
            if param["name"] in [n.strip() for n in names.split(",")]:
                param["description"] = entry["description"]
                if entry["type"]:
                    param["docType"] = entry["type"]
    return {
        "name": fn.name,
        "qualname": f"{owner}.{fn.name}" if owner else fn.name,
        "signature": signature,
        "parameters": params,
        "returns": unparse(fn.returns),
        "decorators": decorator_names(fn),
        "doc": doc,
        "path": path,
        "line": fn.lineno,
        "endLine": fn.end_lineno,
    }


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def extract_exports() -> list[str]:
    for node in parse(PATHS["init"]).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets):
            return list(ast.literal_eval(node.value))
    return []


def extract_game() -> tuple[dict, list[dict]]:
    tree = parse(PATHS["game"])
    constants: list[dict] = []
    game: dict | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.isupper() and not name.startswith("_"):
                try:
                    value = ast.literal_eval(node.value)
                except ValueError:
                    continue
                constants.append({"name": name, "value": value, "path": PATHS["game"], "line": node.lineno})
        if isinstance(node, ast.ClassDef) and node.name == "Game":
            members: list[dict] = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    public = not item.name.startswith("_") or item.name == "__init__"
                    decorators = decorator_names(item)
                    if not public or any(d.endswith(".setter") for d in decorators):
                        continue
                    entry = function_entry(item, path=PATHS["game"], owner="Game")
                    entry["kind"] = "property" if "property" in decorators else "method"
                    if entry["kind"] == "property":
                        entry["settable"] = any(
                            isinstance(other, ast.FunctionDef)
                            and other.name == item.name
                            and f"{item.name}.setter" in decorator_names(other)
                            for other in node.body
                        )
                    members.append(entry)
                elif (
                    isinstance(item, ast.Assign)
                    and len(item.targets) == 1
                    and isinstance(item.targets[0], ast.Name)
                    and isinstance(item.value, ast.Name)
                    and not item.targets[0].id.startswith("_")
                ):
                    members.append(
                        {
                            "kind": "alias",
                            "name": item.targets[0].id,
                            "qualname": f"Game.{item.targets[0].id}",
                            "aliasOf": item.value.id,
                            "path": PATHS["game"],
                            "line": item.lineno,
                            "endLine": item.end_lineno,
                        }
                    )
            game = {
                "name": "Game",
                "qualname": "hedonic.Game",
                "bases": [ast.unparse(base) for base in node.bases],
                "doc": docstring_of(node),
                "path": PATHS["game"],
                "line": node.lineno,
                "endLine": node.end_lineno,
                "members": members,
            }
    if game is None:
        raise SystemExit("class Game not found in " + PATHS["game"])
    return game, constants


def extract_utils() -> list[dict]:
    return [
        {**function_entry(node, path=PATHS["utils"], owner=None), "module": "hedonic.utils"}
        for node in parse(PATHS["utils"]).body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]


def tolerant_literal(node: ast.expr):
    """Like ``ast.literal_eval``, but keep non-literal parts as ``{"$expr": source}``."""
    if isinstance(node, ast.Dict):
        return {
            tolerant_literal(k) if k is not None else "**": tolerant_literal(v)
            for k, v in zip(node.keys, node.values)
        }
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [tolerant_literal(item) for item in node.elts]
    try:
        return ast.literal_eval(node)
    except ValueError:
        return {"$expr": ast.unparse(node)}


def assigned_value(tree: ast.Module, name: str) -> ast.expr | None:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return node.value
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
            return node.value
    return None


def extract_cli() -> dict:
    tree = parse(PATHS["cli"])
    registry = assigned_value(tree, "COMMANDS")
    commands: list[dict] = []
    if isinstance(registry, ast.Dict):
        for key, value in zip(registry.keys, registry.values):
            if not isinstance(value, ast.Call):
                continue
            fields = {kw.arg: tolerant_literal(kw.value) for kw in value.keywords if kw.arg}
            commands.append(
                {
                    "name": ast.literal_eval(key) if key is not None else fields.get("name"),
                    "module": fields.get("module") or None,
                    "summary": fields.get("summary", ""),
                    "needsData": fields.get("needs_data"),
                }
            )
    return {"path": PATHS["cli"], "doc": docstring_of(tree), "commands": commands}


def extract_disjoint() -> dict:
    sweep = parse(PATHS["sbm_sweep"])
    methods = assigned_value(sweep, "METHODS")
    presets = {}
    for name, key in (("V1020_PRESET", "v1020"), ("V1020_SMOKE_PRESET", "v1020-smoke")):
        value = assigned_value(sweep, name)
        if value is not None:
            presets[key] = tolerant_literal(value)
    return {
        "sbmSweep": {
            "path": PATHS["sbm_sweep"],
            "doc": docstring_of(sweep),
            "methods": tolerant_literal(methods) if methods is not None else {},
            "presets": presets,
        },
        "reproduce": {"path": PATHS["reproduce"], "doc": docstring_of(parse(PATHS["reproduce"]))},
        "dataLoader": {"path": PATHS["data_loader"], "doc": docstring_of(parse(PATHS["data_loader"]))},
    }


def extract_package() -> dict:
    text = read(PATHS["pyproject"])
    try:
        import tomllib  # Python >= 3.11

        project = tomllib.loads(text)["project"]
        version = project.get("version")
        requires = project.get("requires-python")
        dependencies = list(project.get("dependencies", []))
        scripts = project.get("scripts", {})
    except ModuleNotFoundError:  # pragma: no cover - Python 3.9/3.10 fallback
        version = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)
        requires = re.search(r'^requires-python\s*=\s*"([^"]+)"', text, re.M).group(1)
        block = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.M | re.S).group(1)
        dependencies = re.findall(r'"([^"]+)"', block)
        scripts = dict(re.findall(r'^([\w-]+)\s*=\s*"([^"]+:[^"]+)"', text, re.M))
    native = next((dep for dep in dependencies if dep.replace("_", "-").startswith("lucas-igraph")), None)
    return {
        "name": "hedonic",
        "version": version,
        "requiresPython": requires,
        "dependencies": dependencies,
        "nativeDependency": native,
        "scripts": scripts,
    }


def repository_info() -> dict:
    repository = os.environ.get("GITHUB_REPOSITORY")
    if repository:
        url = f"https://github.com/{repository}"
    else:
        url = "https://github.com/lucaslopes/hedonic-game"
        try:
            remote = subprocess.run(
                ["git", "-C", str(REPO), "remote", "get-url", "origin"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", remote)
            if match:
                url = f"https://github.com/{match.group(1)}"
        except (OSError, subprocess.CalledProcessError):
            pass
    ref = os.environ.get("HEDONIC_DOCS_REF") or os.environ.get("GITHUB_SHA") or "main"
    return {"url": url, "ref": ref}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=str(REPO / "web/src/generated/api.json"))
    args = parser.parse_args(argv)

    game, constants = extract_game()
    payload = {
        "schemaVersion": 1,
        "generator": "web/scripts/extract_api.py",
        "repository": repository_info(),
        "package": extract_package(),
        "exports": extract_exports(),
        "classes": [game],
        "functions": extract_utils(),
        "constants": constants,
        "cli": extract_cli(),
        "experiments": {"disjoint": extract_disjoint()},
        "sources": sorted(set(PATHS.values())),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Insertion order mirrors the source files, so the output is already
    # deterministic; sorting keys would scramble tables such as METHODS.
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    members = len(game["members"])
    shown = output.resolve()
    shown = shown.relative_to(REPO) if shown.is_relative_to(REPO) else shown
    print(f"api: wrote {shown} ({members} Game members, {len(payload['cli']['commands'])} CLI commands)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
