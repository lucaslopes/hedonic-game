#!/usr/bin/env python3
"""Extract the public Python API of ``hedonic`` into JSON for the docs site.

The script parses source files with :mod:`ast` and never imports the package,
so it runs with a plain Python >= 3.9, without igraph, NumPy or the compiled
core installed. It records the ``Game`` class, ``hedonic.utils`` and the public
functions and constants of ``hedonic.metrics``. The output is deterministic: no
timestamps, stable ordering.

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
    "metrics": "src/hedonic/metrics.py",
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


def extract_constants(key: str, module: str) -> list[dict]:
    """Public upper-case module constants whose value is a Python literal."""
    constants: list[dict] = []
    for node in parse(PATHS[key]).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value_node = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            target, value_node = node.target.id, node.value
        else:
            continue
        if not target.isupper() or target.startswith("_"):
            continue
        try:
            value = ast.literal_eval(value_node)
        except ValueError:
            continue
        constants.append({"name": target, "module": module, "value": value, "path": PATHS[key], "line": node.lineno})
    return constants


def extract_game() -> dict:
    tree = parse(PATHS["game"])
    game: dict | None = None
    for node in tree.body:
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
    return game


def extract_functions(key: str, module: str) -> list[dict]:
    """Public top-level functions of a module (``__all__`` when it defines one)."""
    tree = parse(PATHS[key])
    exported = assigned_value(tree, "__all__")
    names = set(ast.literal_eval(exported)) if exported is not None else None
    return [
        {**function_entry(node, path=PATHS[key], owner=None), "module": module}
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and not node.name.startswith("_")
        and (names is None or node.name in names)
    ]


def assigned_value(tree: ast.Module, name: str) -> ast.expr | None:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return node.value
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
            return node.value
    return None


def extract_package() -> dict:
    text = read(PATHS["pyproject"])
    try:
        import tomllib  # Python >= 3.11

        project = tomllib.loads(text)["project"]
        version = project.get("version")
        requires = project.get("requires-python")
        dependencies = list(project.get("dependencies", []))
        extras = {name: list(deps) for name, deps in project.get("optional-dependencies", {}).items()}
    except ModuleNotFoundError:  # pragma: no cover - Python 3.9/3.10 fallback
        version = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M).group(1)
        requires = re.search(r'^requires-python\s*=\s*"([^"]+)"', text, re.M).group(1)
        block = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.M | re.S).group(1)
        dependencies = re.findall(r'"([^"]+)"', block)
        extras = {}
        section = re.search(r"^\[project\.optional-dependencies\]\s*$(.*?)(?=^\[)", text, re.M | re.S)
        if section:
            for name, body in re.findall(r"^([\w-]+)\s*=\s*\[(.*?)\]", section.group(1), re.M | re.S):
                extras[name] = re.findall(r'"([^"]+)"', body)
    # The graph library the package builds on (stock igraph from PyPI); the
    # detector itself is the package's own compiled core, hedonic._core.
    graph = next((dep for dep in dependencies if re.match(r"^igraph\b", dep)), None)
    return {
        "name": "hedonic",
        "version": version,
        "requiresPython": requires,
        "dependencies": dependencies,
        "optionalDependencies": extras,
        "graphDependency": graph,
    }


def repository_info() -> dict:
    repository = os.environ.get("GITHUB_REPOSITORY")
    if repository:
        url = f"https://github.com/{repository}"
    else:
        url = "https://github.com/lucaslopes/hedonic"
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

    game = extract_game()
    functions = extract_functions("utils", "hedonic.utils") + extract_functions("metrics", "hedonic.metrics")
    constants = extract_constants("game", "hedonic.Game") + extract_constants("metrics", "hedonic.metrics")
    payload = {
        "schemaVersion": 2,
        "generator": "web/scripts/extract_api.py",
        "repository": repository_info(),
        "package": extract_package(),
        "exports": extract_exports(),
        "classes": [game],
        "functions": functions,
        "constants": constants,
        "sources": sorted(set(PATHS.values())),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Insertion order mirrors the source files, so the output is already
    # deterministic; sorting keys would scramble the source order of members.
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    members = len(game["members"])
    shown = output.resolve()
    shown = shown.relative_to(REPO) if shown.is_relative_to(REPO) else shown
    print(f"api: wrote {shown} ({members} Game members, {len(functions)} functions, {len(constants)} constants)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
