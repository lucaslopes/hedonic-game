"""Smoke-test the public API from an installed Hedonic wheel.

The verifier is executed after installing an immutable PyPI release.  It
checks the installed distribution metadata and exercises the direction-
preserving ``Game`` copy/round-trip behavior without importing the checkout.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
from pathlib import Path

import igraph as ig

import hedonic
from hedonic import Game


EXPECTED_NATIVE_VERSION = "1.0.0.5"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_import_origin() -> None:
    """Require the wrapper import to come from the installed distribution."""
    distribution = importlib.metadata.distribution("hedonic")
    expected = Path(distribution.locate_file("hedonic/__init__.py")).resolve()
    actual_file = getattr(hedonic, "__file__", None)
    _require(actual_file is not None, "imported hedonic module has no __file__")
    actual = Path(actual_file).resolve()
    _require(
        actual == expected,
        "release verifier imported hedonic from outside the installed "
        f"distribution: actual={actual!s}, expected={expected!s}",
    )


def main() -> None:
    expected_version = os.environ.get("HEDONIC_EXPECTED_VERSION")
    installed_version = importlib.metadata.version("hedonic")
    if expected_version:
        _require(
            installed_version == expected_version,
            f"installed Hedonic version {installed_version!r} does not match "
            f"the requested uploaded release {expected_version!r}",
        )

    verify_import_origin()

    native_version = importlib.metadata.version("lucas-igraph")
    _require(
        native_version == EXPECTED_NATIVE_VERSION,
        "uploaded Hedonic release resolved an unexpected lucas-igraph "
        f"version: {native_version!r}",
    )
    requirements = importlib.metadata.requires("hedonic") or []
    _require(
        any(
            re.match(
                rf"^lucas-igraph=={re.escape(EXPECTED_NATIVE_VERSION)}(?:$|[;])",
                item,
            )
            for item in requirements
        ),
        f"installed Hedonic metadata does not pin lucas-igraph=={EXPECTED_NATIVE_VERSION}",
    )

    source = ig.Graph(n=3, edges=[(0, 1), (1, 2)], directed=True)
    source.vs["name"] = ["source", "middle", "sink"]
    source.es["weight"] = [2.0, 3.0]
    source["fixture"] = "installed-wheel-directed-copy"

    game = Game(source)
    _require(game.is_directed(), "Game(Graph) dropped the directed flag")
    _require(game.get_edgelist() == source.get_edgelist(), "edge order changed")
    _require(game.vs["name"] == source.vs["name"], "vertex attributes changed")
    _require(game.es["weight"] == source.es["weight"], "edge attributes changed")
    _require(game["fixture"] == source["fixture"], "graph attributes changed")

    round_trip = game.to_igraph()
    _require(round_trip.is_directed(), "to_igraph() dropped the directed flag")
    _require(
        round_trip.get_edgelist() == source.get_edgelist(),
        "to_igraph() changed the edge order",
    )

    for cap in (1, 2):
        try:
            game.community_hedonic(
                max_memberships=cap,
                resolution=0.1,
                n_iterations=1,
            )
        except ValueError as exc:
            _require(
                "community_hedonic requires an undirected graph" in str(exc),
                f"unexpected directed-detector error: {exc}",
            )
        else:  # pragma: no cover - release failure path
            raise AssertionError("directed detection was not rejected")


if __name__ == "__main__":
    main()
    print("installed-wheel metadata and directed-copy smoke passed")
