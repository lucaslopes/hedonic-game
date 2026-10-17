"""Metric bridge for the public Game facade."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

EVALUATION_METHODS = {
    "f1": "f1",
    "symmetric_f1": "f1",
    "symmetric_best_match_f1": "f1",
    "jaccard": "jaccard",
    "symmetric_best_match_jaccard": "jaccard",
    "one_to_one_f1": "matching_f1",
    "matching_f1": "matching_f1",
    "omega": "omega",
    "node_micro_f1": "node_micro_f1",
    "size_weighted_community_f1": "size_weighted_community_f1",
    "size_weighted_f1": "size_weighted_community_f1",
}


def evaluate_against(
    game,
    ground_truth,
    method: str = "f1",
    *,
    singleton_mode: str = "all",
    matching_weight: str = "f1",
    omega_sample_size: int = 100_000,
    omega_seed: int = 0,
) -> float:
    """Score ``game.memberships`` against ``ground_truth`` (see the facade)."""
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    normalized_method = method.strip().lower().replace("-", "_").replace(" ", "_")
    metric_key = EVALUATION_METHODS.get(normalized_method)
    if metric_key is None:
        supported = ", ".join(sorted(EVALUATION_METHODS))
        raise ValueError(
            f"Unknown evaluation method {method!r}; supported methods: {supported}"
        )
    if ground_truth is None:
        raise ValueError("ground_truth is required")

    # A local import: the one-to-one matcher needs SciPy, which is optional.
    from ..metrics import (
        _is_label,
        evaluate_cover,
        flat_labels_to_cover,
        partition_to_cover_lists,
    )

    predicted = game._memberships_as_cover()
    if hasattr(ground_truth, "membership"):
        source = getattr(ground_truth, "graph", None)
        if source is not None and source.vcount() != game.vcount():
            raise ValueError(
                f"ground_truth is a clustering of a graph with {source.vcount()} "
                f"vertices; this graph has {game.vcount()}"
            )
        truth = partition_to_cover_lists(ground_truth)
    else:
        if isinstance(ground_truth, Mapping):
            raise ValueError(
                "ground_truth cannot be a dict; pass one label per vertex in "
                "vertex order, a list of communities (each a list of vertices), "
                "or a VertexClustering/VertexCover"
            )
        try:
            truth_values = list(ground_truth)
        except TypeError as exc:
            raise TypeError(
                "ground_truth must be a community cover, partition vector, "
                "or VertexClustering/VertexCover"
            ) from exc
        # A flat vector (one label per vertex, any hashable labels) is
        # unambiguous; nested entries are communities (lists of vertices).
        if truth_values and all(_is_label(value) for value in truth_values):
            if len(truth_values) != game.vcount():
                raise ValueError(
                    "flat ground_truth must contain one label per vertex "
                    f"({game.vcount()}); a cover is a list of communities, "
                    "each a list of vertices"
                )
            for vertex, label in enumerate(truth_values):
                if label is None or (
                    isinstance(label, (float, np.floating)) and math.isnan(label)
                ):
                    raise ValueError(
                        f"ground_truth: vertex {vertex} has no label ({label!r}); "
                        "a flat reference labels every vertex. To score against "
                        "a reference that leaves vertices out, pass its "
                        "communities (lists of vertices)"
                    )
            truth = flat_labels_to_cover(truth_values)
        else:
            truth = [list(community) for community in truth_values]
    if not any(len(community) for community in truth):
        raise ValueError("ground_truth has no communities")

    scores = evaluate_cover(
        predicted,
        truth,
        game.vcount(),
        compute_omega=metric_key == "omega",
        singleton_mode=singleton_mode,
        matching_weight=matching_weight,
        omega_sample_size=omega_sample_size,
        omega_seed=omega_seed,
        one_to_one=metric_key in ("matching_f1", "node_micro_f1"),
    )
    return float(scores[metric_key])
