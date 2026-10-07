"""Optional experiment-metric bridge for the public Game facade."""

from __future__ import annotations

from numbers import Integral

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

    # This is intentionally a local import: evaluation is an optional
    # experiments concern and its one-to-one matcher uses SciPy.
    from ..experiments.overlapping.metrics import (
        evaluate_cover,
        partition_to_cover_lists,
    )

    predicted = game._memberships_as_cover()
    if hasattr(ground_truth, "membership"):
        truth = partition_to_cover_lists(ground_truth)
    else:
        try:
            truth_values = list(ground_truth)
        except TypeError as exc:
            raise TypeError(
                "ground_truth must be a community cover, partition vector, "
                "or VertexClustering/VertexCover"
            ) from exc
        # A flat vector is unambiguous; nested rows are already interpreted
        # as the list-of-communities cover expected by evaluate_cover.
        if truth_values and all(
            isinstance(value, Integral) and not isinstance(value, bool)
            for value in truth_values
        ):
            if len(truth_values) != game.vcount():
                raise ValueError(
                    "flat ground_truth must contain one label per vertex "
                    f"({game.vcount()})"
                )
            try:
                truth_rows = game._normalize_membership(truth_values)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid ground_truth partition: {exc}") from exc
            by_label: dict[int, list[int]] = {}
            for vertex, row in enumerate(truth_rows):
                by_label.setdefault(row[0], []).append(vertex)
            truth = [by_label[label] for label in sorted(by_label)]
        else:
            truth = [list(community) for community in truth_values]

    scores = evaluate_cover(
        predicted,
        truth,
        game.vcount(),
        compute_omega=metric_key == "omega",
        singleton_mode=singleton_mode,
        matching_weight=matching_weight,
        omega_sample_size=omega_sample_size,
        omega_seed=omega_seed,
    )
    return float(scores[metric_key])
