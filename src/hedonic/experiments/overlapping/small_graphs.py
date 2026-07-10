"""
Quick overlapping hedonic tests on small graphs via
``Game.community_hedonic(..., max_memberships=K)``.
"""

from __future__ import annotations

import igraph as ig
import numpy as np

from hedonic import Game
from hedonic.experiments.overlapping.metrics import (
    cover_quality,
    evaluate_cover,
    in_equilibrium_overlapping,
    partition_to_cover_lists,
    quality_overlapping_cpm,
)

np.random.seed(42)


def separator(title: str) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print("=" * 55)


def run_tests() -> None:
    # -------------------------------------------------------------------
    # Test 1: small famous graph
    # -------------------------------------------------------------------
    separator("TEST 1: Petersen graph (n=10)")

    g = Game(ig.Graph.Famous("Petersen"))
    res = g.density()
    print(f"Resolution (density): {res:.4f}")

    cover = g.community_hedonic(
        resolution=res, n_iterations=-1, max_memberships=4
    )
    cover_lists = partition_to_cover_lists(cover)
    print(f"Communities found: {len(cover_lists)}")
    for i, members in enumerate(cover_lists):
        print(f"  Community {i}: {sorted(members)}")

    q = cover_quality(cover)
    if q is None:
        q = quality_overlapping_cpm(g, cover_lists, resolution=res)
    print(f"Overlapping CPM quality (Q): {q:.6f}")

    eq = in_equilibrium_overlapping(g, cover_lists, resolution=res)
    print(f"Nash equilibrium: {eq}")

    # -------------------------------------------------------------------
    # Test 2: synthetic SBM with known blocks
    # -------------------------------------------------------------------
    separator("TEST 2: Synthetic SBM with two blocks")

    n_per_block = 10
    n_blocks = 2
    n = n_per_block * n_blocks

    block_sizes = [n_per_block] * n_blocks
    pref_matrix = [
        [0.6, 0.05],
        [0.05, 0.6],
    ]
    g_sbm = Game(ig.Graph.SBM(pref_matrix, block_sizes, directed=False))

    gt = [list(range(0, 10)), list(range(10, 20))]

    res_sbm = 0.1
    cover_sbm = g_sbm.community_hedonic(
        resolution=res_sbm, n_iterations=-1, max_memberships=4
    )
    cover_sbm_lists = partition_to_cover_lists(cover_sbm)
    print(f"Resolution: {res_sbm}")
    print(f"Communities found: {len(cover_sbm_lists)}")
    for i, members in enumerate(cover_sbm_lists):
        print(f"  Community {i}: {sorted(members)}")

    q_sbm = cover_quality(cover_sbm)
    if q_sbm is None:
        q_sbm = quality_overlapping_cpm(g_sbm, cover_sbm_lists, resolution=res_sbm)
    eq_sbm = in_equilibrium_overlapping(g_sbm, cover_sbm_lists, resolution=res_sbm)
    print(f"Quality Q: {q_sbm:.6f}")
    print(f"Nash equilibrium: {eq_sbm}")

    metrics = evaluate_cover(cover_sbm_lists, gt, n)
    print(f"F1 vs ground truth:      {metrics['f1']:.4f}")
    print(f"Jaccard vs ground truth: {metrics['jaccard']:.4f}")
    print(f"Omega index:             {metrics['omega']:.4f}")

    # -------------------------------------------------------------------
    # Test 3: overlapping quality vs non-overlapping
    # -------------------------------------------------------------------
    separator("TEST 3: Overlapping quality vs non-overlapping")

    g3 = Game(ig.Graph.Famous("Petersen"))
    res3 = g3.density()

    p_no = g3.community_hedonic(
        resolution=res3, n_iterations=-1, max_memberships=1
    )
    cover_no = partition_to_cover_lists(p_no)
    q_no = cover_quality(p_no)
    if q_no is None:
        q_no = quality_overlapping_cpm(g3, cover_no, resolution=res3)

    cover_ov = g3.community_hedonic(
        resolution=res3,
        n_iterations=-1,
        max_memberships=4,
        initial_membership=list(p_no.membership),
    )
    cover_ov_lists = partition_to_cover_lists(cover_ov)
    q_ov = cover_quality(cover_ov)
    if q_ov is None:
        q_ov = quality_overlapping_cpm(g3, cover_ov_lists, resolution=res3)

    print(f"Q non-overlapping: {q_no:.6f}  ({len(cover_no)} communities)")
    print(f"Q overlapping:     {q_ov:.6f}  ({len(cover_ov_lists)} communities)")
    print(
        f"ΔQ = {q_ov - q_no:+.6f}  "
        f"{'improved' if q_ov >= q_no else 'worsened'}"
    )

    print("\nAll small-graph tests completed.\n")


def main(argv=None) -> int:
    """CLI entry: optional argv accepted for ``hedonic-exp`` compatibility."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Quick overlapping tests on small graphs "
            "(Petersen + tiny SBM). No data directories required."
        )
    )
    parser.parse_args(argv)
    run_tests()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
