"""Game.community_hedonic on stock igraph with the package's own C core
(best-response local moving, partitions only).

The pinned memberships and qualities are results recorded on macOS arm64
(Apple clang -O3). The core is written to give the same bits everywhere
(no contraction, explicit fma(), fixed summation orders), but the pins are
skipped unless HIG_PINNED=1, which ``tools/native_tests.py`` sets.
"""

import os
import random
import unittest

import igraph as ig

from hedonic import Game

ZACHARY_SEED0 = [0, 0, 0, 0, 1, 2, 2, 0, 3, 0, 1, 1, 0, 0, 3, 3, 2, 0, 3, 0, 3, 0, 3, 3, 4,
                 4, 3, 4, 4, 3, 3, 4, 3, 3]

PINNED = unittest.skipUnless(os.environ.get("HIG_PINNED") == "1",
                             "bit-level pins: HIG_PINNED=1")


def _karate():
    return Game(ig.Graph.Famous("Zachary"))


class TestBackendRouting(unittest.TestCase):
    @PINNED
    def test_partition_matches_native_pin(self):
        p = _karate().community_hedonic(initial_membership=None, seed=0)
        self.assertIsInstance(p, ig.VertexClustering)
        self.assertEqual(p.membership, ZACHARY_SEED0)
        self.assertEqual(p.quality.hex(), (0.4306184012066366).hex())

    def test_game_and_native_layer_agree(self):
        from hedonic import _native

        for kw in (dict(), dict(resolution=0.1)):
            a = _karate().community_hedonic(initial_membership=None, seed=0, **kw)
            b = _native.community_hedonic(_karate(), seed=0, **kw)
            self.assertEqual(list(a), list(b), kw)
        with self.assertRaisesRegex(ValueError, "partitions only"):
            _native.community_hedonic(_karate(), seed=0, max_memberships=3, resolution=0.1)

    def test_seed_ignores_the_caller_generator(self):
        a = _karate().community_hedonic(initial_membership=None, seed=5)
        ig.set_random_number_generator(random.Random(999))
        try:
            b = _karate().community_hedonic(initial_membership=None, seed=5)
        finally:
            ig.set_random_number_generator(random)
        self.assertEqual(a.membership, b.membership)

    def test_versions_name_the_package_core_and_binding(self):
        from hedonic import _core

        versions = _karate().community_hedonic(seed=0)._hedonic_provenance["versions"]
        self.assertEqual(set(versions), {"hedonic", "hedonic_core", "python_igraph"})
        self.assertEqual(versions["hedonic_core"], _core.core_version)
        self.assertEqual(versions["python_igraph"], ig.__version__)
        # The C core names the release it belongs to (a .devN or rcN tag only adds a suffix).
        import re
        from importlib.metadata import version

        self.assertEqual(_core.core_version, re.match(r"\d+\.\d+\.\d+", version("hedonic")).group())


if __name__ == "__main__":
    unittest.main()
