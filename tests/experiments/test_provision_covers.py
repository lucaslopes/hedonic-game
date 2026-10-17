"""Every ground-truth cover of a network as its own target, and detection on the subgraph the cover induces."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path

from hedonic.experiments.overlapping import codeseg_reproduction as cr
from hedonic.experiments.overlapping import quickstart as qs

from hedonic.experiments.resources import CONFIG_DIR

COVERS = str(CONFIG_DIR / "paper-benchmark.toml")
PAIR_ORDER = ["dblp:top5000", "amazon:top5000", "youtube:top5000", "livejournal:top5000",
              "dblp:all", "amazon:all", "youtube:all", "livejournal:all"]


def load(path: str = COVERS) -> qs.Config:
    return qs.config_from_args(qs.build_parser().parse_args(["--config", path]))


def digest(argv: list[str]) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
        cr.main([*argv, "--identity-only"])
    return json.loads(buffer.getvalue().strip().splitlines()[-1])["config_digest"]


class TestTargets(unittest.TestCase):
    def test_a_network_may_name_its_cover(self):
        self.assertEqual(qs.parse_targets("dblp,amazon:top5000,lj:all"), ["dblp", "amazon:top5000", "livejournal:all"])

    def test_a_cover_the_network_does_not_have_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no cover 'top5k'"):
            qs.parse_targets("wikipedia:top5000")
        with self.assertRaisesRegex(ValueError, "unknown network"):
            qs.parse_targets("nope:all")

    def test_covers_expand_network_by_network(self):
        self.assertEqual(qs.expand_covers(["dblp", "amazon"], ["all", "top5000"]),
                         ["dblp:all", "dblp:top5000", "amazon:all", "amazon:top5000"])
        # a network that already names its cover is kept as it is, not crossed with the others
        self.assertEqual(qs.expand_covers(["dblp:top5000", "amazon"], "all,top5000"),
                         ["dblp:top5000", "amazon:all", "amazon:top5000"])

    def test_the_catalogue_answers_for_a_target(self):
        self.assertEqual(qs.NETWORKS["dblp:top5000"], ("DBLP · top5k", *qs.NETWORKS["dblp"][1:]))
        self.assertEqual(qs.NETWORKS.get("nope:all"), None)

    def test_the_covers_flag_and_induce_flag(self):
        cfg = qs.config_from_args(qs.build_parser().parse_args(
            ["--networks", "dblp,youtube", "--covers", "all,top5000", "--induce", "covered", "--full"]))
        self.assertEqual(cfg.networks, ["dblp:all", "dblp:top5000", "youtube:all", "youtube:top5000"])
        self.assertEqual(cfg.induce, "covered")
        self.assertIn("--induce covered", cfg.command())
        self.assertIn("dblp:all,dblp:top5k", cfg.command())

    def test_robustness_is_accepted_for_covers_and_the_induced_graph(self):
        for argv in (["--networks", "dblp", "--induce", "covered", "--full", "--robustness"],
                     ["--networks", "dblp:top5000", "--full", "--robustness"]):
            cfg = qs.config_from_args(qs.build_parser().parse_args(argv))
            self.assertTrue(cfg.robustness and cfg.save_covers)
        with self.assertRaisesRegex(ValueError, "complete graphs"):
            qs.config_from_args(qs.build_parser().parse_args(["--networks", "dblp", "--nodes", "500", "--robustness"]))


class TestUnits(unittest.TestCase):
    def cfg(self, **kw):
        return qs.Config(networks=["dblp"], methods=["hedonic"], seeds=[0], nodes=0, output_dir="/o", cache_dir="/c", **kw)

    def test_a_plain_network_keeps_its_folder_and_its_arguments(self):
        argv, out = qs.unit_argv(self.cfg(), "dblp", 0, qs.BY_KEY["hedonic"], None)
        self.assertEqual(out, Path("/o/dblp-full/seed0"))
        self.assertNotIn("--cover", argv)
        self.assertNotIn("--induce", argv)

    def test_a_cover_and_the_induced_graph_get_their_own_folder_and_arguments(self):
        argv, out = qs.unit_argv(self.cfg(induce="covered"), "dblp:top5000", 3, qs.BY_KEY["hedonic"], None)
        self.assertEqual(out, Path("/o/dblp-top5000-full-covered/seed3"))
        self.assertEqual(argv[argv.index("--datasets") + 1], "dblp")
        self.assertEqual(argv[argv.index("--cover") + 1], "top5000")
        self.assertEqual(argv[argv.index("--induce") + 1], "covered")

    def test_the_whole_graph_records_keep_their_identity_and_the_induced_ones_have_another(self):
        cfg = self.cfg()
        whole, _ = qs.unit_argv(cfg, "dblp", 0, qs.BY_KEY["hedonic"], None)
        covered, _ = qs.unit_argv(self.cfg(induce="covered"), "dblp", 0, qs.BY_KEY["hedonic"], None)
        top, _ = qs.unit_argv(cfg, "dblp:top5000", 0, qs.BY_KEY["hedonic"], None)
        self.assertEqual(len({digest(whole), digest(covered), digest(top)}), 3)
        without_flag = [a for a in whole]  # the flag-free argv is what every earlier record was made with
        self.assertEqual(digest(without_flag), digest(whole))

    def test_settings_of_a_network_apply_to_its_covers_and_a_target_may_override(self):
        cfg = self.cfg(method_params={"hedonic": {"dblp": {"max_memberships": 9}, "dblp:top5000": {"max_memberships": 5}}})
        m = qs.BY_KEY["hedonic"]
        self.assertEqual(qs.method_params(m, cfg, "dblp:all")["max_memberships"], 9)
        self.assertEqual(qs.method_params(m, cfg, "dblp:top5000")["max_memberships"], 5)


class TestShippedCoversConfig(unittest.TestCase):
    def test_four_networks_two_covers_nine_methods_ten_seeds(self):
        cfg = load()
        self.assertEqual(cfg.networks, PAIR_ORDER)
        self.assertEqual(cfg.methods, ["hedonic", "codeseg", "fox", "clique_percolation", "angel", "bigclam", "ego_splitting",
                                       "ncgame", "neo_kmeans"])
        self.assertNotIn("hoc_multilevel", cfg.methods)
        self.assertEqual(cfg.induce, "covered")
        self.assertTrue(cfg.robustness and cfg.save_covers)
        self.assertEqual(len(qs.build_plan(cfg, {})), 8 * 9 * 10)

    def test_for_a_seed_each_pair_is_finished_with_all_methods_before_the_next_top5000_covers_first(self):
        cfg = load()
        plan = qs.build_plan(cfg, {})
        self.assertEqual({seed for _, seed, _ in plan[:72]}, {0})
        for number, pair in enumerate(PAIR_ORDER):
            self.assertEqual([net for net, _, _ in plan[9 * number:9 * (number + 1)]], [pair] * 9)
        self.assertEqual([m.key for _, _, m in plan[:9]], cfg.methods)
        self.assertEqual(plan[72][1], 1)

    def test_hedonic_is_called_hedonic(self):
        self.assertEqual(qs.BY_KEY["hedonic"].label, "Hedonic")

    def test_every_network_has_hedonic_settings(self):
        cfg = load()
        for net in cfg.networks:
            params = qs.method_params(qs.BY_KEY["hedonic"], cfg, net)
            self.assertIn("max_memberships", params)
            self.assertIn("absolute_resolution", params)


if __name__ == "__main__":
    unittest.main()


class TestAuditFindsTargetFolders(unittest.TestCase):
    def test_a_target_sees_only_its_own_folders_and_induce_filters_them(self):
        import tempfile

        from hedonic.experiments.overlapping import cover_audit as ca

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("dblp-full", "dblp-top5000-full", "dblp-full-covered", "dblp-top5000-full-covered", "dblp-n5000"):
                record = root / folder / "seed0" / "runs" / "dblp" / "hedonic_local.json"
                record.parent.mkdir(parents=True)
                record.write_text("{}")
            names = lambda net, induce=None: sorted(p.parents[3].name for p in
                                                   ca.record_files(root, net, 0, "hedonic_local", induce))
            self.assertEqual(names("dblp"), ["dblp-full", "dblp-full-covered", "dblp-n5000"])
            self.assertEqual(names("dblp", "full"), ["dblp-full", "dblp-n5000"])
            self.assertEqual(names("dblp:top5000", "covered"), ["dblp-top5000-full-covered"])
            self.assertEqual(names("dblp:top5000"), ["dblp-top5000-full", "dblp-top5000-full-covered"])

    def test_the_audit_plan_accepts_targets_and_refuses_unknown_networks(self):
        import tempfile

        from hedonic.experiments.overlapping import cover_audit as ca

        with tempfile.TemporaryDirectory() as tmp:
            plan = ca.resolve(["--source", tmp, "--networks", "dblp:top5000,amazon", "--methods", "clique_percolation", "--seeds", "0",
                               "--induce", "covered", "--output-dir", str(Path(tmp) / "out")])
            self.assertEqual(plan["networks"], ["dblp:top5000", "amazon"])
            self.assertEqual(plan["induce"], "covered")
            self.assertEqual(ca.key_of("dblp:top5000", 0, "clique_percolation", "cover"), "dblp:top5000/seed0/clique_percolation/cover")
            with self.assertRaises(ValueError):
                ca.resolve(["--source", tmp, "--networks", "nosuchnet:top5000", "--methods", "clique_percolation"])
