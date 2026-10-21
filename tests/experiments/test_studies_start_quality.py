"""The start-quality curve: start levels, resumable records, CLI routing and the dry run (small graphs only)."""

from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import igraph as ig

from hedonic.experiments import CLI, runtime
from hedonic.experiments.overlapping import gt_spectrum as gs
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import robustness as rb
from hedonic.experiments.overlapping import start_quality as sq
from hedonic.experiments.resources import config_path

SHIPPED = "start-quality-hedonic"  # the shipped config, resolved by name through resources.config_path
# Two overlapping cliques: A = {0,1,2,3} and B = {3,4,5,6,7}; vertex 3 is in both.
GT = [[0, 1, 2, 3], [3, 4, 5, 6, 7]]
N = 8


def tiny_graph() -> ig.Graph:
    edges = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    edges += [(a, b) for a in range(3, 8) for b in range(a + 1, 8)]
    return ig.Graph(n=N, edges=edges)


def tiny_job():
    """A real ``gt_spectrum.Job`` on the tiny graph (only the prepared dataset is stubbed)."""
    prepared = SimpleNamespace(dataset=SimpleNamespace(graph=tiny_graph(), cover=GT))
    return gs.Job("tiny", "top5000", prepared, None)


def plan_for(folder, *, levels, seeds, cap=2, gamma=0.5, overwrite=False, phase="local") -> dict:
    return {"config": None, "phase": phase, "networks": ["tiny"], "seeds": list(seeds), "levels": list(levels),
            "output": Path(folder), "settings": {"tiny": {"cap": cap, "gamma": gamma, "source": "test"}},
            "dry_run": False, "overwrite": overwrite}


class TestSwapLevels(unittest.TestCase):
    def test_swaps_preserve_counts_and_sizes_and_record_the_realized_distance(self):
        reference = rb.canonicalize_cover(GT, N)
        reference_memberships = rb.cover_to_vertex_memberships(reference, N)
        incidences = sum(map(len, reference_memberships))
        for distance in (0.0, 0.25, 0.5, 1.0):
            built = sq.swap_start(reference, N, distance, seed=103)
            self.assertEqual([len(labels) for labels in built["memberships"]],
                             [len(labels) for labels in reference_memberships])
            self.assertEqual(sorted(map(len, built["cover"])), sorted(map(len, reference)))
            self.assertTrue(built["meta"]["vertex_membership_counts_preserved"])
            self.assertTrue(built["meta"]["community_sizes_preserved"])
            self.assertEqual(built["swaps_requested"], round(distance * incidences / 2))
            self.assertLessEqual(built["swaps_successful"], built["swaps_requested"])
            realized = built["realized_distance"]
            self.assertTrue(0.0 <= realized <= 1.0)
            lost = realized * incidences  # the lost incidences are a whole number
            self.assertAlmostEqual(lost, round(lost), places=9)
            self.assertLessEqual(lost, 2 * built["swaps_successful"] + 1e-9)
            if distance == 0.0:
                self.assertEqual(realized, 0.0)
                self.assertEqual(built["swaps_successful"], 0)

    def test_one_successful_swap_loses_exactly_two_incidences(self):
        reference = rb.canonicalize_cover(GT, N)
        incidences = sum(map(len, rb.cover_to_vertex_memberships(reference, N)))
        built = sq.swap_start(reference, N, 0.25, seed=103)  # round(0.25 * 9 / 2) = 1 swap requested
        self.assertEqual(built["swaps_requested"], 1)
        if built["swaps_successful"] == 1:
            self.assertAlmostEqual(built["realized_distance"], 2 / incidences)


def random_cover(rng, n_vertices, n_communities, max_size):
    """A random cover of every vertex (a vertex the draw missed joins a random community)."""
    cover = [set(rng.sample(range(n_vertices), rng.randint(1, max_size))) for _ in range(n_communities)]
    for vertex in set(range(n_vertices)) - set().union(*cover):
        rng.choice(cover).add(vertex)
    return [sorted(community) for community in cover]


class TestFastSwaps(unittest.TestCase):
    """``start_quality.perturb_cover_incidence`` must return exactly what ``robustness.perturb_cover_incidence`` does."""

    def assert_same(self, cover, n_vertices, **kwargs):
        reference = rb.perturb_cover_incidence(cover, n_vertices, **kwargs)
        fast = sq.perturb_cover_incidence(cover, n_vertices, **kwargs)
        self.assertEqual(fast, reference)

    def test_identical_to_the_reference_on_random_covers(self):
        import random

        rng = random.Random(20260512)
        for case in range(250):
            n_vertices = rng.randint(4, 40)
            cover = random_cover(rng, n_vertices, rng.randint(2, 12), rng.randint(1, max(1, n_vertices // 2)))
            swaps = rng.choice([0, 1, 2, 5, 20, 80])
            self.assert_same(cover, n_vertices, swaps=swaps, seed=rng.randint(0, 10**6))

    def test_identical_when_the_blocks_split_and_merge(self):
        import random

        rng = random.Random(7)
        for load in (1, 2, 3, 5):
            with patch.object(sq._SortedPairs, "LOAD", load):
                for _ in range(40):
                    n_vertices = rng.randint(6, 30)
                    cover = random_cover(rng, n_vertices, rng.randint(3, 9), n_vertices // 2)
                    self.assert_same(cover, n_vertices, swaps=rng.randint(1, 40), seed=rng.randint(0, 10**6))

    def test_identical_at_the_edges(self):
        self.assert_same(GT, N, swaps=0, seed=1)
        self.assert_same(GT, N, swaps=3, seed=1, max_attempts=2)  # attempts exhausted before the swaps are done
        self.assert_same([[0, 1]], 2, swaps=4, seed=2)  # one incidence pair: nothing to switch
        self.assert_same([[0], [1]], 2, swaps=5, seed=3)  # two incidences, no valid switch against the duplicate rule
        self.assert_same([[0, 1, 2]], 3, swaps=2, seed=4)  # one community
        self.assert_same([[0, 1], [0, 1], [2]], 3, swaps=5, seed=5)  # a duplicate body is merged before the switches
        with self.assertRaises(ValueError):
            sq.perturb_cover_incidence(GT, N, swaps=-1, seed=0)

    def test_memory_does_not_grow_with_the_number_of_switches(self):
        """The first version kept every body it had ever formed (two per switch): on DBLP that was 70-170 GB."""
        import random
        import tracemalloc

        rng = random.Random(3)
        n_vertices = 3000
        cover = [sorted(rng.sample(range(n_vertices), 300)) for _ in range(40)]
        for vertex in set(range(n_vertices)) - set().union(*map(set, cover)):
            cover[vertex % len(cover)].append(vertex)

        def peak(swaps):
            tracemalloc.start()
            try:
                _, meta = sq.perturb_cover_incidence(cover, n_vertices, swaps=swaps, seed=5)
                return tracemalloc.get_traced_memory()[1], meta["successful_swaps"]
            finally:
                tracemalloc.stop()

        small, done_small = peak(100)
        large, done_large = peak(4000)
        self.assertEqual((done_small, done_large), (100, 4000))
        self.assertLess(large, 2 * small)  # 40 times the switches: a leak of two bodies per switch would be ~10x

    def test_sorted_pairs_follow_a_plain_sorted_set(self):
        import random

        rng = random.Random(11)
        for load in (1, 2, 4):
            with patch.object(sq._SortedPairs, "LOAD", load):
                universe = [(v, l) for v in range(12) for l in range(6)]
                present = set(rng.sample(universe, 30))
                order = sq._SortedPairs(sorted(present))
                for _ in range(400):
                    item = rng.choice(universe)
                    if item in present:
                        present.remove(item)
                        order.remove(item)
                    else:
                        present.add(item)
                        order.add(item)
                    self.assertEqual(len(order), len(present))
                    if present:
                        expected = sorted(present)
                        k = rng.randrange(len(expected))
                        self.assertEqual(order.kth(k), expected[k])
                self.assertEqual([x for block in order.blocks for x in block], sorted(present))


class TestRandomLevel(unittest.TestCase):
    def test_random_start_keeps_counts_and_draws_only_existing_labels(self):
        job = tiny_job()
        start = job.start(1)  # cap 1 is below vertex 3's multiplicity: the start is cap-projected
        self.assertIsNotNone(start.projection)
        self.assertEqual(rb.cover_to_vertex_memberships(start.cover, N), start.memberships)
        expected_counts = [min(len(labels), 1) for labels in job.memberships]
        for seed in (200, 201, 202):
            random_memberships = sq.random_start(start.memberships, len(start.cover), seed=seed)
            self.assertEqual([len(labels) for labels in random_memberships], expected_counts)
            for labels in random_memberships:
                self.assertTrue(set(labels) <= set(range(len(start.cover))))
                self.assertEqual(len(set(labels)), len(labels))
            self.assertTrue(0.0 <= sq.label_loss(start.memberships, random_memberships) <= 1.0)

    def test_random_start_is_reproducible_for_a_seed(self):
        reference = [[0], [0, 1], [1]]
        self.assertEqual(sq.random_start(reference, 2, seed=7), sq.random_start(reference, 2, seed=7))

    def test_contiguous_labels_renumbers_gaps_in_order_and_leaves_a_contiguous_start_alone(self):
        self.assertEqual(sq.contiguous_labels([[0, 3], [3], [5, 0]]), [[0, 1], [1], [2, 0]])
        whole = [[0, 1], [1], [2, 0]]
        self.assertIs(sq.contiguous_labels(whole), whole)  # nothing is rebuilt, so no earlier record can change

    def test_a_random_start_that_leaves_a_label_undrawn_still_runs(self):
        """On complete YouTube nine seeds in ten leave 2-4 of 14,827 labels undrawn and the detector refused them."""
        prepared = SimpleNamespace(dataset=SimpleNamespace(graph=tiny_graph(), cover=[[0, 1], [2, 3], [4, 5], [6, 7]]))
        job = gs.Job("tiny", "top5000", prepared, None)
        labels = len(job.start(2).cover)
        gapped = [seed for seed in range(60)
                  if len({label for row in sq.random_start(job.start(2).memberships, labels, seed=sq.RANDOM_SEED_BASE + seed)
                          for label in row}) < labels]
        self.assertTrue(gapped, "the fixture must contain a random start with a gap")
        record = sq.run_condition(job, {"cap": 2, "gamma": 0.5}, "random", gapped[0], "k", "local")
        self.assertTrue(0.0 <= record["final_symmetric_f1"] <= 1.0)
        self.assertLess(record["start_communities"], labels)  # the undrawn label is an empty community

    def test_no_cap_starts_from_the_whole_reference_and_runs(self):
        """cap -1 (no cap): level 0 is the reference itself, never a projection, and the detector runs uncapped."""
        cover = [[0, 1, 2], [2, 3, 4], [2, 5, 6], [6, 7, 0]]  # vertex 2 is in three communities
        prepared = SimpleNamespace(dataset=SimpleNamespace(graph=tiny_graph(), cover=cover))
        job = gs.Job("tiny", "top5000", prepared, None)
        record = sq.run_condition(job, {"cap": -1, "gamma": 0.1}, 0, 0, "k", "local")
        self.assertEqual(record["cap"], -1)
        self.assertEqual(record["start_symmetric_f1"], 1.0)  # the exact reference
        self.assertTrue(0.0 <= record["final_symmetric_f1"] <= 1.0)

    def test_the_reference_cap_is_the_largest_reference_multiplicity_and_rides_along_in_the_key_and_argv(self):
        """--cap reference (leveraging the ground truth): the cap is read from the loaded reference, the key names the
        source, and a resume rebuilds the same plan."""
        cover = [[0, 1, 2], [2, 3, 4], [2, 5, 6], [6, 7, 0]]  # vertex 2 is in three communities
        prepared = SimpleNamespace(dataset=SimpleNamespace(graph=tiny_graph(), cover=cover))
        job = gs.Job("tiny", "top5000", prepared, None)
        setting = sq.with_cap({"cap": -1, "gamma": 0.1, "source": "test"}, "reference")
        self.assertEqual(sq.setting_cap(job, setting), 3)
        record = sq.run_condition(job, setting, 0, 0, "k", "local")
        self.assertEqual((record["cap"], record["cap_source"]), (3, "reference"))
        self.assertLessEqual(record["final_max_memberships"], 3)
        self.assertIn("|cap=reference|", sq.condition_key("tiny", 0, 0, setting, "local"))
        self.assertEqual(sq.with_cap({"cap": 4}, "tuned"), {"cap": 4})
        plan = {"networks": ["tiny"], "phase": "local", "seeds": [0], "levels": [0], "output": "/x", "workers": 1,
                "settings": {"tiny": setting}}
        self.assertEqual(sq.launch_argv(plan)[-2:], ["--cap", "reference"])


class TestResume(unittest.TestCase):
    def test_rerun_skips_recorded_conditions_and_extends_by_new_ones(self):
        job = tiny_job()
        loaded = []

        def loader(network):
            loaded.append(network)
            return job

        with tempfile.TemporaryDirectory() as folder:
            results = Path(folder) / sq.RESULTS_NAME
            emitted: list[str] = []
            plan = plan_for(folder, levels=[0.0, "random"], seeds=[0])
            self.assertEqual(sq.run_study(plan, loader=loader, emit=emitted.append), 0)
            records = [json.loads(line) for line in results.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            for record in records:
                self.assertTrue(0.0 <= record["final_symmetric_f1"] <= 1.0)
                self.assertTrue(0.0 <= record["start_symmetric_f1"] <= 1.0)
                self.assertTrue(math.isfinite(record["seconds"]))
                self.assertIn("runtime", record)
            progress = [line for line in emitted if line.startswith("[")]
            self.assertEqual([line.split()[0] for line in progress], ["[1/2]", "[2/2]"])

            # A plain rerun: nothing pending, so the graph is not even loaded.
            loaded.clear()
            emitted.clear()
            sq.run_study(plan, loader=loader, emit=emitted.append)
            self.assertEqual(loaded, [])
            self.assertIn("2 already", emitted[0])
            self.assertEqual(len(results.read_text().splitlines()), 2)

            # A new seed runs only the new conditions.
            sq.run_study(plan_for(folder, levels=[0.0, "random"], seeds=[0, 1]), loader=loader, emit=lambda _: None)
            self.assertEqual(len(results.read_text().splitlines()), 4)

            # A changed setting (gamma) is a different condition, so it is computed again.
            sq.run_study(plan_for(folder, levels=[0.0], seeds=[0], gamma=0.25), loader=loader, emit=lambda _: None)
            self.assertEqual(len(results.read_text().splitlines()), 5)

            # A torn last line (an interrupted write) is cut off before the next record is appended.
            with results.open("a", encoding="utf-8") as handle:
                handle.write('{"key": "torn')
            sq.run_study(plan_for(folder, levels=[0.0, "random"], seeds=[0, 1, 2]), loader=loader,
                         emit=lambda _: None)
            lines = results.read_text().splitlines()
            self.assertEqual(len(lines), 7)
            parsed = [json.loads(line) for line in lines]
            self.assertEqual(len({record["key"] for record in parsed}), 7)
            self.assertTrue((Path(folder) / sq.SUMMARY_NAME).is_file())

    def test_overwrite_deletes_the_records_first(self):
        job = tiny_job()
        with tempfile.TemporaryDirectory() as folder:
            results = Path(folder) / sq.RESULTS_NAME
            sq.run_study(plan_for(folder, levels=[0.0], seeds=[0]), loader=lambda _: job, emit=lambda _: None)
            self.assertEqual(len(results.read_text().splitlines()), 1)
            sq.run_study(plan_for(folder, levels=[0.0], seeds=[0], overwrite=True), loader=lambda _: job,
                         emit=lambda _: None)
            self.assertEqual(len(results.read_text().splitlines()), 1)


class TestPlanAndConfig(unittest.TestCase):
    def test_the_shipped_config_resolves_by_name_path_or_prefixed_path(self):
        shipped = config_path(SHIPPED)
        self.assertTrue(shipped.is_file())
        # The name it had before the method's rename (a queue or a registered run may still give it) is the same file.
        for spelling in (SHIPPED, f"configs/{SHIPPED}.toml", str(shipped), "start-quality-hoc-local",
                         "configs/start-quality-hoc-local.toml"):
            plan = sq.resolve(["--config", spelling])
            self.assertEqual(plan["config"], str(shipped.resolve()))
            self.assertEqual(len(plan["levels"]), 9)
            self.assertEqual(plan["levels"][-1], "random")
        with self.assertRaises(ValueError):
            sq.resolve(["--config", "no-such-start-quality-config"])

    def test_the_output_is_the_flag_else_the_runs_root_never_the_data(self):
        with tempfile.TemporaryDirectory() as folder:
            for flag in ("--output", "--output-dir"):
                plan = sq.resolve(["--networks", "dblp", "--seeds", "0", flag, folder])
                self.assertEqual(plan["output"], Path(folder).resolve())
        default = sq.resolve(["--networks", "dblp", "--seeds", "0"])
        self.assertEqual(default["output"], Path(sq.DEFAULT_OUTPUT).resolve())
        from hedonic.experiments import config

        self.assertFalse(Path(default["output"]).resolve().is_relative_to(Path(config.NETWORKS_DIR).resolve()))

    def test_config_rejects_unknown_keys_and_bad_levels(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "bad.toml"
            config.write_text('[start_quality]\nnetworks = ["dblp"]\ncap = 3\n')
            with self.assertRaises(ValueError):
                sq.resolve(["--config", str(config)])
        with self.assertRaises(ValueError):
            sq.parse_levels("0,1.5")
        with self.assertRaises(ValueError):
            sq.parse_levels("0,0")
        with self.assertRaises(ValueError):
            sq.parse_networks("dblp,unknown-network")

    def test_dry_run_prints_the_plan_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "study"
            buffer = io.StringIO()
            with patch.object(sq, "run_study", side_effect=AssertionError("dry run ran the study")), \
                    redirect_stdout(buffer):
                code = sq.main(["--networks", "dblp", "--seeds", "0", "--levels", "0,random",
                                "--output", str(output), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())
            text = buffer.getvalue()
            self.assertIn("2 conditions = 1 networks", text)
            self.assertIn("dry run", text)

    def test_front_dry_run_starts_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            buffer = io.StringIO()
            with patch("hedonic.experiments.overlapping.runmanager.launch_startcurve",
                       side_effect=AssertionError("dry run launched a run")), redirect_stdout(buffer):
                code = sq.front_main(["--config", SHIPPED, "--output-dir", str(Path(folder) / "run"), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse((Path(folder) / "run").exists())
            self.assertIn("hedonic run startcurve", buffer.getvalue())


class TestProgressAndViewer(unittest.TestCase):
    def test_progress_line_is_read_by_the_durable_viewer_pattern(self):
        record = {"network": "dblp", "level": "0.05", "detector_seed": 2, "seconds": 1.5,
                  "start_symmetric_f1": 0.7, "final_symmetric_f1": 0.6}
        found = sq.PROGRESS_LINE.match(sq.progress_line(3, 135, record))
        self.assertIsNotNone(found)
        self.assertEqual((found[1], found[2]), ("3", "135"))
        self.assertIn("level=0.05", found[3])

    def test_snapshot_renders_a_completed_run(self):
        entry = {"name": "startcurve-test", "created": 0.0, "backend": "process", "session": "",
                 "output_dir": "/tmp/none"}
        prog = {"status": "completed", "done": 135, "total": 135, "started": 0.0, "finished": 60.0,
                "describe": "start-quality curve", "output": "/tmp/none/start_quality.jsonl"}
        with patch("hedonic.experiments.overlapping.runmanager.effective_status", return_value="completed"):
            text = sq.snapshot(entry, prog, "●")
        self.assertIn("completed", text)
        self.assertIn("135/135", text)


def _write_tuned(path: Path, rows: dict, method: str = "hedonic") -> Path:
    path.write_text(json.dumps({"complete": True, "settings": {method: rows}}))
    return path


class TestPhase(unittest.TestCase):
    def test_phase_defaults_to_local_with_the_hedonic_settings(self):
        plan = sq.resolve(["--networks", "dblp", "--seeds", "0"])
        self.assertEqual(plan["phase"], "local")
        self.assertEqual(plan["settings"]["dblp"]["cap"], qs.HEDONIC_TUNED["hedonic"]["dblp"]["max_memberships"])
        self.assertFalse(hasattr(sq, "MULTILEVEL_TUNED"))  # the 1.x multilevel settings are gone with the phase

    def test_local_takes_the_hedonic_settings_from_the_config_or_the_flag(self):
        plan = sq.resolve(["--config", SHIPPED])
        self.assertEqual(plan["phase"], "local")
        # the top5000 covers first (DBLP, Amazon, YouTube, Friendster, LiveJournal), then the complete (all) covers
        self.assertEqual(plan["networks"], ["dblp:top5000", "amazon:top5000", "youtube:top5000", "friendster:top5000",
                                            "livejournal:top5000", "dblp", "amazon", "youtube"])
        self.assertEqual(plan["seeds"], list(range(10)))
        self.assertEqual(len(sq.conditions(plan)), 720)
        local = qs.HEDONIC_TUNED["hedonic"]
        self.assertEqual(plan["settings"]["dblp"]["cap"], local["dblp"]["max_memberships"])
        self.assertAlmostEqual(plan["settings"]["dblp"]["gamma"], local["dblp"]["absolute_resolution"])
        self.assertEqual(plan["settings"]["amazon"]["cap"], local["amazon"]["max_memberships"])
        self.assertAlmostEqual(plan["settings"]["amazon"]["gamma"], local["amazon"]["absolute_resolution"])
        flagged = sq.resolve(["--networks", "dblp,amazon", "--seeds", "0-9", "--phase", "local"])
        self.assertEqual(flagged["settings"], {k: plan["settings"][k] for k in ("dblp", "amazon")})
        # a pair takes its network's setting until a calibration names it, and the DBLP one when its network has none
        self.assertEqual(plan["settings"]["amazon:top5000"], plan["settings"]["amazon"])
        self.assertEqual(plan["settings"]["friendster:top5000"]["cap"], local["dblp"]["max_memberships"])

    def test_the_multilevel_phase_is_a_clear_error_before_anything_runs(self):
        for spelling in ("multilevel", "multi-phase", "multiphase", "hoc_multilevel"):
            with self.assertRaisesRegex(ValueError, "not available in hedonic 2.0"):
                sq.parse_phase(spelling)
        with self.assertRaisesRegex(ValueError, "must be one of local"):
            sq.parse_phase("global")
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "phase.toml"
            config.write_text('[start_quality]\nnetworks = ["dblp"]\nphase = "multilevel"\n')
            with self.assertRaisesRegex(ValueError, "not available in hedonic 2.0"):
                sq.resolve(["--config", str(config)])
            with self.assertRaisesRegex(ValueError, "not available in hedonic 2.0"):
                sq.resolve(["--config", SHIPPED, "--phase", "multilevel"])  # the flag is checked as the config is
            err = io.StringIO()
            with patch.object(sq, "run_study", side_effect=AssertionError("ran")), redirect_stderr(err), \
                    redirect_stdout(io.StringIO()):
                code = sq.main(["--networks", "dblp", "--phase", "multilevel", "--output", str(Path(folder) / "o")])
            self.assertEqual(code, 2)
            self.assertIn("not available in hedonic 2.0", err.getvalue())
            self.assertFalse((Path(folder) / "o").exists())

    def test_keys_name_the_phase_and_the_code_and_records_carry_the_runtime(self):
        job = tiny_job()
        original = sq.v3._detector_worker
        with tempfile.TemporaryDirectory() as folder:
            results = Path(folder) / sq.RESULTS_NAME
            local = plan_for(folder, levels=[0.0], seeds=[0])
            key = sq.conditions(local)[0]["key"]
            self.assertEqual(key, f"tiny|0|0|cap=2|gamma=0.5|phase=local|code={sq.code_identity()}")
            self.assertTrue(sq.code_identity().startswith(f"hedonic-{runtime.RESULTS_SERIES}+core-"))
            self.assertNotIn("/", sq.code_identity())  # no path ever enters a key
            with patch.object(sq.v3, "_detector_worker", side_effect=original) as worker:
                sq.run_study(local, loader=lambda _: job, emit=lambda _: None)
            self.assertEqual(worker.call_args.args[4], "local")  # the detector runs the configured phase
            records = [json.loads(line) for line in results.read_text().splitlines()]
            self.assertEqual([(r["phase"], r["key"], r["code"]) for r in records], [("local", key, sq.code_identity())])
            self.assertEqual(records[0]["runtime"]["hedonic"], runtime.hedonic_version())
            self.assertFalse({"lucas_igraph", "igraph_c_required"} & set(records[0]["runtime"]))

    def test_records_of_other_code_are_kept_but_never_reused(self):
        job = tiny_job()
        # A 1.x record without a phase field (the multilevel phase), and a 1.x local record without a code token.
        legacy = {"key": "tiny|0|0|cap=2|gamma=0.5", "network": "tiny", "level": "0", "level_kind": "swap",
                  "requested_distance": 0.0, "realized_distance": 0.0, "detector_seed": 0,
                  "start_symmetric_f1": 0.5, "start_onmi": 0.5, "final_symmetric_f1": 0.5, "final_onmi": 0.5,
                  "seconds": 0.1, "final_communities": 2, "cap": 2, "gamma": 0.5}
        old_local = dict(legacy, key="tiny|0|0|cap=2|gamma=0.5|phase=local", phase="local")
        with tempfile.TemporaryDirectory() as folder:
            results = Path(folder) / sq.RESULTS_NAME
            results.write_text(json.dumps(legacy) + "\n" + json.dumps(old_local) + "\n")
            records = sq.read_records(results)
            self.assertEqual(sorted(records), ["tiny|0|0|cap=2|gamma=0.5|phase=local",
                                               "tiny|0|0|cap=2|gamma=0.5|phase=multilevel"])
            loaded = []

            def loader(network):
                loaded.append(network)
                return job

            sq.run_study(plan_for(folder, levels=[0.0], seeds=[0]), loader=loader, emit=lambda _: None)
            self.assertEqual(loaded, ["tiny"])  # neither old record is this code's condition
            lines = [json.loads(line) for line in results.read_text().splitlines()]
            self.assertEqual([line.get("phase") for line in lines], [None, "local", "local"])
            self.assertEqual(lines[-1]["code"], sq.code_identity())
            loaded.clear()
            sq.run_study(plan_for(folder, levels=[0.0], seeds=[0]), loader=loader, emit=lambda _: None)
            self.assertEqual(loaded, [])  # this code's record is reused

    def test_dry_run_plan_names_the_phase(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "study"
            buffer = io.StringIO()
            with patch.object(sq, "run_study", side_effect=AssertionError("dry run ran the study")), \
                    redirect_stdout(buffer):
                code = sq.main(["--networks", "dblp", "--seeds", "0", "--levels", "0,random", "--phase", "local",
                                "--output", str(output), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())
            text = buffer.getvalue()
            self.assertIn(sq.PHASE_NAME["local"], text)
            self.assertIn("phase    local ", text)

    def test_front_dry_run_of_the_local_config_names_the_local_phase(self):
        with tempfile.TemporaryDirectory() as folder:
            buffer = io.StringIO()
            with patch("hedonic.experiments.overlapping.runmanager.launch_startcurve",
                       side_effect=AssertionError("dry run launched a run")), redirect_stdout(buffer):
                code = sq.front_main(["--config", SHIPPED, "--output-dir", str(Path(folder) / "run"), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse((Path(folder) / "run").exists())
            text = buffer.getvalue()
            self.assertIn(f"720 {sq.PHASE_NAME['local']} runs", text)
            self.assertIn("phase    local ", text)

    def test_a_named_run_without_a_folder_goes_under_the_user_output_dir_or_the_runs_root(self):
        import os

        with tempfile.TemporaryDirectory() as folder:
            user = Path(folder) / "config.toml"
            user.write_text(f'[defaults]\noutput_dir = "{folder}/runs"\n')
            with patch.dict(os.environ, {"HEDONIC_CONFIG": str(user)}):
                plan = sq.front_study(["--config", SHIPPED, "--name", "curve"])
            self.assertEqual(plan["output"], (Path(folder) / "runs" / "curve").resolve())
            plan = sq.front_study(["--config", SHIPPED, "--name", "curve", "--output-dir", str(Path(folder) / "x")])
            self.assertEqual(plan["output"], (Path(folder) / "x").resolve())

    def test_covers_cross_every_network_and_the_all_cover_keeps_the_old_record_keys(self):
        self.assertEqual(sq.parse_networks("dblp,amazon", "local", None, "all,top5000"),
                         ["dblp", "dblp:top5000", "amazon", "amazon:top5000"])
        self.assertEqual(sq.parse_networks("dblp:all,youtube:top5000", "local"), ["dblp", "youtube:top5000"])
        with self.assertRaises(ValueError):
            sq.parse_networks("wikipedia", "local", None, "top5000")  # Wikipedia has no such cover
        calibrated = {"hedonic": {"dblp:top5000": {"max_memberships": 9, "absolute_resolution": 0.3},
                                    "dblp": {"max_memberships": 5, "absolute_resolution": 0.2}}}
        self.assertEqual(sq.tuned_setting("dblp:top5000", "local", calibrated)["cap"], 9)
        self.assertEqual(sq.tuned_setting("dblp", "local", calibrated)["cap"], 5)  # the all cover: the network's entry

    def test_tuned_files_are_laid_over_one_another_and_replayed_by_a_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            first = _write_tuned(Path(folder) / "a.json", {"dblp": {"max_memberships": 5, "absolute_resolution": 0.2},
                                                           "amazon": {"max_memberships": 3, "absolute_resolution": 0.1}})
            second = _write_tuned(Path(folder) / "b.json", {"dblp": {"max_memberships": 7, "absolute_resolution": 0.4}},
                                  method="hoc_local")  # an older file names the method hoc_local
            plan = sq.resolve(["--networks", "dblp,amazon", "--seeds", "0", "--tuned", str(first), "--tuned", str(second)])
            self.assertEqual((plan["settings"]["dblp"]["cap"], plan["settings"]["dblp"]["gamma"]), (7, 0.4))
            self.assertEqual(plan["settings"]["amazon"]["cap"], 3)
            self.assertEqual(plan["tuned"], f"{first.resolve()},{second.resolve()}")
            replayed = sq.resolve(sq.launch_argv(plan))
            self.assertEqual(sq.conditions(replayed), sq.conditions(plan))
            self.assertEqual(replayed["settings"], plan["settings"])
            front = sq.front_study(["--config", SHIPPED, "--tuned", str(first), "--output-dir", folder])
            self.assertEqual(front["settings"]["amazon"]["cap"], 3)
            with self.assertRaises(qs.TunedMissing):
                sq.resolve(["--networks", "dblp", "--seeds", "0", "--tuned", str(Path(folder) / "missing.json")])

    def test_a_durable_run_replays_the_phase_and_the_conditions(self):
        plan = sq.resolve(["--config", SHIPPED])
        replayed = sq.resolve(sq.launch_argv(plan))  # a resume rebuilds the plan from the stored argv alone
        self.assertEqual(replayed["phase"], "local")
        self.assertEqual(sq.conditions(replayed), sq.conditions(plan))


class TestCLI(unittest.TestCase):
    def test_registration_help_and_the_run_verb_routes_to_the_front_door(self):
        self.assertIn("start-quality", CLI.COMMANDS)
        self.assertIn("startcurve", CLI.RUN_ALIASES)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(CLI.main(["start-quality", "--help"]), 0)
            self.assertEqual(CLI.hedonic_main(["run", "startcurve", "--help"]), 0)
        with patch("hedonic.experiments.overlapping.start_quality.front_main", return_value=0) as front:
            self.assertEqual(CLI.hedonic_main(["run", "startcurve", "--config", "x.toml", "--dry-run"]), 0)
        front.assert_called_once_with(["--config", "x.toml", "--dry-run"])


if __name__ == "__main__":
    unittest.main()
