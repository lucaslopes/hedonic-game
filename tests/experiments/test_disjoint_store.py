"""Columnar result store of the disjoint SBM sweep (``layout="store"``)."""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hedonic.experiments import CLI
from hedonic.experiments.disjoint import reproduce, sbm_sweep
from hedonic.experiments.disjoint import store as store_mod


def _store_rows(root) -> list[tuple]:
    df = store_mod.load_runs(root)
    df = df.sort_values(list(store_mod.KEY_COLUMNS), na_position="first")
    cols = ["method", "noise", "partition_seed", "run", "run_seed", "partition_hash"]
    cols += ["rand_index", "adjusted_rand", "robustness"]
    frame = df[cols].astype(object)
    frame = frame.where(frame.notna(), None)  # NaN != NaN would mask equality
    return [tuple(row) for row in frame.itertuples(index=False)]


class TestStoreHelpers(unittest.TestCase):
    def test_decimal_text_and_shard_path(self):
        self.assertEqual(store_mod.decimal_text(0.05), "0.05")
        self.assertEqual(store_mod.decimal_text(0.10), "0.1")
        self.assertEqual(store_mod.decimal_text(10.0), "10")
        self.assertEqual(store_mod.decimal_text(0.0), "0")
        self.assertEqual(store_mod.decimal_text(None), "null")
        self.assertEqual(
            store_mod.shard_relative_path(4, 255, 0.05, 0.5, 7).as_posix(),
            "runs/4C_255N/p_in-0.05/difficulty-0.5/network-007.parquet",
        )

    def test_run_seed_is_stable_and_key_sensitive(self):
        keys = dict(
            number_of_communities=2,
            community_size=10,
            p_in=0.3,
            difficulty=0.5,
            network_seed=1,
            method="Hedonic",
            noise=0.1,
            partition_seed=0,
            run=0,
        )
        seed = store_mod.run_seed(**keys)
        self.assertEqual(seed, store_mod.run_seed(**keys))
        self.assertTrue(0 <= seed < 2**63)
        for field, value in [
            ("run", 1),
            ("method", "Leiden"),
            ("noise", 0.5),
            ("partition_seed", 1),
            ("network_seed", 2),
            ("p_in", 0.31),
        ]:
            self.assertNotEqual(store_mod.run_seed(**dict(keys, **{field: value})), seed)

    def test_canonical_partition_and_hash(self):
        a = store_mod.canonical_partition([5, 5, 2, 9, 2])
        self.assertEqual(a.tolist(), [0, 0, 1, 2, 1])
        self.assertEqual(a.dtype, np.uint16)
        b = store_mod.canonical_partition([1, 1, 0, 3, 0])
        self.assertEqual(store_mod.partition_hash(a), store_mod.partition_hash(b))
        self.assertEqual(len(store_mod.partition_hash(a)), 16)
        self.assertNotEqual(
            store_mod.partition_hash(a),
            store_mod.partition_hash(store_mod.canonical_partition([0, 1, 1, 2, 1])),
        )

    def test_partition_agreement_metrics(self):
        truth = [0] * 5 + [1] * 5 + [2] * 5
        same = sbm_sweep.partition_agreement([7] * 5 + [3] * 5 + [1] * 5, truth)
        self.assertEqual(same["overlap"], 1.0)
        self.assertEqual(same["overlap_normalized"], 1.0)
        self.assertAlmostEqual(same["nmi"], 1.0)
        self.assertAlmostEqual(same["vi"], 0.0)
        merged = sbm_sweep.partition_agreement([0] * 15, truth)
        self.assertAlmostEqual(merged["overlap"], 1 / 3)
        self.assertAlmostEqual(merged["overlap_normalized"], 0.0)
        self.assertAlmostEqual(merged["nmi"], 0.0)
        # One vertex moved: overlap 14/15; extra found communities are errors.
        moved = sbm_sweep.partition_agreement([0] * 5 + [1] * 5 + [2] * 4 + [0], truth)
        self.assertAlmostEqual(moved["overlap"], 14 / 15)
        split = sbm_sweep.partition_agreement(list(range(15)), truth)
        self.assertAlmostEqual(split["overlap"], 3 / 15)
        self.assertLess(split["overlap_normalized"], 0)

    def test_vectorized_robustness_matches_reference(self):
        rng = random.Random(3)
        for trial in range(60):
            size = rng.randint(1, 15)
            g = sbm_sweep.generate_graph(
                rng.randint(1, 4), size, rng.choice([0, 0.2, 0.7, 1]), 0.4, trial
            )
            n = g.vcount()
            for membership in (
                [rng.randrange(rng.randint(1, n)) for _ in range(n)],
                list(range(n)),
                [v // size for v in range(n)],
            ):
                self.assertEqual(
                    sbm_sweep.robustness(g, membership),
                    sbm_sweep.robustness_reference(g, membership),
                )

    def test_initial_membership_is_pure(self):
        gt = [v // 10 for v in range(30)]
        state = np.random.get_state()
        first = sbm_sweep.get_initial_membership(gt, 0.5, 4)
        after = np.random.get_state()
        self.assertTrue(np.array_equal(state[1], after[1]))
        self.assertEqual(first, sbm_sweep.get_initial_membership(gt, 0.5, 4))
        self.assertNotEqual(first, sbm_sweep.get_initial_membership(gt, 0.5, 5))
        self.assertEqual(sbm_sweep.get_initial_membership(gt, 1.1, 0), list(range(30)))


class TestSbmSmokeStore(unittest.TestCase):
    """The whole sbm-smoke pipeline, built once."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        code = CLI.main(["disjoint", "--preset", "sbm-smoke", "--output_root", str(cls.root)])
        assert code == 0, code
        cls.store = cls.root / "sbm-smoke"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_layout_manifest_and_shards(self):
        manifest = json.loads((self.store / "manifest.json").read_text())
        self.assertEqual(manifest["schema"], store_mod.STORE_SCHEMA)
        self.assertEqual(manifest["spec"]["graph_generator"], sbm_sweep.SBM_GENERATOR_ID)
        self.assertEqual(manifest["spec"]["partitions"], "full")
        self.assertIn("python_igraph", manifest["spec"]["runtime"])
        shards = store_mod.shard_paths(self.store)
        self.assertEqual(len(shards), 16)
        meta = store_mod.shard_metadata(shards[0])
        self.assertTrue(meta["complete"])
        self.assertEqual(meta["spec_hash"], manifest["spec_hash"])
        self.assertEqual(meta["n_rows"], 50)
        # No graph files are written.
        self.assertFalse(list(self.store.rglob("*.pkl")))

    def test_rows_seeds_and_partitions(self):
        df = store_mod.load_runs(self.store, with_partitions=True)
        self.assertEqual(len(df), 800)
        self.assertTrue((df["status"] == "ok").all())
        once = df[df["method"].isin(["GroundTruth", "Spectral"])]
        self.assertTrue(once["noise"].isna().all())
        self.assertEqual(len(once), 32)
        self.assertTrue((df[df["method"] == "GroundTruth"]["adjusted_rand"] == 1.0).all())
        gt_rows = df[df["method"] == "GroundTruth"]
        self.assertTrue((gt_rows["overlap_normalized"] == 1.0).all())
        self.assertTrue((gt_rows["vi"].abs() < 1e-12).all())
        self.assertEqual(df["run_seed"].nunique(), len(df))
        self.assertEqual(df[df["method"] == "Hedonic"]["run"].max(), 2)
        unique = df[~df["is_duplicate"]]
        self.assertTrue(unique["partition"].notna().all())
        for _, row in unique.head(20).iterrows():
            canonical = np.asarray(row["partition"], dtype=np.uint16)
            self.assertEqual(store_mod.partition_hash(canonical), row["partition_hash"])
        self.assertTrue(df[df["is_duplicate"]]["partition"].isna().all())

    def test_graph_fingerprints_match_regeneration(self):
        df = store_mod.load_runs(self.store)
        graphs = df.drop_duplicates(
            ["number_of_communities", "p_in", "difficulty", "network_seed"]
        )
        for row in graphs.itertuples():
            g = sbm_sweep.generate_graph(
                row.number_of_communities,
                row.community_size,
                row.p_in,
                row.difficulty,
                row.network_seed,
            )
            self.assertEqual(sbm_sweep.graph_fingerprint(g), row.graph_sha256)
            self.assertEqual(g.ecount(), row.n_edges)

    def test_verify_replays_every_shard(self):
        report = store_mod.verify_store(self.store, sample_shards=None, sample_rows=6)
        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["graphs_checked"], 16)
        self.assertEqual(report["rows_checked"], 96)

    def test_duckdb_views(self):
        con = store_mod.open_store(self.store)
        self.assertEqual(con.execute("SELECT count(*) FROM runs").fetchone()[0], 800)
        cells = con.execute(
            "SELECT count(*) FROM cells WHERE method = 'Hedonic'"
        ).fetchone()[0]
        self.assertEqual(cells, 16 * 3 * 2)
        rows = con.execute(
            "SELECT method, avg(adjusted_rand) FROM runs GROUP BY ALL ORDER BY ALL"
        ).fetchall()
        self.assertEqual(rows[0][0], "GroundTruth")
        self.assertEqual(rows[0][1], 1.0)

    def test_plot_table_is_one_row_per_logical_cell(self):
        table = store_mod.plot_table(self.store)
        for column in ("method", "multiplier", "accuracy", "adjusted_rand", "robustness",
                       "duration", "noise", "network_seed", "partition_seed", "outputs"):
            self.assertIn(column, table.columns)
        self.assertFalse(table["noise"].isna().any())
        # Per graph: GroundTruth and Spectral once, repeated over 3 noises (6),
        # plus 4 cell methods x 3 noises x 2 partition seeds (24).
        self.assertEqual(len(table), 16 * 30)
        keys = ["method", "number_of_communities", "p_in", "multiplier", "network_seed",
                "noise", "partition_seed"]
        self.assertFalse(table.duplicated(keys).any())
        self.assertEqual(
            sorted(table[table["method"] == "Spectral"]["noise"].unique()), [0.1, 0.5, 1.0]
        )
        everything = store_mod.plot_table(self.store, unique_only=False)
        self.assertEqual(len(everything), len(table))
        self.assertEqual(everything["outputs"].sum(), 800 + 2 * 16 * 2)
        self.assertLess(table["outputs"].sum(), everything["outputs"].sum())
        raw = store_mod.load_runs(self.store)
        unique_hedonic = raw[(raw["method"] == "Hedonic") & ~raw["is_duplicate"]]
        self.assertAlmostEqual(
            table[table["method"] == "Hedonic"]["outputs"].sum(), len(unique_hedonic)
        )

    def test_paper_stats_from_store(self):
        from hedonic.experiments.disjoint import paper_stats

        out = self.root / "stats"
        self.assertEqual(
            CLI.main(["disjoint-stats", "--data", str(self.store), "--output_dir", str(out)]), 0
        )
        stats = json.loads((out / "paper_stats.json").read_text())
        grid = stats["grid"]
        self.assertEqual(grid["graphs"], 16)
        self.assertEqual(grid["seeds_per_condition_min"], 2)  # counted, not assumed
        self.assertEqual(grid["network_seeds"], [0, 1])
        self.assertEqual(stats["ground_truth"]["graphs"], 16)
        self.assertIn("Hedonic", stats["tracking"]["mean_ari"])
        for metric in ("adjusted_rand", "overlap_normalized", "nmi", "vi"):
            self.assertIn(metric, stats["tracking"]["mean_accuracy"])
            self.assertIn(metric, stats["threshold"]["accuracy_noise1"])
        self.assertIn("below_threshold_spread", stats["threshold"])
        self.assertTrue((out / "tracking_snr_band_all_metrics.csv").is_file())
        for name in ("tables.tex", "summary.md", "tracking_by_noise.csv",
                     "refinement_vs_mirror.csv", "gt_snr_band.csv"):
            self.assertTrue((out / name).is_file(), name)
        self.assertIn("network seeds per condition: 2", (out / "tables.tex").read_text())
        # SNR of one condition, from the manuscript's formula.
        frame = paper_stats.pd.DataFrame(
            {"number_of_communities": [2], "community_size": [510], "p_in": [0.05],
             "multiplier": [0.5]}
        )
        expected = 0.05 * 1020 * 0.25 / (2 * 1.5)
        self.assertAlmostEqual(float(paper_stats.snr(frame).iloc[0]), expected)

    def test_progress_and_writer_lock(self):
        info = store_mod.progress(self.store)
        self.assertEqual(info["completed"], 16)
        self.assertEqual(info["expected"], 16)
        self.assertFalse(info["running"])
        self.assertIn("igraph_extension_sha256", info["runtime"])
        handle, removed = store_mod.acquire_writer_lock(self.store)
        try:
            self.assertEqual(removed, 0)
            self.assertIsNotNone(store_mod.writer_lock_holder(self.store))
            with self.assertRaises(SystemExit) as raised:
                sbm_sweep.main(
                    ["--preset", "sbm-smoke", "--output_root", str(self.root), "--resume"]
                )
            self.assertIn("refusing a second concurrent run", str(raised.exception))
        finally:
            handle.close()
        self.assertIsNone(store_mod.writer_lock_holder(self.store))
        self.assertEqual(CLI.main(["disjoint-store", str(self.store), "--progress"]), 0)

    def test_stale_tmp_files_are_removed_on_resume(self):
        shard = store_mod.shard_paths(self.store)[0]
        stale = shard.parent / f".{shard.name}.abc123.tmp"
        stale.write_bytes(b"partial")
        self.assertEqual(store_mod.progress(self.store)["stale_tmp_files"], 1)
        handle, removed = store_mod.acquire_writer_lock(self.store)
        handle.close()
        self.assertEqual(removed, 1)
        self.assertFalse(stale.exists())

    def test_resume_skips_complete_graphs(self):
        with patch.object(sbm_sweep, "generate_graph", side_effect=AssertionError("rerun")):
            code = sbm_sweep.main(
                ["--preset", "sbm-smoke", "--output_root", str(self.root), "--resume"]
            )
        self.assertEqual(code, 0)

    def test_spec_change_is_refused(self):
        with self.assertRaises(SystemExit) as raised:
            sbm_sweep.main(
                ["--preset", "sbm-smoke", "--output_root", str(self.root), "--n_runs", "2"]
            )
        # Only the restarted methods change with n_runs; the store names them and the way out.
        self.assertIn("holds rows of Hedonic, Leiden", str(raised.exception))
        self.assertIn("--rerun hedonic,leiden", str(raised.exception))

    def test_store_cli(self):
        self.assertEqual(CLI.main(["disjoint-store", str(self.store)]), 0)
        self.assertEqual(
            CLI.main(["disjoint-store", str(self.store), "--sql", "SELECT 1 AS x"]), 0
        )
        out = self.root / "export" / "plot.csv.gzip"
        self.assertEqual(CLI.main(["disjoint-store", str(self.store), "--export",
                                   str(self.root / "export" / "all.parquet"),
                                   "--all-restarts"]), 0)
        self.assertEqual(CLI.main(["disjoint-store", str(self.store), "--export", str(out)]), 0)
        self.assertTrue(out.is_file())
        self.assertEqual(
            CLI.main(["disjoint-store", str(self.store), "--verify", "--sample-shards", "2",
                      "--sample-rows", "3"]),
            0,
        )
        self.assertEqual(CLI.main(["disjoint-store", str(self.root / "nope")]), 1)

    def test_plots_read_store_directory(self):
        from hedonic.experiments.plots import paper_figures

        df = paper_figures.load_data(self.store)
        self.assertEqual(len(df), 16 * 30)


class TestOlderStoreSchema(unittest.TestCase):
    def test_store_without_the_newer_metric_columns_stays_readable(self):
        import pyarrow.parquet as pq

        from hedonic.experiments.disjoint import paper_stats

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sbm_sweep.main(["--preset", "sbm-smoke", "--output_root", str(root), "--partitions", "hash",
                            "--n_communities", "2", "--seeds", "0"])
            store = root / "sbm-smoke"
            for shard in store_mod.shard_paths(store):  # rewrite as a schema-v1 store (ARI only)
                table = pq.read_table(shard)
                table = table.drop_columns(list(store_mod.OPTIONAL_METRICS))
                pq.write_table(table, shard)
            self.assertGreater(store_mod.summarize(store)["rows"], 0)
            self.assertNotIn("nmi", store_mod.plot_table(store).columns)
            self.assertTrue(store_mod.verify_store(store, sample_shards=None, sample_rows=3)["ok"])
            stats = paper_stats.compute_stats(store_mod.plot_table(store))
            self.assertIsNotNone(stats["tracking"]["mean_ari"]["Hedonic"])
            self.assertIsNone(stats["tracking"]["mean_accuracy"]["nmi"]["Hedonic"])


class TestStoreModes(unittest.TestCase):
    ARGS = (
        "--max_n_nodes", "40", "--n_communities", "2", "--seeds", "0", "1",
        "--p_in", "0.3", "--difficulty", "0.4", "--noises", "0.1", "1.0",
        "--partition_seeds", "2", "--methods", "Mirror,Hedonic,Spectral",
        "--n_runs", "2", "--layout", "store", "--folder_name", "s",
    )

    def _run(self, root, *extra):
        code = sbm_sweep.main([*self.ARGS, "--output_root", str(root), *extra])
        self.assertEqual(code, 0)
        return root / "s"

    def test_hash_mode_and_workers_give_identical_rows(self):
        with tempfile.TemporaryDirectory() as d:
            full = self._run(Path(d) / "full")
            hashed = self._run(Path(d) / "hash", "--partitions", "hash")
            parallel = self._run(Path(d) / "par", "--workers", "2")
            self.assertEqual(_store_rows(full), _store_rows(hashed))
            self.assertEqual(_store_rows(full), _store_rows(parallel))
            hashed_df = store_mod.load_runs(hashed)
            self.assertNotIn("partition", hashed_df.columns)
            report = store_mod.verify_store(hashed, sample_shards=None, sample_rows=5)
            self.assertTrue(report["ok"], report["problems"])

    def test_preflight_reports_rows_and_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            argv = [*self.ARGS, "--output_root", str(root), "--preflight"]
            self.assertEqual(sbm_sweep.main(argv), 0)
            self.assertFalse((root / "s").exists())
            self._run(root)
            plan = sbm_sweep.preflight_grid(
                folder_name="s",
                max_n_nodes=40,
                n_communities=[2],
                seeds=[0, 1],
                p_in=[0.3],
                difficulty=[0.4],
                noises=[0.1, 1.0],
                partition_seeds=2,
                methods=sbm_sweep._parse_methods("Mirror,Hedonic,Spectral"),
                n_runs=2,
                layout="store",
                output_root=root,
                spec=store_mod.read_manifest(root / "s")["spec"],
            )
            # Mirror 4 cells + Hedonic 4 cells × 2 runs + Spectral once.
            self.assertEqual(plan["rows_per_graph"], 4 + 8 + 1)
            self.assertEqual(plan["pending_cells"], 0)

    def test_reproduce_store_pipeline_without_plots(self):
        with tempfile.TemporaryDirectory() as d:
            code = reproduce.main(
                ["--preset", "sbm-smoke", "--output_root", d, "--skip-plots"]
            )
            self.assertEqual(code, 0)
            store = Path(d) / "sbm-smoke"
            self.assertTrue((store / "plot_table.parquet").is_file())
            self.assertTrue((store / "stats" / "paper_stats.json").is_file())


class TestDisjointFrontDoor(unittest.TestCase):
    """`hedonic run disjoint` / `hedonic show disjoint` and the run verbs for disjoint runs."""

    def test_flag_parsing_and_validation(self):
        from hedonic.experiments.disjoint import front

        self.assertEqual(front.parse_floats("0.1:0.3:0.1"), [0.1, 0.2, 0.3])
        self.assertEqual(front.parse_floats("0.5,0.1"), [0.1, 0.5])
        self.assertEqual(front.parse_methods("hedonic,full,gt"), ["Hedonic", "Leiden", "GroundTruth"])
        with self.assertRaises(ValueError):
            front.parse_methods("louvain")
        args = front.build_parser().parse_args(
            ["--preset", "paper", "--communities", "2,4", "--seeds", "0-9", "--workers", "3"]
        )
        settings = front.resolve(args)
        self.assertEqual(settings["communities"], [2, 4])
        self.assertEqual(settings["seeds"], list(range(10)))
        self.assertEqual(front.plan(settings)["graphs"], 2 * 10 * 10 * 10)
        with self.assertRaises(ValueError):  # 1020 is not divisible by 7
            front.resolve(front.build_parser().parse_args(["--communities", "7", "--nodes", "1020"]))
        # The stored flags reproduce the same settings.
        again = front.resolve(front.build_parser().parse_args(front.front_argv(settings)))
        self.assertEqual(again, settings)

    def test_paper_preset_plan_show_and_dry_run(self):
        from hedonic.experiments.disjoint import front

        paper = front.resolve(front.build_parser().parse_args(["--preset", "paper", "--workers", "8"]))
        info = front.plan(paper)
        self.assertEqual((info["graphs"], info["rows_per_graph"], info["rows"]), (50_000, 1_102, 55_100_000))
        self.assertEqual(CLI.hedonic_main(["show", "disjoint"]), 0)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "planned"
            self.assertEqual(CLI.hedonic_main(["run", "disjoint", "--preset", "paper", "--dry-run",
                                               "--output-dir", str(out)]), 0)
            self.assertFalse(out.exists())  # a dry run writes nothing

    def test_foreground_run_check_and_counts(self):
        from hedonic.experiments.disjoint import front
        from hedonic.experiments.overlapping import runmanager

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "smoke-run"
            code = CLI.hedonic_main(["run", "disjoint", "--preset", "smoke", "--foreground",
                                     "--workers", "2", "--output-dir", str(out), "--name", "fg-smoke"])
            self.assertEqual(code, 0)
            prog = json.loads((out / "progress.json").read_text())
            self.assertEqual(prog["status"], "completed")
            self.assertEqual((prog["done"], prog["graphs"]), (16, 16))
            self.assertIn("igraph_c", prog["environment"])
            self.assertEqual(prog["methods"]["GroundTruth"]["adjusted_rand"], prog["methods"]["GroundTruth"]["n"])
            self.assertTrue((out / "stats" / "paper_stats.json").is_file())
            entry = {"name": "fg-smoke", "kind": "disjoint", "output_dir": str(out), "backend": "process",
                     "created": prog["started"], "settings": front.resolve(
                         front.build_parser().parse_args(["--preset", "smoke", "--workers", "2"]))}
            self.assertEqual(runmanager.progress_counts(entry, prog), (16, 16))
            self.assertIn("16/16 graphs", front.snapshot(entry, prog, "completed", "*", True))
            self.assertEqual(front.check(entry, shards=4, rows=5), 0)
            # The queue judges the run from its store: complete, and 16 graphs.
            from hedonic.experiments.overlapping import runqueue

            item = ["disjoint", "--preset", "smoke", "--workers", "2", "--name", "fg-smoke",
                    "--output-dir", str(out)]
            self.assertEqual(runqueue.data_state(item, fresh=True), ("complete", str(out.resolve())))
            self.assertEqual(runqueue.unit_counts(item), (16, 16, "graphs"))
            # A store whose Hedonic rows another detector computed (another fingerprint) is refused before
            # anything is written.
            before = (out / "progress.json").read_bytes()
            with _other_build(hedonic_partition_fingerprint="partitions-v1:" + "0" * 64):
                self.assertEqual(runqueue.data_state(item, fresh=True)[0], "partial")
                code = CLI.hedonic_main(["run", "disjoint", "--preset", "smoke", "--foreground",
                                         "--workers", "2", "--output-dir", str(out), "--name", "fg-smoke"])
                self.assertEqual(code, 2)
            self.assertEqual((out / "progress.json").read_bytes(), before)

    def test_shipped_disjoint_queue(self):
        from hedonic.experiments.overlapping import runqueue

        with tempfile.TemporaryDirectory() as d:
            queue = runqueue.load_queue("disjoint", root=d)
            self.assertEqual([item["name"] for item in queue["items"]], ["disjoint-smoke", "disjoint-paper"])
            self.assertEqual(runqueue.requirements(queue), ([], []))  # nothing to download or build
            self.assertEqual(runqueue.data_state(queue["items"][1]["run"], fresh=True),
                             ("missing", str(Path(d).resolve() / "disjoint-paper")))
            bad = Path(d) / "q.toml"
            bad.write_text('name = "q"\n[[items]]\nname = "a"\nrun = ["disjoint", "--preset", "smoke"]\n')
            with self.assertRaises(SystemExit) as raised:
                runqueue.load_queue(str(bad), root=d)
            self.assertIn("needs --name", str(raised.exception))

    def test_default_output_uses_the_user_config(self):
        import os

        from hedonic.experiments.disjoint import front
        from hedonic.experiments.overlapping import userconfig

        with tempfile.TemporaryDirectory() as d:
            with patch.dict(os.environ, {"HEDONIC_CONFIG": str(Path(d) / "config.toml")}):
                userconfig.main(["set", "disjoint_output_root", str(Path(d) / "runs")])
                self.assertEqual(front.default_output("x"), str(Path(d) / "runs" / "x"))
                self.assertNotIn("disjoint_output_root", userconfig.MACHINE_KEYS)


class TestPaperHocProfile(unittest.TestCase):
    def test_per_network_tuned_settings_and_command(self):
        from hedonic.experiments.overlapping import quickstart as qs

        cfg = qs.load_config("paper-hoc")
        self.assertEqual(cfg.methods, ["hedonic"])
        self.assertEqual((cfg.nodes, cfg.seeds, cfg.save_covers), (0, list(range(10)), True))
        local = qs.BY_KEY["hedonic"]
        self.assertEqual(qs.method_params(local, cfg, "dblp"), {"max_memberships": 32, "absolute_resolution": 1 / 7})
        self.assertEqual(qs.method_params(local, cfg, "amazon"),
                         {"max_memberships": 4, "absolute_resolution": 1 / 723})
        # Other configs keep the screened settings.
        self.assertEqual(qs.method_params(local, qs.load_config("paper"), "amazon"), dict(local.screened))
        self.assertIn("--config paper-hoc", cfg.command())
        self.assertEqual(len(qs.build_plan(cfg, {})), 20)
        env = qs.environment()
        self.assertIn("igraph_c", env)
        self.assertIn("igraph_extension_sha256", env)

    def test_youtube_hoc_is_gone(self):
        from hedonic.experiments import CLI
        from hedonic.experiments.overlapping import quickstart as qs

        with self.assertRaisesRegex(ValueError, "unknown config 'youtube-hoc'"):
            qs.load_config("youtube-hoc")
        self.assertNotIn("youtube-hoc", CLI.RUN_ALIASES)
        self.assertIn("--isolate", qs.Config(isolate=True).command())


class TestStoreCLIRegistry(unittest.TestCase):
    def test_registered_and_help(self):
        self.assertIn("disjoint-store", CLI.COMMANDS)
        self.assertIn("disjoint-stats", CLI.COMMANDS)
        self.assertEqual(CLI.main(["disjoint-stats", "--help"]), 0)
        with self.assertRaises(SystemExit) as raised:
            store_mod.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(CLI.main(["disjoint-store", "--help"]), 0)
        self.assertIn("sbm", sbm_sweep.PRESETS)
        self.assertEqual(sbm_sweep.PRESETS["sbm"]["seeds"], list(range(100)))


if __name__ == "__main__":
    unittest.main()


def _other_build(**fields):
    """runtime_identity of another build (only the named fields differ)."""
    real = store_mod.runtime_identity

    def fake():
        return dict(real(), **fields)

    return patch.object(store_mod, "runtime_identity", fake)


class TestPerMethodIdentity(unittest.TestCase):
    """A store keeps each method's rows while that method's identity is unchanged."""

    ARGS = (
        "--max_n_nodes", "40", "--n_communities", "2", "--seeds", "0", "1",
        "--p_in", "0.3", "--difficulty", "0.4", "--noises", "0.1", "1.0",
        "--partition_seeds", "2", "--n_runs", "2", "--layout", "store", "--folder_name", "s",
    )

    def _run(self, root, methods, *extra):
        calls: list[str] = []
        real = sbm_sweep._execute_store_run

        def spy(g, gt, info, **kw):
            calls.append(info["method_call_name"] + ":" + str(kw.get("initial_membership") is not None))
            return real(g, gt, info, **kw)

        with patch.object(sbm_sweep, "_execute_store_run", spy):
            code = sbm_sweep.main(
                [*self.ARGS, "--methods", methods, "--output_root", str(root), "--resume", *extra]
            )
        self.assertEqual(code, 0)
        return calls

    def _rows(self, store, method):
        return [row for row in _store_rows(store) if row[0] == method]

    def test_adding_a_method_computes_only_it(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._run(root, "Mirror,Spectral")
            mirror = self._rows(root / "s", "Mirror")
            calls = self._run(root, "Mirror,Spectral,Hedonic")
            self.assertEqual({c.split(":")[0] for c in calls}, {"community_hedonic"})
            self.assertEqual(self._rows(root / "s", "Mirror"), mirror)
            self.assertEqual(len(self._rows(root / "s", "Hedonic")), 2 * 4 * 2)
            # A run naming fewer methods keeps the others' rows.
            self.assertEqual(self._run(root, "Hedonic"), [])
            self.assertEqual(self._rows(root / "s", "Mirror"), mirror)
            self.assertTrue(store_mod.verify_store(root / "s", sample_shards=None, sample_rows=6)["ok"])

    def test_another_core_build_refuses_hedonic_only_and_reruns_it(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._run(root, "Mirror,Hedonic,Spectral")
            mirror = self._rows(root / "s", "Mirror")
            with _other_build(hedonic_partition_fingerprint="partitions-v1:" + "0" * 64):
                with self.assertRaises(SystemExit) as raised:
                    self._run(root, "Mirror,Hedonic,Spectral")
                self.assertIn("holds rows of Hedonic computed", str(raised.exception))
                plan = sbm_sweep.preflight_grid(
                    folder_name="s", max_n_nodes=40, n_communities=[2], seeds=[0, 1], p_in=[0.3],
                    difficulty=[0.4], noises=[0.1, 1.0], partition_seeds=2,
                    methods=sbm_sweep._parse_methods("Mirror,Hedonic,Spectral"), n_runs=2,
                    layout="store", output_root=root, check_graphs=False,
                    spec=store_mod.build_spec(
                        methods=sbm_sweep._parse_methods("Mirror,Hedonic,Spectral"), n_runs=2,
                        partitions="full", generator_id=sbm_sweep.SBM_GENERATOR_ID),
                )
                self.assertEqual(plan["method_graphs"]["Mirror"], {"current": 2, "other": 0, "missing": 0})
                self.assertEqual(plan["method_graphs"]["Hedonic"], {"current": 0, "other": 2, "missing": 0})
                calls = self._run(root, "Mirror,Hedonic,Spectral", "--rerun", "hedonic")
                self.assertEqual({c.split(":")[0] for c in calls}, {"community_hedonic"})
                self.assertEqual(self._run(root, "Mirror,Hedonic,Spectral"), [])
            self.assertEqual(self._rows(root / "s", "Mirror"), mirror)

    def test_another_source_with_the_same_fingerprint_is_replayed_and_adopted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._run(root, "Mirror,Hedonic")
            store = root / "s"
            hedonic = self._rows(store, "Hedonic")
            first = store_mod.runtime_identity()["hedonic_core_source_sha256"]

            def durations():
                df = store_mod.load_runs(store)
                return list(df[df["method"] == "Hedonic"].sort_values(list(store_mod.KEY_COLUMNS))["duration"])

            timed = durations()
            # hedonic 1 and hedonic 2: other sources, the same partitions bit for bit.
            with _other_build(hedonic_core_source_sha256="1" * 64):
                with patch.object(store_mod, "_replay_shards", wraps=store_mod._replay_shards) as replay:
                    self._run(root, "Mirror,Hedonic")  # its detector calls are the replay's
                self.assertTrue(replay.called)
                self.assertEqual(durations(), timed)  # kept, not recomputed
                adopted = store_mod.read_manifest(store)["adopted"]
                self.assertEqual([(a["method"], a["source"], a["by_source"]) for a in adopted],
                                 [("Hedonic", first, "1" * 64)])
                self.assertGreater(adopted[0]["rows_replayed"], 0)
                self.assertEqual(self._rows(store, "Hedonic"), hedonic)
                # Adopted once: the next run replays nothing.
                with patch.object(store_mod, "_replay_shards", wraps=store_mod._replay_shards) as replay:
                    self.assertEqual(self._run(root, "Mirror,Hedonic"), [])
                self.assertFalse(replay.called)
                # A new graph is written by this source; the store says which source wrote what.
                code = sbm_sweep.main([*self.ARGS[:5], "0", "1", "2", *self.ARGS[7:], "--methods", "Mirror,Hedonic",
                                       "--output_root", str(root), "--resume"])
                self.assertEqual(code, 0)
                self.assertEqual(store_mod.summarize(store)["sources"], {"Hedonic": {first: 2, "1" * 64: 1}})

    def test_another_source_that_does_not_replay_is_refused_then_rerun(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._run(root, "Mirror,Hedonic")
            store = root / "s"
            for shard in store_mod.shard_paths(store):
                table = pq.read_table(shard)
                meta = table.schema.metadata
                column = table.schema.get_field_index("adjusted_rand")
                table = table.set_column(
                    column, "adjusted_rand", pa.array([0.123] * table.num_rows, pa.float64())
                ).replace_schema_metadata(meta)
                pq.write_table(table, shard)
            manifest = (store / "manifest.json").read_bytes()
            with _other_build(hedonic_core_source_sha256="1" * 64):
                with self.assertRaises(SystemExit) as raised:
                    self._run(root, "Mirror,Hedonic")
                self.assertIn("do not replay exactly", str(raised.exception))
                self.assertEqual((store / "manifest.json").read_bytes(), manifest)
                calls = self._run(root, "Mirror,Hedonic", "--rerun", "hedonic")
                self.assertEqual({c.split(":")[0] for c in calls}, {"community_hedonic"})
                self.assertEqual(store_mod.summarize(store)["sources"], {"Hedonic": {"1" * 64: 2}})
                self.assertNotIn("adopted", store_mod.read_manifest(store))

    def test_a_store_of_this_build_under_the_source_rule_stays_current(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._run(root, "Mirror,Hedonic")
            store = root / "s"
            manifest = store_mod.read_manifest(store)
            spec = manifest["spec"]
            legacy = store_mod.spec_hash(store_mod.method_identity(
                "Hedonic", spec["methods"]["Hedonic"], n_runs=spec["n_runs"], runtime=spec["runtime"], legacy=True))
            self.assertNotEqual(legacy, manifest["method_identities"]["Hedonic"])
            manifest["method_identities"]["Hedonic"] = spec["method_identities"]["Hedonic"] = legacy
            (store / "manifest.json").write_text(json.dumps(manifest))
            for shard in store_mod.shard_paths(store):  # as the sweep wrote it before the fingerprint
                table = pq.read_table(shard)
                meta = json.loads(table.schema.metadata[b"hedonic.disjoint_store"])
                meta["methods_meta"]["Hedonic"]["identity"] = legacy
                del meta["methods_meta"]["Hedonic"]["source"]
                pq.write_table(table.replace_schema_metadata(
                    {b"hedonic.disjoint_store": json.dumps(meta).encode()}), shard)
            self.assertEqual(self._run(root, "Mirror,Hedonic"), [])
            self.assertNotIn("adopted", store_mod.read_manifest(store))
            mine = store_mod.runtime_identity()["hedonic_core_source_sha256"]
            self.assertEqual(store_mod.summarize(store)["sources"], {"Hedonic": {mine: 2}})

    def test_reuse_from_carries_replayed_rows_into_a_new_store(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old = root / "old"
            self._run(old, "Mirror,Hedonic,Spectral")
            with _other_build(igraph_c="9.9.9-other"):
                calls = self._run(
                    root / "new", "Mirror,Hedonic,Spectral",
                    "--reuse-from", str(old / "s"), "--reuse-methods", "mirror,spectral",
                )
                self.assertIn("community_hedonic:True", calls)
                new = root / "new" / "s"

                def durations(store, method):
                    df = store_mod.load_runs(store)
                    df = df[df["method"] == method].sort_values(list(store_mod.KEY_COLUMNS))
                    return list(df["duration"])

                # Carried rows are the stored ones (their recorded durations), not recomputed.
                for method in ("Mirror", "Spectral"):
                    self.assertEqual(durations(new, method), durations(old / "s", method))
                self.assertNotEqual(durations(new, "Hedonic"), durations(old / "s", "Hedonic"))
                for method in ("Mirror", "Spectral", "Hedonic"):
                    self.assertEqual(self._rows(new, method), self._rows(old / "s", method))
                manifest = store_mod.read_manifest(new)
                self.assertEqual(manifest["carried"][0]["methods"], ["Mirror", "Spectral"])
                self.assertGreater(manifest["carried"][0]["rows_replayed"], 0)
                self.assertEqual(self._run(root / "new", "Mirror,Hedonic,Spectral"), [])

    def test_reuse_refuses_rows_this_build_does_not_reproduce(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old = root / "old"
            self._run(old, "Mirror")
            for shard in store_mod.shard_paths(old / "s"):
                table = pq.read_table(shard)
                meta = table.schema.metadata
                column = table.schema.get_field_index("adjusted_rand")
                table = table.set_column(
                    column, "adjusted_rand", pa.array([0.123] * table.num_rows, pa.float64())
                ).replace_schema_metadata(meta)
                pq.write_table(table, shard)
            with self.assertRaises(SystemExit) as raised:
                self._run(root / "new", "Mirror", "--reuse-from", str(old / "s"), "--reuse-methods", "mirror")
            self.assertIn("not what this build computes", str(raised.exception))
            self.assertFalse((root / "new" / "s" / "manifest.json").exists())

    def test_store_written_before_method_identities_is_resumed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._run(root, "Mirror,Hedonic")
            store = root / "s"
            manifest = store_mod.read_manifest(store)
            manifest.pop("method_identities")
            manifest["spec"].pop("method_identities")
            (store / "manifest.json").write_text(json.dumps(manifest))
            import pyarrow.parquet as pq

            for shard in store_mod.shard_paths(store):  # shard metadata as the 1.x sweep wrote it
                table = pq.read_table(shard)
                meta = json.loads(table.schema.metadata[b"hedonic.disjoint_store"])
                meta.pop("methods_meta")
                meta["spec_hash"] = manifest["spec_hash"]
                table = table.replace_schema_metadata(
                    {b"hedonic.disjoint_store": json.dumps(meta).encode()}
                )
                pq.write_table(table, shard)
            self.assertEqual(self._run(root, "Mirror,Hedonic"), [])
            self.assertIn("legacy_shard_identities", store_mod.read_manifest(store))
