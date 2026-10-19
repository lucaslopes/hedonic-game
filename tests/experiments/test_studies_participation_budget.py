"""The participation-budget illustration: intensities, pyramid geometry, the figure and CLI registration."""

from __future__ import annotations

import io
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from hedonic.experiments import CLI
from hedonic.experiments.plots import participation_budget as pb


class TestParticipationBudget(unittest.TestCase):
    def test_intensities_spend_a_unit_budget(self):
        for k in range(1, 12):
            self.assertAlmostEqual(k * pb.intensity(k) ** 2, 1.0)
        self.assertGreater(pb.intensity(1), pb.intensity(2))
        with self.assertRaises(ValueError):
            pb.intensity(0)

    def test_pyramids_stand_on_a_common_ground(self):
        def rows(n):
            heights = sorted({round(y, 9) for _, y in pb.pyramid_positions(n)})
            return [sum(round(y, 9) == h for _, y in pb.pyramid_positions(n)) for h in heights]

        self.assertEqual(rows(1), [1])
        self.assertEqual(rows(3), [2, 1])
        self.assertEqual(rows(4), [3, 1])
        self.assertEqual(rows(8), [4, 3, 1])
        self.assertEqual(rows(10), [4, 3, 2, 1])
        for n in range(1, 11):
            positions = pb.pyramid_positions(n)
            self.assertEqual(len(positions), n)
            self.assertEqual(min(y for _, y in positions), 0.0)
            if n > 1:
                nearest = min(math.dist(a, b) for i, a in enumerate(positions) for b in positions[i + 1:])
                self.assertAlmostEqual(nearest, 1.0)
        self.assertEqual(pb.pyramid_positions(0), [])

    def test_figure_has_one_panel_per_k_with_k_circles(self):
        fig = pb.figure(5, colorbar=False)
        panels = fig.axes
        self.assertEqual(len(panels), 5)
        self.assertEqual([len(ax.patches) for ax in panels], [1, 2, 3, 4, 5])
        colours = [panels[k - 1].patches[0].get_facecolor() for k in range(1, 6)]
        self.assertEqual(len(set(colours)), 5)
        with self.assertRaises(ValueError):
            pb.figure(0)

    def test_cli_writes_the_figure(self):
        self.assertIn("participation-budget", CLI.COMMANDS)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(CLI.main(["participation-budget", "--help"]), 0)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "budget.pdf"
                self.assertEqual(CLI.main(["participation-budget", "--k-max", "4", "--output", str(out)]), 0)
                self.assertGreater(out.stat().st_size, 0)

    def test_bar_figure_has_one_bar_per_k_at_its_intensity(self):
        fig = pb.bar_figure(12)
        bars = fig.axes[0].patches
        self.assertEqual(len(bars), 12)
        for k, bar in enumerate(bars, start=1):
            self.assertAlmostEqual(bar.get_height(), pb.intensity(k))
        self.assertGreater(sum(bars[0].get_facecolor()[:3]), 0)
        self.assertLess(sum(bars[0].get_facecolor()[:3]), sum(bars[-1].get_facecolor()[:3]))  # k=1 darkest
        with redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bars.pdf"
            self.assertEqual(pb.main(["--bars", "--k-max", "5", "--width", "3.33", "--height", "1.6",
                                      "--output", str(out)]), 0)
            self.assertGreater(out.stat().st_size, 0)

    def test_help(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
            pb.main(["--help"])
        self.assertEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
