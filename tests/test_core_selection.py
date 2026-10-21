"""HEDONIC_CORE=<name>: another core, hedonic._core_<name>, in place of hedonic._core for every reader of the package.

Each case runs in a fresh interpreter, since the choice is made once, when hedonic is imported."""

import os
import subprocess
import sys
import textwrap
import unittest


def run(code, core):
    env = {k: v for k, v in os.environ.items() if k != "HEDONIC_CORE"}
    if core is not None:
        env["HEDONIC_CORE"] = core
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True, text=True)


# A second core with the functions of hedonic._core: the same extension, loaded from its file before hedonic is
# imported and registered as hedonic._core_probe with its own backend name.
PROBE = """
    import glob, importlib.util, os, sys, types

    folder = os.path.dirname(importlib.util.find_spec("hedonic").origin)
    path = [p for p in glob.glob(os.path.join(folder, "_core.*")) if p.endswith((".so", ".pyd"))][0]
    spec = importlib.util.spec_from_file_location("hedonic._core", path)
    real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real)
    probe = types.ModuleType("hedonic._core_probe")
    probe.__dict__.update({k: v for k, v in vars(real).items() if not k.startswith("__")})
    probe.backend = "probe"
    sys.modules["hedonic._core_probe"] = probe

    import igraph
    import hedonic
    from hedonic import _core, _native
    from hedonic.experiments import runtime

    assert _core is probe and sys.modules["hedonic._core"] is probe and hedonic._core is probe
    assert _native._detect is probe.detect
    p = hedonic.Game(igraph.Graph.Famous("Zachary")).community_hedonic(seed=0)
    print(len(set(p.membership)), runtime.core_built_source_sha256() == real.source_sha256)
"""


class TestCoreSelection(unittest.TestCase):
    def test_unset_or_empty_loads_hedonic_core(self):
        for core in (None, ""):
            out = run("import hedonic; from hedonic import _core; print(_core.__spec__.name)", core)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.strip(), "hedonic._core")

    def test_another_core_replaces_hedonic_core_everywhere(self):
        out = run(PROBE, "probe")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split(), ["5", "True"])

    def test_the_name_of_the_default_core_selects_it(self):
        backend = run("from hedonic import _core; print(getattr(_core, 'backend', ''))", None).stdout.strip()
        if not backend:
            self.skipTest("hedonic._core names no backend")
        out = run("import hedonic; from hedonic import _core; print(_core.__spec__.name)", backend)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "hedonic._core")

    def test_a_core_that_is_not_installed_is_an_error_naming_the_variable(self):
        out = run("import hedonic", "nosuchcore")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("HEDONIC_CORE='nosuchcore'", out.stderr)
        self.assertIn("hedonic._core_nosuchcore", out.stderr)

    def test_a_name_that_is_not_an_identifier_is_refused(self):
        out = run("import hedonic", "../x")
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("HEDONIC_CORE='../x' is not a core's name", out.stderr)


if __name__ == "__main__":
    unittest.main()
