"""Release-environment expectations read from the repository's resolved lock."""

from pathlib import Path
import tomllib


def locked_lucas_igraph_version(lock_path: Path | None = None) -> str:
    """Return the unique exact lucas-igraph version, failing on an invalid lock.

    This intentionally does not inspect the installed runtime or substitute
    pyproject.toml: a release environment must agree with its resolved lock.
    """
    if lock_path is None:
        lock_path = Path(__file__).resolve().parents[1] / "uv.lock"
    data = tomllib.loads(lock_path.read_bytes().decode("utf-8"))
    matches = [row for row in data.get("package", []) if row.get("name") == "lucas-igraph"]
    if len(matches) != 1:
        raise ValueError("uv.lock must contain exactly one lucas-igraph package")
    version = matches[0].get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("uv.lock lucas-igraph package must have an exact version")
    return version


# Both frozen overlapping protocol locks (the 125-condition paper protocol and
# the ground-truth robustness v3 protocol) name the stack that produced their
# ledgers. They are never rewritten (AGENTS.md), so a newer runtime pin in
# uv.lock is expected to differ from them.
FROZEN_PRODUCER_LUCAS_IGRAPH = "1.0.0.3"


def assert_live_stack_reported_against_frozen_lock(test, lucas_identity) -> None:
    """The identity names the frozen producer, reads the running stack, and
    reports whether they agree, whatever the running stack is."""
    test.assertEqual(lucas_identity["expected_version"], FROZEN_PRODUCER_LUCAS_IGRAPH)
    test.assertEqual(lucas_identity["expected_igraph_version"], FROZEN_PRODUCER_LUCAS_IGRAPH)
    test.assertEqual(lucas_identity["actual_version"], locked_lucas_igraph_version())
    test.assertEqual(
        lucas_identity["version_matches_lock"],
        lucas_identity["actual_version"] == lucas_identity["expected_version"],
    )
    test.assertEqual(
        lucas_identity["igraph_version_matches_lock"],
        lucas_identity["actual_igraph_version"] == lucas_identity["expected_igraph_version"],
    )


def live_lucas_igraph_lock_section(production_section: dict, live_identity: dict) -> dict:
    """A lucas_igraph lock section describing the running stack, for tests
    that exercise the accepting path of a lock without rewriting the frozen
    one. The module must still belong to the installed distribution (true for
    a wheel install, false for an editable python-igraph checkout)."""
    live = live_identity["lucas_igraph"]
    section = dict(production_section)
    section.update(
        version=live["actual_version"],
        igraph_version=live["actual_igraph_version"],
        uv_lock_path=str((Path(__file__).resolve().parents[1] / "uv.lock")),
        uv_lock_sha256=live["actual_uv_lock_sha256"],
        package_lock_sha256=live["actual_package_lock_sha256"],
        distribution_tree_sha256=live["actual_distribution_tree_sha256"],
        module=live["actual_module"]["module"],
        module_path=live["actual_module"]["path"],
        module_sha256=live["actual_module"]["sha256"],
    )
    return section


def live_dependency_lock_sections(production_sections: dict, live_sections: dict) -> dict:
    """External or scientific dependency lock sections describing the running
    environment (same purpose as live_lucas_igraph_lock_section)."""
    sections = {}
    for name, spec in production_sections.items():
        live = live_sections[name]
        section = dict(spec)
        section.update(
            version=live["actual_version"],
            package_lock_sha256=live["actual_package_lock_sha256"],
            distribution_tree_sha256=live["actual_distribution_tree_sha256"],
        )
        if isinstance(live.get("actual_module"), dict):
            section.update(
                module=live["actual_module"]["module"],
                module_path=live["actual_module"]["path"],
                module_sha256=live["actual_module"]["sha256"],
            )
        if isinstance(live.get("actual_implementation"), dict):
            section.update(live["actual_implementation"])
        sections[name] = section
    return sections


def lucas_igraph_is_an_installed_distribution() -> bool:
    """False for an editable or locally built python-igraph, whose module
    cannot belong to the lucas-igraph distribution a lock names."""
    from hedonic.experiments.overlapping import protocol

    live = protocol.current_experiment_identity()["lucas_igraph"]
    return live.get("module_belongs_to_distribution") is True


INSTALLED_DISTRIBUTION_REASON = (
    "the accepting path of a protocol lock needs lucas-igraph installed as a "
    "distribution (an editable python-igraph build is never accepted)"
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_EVIDENCE = REPO_ROOT / "artifacts" / "evidence" / "overlapping_communities"
PUBLIC_CHECKOUT_REASON = (
    "needs the private research evidence or files that a frozen lock names; "
    "a public checkout does not carry them"
)


def lock_tracked_files_present(lock_path: Path) -> bool:
    """Are all files a frozen lock tracks present? (False in a public
    checkout, where manuscript-only configuration is not shipped.)"""
    import json

    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    return all((REPO_ROOT / relative).is_file() for relative in lock.get("tracked_files", {}))


def private_evidence_present(*relative: str) -> bool:
    return all((PRIVATE_EVIDENCE / name).exists() for name in relative)
