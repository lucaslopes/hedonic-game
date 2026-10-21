"""Which code of the ground-truth spectrum computes what earlier code computed: the spectrum's equivalence table.

The spectrum (``overlapping/gt_spectrum.py``) files every record under ``tracked_files_sha256``: one SHA-256 over the
bytes of its ``SOURCE_MODULES``, of ``hedonic/*.py`` and of ``hedonic/_game/*.py`` (``tracked_file_hashes``) together
with the runtime identity (results series and core class, ``runtime.build_identity``). A record is reused only under
the digest it was written under, so any byte change in a tracked file, a comment included, reruns the whole study.
This module is the way out for changes that do not touch results: it lives in ``hedonic/experiments/``, outside every
tracked file, so the table can grow without changing the digests it describes.

``ALIASES`` maps the digest of later code to the digest of the earlier code it is proven to compute the same as::

    "<later digest>": {
        "earlier": "<earlier digest>",
        "files": {"<path in the package>": ["<earlier SHA-256 or None>", "<later SHA-256 or None>"], ...},
        "why": "what changed and why no result can",
    }

``files`` lists every tracked file whose bytes differ: ``[None, sha]`` is a file added since, ``[sha, None]`` one
removed. Undoing ``files`` on the later per-file hashes gives the earlier ones, and hashing those with the runtime
identity gives ``earlier``: :func:`earlier_files` checks this at every step and ignores an entry that does not hold.
"Proven" means the change is identity or plumbing (names, help text, the identity machinery, a new module the study
does not run), never a computation; say how in ``why``. ``ROOTS`` are the digests whose records started a code
identity (results changed, or the first code under this rule), with their per-file hashes.

The loaded code names the earliest digest of its chain (``gt_spectrum.tracked_digest``), so records written by it
are filed exactly as the earlier code filed them and the earlier code's records stay current; the real digest and
per-file hashes are kept as provenance (``tracked_files_sha256_built``, ``tracked_files_built``). Stored records are
never rewritten. ``tests/experiments/test_spectrum_code.py`` recomputes the loaded code's digest: editing a tracked
file without an entry here fails it, and its message prints the entry to add (or says to add a root, when results
change).

The next entry (2.1, the Rust core: ``hedonic/__init__.py`` changes and new modules join ``hedonic/*.py``) is added
the same way, on the release line that makes the change: run the test, check that only identity or plumbing
changed, paste the printed entry. It maps 2.1's digest to the digest 2.0 records name (the chain may go through
the 2.0.0rc12 entry; ``canonical`` follows it to the root). A core of the same results class keeps the runtime part
of the digest unchanged; a new class changes it, and no entry may bridge that (the results differ).
"""

from __future__ import annotations

from typing import Callable

# Digests whose records started a code identity: what they are, and their per-file hashes (what an entry of
# ``ALIASES`` is written against).
ROOTS: dict[str, dict] = {
    "18ca62d99ead6c879cdb1cb67e45b3f28eaad9d9d912a1b36f343d5f09efad13": {
        "what": "2.0.0rc11 (results series 2.0, core class ead786fe...): the overlap queue's gt-robustness records",
        "files": {
            "hedonic/Game.py": "524cd68b33e0076960d6e82ba867a28a7a49cea0ac59653d5ea9d8cfa4c0f505",
            "hedonic/__init__.py": "3dfd738c06a31430c417836b0b265dc523c0ac7f429f0d58776235733b85eacb",
            "hedonic/_audit.py": "d8f8e6a259047aa7240dbe0253a15a7ad5d46ec9e29a94860c6151f2ab51ff16",
            "hedonic/_game/__init__.py": "57344739404fa2cf6ce3386641cf95cb6f9765cf414e02c21af21d147f30bbe6",
            "hedonic/_game/detection.py": "82dcab563f39ff9079c4474b38a8e4fceb0883bf927af1ddfe2a90afddfe00b6",
            "hedonic/_game/evaluation.py": "d1b099adc4efe29fc1d57a26ee5a0695e38f3aa5d2dbec78c2111dfeed63dc53",
            "hedonic/_game/helpers.py": "d958d0e8776db670632c3e73e1df3603099dd580b97b4d4c07511ead0253817f",
            "hedonic/_game/memberships.py": "0933215338c836793bfc4de76888771167a41dab80b4bc8386e6303f8445b58b",
            "hedonic/_native.py": "e597ccfe3fe419da718da822a4c78ab48841ec1e1cef201c9314e761b341cdb8",
            "hedonic/experiments/overlapping/execution.py": "8c433a9e37051f82f1be3256908ead5ba290aa63e2781300138726a25fc3d36b",
            "hedonic/experiments/overlapping/ground_truth_data.py": "def7b6401ad7d7d8a36829e8cf02040e092b0dc7f7df4edce867550ea120ef56",
            "hedonic/experiments/overlapping/ground_truth_robustness.py": "84d995082bb30adaf3da531d97a6c49c5898455318cd25b4717432318b71cf60",
            "hedonic/experiments/overlapping/gt_spectrum.py": "71dc4133776741bd9b2ba8ac58105f2e1cfd0f7526a0e65887b9f130327de6fc",
            "hedonic/experiments/overlapping/gt_spectrum_report.py": "88422d34ed3452e6a5091bc51e46628157fd64cea9b3a14a2a692fe655d0ed52",
            "hedonic/experiments/overlapping/metrics.py": "a8c2895e54bbcb7dec01a782642a9077a06e0bca85d8dc0d8fba01fe13fc6989",
            "hedonic/experiments/overlapping/robustness.py": "83a96f4723275fd70da4a16f4afc7541981ce33fa3b7645afffdb0cfd685730c",
            "hedonic/experiments/overlapping/snap.py": "e0e098defd86d75d86a32ff493aa3dc6a7af98de219e0422f925f6decd2eeeac",
            "hedonic/experiments/overlapping/stability.py": "d8892b7ba9b7f34ee06db8f93f2c39e9e2ee7dd9ddd9b70d1e6b8b112470e60c",
            "hedonic/metrics.py": "c6fe6a3c9797d6d522dc81e2e06f2b9c07a65b618b80586efd4ee47b978190db",
            "hedonic/stability.py": "d8095a9cff6d794d7106d735eb85a9fe25be5ad59da96da78fd2e9afdf811751",
            "hedonic/utils.py": "95994a0144deee7754015205bd30ad870a494e1f4c81ee350c6fbff89dec150d",
        },
    },
}

ALIASES: dict[str, dict] = {
    # 2.0.0rc12: the identity machinery (gt_spectrum consults this table, names the core class through
    # runtime.algorithm_identity() rather than a module constant, and keeps the loaded files' hashes as provenance).
    "ed05acde78603c2120ff9bd7e6cf8a080a104ae9f52b07a031fcdabd6914ac74": {
        "earlier": "18ca62d99ead6c879cdb1cb67e45b3f28eaad9d9d912a1b36f343d5f09efad13",
        "files": {
            "hedonic/experiments/overlapping/gt_spectrum.py": ["71dc4133776741bd9b2ba8ac58105f2e1cfd0f7526a0e65887b9f130327de6fc", "15797ad12d44ff5390fb3489e1b4c0133c82b689dc05a9704f2ac48697a61dce"],
        },
        "why": "identity and plumbing only: tracked_digest, implementation_identity, canonical_digest and the drift check "
               "(now implementation_drift) read spectrum_code; _runtime_identity calls runtime.algorithm_identity() "
               "(the same string); docstrings. No detector call, audit, score or record field other than the "
               "identity's provenance (tracked_files_built, tracked_files_sha256_built) changed.",
    },
}


def canonical(digest: str | None) -> str | None:
    """The digest records of ``digest``'s code are filed under: the end of its chain of ``ALIASES`` (itself if none)."""
    seen = set()
    while digest in ALIASES and digest not in seen:
        seen.add(digest)
        digest = ALIASES[digest]["earlier"]
    return digest


def earlier_files(files: dict[str, str | None], digest_of: Callable[[dict[str, str | None]], str]
                  ) -> tuple[dict[str, str | None], str]:
    """The per-file hashes and the digest of the earliest code that ``files`` is proven equivalent to.

    ``digest_of`` hashes a per-file map with the runtime identity (``gt_spectrum._implementation_digest``). Each step
    of the chain must reproduce its entry exactly: the later files are those the entry names, and undoing them gives
    the entry's ``earlier`` digest. A step that does not is not taken (the code is then judged by what it is)."""
    current = dict(files)
    digest = digest_of(current)
    seen = set()
    while digest in ALIASES and digest not in seen:
        seen.add(digest)
        entry = ALIASES[digest]
        if any(current.get(path) != later for path, (_earlier, later) in entry["files"].items()):
            break
        before = dict(current)
        for path, (earlier, _later) in entry["files"].items():
            if earlier is None:
                before.pop(path, None)
            else:
                before[path] = earlier
        before = dict(sorted(before.items()))
        if digest_of(before) != entry["earlier"]:
            break
        current, digest = before, entry["earlier"]
    return current, digest


def suggest(files: dict[str, str | None], earlier: dict[str, str | None], later_digest: str, earlier_digest: str
            ) -> str:
    """The ``ALIASES`` entry that maps code with per-file hashes ``files`` to the code with ``earlier``."""
    changed = {path: [earlier.get(path), files.get(path)] for path in sorted(set(files) | set(earlier))
               if earlier.get(path) != files.get(path)}
    lines = [f'    "{later_digest}": {{', f'        "earlier": "{earlier_digest}",', '        "files": {']
    for path, (old, new) in changed.items():
        lines.append(f'            "{path}": [{_quoted(old)}, {_quoted(new)}],')
    lines += ['        },', '        "why": "<what changed, and why no result can>",', '    },']
    return "\n".join(lines)


def _quoted(value: str | None) -> str:
    return "None" if value is None else f'"{value}"'
