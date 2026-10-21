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

A later entry is added the same way, on the release line that makes the change: run the test, check that only
identity or plumbing changed, paste the printed entry (or chain it to the latest entry, as 2.1's is to rc12's;
``canonical`` follows the chain to the root). A core of the same results class keeps the runtime part of the digest
unchanged; a new class changes it, and no entry may bridge that (the results differ).
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
    # 2.1: the Rust core and the array API. The package's __init__ selects a core (HEDONIC_CORE) and imports Game
    # lazily; detection.plan, _audit.state and helpers.real_vector are the argument checks Game and hedonic.arrays share
    # (the same checks, messages, start and provenance, moved out of community_hedonic); arrays.py is a new module the
    # study does not run; Game.py and _native.py change in docstrings only.
    "26062019eed2a9b65501db573dc08af3d6bd6f67130d68fe34376f66fbcd99a5": {
        "earlier": "ed05acde78603c2120ff9bd7e6cf8a080a104ae9f52b07a031fcdabd6914ac74",
        "files": {
            "hedonic/Game.py": ["524cd68b33e0076960d6e82ba867a28a7a49cea0ac59653d5ea9d8cfa4c0f505", "405c34fd0e9feaeeb0f0e105186c5daf4ea321349793c5e7823db8be6e0e6b79"],
            "hedonic/__init__.py": ["3dfd738c06a31430c417836b0b265dc523c0ac7f429f0d58776235733b85eacb", "cb51a6f79f105459797fd1ca0d55ce427e7678164b13da63b42e9b36a3de40b5"],
            "hedonic/_audit.py": ["d8f8e6a259047aa7240dbe0253a15a7ad5d46ec9e29a94860c6151f2ab51ff16", "8bff73bc42d3cbe0219d7003c26a7359fd80ed8acbdfa55b7b99a42f909eee86"],
            "hedonic/_game/detection.py": ["82dcab563f39ff9079c4474b38a8e4fceb0883bf927af1ddfe2a90afddfe00b6", "ca90b6fc57f8f01fe47eea96536979e1376242cdc2dc3da5b4c46b58caa2198f"],
            "hedonic/_game/helpers.py": ["d958d0e8776db670632c3e73e1df3603099dd580b97b4d4c07511ead0253817f", "7bb582144d7bfcb7cd6d8cd9b844cf8f9725d8897bce329e2d5e4dd4d34e380f"],
            "hedonic/_game/memberships.py": ["0933215338c836793bfc4de76888771167a41dab80b4bc8386e6303f8445b58b", "9830410d7510723580360174df90d682eac7a8b12180eeca48c04f2b93309b64"],
            "hedonic/_native.py": ["e597ccfe3fe419da718da822a4c78ab48841ec1e1cef201c9314e761b341cdb8", "5fd5bf03cf8c8b8346c5e56d3ce54724acdfe4d45f482454f0d9f365cc66b76a"],
            "hedonic/arrays.py": [None, "9cd43a29450b941e5679f839f8dc08a3c5b05e5ae09f582d66a33f1ca8ca302f"],
        },
        "why": "plumbing only: the checks of a detector call and of an audit's cover moved into functions that take a "
               "vertex count and an edge source (plan, state, normalize_rows, real_vector), with the same checks in "
               "the same order, the same start, arguments to the core and provenance; igraph is looked up instead "
               "of imported; the core is chosen at import (HEDONIC_CORE unset: hedonic._core). tests/test_arrays.py "
               "compares Game's and the array path's results bit for bit, and the core class (fingerprints) is "
               "unchanged, so the runtime part of the digest is too.",
    },
    # 2.1, after the first 2.1 entry: hedonic.arrays reaches the core through the binding's buffer entry points
    # (detect_arrays, _audit_prepare_arrays) and passes counts and labels to NumPy as the platform's index type;
    # _audit.state takes the core's prepare call; and the Python layer's shortcuts (the perflog's deferred proposals).
    "4c36ae269c1455c36416d78befc5feb5d54f31f40cea83f96cacf6ab0b4f22b4": {
        "earlier": "26062019eed2a9b65501db573dc08af3d6bd6f67130d68fe34376f66fbcd99a5",
        "files": {
            "hedonic/_audit.py": ["8bff73bc42d3cbe0219d7003c26a7359fd80ed8acbdfa55b7b99a42f909eee86", "3e04f78e601a27c36fedde960ceec913d0c4eccfbd06b95ae1bcecd4557b1ca6"],
            "hedonic/_game/detection.py": ["ca90b6fc57f8f01fe47eea96536979e1376242cdc2dc3da5b4c46b58caa2198f", "5db161f8a18be4296a083b4a88885cbb219e7adeedb75979d3ad886f86299564"],
            "hedonic/_game/helpers.py": ["7bb582144d7bfcb7cd6d8cd9b844cf8f9725d8897bce329e2d5e4dd4d34e380f", "c5e9a7bec782097fa9c5cf5fa062dcf466dcf1888c03dbcc9e6deaa8c2e4b1c9"],
            "hedonic/_game/memberships.py": ["9830410d7510723580360174df90d682eac7a8b12180eeca48c04f2b93309b64", "a85a3985585dfff83e5dd272cca23c16f508e48576bb1a7287e47b163ca7b83f"],
            "hedonic/_native.py": ["5fd5bf03cf8c8b8346c5e56d3ce54724acdfe4d45f482454f0d9f365cc66b76a", "58578d1b6130e213fb69402d06a42924216efdae342b9ce71e9f2b83b82a2114"],
            "hedonic/arrays.py": ["9cd43a29450b941e5679f839f8dc08a3c5b05e5ae09f582d66a33f1ca8ca302f", "c5e1e76721301c215490ccad1046d8e10ed6ee7f3b92b3c67d0ba01638249705"],
        },
        "why": "plumbing and speed only. Game's audit passes _core._audit_prepare on the graph as before (now through a "
               "prepare function); arrays.py, which the study does not run, hands the core the same edges, weights "
               "and start as buffers and casts counts and labels to np.intp (the same values); normalize_rows reads "
               "plain-int memberships without its per-label checks and falls back to them for anything else (the "
               "same rows and the same first error, tests/test_python_layer_paths.py); a default singleton start is "
               "passed to the core as None, whose default is the same singletons (the same bits, checked there and "
               "by the three fingerprints); the self-loop check asks is_simple() first; the occupied count is the "
               "same set built in C; a VertexCover start is read without a second copy of its rows; the cyclic "
               "collector is held while large row lists are built (gc_paused). tests/test_arrays.py compares the "
               "array path with Game bit for bit; provenance, records and the core class are unchanged.",
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
