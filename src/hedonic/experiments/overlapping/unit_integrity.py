"""Is a benchmark unit's record really there, intact and finished? Hash-certified, with no detector work.

A benchmark unit (network, seed, method) leaves a record ``runs/<net>/<runner>.json`` and, under ``--save-covers``,
a cover ``covers/<net>/<runner>.json.gz``. Beside ``runs/`` sits a seal, ``integrity/<net>/<runner>.json``: the SHA-256 of the record's bytes and of the cover's bytes, taken when the unit finished. A unit is certified when

* the record parses and has a final status (completed, unavailable or failed),
* its ``config_digest`` is the one the runner would stamp now (the same settings),
* the cover it names exists, and
* the bytes of record and cover are the sealed ones.

States (``Check.state``):

``verified``  sealed, and every hash matches
``unsealed``  valid, but it has no seal yet (written before seals existed, or placed there by hand); ``adopt`` seals it
``missing``   no usable record (absent, still ``running``, or its cover is absent): the unit has to run
``stale``     a record of other settings: the unit has to run again, nothing is wrong with the file
``corrupt``   unreadable, or its bytes differ from the seal: never trusted; ``quarantine`` moves it aside

Sealing a record that already exists trusts it as it is (the hashes are taken now). A seal made by the run itself
certifies the bytes it wrote. Nothing here deletes a file: a bad record is renamed, so it can still be inspected.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

SEAL_DIR = "integrity"
SCHEMA = 1
FINAL = ("completed", "unavailable", "failed")
TIMEOUT = "timeout"  # a run a queue's time limit stopped (``limits``): final only under a limit no larger than that one


def final(record: dict, limit: float | None = None) -> bool:
    """Is this record the unit's answer for a run under the queue's per-run ``limit`` (None: no limit)?"""
    from hedonic.experiments.overlapping import limits

    status = record.get("status")
    return status in FINAL or (status == TIMEOUT and limits.stands(record, limit))
OK_STATES = ("verified", "unsealed")


@dataclass
class Check:
    state: str
    reason: str = ""
    record: dict | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        """The record can be reused as it is."""
        return self.state in OK_STATES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record_path(out: Path, network: str, runner: str) -> Path:
    return out / "runs" / network / f"{runner}.json"


def seal_path(out: Path, network: str, runner: str) -> Path:
    """Seals live in their own folder, so nothing that lists ``runs/`` ever mistakes one for a record."""
    return out / SEAL_DIR / network / f"{runner}.json"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def read_seal(out: Path, network: str, runner: str) -> dict | None:
    seal = _read_json(seal_path(out, network, runner))
    return seal if isinstance(seal, dict) else None


def check(out: Path, network: str, runner: str, digest: str | None, save_covers: bool,
          limit: float | None = None) -> Check:
    """Certify one unit from the files on disk. ``digest`` is the config digest the runner would stamp now; ``limit``
    the queue's per-run time limit, under which a run that limit (or a smaller one) stopped is final."""
    path = record_path(out, network, runner)
    if not path.is_file():
        return Check("missing", "no record")
    record = _read_json(path)
    if not isinstance(record, dict):
        return Check("corrupt", "the record is not readable JSON")
    if not final(record, limit):
        return Check("missing", f"the record is {record.get('status')!r}, not final")
    if digest is not None and record.get("config_digest") != digest:
        return Check("stale", "the record was made with other settings", record)
    cover = None
    if save_covers and record.get("status") == "completed":
        if not record.get("cover_path"):
            return Check("missing", "the record names no cover", record)
        cover = out / str(record["cover_path"])
        if not cover.is_file():
            return Check("missing", "its cover is missing", record)
    seal_file = seal_path(out, network, runner)
    if not seal_file.is_file():
        return Check("unsealed", "no seal yet", record)
    seal = _read_json(seal_file)
    if not isinstance(seal, dict):
        return Check("corrupt", "the seal is not readable JSON", record)
    if seal.get("record_sha256") != sha256_file(path):
        return Check("corrupt", "the record's hash differs from its seal", record)
    if cover is not None and seal.get("cover_sha256") != sha256_file(cover):
        return Check("corrupt", "the cover's hash differs from its seal", record)
    return Check("verified", "hashes match the seal", record)


def _cover_readable(cover: Path) -> bool:
    try:
        with gzip.open(cover, "rt", encoding="utf-8") as handle:
            return isinstance(json.load(handle), list)
    except (OSError, EOFError, UnicodeDecodeError, ValueError):
        return False


def seal(out: Path, network: str, runner: str, *, by: str) -> dict:
    """Write the seal of a unit whose record is on disk (atomically) and return it. ``by``: "run" or "adopted"."""
    path = record_path(out, network, runner)
    record = _read_json(path) or {}
    body = {"schema": SCHEMA, "network": network, "method": runner, "status": record.get("status"),
            "config_digest": record.get("config_digest"), "record_sha256": sha256_file(path),
            "record_bytes": path.stat().st_size, "sealed_by": by, "sealed_at": time.time()}
    if record.get("status") == "completed" and record.get("cover_path") and (out / str(record["cover_path"])).is_file():
        cover = out / str(record["cover_path"])
        body.update(cover_path=str(record["cover_path"]), cover_sha256=sha256_file(cover), cover_bytes=cover.stat().st_size)
    target = seal_path(out, network, runner)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(body, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    return body


def adopt(out: Path, network: str, runner: str, digest: str | None, save_covers: bool,
          limit: float | None = None) -> Check:
    """``check``, and seal a valid unsealed unit after reading its cover through (its bytes are trusted from now on)."""
    result = check(out, network, runner, digest, save_covers, limit)
    if result.state != "unsealed":
        return result
    record = result.record or {}
    if save_covers and record.get("status") == "completed" and not _cover_readable(out / str(record["cover_path"])):
        return Check("corrupt", "its cover is not a readable gzip of a JSON list", record)
    seal(out, network, runner, by="adopted")
    return Check("verified", "valid; sealed now", record)


def quarantine(out: Path, network: str, runner: str, reason: str) -> list[str]:
    """Rename a corrupt unit's record, seal and cover aside (``*.corrupt-<time>``), so the unit runs again. Returns the new names."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = record_path(out, network, runner)
    record = _read_json(path) or {}
    moved = []
    targets = [path, seal_path(out, network, runner)]
    if record.get("cover_path"):
        targets.append(out / str(record["cover_path"]))
    for target in targets:
        if target.is_file():
            new = target.with_name(f"{target.name}.corrupt-{stamp}")
            os.replace(target, new)
            moved.append(str(new))
    if moved:
        note = path.with_name(f"{path.name}.corrupt-{stamp}.why")
        note.write_text(reason + "\n", encoding="utf-8")
    return moved
