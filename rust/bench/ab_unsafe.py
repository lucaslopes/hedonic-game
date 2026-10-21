"""What the crate's unchecked indexing buys: the crate against a copy with bounds checks.

Builds examples/bench_core twice, from this crate (A: as it is) and from a
copy (B) in which every `get_unchecked`/`get_unchecked_mut` of the chosen
files is checked indexing. Runs the two binaries alternately, keeps the
fastest time of each case and phase, checks that both compute the same
hashes, and prints B / A (above 1: the unchecked code is faster).

    python rust/bench/ab_unsafe.py --group partition --rounds 8
    bash .agents/skills/hedonic-native-core/scripts/guard.sh 8 python rust/bench/ab_unsafe.py --group all --rounds 20

Groups: partition, cover, audit (the files that keep unchecked reads), all.
rust/README.md ("Unsafe code") records the numbers each one was kept on; the
graph's adjacency build and the queue, measured the same way, gained nothing
and are checked code now.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
GROUPS = {
    "partition": ["partition.rs"],
    "cover": ["cover.rs"],
    "audit": ["audit.rs"],
}
GROUPS["all"] = sorted({f for files in GROUPS.values() for f in files})

AT = """
/// Checked stand-ins for get_unchecked (bench/ab_unsafe.py's B build).
pub trait At<T> {
    fn at<I: std::slice::SliceIndex<[T]>>(&self, i: I) -> &I::Output;
    fn at_mut<I: std::slice::SliceIndex<[T]>>(&mut self, i: I) -> &mut I::Output;
}

impl<T> At<T> for [T] {
    #[inline(always)]
    fn at<I: std::slice::SliceIndex<[T]>>(&self, i: I) -> &I::Output {
        &self[i]
    }
    #[inline(always)]
    fn at_mut<I: std::slice::SliceIndex<[T]>>(&mut self, i: I) -> &mut I::Output {
        &mut self[i]
    }
}
"""


def checked(text):
    text = text.replace(".get_unchecked_mut(", ".at_mut(").replace(".get_unchecked(", ".at(")
    lines = text.split("\n")
    head = next(i for i, line in enumerate(lines) if not line.startswith("//!"))
    lines.insert(head, "#[allow(unused_imports)]\nuse crate::At;")
    return "\n".join(lines)


def build(root, target):
    env = dict(os.environ, CARGO_BUILD_JOBS=os.environ.get("CARGO_BUILD_JOBS", "4"),
               RUSTFLAGS=os.environ.get("RUSTFLAGS", "") + " -A unused_unsafe")
    subprocess.run(["cargo", "build", "--release", "--locked", "-q", "--example", "bench_core",
                    "--target-dir", str(target)], cwd=root, check=True, env=env)
    return target / "release" / "examples" / "bench_core"


def parse(out):
    rows = {}
    for line in out.splitlines():
        name, rest = line[:28].strip(), line[28:]
        times = {k: float(v) for k, v in re.findall(r"(build|run|audit)\s+([\d.]+) ms", rest)}
        rows[name] = (times, re.search(r"hash (\w+)", rest).group(1))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", choices=sorted(GROUPS), default="all")
    p.add_argument("--rounds", type=int, default=6, help="alternations of A and B")
    p.add_argument("--reps", type=int, default=3, help="runs of each case per binary call")
    p.add_argument("--filter", default="")
    args = p.parse_args(argv)

    work = CRATE / "target" / "ab" / args.group
    copy = work / "crate"
    if copy.exists():
        shutil.rmtree(copy)
    copy.mkdir(parents=True)
    for name in ("Cargo.toml", "Cargo.lock", "build.rs"):
        shutil.copy2(CRATE / name, copy / name)
    shutil.copytree(CRATE / "src", copy / "src")
    shutil.copytree(CRATE / "examples", copy / "examples")
    for name in GROUPS[args.group]:
        path = copy / "src" / name
        path.write_text(checked(path.read_text()))
    lib = copy / "src" / "lib.rs"
    lib.write_text(lib.read_text() + AT)

    a = build(CRATE, work / "target-a")
    b = build(copy, work / "target-b")
    best = {"A": {}, "B": {}}
    hashes = {"A": {}, "B": {}}
    for _ in range(args.rounds):
        for label, exe in (("A", a), ("B", b)):
            out = subprocess.run([str(exe), str(args.reps), args.filter], check=True, capture_output=True,
                                 text=True).stdout
            for case, (times, h) in parse(out).items():
                hashes[label][case] = h
                for phase, t in times.items():
                    key = (case, phase)
                    best[label][key] = min(best[label].get(key, float("inf")), t)
    if hashes["A"] != hashes["B"]:
        sys.exit(f"the builds compute different results: {hashes}")
    print(f"group {args.group} ({', '.join(GROUPS[args.group])}): B (checked) / A (unchecked), "
          f"fastest of {args.rounds * args.reps}")
    for (case, phase), ta in best["A"].items():
        tb = best["B"][(case, phase)]
        print(f"  {case:<28} {phase:<6} A {ta:8.2f} ms  B {tb:8.2f} ms  B/A {tb / ta:5.3f}")


if __name__ == "__main__":
    main()
