---
name: sibling-release-pipeline
description: Prepare a coordinated sibling release across igraph, python-igraph, and hedonic-game by squashing work onto each 1.0.0 bump commit, fixing the vendored C-core pointer, synchronizing hedonic to 1.N.0, and validating dependency propagation. Use only when explicitly invoked for a release.
---

# Sibling release pipeline

Use this skill only after the user explicitly asks to prepare or execute a
new coordinated release. Creating this skill does not run the pipeline. The
pipeline is a guarded release-candidate workflow. Pushing a candidate branch,
opening an optional draft PR, moving a public branch, tagging, or publishing requires
explicit authorization for that external action; an unapproved run stops at
the local preflight/candidate boundary.

## Release invariant

For a target such as `1.0.0.4`, the C-core and Python-interface release commits
must each be a single direct child of that repository's existing commit whose
message bumps the version to `1.0.0`. The numbered tags are sibling commits,
not a chain where `.4` descends from `.3`. The Python release must record the
exact C-core release commit in the `vendor/source/igraph` submodule. Hedonic
Game then receives the released Python package through its dependency pin and
uses the synchronized package version `1.4.0`: in general, upstream identity
`1.0.0.N` maps to Hedonic version `1.N.0`, and Hedonic-only fixes on the same
vendored upstream are `1.N.x`. Hedonic must never use an
independently chosen patch number for a coordinated sibling release.

`1.0.0.N` is the upstream four-component release identity, not strict SemVer:
the third component remains the patch level and the fourth component is a
sibling release/iteration suffix. For this repository's coordinated release,
the corresponding Hedonic package identity is `1.N.x`: `N` is exactly the
upstream suffix and `x` counts Hedonic releases on that upstream, starting at
0 (for example, `1.0.0.4` → `1.4.0`, then `1.4.1` for a Hedonic-only fix). Preserve both conventions instead of choosing a
different Hedonic patch or prerelease without a user decision.

## Partial releases: when only some repositories change

The suffix `N` is a lookup key: `N` alone names the matching C-core,
Python-interface and Hedonic states. The rule is that every tag `1.0.0.N`
exists in both upstream repositories and that Python `N` pins the C commit
carrying the tag `1.0.0.N`. It does **not** require every repository to have
a new commit for every `N`. Decide the case during preflight by diffing each
source branch against its previous release tree:

| Changed | C-core `1.0.0.N` | Python `1.0.0.N` | Hedonic |
|---------|------------------|------------------|---------|
| C-core (with or without Python) | new one-commit release | new one-commit release; pointer changes | `1.N.0` |
| Python only | **alias tag** on the previous C release commit | new one-commit release; pointer unchanged | `1.N.0` |
| Hedonic only | none | none | `1.N.(x+1)`, see Stage 3 |

A C-core change always forces a Python release, because the submodule pointer
is part of the Python tree. A Python-only change never forces a C-core commit.

**Alias tag (Python-only).** When the C-core delta is empty, tag the existing
release commit as `1.0.0.N` (an additional annotated tag on the same commit)
instead of creating an empty commit. Preconditions: the C-core tree is
byte-identical to the previous release (`git diff --quiet <prev-tag> HEAD` on
the source line and no C-core source branch delta), the previous commit's
hosted CI is green, and the alias is recorded in the manifest as
`igraph 1.0.0.N -> <previous release hash> (alias of 1.0.0.<N-1>)`. No new
commit, no new CI run, no branch moves. Never create an `--allow-empty`
commit for this purpose. An alias tag is immutable like any other tag, and
the `1.0.0.<N-1>` tag and branch stay untouched. Python's pointer verification
then compares against the commit the alias resolves to, and the tag name is
checked to exist, not to be unique to that commit.

If a later release changes the C-core again, the new C release is a normal
one-commit child of the bump; aliases never chain into ancestry.

The observed `.1`–`.3` history is documented in
[references/observed-history.md](references/observed-history.md). Treat those
hashes as an audit record, not as hard-coded inputs: rediscover the current
refs and bump commits before every run.

## Branch roles

Keep these roles distinct in each upstream repository:

- `lucas-dev`: the private, linear integration history with the individual
  development/test commits stacked in order;
- `lucas-1.0.0.N`: the one-commit release snapshot and its matching tag;
- `lucas`: the public fork pointer to the newest validated release snapshot;
- `physica-a`: the article/research pointer to the selected validated release.

The release snapshot is intentionally a squash of the development line; this
does not erase the detailed history retained by `lucas-dev` or the original
Codex source branch.

## Release commit message continuity

The numbered sibling releases use a deliberately stable commit-message
template. The `.4` message must preserve the repository-specific subject style,
section order, and level of detail used by the preceding sibling (currently
`1.0.0.3`), while describing only the new diff. This is editorial continuity,
not commit reuse: never use `--reuse-message` or copy stale claims verbatim.

For each upstream release, before committing the squash:

1. Read the complete message of the preceding sibling with
   `git show -s --format=fuller <previous-tag>` and record its subject/body
   shape. Use the C-core and Python-interface templates independently; they do
   not have to share wording.
2. Draft the new message with the same subject prefix, section order, and
   bullet style. Update the version-specific API, correctness, safety,
   diagnostics, vendor, test, and documentation statements to match the
   staged diff exactly. Do not turn the body into a mechanical file list.
3. Show the drafted subject/body alongside a concise source-to-release change
   summary during preflight. Stop if the structure materially diverges or if
   any claim cannot be supported by the staged diff and executed checks.
4. Create the one release commit with a real multiline message, then verify
   `git log -1 --format=%B | sed -n 'l'`: no literal `\\n` escapes, no stale
   `.3` claims presented as `.4`, and no omitted release-specific safety or
   compatibility note.

The message template does not change ancestry: each release commit remains a
single direct child of the `1.0.0` bump, and each numbered tag remains a sibling
of the earlier tags.

## Required inputs and preflight

Before mutating anything, resolve and display:

- target suffix and complete tag, for example `1.0.0.4`;
- the C-core (`igraph`), Python interface (`python-igraph`), and
  `hedonic-game` repository paths;
- the source work branch in each upstream repository (for this workspace,
  examples are `codex/astra-overlap-contracts` and
  `codex/astra-membership-ownership`);
- an optional canonical local development-preservation branch for each
  upstream. If omitted, propose a new, non-conflicting name such as
  `dev/next-1.0.0.4` and ask before creating it; never repoint an existing
  `lucas-dev` branch automatically;
- the intended Hedonic base branch (`paper` or `main`), respecting the root
  repository's public/private workflow in `AGENTS.md`.
- the derived Hedonic package version `1.N.x` (`x` = 0 for a coordinated release), where `N` is exactly the suffix
  of the upstream `1.0.0.N` target. If the user requests a coordinated
  release, a different Hedonic version is a hard mismatch; only an explicit
  non-release/package-build request may leave the version unchanged;
- the publication mode (artifacts-only, publish `lucas-igraph`, publish
  `hedonic`, or both), target index, and explicit approval for any upload;
- whether the run may push candidate branches and later move public branches.
  Draft PR creation is not part of the default flow and requires a separate
  request. If push is not authorized, perform only read-only checks and local
  candidate preparation, then stop before network mutation.

Resolve the `1.0.0` bump commit in each upstream repository from the local
history and require exactly one unambiguous candidate. There is no assumption
that a literal `1.0.0` tag exists. Record the source tip, bump commit, current
submodule pointer, and all target refs before proceeding.

Stop before any write if a worktree (including a submodule worktree) is dirty,
the target tag or release branch already exists, the source branch is not a
descendant of its bump commit, the bump is ambiguous, or the requested paths
do not resolve to the expected repositories. Never use `reset --hard`,
`checkout --`, force updates, or broad cleanup to make preflight pass.

Use temporary `git worktree` checkouts for release branches so the user's
current checkouts and source branches remain untouched. Keep a manifest of
worktree paths and refs so cleanup is explicit and recoverable.

## Preserve the development history

The Codex source branches are the active development lines and may contain
several commits (four in the audited C-core branch and five in the Python
branch). Before the first squash, create the requested local-only
development-preservation branch at each source tip. This branch is a durable
pointer to the complete iterative history; it is not the release branch and is
not pushed unless the user asks for it. Keep the original Codex branch as well.

Do not assume an existing `lucas-dev` branch is current. Inspect its ancestry
first; in the audited clones those names were stale/divergent and did not
contain the `1.0.0.3` release. If a proposed preservation branch already
exists at a different commit, stop and ask rather than moving it. When fixes
are added after a failed candidate, advance the preservation branch only by a
verified fast-forward (or create a new explicitly named preservation branch).

## Consolidate the canonical `lucas-dev` line

Before creating a release candidate, integrate the supplied Codex source
branch into a canonical local `lucas-dev` line. This is a history-preservation
operation, separate from the public squash:

1. Snapshot the existing `lucas-dev` tip in a uniquely named local backup such
   as `lucas-dev-backup/<target>`. Never move or delete the existing ref before
   that backup is verified.
2. In a temporary worktree, compare the existing `lucas-dev` and supplied
   source histories with `merge-base`, `range-diff`, and patch-id/cherry
   checks. Identify which commits are already represented and which need to be
   transplanted; do not duplicate equivalent patches merely because their
   hashes differ.
3. Build a candidate linear branch, normally by rebasing or cherry-picking the
   missing commits in chronological order. A merge is acceptable only when the
   user explicitly wants merge topology; the default `lucas-dev` result has no
   merge commits after the `1.0.0` bump.
4. Verify that the candidate contains every requested Codex change, that its
   commit order is auditable, and that no source branch was rewritten. Show a
   source-to-candidate commit map before advancing `lucas-dev`.
5. Advance the local `lucas-dev` only after that map and conflict resolutions
   are accepted. Updating a remote `origin/lucas-dev` is separate and requires
   explicit authorization plus `--force-with-lease`; local backup refs and the
   original Codex branches remain untouched.

If the old `lucas-dev` and the supplied source are materially divergent and
the correct transplant set cannot be established mechanically, stop and ask
whether to preserve the old line or rebuild `lucas-dev` from the resolved
`1.0.0` bump. Never silently merge stale release snapshots or reset the branch.

## CI candidate gate

Use the existing GitHub Actions as the cross-platform authority. Both upstream
repositories have workflows triggered by `push` and `pull_request`; the Python
workflow checks out submodules recursively and builds wheels/sdist across its
platform matrix. A local build is useful for fast feedback but never replaces
the hosted checks.

Keep a multi-commit development branch for iteration. When external CI is
authorized, push that branch to the user's fork; the existing `push`-triggered
workflows run without opening a PR. Create a draft PR only if the user asks for
one. Iterate on the same development branch until its checks are green. Do not
squash or tag a failing candidate.

After the development branch is green, create the one-commit release candidate
from the bump commit and push it to a dedicated, untagged release branch. Run
Actions again on this exact squashed commit. If it fails, fix the development
branch, regenerate the candidate on the same untagged branch, and update it
with `--force-with-lease` only after verifying the remote tip. Do not create a
new branch for every failed attempt. Once a tag exists, the release commit and
tag are immutable: never force-update them.

If GitHub status cannot be verified (for example, no authenticated `gh` CLI
and no user-supplied green run links), stop before tagging and report the
candidate commit that still needs CI confirmation.

## Stage 1: C-core `igraph`

1. Consolidate the selected C-core source branch into the canonical local
   `lucas-dev` line as described above. Then create a new release worktree/branch from the resolved
   C-core `1.0.0` bump commit. Follow the repository's established branch naming convention
   (for example `lucas-1.0.0.4`) unless the user supplies another name.
2. Squash the complete delta from the selected source work branch onto that
   base (`merge --squash --no-commit` is preferred when the base is an
   ancestor). Resolve conflicts manually and stop if the resulting staged
   diff is not the intended release.
3. Inspect the staged diff, run the relevant C-core build/tests, and commit
   exactly once using the preceding-sibling message template described above.
   Do not create a chain of cleanup commits.
4. Before creating the local tag, verify both:
   `git rev-list --count <bump>..<release-commit> == 1` and the release
   commit's only parent is exactly `<bump>`.
5. Push the untagged release branch when authorized (without opening a PR by
   default) and wait for the full Actions matrix on this exact one-commit
   candidate. If it fails, revise the
   development branch and regenerate the candidate on the same branch as
   described in the CI gate; do not tag a failing build.
6. Create the new local tag using the existing tag style, then verify that the
   tag resolves to the validated release commit. Record its full hash for the
   Python stage. Push the tag only when explicitly authorized and only after
   the final candidate Actions are green.
7. After the tag is immutable, update the public `lucas` and `physica-a`
   branch pointers to this release commit if that is part of the requested
   release. Because sibling releases are not fast-forwards, use a verified
   `--force-with-lease` update; never use plain `--force`. Preserve the old
   `lucas-1.0.0.N` branch and tag.

## Stage 2: Python interface `python-igraph`

1. Consolidate the selected Python source branch into the canonical local
   `lucas-dev` line as described above. Then create a separate release worktree/branch from the Python
   interface's resolved `1.0.0` bump commit, never from the old `.3` tag.
2. Squash the complete Python delta from its selected source work branch onto
   that base. Preserve the source branch; do not reset it.
3. Check out the `vendor/source/igraph` submodule at the exact C-core release
   commit recorded in Stage 1 (for a Python-only release, the commit the
   `igraph 1.0.0.N` alias tag resolves to), then stage the submodule pointer. Verify the
   tree entry with `git ls-tree` and `git submodule status`; a pointer to a
   branch name, an older tag, or an unrelated descendant is a hard failure.
4. Verify the Python package version, build the extension/package, and run the
   relevant tests. Do not manufacture package or lock-file hashes.
5. Commit exactly once using the preceding-sibling message template described
   above, verify that its sole parent is the Python bump and that the distance
   from the bump is one. Keep this release branch untagged until hosted CI
   passes.
6. Push the Python candidate branch (when authorized, without opening a PR by
   default) and wait for its full Actions matrix on the exact squashed commit.
   If it fails, revise the
   development branch and regenerate the untagged candidate as described in
   the CI gate; do not tag a failing build.
7. Create and verify the matching local tag only after the hosted checks are
   green. Push the tag only with explicit authorization. Record the Python
   commit and package identity for Hedonic Game.
8. After the tag is immutable, update the public `lucas` and `physica-a`
   branch pointers with verified `--force-with-lease` updates when requested.

9. If `lucas-igraph` publication is requested, select a successful GitHub
   Actions run whose `headSha` is exactly the Python release commit. Use
   `gh run download` to retrieve its wheels/sdist, verify every artifact's
   package name and version, and follow
   [references/pypi-artifact-handoff.md](references/pypi-artifact-handoff.md)
   for the hidden-token `uv publish` handoff. Do not rebuild a different local
   artifact or upload before the run/commit match is proven.

Do not advance the Python tag until the C-core hash and the recorded vendor
pointer match exactly. If the package is not available for the next stage,
stop and report that publication or package availability is required rather
than fabricating a dependency resolution.

## Stage 3: Hedonic Game dependency update

Only after both upstream stages pass their gates, and the Python package is
actually available from the intended package index, create a Hedonic worktree
from the explicitly selected `paper` or `main` base. Follow `AGENTS.md`: keep
private manuscript material on `paper`, never merge `paper` wholesale into
`main`, and publish only reviewed public commits.

Update the `lucas-igraph` pin in `pyproject.toml` from the old release to the
new Python package version and set the Hedonic package version to the derived
`1.N.0`. Regenerate `uv.lock` with the package resolver once the package is
genuinely available; never hand-edit artifact URLs or hashes. Verify that the
package metadata says exactly `1.N.0` and that the lock metadata names the
requested Python dependency version, run focused tests and the normal project
checks, and make one dependency/version-update commit. If a Hedonic release
tag is part of the authorized publication, use the repository's established
tag style with the same `1.N.0` version (`v1.N.0`); never create a Hedonic `1.0.0.N` tag.

**Hedonic-only changes.** Hedonic is not proposed upstream, so its version is
the only one that may advance without an upstream release. When no upstream
sibling changed, do not invent an upstream `N`, do not create empty upstream
commits, and do not reuse a version. Bump the last component: `1.N.x` becomes
`1.N.(x+1)`, keeping the `lucas-igraph` pin unchanged, so `1.N.x` always
names the upstream `N` it was built against and users can pin `hedonic~=1.N.0`
for "this vendor, fixes only". The next coordinated release resets to
`1.(N+1).0`. Tag it `v1.N.x` (never a `1.0.0.N`-style tag). Hedonic-only
features also land in `x`. A breaking Hedonic API change is the one case that
would need `2.0.0`, which ends the `N` mapping; do not do that without an
explicit user decision. If the user does not want a release, leave the version
unchanged (non-release package build) instead.

## Stage 4: Hedonic PyPI handoff (optional, explicit)

Publishing the Hedonic package is a separate final gate, not an automatic
consequence of updating the dependency. The repository already contains
[`publish-pypi.yml`](../../.github/workflows/publish-pypi.yml), which builds
with `uv build` and publishes through the configured PyPI action on a matching
`v*` tag or manual dispatch. Reuse that workflow rather than duplicating a
second publishing implementation in this skill.

The workflow file currently carries a comment that publication is disabled and
requires a `PYPI_API_TOKEN`; verify the effective Actions configuration and
credentials before treating either trigger as publishable.

After Stage 3 succeeds:

1. Confirm the derived Hedonic package version `1.N.x`, repository base/branch,
   PyPI target, and availability of the required repository secret or trusted
   publisher configuration. Verify that `N` matches the upstream `1.0.0.N`
   target; do not publish a different Hedonic version in the coordinated run.
2. Run `uv build` and inspect the generated sdist/wheel metadata before any
   upload. Keep the build output separate from source changes and remove or
   retain it only according to the user's artifact policy.
3. If publication is authorized, use the existing workflow's documented
   trigger or the hidden-token `uv publish` procedure in
   [references/pypi-artifact-handoff.md](references/pypi-artifact-handoff.md)
   for the configured index.
   Do not create a production tag, dispatch a publishing workflow, or upload
   artifacts without that explicit authorization.
4. Verify the published version from the intended index, then report the
   package URL and immutable artifact hashes. If publication is not authorized
   or credentials/index configuration are missing, stop with a verified local
   build ready for the user instead.

## Stop conditions and final report

Stop immediately on a dirty checkout, conflict, failed build/test, ambiguous
ancestry, pre-existing target ref, vendor-pointer mismatch, unavailable
package, or any request that would require force-pushing or deleting history.
Leave successful earlier stages intact and report their recorded hashes; do
not silently roll them back.

At completion, report a compact manifest containing, for each upstream
repository: base bump hash, source branch, release branch, tag, release hash,
parent hash, and commit distance. Mark any alias tag explicitly (alias tag, target hash, the real release it
duplicates). Include the Python vendor pointer, the
upstream-to-Hedonic version mapping (`1.0.0.N` → `1.N.x`), and the Hedonic
dependency/lock result. Include the candidate and final CI status/run links
(or state that the user must verify them), and the remote branch update
status. Include the names/tips of the local development-preservation branches.
For the optional PyPI handoff, include the package version, build metadata,
workflow/command used, publication status, and index verification (or the
exact reason it remains unpublished).
Clearly distinguish local-only results from published artifacts and
list any remaining user decision (for example, whether to push tags, move
`lucas`/`physica-a`, or publish the Python package).
