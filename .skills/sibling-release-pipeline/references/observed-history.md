# Observed sibling-release history

This is a local audit of the three existing `1.0.0.N` releases. Re-run the
ancestry checks before using the skill; these values are evidence of the
pattern, not immutable configuration.

## C core (`igraph`)

- Bump commit: `b9b573902ccbe393a78252ab5e94c7876ed92597`,
  `chore: bumped version to 1.0.0`.
- `1.0.0.1` → `3ebcde3df74b1ca33f51a6ab3cb79b70f85187a9`,
  parent exactly the bump.
- `1.0.0.2` → `7793dac95cbf6c874bfdc1c2f6b4801e5cf5a4b7`,
  parent exactly the bump.
- `1.0.0.3` → `99d6fd99a8bd9fc914f765839a2ec2955cd67d16`,
  parent exactly the bump.

Thus each tag is one commit after the bump and the three tags are siblings.

The observed branch roles are also consistent across the two repositories:
`lucas-1.0.0.N` names the release snapshot, while `lucas` and `physica-a`
point at the current release snapshot. Development branches may move ahead of
the release (the audited local Codex branches were four commits ahead in the C
core and five commits ahead in Python). Because the next sibling has the same
bump parent rather than the previous release as its parent, moving `lucas` or
`physica-a` to the new release is a non-fast-forward pointer update; preserve
the old numbered branch/tag and use only a verified `--force-with-lease` when
that move is explicitly authorized.

The existing `lucas-dev` refs are not a dependable substitute for a current
development pointer: the audited C and Python clones had divergent/stale
`lucas-dev` tips that did not descend from `1.0.0.3`. A release run should
snapshot those refs before integrating the supplied Codex branches, then show
a source-to-candidate commit map before advancing the canonical local
`lucas-dev`. Do not repoint or delete the old ref without a verified backup.

## Python interface (`python-igraph`)

- Bump commit: `b16f27618674dd1913007a52855b76075802cbf9`,
  `chore: updated changelog, bumped version to 1.0.0`.
- `1.0.0.1` → `c4b9a965a88e2e61d76f0de2ab0e7726592a26e9`, with the C-core
  submodule at `3ebcde3df74b1ca33f51a6ab3cb79b70f85187a9`.
- `1.0.0.2` → `e441a21c3663062d9ebe6b98451c826325f886ad7`, with the C-core
  submodule at `7793dac95cbf6c874bfdc1c2f6b4801e5cf5a4b7`.
- `1.0.0.3` → `66db4a6932adca38bad7fc319a9a20a0ced46a91`, with the C-core
  submodule at `99d6fd99a8bd9fc914f765839a2ec2955cd67d16`.

Each Python tag is likewise one commit after its bump, and each vendor
pointer matches the corresponding C-core tag exactly.

## CI behavior observed locally

- `igraph/.github/workflows/build-cmake.yml` runs on every branch `push` and
  `pull_request`, with a Windows architecture/shared-library matrix and CMake
  tests. Other C-core validation workflows also use push/PR triggers.
- `python-igraph/.github/workflows/build.yml` runs on `push` and
  `pull_request`; its checkout uses `submodules: true` and `fetch-depth: 0`,
  then builds/tests Linux, macOS, WebAssembly, and Windows wheels plus an
  sdist/sanitizer path.

Therefore a public fork branch is enough to exercise the hosted matrix before
a release tag; a draft PR is optional and is not part of the default flow.
The tag should be created only after the exact one-commit candidate has a
green run.

## Hedonic Game handoff

The current project pin is `lucas-igraph==1.0.0.3` in `pyproject.toml`, with
the same package identity in `uv.lock`. The next handoff must update both via
the resolver after the Python package is available. Hedonic's own version is
separate (`0.1.0` in the audited checkout), so do not infer an upstream
`1.0.0.N` tag for this repository.

The repository also contains `.github/workflows/publish-pypi.yml`, which runs
`uv build` and the PyPI publishing action for `v*` tags or manual dispatch.
Its comments indicate that publication requires enabling/configuring the
workflow and a `PYPI_API_TOKEN`; verify that configuration before triggering
any upload. For upstream `lucas-igraph` artifacts, use the successful
`python-igraph` Actions run whose `headSha` matches the release commit, then
download with `gh run download` and perform the explicit hidden-token `uv
publish` handoff described by the skill.
