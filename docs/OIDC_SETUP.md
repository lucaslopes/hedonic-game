# Package publishing

The public checkout must be built and tagged from a clean worktree. Manuscript
and evidence files are not part of the distribution.

## PyPI

`.github/workflows/publish-pypi.yml` publishes tags matching `v*` to PyPI. It
expects the repository secret `PYPI_API_TOKEN`. Configure that secret in the
GitHub repository before pushing a release tag.

Prepare a release locally with:

```bash
uv version 0.1.0 --no-sync
uv build --no-sources
uv run --with pytest pytest -q
git add pyproject.toml uv.lock
git commit -m "release: prepare 0.1.0"
git tag v0.1.0
```

The helper `scripts/release.sh` performs the same workflow for a patch, minor,
or major bump. It never pushes unless called with `--push`.

Review both files in `dist/` before pushing the public branch and tag.

## TestPyPI rehearsal

`.github/workflows/publish-pypi.yml.disabled` contains the opt-in TestPyPI
workflow. Enable it deliberately for a rehearsal and keep test tags separate
from production `v*` tags. Its trusted-publishing/OIDC configuration must be
configured in the target repository before use.

## Privacy checklist

Before publishing, verify that the public tree and built artifacts contain no
`docs/papers/`, `artifacts/evidence/`, LaTeX/BibTeX manuscript files, local
absolute paths, or generated result ledgers.
