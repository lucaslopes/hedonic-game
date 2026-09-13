# Publishing Hedonic

Hedonic uses independent three-component versions. Release 0.1.1 depends on
`lucas-igraph==1.0.0.4`. The public `main` branch is the publication boundary;
never push the private `paper` branch or merge its history into `main`.

1. Prepare the version, dependency pin, README and changelog on `main`.
   Regenerate `uv.lock` with `uv lock`; preserve historical protocol locks.
2. Run `uv sync --locked --extra experiments`, the focused release gate
   (`uv run --with pytest pytest -q tests/test_release_helper.py
   tests/test_overlapping_and_experiments.py::TestCommunityHedonic`), source
   compilation, and `uv build --no-sources`. The research protocol tests are
   intentionally pinned to the historical `lucas-igraph 1.0.0.3` environment;
   they remain unchanged and are not a release gate for the `.4` dependency
   update.
3. Review the package contents and commit, then push `main`. The
   `Publish to PyPI` workflow tests and saves wheel/sdist artifacts without
   publishing. Wait for success on that exact commit.
4. Run `./scripts/release.sh preflight-pypi` from the release checkout. It
   selects a successful artifact-producing run for the exact commit and checks
   package identity and existing PyPI hashes.
5. Create the new immutable `vVERSION` tag on the validated commit and push
   that tag. The same workflow downloads the successful main run's artifacts
   and publishes via `scripts/release.sh` using `PYPI_API_TOKEN`. It does not
   rebuild or overwrite an existing tag.
6. Verify the PyPI version and artifact hashes. On a partial upload, inspect
   the index and run preflight again before deciding whether to retry.

For an authorized manual upload, run `./scripts/release.sh publish-pypi --publish`
from the tagged release checkout. Without a configured `UV_PUBLISH_TOKEN`, the
script prompts silently on an interactive terminal only after validation.
Never paste a token into command arguments, source files, or logs.

The helper rejects Pyodide wheels and currently excludes PyPy wheels because
of the upstream metadata compatibility issue. Saved artifact directories are
compared with a fresh download before use. Existing filenames are skipped only
when their SHA-256 digest matches. No upload is retried automatically.

The historical `patch|minor|major` helper updates the version and lock, builds,
commits, and tags locally. For hosted releases, prefer the staged procedure
above so tests on the exact public commit pass before creating the tag.
