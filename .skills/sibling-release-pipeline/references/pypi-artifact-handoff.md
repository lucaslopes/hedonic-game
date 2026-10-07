# PyPI artifact handoff

Use this reference only after a release candidate has passed its hosted
Actions checks and the user explicitly authorizes an upload. It covers the
artifact-first path: download the exact successful GitHub Actions outputs with
`gh`, verify their commit/version identity, then upload with `uv` without
placing a token in shell history.

## Download the exact Actions artifacts

Resolve the repository, workflow, branch/tag, and successful run interactively;
never guess a run ID. Verify the run's `headSha` equals the immutable release
commit before downloading:

```bash
gh run view RUN_ID --repo OWNER/REPO --json headSha,conclusion,workflowName,artifacts
ARTIFACT_DIR="$(mktemp -d)"
gh run download RUN_ID --repo OWNER/REPO --dir "$ARTIFACT_DIR"
```

Collect only the expected `.whl` and `.tar.gz` files. Inspect their metadata
with an installed packaging tool (or the project's normal check), and reject
any artifact whose distribution name or version differs from the intended
release. Keep the downloaded directory outside the repository and do not add
it to Git.

## Upload without exposing the token

Use a terminal command that prompts silently; do not put the literal token in
the command, an environment file, a notebook, or a commit. The command text
below is safe to paste because the token is entered interactively and removed
from the environment afterward:

```bash
printf 'PyPI token (hidden): '
read -r -s UV_PUBLISH_TOKEN
printf '\n'
export UV_PUBLISH_TOKEN
uv publish "$ARTIFACT_DIR"/*.whl "$ARTIFACT_DIR"/*.tar.gz
publish_status=$?
unset UV_PUBLISH_TOKEN
exit "$publish_status"
```

Use the explicit index option or `UV_PUBLISH_URL` when targeting TestPyPI or a
private index. Never pass a literal token as a `--token` argument. If the
upload fails, keep the artifacts for diagnosis, report the failure, and do not
blindly retry a partially successful upload.

## Verify and hand off

After a successful upload, query the intended index for the exact distribution
and version and record the published files/hashes. If the package is already
present, compare hashes and stop rather than overwriting it. A PyPI upload is
irreversible; tags and release commits must already be immutable before this
step.
