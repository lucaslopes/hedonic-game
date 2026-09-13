#!/usr/bin/env bash

# Safe release helper for hedonic and related Python distributions.
#
# The version-bump path keeps the historical local workflow. The PyPI path is
# deliberately artifact-first: it verifies an exact successful GitHub Actions
# run, downloads its distributions, rejects unsupported/non-PyPI wheels,
# avoids files already present on the index, and asks for a token only at the
# final upload boundary.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

readonly DEFAULT_PUBLISH_URL="https://upload.pypi.org/legacy/"
readonly DEFAULT_INDEX_URL="https://pypi.org/pypi"

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

warn() {
    printf 'Warning: %s\n' "$*" >&2
}

need_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

index_has_file() {
    local filename="$1"
    local manifest="$2"
    awk -F '\t' -v target="$filename" '$1 == target { found = 1 } END { exit(found ? 0 : 1) }' "$manifest"
}

verify_index_hash() {
    python - "$1" "$2" <<'PYHASH'
import hashlib
import pathlib
import sys
artifact = pathlib.Path(sys.argv[1])
manifest = dict(line.rstrip('\n').split('\t') for line in pathlib.Path(sys.argv[2]).read_text().splitlines())
if hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest.get(artifact.name):
    raise SystemExit('index SHA-256 mismatch: ' + artifact.name)
PYHASH
}

usage() {
    cat <<'EOF'
Usage:
  ./scripts/release.sh patch|minor|major [--push]
  ./scripts/release.sh preflight-pypi [options]
  ./scripts/release.sh publish-pypi [options] --publish
  ./scripts/release.sh publish-pypi --publish

Version bump:
  patch|minor|major  Update pyproject.toml, build, commit, and tag locally.
  --push             Push the current branch and its new tag to origin.

Artifact-first PyPI publication:
  --repo REPO        GitHub repository (default: origin's GitHub repository).
  --run-id ID        Exact Actions run (default: auto-select by commit/artifacts).
  --commit SHA       Expected head SHA (default: current checkout HEAD).
  --workflow NAME    Optional workflow-name filter during auto-selection.
  --version VERSION  Distribution version (default: pyproject.toml).
  --distribution N   Distribution name (default: pyproject.toml).
  --artifacts-dir D  Verify saved artifacts against a fresh exact-run download.
  --publish-url URL  Upload endpoint (default: PyPI production endpoint).
  --index-url URL    JSON index base used to check existing files.
  --publish          Perform the upload. Without it, only preflight runs.

Examples:
  # From the exact release checkout, auto-discover the matching green run:
  ./scripts/release.sh preflight-pypi \
    --repo lucaslopes/python-igraph \
    --commit 9fbd3547cdb87b67d57954bc6a5ba901fbabe391 \
    --version 1.0.0.4 --distribution lucas-igraph

  # For a Hedonic release checkout, repo/version/distribution/commit are inferred:
  ./scripts/release.sh publish-pypi --publish

  # Explicit values remain supported when publishing another repository:
  ./scripts/release.sh preflight-pypi \
    --repo lucaslopes/python-igraph \
    --run-id 34723765016 \
    --commit 9fbd3547cdb87b67d57954bc6a5ba901fbabe391 \
    --version 1.0.0.4 --distribution lucas-igraph

  ./scripts/release.sh publish-pypi \
    --repo lucaslopes/python-igraph \
    --run-id 34723765016 \
    --commit 9fbd3547cdb87b67d57954bc6a5ba901fbabe391 \
    --version 1.0.0.4 --distribution lucas-igraph --publish

The PyPI path never opens a pull request, never rebuilds a downloaded
artifact, skips Pyodide and PyPy wheels that PyPI rejects, and does not retry
an upload that may have partially succeeded. The token is read silently from
/dev/tty only after all preflight gates pass, unless Actions supplies
UV_PUBLISH_TOKEN through its configured secret.
EOF
}

pyproject_field() {
    local field="$1"
    sed -n "s/^[[:space:]]*$field[[:space:]]*=[[:space:]]*\"\([^\"]*\)\".*/\1/p" pyproject.toml | head -n 1
}

normalise_distribution_name() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[_.]/-/g'
}

current_git_branch() {
    git symbolic-ref --quiet --short HEAD 2>/dev/null || true
}

release_bump() {
    local version_type="$1"
    local do_push=false
    local current_version new_version tag branch
    local major minor patch

    shift
    while (($#)); do
        case "$1" in
            -p|--push) do_push=true ;;
            *) die "unknown version-bump option: $1" ;;
        esac
        shift
    done

    need_command git
    need_command uv

    [[ "$(current_git_branch)" == main ]] || die "version bumps must run on public main"
    [[ -f pyproject.toml ]] || die "pyproject.toml not found"
    [[ -z "$(git status --porcelain)" ]] || die "worktree is dirty; clean it before a version bump"

    current_version="$(pyproject_field version)"
    [[ -n "$current_version" ]] || die "could not read version from pyproject.toml"
    IFS='.' read -r major minor patch <<< "$current_version"
    [[ "$major" =~ ^[0-9]+$ && "$minor" =~ ^[0-9]+$ && "$patch" =~ ^[0-9]+$ ]] || \
        die "version must have three numeric components: $current_version"

    case "$version_type" in
        patch) new_version="$major.$minor.$((patch + 1))" ;;
        minor) new_version="$major.$((minor + 1)).0" ;;
        major) new_version="$((major + 1)).0.0" ;;
        *) die "version type must be patch, minor, or major" ;;
    esac

    tag="v$new_version"
    git rev-parse "$tag" >/dev/null 2>&1 && die "tag already exists: $tag"

    printf 'Bumping %s -> %s\n' "$current_version" "$new_version"
    if [[ "$OSTYPE" == darwin* ]]; then
        sed -i '' -E "s/^version[[:space:]]*=[[:space:]]*\"[^\"]+\"/version = \"$new_version\"/" pyproject.toml
    else
        sed -i -E "s/^version[[:space:]]*=[[:space:]]*\"[^\"]+\"/version = \"$new_version\"/" pyproject.toml
    fi

    printf 'Building distributions with uv...\n'
    uv lock
    uv build --no-sources

    git add pyproject.toml uv.lock
    git commit -m "Bump version to $new_version"
    git tag "$tag"
    printf 'Created %s\n' "$tag"

    if [[ "$do_push" == true ]]; then
        branch="$(current_git_branch)"
        [[ -n "$branch" ]] || die "detached HEAD; cannot infer branch for --push"
        printf 'Pushing branch %s and tag %s to origin...\n' "$branch" "$tag"
        git push origin "$branch"
        git push origin "$tag"
    else
        branch="$(current_git_branch)"
        printf 'Nothing was pushed. Review and push %s and %s when ready.\n' "$branch" "$tag"
    fi
}

# Print a tab-separated "Name<TAB>Version" pair from a wheel or source
# distribution using only Python's standard library. This catches a mislabeled
# artifact before uv contacts the upload endpoint.
distribution_metadata() {
    local artifact="$1"
    python - "$artifact" <<'PY'
import sys
from email.parser import Parser
import tarfile
import zipfile

path = sys.argv[1]
if path.endswith('.whl'):
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist()
                 if name.endswith('.dist-info/METADATA')]
        if len(names) != 1:
            raise SystemExit(f"expected one wheel METADATA file, found {len(names)}")
        text = archive.read(names[0]).decode('utf-8')
else:
    with tarfile.open(path, 'r:gz') as archive:
        names = [member for member in archive.getmembers()
                 if member.name == 'PKG-INFO' or member.name.endswith('/PKG-INFO')]
        # Setuptools may include both the public root PKG-INFO and a generated
        # src/*.egg-info/PKG-INFO copy. The root manifest is authoritative.
        root_names = [member for member in names
                      if '/.egg-info/' not in member.name
                      and '/src/' not in member.name]
        if len(root_names) == 1:
            manifest = root_names[0]
        elif len(names) == 1:
            manifest = names[0]
        else:
            raise SystemExit(f"could not identify one authoritative sdist PKG-INFO file (found {len(names)})")
        stream = archive.extractfile(manifest)
        if stream is None:
            raise SystemExit('could not read sdist PKG-INFO')
        text = stream.read().decode('utf-8')

fields = Parser().parsestr(text, headersonly=True)
if len(fields.get_all('Name', [])) != 1 or len(fields.get_all('Version', [])) != 1:
    raise SystemExit('artifact must have exactly one Name and Version header')
print(fields['Name'] + '\t' + fields['Version'])
PY
}

# Query the JSON API for a single version. A 404 means the version has no
# files yet; all other HTTP/network failures are hard stops.
index_manifest() {
    local url="$1"
    python - "$url" <<'PY'
import json
import sys
import urllib.error
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
except urllib.error.HTTPError as exc:
    if exc.code == 404:
        raise SystemExit(0)
    raise SystemExit(f"index query failed with HTTP {exc.code}: {exc.reason}")
except urllib.error.URLError as exc:
    raise SystemExit(f"index query failed: {exc.reason}")

for item in payload.get('urls', []):
    filename = item.get('filename', '')
    digest = (item.get('digests') or {}).get('sha256', '')
    if filename:
        print(filename + '\t' + digest)
PY
}

publish_pypi() {
    local do_publish=false
    local repo="${PYPI_RELEASE_REPO:-}"
    local run_id="${PYPI_RELEASE_RUN_ID:-}"
    local expected_sha="${PYPI_RELEASE_COMMIT:-}"
    local workflow_filter="${PYPI_RELEASE_WORKFLOW:-}"
    local version="${PYPI_RELEASE_VERSION:-}"
    local distribution="${PYPI_RELEASE_DISTRIBUTION:-}"
    local artifacts_dir="${PYPI_RELEASE_ARTIFACTS_DIR:-}"
    local publish_url="${UV_PUBLISH_URL:-$DEFAULT_PUBLISH_URL}"
    local index_url="${PYPI_INDEX_URL:-}"
    local run_view run_head run_conclusion run_workflow run_url
    local work_dir source_dir candidate_dir existing_file index_version_url
    local repo_was_inferred=false
    local file base lower metadata metadata_name metadata_version
    local valid_count=0 new_count=0
    local -a candidate_files=()
    local -a new_files=()

    while (($#)); do
        case "$1" in
            --repo) shift; (($#)) || die "--repo requires a value"; repo="$1" ;;
            --run-id) shift; (($#)) || die "--run-id requires a value"; run_id="$1" ;;
            --commit) shift; (($#)) || die "--commit requires a value"; expected_sha="$1" ;;
            --workflow) shift; (($#)) || die "--workflow requires a value"; workflow_filter="$1" ;;
            --version) shift; (($#)) || die "--version requires a value"; version="$1" ;;
            --distribution) shift; (($#)) || die "--distribution requires a value"; distribution="$1" ;;
            --artifacts-dir) shift; (($#)) || die "--artifacts-dir requires a value"; artifacts_dir="$1" ;;
            --publish-url) shift; (($#)) || die "--publish-url requires a value"; publish_url="$1" ;;
            --index-url) shift; (($#)) || die "--index-url requires a value"; index_url="$1" ;;
            --publish) do_publish=true ;;
            --preflight) do_publish=false ;;
            -h|--help) usage; return 0 ;;
            *) die "unknown PyPI option: $1" ;;
        esac
        shift
    done

    need_command git
    need_command gh
    need_command python
    need_command uv
    need_command find
    need_command awk
    need_command sed

    if [[ -z "$repo" ]]; then
        local origin_url
        origin_url="$(git remote get-url origin 2>/dev/null || true)"
        case "$origin_url" in
            git@github.com:*) repo="${origin_url#git@github.com:}" ;;
            https://github.com/*) repo="${origin_url#https://github.com/}" ;;
            ssh://git@github.com/*) repo="${origin_url#ssh://git@github.com/}" ;;
            *) die "could not infer a GitHub repository from origin; pass --repo OWNER/REPOSITORY" ;;
        esac
        repo="${repo%.git}"
        repo_was_inferred=true
    fi

    [[ -n "$repo" ]] || die "--repo is required"
    [[ "$repo" == */* && "$repo" != */*/* ]] || die "--repo must look like OWNER/REPOSITORY"

    if [[ -z "$version" ]]; then
        version="$(pyproject_field version)"
    fi
    if [[ -z "$distribution" ]]; then
        distribution="$(pyproject_field name)"
    fi
    [[ -n "$version" ]] || die "--version is required (or define version in pyproject.toml)"
    [[ -n "$distribution" ]] || die "--distribution is required (or define name in pyproject.toml)"

    if [[ -z "$expected_sha" ]]; then
        [[ "$repo_was_inferred" == true ]] || \
            die "when --repo is explicit, pass --commit for independent exact run verification"
        expected_sha="$(git rev-parse HEAD)" || die "could not determine current checkout HEAD"
    fi
    if [[ -n "$artifacts_dir" && -z "$run_id" ]]; then
        die "--artifacts-dir requires an explicit --run-id so its provenance remains auditable"
    fi
    if [[ -n "$expected_sha" ]]; then
        [[ "$expected_sha" =~ ^[0-9a-fA-F]{40}$ ]] || die "--commit must be a full 40-character SHA"
    fi
    if [[ -n "$run_id" ]]; then
        [[ "$run_id" =~ ^[0-9]+$ ]] || die "--run-id must be numeric"
        [[ "$run_id" != 123456789 ]] || \
            die "123456789 is an example placeholder; omit --run-id for automatic selection"
    fi
    if [[ -n "$expected_sha" ]]; then
        [[ "$expected_sha" != abcdef0123456789abcdef0123456789abcdef01 ]] || \
            die "the supplied commit SHA is an example placeholder; omit --commit or provide the real SHA"
    fi

    if [[ -z "$index_url" ]]; then
        case "$publish_url" in
            https://upload.pypi.org/legacy/*) index_url="$DEFAULT_INDEX_URL" ;;
            https://test.pypi.org/legacy/*) index_url="https://test.pypi.org/pypi" ;;
            *) die "--index-url is required for a nonstandard publish URL" ;;
        esac
    fi
    index_version_url="${index_url%/}/$distribution/$version/json"

    if [[ -z "$run_id" ]]; then
        local runs_json runs_file candidates_file artifacts_json artifact_candidates_file candidate_id artifact_has_payload
        candidates_file="$(mktemp -t hedonic-release-runs.XXXXXX)"
        runs_file="$(mktemp -t hedonic-release-runs-json.XXXXXX)"
        artifact_candidates_file="$(mktemp -t hedonic-release-artifact-runs.XXXXXX)"
        printf 'Finding a successful Actions run for commit %s in %s...\n' "$expected_sha" "$repo"
        runs_json="$(gh api "repos/$repo/actions/runs?status=success&head_sha=$expected_sha&per_page=100")" || \
            die "could not list GitHub Actions runs"
        printf '%s' "$runs_json" > "$runs_file"
        python - "$expected_sha" "$workflow_filter" "$runs_file" > "$candidates_file" <<'PY'
import json
import sys

expected_sha = sys.argv[1].lower()
workflow_filter = sys.argv[2].lower()
with open(sys.argv[3], encoding="utf-8") as stream:
    payload = json.load(stream)
runs = []
for run in payload.get("workflow_runs", []):
    if run.get("head_sha", "").lower() != expected_sha:
        continue
    if run.get("conclusion") != "success":
        continue
    name = run.get("name", "")
    if workflow_filter and workflow_filter not in name.lower():
        continue
    runs.append(run)

for run in sorted(runs, key=lambda item: item.get("created_at", ""), reverse=True):
    print("\t".join(str(run.get(key, "")) for key in ("id", "created_at", "name", "html_url")))
PY
        [[ -s "$candidates_file" ]] || {
            warn "no successful Actions run matches commit $expected_sha"
            warn "push the exact release commit and wait for its checks, or pass --run-id explicitly"
            die "could not auto-select an Actions run"
        }

        while IFS=$'\t' read -r candidate_id _; do
            artifacts_json="$(gh api "repos/$repo/actions/runs/$candidate_id/artifacts?per_page=100")" || \
                die "could not inspect artifacts for Actions run $candidate_id"
            artifact_has_payload="$(python -c 'import json,sys; p=json.load(sys.stdin); print(any(not a.get("expired", False) and a.get("size_in_bytes", 0) > 0 for a in p.get("artifacts", [])))' <<< "$artifacts_json")"
            if [[ "$artifact_has_payload" == true || "$artifact_has_payload" == True ]]; then
                printf '%s\n' "$candidate_id" >> "$artifact_candidates_file"
            fi
        done < "$candidates_file"

        [[ -s "$artifact_candidates_file" ]] || {
            warn "matching successful runs exist, but none has downloadable, non-expired artifacts"
            die "could not auto-select an artifact-producing Actions run"
        }
        run_id="$(sed -n '1p' "$artifact_candidates_file")"
        if [[ "$(wc -l < "$artifact_candidates_file" | tr -d ' ')" -gt 1 ]]; then
            warn "multiple artifact-producing runs matched; selecting the newest successful run $run_id"
        fi
    fi

    printf 'Checking GitHub Actions run %s in %s...\n' "$run_id" "$repo"
    run_view="$(gh run view "$run_id" --repo "$repo" --json headSha,conclusion,workflowName,url)" || \
        die "could not inspect GitHub Actions run $run_id"

    run_head="$(printf '%s' "$run_view" | python -c 'import json,sys; print(json.load(sys.stdin)["headSha"])')" || \
        die "gh returned invalid run JSON"
    run_conclusion="$(printf '%s' "$run_view" | python -c 'import json,sys; print(json.load(sys.stdin)["conclusion"])')" || \
        die "gh returned invalid run JSON"
    run_workflow="$(printf '%s' "$run_view" | python -c 'import json,sys; print(json.load(sys.stdin)["workflowName"])')" || \
        die "gh returned invalid run JSON"
    run_url="$(printf '%s' "$run_view" | python -c 'import json,sys; print(json.load(sys.stdin)["url"])')" || \
        die "gh returned invalid run JSON"

    if [[ -z "$expected_sha" ]]; then
        expected_sha="$run_head"
    fi
    [[ "$run_head" == "$expected_sha" ]] || die "run head SHA $run_head does not match expected $expected_sha"
    [[ "$run_conclusion" == success ]] || die "Actions run conclusion is $run_conclusion, not success"
    printf '  workflow: %s\n  commit:   %s\n  run:      %s\n' "$run_workflow" "$run_head" "$run_url"

    source_dir="$(mktemp -d -t hedonic-release-artifacts.XXXXXX)"
    printf 'Downloading artifacts into %s...\n' "$source_dir"
    gh run download "$run_id" --repo "$repo" --dir "$source_dir" || {
        warn "artifact directory preserved for diagnosis: $source_dir"
        die "could not download Actions artifacts"
    }
    if [[ -n "$artifacts_dir" ]]; then
        [[ -d "$artifacts_dir" ]] || die "artifact directory does not exist: $artifacts_dir"
        # A directory/run ID association alone is not provenance. Compare bytes
        # against a fresh download, then use the authoritative downloaded files.
        python - "$artifacts_dir" "$source_dir" <<'PYVERIFY'
import hashlib
import pathlib
import sys

def inventory(root):
    result = {}
    for path in pathlib.Path(root).rglob('*'):
        if path.is_file() and (path.name.endswith('.whl') or path.name.endswith('.tar.gz')):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.name in result and result[path.name] != digest:
                raise SystemExit('conflicting duplicate artifact: ' + path.name)
            result[path.name] = digest
    return result
if inventory(sys.argv[1]) != inventory(sys.argv[2]):
    raise SystemExit('supplied artifacts do not match the exact Actions run')
PYVERIFY
    fi

    work_dir="$(mktemp -d -t hedonic-release-pypi.XXXXXX)"
    candidate_dir="$work_dir/validated"
    existing_file="$work_dir/index-before.tsv"
    mkdir -p "$candidate_dir"

    printf 'Inspecting distributions for %s %s...\n' "$distribution" "$version"
    while IFS= read -r -d '' file; do
        base="$(basename "$file")"
        lower="$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')"

        if [[ "$lower" == *pyodide* ]]; then
            printf '  SKIP Pyodide wheel (PyPI rejects this platform): %s\n' "$base"
            continue
        fi
        if [[ "$base" == *-pp[0-9]*-* ]]; then
            printf '  SKIP PyPy wheel (known incompatible metadata): %s\n' "$base"
            continue
        fi

        metadata="$(distribution_metadata "$file")" || die "invalid package metadata: $base"
        IFS=$'\t' read -r metadata_name metadata_version <<< "$metadata"
        [[ "$(normalise_distribution_name "$metadata_name")" == "$(normalise_distribution_name "$distribution")" ]] || \
            die "$base contains distribution $metadata_name, expected $distribution"
        [[ "$metadata_version" == "$version" ]] || \
            die "$base contains version $metadata_version, expected $version"

        if [[ -e "$candidate_dir/$base" ]]; then
            cmp -s "$file" "$candidate_dir/$base" || die "conflicting duplicate artifact: $base"
            continue
        fi
        cp -p "$file" "$candidate_dir/$base"
        candidate_files+=("$candidate_dir/$base")
        valid_count=$((valid_count + 1))
    done < <(find "$source_dir" -type f \( -name '*.whl' -o -name '*.tar.gz' \) -print0)

    ((valid_count > 0)) || die "no valid non-Pyodide/non-PyPy distributions were found"

    printf 'Checking existing files at %s...\n' "$index_version_url"
    index_manifest "$index_version_url" > "$existing_file" || {
        warn "artifact directory preserved: $source_dir"
        die "could not query the package index; no upload was attempted"
    }

    printf 'Files already present on the index:\n'
    if [[ -s "$existing_file" ]]; then
        sed 's/^/  /' "$existing_file"
    else
        printf '  (none)\n'
    fi

    for file in "${candidate_files[@]}"; do
        base="$(basename "$file")"
        if index_has_file "$base" "$existing_file"; then
            verify_index_hash "$file" "$existing_file"
            printf '  SKIP already published: %s\n' "$base"
        else
            new_files+=("$file")
            new_count=$((new_count + 1))
            printf '  READY to publish: %s\n' "$base"
        fi
    done

    printf 'Artifacts and diagnostics are preserved in:\n  source: %s\n  work:   %s\n' "$source_dir" "$work_dir"
    ((new_count > 0)) || {
        printf 'No new files remain. No token was requested and no upload was attempted.\n'
        return 0
    }

    if [[ "$do_publish" != true ]]; then
        printf 'Preflight complete. Re-run the same command with --publish to upload.\n'
        return 0
    fi

    if [[ "$distribution" == hedonic ]]; then
        [[ "$(git rev-parse HEAD)" == "$expected_sha" ]] || die "Hedonic publication requires the release checkout"
        [[ "$(git rev-parse "refs/tags/v$version^{commit}")" == "$expected_sha" ]] || die "version tag must match the release commit"
        git merge-base --is-ancestor "$expected_sha" origin/main || die "release commit is not on public origin/main"
    fi
    printf 'All gates passed. The next step uploads permanently to %s.\n' "$publish_url"
    if (
        # Actions supplies its configured secret only to this final upload step.
        # Interactive users are prompted only when no token is already supplied.
        if [[ -z "${UV_PUBLISH_TOKEN:-}" ]]; then
            [[ -t 0 ]] || die "no publishing credential; run interactively for hidden token input"
            printf 'PyPI token (hidden; never shown or stored): ' > /dev/tty
            IFS= read -r -s UV_PUBLISH_TOKEN < /dev/tty
            printf '\n' > /dev/tty
        fi
        [[ -n "${UV_PUBLISH_TOKEN:-}" ]] || die "empty PyPI token"
        export UV_PUBLISH_TOKEN
        UV_PUBLISH_URL="$publish_url" uv publish "${new_files[@]}"
    ); then
        printf 'Upload command completed. Verifying the index...\n'
    else
        warn "upload failed and may have been partial; no automatic retry was attempted"
        if index_manifest "$index_version_url" > "$work_dir/index-after-failure.tsv" 2>/dev/null; then
            warn "current index state (re-run preflight before any retry):"
            sed 's/^/  /' "$work_dir/index-after-failure.tsv" >&2
        fi
        warn "re-run preflight to identify only files still missing"
        return 1
    fi

    local after_file="$work_dir/index-after.tsv"
    index_manifest "$index_version_url" > "$after_file" || \
        die "upload may have succeeded, but post-upload index verification failed"

    for file in "${new_files[@]}"; do
        base="$(basename "$file")"
        index_has_file "$base" "$after_file" || \
            die "post-upload verification did not find $base"
        verify_index_hash "$file" "$after_file"
    done

    printf 'Published files and SHA-256 digests:\n'
    sed 's/^/  /' "$after_file"
    printf 'PyPI publication verified for %s %s.\n' "$distribution" "$version"
}

main() {
    local command="${1:-}"
    case "$command" in
        patch|minor|major)
            shift
            release_bump "$command" "$@"
            ;;
        publish-pypi|publish)
            shift
            publish_pypi "$@"
            ;;
        preflight-pypi|preflight)
            shift
            publish_pypi --preflight "$@"
            ;;
        -h|--help|help)
            usage
            ;;
        '')
            usage >&2
            exit 1
            ;;
        *)
            die "unknown command: $command (use --help)"
            ;;
    esac
}

main "$@"
