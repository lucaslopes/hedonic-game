#!/usr/bin/env bash

# Prepare a release from a clean public checkout.
# Usage: ./scripts/release.sh [patch|minor|major] [--push]

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [patch|minor|major] [--push]" >&2
    exit 1
fi

VERSION_TYPE=""
DO_PUSH=false
for arg in "$@"; do
    case "$arg" in
        patch|minor|major)
            if [[ -n "$VERSION_TYPE" ]]; then
                echo "Error: specify only one version bump" >&2
                exit 1
            fi
            VERSION_TYPE="$arg"
            ;;
        --push|-p)
            DO_PUSH=true
            ;;
        *)
            echo "Error: unknown argument: $arg" >&2
            echo "Usage: $0 [patch|minor|major] [--push]" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$VERSION_TYPE" ]]; then
    echo "Error: version bump is required" >&2
    exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: release requires a clean worktree" >&2
    exit 1
fi

CURRENT_VERSION=$(uv version --short)
echo "Current version: $CURRENT_VERSION"
uv version --bump "$VERSION_TYPE" --no-sync
NEW_VERSION=$(uv version --short)
echo "New version: $NEW_VERSION"

uv build --no-sources
git add pyproject.toml uv.lock
git commit -m "release: prepare $NEW_VERSION"

TAG="v$NEW_VERSION"
git tag "$TAG"
echo "Created local tag: $TAG"

if [[ "$DO_PUSH" == true ]]; then
    BRANCH=$(git branch --show-current)
    git push origin "$BRANCH" "$TAG"
    echo "Pushed $BRANCH and $TAG."
else
    echo "Nothing was pushed. Review the artifacts, then push $TAG from the public branch when ready."
fi
