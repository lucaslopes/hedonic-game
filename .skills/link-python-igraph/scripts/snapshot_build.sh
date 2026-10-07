#!/usr/bin/env bash
# Build an isolated, distinctly versioned snapshot of one igraph C commit.
#
# Snapshots let you attribute a behaviour change to a commit: load the
# snapshot's library with DYLD_LIBRARY_PATH (macOS) or LD_LIBRARY_PATH
# (Linux) under an unchanged python-igraph build, and compare results.
#
#   snapshot_build.sh <igraph-worktree> <commit> <name> [dev-root]
#
#   <igraph-worktree>  any checkout of the igraph C repository
#   <commit>           commit-ish to snapshot (resolved and recorded)
#   <name>             short, path-safe name, e.g. opus-preC2
#   [dev-root]         defaults to the worktree's parent directory
#
# It creates, and refuses to reuse:
#   <dev-root>/_scratch/igraph-<name>   sources from `git archive <commit>`
#   ~/dev/igraph_<name>                 CMake build directory (Release)
#   <dev-root>/_prefix/igraph-<name>    install prefix
#
# The snapshot reports igraph_version() = "<base>-<name>" (base from the
# worktree's IGRAPH_VERSION file, or $SNAPSHOT_BASE_VERSION), so every result
# record that stores igraph._igraph.__igraph_version__ proves which library
# produced it. Only the igraph library target is built; nothing is installed
# outside the prefix.
set -euo pipefail

if [ $# -lt 3 ]; then
  sed -n '2,24p' "$0" >&2
  exit 2
fi

IGRAPH_SRC=$(cd "$1" && pwd)
COMMIT=$2
NAME=$3
DEV_ROOT=${4:-$(dirname "$IGRAPH_SRC")}

case "$NAME" in
  ''|*/*|*' '*) echo "name must be short and path-safe: '$NAME'" >&2; exit 2 ;;
esac

REV=$(git -C "$IGRAPH_SRC" rev-parse --verify "${COMMIT}^{commit}")
SRC_DIR="$DEV_ROOT/_scratch/igraph-$NAME"
BUILD_DIR="$HOME/dev/igraph_$NAME"
PREFIX="$DEV_ROOT/_prefix/igraph-$NAME"
for path in "$SRC_DIR" "$BUILD_DIR" "$PREFIX"; do
  if [ -e "$path" ]; then
    echo "refusing to reuse $path; choose a new name" >&2
    exit 3
  fi
done

if [ -n "${SNAPSHOT_BASE_VERSION:-}" ]; then
  BASE_VERSION=$SNAPSHOT_BASE_VERSION
elif [ -f "$IGRAPH_SRC/IGRAPH_VERSION" ]; then
  BASE_VERSION=$(sed 's/-.*//' "$IGRAPH_SRC/IGRAPH_VERSION")
else
  BASE_VERSION=snapshot
fi

mkdir -p "$SRC_DIR" "$HOME/dev" "$DEV_ROOT/_prefix"
git -C "$IGRAPH_SRC" archive "$REV" | tar -x -C "$SRC_DIR"
# git archive has no .git, so CMake reads the version from this file.
printf '%s-%s\n' "$BASE_VERSION" "$NAME" > "$SRC_DIR/IGRAPH_VERSION"

cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_SHARED_LIBS=ON \
  -DIGRAPH_WARNINGS_AS_ERRORS:BOOL=OFF > "$BUILD_DIR.configure.log"
cmake --build "$BUILD_DIR" --parallel --target igraph > "$BUILD_DIR.build.log"
cmake --install "$BUILD_DIR" > "$BUILD_DIR.install.log"

if [ "$(uname)" = Darwin ]; then LOADER=DYLD_LIBRARY_PATH; else LOADER=LD_LIBRARY_PATH; fi
cat <<EOF
snapshot  $NAME
commit    $REV
version   $BASE_VERSION-$NAME
prefix    $PREFIX
logs      $BUILD_DIR.{configure,build,install}.log
use       $LOADER=$PREFIX/lib <python> ...   (set it on the python command itself;
          macOS strips DYLD_* when a SIP-protected binary such as /usr/bin/nohup,
          /usr/bin/env, /bin/sh or /usr/bin/time starts the process)
EOF
