# Source this file to build python-igraph against a local igraph C prefix:
#
#   . .skills/link-python-igraph/scripts/python_build_env.sh "$C_PREFIX"
#
# It exports, for the current shell only:
#   PKG_CONFIG_PATH        so pkg-config finds <prefix>/lib/pkgconfig/igraph.pc
#   IGRAPH_USE_PKG_CONFIG  so python-igraph's setup.py links the prefix
#                          instead of building its vendored C core
#   LDFLAGS                with an rpath to <prefix>/lib, so the extension
#                          loads that library without DYLD/LD_LIBRARY_PATH
# and on macOS additionally SDKROOT and -isysroot in CFLAGS/LDFLAGS: uv's
# standalone Pythons record the SDK of the machine that built them (for
# example an Xcode_15.2 path that does not exist here), and the link then
# fails with "library 'c++' not found".
#
# Nothing is written to shell startup files.

if [ -z "${1:-}" ] || [ ! -d "$1/lib/pkgconfig" ]; then
  echo "usage: . python_build_env.sh <igraph-prefix>   (needs <prefix>/lib/pkgconfig)" >&2
  return 2 2>/dev/null || exit 2
fi
_igraph_prefix=$(cd "$1" && pwd)
export PKG_CONFIG_PATH="$_igraph_prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export IGRAPH_USE_PKG_CONFIG=1
if [ "$(uname)" = Darwin ]; then
  SDKROOT=$(xcrun --show-sdk-path)
  export SDKROOT
  export CFLAGS="-isysroot $SDKROOT${CFLAGS:+ $CFLAGS}"
  export LDFLAGS="-isysroot $SDKROOT -Wl,-rpath,$_igraph_prefix/lib${LDFLAGS:+ $LDFLAGS}"
else
  export LDFLAGS="-Wl,-rpath,$_igraph_prefix/lib${LDFLAGS:+ $LDFLAGS}"
fi
echo "igraph prefix: $_igraph_prefix ($(pkg-config --modversion igraph 2>/dev/null || echo 'pkg-config cannot find igraph'))"
unset _igraph_prefix
