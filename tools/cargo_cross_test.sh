#!/usr/bin/env bash
# The Rust core's cargo tests (the C core's bit pins, the soft fma against the
# target's own fma, the verifier, the audit) on a Linux target other than the
# host's, cross-compiled and run under qemu-user, on Debian or Ubuntu:
#
#   tools/cargo_cross_test.sh TARGET [CARGO TEST ARGS...]
#
# TARGET is one of s390x-unknown-linux-gnu (big-endian), powerpc64le-unknown-
# linux-gnu, armv7-unknown-linux-gnueabihf, riscv64gc-unknown-linux-gnu,
# i686-unknown-linux-gnu (run natively on an x86_64 host). Installs the cross
# compiler and qemu-user with apt (sudo when not root) and the target with
# rustup, then runs `cargo test --release --locked` and the debug build's tests
# (every debug assertion, overflow checks on 32-bit usize). No Python here:
# the wheels' own tests run through Python on these targets (tools/core_pins.py).
set -euo pipefail

target=${1:?usage: tools/cargo_cross_test.sh TARGET [CARGO TEST ARGS...]}
shift
case "$target" in
    s390x-unknown-linux-gnu)        gnu=s390x-linux-gnu        deb=s390x    qemu=s390x ;;
    powerpc64le-unknown-linux-gnu)  gnu=powerpc64le-linux-gnu  deb=ppc64el  qemu=ppc64le ;;
    armv7-unknown-linux-gnueabihf)  gnu=arm-linux-gnueabihf    deb=armhf    qemu=arm ;;
    riscv64gc-unknown-linux-gnu)    gnu=riscv64-linux-gnu      deb=riscv64  qemu=riscv64 ;;
    i686-unknown-linux-gnu)         gnu=i686-linux-gnu         deb=i386     qemu=i386 ;;
    *) echo "unknown target $target" >&2; exit 2 ;;
esac

sudo=""
[ "$(id -u)" = 0 ] || sudo=sudo
packages="gcc-$gnu libc6-dev-$deb-cross"
native=no
if [ "$qemu" = i386 ] && [ "$(uname -m)" = x86_64 ]; then native=yes; fi
[ "$native" = yes ] || packages="$packages qemu-user"
$sudo apt-get update -qq
# shellcheck disable=SC2086
DEBIAN_FRONTEND=noninteractive $sudo apt-get install -y -qq --no-install-recommends $packages >/dev/null

rustup target add "$target"
upper=$(echo "$target" | tr 'a-z-' 'A-Z_')
export "CARGO_TARGET_${upper}_LINKER=$gnu-gcc"
if [ "$native" = no ]; then
    export "CARGO_TARGET_${upper}_RUNNER=qemu-$qemu -L /usr/$gnu"
fi

cd "$(dirname "$0")/../rust"
echo "== cargo test --release --target $target ($([ "$native" = yes ] && echo native || echo "qemu-$qemu"))"
cargo test --release --locked --target "$target" "$@"
echo "== cargo test (debug) --target $target"
cargo test --locked --target "$target" "$@"
