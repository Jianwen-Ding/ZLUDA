#!/usr/bin/env bash
# Builds and runs the Metal backend end-to-end tests. Only clang is used:
# clang compiles the CUDA kernels to PTX and the host program; no nvcc, no
# Metal toolchain. On Linux the kernels execute on the software Metal device
# from zluda_metal/linux_shim.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)" # the zluda_metal workspace
out="${OUT_DIR:-$root/target/zluda_metal_e2e}"
profile="${PROFILE:-debug}"
mkdir -p "$out"
CLANG="${CLANG:-clang}"
export ZLUDA_SYSTEM_LLVM_CONFIG="${ZLUDA_SYSTEM_LLVM_CONFIG:-$(command -v llvm-config-18 || command -v llvm-config)}"
cargo_flags=()
[ "$profile" = release ] && cargo_flags+=(--release)
(cd "$root" && CC="$CLANG" CXX="${CLANG}++" cargo build -p zluda_metal "${cargo_flags[@]}")
lib="$root/target/$profile/libzluda_metal.so"
[ -f "$lib" ] || lib="$root/target/$profile/libzluda_metal.dylib"
ln -sf "$lib" "$out/libcuda.so.1"
ln -sf "$lib" "$out/libcuda.so"
"$CLANG"++ -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_70 \
    -Xclang -target-feature -Xclang +ptx70 -O2 -S -include "$here/clang_cuda.h" \
    -o "$out/kernels.ptx" "$here/kernels.cu"
"$CLANG" -std=c11 -O1 -Wall -o "$out/zluda_metal_e2e" "$here/test_main.c" -L"$out" -lcuda -lm \
    -Wl,-rpath,"$out"
"$out/zluda_metal_e2e" "$out/kernels.ptx"
