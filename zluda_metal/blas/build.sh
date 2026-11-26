#!/bin/sh
# cuBLAS device code: gen_kernels.py -> MLIR (GPU dialect) -> AIR -> AGX object, by the LLVM fork's mlir-opt
# (MLIR_OPT). Writes OUT (default blas/zluda_blas.agx.o, the checked-in copy build.rs uses without MLIR).
set -e
here=$(dirname "$0")
out=${1:-$here/zluda_blas.agx.o}
mlir_opt=${MLIR_OPT:-$here/../../../llvm-project/build/bin/mlir-opt}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python3 "$here/gen_kernels.py" > "$tmp/kernels.mlir"
"$mlir_opt" "$tmp/kernels.mlir" --convert-scf-to-cf --convert-gpu-to-air=use-bare-ptr-memref-call-conv=1 \
    --convert-math-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm \
    --reconcile-unrealized-casts --gpu-module-to-binary -o "$tmp/binary.mlir"
python3 "$here/extract_binary.py" "$tmp/binary.mlir" "$out"
