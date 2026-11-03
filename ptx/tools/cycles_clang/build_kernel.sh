#!/bin/bash
# build_kernel.sh SRC OUT.ptx [clang flags]: Cycles' CUDA kernel (SRC/kernel/device/cuda/kernel.cu) with clang alone,
# for a Cycles kernel that no NVIDIA tool generated. SM picks the target the host expects (default 75).
set -e
HERE=$(cd "$(dirname "$0")" && pwd); SRC=$1; OUT=$2; shift 2
CLANG=${CLANG:-/scratch/clang-build/bin/clang++}; SM=${SM:-75}
exec $CLANG -x cuda --cuda-device-only --cuda-gpu-arch=sm_$SM -nocudainc -nocudalib -nostdinc++ -nostdlibinc \
  -Xclang -target-feature -Xclang +ptx87 -std=c++17 -O3 -ffp-contract=fast -fno-math-errno -fcuda-flush-denormals-to-zero -Wno-everything \
  -include "$HERE/cycles_cuda_prelude.h" -I"$HERE/include" -I"$SRC" \
  -D CCL_NAMESPACE_BEGIN= -D CCL_NAMESPACE_END= -DNVCC -D__CUDACC__ "$@" \
  -S "$SRC/kernel/device/cuda/kernel.cu" -o "$OUT"
