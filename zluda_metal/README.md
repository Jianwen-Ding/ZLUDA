# ZLUDA on Metal

`zluda_metal` runs unmodified CUDA programs (driver API, PTX kernels) on
Metal. It reuses ZLUDA's PTX front end and replaces the ROCm/HIP back end with
an open **PTX → LLVM → `.air` → `.metallib` → metal-cpp** pipeline:

```
PTX ──ptx_parser + ptx::to_llvm_module──▶ ZLUDA LLVM IR (AMDGPU address spaces)
    ──cpp/air_lower.cpp────────────────▶ .air  (one AIR bitcode module per kernel)
    ──cpp/metallib.cpp─────────────────▶ .metallib
    ──cpp/metal_backend.cpp (metal-cpp)▶ MTLLibrary / MTLComputePipelineState / dispatch
```

No proprietary component is used anywhere in the pipeline: no Apple compiler
(`metal`, `metallib`, `air-lld`), no Metal Shader Converter, no NVIDIA tool
or header. AIR is produced with upstream LLVM, the metallib container is
written by our own code, and the runtime only uses Apple's open-source
(Apache-2.0) [metal-cpp](https://developer.apple.com/metal/cpp/) headers.
Everything is built and tested with clang.

## Layout

| Path | What |
| --- | --- |
| `src/compile.rs`, `src/bin/ptx2metallib.rs` | PTX → metallib driver, `ptx2metallib` CLI |
| `cpp/air_lower.cpp` | ZLUDA IR → AIR lowering (LLVM C++ API) |
| `cpp/metallib.cpp`, `cpp/sha256.h` | open `.metallib` writer and reader |
| `../ptx/lib/zluda_ptx_metal_impl.cpp` | Metal versions of the `__zluda_ptx_impl_*` helpers (sregs, shuffles, votes, bit ops, ...) compiled to bitcode by clang at build time |
| `cpp/metal_backend.cpp` | metal-cpp runtime: buffers, libraries, pipelines, dispatch |
| `src/cuda.rs` | CUDA driver API exported from `libzluda_metal.so`/`.dylib` (use it as `libcuda.so.1`) |
| `linux_shim/` | Linux test harness: an objc-compatible runtime + Foundation subset and a software `MTLDevice` that JIT-executes AIR, so the real metal-cpp code path runs on Linux |
| `tests/` | cargo tests and the clang-only end-to-end suite |

## The AIR lowering

ZLUDA's emitter produces IR for AMDGPU. `air_lower.cpp`:

1. links the Metal helper library and inlines everything into the kernels
   (Metal has no recursion; helpers become `alwaysinline`);
2. runs SROA/InstCombine and `InferAddressSpaces` on the flat address space,
   then recovers pointer provenance from PTX's integer pointer arithmetic
   (`ptrtoint`/`add`/`inttoptr`, also through loop phis and compares) so that
   threadgroup and thread pointers never go through integers;
3. remaps address spaces to AIR (private 5 → thread 0, global/flat → device 1,
   constant/param 4 → constant 2, shared 3 → threadgroup 3);
4. builds the Metal entry points: CUDA parameters are packed in one constant
   buffer (`[[buffer(0)]]`, CUDA layout and alignment), thread/block indices
   come from `thread_position_in_threadgroup`, `threadgroup_position_in_grid`,
   `threads_per_threadgroup`, `threadgroups_per_grid`,
   `thread_index_in_simdgroup`; `extern .shared` becomes a `[[threadgroup(0)]]`
   argument; the `!air.kernel` argument metadata describes all of it;
5. lowers atomics (`air.atomic.{global,local}.*`, CAS loops for the rest),
   fences, barriers (`air.wg.barrier`), SIMD-group operations
   (`air.simd_shuffle`, `air.simd_ballot`) and math to `air.*` intrinsics;
6. adds `!air.version`, `!air.language_version`, `!air.compile_options` and the
   AIR module flags, and splits the module into one module per kernel.

Writable module-scope `__device__` variables have no Metal equivalent; they
are packed into a per-module device buffer whose address is appended to the
parameter buffer of the kernels that use them, and `cuModuleGetGlobal` returns
addresses inside that buffer. `__constant__` data becomes AIR constant memory.

Device pointers are the GPU addresses of shared-storage `MTLBuffer`s
(`gpuAddress`, Metal 3), made resident with `useResources`, so CUDA pointer
arithmetic and pointers stored in memory work unchanged.

## Building and testing (Linux, clang only)

Requirements: Rust, clang/LLVM 18 with development files (`llvm-18-dev`), and a
metal-cpp checkout (`METAL_CPP_DIR`, defaults to `../ext/metal-cpp` or a
sibling `../../metal-cpp`). `.cargo/config.toml` points
`ZLUDA_SYSTEM_LLVM_CONFIG` at `llvm-config-18`, which makes `llvm_zluda` link
the installed LLVM instead of building the ROCm LLVM submodule. This crate is a
separate cargo workspace so that the main ZLUDA workspace is unaffected.

```bash
cd zluda_metal
CC=clang CXX=clang++ cargo test
```

```bash
CC=clang CXX=clang++ tests/e2e/run.sh
```

`run.sh` compiles `tests/e2e/kernels.cu` to PTX with
`clang -x cuda --cuda-device-only -nocudainc -nocudalib` (the tiny
`clang_cuda.h` prelude replaces the CUDA headers), builds the C host program
`test_main.c` with clang against `libcuda.so.1 → libzluda_metal.so`, and runs
it. On Linux the kernels run on the software Metal device in `linux_shim/`:
metal-cpp is compiled unmodified, its `objc_msgSend` calls go through the
shim's dispatcher, and the device JIT-compiles the AIR found in the metallib
with LLVM ORC. That executor interprets the `!air.kernel` metadata and the
`air.*` intrinsics independently of the producer, with threadgroup barriers
and 32-wide SIMD groups backed by real threads, so it checks the generated AIR
rather than just echoing it.

The suite covers element-wise kernels, grid-stride loops, static and dynamic
shared memory with barriers, warp shuffles/ballots, 2D/3D grids and blocks,
global atomics (including CAS loops and float add), `.func` calls with private
arrays, by-value struct parameters, integer/float math, pointer loops over
shared memory, device globals and constant tables, plus driver API basics
(memset, DtoD, address ranges, streams, events).

Useful tools:

```bash
target/debug/ptx2metallib kernels.ptx -o kernels.metallib --emit-air air/ --emit-ll ll/
```

* `ZLUDA_METAL_DUMP_DIR=dir` dumps every loaded module's PTX, AIR and metallib.
* `ZLUDA_METAL_LOG=1` prints detailed errors (also available through the
  `zludaMetalGetLastError` export).

On macOS, `build.rs` compiles the same `metal_backend.cpp` against Metal.framework
(no shim) — set `ZLUDA_SYSTEM_LLVM_CONFIG` to an LLVM 18 `llvm-config`.

## The AGX backend (Apple GPU machine code, simulated)

`ZLUDA_METAL_BACKEND=agx` runs every kernel as AGX machine code instead of on the
software Metal device:

```
PTX ──ZLUDA──▶ portable AIR (.metallib) ──llvm-air2agx──▶ AGX code + .agx.kernels ──▶ AGX Performance Simulator
```

* `src/agx.rs` compiles each module's `.metallib` with the LLVM fork's `llvm-air2agx`
  (`ZLUDA_AGX_AIR2AGX`, `-mcpu=agx-g14c`, `ZLUDA_AGX_FLAGS`, default `-mattr=+canonical-nan` in the strict floating point mode), fills
  the launch's uniforms from the `.agx.kernels` record (buffer table, threadgroups per grid, global
  data buffer, extensions 4 / 6 / 7 / 8 / 9, clusters) and runs it on the AGX Performance
  Simulator's functional model (`agx_sim`, linked in). Device memory is below 4 GiB, so
  `.address_size 32` modules work too.
* Portable AIR (`air_lower.cpp`, `portable_air`): doubles and directed-rounding constrained
  intrinsics are kept for the fork's `air-emulate-f64`; a dataflow over ZLUDA's mode-register writes
  gives each operation its PTX rounding and denormal mode (`!agx.keep-denormals` for IEEE binary32
  operations, which the AGX ALU otherwise flushes). Helpers can request IEEE binary32 operations with
  `__zluda_ieee_*_f32`.
* Instructions outside ZLUDA's grammar are lowered by `ptx_parser/src/helpers.rs` and
  `helper_table/` (one module per family: video, misc, cvt, float, matrix, tensor, memory) to calls of
  helpers in `ptx/lib/metal_impl/*.inc`, written from PTX-formalization's Sail model.
* `ZLUDA_COMPILE_TIMES=1` prints the time of each compilation phase (PTX parse, translation, each step of
  `air_lower.cpp`). The helper library is read lazily and only the helpers a module calls are linked.
* `ZLUDA_AGX_STATS=FILE` appends each launch's dynamic instructions and the simulator's predicted time
  (dispatch model, M2 Max, CUDA-runtime launch profile) as JSON lines.

### cuBLAS (`src/blas.rs`)

The library also exports cuBLAS's entry points (an application linked with cuBLAS uses it through a
`libcublas.so.12` link to `libzluda_metal.so`), on the AGX backend. Its device code is written in MLIR's GPU
dialect (`blas/gen_kernels.py`) and compiled by the LLVM fork's MLIR, GPU -> AIR -> AGX object
(`blas/build.sh`: `convert-gpu-to-air`, `gpu-module-to-binary` with `#air.target<chip = "agx-g14c">`); build.rs
runs it when `mlir-opt` exists (`ZLUDA_MLIR_OPT`, or `../../llvm-project/build/bin/mlir-opt`, a build with
`-DLLVM_ENABLE_PROJECTS=mlir`) and otherwise uses the checked-in `blas/zluda_blas.agx.o`.

* handle, stream, pointer mode (host and device scalars), math mode, workspace; `Set/GetVector`, `Set/GetMatrix`
* S / D `axpy scal copy swap dot nrm2 asum amax amin gemv gemm gemmStridedBatched gemmBatched`
* `Hgemm`, `HgemmStridedBatched`, `GemmEx`, `GemmStridedBatchedEx`, `GemmBatchedEx` for f16 / bf16 data
  (float accumulation, output in the data type or f32), f32 and f64
* not implemented: complex types, the other level 2 / 3 routines (`CUBLAS_STATUS_NOT_SUPPORTED` is not
  returned for them: they are not exported), matrices beyond 2^31 elements

GEMMs with a float C (`Sgemm`, f16 / bf16 `GemmEx` to f32) whose m and n are multiples of 16 and k of 8 run
on AGX's SIMD-group matrix multiply (`air.simdgroup_matrix_*`, one SIMD group per 16 x 16 or, for large C, 32 x 32 tile; alpha and beta
as products with alpha I / beta I): a 256^3 Sgemm is predicted at 17.3 us against 120.8 us for the
register-tiled kernel the other shapes use (`ZLUDA_BLAS_NO_MMA=1` forces it). Double precision is emulated
(AIR's portable passes). `tests/blas/test_blas.py` checks every routine against
numpy (transposes, leading dimensions, negative increments, batches, both pointer modes).

### Floating point modes (`ZLUDA_FP_MODE`)

| Mode | Behavior |
| --- | --- |
| `fast` (default) | Results within a few ulp of PTX's. The `.approx` transcendentals (`sin`, `cos`, `ex2`, `rsqrt` and the half forms of `ex2` / `tanh`) are the hardware's approximations, `tanh.approx` a short float evaluation (about 2 ulp of the true tanh), NaN results keep the hardware's bit patterns. Denormals are IEEE (as CUDA's default `-ftz=false`); `div`, `sqrt` (correctly rounded), directed roundings and double precision (software) are exact. |
| `fast-ftz` | `fast` with binary32 denormal operands and results flushed to zero, as `-ftz=true` / `--use_fast_math`: the hardware's own arithmetic. |
| `strict` | Bit for bit the formal model: IEEE binary32 with denormals (`!agx.keep-denormals`), the model's transcendentals (`sin`, `cos`, `ex2`, `lg2`, `tanh` correctly rounded, `rsqrt` as two roundings, `sqrt.approx` correctly rounded), canonical NaN (`-mattr=+canonical-nan`). |

In strict mode the binary32 `tanh`, `sin` / `cos` (|x| < 200), `ex2` and `lg2` are evaluated in float-float
arithmetic (`float.inc` `f_exp_pm`, `f_sincos_fast`, `f_exp2_fast`, `f_lg2_fast`) and returned when both ends of
the result's error interval round alike, otherwise by the exact path; their operations never see a subnormal and are
emitted without `!agx.keep-denormals` (`__zluda_nd_*`). `tests/stress/exhaustive/fastpath.cpp` checks each fast path
against the exact path, and `ex2` / `lg2` / `rsqrt` against binary128 references of the model, for all 2^32 operands.
`air_lower.cpp` also leaves `!agx.keep-denormals` off binary32 operations that provably cannot meet a subnormal
(`FRangeAnalysis`: magnitude bounds and granularity from constants and integer or half conversions;
`ZLUDA_AGX_NO_FRANGE=1` tags them all).

Division and square root are not relaxed in any mode: a reciprocal-based division is wrong by far more
than a few ulp for divisors beyond 2^126 (as CUDA's `__fdividef`). Note that the model's own `.approx`
definitions are not the exact functions (its `tanh.approx` is 13 ulp from tanh near 0.013), so `fast`
results can be closer to the true function than `strict` ones. `ptxconf.py --fp-mode fast` checks with
a ulp tolerance (`--ulp`, flushed denormals and NaN bit patterns accepted): `APPROX` for float typed
expects, `APPROX?` for float bits in integer typed ones; differences of the `.approx` instructions
beyond the tolerance are expected there (PTX only bounds their error, e.g. 2^-11 relative for tanh).

Tests and tools (no NVIDIA or Apple tool anywhere: PTX is hand written or from clang's NVPTX target):

| Tool | What |
| --- | --- |
| `tests/ptxconf/ptxconf.py` | PTX-formalization's harness programs (`.buffer`/`.launch`/`.expect`, `.tensormap`), every `.expect` checked bit for bit; `triage.py` ranks blockers |
| `tests/stress/gen.sh OUT` | a large generated suite (the formalization's ISA generators with new seeds, 512 cases per launch, ~880 programs, with `stress/gen/gen_sincos.py` and `gen_transc.py`: `sin` / `cos` / `ex2` / `lg2` / `rsqrt` / `sqrt` / `rcp` `.approx` against the model, where the hardware's approximations differ); `stress/gen/gen_tanh.py` adds exact `tanh.approx` cases |
| `tests/stress/exhaustive/fastpath.cpp` | the helper library's binary32 fast paths against its exact paths and the model, for every operand, on the host |
| `tests/alive/opt_check.py` | alive2 (`alive-tv --fp-uf`, the fork in `../../alive2`) proves that each kernel's AGX IR at -O3 refines -O0 |
| `tests/bench/bench.py`, `cuda_bench.py` | predicted time of the conformance corpus, and of the LLVM fork's agx-bench CUDA kernels at realistic sizes (outputs hashed against `bench_ref.json`) |

```bash
ZLUDA_AGX_AIR2AGX=../../llvm-project/build/bin/llvm-air2agx \
    python3 tests/ptxconf/ptxconf.py --backend agx ../../PTX-formalization/tests ../../PTX-formalization/tests/zluda
```

## Limitations / status

* **Not yet validated on Apple hardware.** Everything above is tested on Linux
  through the shim. The AIR follows the conventions observable in Metal
  bitcode, but Apple's AIR reader may require an older bitcode flavour (Metal
  toolchains historically consumed typed-pointer, LLVM-5-era bitcode, as the
  open-source Metal.jl project handles with a bitcode downgrader), and the
  `.metallib` header/tag values come from public reverse-engineering notes.
  Expect that loading on a real device needs adjustments in `metallib.cpp`
  and possibly a downgrade step after `air_lower.cpp`.
* No fp64 (Metal has none): such kernels are rejected with a clear error.
* Textures/surfaces, `printf`, cooperative groups beyond warps, tensor core
  (`mma`, `ldmatrix`) instructions and cubin/fatbin images are not supported;
  modules must be loaded as PTX text.
* All CUDA streams share one in-order Metal command queue.
* On the software device, warp-level operations inside divergent code where
  only part of a SIMD group participates can deadlock.
