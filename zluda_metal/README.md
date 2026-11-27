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
| `src/cuda.rs`, `src/cuda/*.rs` | CUDA driver API exported from `libzluda_metal.so`/`.dylib` (use it as `libcuda.so.1`); see [The CUDA driver API](#the-cuda-driver-api) |
| `linux_shim/` | Linux test harness: an objc-compatible runtime + Foundation subset and a software `MTLDevice` that JIT-executes AIR, so the real metal-cpp code path runs on Linux |
| `tests/` | cargo tests and the clang-only end-to-end suite |

## The CUDA driver API

`src/cuda.rs` (modules, functions, launches) and `src/cuda/` implement the driver API from the documented
semantics of the CUDA Driver API reference (every declaration is our own). `tests/driver/test_driver.py` checks
each behavior below (about 500 checks, each group in its own process, AGX backend), with the documented behavior it
asserts next to each check.

| Area | Supported |
| --- | --- |
| Initialization (`cuda.rs`, `errors.rs`) | `cuInit` (flags must be 0), `CUDA_ERROR_NOT_INITIALIZED` before it from every call but `cuDriverGetVersion` / `cuGetErrorName` / `cuGetErrorString` / `cuGetProcAddress`, `CUDA_ERROR_DEINITIALIZED` from calls made after the driver's exit handler; every `CUresult` has its name and a description |
| Devices (`device.rs`) | one device; `cuDeviceGetAttribute` answers every `CUdevice_attribute` (limits of the simulated M2 Max: 38 multiprocessors, 3072 threads / 64 KiB shared memory / 53248 registers each, 48 KiB shared per block, 64 KiB opt-in; `CUDA_ERROR_INVALID_VALUE` for values outside the enumeration), name, UUID, PCI bus id (and back), properties, peer queries (no peer) |
| Contexts (`context.rs`) | a real per-thread context stack (`cuCtxCreate` v2 / v3 / v4, `Push` / `Pop` / `SetCurrent` / `GetCurrent`, `cuCtxDestroy` pops it if current, other threads then get `CUDA_ERROR_CONTEXT_IS_DESTROYED`); no implicit context (`CUDA_ERROR_INVALID_CONTEXT`); flags, id, API version, limits (`cuCtxSetLimit` / `GetLimit` for every `CUlimit`: the device runtime's, `src/device_rt.rs`), cache and shared memory configuration, stream priority range (0 .. -5); the primary context is reference counted (`Retain` / `Release` / `GetState` / `SetFlags`; released to 0 or `cuDevicePrimaryCtxReset` it frees its resources, a reset one stays retained); peer access returns the documented errors for a single device |
| Resources | allocations, modules, streams and events belong to their context: destroying it (or releasing / resetting the primary context) frees them; handles are validated (`CUDA_ERROR_INVALID_HANDLE` for stale or bogus ones) |
| Memory (`memory.rs`) | `cuMemAlloc` (0 bytes: `CUDA_ERROR_INVALID_VALUE`), `cuMemGetInfo` tracks what is allocated, `cuMemAllocPitch` (512-byte pitch), 1D / 2D / 3D copies and their async forms (`cuMemcpy2D`, `cuMemcpy3D`, unified `cuMemcpy`, peer copies), `cuMemsetD8/16/32`, `cuMemsetD2D8/16/32` and async forms, bounds checked; pinned (`cuMemAllocHost`, `cuMemHostAlloc`), registered (`cuMemHostRegister`, with the documented overlap errors) and managed (`cuMemAllocManaged`, `cuMemPrefetchAsync`, `cuMemAdvise`) memory is mapped for the device at its host address (zero copy: a kernel writes the host's buffer); `cuMemHostGetDevicePointer`, `cuMemHostGetFlags`; `cuPointerGetAttribute(s)` (every attribute; unknown pointers per the documented `cuPointerGetAttributes` defaults), `cuPointerSetAttribute` (SYNC_MEMOPS); the stream ordered allocator (`cuMemAllocAsync`, `cuMemFreeAsync` in stream order, the default memory pool and its attributes) |
| Streams and events (`stream.rs`) | streams run their work in order: in the calling thread when idle, otherwise on the stream's worker thread, so `cuLaunchHostFunc` / `cuStreamAddCallback` run asynchronously and `cuStreamQuery` / `cuEventQuery` return `CUDA_ERROR_NOT_READY` while work waits behind them; the legacy default stream synchronizes with blocking streams, non-blocking ones are independent; `cuStreamWaitEvent` orders streams; priorities (clamped), flags, ids, context; `cuEventElapsedTime` measures the simulated GPU time (the AGX Performance Simulator's prediction of each kernel; kernels are timed once a timing event exists; `ZLUDA_EVENT_TIMING=wall` for host time) and returns the documented `INVALID_HANDLE` / `NOT_READY` errors |
| Modules and functions (`cuda.rs`) | `cuModuleLoad(Data(Ex))`, `cuModuleGetFunctionCount`, `cuModuleEnumerateFunctions`, `cuModuleGetLoadingMode`; `cuFuncGetAttribute` for every attribute (registers and stack from the AGX code, PTX target version), `cuFuncSetAttribute` (dynamic shared memory opt-in, carveout, cluster hints; read-only ones rejected), `cuFuncGetName` / `GetModule` / `GetParamInfo` / `IsLoaded` |
| Launches | grid, block and shared memory limits checked as CUDA does (the dynamic shared memory against the function's `MAX_DYNAMIC_SHARED_SIZE_BYTES`, registers against the core); `cuLaunchCooperativeKernel` and `cuLaunchKernelEx`'s cooperative attribute (the grid must be co-resident by the occupancy model, and one block while the functional model runs blocks one after another, see the device runtime below: `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`), `cuLaunchKernelEx` with launch attributes |
| Occupancy (`occupancy.rs`) | `cuOccupancyMaxActiveBlocksPerMultiprocessor(WithFlags)`, `cuOccupancyMaxPotentialBlockSize(WithFlags)` (with the block-size-to-shared-memory callback), `cuOccupancyAvailableDynamicSMemPerBlock`, from the simulator's occupancy model (registers, threads, shared memory) |
| Library API (`library.rs`) | CUDA 12's context-independent `cuLibraryLoadData` / `LoadFromFile` / `Unload` / `GetKernel` / `GetModule` / `GetGlobal` / `GetKernelCount` / `EnumerateKernels`, `cuKernelGetFunction` / `GetAttribute` / `SetAttribute` / `GetName` / `GetLibrary` / `GetParamInfo`; a `CUkernel` launches directly; `cuLink*` with one PTX input |
| Graphs (`graph.rs`) | explicit graphs (kernel, memcpy, memset, host, empty, event record / wait nodes, dependencies, node and edge queries, clone), stream capture (`cuStreamBeginCapture` in the three modes, fork / join through events, `cuStreamIsCapturing`, `cuStreamGetCaptureInfo` v1-v3, the documented invalidation errors: synchronizing a capturing stream, unsafe calls in global mode, implicit legacy-stream dependencies, unjoined forks, isolation), `cuGraphInstantiate*`, `cuGraphLaunch` (nodes in a topological order on the stream) |
| Entry points (`proc_address.rs`) | `cuGetProcAddress(_v2)` finds every exported `cu*` function, choosing among `_v2` / `_v3` / ... by the requested CUDA version, with the documented symbol status |

Not implemented: CUDA arrays, textures and surfaces (`CU_MEMORYTYPE_ARRAY` copies return `CUDA_ERROR_NOT_SUPPORTED`),
the virtual memory management API (`cuMemCreate`, `cuMemMap`, ...), IPC, external memory / semaphores, graphics
interop, stream memory operations (`cuStreamWaitValue*`), launch-time thread block clusters, grid-wide
synchronization inside a cooperative kernel (the functional model runs blocks one after another), graph updates
(`cuGraphExec*SetParams`, `cuGraphExecUpdate`) and child graph nodes, linking several PTX inputs, and the CUDA
runtime's private export tables (`cuGetExportTable`). Managed and mapped host memory need the AGX backend.

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
(memset, DtoD, address ranges, streams, events). `tests/driver/test_driver.py` is the driver API conformance
suite (see [The CUDA driver API](#the-cuda-driver-api)).

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

* handle, stream, pointer mode (host and device scalars), math mode, atomics mode, SM count target, workspace,
  `GetProperty`, `GetCudartVersion`; `Set/GetVector`, `Set/GetMatrix` and their `Async` forms (synchronous)
* level 1: S / D `axpy scal copy swap dot nrm2 asum amax amin rot rotg rotm rotmg`; C / Z `axpy scal
  (Cs/Zd)scal copy swap dotu dotc (Sc/Dz)nrm2 (Sc/Dz)asum (Cs/Zd)rot`
* level 2: S / D `gemv gemvStridedBatched gemvBatched gbmv symv sbmv spmv ger syr syr2 trmv tbmv tpmv trsv tbsv
  tpsv`; C / Z `gemv`
* level 3: S / D `gemm gemmStridedBatched gemmBatched symm syrk syr2k syrkx trmm trsm trsmBatched geam dgmm`; C / Z
  `gemm gemm3m gemmStridedBatched gemmBatched`; `Hgemm`, `HgemmStridedBatched`, `GemmEx`, `GemmStridedBatchedEx`,
  `GemmBatchedEx` for f16 / bf16 data (float accumulation, output in the data type or f32), f32 and f64
* the `_64` entry points of all of these (dimensions beyond int32: `CUBLAS_STATUS_NOT_SUPPORTED`)
* not implemented (not exported): the complex level 2 / 3 routines other than gemv / gemm (hemv, herk, ...), `Icamax`,
  `Crot` with a complex sine, packed rank updates (`spr`, `spr2`), the batched LU / QR / least-squares routines,
  cublasLt; matrices beyond 2^31 elements

Argument checks follow cuBLAS's documented rules (`CUBLAS_STATUS_INVALID_VALUE` for bad enums, negative
dimensions, short leading dimensions, zero increments, in-place geam other than C = A / B). How the routines map
onto the kernels:

* GEMMs with a float or half C (`Sgemm`, `Hgemm`, f16 / bf16 `GemmEx`) run on AGX's SIMD-group matrix multiply
  (`air.simdgroup_matrix_*`: one SIMD group per 16 x 16 or, when that still gives 256 of them, 32 x 32 tile of an
  f32 C, k in steps of 8; alpha and beta as products with alpha I / beta I). Shapes the tiles do not divide run on
  zero-padded copies of op(A), op(B) and C (`pad_*` kernels), an f16 / bf16 C on an f32 copy (`cvt_*`), so every
  shape from 16 x 16 x 8 up (and, for padded copies, 2^17 multiply-adds) uses the matrix hardware; smaller ones and
  doubles use a register-tiled kernel (4 x 4, f64 2 x 2 per thread). `ZLUDA_BLAS_NO_MMA=1` forces it,
  `ZLUDA_BLAS_MMA_TILE=16 / 32` the tile.
* `gemmBatched` / `GemmBatchedEx` read the device arrays of matrix addresses in the kernels (one launch).
* gemv: 32 rows (a SIMD group's lanes) x 32 (f64: 16) column groups per block with the partial sums added in
  threadgroup memory, or, for row-contiguous op(A) / few rows, a SIMD group per row; batched over grid y.
* dot / nrm2 / asum / amax / amin are single launches: blocks of 256 threads write partials and the last block to
  finish (an atomic count) combines them in block order (deterministic).
* symmetric / triangular / banded / packed matrices are expanded by one elementwise kernel (`geam`, which also
  serves geam, dgmm and triangle-masked copies) to dense copies for gemv / gemm: symv, sbmv, spmv, gbmv, trmv,
  tbmv, tpmv, symm, trmm; syrk / syr2k / syrkx are GEMMs into a copy of C whose triangle is copied back.
* trsv / tbsv / tpsv / trsm: one SIMD group per right-hand side (trsm: B's columns, or rows on the right); lane l
  owns the elements i = l mod 32, the solved unknown is broadcast by a shuffle (no shared memory).
* complex data is the real type's memory interleaved; products are real fmas with per-operand conjugation.

Predicted times (`ZLUDA_AGX_STATS`, `tests/blas/bench_blas.py`, the simulated M2 Max):

| Call | before | now |
| --- | --- | --- |
| Sgemm 256^3 / 512^3 / 1024^3 | 17.3 / 46.2 / 238.9 us | 17.3 / 46.1 / 238.3 us |
| Sgemm 250^3 / 500^3 / 1000^3 | 118.4 / 226.0 / (not simulated) us | 32.5 / 76.1 / 319.9 us |
| Sgemm 255 x 257 x 129 | 62.6 us | 25.1 us |
| Hgemm 256^3 / 512^3 | 122.8 / 241.2 us | 21.9 / 53.1 us |
| Dgemm 128^3 | 2197.9 us | 410.2 us |
| Sgemv 1000 x 1000 (N / T) | 321.6 / 101.6 us | 16.2 / 13.6 us |
| Dgemv 1000 x 1000 | 1198.6 us | 75.6 us |
| Sdot / Snrm2 / Isamax 1e6 | 89.1 / 52.5 / 56.1 us | 27.3 / 15.6 / 16.5 us |

Double precision is emulated (AIR's portable passes). `tests/blas/test_blas.py` checks every routine against
numpy (transposes, leading dimensions, negative increments, batches, both pointer modes, structured storage,
argument errors). The kernels' object is 376 KB (49 kernels; the f64 ones are most of it: emulated doubles); it
loads in 0.3 ms (kernels' code is copied out when first used).

### Device runtime (`src/device_rt.rs`, `ptx/lib/metal_impl/devrt.inc`)

What CUDA's device runtime provides to kernels, with its host-side semantics. The device side keeps its state in
buffers the host owns, found through module variables (`__zluda_devrt_printf`, `__zluda_devrt_heap`,
`__zluda_devrt_clock`) that the launch fills; state is per context.

* **printf** (`vprintf`, as clang lowers device `printf`): the device parses the format, copies the arguments (one
  8-byte slot each; `%s` strings by value, at most 4096 bytes) into a circular FIFO, and the host formats each record
  with the C library's `snprintf`, spec by spec, so `%d %i %u %x %X %o %c %s %p %f %e %g %a`, flags, width,
  precision, `*`, `h hh l ll z j t` and `%%` behave as C's (at most 32 arguments, as CUDA). Output goes to the
  process's C `stdout` at CUDA's flush points: the start of the next launch, `cuCtxSynchronize` /
  `cuStreamSynchronize` / `cuEventSynchronize`, the blocking `cuMemcpy*`, `cuModuleLoad*` / `cuModuleUnload`,
  `cuCtxDestroy` / `cuDevicePrimaryCtxReset`. Records print in the order the threads wrote them. The FIFO is
  `CU_LIMIT_PRINTF_FIFO_SIZE` bytes (1 MiB by default; changing it prints the pending output and replaces the FIFO);
  a launch that prints more loses its oldest output (16 bytes per record plus 8 per argument). `printf` returns the
  number of arguments, -1 for a null format, -2 when the record does not fit.
* **malloc / free** (PTX `call malloc` / `call free`): a lock-free size-class allocator in a heap of
  `CU_LIMIT_MALLOC_HEAP_SIZE` bytes (8 MiB by default), allocated at the first launch of a kernel that uses it and kept
  for the context's life, so blocks stay valid across launches until freed. Pointers are 16-byte aligned; freed blocks
  are reused by allocations of the same class (four classes per power of two, at most 25% padding); blocks are not
  split or coalesced. An exhausted heap returns NULL. The size cannot change once the heap exists
  (`CUDA_ERROR_INVALID_VALUE`).
* **Clocks**: one cycle counter per launch: each SIMD group's read advances it by 25 cycles (the lanes of a group read
  one value), `nanosleep.u32 t` by `t` ns at the reported 1.398 GHz (capped at 1 ms, PTX's bound), and between
  launches it moves on by the simulator's instruction count of each launch spread over the 8 reported
  multiprocessors. So `%clock` / `%clock64` are monotonic within and across launches and `%globaltimer` is
  nanoseconds since the Unix epoch (the launch's start, then the counter's cycles at the clock rate), monotonic too.
  Values model time only that coarsely: the functional model has no time of its own.
* **Special registers**: `%laneid`, `%lanemask_*`, `%warpid` (< `%nwarpid`, the CTA's warps), `%smid` (the CTA's
  linear index modulo `%nsmid` = 8), `%warpsize` as PTX defines them. Dynamic shared memory is 16-byte aligned.
* **Limits** (`cuCtxSetLimit` / `cuCtxGetLimit`): the printf FIFO and heap as above; `CU_LIMIT_STACK_SIZE` (1 KiB),
  the device-runtime sync depth / pending launches and the L2 limits are recorded and reported but change nothing
  (a kernel whose local memory overflows its stack faults with `CUDA_ERROR_HARDWARE_STACK_ERROR`); other limits are
  `CUDA_ERROR_UNSUPPORTED_LIMIT`.
* **Cooperative launches** (`cuLaunchCooperativeKernel`): the AGX functional model runs a dispatch's workgroups one
  after the other, each to its end, and a threadgroup's SIMD groups each to its next barrier: a grid-wide barrier
  (every block waiting for every other) can only complete when the grid is one block, so larger cooperative grids
  fail with `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE` (720), as CUDA fails a grid that cannot be co-resident.
  The same scheduling makes cross-block message passing work when the producer block has the lower index (blocks run
  in order) and hang (`CUDA_ERROR_LAUNCH_TIMEOUT`, sticky) the other way round, and any spin-wait on another SIMD group
  of the same block that is not separated by a barrier hang too; CUDA (independent thread scheduling) completes both
  when the blocks are co-resident. Supporting them needs a preemptive scheduler in the functional model (run every
  workgroup of a cooperative launch, switching groups after a step budget).
* Memory ordering: `membar` / `fence` / `__threadfence` and atomics are sequentially consistent in the functional
  model (one SIMD group runs at a time).
* Not reproduced: out-of-bounds shared memory accesses are dropped (loads read 0) by the functional model, as by the
  hardware it models, so the kernel completes where CUDA reports `CUDA_ERROR_ILLEGAL_ADDRESS`.

`tests/device/test_device.py` checks all of it (`kernels.cu`, compiled by clang alone to `kernels.ptx`; `--regen`).

### Floating point modes (`ZLUDA_FP_MODE`)

| Mode | Behavior |
| --- | --- |
| `fast` (default) | Results within a few ulp of PTX's. The `.approx` transcendentals (`sin`, `cos`, `ex2`, `rsqrt` and the half forms of `ex2` / `tanh`) are the hardware's approximations, `tanh.approx` a short float evaluation (about 2 ulp of the true tanh), NaN results keep the hardware's bit patterns. Denormals are IEEE (as CUDA's default `-ftz=false`); `div`, `sqrt` (correctly rounded), directed roundings and double precision (software) are exact. |
| `fast-ftz` | `fast` with binary32 denormal operands and results flushed to zero, as `-ftz=true` / `--use_fast_math`: the hardware's own arithmetic. |
| `strict` | Bit for bit the formal model: IEEE binary32 with denormals (`!agx.keep-denormals`), the model's transcendentals (`sin`, `cos`, `ex2`, `lg2`, `tanh` correctly rounded, `rsqrt` as two roundings, `sqrt.approx` correctly rounded), canonical NaN (`-mattr=+canonical-nan`). |

In strict mode the binary32 `tanh`, `sin` / `cos` (|x| < 200), `ex2` and `lg2` are evaluated in float-float
arithmetic (`float.inc` `f_exp_pm`, `f_sincos_fast`, `f_exp2_all`, `f_lg2_fast`). `sin` / `cos` / `lg2` return it when
both ends of the result's error interval round alike, otherwise take the exact path; `tanh` and `ex2` always round it,
with a 64-slot table of the 23 operands each where that rounding differs from the model (found exhaustively); their operations never see a subnormal and are
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
| `tests/device/test_device.py` | device runtime: printf, malloc / free, clocks, limits, special registers, cooperative launches |
| `tests/ptxconf/ptxconf.py` | PTX-formalization's harness programs (`.buffer`/`.launch`/`.expect`, `.tensormap`), every `.expect` checked bit for bit; `triage.py` ranks blockers |
| `tests/stress/gen.sh OUT` | a large generated suite (the formalization's ISA generators with new seeds, 512 cases per launch, ~880 programs, with `stress/gen/gen_sincos.py` and `gen_transc.py`: `sin` / `cos` / `ex2` / `lg2` / `rsqrt` / `sqrt` / `rcp` `.approx` against the model, where the hardware's approximations differ); `stress/gen/gen_tanh.py` adds exact `tanh.approx` cases |
| `tests/stress/exhaustive/fastpath.cpp` | the helper library's binary32 fast paths against its exact paths and the model, for every operand, on the host |
| `tests/alive/opt_check.py` | alive2 (`alive-tv --fp-uf`, the fork in `../../alive2`) proves that each kernel's AGX IR at -O3 refines -O0; `--per-pass` checks every pass that changes a kernel on its own (run one file per process, since a run over all 597 conformance files takes about 11 hours on 4 cores). The last per-pass run checked 15710 pass applications and found 0 incorrect: 10491 correct, 243 correct-bounded (loops unrolled), 1717 unknown (60 s solver timeout), 3259 error (alive2 cannot translate the IR, mostly passes it sees no change in). 62 files hit a 30-minute per-file cap and `tex-mip` a pipeline timeout. The three incorrect results it found were gaps in the fork's GPU model, now fixed: stores to a write-only, kernel-private threadgroup variable (`dynshared`), and the alignment of dynamic threadgroup memory (`extern_shared`, `extern_shared_call`) |
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
* Textures/surfaces, cooperative groups beyond warps and one-block grids, tensor core
  (`mma`, `ldmatrix`) instructions and cubin/fatbin images are not supported;
  modules must be loaded as PTX text.
* Kernels of different streams do not run concurrently (the device executes one kernel at a time).
* AGX pipeline, texture code size: texture fetches are the helper library's software sampler
  (`ptx/lib/metal_impl/texture.inc`, the formalization's filtering computed in emulated binary64), inlined at
  every call because AGX calls are still experimental (`-agx-experimental-leaf-calls`). PTX-formalization's
  `tex-mip.ptx`, 35 fetches in one kernel, becomes 8.3 MB of AGX IR and takes 22 s at -O3 (10.8 MB and 24 s before its descriptor and texel reads stopped being volatile) (too large for
  `opt_check.py --per-pass`, which prints the module after every pass). Ways out: outlined sampler calls once
  leaf calls are validated on hardware, or a binary32 / hardware-sampler path for fast mode.
* On the software device, warp-level operations inside divergent code where
  only part of a SIMD group participates can deadlock.
