# Independent inference validation — 2026-11-01

Tested on the project's Asahi Linux / Apple M2 Max environment using our ZLUDA AGX driver.
No NVIDIA runtime, SDK, vendor engine, binary inspection or vendor numerical oracle was used.
The library's scope and supported formats are described in [the crate README](../README.md).

## Correctness and lifecycle

| Check | Result |
| --- | --- |
| Rust graph/ONNX/serialization tests | 7 passed |
| GPU graph tests | 42 passed; maximum absolute error 1.90735e-6 against NumPy |
| Independently encoded ONNX models | 17 passed; maximum absolute error 5.96047e-8 against NumPy |
| AGX Performance Simulator | All 8 new inference primitive kernels replayed and matched NumPy references |
| C++ source interface | Builder/configuration, shapes, native engine roundtrip, valid ONNX import, rejection paths passed |
| CUDA execution lifetime | Graph capture/replay with changed input, stream changes, shared resources, foreign-context rejection/restoration, engine-before-context destruction passed |
| CPU-only compiler | Public-schema ONNX compiled to our format; 642-byte MLP engine, 256-byte workspace |
| Packaged C++ example | `[11, 11, 0, 0]` after native engine serialization/deserialization |
| Kernel packaging | Same portable MLIR emitted upstream AIR bitcode/text and AGX fat object |

The optional official ONNX Python checker was unavailable. Fixtures follow the public protobuf schema and were
validated by our reader and numerical references; no claim of official-checker validation is made.
Simulator replay uses our own captured primitive launches, including subgroup softmax and fused convolution.
It does not independently replay the entire graph scheduler or claim full PTX compliance.

## Performance and memory

`baseline.json` and `optimized.json` contain 12 samples, 4 enqueues per sample, median end-to-end enqueue/sync
latency. Clocks and system load were not controlled. These small measurements are indicative, not a comparative
benchmark against NVIDIA TensorRT or Metal.

| Case | Initial latency (us) | Optimized latency (us) |
| --- | ---: | ---: |
| Softmax 64 x 4096 | 1855.13 | 235.37 |
| Bias + ReLU, 1M elements | 85.98 | 133.46 |
| MatMul 256 x 256 | 63.68 | 57.77 |
| Conv 8 -> 16, 32 x 32 | 86.58 | 62.51 |

The pointwise kernel was unchanged; its measured timing varied between runs. Softmax's standard MLIR subgroup
reduction and convolution's fused bias/ReLU account for the targeted changes. Approximate-exp hints were tried,
but the incremental gain did not justify keeping them. No new target-specific hints are present.

GPU weights/metadata use one packed immutable allocation per engine and CUDA context, shared by execution
contexts. Each context owns private liveness-reused activations. The benchmark's second context creation took
14–22 us versus 33–129 us for first preparation. Peak benchmark process RSS was 54,880 KiB. Serialization bounds
writes while constructing one header/payload allocation, avoiding a second full payload copy. These figures are
for the small synthetic workloads; they do not establish memory use for large production models.

## PTX smoke check and known failure

`ptx-smoke.json`: `addr32`, `barrier`, `launch-ok`, `fma`, and `mov` passed through AIR and the simulator.
`cvt-0.ptx` failed: `O22[15] expected 0xffff, got 0x0`. The exact failure also appears in existing scratch baseline
records `air12-strict.txt`, `air17-strict.txt`, and `air27-strict.txt`. This work does not change the PTX compiler
and does not resolve that existing conversion issue. The smoke check is not the full PTX-formalization suite.

## Application limits

Geekbench's opaque TensorRT engines remain unsupported: this is an independent source API and native format,
not a binary-compatible drop-in for `libtensorrt_rtx.so`. Existing Geekbench CUDA fallback runs do not validate
this inference implementation. No llama.cpp or Blender integration/performance claim is made. The embedded
C++ MLP and a complete ONNX convolution/pooling/dense/softmax classifier exercise the new API directly.

Reproduction commands are in [the crate README](../README.md). Generated libraries, executables, AIR and AGX
objects, simulator captures and the packaged SDK remain in scratch; this directory records source and results.

## Follow-up: binding dispatch and expanded PTX run

Unchanged execution bindings now reuse successful missing/overlap validation, avoiding its temporary allocation
and pairwise comparisons on each enqueue. Every successful rebind invalidates that result. The lifecycle test
checks overlap rejection after a previously successful enqueue and recovery after restoring the output binding.
All 42 GPU graph tests and lifecycle tests pass. `bindings-cached.json` records a follow-up microbenchmark;
GPU-dominated timings do not isolate the CPU validation saving, so no percentage speedup is claimed.

Expanded strict-fp AIR/simulator check of all 184 `tests/zluda/*.ptx` files: 179 PASS, 2 FAIL, 1 COMPILE-ERROR,
1 SIM-ERROR, 1 UNSUPPORTED. Exact results are in `ptx-expanded.json`. Nonpassing cases:

* `call_rnd.ptx`: OUT[0] expected 0x1, got 0x0.
* `div_noftz.ptx`: OUT[0] expected 0x26, got 0x0.
* `const_ident.ptx`: compiler rejects variable `x` referenced outside kernels.
* `ldmatrix.ptx`: simulator read of unmapped address 0x400017c at pc 56.
* `bar_red_and_pred.ptx`: harness does not support initializer IN.

These were run with the same existing compiler used for the earlier smoke check, without PTX compiler edits.
They remain unresolved; this run alone does not classify their historical regression status. The unsupported
initializer is a harness limitation rather than a numerical pass or failure. No proprietary code was inspected.

## Follow-up: coalesced broadcast indexing

Broadcast metadata now removes unit axes and merges adjacent axes when both operands retain a linear address
map. Runtime dispatch passes the reduced metadata rank to the unchanged portable MLIR kernel. This cuts loop
iterations and integer quotient/remainder work, while preserving broadcasting and scalar semantics. No math
precision changes or target hints were added.

An exhaustive unit check compares every element address for all compatible pairs among 81 four-dimensional
shapes with axis extents 1, 2 or 3. All 8 Rust tests, 42 GPU graphs, 17 ONNX models and lifecycle tests pass.
The eight unchanged kernels also pass replay of existing simulator captures; those captures predate this metadata
optimization, so the new host-generated indexing is validated by the exhaustive address check and GPU tests.

`broadcast-before.json` and `broadcast-after.json` record 30-sample comparisons. Equal-shape addition of 65,536
values (shape 16x16x16x16) measured 52.26 us before and 44.02 us after. The four-axis address loop becomes one axis.
Other workloads varied substantially (including unchanged softmax); these are preliminary uncontrolled-clock
measurements, not a general speedup claim. The new benchmark remains available for repeat measurements.

## Custom serializer, version 2

New engines use our ZTRTAGX2 framing with JSON graph metadata and raw little-endian f32 weight records.
The serializer borrows graph weights and converts them through a fixed 4 KiB stack buffer; it does not clone
weights or encode them as decimal JSON. The reader retains support for our legacy ZTRTAGX1 JSON engines.
All 9 Rust tests pass, including exact float-bit preservation (negative zero and subnormals), duplicate weight
record/invalid tensor rejection, and legacy loading. The 42 GPU graphs, 17 ONNX models, lifecycle tests and C++
source-interface tests pass with the new format. This does not add NVIDIA engine-format compatibility.

`custom-serializer.json` records the same small benchmark: MatMul model build measured 1.87 ms versus 3.74 ms
in `broadcast-after.json`; convolution model build measured 0.052 ms versus 0.098 ms. Build includes JSON input
parsing and validation as well as serialization, so these are not isolated serializer throughput measurements.

## Inference BatchNormalization

The ONNX importer now accepts inference BatchNormalization with static f32 input and constant per-channel scale,
bias, running mean and variance. Coefficients are precomputed once, then executed as portable MLIR multiply/add;
following ReLU uses existing add/ReLU fusion. This removes runtime square roots and parameter normalization.
It still uses two pointwise launches; a dedicated affine fusion remains a possible optimization. Folding can
change floating-point rounding compared with computing the unfused expression.

Four new GPU numerical fixtures cover ranks 2, 3, 4 and 5, nondefault epsilon and mixed-sign scales with ReLU.
Training mode and negative variance rejection are checked. All 21 ONNX models pass (maximum absolute error
4.76837e-7); all 9 Rust tests, 42 graph tests and lifecycle tests also pass. Official ONNX checker remains unavailable.
No PTX compiler or MLIR kernel changes were made in this checkpoint.
Public semantics: https://onnx.ai/onnx/operators/onnx__BatchNormalization.html

## Broadcast-batched and vector MatMul

Native MatMul now accepts rank-2 through rank-8 matrices, broadcasting leading batch axes and applying transpose
only to the final two dimensions. Matching batches and shared-matrix cases use one existing strided-batched
MLIR/AIR GEMM dispatch. Mixed broadcast axes use per-matrix offsets without replicating inputs. This fallback
currently performs one GEMM call per matrix, so mixed-broadcast launch overhead remains an optimization target.
ONNX vector operands are promoted to matrices and the result squeezed with reshape operations, including scalar
dot products. Those reshape operations currently perform device copies; no zero-copy view claim is made.
ONNX Gemm explicitly remains rank-2.

24 new graph tests cover all transpose combinations, shared operands, mixed broadcast axes and rank-8 operands.
Eight new ONNX cases cover batch broadcasting and vector combinations. All 66 GPU graphs and 29 ONNX models
pass; maximum errors are 1.90735e-6 and 9.53674e-7 respectively. Existing lifecycle checks and 9 Rust tests pass.
`batched-matmul.json` records a new eight-batch 64x64 shared-weight microbenchmark: 55.23 us median enqueue/sync,
zero activation workspace, 16,384 constant bytes. This is an initial absolute measurement, not a speedup claim.
No PTX compiler changes, target hints or new MLIR kernels were introduced.

Public semantics: https://onnx.ai/onnx/operators/onnx__MatMul.html and
https://onnx.ai/onnx/operators/onnx__Gemm.html

## Zero-copy internal reshapes

Internal contiguous reshapes now alias their root tensor. The memory planner propagates all view-consumer
lifetimes to the root and allocates no separate view workspace. Enqueue refreshes view addresses after external
bindings change. Output reshapes retain a copy into the caller's independent output buffer, preserving the
nonoverlapping external-binding contract. Thus vector promotion copies are eliminated, while final squeeze
copies remain when a reshape is itself a graph output.

All 66 GPU graphs and 29 ONNX models pass. All 10 Rust tests pass, including chained-view root lifetime and
workspace checks. The lifecycle test now passes input rebinding through a view as well as CUDA graph replay,
cross-stream reuse and engine/context destruction checks.

`tests/bench_views.py` compares reshape -> tanh against the saved pre-view SDK library, checking numerical
output. `views-before.json` and `views-after.json` record 30 samples of four enqueues each. Workspace drops from
1 KiB / 256 KiB / 4 MiB to zero for 256 / 65,536 / 1,048,576 elements. Initial median enqueue/sync measurements
were 150.8 -> 51.5 us, 164.2 -> 62.1 us and 370.7 -> 86.7 us respectively. These compare checkpoints on an
uncontrolled-clock system, not an isolated hardware throughput result; the directly established benefits are
eliminating the reshape copy and its allocation. No target hints or GPU math/precision changes were introduced.

## Versioned sum/mean reductions

ONNX ReduceSum (opsets 13–22) and ReduceMean (13–17 attribute axes, 18–22 input axes) now lower to existing
portable transpose/reshape/GEMM operations. Axes are normalized and duplicates rejected; flags, omitted/empty
axes and scalar inputs have explicit handling. A bounded constant coefficient vector supplies sum or mean
weights. No proprietary code, new target hints or GPU compiler modifications are involved.

44 new numerical fixtures cover both versions, both keepdims settings, multiple/negative/default axes, scalar
inputs and no-op reductions. Additional cases reject duplicate/out-of-range axes, invalid flags and a legacy
axes attribute used with Mean-18. All 73 ONNX cases pass with maximum absolute error 1.43051e-6. The 66 graph
cases, lifecycle tests and 10 Rust tests also pass. The optional official ONNX checker is still unavailable.

`reductions.json` adds a synthetic 64x4096 sum using the exact lowered graph pattern. It reports initial absolute
timing and owned memory, not a speedup over a dedicated reducer. Coefficient storage is O(reduction length),
noncontiguous axes require a transpose, and output reshapes still copy. Dedicated portable reduction kernels
remain necessary performance work; this checkpoint does not claim optimal reduction throughput or memory use.

Public semantics: https://onnx.ai/onnx/operators/onnx__ReduceMean.html and
https://onnx.ai/onnx/operators/onnx__ReduceSum.html
