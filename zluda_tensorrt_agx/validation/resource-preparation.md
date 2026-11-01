# Resource preparation: bounded constant staging

GPU context preparation previously copied every graph constant into a combined host vector before upload.
It now packs at most 1 MiB of small weight data (including alignment gaps). Any constant that would exceed
that budget is uploaded directly from its immutable graph vector. Broadcast/transpose/gather metadata is
still packed afterward and can exceed the small-weight budget; there is no claim of a total 1 MiB cap.

A single device allocation contains the packed region and all direct-upload weights, with 256-byte aligned
starts and checked size/address arithmetic. Padding is not read by kernels. Metadata and constants retain
shared engine/CUDA-context ownership. The host write API is synchronous, so borrowed graph slices remain
valid through transfer. There is no new GPU instruction or target hint.

This removes the model-sized temporary host copy for large weights. It can increase transfer-call count:
one for packed data and one per directly uploaded constant. Small models retain packed upload behavior.
A local synthetic preparation comparison is recorded below; broader model validation is still needed.
No inference-throughput improvement is claimed. Kernel-function resolution is also deduplicated per distinct
kernel name during preparation; repeated graph nodes reuse that resolved function.

Validation uses generated independent models only. The GPU suite now passes 101 graphs. New cases cover a
constant exactly at the 1 MiB packing boundary and a model with two larger-than-budget weights, a small
scalar, repeated elementwise steps and broadcast metadata. Both paths pass serialization round trips and
NumPy numerical comparisons, including output tails. Native lifecycle and 109 ONNX model checks pass.
The C++ runtime/metadata/context probes pass with AddressSanitizer instrumentation on adapters/probes;
the native backend itself is not ASan-instrumented. Optional official ONNX checking remains unavailable.


## Fresh-process comparison

`bench_load.py --prepare-context --repetitions 5` now measures engine loading, first-context preparation,
and preparation of a second context on the same engine/CUDA context. Each sample uses a fresh process;
within it both execution contexts coexist, then are destroyed before engine and primary-context cleanup.
The benchmark prepares resources but does not enqueue inference. It reports process peak RSS, not device
memory usage or a phase-specific allocation counter. CUDA initialization is outside context-preparation timing.

Baseline source: `ba6204d`. Candidate source: `81a8b04`. Both release libraries were built locally offline
with the same toolchain; no vendor runtime or engine was loaded. The fixture is our generated 32 MiB
single-constant addition model. Results are medians across five fresh processes per path:

| Build and load path | Peak RSS KiB | First context ms | Cached context µs |
| --- | ---: | ---: | ---: |
| Baseline, buffer | 114816 | 8.160 | 18.71 |
| Candidate, buffer | 82080 | 5.453 | 21.25 |
| Baseline, reader | 81616 | 8.951 | 20.00 |
| Candidate, reader | 48848 | 5.479 | 22.04 |

The reader-path median peak RSS decreased by 32768 KiB (40.1%), corresponding to the removed 32 MiB
host staging copy. First-context median time decreased in this fixture; cached-context timing remained
around 20 µs and does not establish an improvement. Baseline samples ran before candidate samples, and
file/compiler caches and scheduling affect timings. This is not a controlled inference-throughput result,
a confidence interval, or a guarantee for many-small-weight models. The latter still need end-to-end testing.
Raw results are in [resource-preparation-samples.json](resource-preparation-samples.json).

```sh
ZLUDA_AGX_LIBCUDA=/path/to/our/libcuda.so ZLUDA_AGX_ZOC=/path/to/our/zoc \
python3 zluda_tensorrt_agx/tests/bench_load.py --library /path/to/libtensorrt_agx.so \
  --weights-mib 32 --prepare-context --repetitions 5
```

A 1 MiB fixture also passed the preparation benchmark. The restored current release build passes all 101
GPU graph checks after the comparison. Build artifacts for different checkouts should use separate target
directories; this experiment rebuilt the current package after using the temporary baseline checkout.
