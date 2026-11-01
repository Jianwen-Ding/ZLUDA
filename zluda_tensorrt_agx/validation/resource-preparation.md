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
A representative end-to-end preparation-time/RSS comparison is still needed before claiming a speedup.
No inference-throughput improvement is claimed. Kernel-function resolution is also deduplicated per distinct
kernel name during preparation; repeated graph nodes reuse that resolved function.

Validation uses generated independent models only. The GPU suite now passes 101 graphs. New cases cover a
constant exactly at the 1 MiB packing boundary and a model with two larger-than-budget weights, a small
scalar, repeated elementwise steps and broadcast metadata. Both paths pass serialization round trips and
NumPy numerical comparisons, including output tails. Native lifecycle and 109 ONNX model checks pass.
The C++ runtime/metadata/context probes pass with AddressSanitizer instrumentation on adapters/probes;
the native backend itself is not ASan-instrumented. Optional official ONNX checking remains unavailable.
