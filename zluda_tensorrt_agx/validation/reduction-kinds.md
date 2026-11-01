# Maximum, minimum and product reductions

Portable MLIR inference now includes serial and subgroup maximum/minimum/product kernels, in addition to
sum/mean. Kernels read inner-strided rows directly without temporary transposes or coefficient tensors.
The host picks subgroup kernels for axes of at least 32 elements, matching the existing sum/mean policy.
No AGX-specific operation, ISA feature or target hint was added.

The GPU-to-AIR converter currently accepts maxnumf/minnumf subgroup reductions but not maximumf/minimumf.
The new subgroup extrema kernels therefore combine supported min/max with a portable subgroup OR of local
NaN flags, selecting a quiet NaN if any lane saw NaN. Local arithmetic extrema also propagate NaNs.
Product uses a neutral value of one and a standard subgroup multiply. Empty tensor dimensions remain
unsupported by the graph format; empty ONNX axis lists use the existing versioned identity/all-axes rules.
Floating-point product association differs between serial/subgroup paths, so numerical tolerances apply.
No exact signed-zero or NaN-payload preservation guarantee is added for extrema.

Graph reduction metadata adds an optional kind (sum/max/min/product); the default sum field is omitted during
serialization, preserving old sum/mean metadata. Setting mean=true with another kind is rejected before GPU
work. Custom engine magic stays ZTRTAGX2. Native builder kind codes are 0=sum, 1=mean, 2=max, 3=min, 4=product.
The C++ source wrapper explicitly maps SUM/AVG/MAX/MIN/PROD rather than casting the public enum.

ONNX reduction axis handling follows the public [ReduceMax](https://onnx.ai/onnx/operators/onnx__ReduceMax.html),
[ReduceMin](https://onnx.ai/onnx/operators/onnx__ReduceMin.html) and
[ReduceProd](https://onnx.ai/onnx/operators/onnx__ReduceProd.html) specifications: attribute axes for version 13,
constant int64 input axes from version 18, multiple/negative axes, keepdims, scalar inputs and empty-axis
no-op behavior. The runtime continues to accept only its documented static f32 scope.

Validation:

* 131 native GPU graph checks, including serial/subgroup boundaries, strided axes, tails, keepdims,
  serialization round trips, and NaN/infinity cases for each new reduction.
* 175 independently encoded ONNX model checks, including all five reduction kinds at versions 13 and 18.
  The optional official ONNX checker remains unavailable.
* All 19 inference primitive families replayed against NumPy using AGX-Performance-Simulator captures;
  these captures exercise finite inputs, while exceptional-value cases run through the GPU driver harness.
* 14 Rust unit tests, including legacy metadata defaults, exact round trips and ambiguous-mean rejection.
* C++ source tests and actual GPU execution for all five public reduction enum mappings.
* Existing runtime/engine/context ABI probes with C++ adapter/probe AddressSanitizer instrumentation.
  The native Rust backend itself is not ASan-instrumented.

This expands operator support, not full TensorRT ABI/dtype/dynamic-shape coverage. No vendor binary or engine
was inspected or used as an oracle. Compiler/PTX translation was unchanged; this is not a new PTX compliance
claim. Performance beyond the existing subgroup selection policy still needs representative benchmarking.
