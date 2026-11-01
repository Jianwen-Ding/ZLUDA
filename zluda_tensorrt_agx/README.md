# Independent MLIR/AIR inference runtime

`zluda_tensorrt_agx` builds `libtensorrt_agx.so`: an independent graph builder and inference runtime for Apple GPUs
on Asahi Linux. Its math kernels are portable MLIR, lowered to upstream AIR and then by LLVM to AGX. It shares the
existing MLIR/AIR GEMM implementation with the other AGX math libraries. No new AGX-specific
intrinsics or handwritten AGX instructions are used.

The library has an independent C ABI (`include/zluda_tensorrt.h`) and a **source-level subset** of the documented
TensorRT C++ builder/runtime API (`include/NvInfer.h`, `include/NvOnnxParser.h`). Applications using that subset can
be compiled against these headers and `-ltensorrt_agx`. This is **not binary ABI compatibility** with NVIDIA's
`libtensorrt_rtx.so` or `libnvinfer.so`, and it does not consume NVIDIA's serialized engine format. Do not rename it
to impersonate those libraries or claim it loads Geekbench's opaque TensorRT engines.

The supported input formats are public ONNX and our documented JSON graph; the runtime generates and reads its
own `ZTRTAGX2` engines. No NVIDIA runtime, SDK, compiler, library binary, or vendor-generated engine is required.
In particular, do not install `libcudart.so` for this implementation. CUDA Driver API calls go to our ZLUDA AGX driver.

## Supported execution

* Contiguous **f32**, static positive dimensions, rank 0 through 8, with at most `INT32_MAX` elements per tensor.
* Matrix multiplication of rank-2 through rank-8 operands, with batch broadcasting and either matrix operand transposed, using the tiled MLIR/AIR GEMM engine.
* NCHW 2-D convolution, including grouped/depthwise convolution, symmetric padding, strides, dilation, and bias.
* ReLU, sigmoid, tanh, identity; add/multiply/subtract/divide/min/max with multidimensional broadcasting.
* Stable softmax along one axis; long contiguous rows use portable `gpu.subgroup_reduce` operations.
* Max/average 2-D pooling with padding accounting; reshape, transpose, and concatenation along any axis.
* Named device bindings, asynchronous stream execution, independently owned execution contexts, serialization.

The graph front end and dispatch/memory planner are Rust. GPU math is implemented in MLIR/AIR. Kernels compile
when the library is built; model building does not need a running GPU or a compiler subprocess. Unreachable nodes
and constants are removed, single-use elementwise/convolution + ReLU is fused, activation slots are reused by
liveness, and weights and immutable metadata are packed into one shared device allocation per engine and CUDA context.
Convolution, bias and a single-use ReLU share one output kernel. Bindings cannot overlap. A context serializes reuse of its workspace across different streams, and its destructor
waits for outstanding work before freeing owned buffers. The CUDA context must outlive its prepared engines and execution contexts.

ONNX import supports standard-domain opsets 13–22, IR versions 4–13, static f32 I/O, embedded f32 weights, and
constant i64 reshape shapes. Supported operators: `MatMul`, `Gemm`, `Conv`, `Relu`, `Sigmoid`, `Tanh`, `Identity`,
`BatchNormalization` (inference with constant parameters), `Add`, `Mul`, `Sub`, `Div`, binary `Max`/`Min`, `Softmax`, `MaxPool`, `AveragePool`, `GlobalAveragePool`, `Flatten`,
`Reshape`, `Transpose`, `Concat`, and tensor-valued `Constant` nodes. External weights, dynamic shapes, non-f32
execution, custom operators/plugins, control flow, asymmetric/automatic padding, dilated/ceil pooling, and opaque
vendor engines return explicit errors. This operator set is not the entire TensorRT API.

MatMul uses one strided-batched GEMM for matching batches or a shared matrix. Mixed broadcast axes dispatch
individual GEMMs without replicating input data. ONNX vector MatMul promotes/squeezes dimensions through reshape operations (currently device copies). The native
MatMul operation requires matrix operands. ONNX Gemm remains
rank-2 as required by its schema.

Inference BatchNormalization precomputes channel scale/shift at model build time and lowers to existing portable
MLIR multiply/add kernels. A following ReLU is fused into the add. Training mode and dynamic normalization
parameters are rejected; this does not implement training statistics updates.

## Build and run

Use the project's LLVM/MLIR build with the AIR and AGX targets:

```sh
ZLUDA_LLVM_BUILD_DIR=/path/to/llvm-build cargo build --release -p zluda_tensorrt_agx
```

`ZLUDA_XTASK_AGX=1 cargo xtask --release` includes the library in AGX builds. Build artifacts remain in the Cargo
target directory, not in the repository. No NVIDIA headers or libraries are needed to build it.

Compile and run the complete C++ MLP example (CUDA Driver API only):

```sh
c++ -std=c++17 -I zluda_tensorrt_agx/include zluda_tensorrt_agx/examples/native.cpp \
    -L target/release -Wl,-rpath,"$PWD/target/release" -ltensorrt_agx -ldl -o /tmp/ztrt-example
ZLUDA_AGX_LIBCUDA=/absolute/path/to/libcuda.so /tmp/ztrt-example
```

The expected output is `[11, 11, 0, 0]`. The example builds and serializes its own network, reloads the engine,
creates an execution context, binds device buffers and executes inference. The C++ header also supports the
convolution → buildSerializedNetwork → deserializeCudaEngine → setTensorAddress → enqueueV3 workflow.
Unsupported precision flags and invalid layer attributes prevent engine construction rather than silently falling
back to another calculation.

Compile an ONNX model without a GPU:

```sh
target/release/ztrt-compile --format onnx --workspace 1073741824 model.onnx model.ztrt
```

`--format json` accepts the independent JSON graph (see the generated test models in `tests/test_gpu.py`). A native
engine contains `ZTRTAGX2`, a little-endian u64 payload length, a u64 JSON metadata length,
then metadata without weights. Remaining records contain a u32 tensor ID, u64 element count and little-endian
IEEE f32 weight bytes. Duplicate records, invalid IDs/counts, nonfinite weights and truncated payloads are rejected.
The payload is bounded to 256 MiB. The serialized workspace ceiling is preserved and can be tightened at load time.
Legacy independently generated `ZTRTAGX1` JSON engines remain readable. This format contains our graph and
weights, not NVIDIA code. Serialization borrows weights and uses a fixed-size conversion buffer rather than
cloning a model or converting weights to decimal strings.

To emit the math kernels separately as portable AIR bitcode/text and an AGX fat object:

```sh
python3 zluda_tensorrt_agx/tools/build_kernels.py --llvm-build /path/to/llvm-build \
    --output-dir /tmp/inference-kernels --emit both
```

The same MLIR source produces `inference.air`, `inference.air.ll`, and `inference.agxfat`. `--emit air` and `--emit agx`
select individual outputs. These are upstream AIR artifacts; macOS packaging/downgrading is a separate toolchain step.

To package a relocatable SDK containing only our own library, headers, compiler, example and kernel artifacts:

```sh
python3 zluda_tensorrt_agx/tools/package_sdk.py --target-dir target/release \
    --kernels-dir /tmp/inference-kernels --output-dir /tmp/zluda-inference-sdk
```

The package includes a runnable C++ example with a relative rpath, licenses and a checksum manifest.
It uses the existing ZLUDA AGX driver selected by `ZLUDA_AGX_LIBCUDA`.

## Verification

```sh
ZLUDA_LLVM_BUILD_DIR=/path/to/llvm-build cargo test --release -p zluda_tensorrt_agx --lib
export ZTRT_LIBRARY=/absolute/path/to/libtensorrt_agx.so
export ZLUDA_AGX_LIBCUDA=/absolute/path/to/our/libcuda.so
python3 zluda_tensorrt_agx/tests/test_gpu.py
python3 zluda_tensorrt_agx/tests/test_onnx.py
python3 zluda_tensorrt_agx/tests/test_lifecycle.py
```

The Python numerical tests require NumPy. The ONNX fixtures are independently generated from the public schema;
if the optional open-source `onnx` package is installed, the test also runs its checker. No vendor engines or
vendor binaries are used in these tests.

Capture our primitive launches with the driver's `ZLUDA_AGX_CAPTURE` facility, then run the same computations on
the AGX Performance Simulator:

```sh
ZLUDA_AGX_CAPTURE=/tmp/inference-captures \
ZLUDA_AGX_CAPTURE_KERNELS=agx_infer_conv2d,agx_infer_binary,agx_infer_unary,agx_infer_softmax,agx_infer_softmax_simd,agx_infer_pool,agx_infer_transpose,agx_infer_concat \
ZLUDA_AGX_CAPTURE_MAX=1 python3 zluda_tensorrt_agx/tests/test_gpu.py
python3 zluda_tensorrt_agx/tests/test_sim.py /tmp/inference-captures --agx-run /path/to/agx_run
```

`tests/bench.py` measures independent embedded inference, pointwise, GEMM, convolution, and softmax examples. It
reports model-build time, context preparation, end-to-end enqueue/synchronization latency and owned workspace.
It is not a comparison with NVIDIA TensorRT, nor a Geekbench score. Geekbench/llama.cpp/Blender runs that do not
call this new API do not validate this runtime's model importer or engine format.

Immutable weights and kernel metadata are uploaded once per engine and CUDA context, then shared across its
execution contexts. Activation workspace remains private to each execution context. Keep the owning CUDA context
alive until its engines and execution contexts have been destroyed. An execution context retains its plan and
resources if the engine handle is released first.

## Public sources and provenance

The implementations are original project source. The interface subset follows the public
[TensorRT builder/runtime workflow](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/runtime-api.html).
The ONNX reader follows the open [ONNX protobuf schema](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto).
No NVIDIA library has been inspected, disassembled, restored, loaded, or used as an implementation oracle for this
work. Tests use synthetic networks, independently encoded ONNX, CPU references, our AGX driver and the simulator.
