# Public interface alignment and unfinished ABI work

The requested end state remains a complete independent TensorRT implementation backed by portable MLIR/AIR,
with our custom serializer, performance validation and no proprietary binary inspection. The current library
is not that complete end state. More operators alone do not establish ABI compatibility.

## Public sources

NVIDIA's current TensorRT RTX documentation publishes interface header source marked Apache-2.0:

* [Runtime base types](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/_nv_infer_runtime_base_8h_source.html)
* [Runtime and versioned logger](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/_nv_infer_runtime_8h_source.html)
* [Dispatch interfaces](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/_nv_infer_impl_8h_source.html)
* [Builder/layer enums](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/_nv_infer_8h_source.html)
* [Version macros](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/_nv_infer_version_8h_source.html)

These are public interface sources, not disassembled binaries or a proprietary runtime implementation. No SDK,
NVIDIA shared library or libcudart was installed or used for this work. The local runtime-base subset retains
Apache attribution for adapted public declarations. Other project implementation remains original source.
The documentation's `latest` URLs returned pages labeled both 1.6.1 and 1.7.1 during research. They are not a
pinned release snapshot. Exact target-version compatibility must be established against consistent reference
headers before any ABI claim, especially for Geekbench's documented RTX 1.6 path.

## Concrete corrections

* Public TensorIOMode values are NONE=0, INPUT=1, OUTPUT=2. The C++ bridge translates our existing C values
  (-1/0/1) explicitly. Native C semantics and existing custom engines are unchanged.
* Public elementwise values are SUM=0, PROD=1, MAX=2, MIN=3, SUB=4, DIV=5. The bridge translates these to our
  stable C dispatch codes; other public elementwise values fail explicitly rather than select another operation.
* TF32's public BuilderFlag value is 6. Setting unsupported precision flags still prevents a build.
* Dims aliases Dims64 with eight int64 dimensions. The versioned logger uses APILanguage, InterfaceInfo,
  IVersionedInterface and the v_1_0::ILogger namespace/alias and public callback signatures.

C++ checks cover numeric values, 64-bit structure sizes/field offsets, logger metadata/virtual calls, every
supported elementwise-to-native graph mapping, unsupported operation rejection and input/output role mapping.
The compiled C++ MLP executes through our own driver after these changes. These checks cover our subset and
cannot substitute for an independently compiled full-reference-header client ABI test.

## Required remaining work and completion evidence

| Requirement | Current state | Evidence required for completion |
| --- | --- | --- |
| Public enums, data layouts and callbacks | Initial subset aligned | Complete version-specific interface audit and independent client tests |
| Public C factory/version exports | Isolated 1.7.1 configuration prototype | Client resolves documented exports and version negotiation works |
| Runtime/engine/context dispatch objects | Runtime, engine and static-f32 context prototypes with custom loading and user-managed workspace | Full applicable public pImpl/vtable contracts, ownership and error behavior verified |
| Builder/network/layer dispatch objects | Source wrapper subset only | Reference-header model construction, configuration and destruction work |
| Custom engine serialization | Native ZTRTAGX2 works | Roundtrip through public runtime ABI and independent client, without NVIDIA format assumptions |
| Precision and shape coverage | Primarily static f32 | Required dtypes, shape tensors, profiles and dynamic execution validated |
| Operator and plugin coverage | Documented subset | Public supported-scope inventory with numerical/integration tests; unsupported paths explicitly tracked |
| Performance | Synthetic kernels/models measured | Representative end-to-end integrations plus compiler/runtime time and memory measurements |
| Geekbench TensorRT path | Not established | Public ABI integration and compatible model input, with actual workload correctness evidence |

Public runtime wrapper objects contain a dispatch-interface pointer; the dispatch classes have ordered virtual
methods. Our current own C++ objects do not reproduce these contracts. Merely exporting a factory name or
renaming libtensorrt_agx.so would be insufficient and must not be used to claim compatibility. The public
serialized NVIDIA engine format remains a separate unresolved input-format issue; the user-requested custom
serializer is not a decoder for that format. No opaque vendor engine is used as an oracle.

## Isolated runtime configuration prototype

`runtime.cpp` and `runtime_contract.hpp` implement a separate experimental library named
`libtensorrt_agx_runtime_abi.so`. It exports `createInferRuntime_INTERNAL` and `getInferLibVersion`, with the
reviewed RTX 1.7.1 contract number 10701. It accepts only that number; other versions, including 1.6, are rejected.
This number identifies the prototype's interface target, not a claim that the implementation is NVIDIA TensorRT.
It is not installed under NVIDIA's library names and is not linked into the main inference library yet. The adapter now links against our native library to wrap native engine handles.

The wrapper has a virtual destructor and implementation pointer. The dispatch declarations follow the reviewed
public method order, including getPImpl, configuration, logger and plugin-registry access. Positive thread limits,
temporary-directory copies and tempfile-control flags are implemented. An empty registry reports no creators.
Allocator/error-recorder/plugin/DLA/host-code/deferred-loading requests that cannot be supported are rejected
or logged explicitly. Directory strings remain valid until the next setter; callers must coordinate pointer use
with mutation and keep the logger alive through runtime destruction.

`deserializeCudaEngine` accepts only our independently serialized ZTRTAGX1/ZTRTAGX2 engines.
It checks buffer bounds and the eight-byte format tag before invoking our native decoder; unknown formats
are rejected without decoding their payload. The caller must provide a readable buffer of the declared size.
Header validity checks only our 24-byte custom header and reports the public MALFORMED_ENGINE bit for invalid headers. It does not validate the body or initialize a GPU. Native engines can also enter through the explicit
`ztrtRtxWrapNativeEngine` extension. External runtime and plugin loading do not load files.


Build and run the separate dispatch-prefix probe:

```sh
python3 zluda_tensorrt_agx/tools/build_runtime_abi.py --output-dir /tmp/ztrt-runtime-abi --native-library /path/to/libtensorrt_agx.so
python3 zluda_tensorrt_agx/tools/build_runtime_abi.py --output-dir /tmp/ztrt-runtime-abi-asan --native-library /path/to/libtensorrt_agx.so --sanitize
```

The probe compiles in another translation unit without including the adapter dispatch header. It checks
factory/version negotiation, wrapper/dispatch prefix calls, logger callbacks, configuration persistence, empty
registry behavior, unknown-format rejection, and 1000 destruction cycles. Both ordinary and AddressSanitizer runs
pass on Linux aarch64. This is a targeted binary-prefix test using independently declared call shapes, not a
complete vendor-reference-header or cross-version ABI conformance test. Plugins, error recorders, complete
engine/context interfaces, builder factories and reference-header client integration remain required work.

## Native engine metadata and serialization adapter

`engine.cpp` adds the `ztrtRtxWrapNativeEngine` extension declared in `extensions.h`. It accepts an engine handle
created by our native C API, validates the borrowed versioned logger, and transfers ownership only on success.
No NVIDIA engine bytes enter this path. The returned wrapper has a public-style engine dispatch object for
static f32 metadata queries and custom serialization. Static execution contexts now run native GPU inference through the public-style context dispatch. Unsupported allocation strategies and optional features return explicit failures.

Implemented queries cover I/O names and roles, shapes, static profile 0, f32/device/linear layout, name/layer
count, workspace and serialization. Unknown-name fallback values follow the reviewed public docs: datatype
FLOAT, location DEVICE, format LINEAR, empty format description and NONE I/O role. Component/vector queries
return -1 for the unvectorized linear layout, following the documented contract. The native layer-count query
reports planned graph steps, not a physical GPU launch count. The host-memory adapter owns custom ZTRTAGX2
bytes independently of the engine wrapper. Runtime deserialization now supports these custom bytes through the same native decoder.

The build helper requires our native library (standard target/release path, ZTRT_LIBRARY, or --native-library).
It now runs a second independently declared dispatch-prefix probe. One hundred rounds wrap a native engine,
query metadata (including a by-value 72-byte Dims result), serialize, destroy the engine, deserialize the still
live bytes through our native API, wrap the restored engine and destroy both wrapper and host memory. The fixture includes weights and the
restored serialization is byte-identical. Metadata queries also pass with an unavailable CUDA driver. Failed
wrapping leaves ownership with the caller. Normal and AddressSanitizer builds pass without diagnostics.

These probes still do not compile against a complete independently obtained, version-pinned reference header
set. Broader engine/context execution coverage, incremental decoding/device streaming, serializer options, refitting, inspection,
weight management and complete version-specific compatibility remain required work. The prototype does not
alias libtensorrt_rtx.so and does not establish Geekbench support.


## Static execution-context dispatch

`context.cpp` bridges static f32 engines to GPU execution through the reviewed context dispatch order.
Supported calls include named I/O bindings, fixed shapes and contiguous strides, profile 0, synchronous
`executeV2`, asynchronous `enqueueV3`, and context metadata. The public engine must outlive its contexts.
Dynamic shape changes, allocator/profiler callbacks, auxiliary streams and debug
instrumentation are explicitly unsupported. Runtime deserialization supports only our custom engine format.

Run the GPU probe with our own driver and compiler:

```sh
ZLUDA_AGX_LIBCUDA=/path/to/our/libcuda.so ZLUDA_AGX_ZOC=/path/to/our/zoc \
python3 zluda_tensorrt_agx/tools/build_runtime_abi.py --output-dir /tmp/ztrt-context-abi \
  --native-library /path/to/libtensorrt_agx.so --gpu
```

Add `--sanitize` to instrument the C++ adapter and probes; the linked native Rust backend is not
AddressSanitizer-instrumented. The independent context-prefix probe verifies GPU results, rebinding and
null unbinding, workspace size/alignment/overlap validation, synchronous execution, explicit streams, CUDA graph
capture/replay with changed inputs, and foreign-context restoration during destruction. Capture capability
is conservatively reported only for the same explicit stream after successful warmup with complete bindings.
These tests are not complete version-pinned vendor-header ABI validation or Geekbench compatibility.

A native lifecycle regression also covers a caller synchronizing and destroying a previously used stream
before reusing the execution context. If synchronizing that old stream returns INVALID_HANDLE, the runtime
waits on the owning CUDA context before reusing workspace; other synchronization errors propagate.
The ordinary path still waits only on the previous stream. The full native graph suite passes: 99 GPU graphs,
109 ONNX models, and the stream/context lifecycle probe. Normal and adapter-ASan context probes pass.


## Custom engine runtime loading

The runtime buffer loader now returns public-style engine wrappers directly from our custom serializer.
Native runtime/engine handles are guarded during failure; ownership transfers only after wrapper creation.
The default workspace ceiling is 1 GiB and the serialized ceiling can further restrict it, as in the native API.
No host code or plugin binaries are loaded. This adds no decoder for NVIDIA engine formats.

The engine probe checks 100 byte-identical custom serialization/runtime-load cycles without a GPU,
all truncated prefixes of its fixture, malformed payload/metadata lengths, unknown tags and oversized input.
The GPU probe uses the runtime load path, destroys the runtime and overwrites caller bytes before executing
inference, then checks streams and graph replay. Ordinary and adapter-ASan probes pass. These tests use
only our own JSON model and serializer output. Full reference-header ABI validation remains outstanding.


## Header-only validity

`getEngineHeaderSize()` returns 24. `getEngineValidity()` reads at most those bytes, accepts our current
binary format or legacy JSON format, and checks the nonzero payload length against the 256 MiB ceiling.
For the binary format, metadata must be nonempty and fit inside the declared payload after its length field.
Unknown tags, null/short buffers and invalid lengths return INVALID with MALFORMED_ENGINE (`1ULL << 5`);
accepted headers return VALID with diagnostics zero. A null diagnostics pointer is supported.
No CUDA/device/version requirements are encoded in this portable custom header, so this check makes no
claim about available GPU memory or driver readiness. Actual graph/body validation occurs during loading.

This follows the public [runtime validity contract](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/classnvinfer1_1_1_i_runtime.html)
and the Apache-2.0 [diagnostic enum declaration](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/_nv_infer_runtime_8h_source.html).
A protected-page probe places the 24-byte custom header immediately before inaccessible memory and checks
both exact-header and larger declared sizes without touching the body. Tests cover both format tags,
negative/short/null inputs, unknown versions, oversized payloads and impossible metadata lengths. A real
serialized fixture also proves that a valid header with a corrupt body fails full deserialization. Runtime,
metadata and GPU context probes pass ordinarily and with C++ adapter/probe AddressSanitizer instrumentation.


## Reader-based custom engine loading

`deserializeCudaEngineV2` now accepts the reviewed version 1.0 C++ `IStreamReaderV2` interface.
The reader is borrowed, starts at its current position and is never destroyed or rewound. Only host buffers
and a null CUDA stream are passed to `read`. Requests are capped at 1 MiB; short positive reads are retried.
Zero, negative and over-reported read counts fail. The format tag and declared payload limit are checked before decoding metadata/weights, and the payload remains bounded to 256 MiB. The loader consumes exactly one
engine, leaving trailing stream bytes untouched. Invalid interfaces are rejected before reads.

The native reader decoder now parses bounded metadata and reads each binary weight record directly into its
final host vector. It avoids staging the full serialized engine. The native engine caches only header/metadata and an exact serialized size; weights are written directly
from the immutable graph into caller-owned serialization buffers. GPUDirect
and deferred device loading remain unsupported. No NVIDIA engine format is decoded.
The adapted interface declarations follow the Apache-2.0 [public reader declarations](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/_nv_infer_runtime_8h_source.html).

The independently declared reader probe checks 1-byte/7-byte/large reads, every truncated prefix of a real
custom fixture, reader errors and invalid counts, unknown format tags rejected after the 16-byte prefix, version rejection before reading,
trailing bytes, and a real weight payload larger than 1 MiB. Serialization remains byte-identical after stream
loading. The GPU context probe now loads through seven-byte reader calls and still checks inference, streams
and CUDA graph replay after caller data is overwritten. Normal and adapter/probe-ASan checks pass.


`ztrtDeserializeEngineReader` exposes the same incremental decoder through the independent native C ABI.
Callbacks are synchronous and borrowed, must not unwind or write beyond the destination, and return bytes
read (zero EOF, negative error). Metadata parsing uses a bounded buffered reader; the payload limiter prevents
consuming trailing bytes. Binary float weights preserve their little-endian bit patterns, with byte swapping on
big-endian targets. The current platform is little-endian; the big-endian path is not hardware-tested.
See [stream loading measurements](../validation/stream-loading.md) for the memory benchmark and limits.


## Caller-managed execution workspace

Static strategy 0 and user-managed strategy 2 now create GPU execution contexts. Strategy 2 omits activation
workspace allocation; its immutable kernel/constant resources are still prepared and shared per engine/CUDA
context. Strategy 1 remains unsupported. The runtime-config factory path remains incomplete.
`setDeviceMemoryV2` connects caller scratch storage to the native context. Legacy `setDeviceMemory` uses the
engine's required size, so the caller must ensure the actual allocation is large enough. Scratch must be
256-byte aligned and at least the reported workspace size. Negative sizes, invalid alignment and insufficient
sizes reject configuration; enqueue then fails until a successful setter call. Null/zero unbinds scratch, so
nonzero-workspace engines cannot enqueue until memory is supplied again. Zero-workspace graphs need none.

Switching workspace synchronizes this execution context before releasing any old internally owned buffer.
Caller memory is never freed. I/O overlap with the scratch range is rejected before execution. Callers own
device-pointer validity/lifetimes and must serialize contexts that share scratch. Existing captured graphs
retain their recorded addresses and must be recaptured when their memory bindings change.

The independent C ABI provides `ztrtCreateExecutionContextWithoutDeviceMemory` and `ztrtSetDeviceMemory`.
Its failed setter leaves the previous configuration intact; the public void setter adapter separately blocks
enqueue after a configuration error. All changes require the owning CUDA context to be current.
The adapter probe tests undersized/unaligned storage, overlap rejection, null unbinding, successful execution
with borrowed scratch, strategy-2 creation, and sequential scratch reuse after destroying one context.
Final caller `cuMemFree` succeeds, checking that borrowed scratch was not freed by context destruction.
Normal and C++ adapter/probe-ASan tests pass, as do native lifecycle, 99 GPU graph and 109 ONNX model checks.
