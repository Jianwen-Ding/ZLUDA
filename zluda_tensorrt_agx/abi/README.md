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
Allocator/plugin/DLA/host-code/deferred-loading requests that cannot be supported are rejected
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
complete vendor-reference-header or cross-version ABI conformance test. Plugins, complete error-path routing, complete
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
set. Broader engine/context execution coverage, incremental decoding/device streaming, weight-stripped/refit serialization options, refitting, detailed inspection,
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
context. Strategy 1 remains unsupported. The runtime-config factory path now supports those same two strategies; cache/graph controls remain incomplete.
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


## Runtime configuration objects

`config.cpp` implements the reviewed `VRuntimeConfig` method order, from getPImpl through graph strategy.
`createRuntimeConfig()` returns a caller-owned object associated with its engine. Allocation defaults to
STATIC (0); STATIC and USER_MANAGED (2) are supported. Unsupported/invalid values passed to the void
allocation setter preserve the last reported value but invalidate creation until a supported value is set.
The engine rejects configs belonging to a different engine, unknown implementations, or invalid settings.

`createExecutionContextWithRuntimeConfig()` copies valid settings into a context-owned snapshot. A null config
selects defaults. The caller can change/delete the source config after creation. `getRuntimeConfig()` returns
a borrowed snapshot, which must not be deleted by the caller; creation settings are not live allocation
controls, and changing them after creation does not reallocate context memory. Direct strategy-based creation
still returns no runtime-config object. This explicit snapshot behavior is our current subset contract;
complete reference-header/version-specific behavioral validation remains outstanding.

Dynamic specialization enum values LAZY=0, EAGER=1 and NONE=2 are stored for static-engine configuration
round trips. There are no dynamic profiles or dynamically specialized kernels yet, so this does not add
dynamic-shape execution. Invalid specialization values block subsequent creation until corrected.
Automatic graph capture accepts only DISABLED=0; WHOLE_GRAPH_CAPTURE and unknown values return false.
This does not disable caller-controlled explicit graph capture. Runtime cache creation/attachment is
unsupported and returns null/false with a logger message.

Declarations follow the Apache-2.0 [public dispatch source](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/_nv_infer_impl_8h_source.html)
and [runtime config enums](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/_nv_infer_runtime_8h_source.html).
Independent prefix probes exercise config metadata/getPImpl, allocation changes, invalid configuration
rejection before GPU access, specialization values, cache/graph rejection, cross-engine rejection and 100
ownership cycles. The GPU probe checks default creation, config-selected caller scratch, source-config
mutation/destruction, snapshot persistence, inference results and sequential shared-scratch execution.
Normal and C++ adapter/probe-ASan builds pass. The native backend is not ASan-instrumented.


## Serialization configuration

`createSerializationConfig()` now returns a caller-owned wrapper with the reviewed five-method
`VSerializationConfig` dispatch. Flags default to zero. `setFlags`, `getFlags`, `setFlag`, `clearFlag` and
`getFlag` handle the public enum positions EXCLUDE_WEIGHTS=0, EXCLUDE_LEAN_RUNTIME=1 and INCLUDE_REFIT=2.
Unknown bits or enum positions fail without changing the flag state; shifts are performed only after bounds
checks. Configuration accepts the three defined flags for introspection, but engine serialization rejects
EXCLUDE_WEIGHTS and INCLUDE_REFIT because stripped/refittable engines are not implemented yet.

`serializeWithConfig` supports flags zero or EXCLUDE_LEAN_RUNTIME only. Our independent engine format never
embeds an executable runtime, so exclusion leaves the bytes unchanged. The host-memory result retains its
independent ownership. Unknown config implementations fail rather than being cast to unchecked private data.
This does not add support for proprietary serialized engines, weight refitting or NVIDIA runtime packaging.

The adapted declarations follow the Apache-2.0 [public serialization dispatch](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/_nv_infer_impl_8h_source.html)
and [public flag declarations](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/_nv_infer_runtime_8h_source.html).
The separate ABI-prefix probe tests all three flag positions, negative/out-of-range positions, unknown low
and high bit masks, rejected-state preservation, unknown config rejection, individually unsupported options,
and byte-identical default/exclude-runtime serialization across 100 native-engine round trips. Normal and
C++ adapter/probe-ASan runs pass, including the existing GPU context probe. Full independently pinned
reference-header ABI validation remains incomplete.


## Inspector for independent compiled graphs

`createEngineInspector()` now returns the reviewed seven-method inspector dispatch wrapper. It inspects only
our in-memory planned graph names, using the independent native `ztrtGetLayerName` accessor. It does not open
or inspect binaries, serialized vendor engines, weight bytes or generated GPU code. Planned-step names are
their output tensor names; a fused step uses its final output name. Metadata lookup requires no GPU access.

Engine profiling verbosity is now LAYER_NAMES_ONLY (0), matching the inspector's actual output. ONELINE (0)
returns names separated by newlines. JSON (1) returns `{"Name":"..."}` per layer or
`{"Layers":[{"Name":"..."},...]}` for the engine. Quote, backslash and control characters are escaped;
UTF-8 names remain UTF-8. This is our names-only schema, not a claim of complete detailed NVIDIA tooling
schema compatibility. Layer names can contain line breaks, so JSON is the unambiguous representation.
Invalid layer indices/formats return null and log an error. Returned strings are borrowed until the next
information query on that inspector or its destruction. Inspectors are not concurrently queried/mutated.

An inspector can borrow a context from the same public engine, or detach with null. Unknown implementations
and other-engine contexts are rejected without changing the previous attachment. The engine and any attached
context must outlive the inspector/attachment; detach before destroying an attached context. Error recorders are supported as described below. The public dispatch declarations follow the Apache-2.0
[interface source](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/_nv_infer_impl_8h_source.html).

Independent ABI-prefix tests cover names/text/JSON, invalid indices/formats, output escaping, getPImpl,
unknown context rejection, GPU-context attachment/detachment and cross-engine rejection. The native lifecycle
probe checks fused-step name selection and null/negative/out-of-range access. Normal and C++ adapter/probe-ASan
runs pass, including 100 engine ownership/serialization cycles. No proprietary artifact is used as a test oracle.


## Versioned error recorders and callback ownership

The public runtime-base subset now declares the reviewed ErrorCode values and version-1.0 IErrorRecorder
callback order. Runtime, engine, execution context, inspector and empty plugin registry support recorder
registration/getters. Unknown interface language/kind/version is rejected before taking a reference.
Each registration takes an incRefCount reference; replacement, unregistration and object destruction release
it with decRefCount. Re-registering the same pointer remains balanced. The application owns the recorder
and must keep it alive while references remain outstanding.

Successfully loaded engines inherit the runtime's recorder; new contexts and inspectors inherit the engine's
recorder. Children retain their own registration if the parent changes or is destroyed. The independent
native-engine wrapping extension starts without a recorder until one is attached explicitly.
Callback routing uses a shared lease, with report callbacks and reference callbacks outside the slot mutex.
A report can unregister itself without deadlock or premature release. Concurrent reports and recorder
replacement are supported by the slot; other operations on a single runtime/context follow their existing
thread-safety rules. Borrowed getter results still require caller synchronization against replacement; child creation now
uses an internal pinned lease, as described below. The recorder's callbacks must themselves satisfy the public thread-safety contract.

Existing adapter error-message paths report UNSPECIFIED_ERROR (1) to an attached recorder, otherwise to the
logger. Descriptions are bounded to 127 bytes plus a null terminator. A true fatal return does not make the
failing operation succeed or retry it. Granular per-error classification, every currently silent false/null
return path, configuration-object failures and some construction/destruction failures still require routing
work; complete error-reporting compatibility is not claimed. No asynchronous driver polling was added.

The declarations follow the Apache-2.0 [public recorder source](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/_static/c-api/_nv_infer_runtime_base_8h_source.html).
Probes verify inherited registration through GPU execution, replacement and destruction reference balance,
unknown-version rejection, logger fallback, reentrant self-unregistration, fatal-return behavior, 127-byte
truncation, registry registration, 4000 concurrent reports during 1000 same-recorder replacements, and engine/
inspector lifetime cycles. Ordinary and C++ adapter/probe-ASan runs pass. This is not a ThreadSanitizer result
or a full independently pinned reference-header ABI proof; the native backend remains uninstrumented.


## Classified context validation failures

The context adapter now records failures that previously returned false/-1 silently: null execute bindings,
unknown binding names, wrong input/output roles, invalid inferShapes arrays, invalid shapes/ranks, unsupported
NVTX/debug/listener/communicator settings and missing profiler state. Invalid arguments use INVALID_ARGUMENT
(3); valid shapes incompatible with the static engine and missing I/O/workspace configuration use INVALID_CONFIG
(4); reporting without a supported profiler uses INVALID_STATE (9); unsupported optional features use
UNSUPPORTED_STATE (10). The runtime's nonpositive thread limit also uses INVALID_ARGUMENT.

Validation still occurs before GPU execution or unsupported callback dereferences. Valid null unbinding is
supported. Pure metadata/predicate queries keep their fallback semantics and do not create errors merely
because a queried name is absent. Native-backend errors and some older message paths still use UNSPECIFIED_ERROR;
full per-path classification, configuration-object routing and construction/destruction routing remain unfinished.

The context ABI probe checks exactly one recorder event and its numeric classification for nineteen distinct
validation failures, then performs normal GPU execution/graph replay. It also checks that unknown-name query
fallbacks leave the error count unchanged. Runtime tests check the thread-limit error code. Normal and C++
adapter/probe-ASan probes pass; this does not instrument the native Rust backend or establish full ABI coverage.


## Pin recorder references during inheritance

Runtime-to-engine, engine-to-context (both factory paths) and engine-to-inspector recorder inheritance now
holds a shared lease until the child has acquired its own registration. Previously these paths used the public
borrowed getter, leaving a gap where concurrent parent unregistration could release the last reference before
the child called incRefCount. Inheritance no longer has that gap. Pinning copies the existing lease and does
not allocate a new control block or invoke recorder callbacks under the slot lock. Public getter semantics
remain borrowed; this change does not make arbitrary simultaneous object mutations safe.

A deterministic inspector-creation regression pauses the recorder's interface-validation callback before the
child registration increments its reference, unregisters the parent on the main thread, and verifies that a
reference remains until the child resumes. Both waits have five-second deadlines. Child registration and
destruction then balance the final reference. Normal and C++ adapter/probe-ASan runs pass, including existing
GPU context probes. An isolated source-copy negative control reverting only inspector inheritance to the old
borrowed getter triggers the expected reference-lifetime assertion. No vendor code or artifact is used.


## Independent reference-header verification gate

A real reference-header client and strict snapshot/compile harness are now prepared. The reference client
has not yet compiled or run: a complete consistent public header set is still missing in this environment.
Four offline parser/fail-closed tests pass and do not substitute for that ABI result. See
[reference-header-client.md](../validation/reference-header-client.md) for commands, provenance checks,
current network limitation and the exact test scope. Full reference-header validation remains outstanding.
