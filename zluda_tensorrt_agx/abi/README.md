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
| Public C factory/version exports | Not implemented | Client resolves documented exports and version negotiation works |
| Runtime/engine/context dispatch objects | Not implemented | Full applicable public pImpl/vtable contracts, ownership and error behavior verified |
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
