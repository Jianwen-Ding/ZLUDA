# User-supplied TensorRT Developer Guide review

Source: `/scratch/TensorRT-Developer-Guide.pdf`, supplied by the user.
SHA-256: `d2a3a00d2c4a16341fa521f63186d749ce6b6701c1fab5d565576c289c3da87e`.
The cover identifies **PG-08540-001_v8.6.1, April 2024**. The document has 201 PDF pages.
It is the TensorRT 8.6.1 Developer Guide, not TensorRT-RTX 1.7.1 headers or an ABI declaration set.
The PDF remains in scratch; no document body, vendor code or executable artifact is copied into the repository.

Reviewed sections, with physical PDF page numbers:

| Section | PDF page | Relevance and current evidence |
| --- | ---: | --- |
| 1.7 API Versioning | 17 | Describes API/ABI versioning separately from serialized-object compatibility. Does not validate our RTX 1.7.1 declarations. |
| Plan serialization portability note | 28 | Describes platform/GPU restrictions for vendor plans. Our custom ZTRTAGX format is independent and does not decode such plans. |
| 5.2 Error Handling and Logging | 33–34 | Describes recorder inheritance, child override and associated-logger fallback. New adapter regression checks those behaviors. |
| 5.3 Runtime memory | 34–35 | Distinguishes engine weights, context-persistent state and scratch. Our shared immutable resources/private or borrowed workspace fit that separation, but our GPU preparation is deferred until context creation. This is not a claim of identical allocation timing. |
| 6.3 Compatibility Checks | 41–42 | Describes plan version and device checks. These rules belong to the vendor format; they do not supply its schema or establish decoder compatibility. |

The new context probe installs a child-specific recorder, triggers an engine error and verifies only the parent
recorder receives it, triggers a context error and verifies only the child recorder receives it, unregisters the
child recorder and checks logger fallback, then restores inheritance and continues normal GPU execution.
Reference counts remain balanced. Normal and C++ adapter/probe-ASan checks pass; the native Rust backend
is not ASan-instrumented. This tests the stated ownership behavior, not a complete TensorRT 8.6 or RTX ABI.

The PDF is useful behavioral guidance. It cannot replace a consistent public RTX header snapshot for compiling
an independent client, or prove builder/network/layer dispatch order and layouts. The reference-header client
gate remains unverified. No proprietary binary, vendor engine or generated vendor artifact was examined.
