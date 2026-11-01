# Incremental custom engine loading

The native reader decoder and public-style reader adapter avoid a complete serialized input buffer.
This work uses only our independent ZTRTAGX format and generated fixtures. No proprietary engine is used.

Before removal of the persistent serialized weight cache, on Linux aarch64 fresh-process loading of a generated 32 MiB weight model gave:

| Path | Peak RSS (KiB) | Load seconds |
| --- | ---: | ---: |
| Native buffer API, one preallocated input buffer | 112752 | 0.02728 |
| Native reader API, direct file reads into destination | 80528 | 0.02196 |

The observed reduction was 32224 KiB (28.6%). Timings are one local sample, not a statistically established
speedup. File-cache state and allocator behavior affect results; this is CPU engine loading, not GPU inference.
That version still stored host graph weights and a serialized cache. The initial reader change removed input staging.

Reproduce with our built library, without a GPU or vendor SDK:

```sh
python3 zluda_tensorrt_agx/tests/bench_load.py --library /path/to/libtensorrt_agx.so --weights-mib 32
```

The benchmark generates its own custom engine and compares fresh subprocesses. The buffer mode fills one
preallocated bytearray directly from the file; reader mode uses synchronous file readinto callbacks. Peak RSS
includes the process and shared library. Both modes build the same graph and retain the same native engine.

Validation: 12 Rust unit tests; 99 GPU graph and 109 ONNX model checks; ordinary and C++ adapter/probe-ASan
runtime/metadata/context checks. Decoder tests cover three-byte reads, bit-exact float constants, every
truncated prefix, invalid IDs, duplicate records, nonfinite values, legacy JSON engines, and trailing bytes.
Adapter tests also cover negative/over-reported callback counts, incompatible interfaces, a weight model over
1 MiB, GPU execution, stream reuse and graph capture/replay. Rust backend code is not ASan-instrumented.
The ONNX fixtures do not run the optional official ONNX checker. Full ABI compatibility remains incomplete.


## Removing the persistent serialized weight cache

The engine now retains only serialized header/metadata plus the decoded graph weights. It computes the
exact serialized size once while validating the bounded engine format. Size queries remain constant time.
Serialization copies metadata and writes weights directly into the caller's buffer; little-endian targets
borrow weight bytes without per-float conversion. Big-endian conversion remains available but untested on
big-endian hardware. Public host-memory serialization still owns its returned complete byte buffer.

A subsequent fresh-process run with the same generated 32 MiB fixture gave:

| Path | Peak RSS (KiB) | Load seconds |
| --- | ---: | ---: |
| Buffer API, no serialized weight cache | 80032 | 0.02429 |
| Reader API, no serialized weight cache | 47216 | 0.01764 |

Reader peak RSS fell about 41% relative to the earlier reader result (80528 KiB); the current reader path
uses about 41% less peak RSS than the current buffer path. These are local single samples, not timing
statistics. Metadata, graph planning and caller-owned serialization outputs still require allocations.
Repeated serialization now reads graph weights on demand rather than memcpy from a retained duplicate.

All 13 Rust unit tests pass, including exact size, destination sentinel preservation and undersized writer
rejection. Both ABI probe modes, 100 byte-identical round trips, 99 GPU graph checks and 109 ONNX checks pass.
