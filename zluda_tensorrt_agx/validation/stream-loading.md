# Incremental custom engine loading

The native reader decoder and public-style reader adapter avoid a complete serialized input buffer.
This work uses only our independent ZTRTAGX format and generated fixtures. No proprietary engine is used.

On the current Linux aarch64 environment, fresh-process loading of a generated 32 MiB weight model gave:

| Path | Peak RSS (KiB) | Load seconds |
| --- | ---: | ---: |
| Native buffer API, one preallocated input buffer | 112752 | 0.02728 |
| Native reader API, direct file reads into destination | 80528 | 0.02196 |

The observed reduction was 32224 KiB (28.6%). Timings are one local sample, not a statistically established
speedup. File-cache state and allocator behavior affect results; this is CPU engine loading, not GPU inference.
The engine still stores host graph weights and a serialized cache. The new reader removes input staging,
not all duplicated persistent data. Model planning/canonical serialization can still allocate memory.

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
