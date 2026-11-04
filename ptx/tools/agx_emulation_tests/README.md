# Emulated operations on AGX: exactness tests

AGX has no double precision, no 64-bit division, no wide atomics and no directed float rounding in hardware;
ZLUDA's PTX → AIR → AGX route emulates them (the AGX backend's AGXSoftF64, AGXStrictFP, AGXCodeGenPrepare and
AGXExpandWideAtomics in the LLVM fork). These tests check the emulations bit-exactly, in the
PTX-formalization test format (`.buffer` / `.launch` / `.expect`), with expected values computed in Python
(IEEE doubles, exact rationals, integers), not recorded from any GPU:

- `atomic-uniform.ptx`: 32 threads of one SIMD group on one address: f64 add, u64 add, s64 min and a
  compare-and-swap chain, every returned value and the memory (the lane-order serialization the backend uses);
- `atomic-groups.ptx`: 256 threads in 8 SIMD groups: the final u64 sum, s64 minimum and count;
- `brev64.ptx`: `brev.b64` of 256 values;
- `div64c.ptx`: `div.u64` / `rem.u64` by constants;
- `mulhi64.ptx`: `mul.hi.u64`, `mul.hi.s64` and `mul.lo.u64` of 512 pairs (edges, random, ones-heavy).

`gen_all.sh DIR` writes the larger generated sets (several MB): f64 add / sub / mul, fma and div / sqrt (random,
edge, denormal, overflow and cancellation operands; with `ZEROS=1` 40% zeros), float fma rounded toward zero / up
/ down, the f64 conversions to and from f32 and 32- / 64-bit integers, rounding to integral values (cvt.rmi / rpi / rzi / rni),
and 2048 64-bit div / rem pairs.

    ptx/tools/agx_emulation_tests/gen_all.sh /tmp/emu
    python3 ptx/tools/agx_ptx_formalization_check.py --zoc target/debug/zoc --agx-run .../agx_run --target air \
        [--strict-fp] --flush-ok /tmp/emu/*.ptx ptx/tools/agx_emulation_tests/*.ptx

`--flush-ok` is for the default mode, which flushes float denormal results of `cvt.rn.f32.f64`.
