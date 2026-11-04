# Emulated operations on AGX: exactness tests

AGX has no double precision, no 64-bit division, no wide atomics and no directed float rounding in hardware;
ZLUDA's PTX → AIR → AGX route emulates them (the AGX backend's AGXSoftF64, AGXStrictFP, AGXCodeGenPrepare and
AGXExpandWideAtomics in the LLVM fork). These tests check the emulations bit-exactly, in the
PTX-formalization test format (`.buffer` / `.launch` / `.expect`), with expected values computed in Python
(IEEE doubles, exact rationals, integers), not recorded from any GPU:

- `atomic-uniform.ptx`: 32 threads of one SIMD group on one address: f64 add, u64 add, s64 min and a
  compare-and-swap chain, every returned value and the memory (the lane-order serialization the backend uses);
- `atomic-incdec.ptx`: `atom.inc.u32` / `atom.dec.u32` on one address: 32 threads with per-thread limits (each
  returned value, in lane order) and 256 threads with one limit (the final value);
- `atomic-16bit.ptx`: `atom.add.noftz.f16` and `atom.cas.b16` counters on both halves of one word from one SIMD
  group (the returned values in lane order, the final word);
- `atomic-groups.ptx`: 256 threads in 8 SIMD groups: the final u64 sum, s64 minimum and count;
- `div16.ptx`: `div` / `rem` of s8, u8, s16 and u16 values, 1,024 pairs each (edges, random, small and zero
  divisors);
- `div16c.ptx` / `div64k.ptx`: 16- and 64-bit `div` / `rem` by constants (7 and 15 divisors);
- `brev64.ptx`: `brev.b64` of 256 values;
- `div64c.ptx`: `div.u64` / `rem.u64` by constants;
- `mulhi64.ptx`: `mul.hi.u64`, `mul.hi.s64` and `mul.lo.u64` of 512 pairs (edges, random, ones-heavy).
- `div32.ptx`: `div` / `rem` `.u32` / `.s32` of 512 pairs (all pairs of edge values, random), division by 0 and -1 as
  ZLUDA defines them;
- `lop3.ptx`: `lop3.b32` with each of the 256 constant tables on 128 operand triples;
- `shf.ptx`: `shf.l` / `shf.r` `.wrap` / `.clamp` of 256 pairs with every shift amount 0..63;
- `f16_directed.ptx`: `cvt.{rz,rm,rp,rn}.f16.f32` of 4,096 floats (half boundaries and midpoints ±1 float ulp,
  half and float denormals, overflow edges, infinities, random).

`strict/` holds tests of exactness that only `--strict-fp` promises (the default mode rounds bfloat fma twice, through
float): `strict/bf16_fma.ptx`, `fma.rn.bf16x2` of 4,096 bfloat triples (realistic, subnormal, edge exponents,
cancellation; `gen/bf16_fma.py`).

`gen_all.sh DIR` writes the larger generated sets (several MB): f64 add / sub / mul, fma and div / sqrt (random,
edge, denormal, overflow and cancellation operands; with `ZEROS=1` 40% zeros), float fma rounded toward zero / up
/ down, float add / sub / mul in those directions (operands around the fast paths' bounds 2^-102, 2^-70, 2^126,
cancellations, exact zero sums), the f64 conversions to and from f32 and 32- / 64-bit integers, rounding to integral values (cvt.rmi / rpi / rzi / rni),
and 2048 64-bit div / rem pairs.

    ptx/tools/agx_emulation_tests/gen_all.sh /tmp/emu
    python3 ptx/tools/agx_ptx_formalization_check.py --zoc target/debug/zoc --agx-run .../agx_run --target air \
        [--strict-fp] --flush-ok /tmp/emu/*.ptx ptx/tools/agx_emulation_tests/*.ptx

`--flush-ok` is for the default mode, which flushes float denormal results of `cvt.rn.f32.f64`.
