// Approximate double precision instructions of PTX that have an exact definition: rcp.approx.ftz.f64 and
// rsqrt.approx.ftz.f64 (PTX ISA 9.7.3.13, 9.7.3.16) work on the upper word of the operand (1.11.20), put the result
// in 1.11.20 into the upper word and zero the lower one; subnormal inputs and results are sign-preserving zeros and
// NaN inputs give 0x7fffffff00000000. The result is the exact one rounded to nearest even: the double precision
// quotient (root) is rounded once more, which cannot cross a midpoint of 21 bits.
//
// Compiled to LLVM IR by ptx/lib/gen_agx_tex.py (clang -target nvptx64) into the generated section of
// zluda_ptx_impl_agx.ll; regenerate after editing.

#include <stdint.h>

#define API __attribute__((always_inline))

// The hardware's reciprocal and reciprocal square root (within 1 ulp; gen_agx_tex.py renames the declarations
// to llvm.agx.rcp.f32 / llvm.agx.rsqrt.f32): estimates only, corrected exactly below.
#ifndef HW_RCP
__attribute__((const)) float __zluda_agx_hw_rcp(float);
__attribute__((const)) float __zluda_agx_hw_rsqrt(float);
#define HW_RCP(x) __zluda_agx_hw_rcp(x)
#define HW_RSQRT(x) __zluda_agx_hw_rsqrt(x)
#endif

static uint64_t bits_of(double x) {
    uint64_t b;
    __builtin_memcpy(&b, &x, 8);
    return b;
}

static double of_bits(uint64_t b) {
    double x;
    __builtin_memcpy(&x, &b, 8);
    return x;
}

static const uint64_t SIGN = 1ull << 63, EXP = 0x7ffull << 52, NAN_RESULT = 0x7fffffff00000000ull;

// the upper word, subnormals flushed
static uint64_t operand(uint64_t a) {
    a &= 0xffffffff00000000ull;
    return (a & EXP) == 0 ? a & SIGN : a;
}

// The significands are 21-bit integers, so the result needs no double arithmetic: a float estimate of the 21-bit
// result integer (the hardware rcp / rsqrt, within 1 ulp of the float result), within 1 of the nearest one, corrected by the exact sign of the remainder against the midpoints on
// either side (integers; a midpoint is never the exact result, which would need a power-of-two significand, handled
// apart).

// x, 1.11.20 in the upper word with the sign s and exponent field e; 0 below the normal range
static uint64_t pack(uint64_t s, int32_t e, uint32_t q) {
    return e <= 0 ? s : s | (uint64_t)e << 52 | (uint64_t)(q & 0xfffff) << 32;
}

// RN(2^41 / m) for a 21-bit m, from an estimate within 1 (no midpoint is exact)
static uint32_t quotient(uint32_t m) {
    uint32_t q = (uint32_t)(0x1p41f * HW_RCP((float)m) + 0.5f);
    uint64_t two42 = 1ull << 42;
    q += ((uint64_t)(2 * q + 1) * m < two42) - ((uint64_t)(2 * q - 1) * m > two42);
    return q;
}

// RN(2^31 / sqrt(m)) for a 21- or 22-bit m, likewise
static uint32_t root(uint32_t m) {
    uint32_t q = (uint32_t)(0x1p31f * HW_RSQRT((float)m) + 0.5f);
    unsigned __int128 two64 = (unsigned __int128)1 << 64;
    uint64_t up = 2 * q + 1, down = 2 * q - 1;
    q += ((unsigned __int128)m * (up * up) < two64) - ((unsigned __int128)m * (down * down) > two64);
    return q;
}

// (selects, not branches: the special operands are rare and the SIMD group runs both sides of a branch anyway)
API double __zluda_ptx_impl_rcp_approx_ftz_f64(double x) {
    uint64_t a = operand(bits_of(x));
    uint64_t s = a & SIGN, mag = a & ~SIGN;
    int32_t e = (int32_t)(mag >> 52);
    uint32_t f = (uint32_t)(a >> 32) & 0xfffff;
    // 1/x = (Q / 2^20) 2^(1022 - e), Q = RN(2^41 / (2^20 + f)) in [2^20, 2^21); a power of two exactly
    uint64_t r = f ? pack(s, 2045 - e, quotient(0x100000u | f)) : pack(s, 2046 - e, 0);
    r = mag == EXP ? s : r;
    r = mag == 0 ? s | EXP : r;
    return of_bits(mag > EXP ? NAN_RESULT : r);
}

API double __zluda_ptx_impl_rsqrt_approx_ftz_f64(double x) {
    uint64_t a = operand(bits_of(x));
    uint64_t mag = a & ~SIGN;
    int32_t e = (int32_t)(mag >> 52) - 1023;
    uint32_t f = (uint32_t)(a >> 32) & 0xfffff;
    // x = (M / 2^20) 2^(2k), M = 2^20 + f, doubled for an odd exponent;
    // 1/sqrt(x) = (S / 2^20) 2^(-1 - k), S = RN(2^31 / sqrt(M)) in [2^20, 2^21); 2^-k for M = 2^20
    int32_t k = e >> 1;
    uint32_t m = (0x100000u | f) << (e & 1);
    uint64_t r = m == 0x100000u ? pack(0, 1023 - k, 0) : pack(0, 1022 - k, root(m));
    r = a == EXP ? 0 : r;
    r = mag == 0 ? (a & SIGN) | EXP : r;
    return of_bits(mag > EXP || ((a & SIGN) && mag != 0) ? NAN_RESULT : r);
}
