// Exhaustive check of the helper library's binary32 fast paths (ptx/lib/metal_impl/float.inc) against its
// exact paths, on the host: every binary32 operand of tanh.approx.f32 (and the other fast paths listed below),
// with the library compiled twice, once as is (namespace fast) and once with -DZLUDA_NO_FAST_PATHS (namespace
// ref). The __zluda_nd_* operations (operands and results claimed never subnormal) are checked both as IEEE
// operations and as flushing ones (subnormal operands and results to zero, as the AGX ALU): the results must
// be the same, which is what makes them safe to emit without !agx.keep-denormals. The __zluda_ieee_* operations
// are IEEE operations, as on AGX.
//
//   clang++ -std=c++17 -O2 -mfma -ffp-contract=off -pthread -I ../../../../ptx/lib/metal_impl fastpath.cpp -o fastpath \
//       -I /usr/lib/gcc/x86_64-linux-gnu/13/include -lquadmath   (GCC's libquadmath: binary128 references)
//   ./fastpath [threads [case|all [stride]]]
//
// (-ffp-contract=off: the library is built by clang with the default contraction, but its operations are
// written as calls or separate statements; the harness must not fuse them either.)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <quadmath.h>
#include <atomic>
#include <thread>
#include <vector>

static thread_local int g_flush; // the __zluda_nd_* operations flush subnormals
static inline float fl(float x)
{
    uint32_t u = __builtin_bit_cast(uint32_t, x);
    return (u & 0x7F800000u) == 0 ? __builtin_bit_cast(float, u & 0x80000000u) : x;
}
static inline float ndr(float x) { return g_flush ? fl(x) : x; }
static inline float ndi(float x) { return g_flush ? fl(x) : x; }
#define ND_PRELUDE                                                                                            \
    static inline float __zluda_ieee_fadd_f32(float a, float b) { return a + b; }                              \
    static inline float __zluda_ieee_fsub_f32(float a, float b) { return a - b; }                              \
    static inline float __zluda_ieee_fmul_f32(float a, float b) { return a * b; }                              \
    static inline float __zluda_ieee_fdiv_f32(float a, float b) { return a / b; }                              \
    static inline float __zluda_ieee_sqrt_f32(float a) { return __builtin_sqrtf(a); }                          \
    static inline float __zluda_ieee_fma_f32(float a, float b, float c) { return __builtin_fmaf(a, b, c); }    \
    static inline float __zluda_nd_fadd_f32(float a, float b) { return ndr(ndi(a) + ndi(b)); }                 \
    static inline float __zluda_nd_fsub_f32(float a, float b) { return ndr(ndi(a) - ndi(b)); }                 \
    static inline float __zluda_nd_fmul_f32(float a, float b) { return ndr(ndi(a) * ndi(b)); }                 \
    static inline float __zluda_nd_fdiv_f32(float a, float b) { return ndr(ndi(a) / ndi(b)); }                 \
    static inline float __zluda_nd_fma_f32(float a, float b, float c) { return ndr(__builtin_fmaf(ndi(a), ndi(b), ndi(c))); }

#define FUNC(N) __zluda_ptx_impl_##N

namespace fast
{
ND_PRELUDE
#include "fallback_core.inc"
#include "cvt.inc"
#include "float.inc"
} // namespace fast
#define ZLUDA_NO_FAST_PATHS
namespace ref
{
ND_PRELUDE
#include "fallback_core.inc"
#include "cvt.inc"
#include "float.inc"
} // namespace ref

// The model's correctly rounded definitions (ptx_math.sail), from binary128 (libquadmath, within an ulp of 2^-112):
// a result is taken only when the binary128 value is farther than 2^-100 (relative) from a binary32 rounding
// boundary; closer cases are counted as undecided (none occur).
static std::atomic<uint64_t> g_undecided{0};
static uint32_t fbits(float f) { return __builtin_bit_cast(uint32_t, f); }
static float rn32q(__float128 q)
{
    float a = (float)(q * (1 - 0x1p-100Q)), b = (float)(q * (1 + 0x1p-100Q));
    if (fbits(a) != fbits(b))
        g_undecided++;
    return (float)q;
}
static uint32_t flushb(uint32_t u) { return (u & 0x7F800000u) == 0 ? u & 0x80000000u : u; }
static uint32_t model_ex2(uint32_t a, uint32_t ctrl)
{
    bool ftz = ctrl & (1 << 12);
    if (ftz)
        a = flushb(a);
    float x = __builtin_bit_cast(float, a);
    if (x != x)
        return 0x7FFFFFFF;
    uint32_t r;
    if (__builtin_isinf(x))
        r = x < 0 ? 0 : 0x7F800000;
    else if (x == 0)
        r = 0x3F800000;
    else if (x == __builtin_floorf(x)) // exact powers of two (rounded below the subnormals)
        r = fbits((float)exp2q((__float128)x));
    else
        r = fbits(rn32q(exp2q((__float128)x)));
    return ftz ? flushb(r) : r;
}
static uint32_t model_lg2(uint32_t a, uint32_t ctrl)
{
    bool ftz = ctrl & (1 << 12);
    if (ftz)
        a = flushb(a);
    float x = __builtin_bit_cast(float, a);
    if (x != x || (x < 0))
        return 0x7FFFFFFF;
    if (x == 0)
        return 0xFF800000;
    if (__builtin_isinf(x))
        return 0x7F800000;
    int e;
    float m = __builtin_frexpf(x, &e);
    if (m == 0.5f) // powers of two: exact
        return fbits((float)(e - 1));
    return fbits(rn32q(log2q((__float128)x)));
}
static uint32_t model_rsqrt(uint32_t a, uint32_t ctrl)
{
    bool ftz = ctrl & (1 << 12);
    if (ftz)
        a = flushb(a);
    float x = __builtin_bit_cast(float, a);
    float s = __builtin_sqrtf(x), r = 1.0f / s;
    if (r != r)
        return 0x7FFFFFFF;
    return ftz ? flushb(fbits(r)) : fbits(r);
}

struct Case
{
    const char *name;
    uint32_t (*fast)(uint32_t, uint32_t);
    uint32_t (*ref)(uint32_t, uint32_t);
    uint32_t ctrl;
    uint32_t (*model)(uint32_t, uint32_t); // when set, fast and ref are both compared with it
};
static const Case cases[] = {
    {"tanh.approx.f32", fast::__zluda_ptx_impl_ftanh_32, ref::__zluda_ptx_impl_ftanh_32, 1 << 4},
    {"tanh.approx.ftz.f32", fast::__zluda_ptx_impl_ftanh_32, ref::__zluda_ptx_impl_ftanh_32, 1 << 4 | 1 << 12},
    {"sin.approx.f32", fast::__zluda_ptx_impl_fsincos_f32, ref::__zluda_ptx_impl_fsincos_f32, 0},
    {"sin.approx.ftz.f32", fast::__zluda_ptx_impl_fsincos_f32, ref::__zluda_ptx_impl_fsincos_f32, 1 << 12},
    {"cos.approx.f32", fast::__zluda_ptx_impl_fsincos_f32, ref::__zluda_ptx_impl_fsincos_f32, 1},
    {"cos.approx.ftz.f32", fast::__zluda_ptx_impl_fsincos_f32, ref::__zluda_ptx_impl_fsincos_f32, 1 | 1 << 12},
    {"rsqrt.approx.f32", fast::__zluda_ptx_impl_frsqrt_f32, ref::__zluda_ptx_impl_frsqrt_f32, 0},
    {"rsqrt.approx.ftz.f32", fast::__zluda_ptx_impl_frsqrt_f32, ref::__zluda_ptx_impl_frsqrt_f32, 1 << 12},
    {"rsqrt.approx.f32/model", fast::__zluda_ptx_impl_frsqrt_f32, ref::__zluda_ptx_impl_frsqrt_f32, 0, model_rsqrt},
    {"rsqrt.approx.ftz.f32/model", fast::__zluda_ptx_impl_frsqrt_f32, ref::__zluda_ptx_impl_frsqrt_f32, 1 << 12, model_rsqrt},
    {"ex2.approx.f32", fast::__zluda_ptx_impl_fex2_f32, ref::__zluda_ptx_impl_fex2_f32, 0, model_ex2},
    {"ex2.approx.ftz.f32", fast::__zluda_ptx_impl_fex2_f32, ref::__zluda_ptx_impl_fex2_f32, 1 << 12, model_ex2},
    {"lg2.approx.f32", fast::__zluda_ptx_impl_flg2_f32, ref::__zluda_ptx_impl_flg2_f32, 0, model_lg2},
    {"lg2.approx.ftz.f32", fast::__zluda_ptx_impl_flg2_f32, ref::__zluda_ptx_impl_flg2_f32, 1 << 12, model_lg2},
};

int main(int argc, char **argv)
{
    unsigned nt = argc > 1 ? atoi(argv[1]) : 2;
    const char *only = argc > 2 && strcmp(argv[2], "all") ? argv[2] : nullptr;
    uint64_t stride = argc > 3 ? strtoull(argv[3], nullptr, 0) : 1; // a sample: every stride-th operand
    int bad = 0;
    for (const Case &c : cases)
    {
        if (only && strcmp(only, c.name))
            continue;
        // per operand: the fast path with IEEE and with flushing __zluda_nd_* operations, the exact path alone
        // (ZLUDA_NO_FAST_PATHS) and, when the case has one, the model
        std::atomic<uint64_t> mism{0}, mismf{0}, refbad{0};
        g_undecided = 0;
        std::vector<std::thread> ts;
        for (unsigned t = 0; t < nt; t++)
            ts.emplace_back([&, t] {
                for (uint64_t i = t * stride; i < (1ull << 32); i += nt * stride)
                {
                    uint32_t a = (uint32_t)i;
                    g_flush = 0;
                    uint32_t f = c.fast(a, c.ctrl), r = c.ref(a, c.ctrl);
                    g_flush = 1;
                    uint32_t ff = c.fast(a, c.ctrl);
                    if (c.model)
                    {
                        uint32_t m = c.model(a, c.ctrl);
                        if (r != m && refbad.fetch_add(1) < 4)
                            printf("  %s %08x: ref %08x model %08x (the exact path alone)\n", c.name, a, r, m);
                        r = m;
                    }
                    if (f != r && mism.fetch_add(1) < 8)
                        printf("  %s %08x: fast %08x %s %08x\n", c.name, a, f, c.model ? "model" : "ref", r);
                    if (ff != r && mismf.fetch_add(1) < 8)
                        printf("  %s %08x: fast (flushing) %08x %s %08x\n", c.name, a, ff, c.model ? "model" : "ref", r);
                }
            });
        for (auto &th : ts)
            th.join();
        printf("%s: %llu mismatches, %llu with flushing nd operations", c.name, (unsigned long long)mism.load(),
               (unsigned long long)mismf.load());
        if (c.model)
            printf(" (against the model; the exact path alone: %llu; undecided %llu)", (unsigned long long)refbad.load(),
                   (unsigned long long)g_undecided.load());
        printf("\n");
        fflush(stdout);
        bad |= mism.load() != 0 || mismf.load() != 0;
    }
    return bad;
}
