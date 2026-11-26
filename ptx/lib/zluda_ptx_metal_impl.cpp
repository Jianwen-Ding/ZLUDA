// Metal (AIR) implementation of the `__zluda_ptx_impl_*` helper functions that
// the PTX translator emits calls to.
//
// This file is the Metal counterpart of zluda_ptx_impl.cpp. It is plain C++
// (no HIP, no vendor headers) and is compiled with upstream clang into LLVM
// bitcode using the same address space numbering as the ZLUDA emitter
// (flat = 0, global = 1, shared = 3, constant = 4, private = 5):
//
//   clang++ -std=c++17 --target=amdgcn-amd-amdhsa -nogpulib -O2 -ffreestanding \
//       -fno-exceptions -emit-llvm -c zluda_ptx_metal_impl.cpp -o zluda_ptx_metal_impl.bc
//
// (zluda_metal/build.rs does this automatically). The amdgcn triple is only
// used to get the address-space layout; no AMD intrinsic is used here. The AIR
// lowering (zluda_metal/cpp/air_lower.cpp) retargets the module to
// air64-apple-macosx and replaces the `__zluda_metal_*` placeholders below by
// Metal kernel builtins (thread_position_in_threadgroup, ...).
//
// Functions starting with `air.` are Metal AIR intrinsics. Their names and
// signatures follow the publicly documented / observable AIR conventions
// (they are what Metal Shading Language builtins lower to); no Apple code or
// header is used.

#include <stdint.h>
#ifndef INT64_MIN
#define INT64_MIN (-9223372036854775807ll - 1)
#define INT64_MAX 9223372036854775807ll
#endif

#define FUNC(NAME) __attribute__((used)) __zluda_ptx_impl_##NAME
#define SHARED_SPACE __attribute__((address_space(3)))

extern "C"
{
    // ---------------------------------------------------------------------
    // Placeholders resolved by the AIR lowering into kernel builtin arguments
    // ---------------------------------------------------------------------
    uint32_t __zluda_metal_thread_position_in_threadgroup(uint32_t dim);
    uint32_t __zluda_metal_threads_per_threadgroup(uint32_t dim);
    uint32_t __zluda_metal_threadgroup_position_in_grid(uint32_t dim);
    uint32_t __zluda_metal_threadgroups_per_grid(uint32_t dim);
    uint32_t __zluda_metal_thread_index_in_simdgroup();

    // ---------------------------------------------------------------------
    // AIR intrinsics
    // ---------------------------------------------------------------------
    // mem_flags: 1 = device, 2 = threadgroup; scope: 1 = threadgroup
    void air_wg_barrier(uint32_t mem_flags, uint32_t scope) __asm("air.wg.barrier");
    void air_simdgroup_barrier(uint32_t mem_flags, uint32_t scope) __asm("air.simdgroup.barrier");
    uint32_t air_simd_shuffle(uint32_t value, uint16_t lane) __asm("air.simd_shuffle.u.i32");
    uint64_t air_simd_ballot(bool value) __asm("air.simd_ballot.i64");
    float air_fast_sqrt(float) __asm("air.fast_sqrt.f32");
    float air_fast_rsqrt(float) __asm("air.fast_rsqrt.f32");
    float air_fast_exp2(float) __asm("air.fast_exp2.f32");
    float air_fast_log2(float) __asm("air.fast_log2.f32");
    float air_fast_tanh(float) __asm("air.fast_tanh.f32");
    float air_sqrt(float) __asm("air.sqrt.f32");
    _Float16 air_tanh_f16(_Float16) __asm("air.tanh.f16");

    // ---------------------------------------------------------------------
    // Special registers
    // ---------------------------------------------------------------------
    uint32_t FUNC(sreg_tid)(uint8_t member) { return __zluda_metal_thread_position_in_threadgroup(member); }
    uint32_t FUNC(sreg_ntid)(uint8_t member) { return __zluda_metal_threads_per_threadgroup(member); }
    uint32_t FUNC(sreg_ctaid)(uint8_t member) { return __zluda_metal_threadgroup_position_in_grid(member); }
    uint32_t FUNC(sreg_nctaid)(uint8_t member) { return __zluda_metal_threadgroups_per_grid(member); }
    uint32_t FUNC(sreg_laneid)() { return __zluda_metal_thread_index_in_simdgroup(); }
    uint32_t FUNC(sreg_lanemask_le)() { return (2U << __zluda_ptx_impl_sreg_laneid()) - 1U; }
    uint32_t FUNC(sreg_lanemask_lt)() { return (1U << __zluda_ptx_impl_sreg_laneid()) - 1U; }
    uint32_t FUNC(sreg_lanemask_ge)() { return (~0U) << __zluda_ptx_impl_sreg_laneid(); }

    // ---------------------------------------------------------------------
    // Synchronization
    // ---------------------------------------------------------------------
    void FUNC(bar_sync)(uint32_t) { air_wg_barrier(3, 1); }

    static uint32_t ballot32(bool value) { return (uint32_t)air_simd_ballot(value); }

    uint32_t FUNC(activemask)() { return ballot32(true); }

    bool FUNC(vote_sync_any_pred)(bool v, uint32_t) { return ballot32(v) != 0; }
    bool FUNC(vote_sync_any_pred_negate)(bool v, uint32_t) { return ballot32(!v) != 0; }
    bool FUNC(vote_any_pred)(bool v) { return ballot32(v) != 0; }
    bool FUNC(vote_any_pred_negate)(bool v) { return ballot32(!v) != 0; }
    bool FUNC(vote_sync_all_pred)(bool v, uint32_t) { return ballot32(v) == ballot32(true); }
    bool FUNC(vote_sync_all_pred_negate)(bool v, uint32_t) { return ballot32(v) == 0; }
    bool FUNC(vote_all_pred)(bool v) { return ballot32(v) == ballot32(true); }
    bool FUNC(vote_all_pred_negate)(bool v) { return ballot32(v) == 0; }
    uint32_t FUNC(vote_sync_ballot_b32)(bool v, uint32_t) { return ballot32(v); }
    uint32_t FUNC(vote_sync_ballot_b32_negate)(bool v, uint32_t) { return ballot32(!v); }
    uint32_t FUNC(vote_ballot_b32)(bool v) { return ballot32(v); }
    uint32_t FUNC(vote_ballot_b32_negate)(bool v) { return ballot32(!v); }

    // shfl.sync: `opts` holds the clamp value (bits 0..4) and the segment mask
    // (bits 8..12), exactly as in PTX. See zluda_ptx_impl.cpp for the details.
    typedef uint32_t ShflSyncResult __attribute__((ext_vector_type(2)));
#define SHFL_SYNC_IMPL(mode, calculate_index, CMP)                                                         \
    ShflSyncResult FUNC(shfl_sync_##mode##_b32_pred)(uint32_t input, int32_t delta, uint32_t opts, uint32_t) \
    {                                                                                                        \
        int32_t section_mask = (opts >> 8) & 0b11111;                                                        \
        int32_t warp_end = opts & 0b11111;                                                                   \
        delta &= 0b11111;                                                                                    \
        int32_t self = (int32_t)__zluda_ptx_impl_sreg_laneid();                                              \
        int32_t subsection = section_mask & self;                                                            \
        int32_t subsection_end = subsection | (~section_mask & warp_end);                                    \
        int32_t idx = calculate_index;                                                                       \
        bool out_of_bounds = idx CMP subsection_end;                                                         \
        if (out_of_bounds)                                                                                   \
            idx = self;                                                                                      \
        uint32_t output = air_simd_shuffle(input, (uint16_t)idx);                                            \
        return ShflSyncResult{output, uint32_t(!out_of_bounds)};                                             \
    }                                                                                                        \
    uint32_t FUNC(shfl_sync_##mode##_b32)(uint32_t input, int32_t delta, uint32_t opts, uint32_t mask)       \
    {                                                                                                        \
        return __zluda_ptx_impl_shfl_sync_##mode##_b32_pred(input, delta, opts, mask).x;                     \
    }
    SHFL_SYNC_IMPL(up, self - delta, <)
    SHFL_SYNC_IMPL(down, self + delta, >)
    SHFL_SYNC_IMPL(bfly, self ^ delta, >)
    SHFL_SYNC_IMPL(idx, (delta & ~section_mask) | subsection, >)

    uint32_t FUNC(match_any_sync_b32)(uint32_t value, uint32_t)
    {
        uint32_t result = 0;
        for (uint16_t lane = 0; lane < 32; lane++)
        {
            uint32_t other = air_simd_shuffle(value, lane);
            if (other == value)
                result |= 1U << lane;
        }
        return result & ballot32(true);
    }

    uint32_t FUNC(match_any_sync_b64)(uint64_t value, uint32_t)
    {
        uint32_t result = 0;
        for (uint16_t lane = 0; lane < 32; lane++)
        {
            uint32_t lo = air_simd_shuffle((uint32_t)value, lane);
            uint32_t hi = air_simd_shuffle((uint32_t)(value >> 32), lane);
            if ((((uint64_t)hi << 32) | lo) == value)
                result |= 1U << lane;
        }
        return result & ballot32(true);
    }

#define REDUX_SYNC_IMPL(name, ptx_type, cpp_type, OP)                                  \
    cpp_type FUNC(redux_sync_##name##_##ptx_type)(cpp_type src, uint32_t)            \
    {                                                                                  \
        uint32_t active = ballot32(true);                                              \
        cpp_type acc = src;                                                            \
        bool first = true;                                                             \
        for (uint16_t lane = 0; lane < 32; lane++)                                     \
        {                                                                              \
            cpp_type other = (cpp_type)air_simd_shuffle((uint32_t)src, lane);          \
            if (!(active & (1U << lane)))                                              \
                continue;                                                              \
            if (first)                                                                 \
            {                                                                          \
                acc = other;                                                           \
                first = false;                                                         \
            }                                                                          \
            else                                                                       \
                acc = OP(acc, other);                                                  \
        }                                                                              \
        return acc;                                                                    \
    }
#define OP_ADD(a, b) ((a) + (b))
#define OP_MIN(a, b) ((a) < (b) ? (a) : (b))
#define OP_MAX(a, b) ((a) > (b) ? (a) : (b))
    REDUX_SYNC_IMPL(add, u32, uint32_t, OP_ADD)
    REDUX_SYNC_IMPL(add, s32, int32_t, OP_ADD)
    REDUX_SYNC_IMPL(min, u32, uint32_t, OP_MIN)
    REDUX_SYNC_IMPL(min, s32, int32_t, OP_MIN)
    REDUX_SYNC_IMPL(max, u32, uint32_t, OP_MAX)
    REDUX_SYNC_IMPL(max, s32, int32_t, OP_MAX)

    // bar.red.{and,or}.pred: a threadgroup-wide reduction through threadgroup memory
    static SHARED_SPACE uint32_t bar_red_scratch;
#define BAR_RED_IMPL(reducer, INIT, COMBINE)                                                     \
    bool FUNC(bar_red_##reducer##_pred)(uint32_t, bool predicate, bool invert_predicate)       \
    {                                                                                            \
        bool p = predicate ^ invert_predicate;                                                   \
        bool leader = __zluda_metal_thread_position_in_threadgroup(0) == 0 &&                    \
                      __zluda_metal_thread_position_in_threadgroup(1) == 0 &&                    \
                      __zluda_metal_thread_position_in_threadgroup(2) == 0;                      \
        if (leader)                                                                              \
            bar_red_scratch = INIT;                                                              \
        air_wg_barrier(2, 1);                                                                    \
        COMBINE;                                                                                 \
        air_wg_barrier(2, 1);                                                                    \
        bool result = bar_red_scratch != 0;                                                      \
        air_wg_barrier(2, 1);                                                                    \
        return result;                                                                           \
    }
    BAR_RED_IMPL(and, 1, if (!p) *(volatile SHARED_SPACE uint32_t *)&bar_red_scratch = 0u)
    BAR_RED_IMPL(or, 0, if (p) *(volatile SHARED_SPACE uint32_t *)&bar_red_scratch = 1u)

    void FUNC(nanosleep_u32)(uint32_t) {}

    void FUNC(__assertfail)(uint64_t, uint64_t, uint32_t, uint64_t, uint64_t) {}

    // printf is not supported by the Metal backend; the arguments are dropped
    int FUNC(vprintf)(const char *, void *) { return 0; }

    // ---------------------------------------------------------------------
    // Bit manipulation (portable)
    // ---------------------------------------------------------------------
    uint32_t FUNC(bfe_u32)(uint32_t base, uint32_t pos_32, uint32_t len_32)
    {
        uint32_t pos = pos_32 & 0xFFU;
        uint32_t len = len_32 & 0xFFU;
        if (pos >= 32 || len == 0)
            return 0;
        if (pos + len >= 32)
            return base >> pos;
        return (base >> pos) & ((1U << len) - 1U);
    }

    uint64_t FUNC(bfe_u64)(uint64_t base, uint32_t pos, uint32_t len)
    {
        if (pos >= 64 || len == 0)
            return 0;
        if (len >= 64 || pos + len >= 64)
            return base >> pos;
        return (base >> pos) & ((1UL << len) - 1UL);
    }

    int32_t FUNC(bfe_s32)(int32_t base, uint32_t pos_32, uint32_t len_32)
    {
        uint32_t pos = pos_32 & 0xFFU;
        uint32_t len = len_32 & 0xFFU;
        if (len == 0)
            return 0;
        if (pos >= 32)
            return (base >> 31);
        if (pos + len >= 32)
            return base >> pos;
        return (int32_t)((uint32_t)base << (32U - pos - len)) >> (32U - len);
    }

    int64_t FUNC(bfe_s64)(int64_t base, uint32_t pos, uint32_t len)
    {
        if (len == 0)
            return 0;
        if (pos >= 64)
            return (base >> 63U);
        if ((uint64_t)pos + len >= 64)
            len = 64 - pos;
        return (int64_t)((uint64_t)base << (64U - pos - len)) >> (64U - len);
    }

    uint32_t FUNC(bfi_b32)(uint32_t insert, uint32_t base, uint32_t pos_32, uint32_t len_32)
    {
        uint32_t pos = pos_32 & 0xFFU;
        uint32_t len = len_32 & 0xFFU;
        if (pos >= 32)
            return base;
        uint32_t mask = len >= 32 ? (UINT32_MAX << pos) : (((1U << len) - 1U) << pos);
        return (~mask & base) | (mask & (insert << pos));
    }

    uint64_t FUNC(bfi_b64)(uint64_t insert, uint64_t base, uint32_t pos, uint32_t len)
    {
        if (pos >= 64)
            return base;
        uint64_t mask = len >= 64 ? (UINT64_MAX << pos) : (((1UL << len) - 1UL) << pos);
        return (~mask & base) | (mask & (insert << pos));
    }

    uint32_t FUNC(bmsk_clamp_b32)(uint32_t position, uint32_t width)
    {
        position &= 0xFF;
        width &= 0xFF;
        if (position >= 32)
            return 0;
        uint32_t end = position + width;
        if (end >= 32)
            return UINT32_MAX << position;
        return ((1U << width) - 1U) << position;
    }

    uint32_t FUNC(prmt_b32)(uint32_t x, uint32_t y, uint32_t s)
    {
        uint64_t input = ((uint64_t)y << 32) | x;
        uint32_t out = 0;
        for (uint32_t i = 0; i < 4; i++)
        {
            uint32_t sel = (s >> (i * 4)) & 0xF;
            uint32_t byte = (uint32_t)(input >> ((sel & 7) * 8)) & 0xFF;
            if (sel & 8)
                byte = (byte & 0x80) ? 0xFF : 0x00;
            out |= byte << (i * 8);
        }
        return out;
    }

#define DP2A(name, RET, A, B, AE, BE, OFF)                                         \
    RET FUNC(dp2a_##name)(A a, B b, RET c)                                          \
    {                                                                               \
        RET acc = c;                                                                \
        for (int i = 0; i < 2; i++)                                                 \
        {                                                                           \
            AE av = (AE)((uint32_t)a >> (16 * i));                                  \
            BE bv = (BE)((uint32_t)b >> (8 * (i + OFF)));                           \
            acc += (RET)((int64_t)av * (int64_t)bv);                                \
        }                                                                           \
        return acc;                                                                 \
    }
    DP2A(lo_u32_u32, uint32_t, uint32_t, uint32_t, uint16_t, uint8_t, 0)
    DP2A(hi_u32_u32, uint32_t, uint32_t, uint32_t, uint16_t, uint8_t, 2)
    DP2A(lo_s32_s32, int32_t, int32_t, int32_t, int16_t, int8_t, 0)
    DP2A(hi_s32_s32, int32_t, int32_t, int32_t, int16_t, int8_t, 2)
    DP2A(lo_s32_u32, int32_t, int32_t, int32_t, int16_t, uint8_t, 0)
    DP2A(hi_s32_u32, int32_t, int32_t, int32_t, int16_t, uint8_t, 2)
    DP2A(lo_u32_s32, int32_t, uint32_t, int32_t, uint16_t, int8_t, 0)
    DP2A(hi_u32_s32, int32_t, uint32_t, int32_t, uint16_t, int8_t, 2)

    // ---------------------------------------------------------------------
    // Floating point
    // ---------------------------------------------------------------------
    float FUNC(sqrt_approx_f32)(float x) { return air_fast_sqrt(x); }
    float FUNC(rsqrt_approx_f32)(float x) { return air_fast_rsqrt(x); }
    float FUNC(rcp_approx_f32)(float x) { return 1.0f / x; }
    float FUNC(ex2_approx_f32)(float x) { return air_fast_exp2(x); }
    float FUNC(lg2_approx_f32)(float x) { return air_fast_log2(x); }
    float FUNC(sqrt_rn_f32)(float x) { return air_sqrt(x); }
    float FUNC(sqrt_rn_ftz_f32)(float x) { return air_sqrt(x); }
    float FUNC(tanh_f32)(float x) { return air_fast_tanh(x); }
    _Float16 FUNC(tanh_f16)(_Float16 x) { return air_tanh_f16(x); }
    float FUNC(div_full_f32)(float a, float b) { return a / b; }

    // The translator splits div.rn into two calls shaped after the AMD
    // division sequence. On Metal we simply carry the operands through and
    // perform an IEEE division in the second part.
    struct DivF32Part1Result
    {
        float fma_4;
        float fma_1;
        float fma_3;
        uint8_t numerator_scaled_flag;
    };
    DivF32Part1Result FUNC(div_f32_part1)(float lhs, float rhs) { return {lhs, rhs, 0.0f, 0}; }
    float FUNC(div_f32_part2)(float x, float y, float, float, float, uint8_t) { return x / y; }

    struct DivF64Part1Result
    {
        double fma4;
        double fma3;
        double mul;
        uint8_t num_scaled;
    };
    DivF64Part1Result FUNC(div_f64_part1)(double x, double y) { return {x, y, 0.0, 0}; }
    double FUNC(div_f64_part2)(double x, double y, double, double, double, uint8_t) { return x / y; }

    // ---------------------------------------------------------------------
    // Instructions lowered by the fallback parser (ptx_parser/src/helper_table.rs).
    // Control words are immediates; after inlining they fold away.
    // Semantics: PTX-formalization's Sail model (sail/ptx/ptx_half.sail, ...).
    // ---------------------------------------------------------------------
    static inline uint32_t lop3_eval(uint32_t a, uint32_t b, uint32_t c, uint32_t lut)
    {
        uint32_t r = 0;
        if (lut & 0x01) r |= ~a & ~b & ~c;
        if (lut & 0x02) r |= ~a & ~b & c;
        if (lut & 0x04) r |= ~a & b & ~c;
        if (lut & 0x08) r |= ~a & b & c;
        if (lut & 0x10) r |= a & ~b & ~c;
        if (lut & 0x20) r |= a & ~b & c;
        if (lut & 0x40) r |= a & b & ~c;
        if (lut & 0x80) r |= a & b & c;
        return r;
    }
    uint32_t FUNC(lop3_b32)(uint32_t a, uint32_t b, uint32_t c, uint32_t lut) { return lop3_eval(a, b, c, lut & 0xFF); }
    typedef uint32_t B32Pred __attribute__((ext_vector_type(2)));
    B32Pred FUNC(lop3_and_b32)(uint32_t a, uint32_t b, uint32_t c, uint32_t lut, bool q, uint32_t negq)
    {
        uint32_t d = lop3_eval(a, b, c, lut & 0xFF);
        bool qq = negq ? !q : q;
        return B32Pred{d, uint32_t((d != 0) && qq)};
    }
    bool FUNC(lop3_and_b32_pred)(uint32_t a, uint32_t b, uint32_t c, uint32_t lut, bool q, uint32_t negq)
    {
        return (lop3_eval(a, b, c, lut & 0xFF) != 0) && (negq ? !q : q);
    }
    bool FUNC(lop3_or_b32_pred)(uint32_t a, uint32_t b, uint32_t c, uint32_t lut, bool q, uint32_t negq)
    {
        return (lop3_eval(a, b, c, lut & 0xFF) != 0) || (negq ? !q : q);
    }
    B32Pred FUNC(lop3_or_b32)(uint32_t a, uint32_t b, uint32_t c, uint32_t lut, bool q, uint32_t negq)
    {
        uint32_t d = lop3_eval(a, b, c, lut & 0xFF);
        bool qq = negq ? !q : q;
        return B32Pred{d, uint32_t((d != 0) || qq)};
    }

    // ---- video ----
    static inline int64_t vext(bool sgn, uint32_t v, int off, int w)
    {
        uint64_t x = ((uint64_t)v >> off) & ((w == 32) ? 0xFFFFFFFFull : ((1ull << w) - 1));
        if (sgn && (x >> (w - 1)) & 1)
            return (int64_t)x - (int64_t)(1ull << w);
        return (int64_t)x;
    }
    static inline int64_t vsel(uint32_t code, bool sgn, uint32_t v)
    {
        switch (code)
        {
        case 1: return vext(sgn, v, 0, 8);
        case 2: return vext(sgn, v, 8, 8);
        case 3: return vext(sgn, v, 16, 8);
        case 4: return vext(sgn, v, 24, 8);
        case 5: return vext(sgn, v, 0, 16);
        case 6: return vext(sgn, v, 16, 16);
        default: return vext(sgn, v, 0, 32);
        }
    }
    static inline int64_t clamp32(int64_t v, bool sgn)
    {
        if (sgn)
            return v < -2147483648ll ? -2147483648ll : v > 2147483647ll ? 2147483647ll : v;
        return v < 0 ? 0 : v > 4294967295ll ? 4294967295ll : v;
    }
    static inline int64_t satw(bool sat, bool sgn, int w, int64_t v)
    {
        if (!sat)
            return v;
        if (sgn)
        {
            int64_t lo = -(1ll << (w - 1)), hi = (1ll << (w - 1)) - 1;
            return v < lo ? lo : v > hi ? hi : v;
        }
        int64_t hi = (w == 32) ? 4294967295ll : (1ll << w) - 1;
        return v < 0 ? 0 : v > hi ? hi : v;
    }
    static inline bool vcmp(uint32_t op, int64_t a, int64_t b)
    {
        switch (op)
        {
        case 0: return a == b;
        case 1: return a != b;
        case 2: return a < b;
        case 3: return a <= b;
        case 4: return a > b;
        default: return a >= b;
        }
    }
    // shift a 33-bit value; results that do not fit in int64 only matter saturated
    static inline int64_t vshift(bool left, bool clampm, int64_t a, int64_t b, bool *big, uint32_t *low)
    {
        uint32_t s = clampm ? (uint32_t)((uint64_t)b & 0xFFFFFFFFull) : (uint32_t)(b & 31);
        if (clampm && s > 32)
            s = 32;
        *big = false;
        *low = left ? (uint32_t)((uint64_t)a << s) : 0;
        if (!left)
            return s >= 63 ? (a < 0 ? -1 : 0) : (a >> s);
        if (a != 0 && s >= 30)
        {
            // |a| < 2^33: a << s may exceed 2^62
            int64_t m = a < 0 ? -a : a;
            if (m >= (1ll << (62 - s)))
            {
                *big = true;
                return a < 0 ? INT64_MIN / 2 : INT64_MAX / 2;
            }
        }
        return a * (1ll << s);
    }
    // ctrl: op 0-3 (vadd vsub vabsdiff vmin vmax vshl vshr, 7 = vset), dt/at/bt signed 4-6, sat 7,
    // secondary op 8-9, asel 10-13, bsel 14-17, clamp 18, merge 19, dsel 20-23, cmp 24-27
    uint32_t FUNC(video_scalar)(uint32_t a, uint32_t b, uint32_t c, uint32_t ctrl)
    {
        uint32_t op = ctrl & 15;
        bool ds = (ctrl >> 4) & 1, as = (ctrl >> 5) & 1, bs = (ctrl >> 6) & 1, sat = (ctrl >> 7) & 1;
        uint32_t sec = (ctrl >> 8) & 3;
        int64_t ta = vsel((ctrl >> 10) & 15, as, a), tb = vsel((ctrl >> 14) & 15, bs, b);
        int64_t v;
        bool big = false;
        uint32_t low = 0;
        switch (op)
        {
        case 0: v = ta + tb; break;
        case 1: v = ta - tb; break;
        case 2: v = ta > tb ? ta - tb : tb - ta; break;
        case 3: v = ta < tb ? ta : tb; break;
        case 4: v = ta > tb ? ta : tb; break;
        case 5: v = vshift(true, (ctrl >> 18) & 1, ta, tb, &big, &low); break;
        case 6: v = vshift(false, (ctrl >> 18) & 1, ta, tb, &big, &low); break;
        default: v = vcmp((ctrl >> 24) & 15, ta, tb); ds = false; sat = false; break;
        }
        if ((ctrl >> 19) & 1)
        {
            // d.dsel = merge(c, saturate-to-lane(v))
            uint32_t dsel = (ctrl >> 20) & 15;
            int w = dsel >= 5 ? 16 : 8;
            int pos = dsel >= 5 ? (dsel - 5) * 16 : (dsel - 1) * 8;
            uint32_t mask = (w == 16 ? 0xFFFFu : 0xFFu);
            uint32_t lane = (uint32_t)((big && !sat) ? (int64_t)low : satw(sat, ds, w, v)) & mask;
            return (c & ~(mask << pos)) | (lane << pos);
        }
        if (sat)
            v = clamp32(v, ds);
        else if (big)
            v = (int64_t)low;
        int64_t cv = ds ? (int64_t)(int32_t)c : (int64_t)c;
        if (op == 7)
            cv = (int64_t)c;
        switch (sec)
        {
        case 1: v = v + cv; break;
        case 2: v = v < cv ? v : cv; break;
        case 3: v = v > cv ? v : cv; break;
        default: break;
        }
        return (uint32_t)v;
    }
    // vmad: ctrl as video_scalar, plus po 18, neg a/b/c 19-21, shift 22-23 (1 = shr7, 2 = shr15)
    uint32_t FUNC(vmad)(uint32_t a, uint32_t b, uint32_t c, uint32_t ctrl)
    {
        bool as = (ctrl >> 5) & 1, bs = (ctrl >> 6) & 1, sat = (ctrl >> 7) & 1, po = (ctrl >> 18) & 1;
        bool na = (ctrl >> 19) & 1, nb = (ctrl >> 20) & 1, nc = (ctrl >> 21) & 1;
        int64_t ta = vsel((ctrl >> 10) & 15, as, a), tb = vsel((ctrl >> 14) & 15, bs, b);
        bool sf = as || bs || (na != nb) || nc;
        int64_t p = ta * tb;
        if (!po && na != nb)
            p = ~p;
        int64_t cr = (int64_t)c;
        int64_t cc = (po || na != nb) ? cr : (nc ? ~cr : cr);
        int64_t l = (po || na != nb || nc) ? 1 : 0;
        int64_t c32 = cc & 0xFFFFFFFFll;
        if (sf && (c32 >> 31))
            c32 -= 0x100000000ll;
        int64_t v = p + c32 + l;
        uint32_t sh = (ctrl >> 22) & 3;
        if (sh == 1)
            v >>= 7;
        else if (sh == 2)
            v >>= 15;
        if (sat)
            v = clamp32(v, sf);
        return (uint32_t)v;
    }
    static inline int64_t velem(bool sgn, uint32_t a, uint32_t b, uint32_t k, int w)
    {
        int n = 32 / w;
        return (int)k < n ? vext(sgn, a, k * w, w) : vext(sgn, b, (k - n) * w, w);
    }
    // ctrl: op 0-2 (vadd vsub vavrg vabsdiff vmin vmax vset), four lanes 3, dt/at/bt signed 4-6, sat 7,
    // accumulate 8, lane mask 10-13, cmp 24-27; sels: a lane digits 0-15, b lane digits 16-31
    uint32_t FUNC(video_simd)(uint32_t a, uint32_t b, uint32_t c, uint32_t ctrl, uint32_t sels)
    {
        uint32_t op = ctrl & 7;
        int n = ((ctrl >> 3) & 1) ? 4 : 2, w = 32 / n;
        bool ds = (ctrl >> 4) & 1, as = (ctrl >> 5) & 1, bs = (ctrl >> 6) & 1, sat = (ctrl >> 7) & 1;
        bool acc = (ctrl >> 8) & 1;
        uint32_t mask = (ctrl >> 10) & 15;
        int64_t sum = 0;
        uint32_t r = 0;
        for (int l = 0; l < n; l++)
        {
            int64_t x = velem(as, a, b, (sels >> (4 * l)) & 15, w);
            int64_t y = velem(bs, a, b, (sels >> (16 + 4 * l)) & 15, w);
            int64_t v;
            switch (op)
            {
            case 0: v = x + y; break;
            case 1: v = x - y; break;
            case 2: v = (x + y >= 0) ? ((x + y + 1) >> 1) : ((x + y) >> 1); break;
            case 3: v = x > y ? x - y : y - x; break;
            case 4: v = x < y ? x : y; break;
            case 5: v = x > y ? x : y; break;
            default: v = vcmp((ctrl >> 24) & 15, x, y); break;
            }
            if (op != 6)
                v = satw(sat, ds, w, v);
            if (acc)
            {
                if ((mask >> l) & 1)
                    sum += v;
            }
            else
            {
                uint32_t lm = (1u << w) - 1;
                uint32_t lv = ((mask >> l) & 1) ? ((uint32_t)v & lm) : ((c >> (l * w)) & lm);
                r |= lv << (l * w);
            }
        }
        return acc ? (uint32_t)((int64_t)c + sum) : r;
    }

    // ---- testp / slct ----
    bool FUNC(testp_f32)(float x, uint32_t k)
    {
        uint32_t u = __builtin_bit_cast(uint32_t, x);
        uint32_t e = (u >> 23) & 0xFF, f = u & 0x7FFFFF;
        switch (k)
        {
        case 0: return e != 0xFF;
        case 1: return e == 0xFF && f == 0;
        case 2: return !(e == 0xFF && f != 0);
        case 3: return e == 0xFF && f != 0;
        case 4: return (e != 0 || f == 0) && e != 0xFF; // zeros are normal (9.7.3.1)
        default: return e == 0 && f != 0;
        }
    }
    bool FUNC(testp_f64)(double x, uint32_t k)
    {
        uint64_t u = __builtin_bit_cast(uint64_t, x);
        uint64_t e = (u >> 52) & 0x7FF, f = u & 0xFFFFFFFFFFFFFull;
        switch (k)
        {
        case 0: return e != 0x7FF;
        case 1: return e == 0x7FF && f == 0;
        case 2: return !(e == 0x7FF && f != 0);
        case 3: return e == 0x7FF && f != 0;
        case 4: return (e != 0 || f == 0) && e != 0x7FF;
        default: return e == 0 && f != 0;
        }
    }
    // c >= 0 on the bits (exact for subnormals, which the hardware's compare flushes);
    // with .ftz a subnormal c is a sign-preserving zero, so it selects a
    static inline bool slct_nonneg(float c, bool ftz)
    {
        uint32_t u = __builtin_bit_cast(uint32_t, c), m = u & 0x7FFFFFFFu;
        if (m > 0x7F800000u)
            return false;
        return (u >> 31) == 0 || m == 0 || (ftz && m < 0x00800000u);
    }
#define SLCT(BITS, T)                                                                                     \
    T FUNC(slct_b##BITS##_s32)(T a, T b, int32_t c) { return c >= 0 ? a : b; }                          \
    T FUNC(slct_b##BITS##_f32)(T a, T b, float c) { return slct_nonneg(c, false) ? a : b; }             \
    T FUNC(slct_b##BITS##_f32_ftz)(T a, T b, float c) { return slct_nonneg(c, true) ? a : b; }
    SLCT(16, uint16_t)
    SLCT(32, uint32_t)
    SLCT(64, uint64_t)
}
