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

    // IEEE binary32 operations (subnormals kept, correctly rounded) whatever the kernel's
    // denormal mode: replaced by air_lower.cpp (lowerIEEEHelperCalls)
    float __zluda_ieee_fadd_f32(float, float);
    float __zluda_ieee_fsub_f32(float, float);
    float __zluda_ieee_fmul_f32(float, float);
    float __zluda_ieee_fdiv_f32(float, float);
    float __zluda_ieee_sqrt_f32(float);
    float __zluda_ieee_fma_f32(float, float, float);

    // ---------------------------------------------------------------------
    // Special registers
    // ---------------------------------------------------------------------
    // Thread-block clusters: a cluster of C CTAs is one threadgroup whose z is C times the CTA's (the AIR lowering's
    // resolveCluster resolves __zluda_metal_cluster_dim to the kernel's .reqnctapercluster, 1 without)
    uint32_t __zluda_metal_cluster_dim(uint32_t dim);
    uint32_t __zluda_metal_explicit_cluster();
    static inline uint32_t cluster_size()
    {
        return __zluda_metal_cluster_dim(0) * __zluda_metal_cluster_dim(1) * __zluda_metal_cluster_dim(2);
    }
    static inline uint32_t cta_ntid_z() { return __zluda_metal_threads_per_threadgroup(2) / cluster_size(); }
    static inline uint32_t cluster_rank() { return __zluda_metal_thread_position_in_threadgroup(2) / cta_ntid_z(); }
    // the CTA's position in its cluster along dim
    static inline uint32_t cluster_cta(uint32_t dim)
    {
        uint32_t r = cluster_rank();
        uint32_t x = __zluda_metal_cluster_dim(0), y = __zluda_metal_cluster_dim(1);
        return dim == 0 ? r % x : dim == 1 ? (r / x) % y : dim == 2 ? r / (x * y) : 0;
    }
    uint32_t FUNC(sreg_tid)(uint8_t member)
    {
        uint32_t t = __zluda_metal_thread_position_in_threadgroup(member);
        return member == 2 ? t % cta_ntid_z() : t;
    }
    uint32_t FUNC(sreg_ntid)(uint8_t member)
    {
        return member == 2 ? cta_ntid_z() : __zluda_metal_threads_per_threadgroup(member);
    }
    uint32_t FUNC(sreg_ctaid)(uint8_t member)
    {
        if (member > 2)
            return __zluda_metal_threadgroup_position_in_grid(member);
        return __zluda_metal_threadgroup_position_in_grid(member) * __zluda_metal_cluster_dim(member) + cluster_cta(member);
    }
    uint32_t FUNC(sreg_nctaid)(uint8_t member)
    {
        if (member > 2)
            return __zluda_metal_threadgroups_per_grid(member);
        return __zluda_metal_threadgroups_per_grid(member) * __zluda_metal_cluster_dim(member);
    }
    uint32_t FUNC(sreg_cluster_ctarank)() { return cluster_rank(); }
    uint32_t FUNC(sreg_cluster_nctarank)() { return cluster_size(); }
    uint32_t FUNC(sreg_clusterid)(uint8_t member) { return member > 2 ? 0 : __zluda_metal_threadgroup_position_in_grid(member); }
    uint32_t FUNC(sreg_nclusterid)(uint8_t member) { return member > 2 ? 1 : __zluda_metal_threadgroups_per_grid(member); }
    uint32_t FUNC(sreg_cluster_ctaid)(uint8_t member) { return cluster_cta(member); }
    uint32_t FUNC(sreg_cluster_nctaid)(uint8_t member) { return member > 2 ? 1 : __zluda_metal_cluster_dim(member); }
    bool FUNC(sreg_is_explicit_cluster)() { return __zluda_metal_explicit_cluster() != 0; }
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

    // Instructions lowered by the fallback parser (ptx_parser/src/helper_table), one file per
    // family. Control words are immediates; after inlining they fold away.
#include "metal_impl/fallback_core.inc"
#include "metal_impl/cvt.inc"
#include "metal_impl/float.inc"
#include "metal_impl/matrix.inc"
#include "metal_impl/memory.inc"
#include "metal_impl/tensor.inc"
#include "metal_impl/warp.inc"
#include "metal_impl/sreg.inc"
#include "metal_impl/texture.inc"
#include "metal_impl/multimem.inc"
}
