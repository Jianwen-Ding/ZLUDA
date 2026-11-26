// Included inside extern "C" of zluda_ptx_metal_impl.cpp: helpers of ptx_parser/src/helper_table/matrix.rs.
//
// Warp-level matrix instructions (mma, mma.sp, block-scaled mma, wmma, ldmatrix, stmatrix,
// movmatrix), following PTX-formalization's Sail model (sail/ptx/ptx_mma.sail, ptx_ldm.sail).
// A warp is one AGX SIMD-group: every lane computes its own result elements, reading the
// elements of A and B it needs from the lanes that hold them (simd shuffles through the inverse
// of the fragment layouts). Float products and sums are exact (a fixed-point accumulator held in
// registers) and rounded once into the accumulator format; .f64 is a chain of exactly rounded
// f64 fmas. Control words (see matrix.rs) are immediates and fold after inlining.
//
// No private arrays are indexed dynamically: llvm-air2agx cannot encode a stack-frame offset
// above 255 that it folds into an ALU immediate (see the matrix family report), so all loops over
// register arrays are fully unrolled and vector elements are picked with select chains.

#define MX_INL static inline __attribute__((always_inline))
#define MX_GLOBAL __attribute__((address_space(1)))
#define MX_UNROLL _Pragma("clang loop unroll(full)")
    typedef uint32_t mx_u32x8 __attribute__((ext_vector_type(8)));
    typedef uint32_t mx_u32x4 __attribute__((ext_vector_type(4)));
    typedef uint32_t mx_u32x2 __attribute__((ext_vector_type(2)));
    typedef uint32_t mx_u32x1 __attribute__((ext_vector_type(1)));
    typedef uint64_t mx_u64x8 __attribute__((ext_vector_type(8)));
    typedef uint64_t mx_u64x4 __attribute__((ext_vector_type(4)));
    typedef float mx_f32x4 __attribute__((ext_vector_type(4)));

    MX_INL int mx_lane() { return (int)__zluda_metal_thread_index_in_simdgroup(); }
    MX_INL uint32_t mx_shfl(uint32_t v, int lane) { return air_simd_shuffle(v, (uint16_t)(lane & 31)); }
    MX_INL uint64_t mx_shfl64(uint64_t v, int lane)
    {
        uint32_t lo = air_simd_shuffle((uint32_t)v, (uint16_t)(lane & 31));
        uint32_t hi = air_simd_shuffle((uint32_t)(v >> 32), (uint16_t)(lane & 31));
        return ((uint64_t)hi << 32) | lo;
    }
    MX_INL uint32_t mx_pick8(mx_u32x8 v, int i)
    {
        uint32_t r = v[0];
        MX_UNROLL for (int j = 1; j < 8; j++) r = (i == j) ? v[j] : r;
        return r;
    }
    MX_INL mx_u32x8 mx_put8(mx_u32x8 v, int i, uint32_t x)
    {
        MX_UNROLL for (int j = 0; j < 8; j++) v[j] = (i == j) ? x : v[j];
        return v;
    }
    MX_INL uint32_t mx_pick4(mx_u32x4 v, int i)
    {
        uint32_t r = v[0];
        MX_UNROLL for (int j = 1; j < 4; j++) r = (i == j) ? v[j] : r;
        return r;
    }
    MX_INL uint64_t mx_pick8_64(mx_u64x8 v, int i)
    {
        uint64_t r = v[0];
        MX_UNROLL for (int j = 1; j < 8; j++) r = (i == j) ? v[j] : r;
        return r;
    }
    MX_INL uint64_t mx_pick4_64(mx_u64x4 v, int i)
    {
        uint64_t r = v[0];
        MX_UNROLL for (int j = 1; j < 4; j++) r = (i == j) ? v[j] : r;
        return r;
    }
    // register `reg` (per lane) of lane `lane`: every one of the first n registers is shuffled
    MX_INL uint32_t mx_fetch8(mx_u32x8 v, int n, int lane, int reg)
    {
        uint32_t r = 0;
        MX_UNROLL for (int j = 0; j < 8; j++)
        {
            if (j < n)
            {
                uint32_t x = mx_shfl(v[j], lane);
                r = (reg == j) ? x : r;
            }
        }
        return r;
    }
    MX_INL uint64_t mx_fetch8_64(mx_u64x8 v, int n, int lane, int reg)
    {
        uint64_t r = 0;
        MX_UNROLL for (int j = 0; j < 8; j++)
        {
            if (j < n)
            {
                uint64_t x = mx_shfl64(v[j], lane);
                r = (reg == j) ? x : r;
            }
        }
        return r;
    }
    MX_INL uint64_t mx_fetch4_64(mx_u64x4 v, int n, int lane, int reg)
    {
        uint64_t r = 0;
        MX_UNROLL for (int j = 0; j < 4; j++)
        {
            if (j < n)
            {
                uint64_t x = mx_shfl64(v[j], lane);
                r = (reg == j) ? x : r;
            }
        }
        return r;
    }

    // ---------------------------------------------------------------- configuration
    // element type codes (matrix.rs ab_code); accumulator codes: f16 0, f32 1, s32 2, f64 3
    enum
    {
        MX_F16 = 0, MX_BF16 = 1, MX_TF32 = 2, MX_E4M3 = 3, MX_E5M2 = 4, MX_E2M3 = 5, MX_E3M2 = 6,
        MX_E2M1 = 7, MX_E2M1P = 8, MX_S8 = 9, MX_U8 = 10, MX_S4 = 11, MX_U4 = 12, MX_B1 = 13, MX_F64 = 15
    };
    // kinds: 0 dense mma, 1 m8n8k4 .f16 (quad pairs), 2 wmma, 3 sparse, 4 block scaled, 5 sparse block scaled
    struct MxCfg
    {
        int K, M, N, at, bt, kind;
        int ct, dt, rnd, sat, bop, acol, bcol, nb, e43, bidA, tidA, bidB, tidB, sel;
        int na, nbr, nc, nd;
    };
    MX_INL MxCfg mx_cfg(uint32_t c0, uint32_t c1, uint32_t c2)
    {
        MxCfg c;
        c.K = c0 & 511;
        c.M = (c0 >> 9) & 63;
        c.N = (c0 >> 15) & 63;
        c.at = (c0 >> 21) & 15;
        c.bt = (c0 >> 25) & 15;
        c.kind = (c0 >> 29) & 7;
        c.ct = c1 & 3;
        c.dt = (c1 >> 2) & 3;
        c.rnd = (c1 >> 4) & 3;
        c.sat = (c1 >> 6) & 1;
        c.bop = (c1 >> 7) & 3;
        c.acol = (c1 >> 9) & 1;
        c.bcol = (c1 >> 10) & 1;
        c.nb = (c1 >> 11) & 7;
        c.e43 = (c1 >> 14) & 1;
        c.bidA = (c1 >> 15) & 3;
        c.tidA = (c1 >> 17) & 3;
        c.bidB = (c1 >> 19) & 3;
        c.tidB = (c1 >> 21) & 3;
        c.sel = (c1 >> 23) & 3;
        c.na = c2 & 15;
        c.nbr = (c2 >> 4) & 15;
        c.nc = (c2 >> 8) & 15;
        c.nd = (c2 >> 12) & 15;
        return c;
    }
    MX_INL int mx_ew(int t)
    {
        switch (t)
        {
        case MX_F16:
        case MX_BF16: return 16;
        case MX_TF32: return 32;
        case MX_E2M1P:
        case MX_S4:
        case MX_U4: return 4;
        case MX_B1: return 1;
        case MX_F64: return 64;
        default: return 8;
        }
    }
    MX_INL bool mx_isint(int t) { return t >= MX_S8 && t <= MX_B1; }
    MX_INL int mx_cdew(int t) { return t == 0 ? 16 : t == 3 ? 64 : 32; }
    MX_INL uint32_t mx_mask(int ew) { return ew >= 32 ? 0xFFFFFFFFu : ((1u << ew) - 1u); }

    // ---------------------------------------------------------------- fragment layouts
    // The accumulator (posC of ptx_mma.sail)
    MX_INL void mx_posC(int M, int l, int i, int &r, int &c)
    {
        if (M == 8) { r = l >> 2; c = 2 * (l & 3) + i; }
        else { r = (l >> 2) + 8 * (i >> 1); c = 2 * (l & 3) + (i & 1); }
    }
    // m8n8k4 .f16 accumulator (Figures 50-52), in the 32x32 frame of the four quad-pair products
    MX_INL void mx_posC884(int ew, int l, int i, int &r, int &c)
    {
        int o = 8 * ((l & 15) >> 2), h = 4 * (l >> 4);
        if (ew == 16) { r = o + (l & 3) + h; c = o + i; }
        else { r = o + (l & 1) + (i & 2) + h; c = o + (i & 4) + (l & 2) + (i & 1); }
    }
    // the lane of m8n8k4 holding element (r, c) of C of element width ew, and its index
    MX_INL void mx_invC884(int ew, int r, int c, int &lane, int &i)
    {
        int Q = r >> 3, rr = r & 7, cc = c & 7;
        if (ew == 16) { lane = 16 * (rr >> 2) + 4 * Q + (rr & 3); i = cc; }
        else { lane = 16 * (rr >> 2) + 4 * Q + (cc & 2) + (rr & 1); i = (cc & 4) | (rr & 2) | (cc & 1); }
    }
    // wmma: element i of lane l holds element (l * nel + i) mod size, row-major
    MX_INL void mx_wmPos(int rows, int cols, int nel, int l, int i, int &r, int &c)
    {
        int x = (l * nel + i) % (rows * cols);
        r = x / cols;
        c = x % cols;
    }
    // the last lane (the one whose value the model keeps) holding wmma element x, and its index
    MX_INL void mx_wmInv(int size, int nel, int x, int &lane, int &i)
    {
        int p = size / nel; // lanes between two copies
        int l0 = (x / nel) % p;
        lane = l0 + p * ((31 - l0) / p);
        i = x % nel;
    }
    // Dense mma A / B: the 32-bit unit w of row r of A (column c of B) is register `reg` of `lane`
    // (checked against posA / posB of ptx_mma.sail for every shape and element width)
    MX_INL void mx_invA(int M, int r, int w, int &lane, int &reg)
    {
        lane = 4 * (r & 7) + (w & 3);
        reg = M == 16 ? 2 * (w >> 2) + (r >> 3) : (w >> 2);
    }
    MX_INL void mx_invB(int c, int w, int &lane, int &reg)
    {
        lane = 4 * c + (w & 3);
        reg = w >> 2;
    }

    // ---------------------------------------------------------------- exact float values
    struct MxF
    {
        int cls; // 0 finite, 1 infinity, 2 NaN
        bool s;
        uint32_t m;
        int e;
    };
    MX_INL MxF mx_fin(bool s, uint32_t m, int e) { MxF f; f.cls = 0; f.s = s; f.m = m; f.e = e; return f; }
    MX_INL MxF mx_nan() { MxF f; f.cls = 2; f.s = false; f.m = 0; f.e = 0; return f; }
    MX_INL MxF mx_inf(bool s) { MxF f; f.cls = 1; f.s = s; f.m = 0; f.e = 0; return f; }
    MX_INL MxF mx_decIEEE(uint32_t b, int p, int eb, int w)
    {
        bool s = (b >> (w - 1)) & 1;
        uint32_t E = (b >> (p - 1)) & ((1u << eb) - 1u);
        uint32_t f = b & ((1u << (p - 1)) - 1u);
        int bias = (1 << (eb - 1)) - 1;
        if (E == (1u << eb) - 1u)
            return f == 0 ? mx_inf(s) : mx_nan();
        if (E == 0)
            return mx_fin(s, f, 1 - bias - (p - 1));
        return mx_fin(s, f | (1u << (p - 1)), (int)E - bias - (p - 1));
    }
    MX_INL MxF mx_decMini(uint32_t c, bool s, int mb, int bias)
    {
        uint32_t hi = c >> mb, m = c & ((1u << mb) - 1u);
        if (hi == 0)
            return mx_fin(s, m, 1 - bias - mb);
        return mx_fin(s, m + (1u << mb), (int)hi - bias - mb);
    }
    // elemF of ptx_mma.sail (the mini formats decode exactly through f16)
    MX_INL MxF mx_elemF(int t, uint32_t b)
    {
        switch (t)
        {
        case MX_F16: return mx_decIEEE(b, 11, 5, 16);
        case MX_BF16: return mx_decIEEE(b, 8, 8, 16);
        case MX_TF32:
        {
            MxF f = mx_decIEEE(b & ~8191u, 24, 8, 32);
            if (f.cls == 0) { f.m >>= 13; f.e += 13; }
            return f;
        }
        case MX_E4M3:
            if ((b & 127) == 127) return mx_nan();
            return mx_decMini(b & 127, (b >> 7) & 1, 3, 7);
        case MX_E5M2:
            if ((b & 127) > 124) return mx_nan();
            if ((b & 127) == 124) return mx_inf((b >> 7) & 1);
            return mx_decMini(b & 127, (b >> 7) & 1, 2, 15);
        case MX_E2M3: return mx_decMini(b & 31, (b >> 5) & 1, 3, 1);
        case MX_E3M2: return mx_decMini(b & 31, (b >> 5) & 1, 2, 3);
        case MX_E2M1: { uint32_t x = (b >> 2) & 15; return mx_decMini(x & 7, x >> 3, 1, 1); }
        case MX_E2M1P: { uint32_t x = b & 15; return mx_decMini(x & 7, x >> 3, 1, 1); }
        default: return mx_fin(false, 0, 0);
        }
    }
    MX_INL MxF mx_mul(MxF a, MxF b)
    {
        if (a.cls == 2 || b.cls == 2) return mx_nan();
        if (a.cls == 1 || b.cls == 1)
        {
            if ((a.cls == 0 && a.m == 0) || (b.cls == 0 && b.m == 0)) return mx_nan();
            return mx_inf(a.s != b.s);
        }
        return mx_fin(a.s != b.s, a.m * b.m, a.e + b.e);
    }
    // block scale factor: ue8m0 (0xFF is NaN) or ue4m3 (ISA 5.2)
    MX_INL MxF mx_scale(int e43, uint32_t x)
    {
        if (!e43) return x == 255 ? mx_nan() : mx_fin(false, 1, (int)x - 127);
        x &= 127;
        if (x == 127) return mx_nan();
        if ((x >> 3) == 0) return mx_fin(false, x & 7, -9);
        return mx_fin(false, (x & 7) + 8, (int)(x >> 3) - 10);
    }

extern "C++"
{
    // ---------------------------------------------------------------- the exact accumulator
    // W words of two's complement, value = w * 2^e0 (+ a fraction in (0, 1) when sticky). All
    // word loops are unrolled, so the words stay in registers.
    template <int W>
    struct MxAcc
    {
        uint32_t w[W];
        int e0;
        bool sticky, nan, pinf, ninf, nz, zset, zsign, zmix;
    };
    template <int W>
    MX_INL void mxa_init(MxAcc<W> &a, int e0)
    {
        MX_UNROLL for (int i = 0; i < W; i++) a.w[i] = 0;
        a.e0 = e0;
        a.sticky = a.nan = a.pinf = a.ninf = a.nz = a.zset = a.zsign = a.zmix = false;
    }
    template <int W>
    MX_INL void mxa_zero(MxAcc<W> &a, bool s)
    {
        if (!a.zset) { a.zset = true; a.zsign = s; }
        else if (a.zsign != s) a.zmix = true;
    }
    // add (-1)^s * m * 2^e exactly (e >= e0, m < 2^32)
    template <int W>
    MX_INL void mxa_add32(MxAcc<W> &a, bool s, uint32_t m, int e)
    {
        if (m == 0) { mxa_zero(a, s); return; }
        a.nz = true;
        int off = e - a.e0, wi = off >> 5, sh = off & 31;
        uint64_t v = (uint64_t)m << sh;
        uint32_t lo = (uint32_t)v, hi = (uint32_t)(v >> 32);
        uint32_t x = s ? 0xFFFFFFFFu : 0u; // the addend, sign-extended (two's complement of a negative)
        uint64_t c = s ? 1 : 0;
        MX_UNROLL for (int j = 0; j < W; j++)
        {
            uint32_t t = j == wi ? lo : j == wi + 1 ? hi : 0u;
            uint32_t addend = j < wi ? (s ? 0xFFFFFFFFu : 0u) : (t ^ x);
            uint64_t r = (uint64_t)a.w[j] + addend + c;
            a.w[j] = (uint32_t)r;
            c = r >> 32;
        }
    }
    template <int W>
    MX_INL void mxa_addF(MxAcc<W> &a, MxF f)
    {
        if (f.cls == 2) a.nan = true;
        else if (f.cls == 1) { if (f.s) a.ninf = true; else a.pinf = true; }
        else mxa_add32(a, f.s, f.m, f.e);
    }
    template <int N>
    MX_INL uint32_t mx_wordAt(const uint32_t (&m)[N], int i)
    {
        uint32_t r = 0;
        MX_UNROLL for (int j = 0; j < N; j++) r = (i == j) ? m[j] : r;
        return r;
    }
    // bits [pos, pos + 32) of the little-endian word array m; pos may be negative
    template <int N>
    MX_INL uint32_t mx_bits32(const uint32_t (&m)[N], int pos)
    {
        if (pos <= -32) return 0;
        if (pos < 0) return m[0] << (-pos);
        int wi = pos >> 5, sh = pos & 31;
        uint32_t lo = mx_wordAt(m, wi), hi = mx_wordAt(m, wi + 1);
        return sh ? (lo >> sh) | (hi << (32 - sh)) : lo;
    }
    template <int N>
    MX_INL bool mx_anybelow(const uint32_t (&m)[N], int n)
    {
        bool any = false;
        MX_UNROLL for (int j = 0; j < N; j++)
        {
            uint32_t w = m[j];
            if (j * 32 >= n) w = 0;
            else if (n < j * 32 + 32) w &= (1u << (n - j * 32)) - 1u;
            any |= w != 0;
        }
        return any;
    }
    // add (-1)^s * m * 2^e for a 4-word magnitude; bits below e0 are truncated (floor) into sticky
    template <int W>
    MX_INL void mxa_add128(MxAcc<W> &a, bool s, const uint32_t (&m)[4], int e)
    {
        if ((m[0] | m[1] | m[2] | m[3]) == 0) { mxa_zero(a, s); return; }
        a.nz = true;
        int off = e - a.e0;
        bool dropped = off < 0 && mx_anybelow(m, -off);
        uint32_t t[W];
        MX_UNROLL for (int j = 0; j < W; j++) t[j] = mx_bits32(m, j * 32 - off);
        if (dropped)
        {
            a.sticky = true;
            if (s) // the floor of a negative value: one more unit
            {
                uint64_t c = 1;
                MX_UNROLL for (int j = 0; j < W; j++)
                {
                    uint64_t r = (uint64_t)t[j] + c;
                    t[j] = (uint32_t)r;
                    c = r >> 32;
                }
            }
        }
        uint64_t c = s ? 1 : 0;
        MX_UNROLL for (int j = 0; j < W; j++)
        {
            uint64_t r = (uint64_t)a.w[j] + (s ? ~t[j] : t[j]) + c;
            a.w[j] = (uint32_t)r;
            c = r >> 32;
        }
    }
    MX_INL uint64_t mx_maxfinite(bool s, int p, int eb, int w)
    {
        uint64_t sb = s ? (1ull << (w - 1)) : 0;
        return sb | ((((1ull << eb) - 2) << (p - 1)) | ((1ull << (p - 1)) - 1));
    }
    // round to precision p / eb exponent bits / w container bits (fround + encExact of ieee.sail);
    // rnd: 0 rn, 1 rz, 2 rm, 3 rp; NaN results are the canonical mask(w - 1)
    template <int W>
    MX_INL uint64_t mxa_round(MxAcc<W> &a, int p, int eb, int w, int rnd, bool satfinite)
    {
        uint64_t sb = 1ull << (w - 1);
        uint64_t emask = (1ull << eb) - 1;
        if (a.nan || (a.pinf && a.ninf)) return sb - 1;
        if (a.pinf || a.ninf)
        {
            if (satfinite) return mx_maxfinite(a.ninf, p, eb, w);
            return (a.ninf ? sb : 0) | (emask << (p - 1));
        }
        bool neg = a.w[W - 1] >> 31;
        uint32_t mag[W];
        {
            uint64_t c = (neg && !a.sticky) ? 1 : 0;
            MX_UNROLL for (int j = 0; j < W; j++)
            {
                uint64_t r = (uint64_t)(neg ? ~a.w[j] : a.w[j]) + c;
                mag[j] = (uint32_t)r;
                c = r >> 32;
            }
        }
        int top = -1;
        uint32_t topw = 0;
        MX_UNROLL for (int j = 0; j < W; j++)
            if (mag[j]) { top = j; topw = mag[j]; }
        if (top < 0 && !a.sticky)
        {
            bool zs = (a.nz || a.zmix) ? (rnd == 2) : a.zsign;
            return zs ? sb : 0;
        }
        int lead = top < 0 ? -1 : top * 32 + 31 - __builtin_clz(topw);
        int bias = (1 << (eb - 1)) - 1, emin = 1 - bias, emax = bias;
        int leadE = lead + a.e0;
        int lsbE = (leadE < emin ? emin : leadE) - (p - 1);
        int sh = lsbE - a.e0;
        uint64_t q;
        bool half = false, below = a.sticky;
        if (sh >= 0)
        {
            q = ((uint64_t)mx_bits32(mag, sh + 32) << 32) | mx_bits32(mag, sh);
            q &= (1ull << p) - 1;
            if (sh >= 1)
            {
                half = (mx_bits32(mag, sh - 1) & 1) != 0;
                below = below || mx_anybelow(mag, sh - 1);
            }
        }
        else
            q = (uint64_t)mag[0] << (-sh);
        bool inexact = half || below;
        bool up = false;
        switch (rnd)
        {
        case 0: up = half && (below || (q & 1)); break;
        case 2: up = inexact && neg; break;
        case 3: up = inexact && !neg; break;
        default: break;
        }
        if (up) q++;
        if (q == (1ull << p)) { q = 1ull << (p - 1); lsbE++; }
        uint64_t sgn = neg ? sb : 0;
        if (q >= (1ull << (p - 1)) && lsbE + p - 1 > emax)
        {
            bool inf = rnd == 0 || (rnd == 3 && !neg) || (rnd == 2 && neg);
            if (inf && !satfinite) return sgn | (emask << (p - 1));
            return mx_maxfinite(neg, p, eb, w);
        }
        if (q < (1ull << (p - 1))) return sgn | q;
        return sgn | ((uint64_t)(lsbE + p - 1 + bias) << (p - 1)) | (q - (1ull << (p - 1)));
    }
}

    // ---------------------------------------------------------------- integer elements
    MX_INL int64_t mx_elemI(int t, uint32_t b)
    {
        switch (t)
        {
        case MX_S8: return (int64_t)(int8_t)(uint8_t)b;
        case MX_U8: return (int64_t)(b & 255);
        case MX_S4: return (int64_t)((int32_t)(b << 28) >> 28);
        case MX_U4: return (int64_t)(b & 15);
        default: return (int64_t)(b & 1);
        }
    }

    // ---------------------------------------------------------------- one result element
    struct MxDot
    {
        int64_t iacc;
        MxAcc<20> f;
    };
    MX_INL void mxd_init(MxDot &d, const MxCfg &cf, uint32_t cv, int ewc)
    {
        d.iacc = (int64_t)(int32_t)cv;
        mxa_init(d.f, -320);
        if (!mx_isint(cf.at))
            mxa_addF(d.f, ewc == 16 ? mx_decIEEE(cv, 11, 5, 16) : mx_decIEEE(cv, 24, 8, 32));
    }
    // one product a * b (* scales)
    MX_INL void mxd_add(MxDot &d, const MxCfg &cf, uint32_t av, uint32_t bv, bool block, uint32_t sa, uint32_t sb)
    {
        if (mx_isint(cf.at))
        {
            if (cf.at == MX_B1) d.iacc += (cf.bop == 2) ? (av & bv & 1) : ((av ^ bv) & 1);
            else d.iacc += mx_elemI(cf.at, av) * mx_elemI(cf.bt, bv);
            return;
        }
        MxF p = mx_mul(mx_elemF(cf.at, av), mx_elemF(cf.bt, bv));
        if (block)
        {
            p = mx_mul(p, mx_scale(cf.e43, sa));
            p = mx_mul(p, mx_scale(cf.e43, sb));
        }
        mxa_addF(d.f, p);
    }
    MX_INL uint32_t mxd_result(MxDot &d, const MxCfg &cf, bool block, int ewd)
    {
        if (mx_isint(cf.at))
        {
            int64_t acc = d.iacc;
            if (cf.sat && cf.at != MX_B1)
                acc = acc < -2147483648ll ? -2147483648ll : acc > 2147483647ll ? 2147483647ll : acc;
            return (uint32_t)acc;
        }
        if (block) return (uint32_t)mxa_round(d.f, 24, 8, 32, 0, false);
        if (ewd == 16) return (uint32_t)mxa_round(d.f, 11, 5, 16, cf.rnd, cf.sat);
        return (uint32_t)mxa_round(d.f, 24, 8, 32, cf.rnd, cf.sat);
    }

    // ---------------------------------------------------------------- the generic mma
    MX_INL mx_u32x8 mx_mma(mx_u32x8 a, mx_u32x8 b, mx_u32x8 c, mx_u32x4 x, uint32_t ctl0, uint32_t ctl1)
    {
        const MxCfg cf = mx_cfg(ctl0, ctl1, x[3]);
        const int l = mx_lane();
        const int M = cf.M, N = cf.N, K = cf.K;
        const int ewa = mx_ew(cf.at), ewb = mx_ew(cf.bt);
        const int pera = 32 / ewa, perb = 32 / ewb;
        const bool m884 = cf.kind == 1, wm = cf.kind == 2;
        const bool sparse = cf.kind == 3 || cf.kind == 5, block = cf.kind == 4 || cf.kind == 5;
        const int ewc = mx_cdew(cf.ct), ewd = mx_cdew(cf.dt);
        const int perc = 32 / ewc, perd = 32 / ewd;
        const int nelC = m884 ? 8 : M * N / 32;
        const uint32_t ma = mx_mask(ewa), mb = mx_mask(ewb);
        // block scales: rows g, g + 8 and columns 2q, 2q + 1 of this lane (mmabs_core)
        uint32_t sar0 = 0, sar1 = 0, sbc0 = 0, sbc1 = 0;
        if (block)
        {
            int g = l >> 2, q = l & 3;
            sar0 = mx_shfl(x[1], g * 4 + 2 * cf.tidA);
            sar1 = mx_shfl(x[1], g * 4 + 2 * cf.tidA + 1);
            sbc0 = mx_shfl(x[2], 8 * q + cf.tidB);
            sbc1 = mx_shfl(x[2], 8 * q + 4 + cf.tidB);
        }
        const int blk = block ? K / (cf.nb ? cf.nb : 1) : 1;
        mx_u32x8 d = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < nelC; i++)
        {
            int r, cc;
            if (m884) mx_posC884(ewd, l, i, r, cc);
            else if (wm) mx_wmPos(M, N, nelC, l, i, r, cc);
            else mx_posC(M, l, i, r, cc);
            uint32_t cv;
            if (m884 && cf.ct != cf.dt)
            {
                int ln, ci;
                mx_invC884(ewc, r, cc, ln, ci);
                cv = (mx_fetch8(c, 8 / perc, ln, ci / perc) >> (ewc * (ci % perc))) & mx_mask(ewc);
            }
            else
                cv = (mx_pick8(c, i / perc) >> (ewc * (i % perc))) & mx_mask(ewc);
            MxDot dot;
            mxd_init(dot, cf, cv, ewc);
            const uint32_t sa = (r >= 8) ? sar1 : sar0, sb = (cc & 1) ? sbc1 : sbc0;
            if (m884)
            {
                int Q = r >> 3, rr = r & 7, Qb = cc >> 3, nn = cc & 7;
                for (int k = 0; k < 4; k++)
                {
                    int la, ia, lb, ib;
                    if (cf.acol) { la = 16 * (rr >> 2) + 4 * Q + k; ia = rr & 3; }
                    else { la = 16 * (rr >> 2) + 4 * Q + (rr & 3); ia = k; }
                    if (cf.bcol) { lb = 16 * (nn >> 2) + 4 * Qb + (nn & 3); ib = k; }
                    else { lb = 16 * (nn >> 2) + 4 * Qb + k; ib = nn & 3; }
                    uint32_t av = (mx_fetch8(a, 2, la, ia >> 1) >> (16 * (ia & 1))) & 0xFFFF;
                    uint32_t bv = (mx_fetch8(b, 2, lb, ib >> 1) >> (16 * (ib & 1))) & 0xFFFF;
                    mxd_add(dot, cf, av, bv, false, 0, 0);
                }
            }
            else if (wm)
            {
                const int nelA = cf.na * pera, nelB = cf.nbr * perb;
                for (int k = 0; k < K; k++)
                {
                    int la, ia, lb, ib;
                    mx_wmInv(M * K, nelA, r * K + k, la, ia);
                    mx_wmInv(K * N, nelB, k * N + cc, lb, ib);
                    uint32_t av = (mx_fetch8(a, cf.na, la, ia / pera) >> (ewa * (ia % pera))) & ma;
                    uint32_t bv = (mx_fetch8(b, cf.nbr, lb, ib / perb) >> (ewb * (ib % perb))) & mb;
                    mxd_add(dot, cf, av, bv, false, 0, 0);
                }
            }
            else if (sparse)
            {
                // A from its compressed values and metadata (spFillAll), B dense
                const int t = cf.at, ewt = ewa;
                const int nz = t == MX_TF32 ? 1 : ewt == 4 ? 4 : 2;
                const int cw = t == MX_TF32 ? 2 : ewt == 4 ? 8 : 4;
                const int nch = K / cw;
                const int g = l >> 2, rh = r >> 3;
                for (int ch = 0; ch < nch; ch++)
                {
                    int tg, u;
                    if (ewt >= 16) { tg = ch & 3; u = 2 * (ch >> 2) + rh; }
                    else { tg = (ch & 7) >> 1; u = (ch & 1) | (rh << 1) | ((ch >> 3) << 2); }
                    int ln, bo;
                    if (nch == 4) { ln = cf.sel; bo = 16 * rh + 4 * ch; }
                    else if (nch == 8 && ewt >= 16) { ln = 2 * cf.sel + (ch >> 2); bo = 16 * rh + 4 * (ch & 3); }
                    else if (nch == 8) { ln = 2 * cf.sel + rh; bo = 4 * ch; }
                    else { ln = 2 * (ch >> 3) + rh; bo = 4 * (ch & 7); }
                    uint32_t nib = (mx_shfl(x[0], 4 * g + ln) >> bo) & 15;
                    int i0 = u * nz;
                    uint32_t areg = mx_shfl(mx_pick8(a, i0 / pera), 4 * g + tg);
                    int sub0 = i0 % pera;
                    for (int p = 0; p < cw; p++)
                    {
                        uint32_t av = 0;
                        for (int j = 0; j < nz; j++)
                        {
                            int sp;
                            if (t == MX_TF32) sp = (int)((nib & 3) >> 1);
                            else if (ewt == 4) sp = 2 * (int)((nib >> (2 * (j >> 1))) & 3) + (j & 1);
                            else sp = (int)((nib >> (2 * j)) & 3);
                            if (sp == p) av = (areg >> (ewa * (sub0 + j))) & ma;
                        }
                        int k = cw * ch + p;
                        int wb = (k * ewb) >> 5, lb, rb;
                        mx_invB(cc, wb, lb, rb);
                        uint32_t bv = (mx_shfl(mx_pick8(b, rb), lb) >> ((k * ewb) & 31)) & mb;
                        int jb = k / blk;
                        mxd_add(dot, cf, av, bv, block, (sa >> (8 * (cf.bidA + jb))) & 255,
                                (sb >> (8 * (cf.bidB + jb))) & 255);
                    }
                }
            }
            else
            {
                // dense: 32-bit units of row r of A and column cc of B
                const int nw = (K * ewa) >> 5;
                for (int w = 0; w < nw; w++)
                {
                    int la, ra, lb, rb;
                    mx_invA(M, r, w, la, ra);
                    uint32_t aw = mx_shfl(mx_pick8(a, ra), la);
                    if (ewa == ewb)
                    {
                        mx_invB(cc, w, lb, rb);
                        uint32_t bw = mx_shfl(mx_pick8(b, rb), lb);
                        if (cf.at == MX_B1)
                        {
                            uint32_t v = (cf.bop == 2) ? (aw & bw) : (aw ^ bw);
                            dot.iacc += __builtin_popcount(v);
                            continue;
                        }
                        for (int j = 0; j < pera; j++)
                        {
                            int k = w * pera + j, jb = k / blk;
                            mxd_add(dot, cf, (aw >> (ewa * j)) & ma, (bw >> (ewb * j)) & mb, block,
                                    (sa >> (8 * (cf.bidA + jb))) & 255, (sb >> (8 * (cf.bidB + jb))) & 255);
                        }
                    }
                    else
                    {
                        for (int j = 0; j < pera; j++)
                        {
                            int k = w * pera + j, jb = k / blk;
                            int wb = (k * ewb) >> 5;
                            mx_invB(cc, wb, lb, rb);
                            uint32_t bv = (mx_shfl(mx_pick8(b, rb), lb) >> ((k * ewb) & 31)) & mb;
                            mxd_add(dot, cf, (aw >> (ewa * j)) & ma, bv, block,
                                    (sa >> (8 * (cf.bidA + jb))) & 255, (sb >> (8 * (cf.bidB + jb))) & 255);
                        }
                    }
                }
            }
            uint32_t res = mxd_result(dot, cf, block, ewd);
            int rg = i / perd;
            d = mx_put8(d, rg, mx_pick8(d, rg) | ((res & mx_mask(ewd)) << (ewd * (i % perd))));
        }
        return d;
    }

    mx_u32x8 FUNC(mma_gen)(mx_u32x8 a, mx_u32x8 b, mx_u32x8 c, mx_u32x4 x, uint32_t ctl0, uint32_t ctl1)
    {
        return mx_mma(a, b, c, x, ctl0, ctl1);
    }

    // ---------------------------------------------------------------- .f64: chains of f64 fmas
    MX_INL void mx_dec64(uint64_t b, int &cls, bool &s, uint64_t &m, int &e)
    {
        s = b >> 63;
        uint32_t E = (b >> 52) & 2047;
        uint64_t f = b & ((1ull << 52) - 1);
        cls = 0;
        if (E == 2047) { cls = f ? 2 : 1; m = 0; e = 0; return; }
        if (E == 0) { m = f; e = -1074; return; }
        m = f | (1ull << 52);
        e = (int)E - 1075;
    }
    // fma(a, b, c) rounded once with the mode (fpFma of ieee.sail; NaN results are canonical)
    MX_INL uint64_t mx_fma64(uint64_t a, uint64_t b, uint64_t c, int rnd)
    {
        int ca, cb, cc, ea, eb, ec;
        bool sa, sb, sc;
        uint64_t ma, mb, mc;
        mx_dec64(a, ca, sa, ma, ea);
        mx_dec64(b, cb, sb, mb, eb);
        mx_dec64(c, cc, sc, mc, ec);
        const uint64_t NaN = 0x7FFFFFFFFFFFFFFFull;
        bool sp = sa != sb;
        if (ca == 2 || cb == 2 || cc == 2) return NaN;
        if (ca == 1 || cb == 1)
        {
            if ((ca == 0 && ma == 0) || (cb == 0 && mb == 0)) return NaN;
            if (cc == 1 && sc != sp) return NaN;
            return (sp ? (1ull << 63) : 0) | 0x7FF0000000000000ull;
        }
        if (cc == 1) return c;
        bool pz = ma == 0 || mb == 0;
        if (pz && mc == 0) return ((sp == sc) ? sp : (rnd == 2)) ? (1ull << 63) : 0;
        if (pz) return c;
        uint32_t pm[4];
        {
            uint64_t a0 = (uint32_t)ma, a1 = ma >> 32, b0 = (uint32_t)mb, b1 = mb >> 32;
            uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
            uint64_t mid = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
            pm[0] = (uint32_t)p00;
            pm[1] = (uint32_t)mid;
            uint64_t hi = (mid >> 32) + (p01 >> 32) + (p10 >> 32) + p11;
            pm[2] = (uint32_t)hi;
            pm[3] = (uint32_t)(hi >> 32);
        }
        int ep = ea + eb;
        int topP = ep + 106, topC = ec + 53;
        int top = topP > topC ? topP : topC;
        MxAcc<10> acc;
        mxa_init(acc, top - 300);
        uint32_t cm[4] = {(uint32_t)mc, (uint32_t)(mc >> 32), 0, 0};
        if (mc != 0) mxa_add128(acc, sc, cm, ec);
        mxa_add128(acc, sp, pm, ep);
        return mxa_round(acc, 53, 11, 64, rnd, false);
    }

    mx_u64x4 FUNC(mma_f64)(mx_u64x8 a, mx_u64x4 b, mx_u64x4 c, uint32_t ctl0, uint32_t ctl1)
    {
        const MxCfg cf = mx_cfg(ctl0, ctl1, 0);
        const int l = mx_lane();
        const int M = cf.M, N = cf.N, K = cf.K;
        const bool wm = cf.kind == 2;
        const int nelA = M * K / 32, nelB = K * N / 32, nelC = M * N / 32;
        mx_u64x4 d = {0, 0, 0, 0};
        for (int i = 0; i < nelC; i++)
        {
            int r, cc;
            if (wm) mx_wmPos(M, N, nelC, l, i, r, cc);
            else mx_posC(M, l, i, r, cc);
            uint64_t acc = mx_pick4_64(c, i);
            for (int k = 0; k < K; k++)
            {
                uint64_t av, bv;
                if (wm)
                {
                    int la, ia, lb, ib;
                    mx_wmInv(M * K, nelA, r * K + k, la, ia);
                    mx_wmInv(K * N, nelB, k * N + cc, lb, ib);
                    av = mx_fetch8_64(a, nelA, la, ia);
                    bv = mx_fetch4_64(b, nelB, lb, ib);
                }
                else
                {
                    int la, ra, lb, rb;
                    mx_invA(M, r, k, la, ra);
                    mx_invB(cc, k, lb, rb);
                    av = mx_shfl64(mx_pick8_64(a, ra), la);
                    bv = mx_shfl64(mx_pick4_64(b, rb), lb);
                }
                acc = mx_fma64(av, bv, acc, cf.rnd);
            }
            MX_UNROLL for (int j = 0; j < 4; j++) d[j] = (i == j) ? acc : d[j];
        }
        return d;
    }

    // ---------------------------------------------------------------- the grammar's mma forms
    MX_INL mx_u32x8 mx_mma_m16n8(mx_u32x4 a4, mx_u32x2 b2, mx_u32x4 c4, uint32_t at, uint32_t k, uint32_t cd)
    {
        mx_u32x8 a = {a4[0], a4[1], a4[2], a4[3], 0, 0, 0, 0};
        mx_u32x8 b = {b2[0], b2[1], 0, 0, 0, 0, 0, 0};
        mx_u32x8 c = {c4[0], c4[1], c4[2], c4[3], 0, 0, 0, 0};
        mx_u32x4 x = {0, 0, 0, 4u | (2u << 4) | (4u << 8) | (4u << 12)};
        uint32_t ctl0 = k | (16u << 9) | (8u << 15) | (at << 21) | (at << 25);
        uint32_t ctl1 = cd | (cd << 2);
        return mx_mma(a, b, c, x, ctl0, ctl1);
    }
    mx_f32x4 FUNC(mma_sync_aligned_m16n8k16_row_col_f32_f16_f16_f32)(mx_u32x4 a, mx_u32x2 b, mx_f32x4 c)
    {
        mx_u32x8 d = mx_mma_m16n8(a, b, __builtin_bit_cast(mx_u32x4, c), MX_F16, 16, 1);
        mx_u32x4 r = {d[0], d[1], d[2], d[3]};
        return __builtin_bit_cast(mx_f32x4, r);
    }
    mx_f32x4 FUNC(mma_sync_aligned_m16n8k16_row_col_f32_bf16_bf16_f32)(mx_u32x4 a, mx_u32x2 b, mx_f32x4 c)
    {
        mx_u32x8 d = mx_mma_m16n8(a, b, __builtin_bit_cast(mx_u32x4, c), MX_BF16, 16, 1);
        mx_u32x4 r = {d[0], d[1], d[2], d[3]};
        return __builtin_bit_cast(mx_f32x4, r);
    }
    mx_u32x4 FUNC(mma_sync_aligned_m16n8k32_row_col_s32_s8_s8_s32)(mx_u32x4 a, mx_u32x2 b, mx_u32x4 c)
    {
        mx_u32x8 d = mx_mma_m16n8(a, b, c, MX_S8, 32, 2);
        mx_u32x4 r = {d[0], d[1], d[2], d[3]};
        return r;
    }

    // ---------------------------------------------------------------- wmma.load / wmma.store
    // ctl: mat (0 a, 1 b, 2 c) | col << 2 | M << 3 | N << 9 | K << 15 | ew << 24; nel elements per lane
    MX_INL void mx_wm_geom(uint32_t ctl, int &rows, int &cols, int &ew, bool &col)
    {
        int mat = ctl & 3, M = (ctl >> 3) & 63, N = (ctl >> 9) & 63, K = (ctl >> 15) & 511;
        col = (ctl >> 2) & 1;
        ew = (ctl >> 24) & 127;
        rows = mat == 1 ? K : M;
        cols = mat == 0 ? K : N;
    }
    // (byte offset, bit offset) of element i of this lane's fragment (wmLoc)
    MX_INL uint64_t mx_wm_bit(uint32_t stride, uint32_t ctl, uint32_t nel, int i, int &ew)
    {
        int rows, cols;
        bool col;
        mx_wm_geom(ctl, rows, cols, ew, col);
        int r, c;
        mx_wmPos(rows, cols, (int)nel, mx_lane(), i, r, c);
        uint64_t idx = col ? (uint64_t)c * stride + r : (uint64_t)r * stride + c;
        return idx * (uint64_t)ew;
    }
#define MX_WMMA_LDST(SPACE, SNAME)                                                                        \
    MX_INL uint64_t mx_wm_rd_##SNAME(const uint8_t SPACE *p, uint32_t stride, uint32_t ctl, uint32_t nel, \
                                     int i)                                                               \
    {                                                                                                     \
        int ew;                                                                                           \
        uint64_t bit = mx_wm_bit(stride, ctl, nel, i, ew);                                                \
        const uint8_t SPACE *q = p + (bit >> 3);                                                          \
        if (ew < 8) return (q[0] >> (bit & 7)) & ((1u << ew) - 1u);                                       \
        uint64_t v = 0;                                                                                   \
        for (int j = 0; j < ew / 8; j++) v |= (uint64_t)q[j] << (8 * j);                                  \
        return v;                                                                                         \
    }                                                                                                     \
    MX_INL void mx_wm_wr_##SNAME(uint8_t SPACE *p, uint32_t stride, uint32_t ctl, uint32_t nel, int i,    \
                                 uint64_t v)                                                              \
    {                                                                                                     \
        int ew;                                                                                           \
        uint64_t bit = mx_wm_bit(stride, ctl, nel, i, ew);                                                \
        uint8_t SPACE *q = p + (bit >> 3);                                                                \
        if (ew < 8)                                                                                       \
        {                                                                                                 \
            uint8_t m = (uint8_t)(((1u << ew) - 1u) << (bit & 7));                                        \
            q[0] = (uint8_t)((q[0] & ~m) | ((v << (bit & 7)) & m));                                       \
            return;                                                                                       \
        }                                                                                                 \
        for (int j = 0; j < ew / 8; j++) q[j] = (uint8_t)(v >> (8 * j));                                  \
    }                                                                                                     \
    mx_u32x8 FUNC(wmma_load_##SNAME)(uint8_t SPACE * p, uint32_t stride, uint32_t ctl, uint32_t nel)      \
    {                                                                                                     \
        int ew = (ctl >> 24) & 127, per = 32 / ew;                                                        \
        mx_u32x8 d = {0, 0, 0, 0, 0, 0, 0, 0};                                                            \
        for (int i = 0; i < (int)nel; i++)                                                                \
        {                                                                                                 \
            uint32_t v = (uint32_t)mx_wm_rd_##SNAME(p, stride, ctl, nel, i) & mx_mask(ew);                \
            d = mx_put8(d, i / per, mx_pick8(d, i / per) | (v << (ew * (i % per))));                      \
        }                                                                                                 \
        return d;                                                                                         \
    }                                                                                                     \
    mx_u64x4 FUNC(wmma_load_f64_##SNAME)(uint8_t SPACE * p, uint32_t stride, uint32_t ctl, uint32_t nel)  \
    {                                                                                                     \
        mx_u64x4 d = {0, 0, 0, 0};                                                                        \
        MX_UNROLL for (int i = 0; i < 4; i++) if (i < (int)nel) d[i] = mx_wm_rd_##SNAME(p, stride, ctl, nel, i); \
        return d;                                                                                         \
    }                                                                                                     \
    void FUNC(wmma_store_##SNAME)(uint8_t SPACE * p, mx_u32x8 v, uint32_t stride, uint32_t ctl, uint32_t nel) \
    {                                                                                                     \
        int ew = (ctl >> 24) & 127, per = 32 / ew;                                                        \
        for (int i = 0; i < (int)nel; i++)                                                                \
            mx_wm_wr_##SNAME(p, stride, ctl, nel, i, (mx_pick8(v, i / per) >> (ew * (i % per))) & mx_mask(ew)); \
    }                                                                                                     \
    void FUNC(wmma_store_f64_##SNAME)(uint8_t SPACE * p, mx_u64x4 v, uint32_t stride, uint32_t ctl,       \
                                      uint32_t nel)                                                       \
    {                                                                                                     \
        MX_UNROLL for (int i = 0; i < 4; i++) if (i < (int)nel) mx_wm_wr_##SNAME(p, stride, ctl, nel, i, v[i]); \
    }
    MX_WMMA_LDST(MX_GLOBAL, global)
    MX_WMMA_LDST(SHARED_SPACE, shared)
    MX_WMMA_LDST(, generic)

    // ---------------------------------------------------------------- ldmatrix / stmatrix / movmatrix
    MX_INL uint32_t mx_half(uint32_t v, int h) { return (v >> (16 * h)) & 0xFFFF; }
    // ctl: num | trans << 4 | kind << 5 (0 m8n8 b16, 1 m8n16, 2 m16n16) | fmt << 8 (0 b8, 6 b6x16_p32, 4 b4x16_p64, 5 s4)
#define MX_LDM(SPACE, SNAME, PTRBITS)                                                                       \
    MX_INL uint32_t mx_ldmEl_##SNAME(const uint8_t SPACE *row, int f, int c)                                \
    {                                                                                                       \
        switch (f)                                                                                          \
        {                                                                                                   \
        case 6:                                                                                             \
        {                                                                                                   \
            int bit = 6 * c;                                                                                \
            uint32_t w = row[bit >> 3] | ((uint32_t)row[(bit >> 3) + 1] << 8);                              \
            return (w >> (bit & 7)) & 63;                                                                   \
        }                                                                                                   \
        case 4: return ((row[c >> 1] >> (4 * (c & 1))) & 15) << 2;                                          \
        case 5:                                                                                             \
        {                                                                                                   \
            uint32_t x = (row[c >> 1] >> (4 * (c & 1))) & 15;                                               \
            return (x & 8) ? (x | 240) : x;                                                                 \
        }                                                                                                   \
        default: return row[c];                                                                             \
        }                                                                                                   \
    }                                                                                                       \
    MX_INL const uint8_t SPACE *mx_addr_##SNAME(PTRBITS a, int lane)                                        \
    {                                                                                                       \
        if (sizeof(PTRBITS) == 8) return (const uint8_t SPACE *)(uintptr_t)mx_shfl64((uint64_t)a, lane);   \
        return (const uint8_t SPACE *)(uintptr_t)mx_shfl((uint32_t)a, lane);                               \
    }                                                                                                       \
    MX_INL mx_u32x8 mx_ldmatrix_##SNAME(const uint8_t SPACE *p, uint32_t ctl)                               \
    {                                                                                                       \
        int n = ctl & 15, tr = (ctl >> 4) & 1, kind = (ctl >> 5) & 7, f = (ctl >> 8) & 15;                  \
        int l = mx_lane();                                                                                  \
        PTRBITS me = (PTRBITS)(uintptr_t)p;                                                                 \
        mx_u32x8 d = {0, 0, 0, 0, 0, 0, 0, 0};                                                              \
        if (kind == 0)                                                                                      \
        {                                                                                                   \
            for (int i = 0; i < n; i++)                                                                     \
            {                                                                                               \
                uint32_t v = 0;                                                                             \
                for (int h = 0; h < 2; h++)                                                                 \
                {                                                                                           \
                    int src = tr ? 8 * i + 2 * (l & 3) + h : 8 * i + (l >> 2);                              \
                    int off = tr ? 2 * (l >> 2) : 2 * (2 * (l & 3) + h);                                    \
                    const uint8_t SPACE *q = mx_addr_##SNAME(me, src) + off;                                \
                    v |= ((uint32_t)q[0] | ((uint32_t)q[1] << 8)) << (16 * h);                              \
                }                                                                                           \
                d = mx_put8(d, i, v);                                                                       \
            }                                                                                               \
        }                                                                                                   \
        else if (kind == 1)                                                                                 \
        {                                                                                                   \
            for (int i = 0; i < n; i++)                                                                     \
            {                                                                                               \
                const uint8_t SPACE *row = mx_addr_##SNAME(me, 8 * i + (l >> 2));                           \
                uint32_t v = 0;                                                                             \
                for (int j = 0; j < 4; j++) v |= mx_ldmEl_##SNAME(row, f, 4 * (l & 3) + j) << (8 * j);      \
                d = mx_put8(d, i, v);                                                                       \
            }                                                                                               \
        }                                                                                                   \
        else                                                                                                \
        {                                                                                                   \
            for (int i = 0; i < n; i++)                                                                     \
            {                                                                                               \
                uint32_t v0 = 0, v1 = 0;                                                                    \
                for (int j = 0; j < 4; j++)                                                                 \
                {                                                                                           \
                    const uint8_t SPACE *row = mx_addr_##SNAME(me, 16 * i + 4 * (l & 3) + j);               \
                    v0 |= mx_ldmEl_##SNAME(row, f, l >> 2) << (8 * j);                                      \
                    v1 |= mx_ldmEl_##SNAME(row, f, (l >> 2) + 8) << (8 * j);                                \
                }                                                                                           \
                d = mx_put8(d, 2 * i, v0);                                                                  \
                d = mx_put8(d, 2 * i + 1, v1);                                                              \
            }                                                                                               \
        }                                                                                                   \
        return d;                                                                                           \
    }                                                                                                       \
    mx_u32x8 FUNC(ldmatrix_gen_##SNAME)(uint8_t SPACE * p, uint32_t ctl) { return mx_ldmatrix_##SNAME(p, ctl); } \
    void FUNC(stmatrix_gen_##SNAME)(uint8_t SPACE * p, mx_u32x4 v, uint32_t ctl)                            \
    {                                                                                                       \
        int n = ctl & 15, tr = (ctl >> 4) & 1, kind = (ctl >> 5) & 7;                                      \
        int l = mx_lane();                                                                                  \
        int i = l >> 3, a = l & 7;                                                                          \
        bool st = l < 8 * n;                                                                                \
        if (kind == 0)                                                                                      \
        {                                                                                                   \
            for (int c = 0; c < 8; c++)                                                                     \
            {                                                                                               \
                int src = tr ? c * 4 + (a >> 1) : a * 4 + (c >> 1);                                         \
                int h = tr ? (a & 1) : (c & 1);                                                             \
                uint32_t sel = 0;                                                                           \
                MX_UNROLL for (int r = 0; r < 4; r++)                                                       \
                    if (r < n)                                                                              \
                    {                                                                                       \
                        uint32_t x = mx_shfl(v[r], src);                                                    \
                        sel = (r == i) ? x : sel;                                                           \
                    }                                                                                       \
                uint32_t e = mx_half(sel, h);                                                               \
                if (st)                                                                                     \
                {                                                                                           \
                    p[2 * c] = (uint8_t)e;                                                                  \
                    p[2 * c + 1] = (uint8_t)(e >> 8);                                                       \
                }                                                                                           \
            }                                                                                               \
        }                                                                                                   \
        else                                                                                                \
        {                                                                                                   \
            for (int r = 0; r < 16; r++)                                                                    \
            {                                                                                               \
                int src = 4 * (r & 7) + (a >> 1);                                                           \
                uint32_t sel = 0;                                                                           \
                MX_UNROLL for (int rr = 0; rr < 4; rr++)                                                    \
                    if (rr < n)                                                                             \
                    {                                                                                       \
                        uint32_t x = mx_shfl(v[rr], src);                                                   \
                        sel = (rr == i) ? x : sel;                                                          \
                    }                                                                                       \
                if (st) p[r] = (uint8_t)(sel >> (8 * (2 * (r >> 3) + (a & 1))));                           \
            }                                                                                               \
        }                                                                                                   \
    }
    MX_LDM(SHARED_SPACE, shared, uint32_t)
    MX_LDM(, generic, uint64_t)

    uint32_t FUNC(movmatrix_m8n8_trans_b16)(uint32_t a)
    {
        int l = mx_lane();
        int c = l >> 2;
        uint32_t lo = mx_shfl(a, (2 * (l & 3)) * 4 + (c >> 1));
        uint32_t hi = mx_shfl(a, (2 * (l & 3) + 1) * 4 + (c >> 1));
        return mx_half(lo, c & 1) | (mx_half(hi, c & 1) << 16);
    }

    // the grammar's ldmatrix forms (ptx/src/pass/replace_instructions_with_functions.rs)
#define MX_LDM_NAMED(NAME, VT, CTL, NOUT)                                                \
    VT FUNC(NAME)(void SHARED_SPACE * p)                                                 \
    {                                                                                    \
        mx_u32x8 d = mx_ldmatrix_shared((const uint8_t SHARED_SPACE *)p, CTL);           \
        VT r;                                                                            \
        MX_UNROLL for (int i = 0; i < NOUT; i++) r[i] = d[i];                            \
        return r;                                                                        \
    }                                                                                    \
    VT FUNC(NAME##_generic)(void *p)                                                     \
    {                                                                                    \
        mx_u32x8 d = mx_ldmatrix_generic((const uint8_t *)p, CTL);                       \
        VT r;                                                                            \
        MX_UNROLL for (int i = 0; i < NOUT; i++) r[i] = d[i];                            \
        return r;                                                                        \
    }
    MX_LDM_NAMED(ldmatrix_m8n8_x1_b16, mx_u32x1, 1, 1)
    MX_LDM_NAMED(ldmatrix_m8n8_x2_b16, mx_u32x2, 2, 2)
    MX_LDM_NAMED(ldmatrix_m8n8_x4_b16, mx_u32x4, 4, 4)
    MX_LDM_NAMED(ldmatrix_m8n8_x1_trans_b16, mx_u32x1, 1 | 16, 1)
    MX_LDM_NAMED(ldmatrix_m8n8_x2_trans_b16, mx_u32x2, 2 | 16, 2)
    MX_LDM_NAMED(ldmatrix_m8n8_x4_trans_b16, mx_u32x4, 4 | 16, 4)
    MX_LDM_NAMED(ldmatrix_m16n16_x1_trans_b8, mx_u32x2, 1 | 16 | 64, 2)
    MX_LDM_NAMED(ldmatrix_m16n16_x2_trans_b8, mx_u32x4, 2 | 16 | 64, 4)
    MX_LDM_NAMED(ldmatrix_m16n16_x4_trans_b8, mx_u32x8, 4 | 16 | 64, 8)
