// Included inside extern "C" of zluda_ptx_metal_impl.cpp: helpers of ptx_parser/src/helper_table/float.rs.
//
// Floating-point arithmetic variants computed exactly with the integer soft-float core of cvt.inc
// (sail/common/ieee.sail): half / bfloat16 / packed / f32x2 / mixed precision add, sub, mul, fma
// with any rounding and .ftz .sat .relu .satfinite .oob, min / max with .NaN .xorsign .abs and
// three inputs, the alternate floating point x4 forms, packed half comparisons, and f32
// div / sqrt with directed roundings and .sat.

    // ---------------------------------------------------------------- exact operations
    // normalize a finite nonzero value so that its leading bit is bit 61
    static inline XF xf_n61(XF a)
    {
        int sh = 61 - (xf_bitlen(a.m) - 1);
        a.m <<= sh;
        a.e -= sh;
        return a;
    }
    // the sum, exact or with the shifted-out bits jammed into bit 0 (enough for one rounding to
    // at most 24 bits of precision)
    static inline XF xf_add(XF a, XF b, int rm)
    {
        if (a.c == XF_NAN || b.c == XF_NAN)
            return xf_nan();
        if (a.c == XF_INF && b.c == XF_INF)
            return a.s == b.s ? a : xf_nan();
        if (a.c == XF_INF)
            return a;
        if (b.c == XF_INF)
            return b;
        if (a.m == 0 && b.m == 0)
            return xf_fin(a.s == b.s ? a.s : rm == RM_RM, 0, 0);
        if (a.m == 0)
            return b;
        if (b.m == 0)
            return a;
        a = xf_n61(a);
        b = xf_n61(b);
        if (a.e < b.e)
        {
            XF t = a;
            a = b;
            b = t;
        }
        int d = a.e - b.e;
        uint64_t mb = d >= 63 ? 1 : ((b.m >> d) | ((b.m & ((1ull << d) - 1)) != 0));
        if (a.s == b.s)
            return xf_fin(a.s, a.m + mb, a.e);
        if (a.m == mb)
            return xf_fin(rm == RM_RM, 0, 0);
        if (a.m > mb)
            return xf_fin(a.s, a.m - mb, a.e);
        return xf_fin(b.s, mb - a.m, a.e);
    }
    static inline XF xf_mul(XF a, XF b)
    {
        if (a.c == XF_NAN || b.c == XF_NAN)
            return xf_nan();
        bool s = a.s != b.s;
        if (a.c == XF_INF || b.c == XF_INF)
        {
            if ((a.c == XF_FIN && a.m == 0) || (b.c == XF_FIN && b.m == 0))
                return xf_nan();
            return xf_inf(s);
        }
        return xf_fin(s, a.m * b.m, a.e + b.e);
    }
    static inline XF xf_fma(XF a, XF b, XF c, int rm)
    {
        XF p = xf_mul(a, b);
        if (p.c == XF_FIN && c.c == XF_FIN && p.m == 0 && c.m == 0)
            return xf_fin(p.s == c.s ? p.s : rm == RM_RM, 0, 0);
        return xf_add(p, c, rm);
    }
    // quotient of finite nonzero values with >= p + 4 bits and a jammed sticky bit
    static inline XF xf_div_fin(XF x, XF y, int p)
    {
        // mantissas normalized to p bits (p <= 30): q = (mx << (p + 4)) / my
        int sx = p - xf_bitlen(x.m), sy = p - xf_bitlen(y.m);
        uint64_t mx = x.m << sx, my = y.m << sy;
        uint64_t num = mx << (p + 4);
        uint64_t q = 0, r = 0;
        for (int i = 2 * p + 4; i >= 0; i--)
        {
            r = (r << 1) | ((num >> i) & 1);
            q <<= 1;
            if (r >= my)
            {
                r -= my;
                q |= 1;
            }
        }
        return xf_fin(x.s != y.s, (q << 1) | (r != 0), (x.e - sx) - (y.e - sy) - (p + 4) - 1);
    }
    static inline XF xf_div(XF x, XF y, int p)
    {
        if (x.c == XF_NAN || y.c == XF_NAN || (x.c == XF_INF && y.c == XF_INF))
            return xf_nan();
        bool s = x.s != y.s;
        if (x.c == XF_INF)
            return xf_inf(s);
        if (y.c == XF_INF)
            return xf_fin(s, 0, 0);
        if (y.m == 0)
            return x.m == 0 ? xf_nan() : xf_inf(s);
        if (x.m == 0)
            return xf_fin(s, 0, 0);
        return xf_div_fin(x, y, p);
    }
    // square root with >= p + 4 bits and a jammed sticky bit
    static inline XF xf_sqrt(XF x, int p)
    {
        if (x.c == XF_NAN)
            return x;
        if (x.c == XF_INF)
            return x.s ? xf_nan() : x;
        if (x.m == 0)
            return x;
        if (x.s)
            return xf_nan();
        // n = m * 2^k with 2 (p + 4) + 2 bits and an even exponent
        int k = 2 * (p + 4) + 2 - xf_bitlen(x.m);
        if (k < 0)
            k = 0;
        if ((x.e - k) & 1)
            k++;
        uint64_t n = x.m << k;
        uint64_t r = 0, rem = 0;
        for (int i = 62; i >= 0; i -= 2)
        {
            rem = (rem << 2) | ((n >> i) & 3);
            uint64_t t = (r << 2) | 1;
            r <<= 1;
            if (rem >= t)
            {
                rem -= t;
                r |= 1;
            }
        }
        return xf_fin(false, (r << 1) | (rem != 0), (x.e - k) / 2 - 1);
    }

    // three-way comparison of non-NaN values (-0 == +0)
    static inline int xf_cmp(XF a, XF b)
    {
        bool za = a.c == XF_FIN && a.m == 0, zb = b.c == XF_FIN && b.m == 0;
        if (za && zb)
            return 0;
        bool sa = za ? false : a.s, sb = zb ? false : b.s;
        // sign of a - b from the signs
        if (za)
            return sb ? 1 : -1;
        if (zb)
            return sa ? -1 : 1;
        if (sa != sb)
            return sa ? -1 : 1;
        int mag;
        if (a.c == XF_INF || b.c == XF_INF)
            mag = (a.c == XF_INF && b.c == XF_INF) ? 0 : a.c == XF_INF ? 1 : -1;
        else
        {
            int la = xf_bitlen(a.m) + a.e, lb = xf_bitlen(b.m) + b.e;
            if (la != lb)
                mag = la > lb ? 1 : -1;
            else
            {
                uint64_t ma = a.m << (64 - xf_bitlen(a.m)), mb = b.m << (64 - xf_bitlen(b.m));
                mag = ma > mb ? 1 : ma < mb ? -1 : 0;
            }
        }
        return sa ? -mag : mag;
    }

    // ---------------------------------------------------------------- min / max (fpMinMax)
    static inline bool bits_nan(uint64_t b, uint32_t t)
    {
        int p = t_p(t), eb = t_eb(t);
        return ((b >> (p - 1)) & ((1ull << eb) - 1)) == ((1ull << eb) - 1) && (b & ((1ull << (p - 1)) - 1)) != 0;
    }
    static inline uint64_t t_signbit(uint32_t t) { return 1ull << (t_cw(t) - 1); }
    static inline uint64_t canon_nan(uint32_t t) { return t_signbit(t) - 1; }
    static inline uint64_t fp_minmax(bool ftz, bool nanf, bool xs_, uint32_t t, uint64_t a, uint64_t b, bool isMin)
    {
        uint64_t sbit = t_signbit(t), mag = sbit - 1;
        if (ftz)
        {
            a = flush_bits(a, t);
            b = flush_bits(b, t);
        }
        uint64_t xs = (a ^ b) & sbit;
        bool na = bits_nan(a, t), nb = bits_nan(b, t);
        uint64_t r;
        if (na || nb)
        {
            if (nanf || (na && nb))
                return canon_nan(t);
            r = na ? b : a;
        }
        else if (xs_ || (a & mag) != 0 || (b & mag) != 0)
        {
            XF x = xdec_t(xs_ ? a & mag : a, t, false), y = xdec_t(xs_ ? b & mag : b, t, false);
            int c = xf_cmp(x, y);
            if (c < 0)
                r = isMin ? a : b;
            else if (c > 0)
                r = isMin ? b : a;
            else
                r = isMin ? (a | b) : (a & b);
        }
        else
            r = isMin ? (a | b) : (a & b);
        return xs_ ? ((r & mag) | xs) : r;
    }

    // ---------------------------------------------------------------- one element of a float op
    // ctrl: 0-3 op (0 add 1 sub 2 mul 3 fma 4 min 5 max 6 min3 7 max3), 4-7 element type, 8 packed,
    // 9-11 rounding, 12 .ftz, 13 .sat, 14 .relu, 15 .oob, 16 .NaN, 17 .xorsign, 18 .abs, 19 .satfinite
    enum
    {
        F_PK = 1u << 8,
        F_FTZ = 1u << 12,
        F_SAT = 1u << 13,
        F_RELU = 1u << 14,
        F_OOB = 1u << 15,
        F_NANM = 1u << 16,
        F_XS = 1u << 17,
        F_ABS = 1u << 18,
        F_SATF = 1u << 19
    };
    static inline uint32_t f_pf(uint32_t ctrl, uint32_t t)
    {
        return ((ctrl & F_FTZ) ? PF_FTZ : 0) | ((ctrl & F_SAT) ? PF_SAT : 0) | ((ctrl & F_RELU) ? PF_RELU : 0) |
               ((ctrl & F_SATF) ? PF_SATF : 0);
    }
    static inline uint64_t oob_nan(uint32_t t) { return t == T_F16 ? 0x7E00 : t == T_BF16 ? 0x7FC0 : ~0ull; }
    // the exact result of add / sub / mul / fma on decoded values
    static inline XF f_exact(uint32_t op, XF x, XF y, XF z, int rm)
    {
        switch (op)
        {
        case 0: return xf_add(x, y, rm);
        case 1: return xf_add(x, xf_neg(y), rm);
        case 2: return xf_mul(x, y);
        default: return xf_fma(x, y, z, rm);
        }
    }
    // add / sub / mul of halves to nearest without .ftz / .sat / .relu: the f32 operation, rounded once
    // more (innocuous double rounding: 24 >= 2 * 11 + 2 and 2 * 8 + 2). f16 operands, their sums and
    // products are normal f32 values (multiples of 2^-24, products of 2^-48), so the hardware's
    // flushing f32 ALU is exact; bf16 shares f32's exponent range and uses the IEEE operations.
    static inline bool f_half_fast(uint32_t op, uint32_t t, uint64_t a, uint64_t b, uint64_t *r)
    {
        if (op > 2 || t != T_F16)
            return false;
        {
            float x = (float)__builtin_bit_cast(_Float16, (uint16_t)a), y = (float)__builtin_bit_cast(_Float16, (uint16_t)b);
            float v = op == 0 ? x + y : op == 1 ? x - y : x * y;
            uint16_t h = __builtin_bit_cast(uint16_t, (_Float16)v);
            *r = ((h & 0x7FFF) > 0x7C00) ? 0x7FFF : h;
            return true;
        }
        // bf16: subnormal bf16 values are subnormal f32 ones; a constant operation of them would be
        // folded with the kernel's denormal mode before the IEEE tagging applies: the exact path
        return false;
    }
    static inline uint64_t f_elem(uint64_t a, uint64_t b, uint64_t c, uint32_t ctrl)
    {
        uint32_t op = ctrl & 15, t = (ctrl >> 4) & 15;
        int rm = (ctrl >> 9) & 7;
        bool ftz = (ctrl & F_FTZ) && t != T_F64;
        uint64_t fast;
        if (rm == RM_RN && f_pf(ctrl, t) == 0 && !(ctrl & F_OOB) && f_half_fast(op, t, a, b, &fast))
            return fast;
        if (op >= 4)
        {
            bool isMin = op == 4 || op == 6;
            uint64_t mag = t_signbit(t) - 1;
            if (op >= 6)
            {
                // three inputs: .abs on each, then min(min(a, b), c)
                if (ftz)
                {
                    a = flush_bits(a, t);
                    b = flush_bits(b, t);
                    c = flush_bits(c, t);
                }
                if (ctrl & F_ABS)
                {
                    a &= mag;
                    b &= mag;
                    c &= mag;
                }
                return fp_minmax(ftz, ctrl & F_NANM, false, t, fp_minmax(ftz, ctrl & F_NANM, false, t, a, b, isMin), c, isMin);
            }
            // f64 with two NaNs: the first, quieted (NaN payloads of f64)
            if (t == T_F64 && bits_nan(a, t) && bits_nan(b, t))
                return a | (1ull << 51);
            return fp_minmax(ftz, ctrl & F_NANM, ctrl & F_XS, t, a, b, isMin);
        }
        if ((ctrl & F_OOB) && op == 3 && (a == oob_nan(t) || b == oob_nan(t) || c == oob_nan(t)))
            return 0;
        if (t == T_F64)
        {
            if (bits_nan(a, t))
                return a | (1ull << 51);
            if (bits_nan(b, t))
                return b | (1ull << 51);
            if (op == 3 && bits_nan(c, t))
                return c | (1ull << 51);
        }
        XF x = xdec_t(a, t, ftz), y = xdec_t(b, t, ftz), z = xdec_t(c, t, ftz);
        return xfin(f_exact(op, x, y, z, rm), t, rm, f_pf(ctrl, t), 0, 0, false);
    }
    static inline uint64_t f_vec(uint64_t a, uint64_t b, uint64_t c, uint32_t ctrl)
    {
        if (!(ctrl & F_PK))
            return f_elem(a, b, c, ctrl);
        uint32_t t = (ctrl >> 4) & 15;
        int w = t == T_F32 ? 32 : 16;
        uint64_t m = (1ull << w) - 1;
        return f_elem(a & m, b & m, c & m, ctrl) | (f_elem((a >> w) & m, (b >> w) & m, (c >> w) & m, ctrl) << w);
    }
    uint16_t FUNC(fop_16)(uint16_t a, uint16_t b, uint16_t c, uint32_t ctrl) { return (uint16_t)f_vec(a, b, c, ctrl); }
    uint32_t FUNC(fop_32)(uint32_t a, uint32_t b, uint32_t c, uint32_t ctrl) { return (uint32_t)f_vec(a, b, c, ctrl); }
    uint64_t FUNC(fop_64)(uint64_t a, uint64_t b, uint64_t c, uint32_t ctrl) { return f_vec(a, b, c, ctrl); }

    // mixed precision: {add,sub,mul}{.rnd}{.sat}.f32.{f16,bf16} d, a, c and fma{.rnd}{.sat}.f32.{f16,bf16} d, a, b, c:
    // the half operands converted to f32 (exactly); ctrl as above with the half type in bits 20-23
    uint32_t FUNC(fmix_f32)(uint16_t a, uint16_t b, uint32_t c, uint32_t ctrl)
    {
        uint32_t op = ctrl & 15, ht = (ctrl >> 20) & 15;
        int rm = (ctrl >> 9) & 7;
        bool ftz = ctrl & F_FTZ;
        XF x = xdec_t(a, ht, false), y = xdec_t(b, ht, false), z = xdec_t(c, T_F32, ftz);
        if (op != 3)
            y = z;
        return (uint32_t)xfin(f_exact(op, x, y, z, rm), T_F32, rm, f_pf(ctrl, T_F32), 0, 0, false);
    }
    // {add,sub,mul}{.rnd}{.ftz}{.sat}.{f16x2,bf16x2}.f32x2.f32x2 d, a, c: each lane an f32 operation rounded
    // toward zero, converted toward zero; .ftz also flushes the half result
    uint32_t FUNC(fpair_f32x2)(uint64_t a, uint64_t c, uint32_t ctrl)
    {
        uint32_t ht = (ctrl >> 20) & 15;
        uint32_t ctrl32 = (ctrl & ~(15u << 4) & ~(7u << 9) & ~F_PK) | (T_F32 << 4) | (RM_RZ << 9);
        uint32_t r = 0;
        for (int i = 0; i < 2; i++)
        {
            uint32_t v = (uint32_t)f_elem((a >> (32 * i)) & 0xFFFFFFFF, (c >> (32 * i)) & 0xFFFFFFFF, 0, ctrl32);
            uint64_t h = xfin(xdec_t(v, T_F32, false), ht, RM_RZ, 0, 0, 0, false);
            if (ctrl & F_FTZ)
                h = flush_bits(h, ht);
            r |= (uint32_t)h << (16 * i);
        }
        return r;
    }
    // fma.rnd{.ftz}{.sat}.f32x2.{f16x2,bf16x2}.{f32x2|same}.f32x2 d, a, b, c: lanes converted to f32
    uint64_t FUNC(ffma_f32x2_pair)(uint32_t a, uint32_t b, uint64_t c, uint32_t ctrl)
    {
        uint64_t r = 0;
        for (int i = 0; i < 2; i++)
            r |= (uint64_t)__zluda_ptx_impl_fmix_f32((a >> (16 * i)) & 0xFFFF, (b >> (16 * i)) & 0xFFFF, (c >> (32 * i)) & 0xFFFFFFFF, ctrl) << (32 * i);
        return r;
    }

    // ---------------------------------------------------------------- alternate floating point x4 (exec_altfp)
    // element type codes: 0 e5m2x4 1 e4m3x4 2 e3m2x4 3 e2m3x4 4 e2m1x4 5 e2m1p4x4 6 ue8m0x4
    static inline XF alt4_elem(uint32_t k, uint32_t v, int i)
    {
        switch (k)
        {
        case 0: return mk_dec(T_E5M2, (v >> (8 * i)) & 255);
        case 1: return mk_dec(T_E4M3, (v >> (8 * i)) & 255);
        case 2: return mk_dec(T_E3M2, (v >> (8 * i)) & 63);
        case 3: return mk_dec(T_E2M3, (v >> (8 * i)) & 63);
        case 4: return mk_dec(T_E2M1, (v >> (4 * i)) & 15);
        case 5: return mk_dec(T_E2M1, (v >> (8 * i)) & 15);
        default:
        {
            uint32_t c = (v >> (8 * i)) & 255;
            return c == 255 ? xf_nan() : xf_fin(false, 1, (int)c - 127);
        }
        }
    }
    static inline uint32_t alt4_enc(uint32_t k, XF x, bool satf)
    {
        uint32_t sg = x.s && x.c != XF_NAN ? 128 : 0;
        uint32_t over = satf ? sg + mk_top(k) : k == T_E5M2 ? sg + 124 : 127;
        if (x.c == XF_NAN)
            return 127;
        if (x.c == XF_INF)
            return over;
        if (x.m == 0)
            return sg;
        int M = mk_M(k), bias = mk_bias(k);
        XF r = xround(false, x.m, x.e, false, M + 1, 1 - bias, 1 << 20, RM_RN, 0, 0);
        int64_t raw = r.m < (1ull << M) ? (int64_t)r.m : ((int64_t)(r.e + M + bias) << M) + (int64_t)(r.m - (1ull << M));
        return raw <= mk_top(k) ? sg + (uint32_t)raw : over;
    }
    // ctrl: 0-1 op (0 add 1 sub 2 mul 3 fma), 2 dst e4m3 (else e5m2), 3 .satfinite, 4-7 a type, 8-11 b type, 12-15 c type
    static inline uint32_t alt4(uint32_t a, uint32_t b, uint32_t c, uint32_t ctrl)
    {
        uint32_t op = ctrl & 3, dk = (ctrl & 4) ? T_E4M3 : T_E5M2;
        uint32_t ka = (ctrl >> 4) & 15, kb = (ctrl >> 8) & 15, kc = (ctrl >> 12) & 15;
        uint32_t r = 0;
        for (int i = 0; i < 4; i++)
        {
            XF x = alt4_elem(ka, a, i), y = alt4_elem(kb, b, i), z = alt4_elem(kc, c, i);
            XF v = op == 0 ? xf_add(x, y, RM_RN) : op == 1 ? xf_add(x, xf_neg(y), RM_RN) : op == 2 ? xf_mul(x, y) : xf_fma(x, y, z, RM_RN);
            r |= alt4_enc(dk, v, ctrl & 8) << (8 * i);
        }
        return r;
    }
    uint32_t FUNC(alt4_32_32)(uint32_t a, uint32_t b, uint32_t c, uint32_t ctrl) { return alt4(a, b, c, ctrl); }
    uint32_t FUNC(alt4_16_32)(uint16_t a, uint32_t b, uint32_t c, uint32_t ctrl) { return alt4(a, b, c, ctrl); }
    uint32_t FUNC(alt4_32_16)(uint32_t a, uint16_t b, uint32_t c, uint32_t ctrl) { return alt4(a, b, c, ctrl); }
    uint32_t FUNC(alt4_16_16)(uint16_t a, uint16_t b, uint32_t c, uint32_t ctrl) { return alt4(a, b, c, ctrl); }

    // ---------------------------------------------------------------- comparisons of half types
    // cmp code: 0 eq 1 ne 2 lt 3 le 4 gt 5 ge 6 equ 7 neu 8 ltu 9 leu 10 gtu 11 geu 12 num 13 nan
    static inline bool f_cmp(uint32_t cmpc, uint32_t t, uint64_t a, uint64_t b, bool ftz)
    {
        if (ftz)
        {
            a = flush_bits(a, t);
            b = flush_bits(b, t);
        }
        bool n = bits_nan(a, t) || bits_nan(b, t);
        if (cmpc == 12)
            return !n;
        if (cmpc == 13)
            return n;
        if (n)
            return cmpc >= 6;
        int c = xf_cmp(xdec_t(a, t, false), xdec_t(b, t, false));
        switch (cmpc % 6)
        {
        case 0: return c == 0;
        case 1: return c != 0;
        case 2: return c < 0;
        case 3: return c <= 0;
        case 4: return c > 0;
        default: return c >= 0;
        }
    }
    static inline bool bool_op(uint32_t bop, bool v, bool c)
    {
        switch (bop)
        {
        case 1: return v && c;
        case 2: return v || c;
        case 3: return v != c;
        default: return v;
        }
    }
    // ctrl: 0-3 cmp, 4-7 source type, 8 packed source, 9-10 BoolOp (0 none 1 and 2 or 3 xor), 11 .ftz, 12 negate c
    typedef uint32_t PredPair __attribute__((ext_vector_type(2)));
    PredPair FUNC(fsetp_pair)(uint32_t a, uint32_t b, bool c, uint32_t ctrl)
    {
        uint32_t t = (ctrl >> 4) & 15, bop = (ctrl >> 9) & 3;
        bool cc = (ctrl & (1u << 12)) ? !c : c;
        bool r0 = bool_op(bop, f_cmp(ctrl & 15, t, a & 0xFFFF, b & 0xFFFF, ctrl & (1u << 11)), cc);
        bool r1 = bool_op(bop, f_cmp(ctrl & 15, t, a >> 16, b >> 16, ctrl & (1u << 11)), cc);
        return PredPair{r0, r1};
    }
    bool FUNC(fsetp_pair_lo)(uint32_t a, uint32_t b, bool c, uint32_t ctrl) { return __zluda_ptx_impl_fsetp_pair(a, b, c, ctrl).x; }
    bool FUNC(fsetp_pair_hi)(uint32_t a, uint32_t b, bool c, uint32_t ctrl) { return __zluda_ptx_impl_fsetp_pair(a, b, c, ctrl).y; }
    bool FUNC(fsetp_16)(uint16_t a, uint16_t b, bool c, uint32_t ctrl);
    bool FUNC(fsetp_32)(uint32_t a, uint32_t b, bool c, uint32_t ctrl)
    {
        uint32_t t = (ctrl >> 4) & 15, bop = (ctrl >> 9) & 3;
        bool cc = (ctrl & (1u << 12)) ? !c : c;
        return bool_op(bop, f_cmp(ctrl & 15, t, a, b, ctrl & (1u << 11)), cc);
    }
    bool FUNC(fsetp_16)(uint16_t a, uint16_t b, bool c, uint32_t ctrl)
    {
        uint32_t t = (ctrl >> 4) & 15, bop = (ctrl >> 9) & 3;
        bool cc = (ctrl & (1u << 12)) ? !c : c;
        return bool_op(bop, f_cmp(ctrl & 15, t, a, b, ctrl & (1u << 11)), cc);
    }
    // set: dst type in bits 16-20 (cvt type codes), its "true" value 1.0 or all ones
    static inline uint64_t set_true(uint32_t dt)
    {
        switch (dt)
        {
        case T_F32: return 0x3F800000;
        case T_F16: return 0x3C00;
        case T_BF16: return 0x3F80;
        case T_F64: return 0x3FF0000000000000ull;
        default: return ~0ull;
        }
    }
    uint32_t FUNC(fset_pair)(uint32_t a, uint32_t b, bool c, uint32_t ctrl)
    {
        PredPair p = __zluda_ptx_impl_fsetp_pair(a, b, c, ctrl);
        uint32_t dt = (ctrl >> 16) & 31;
        uint32_t tv = dt == T_F16 ? 0x3C00 : dt == T_BF16 ? 0x3F80 : 0xFFFF;
        return (p.x ? tv : 0) | ((p.y ? tv : 0) << 16);
    }
#define FSET(DN, DT, SN, ST) \
    DT FUNC(fset_##DN##_##SN)(ST a, ST b, bool c, uint32_t ctrl) { return __zluda_ptx_impl_fsetp_##SN(a, b, c, ctrl) ? (DT)set_true((ctrl >> 16) & 31) : 0; }
    FSET(16, uint16_t, 16, uint16_t)
    FSET(32, uint32_t, 16, uint16_t)
    FSET(16, uint16_t, 32, uint32_t)
    FSET(32, uint32_t, 32, uint32_t)

    // RN(sqrt x) of a binary32 x with IEEE subnormals: a subnormal m 2^-149 = 2m 2^-150 has the root
    // RN(sqrt(2m)) 2^-75 (2m < 2^24 converts exactly; the scaling is exact, the root normal), so the root itself
    // never meets a subnormal (__zluda_nd_sqrt_f32: the hardware's correctly rounded root)
    static inline float f_sqrt_rn(float x)
    {
        uint32_t u = __builtin_bit_cast(uint32_t, x);
        // (a nonzero subnormal: 0 < 2u < 2^24, tested without masks, which LLVM would turn into a comparison of
        // fabs x with 0, false for a subnormal in a flushing kernel)
        bool sub = (u << 1) - 1u < 0x00FFFFFFu;
        float xs = sub ? (float)(int)((u & 0x7FFFFFu) << 1) : x;
        float r = __zluda_nd_sqrt_f32(xs);
        if (sub)
        {
            r = __zluda_nd_fmul_f32(r, 0x1p-75f);
            if (u >> 31)
                r = __builtin_bit_cast(float, 0x7FFFFFFFu); // the root of a negative subnormal: NaN
        }
        return r;
    }

    // ---------------------------------------------------------------- f32 div / sqrt / rcp, exact
    // ctrl: 0-1 op (0 div 1 sqrt 2 rcp), 9-11 rounding, 12 .ftz, 13 .sat
    uint32_t FUNC(fdivsqrt_f32)(uint32_t a, uint32_t b, uint32_t ctrl)
    {
        uint32_t op = ctrl & 3;
        int rm = (ctrl >> 9) & 7;
        bool ftz = ctrl & F_FTZ;
        if (rm == RM_RN && f_pf(ctrl, T_F32) == (ftz ? PF_FTZ : 0u))
        {
            // to nearest without saturation: the backend's correctly rounded IEEE operation
            // (subnormal operands and results kept: ftz flushes them explicitly), canonical NaN
            auto fl = [&](uint32_t u) { return (ftz && (u & 0x7F800000u) == 0) ? (u & 0x80000000u) : u; };
            float fx = __builtin_bit_cast(float, op == 2 ? 0x3F800000u : fl(a));
            float fy = __builtin_bit_cast(float, op == 2 ? fl(a) : fl(b));
#ifndef ZLUDA_NO_FAST_PATHS
            float fr = op == 1 ? f_sqrt_rn(fx) : __zluda_ieee_fdiv_f32(fx, fy);
#else
            float fr = op == 1 ? __zluda_ieee_sqrt_f32(fx) : __zluda_ieee_fdiv_f32(fx, fy);
#endif
            uint32_t r = __builtin_bit_cast(uint32_t, fr);
            if ((r & 0x7FFFFFFFu) > 0x7F800000u)
                return 0x7FFFFFFFu;
            return fl(r);
        }
        XF x = xdec_t(op == 2 ? 0x3F800000u : a, T_F32, ftz), y = xdec_t(op == 2 ? a : b, T_F32, ftz);
        XF v = op == 1 ? xf_sqrt(x, 24) : xf_div(x, y, 24);
        return (uint32_t)xfin(v, T_F32, rm, f_pf(ctrl, T_F32), 0, 0, false);
    }

    // neg / abs{.ftz}: ctrl 0 abs, 4-7 type, 8 packed, 12 .ftz
    static inline uint64_t f_negabs(uint64_t a, uint32_t ctrl)
    {
        uint32_t t = (ctrl >> 4) & 15;
        if ((ctrl & F_FTZ) && t != T_F64)
            a = flush_bits(a, t);
        if (t == T_F64 && bits_nan(a, t))
            return a;
        return (ctrl & 1) ? a & (t_signbit(t) - 1) : a ^ t_signbit(t);
    }
    static inline uint64_t f_negabs_v(uint64_t a, uint32_t ctrl)
    {
        if (!(ctrl & F_PK))
            return f_negabs(a, ctrl);
        return f_negabs(a & 0xFFFF, ctrl) | (f_negabs(a >> 16, ctrl) << 16);
    }
    uint16_t FUNC(fnegabs_16)(uint16_t a, uint32_t ctrl) { return (uint16_t)f_negabs_v(a, ctrl); }
    uint32_t FUNC(fnegabs_32)(uint32_t a, uint32_t ctrl) { return (uint32_t)f_negabs_v(a, ctrl); }
    uint64_t FUNC(fnegabs_64)(uint64_t a, uint32_t ctrl) { return f_negabs(a, ctrl); }

    // ---------------------------------------------------------------- transcendental helpers in double
    // (soft-f64 on AGX: correctly rounded IEEE operations)
    static inline double d_from_bits(uint64_t b) { return __builtin_bit_cast(double, b); }
    static inline uint64_t d_bits(double d) { return __builtin_bit_cast(uint64_t, d); }
    // 2^k for -1022 <= k <= 1023
    static inline double d_pow2(int k) { return d_from_bits((uint64_t)(k + 1023) << 52); }
    // 2^(j/32), correctly rounded (bits)
    static const __attribute__((address_space(4))) uint64_t D_EXP2_32[32] = {
        0x3ff0000000000000ull, 0x3ff059b0d3158574ull, 0x3ff0b5586cf9890full, 0x3ff11301d0125b51ull,
        0x3ff172b83c7d517bull, 0x3ff1d4873168b9aaull, 0x3ff2387a6e756238ull, 0x3ff29e9df51fdee1ull,
        0x3ff306fe0a31b715ull, 0x3ff371a7373aa9cbull, 0x3ff3dea64c123422ull, 0x3ff44e086061892dull,
        0x3ff4bfdad5362a27ull, 0x3ff5342b569d4f82ull, 0x3ff5ab07dd485429ull, 0x3ff6247eb03a5585ull,
        0x3ff6a09e667f3bcdull, 0x3ff71f75e8ec5f74ull, 0x3ff7a11473eb0187ull, 0x3ff82589994cce13ull,
        0x3ff8ace5422aa0dbull, 0x3ff93737b0cdc5e5ull, 0x3ff9c49182a3f090ull, 0x3ffa5503b23e255dull,
        0x3ffae89f995ad3adull, 0x3ffb7f76f2fb5e47ull, 0x3ffc199bdd85529cull, 0x3ffcb720dcef9069ull,
        0x3ffd5818dcfba487ull, 0x3ffdfc97337b9b5full, 0x3ffea4afa2a490daull, 0x3fff50765b6e4540ull};
    // e^y for |y| <= 60 (relative error a few units of 2^-53): e^y = 2^m 2^(j/32) e^r, k = 32 m + j the nearest
    // integer to 32 y / ln 2 (the 1.5 * 2^52 shift: no conversion), r = y - k ln2/32 (Cody-Waite, ln2/32 = hi + lo,
    // hi with 40 bits: k hi exact), |r| <= ln2/64 and e^r by Taylor to degree 6 (remainder < 2^-58); 11 soft-f64
    // operations
    static inline double d_exp(double y)
    {
        const double shift = 0x1.8p52, inv = 0x1.71547652b82fep+5, hi = 0x1.62e42fefa2000p-6,
                     lo = 0x1.9ef35793c7673p-46;
        double s = __builtin_fma(y, inv, shift);
        int k = (int)(uint32_t)d_bits(s);
        double kf = s - shift;
        double r = __builtin_fma(-kf, hi, y);
        r = __builtin_fma(-kf, lo, r);
        double p = __builtin_fma(r, 1.0 / 720.0, 1.0 / 120.0);
        p = __builtin_fma(r, p, 1.0 / 24.0);
        p = __builtin_fma(r, p, 1.0 / 6.0);
        p = __builtin_fma(r, p, 0.5);
        p = __builtin_fma(r, p, 1.0);
        p = __builtin_fma(r, p, 1.0);
        return (d_from_bits(D_EXP2_32[k & 31]) * p) * d_pow2(k >> 5);
    }
    // e^y - 1 for |y| <= 0.5
    static inline double d_expm1_small(double y)
    {
        // Taylor to degree 22 in Horner form (no divisions in soft-f64)
        return y * (1.0 + y * (0.5 + y * (0.16666666666666666 + y * (0.041666666666666664 + y * (0.008333333333333333 + y * (0.001388888888888889 + y * (0.0001984126984126984 + y * (2.48015873015873e-05 + y * (2.7557319223985893e-06 + y * (2.755731922398589e-07 + y * (2.505210838544172e-08 + y * (2.08767569878681e-09 + y * (1.6059043836821613e-10 + y * (1.1470745597729725e-11 + y * (7.647163731819816e-13 + y * (4.779477332387385e-14 + y * (2.8114572543455206e-15 + y * (1.5619206968586225e-16 + y * (8.22063524662433e-18 + y * (4.110317623312165e-19 + y * (1.9572941063391263e-20 + y * (8.896791392450574e-22))))))))))))))))))))));
    }
    // tanh(x) for x >= 0
    static inline double d_tanh_pos(double x)
    {
        if (x >= 20.0)
            return 1.0;
        if (x < 0.25)
        {
            double e = d_expm1_small(2.0 * x);
            return e / (e + 2.0);
        }
        double E = d_exp(2.0 * x);
        return 1.0 - 2.0 / (E + 1.0);
    }
    // round a double to format t (nearest even)
    static inline uint64_t d_round_to(double d, uint32_t t, uint32_t pf)
    {
        return xfin(xdec_t(d_bits(d), T_F64, false), t, RM_RN, pf, 0, 0, false);
    }
    static inline double xf_to_d(XF v)
    {
        return d_from_bits(xenc_ieee(xround_t(v, T_F64, RM_RN, 0, 0, false), 53, 11, 64));
    }

    // ---------------------------------------------------------------- correctly rounded binary32 exponentials
    // A fast path for the model's correctly rounded e^a and e^-a (tanh.approx.f32), in binary32 arithmetic on
    // values that are never subnormal (the __zluda_nd_* operations: any denormal mode gives the IEEE result):
    // k = rint(32 a / ln2), r = a - k ln2/32 as rh + rl (ln2/32 = L1 + L2 + L3, k L1 and k L2 exact, TwoSum),
    // e^(+-r) = 1 +- r + r^2/2 +- r^3/6 + r^4/24 +- r^5/120 as a float-float hi + lo, times 2^(+-k/32) from
    // F_EXP2_32 (hi, lo pairs). The float-float value is within about 2^-44 of e^(+-a) (relative); its rounding
    // to binary32 is returned only when both ends of the interval v + d +- 2^-40 v round to the same v, i.e.
    // the exact value cannot lie on the other side of a rounding boundary, and false otherwise (the caller
    // then takes the double precision path). Equality with that path for every binary32 operand is checked
    // exhaustively (zluda_metal/tests/stress/exhaustive/tanh_fast.cpp, IEEE and flushing operations).
    static const __attribute__((address_space(4))) float F_EXP2_32[64] = {
        0x1p+0f, 0x0p+0f, 0x1.059b0ep+0f, -0x1.9d4f52p-25f, 0x1.0b5586p+0f, 0x1.9f3122p-25f,
        0x1.11301ep+0f, -0x1.fdb496p-25f, 0x1.172b84p+0f, -0x1.c15742p-27f, 0x1.1d4874p+0f, -0x1.d2e8cap-25f,
        0x1.2387a6p+0f, 0x1.ceac48p-25f, 0x1.29e9ep+0f, -0x1.5c0424p-25f, 0x1.306fep+0f, 0x1.4636e2p-25f,
        0x1.371a74p+0f, -0x1.18aac6p-25f, 0x1.3dea64p+0f, 0x1.824684p-25f, 0x1.44e086p+0f, 0x1.8624b4p-30f,
        0x1.4bfdaep+0f, -0x1.593abcp-25f, 0x1.5342b6p+0f, -0x1.2c561p-25f, 0x1.5ab07ep+0f, -0x1.5bd5ecp-27f,
        0x1.6247ecp+0f, -0x1.f8b55p-25f, 0x1.6a09e6p+0f, 0x1.9fcef4p-26f, 0x1.71f75ep+0f, 0x1.1d8beep-25f,
        0x1.7a1148p+0f, -0x1.829fdp-25f, 0x1.82589ap+0f, -0x1.accc7cp-26f, 0x1.8ace54p+0f, 0x1.15506ep-27f,
        0x1.93737cp+0f, -0x1.e64744p-25f, 0x1.9c4918p+0f, 0x1.51f848p-27f, 0x1.a5503cp+0f, -0x1.b83b54p-25f,
        0x1.ae89fap+0f, -0x1.a94b14p-26f, 0x1.b7f77p+0f, -0x1.a09438p-25f, 0x1.c199bep+0f, -0x1.3d56b2p-27f,
        0x1.cb720ep+0f, -0x1.8837ccp-27f, 0x1.d5818ep+0f, -0x1.822dbcp-27f, 0x1.dfc974p+0f, -0x1.908c94p-25f,
        0x1.ea4afap+0f, 0x1.52486cp-27f, 0x1.f50766p+0f, -0x1.246ebp-26f};
#define ND_ADD __zluda_nd_fadd_f32
#define ND_SUB __zluda_nd_fsub_f32
#define ND_MUL __zluda_nd_fmul_f32
#define ND_FMA __zluda_nd_fma_f32
    // RN(T 2^m (h + l)) with T = F_EXP2_32[j], if certain
    static inline bool f_exp_round(int j, int m, float h, float l, float *out)
    {
        float th = F_EXP2_32[2 * j], tl = F_EXP2_32[2 * j + 1];
        float ph = ND_MUL(th, h);
        float pl = ND_ADD(ND_FMA(th, h, -ph), ND_FMA(th, l, ND_MUL(tl, h)));
        float v = ND_ADD(ph, pl);
        float d = ND_ADD(ND_SUB(ph, v), pl);
        float dl = ND_MUL(v, 0x1p-40f);
        if (ND_ADD(v, ND_ADD(d, dl)) != v || ND_ADD(v, ND_SUB(d, dl)) != v)
            return false;
        *out = ND_MUL(v, __builtin_bit_cast(float, (uint32_t)(m + 127) << 23));
        return true;
    }
    // e^a and e^-a correctly rounded for 2^-12 <= a < 20
    static inline bool f_exp_pm(float a, float *ep, float *em)
    {
        const float shift = 0x1.8p23f;
        float kf = ND_SUB(ND_FMA(a, 0x1.715476p+5f, shift), shift);
        int k = (int)kf;
        float r1 = ND_FMA(-kf, 0x1.62ep-6f, a);
        float p = ND_MUL(kf, 0x1.0bf8p-20f);
        // TwoSum(r1, -p)
        float s = ND_SUB(r1, p);
        float bb = ND_SUB(s, r1);
        float e = ND_ADD(ND_SUB(r1, ND_SUB(s, bb)), ND_SUB(-p, bb));
        float rl = ND_FMA(-kf, 0x1.f473dep-35f, e);
        float z = ND_MUL(s, s);
        float ze = ND_FMA(s, s, -z);
        float codd = ND_MUL(ND_MUL(s, z), ND_FMA(z, 0x1.111112p-7f, 0x1.555556p-3f));
        float ceven = ND_FMA(ND_MUL(z, z), 0x1.555556p-5f, ND_FMA(ze, 0.5f, ND_MUL(s, rl)));
        float hz = ND_MUL(z, 0.5f);
        float odd = ND_ADD(rl, codd);
        // e^r
        float h1 = ND_ADD(1.0f, s), t1 = ND_ADD(ND_SUB(1.0f, h1), s);
        float h2 = ND_ADD(h1, hz), t2 = ND_ADD(ND_SUB(h1, h2), hz);
        float lp = ND_ADD(ND_ADD(t1, t2), ND_ADD(odd, ceven));
        // e^-r
        float g1 = ND_SUB(1.0f, s), u1 = ND_SUB(ND_SUB(1.0f, g1), s);
        float g2 = ND_ADD(g1, hz), u2 = ND_ADD(ND_SUB(g1, g2), hz);
        float lm = ND_ADD(ND_ADD(u1, u2), ND_SUB(ceven, odd));
        int j = k & 31, m = k >> 5;
        int jm = (32 - j) & 31, mm = -m - (j != 0);
        return f_exp_round(j, m, h2, lp, ep) && f_exp_round(jm, mm, g2, lm, em);
    }
    // 2^x correctly rounded for -125 <= x < 127.5 (normal results): k = rint(32 x), f = x - k/32 (exact, |f| <=
    // 1/64), 2^x = 2^(k >> 5) 2^((k & 31)/32) e^(f ln2), r = f ln2 as rh + rl and e^r as in f_exp_pm
    static inline bool f_exp2_fast(float x, float *out)
    {
        const float shift = 0x1.8p23f;
        float kf = ND_SUB(ND_FMA(x, 32.0f, shift), shift);
        int k = (int)kf;
        float f = ND_FMA(-kf, 0.03125f, x);
        float s = ND_MUL(f, 0x1.62e430p-1f);
        float rl = ND_FMA(f, -0x1.05c610p-29f, ND_FMA(f, 0x1.62e430p-1f, -s));
        float z = ND_MUL(s, s);
        float ze = ND_FMA(s, s, -z);
        float codd = ND_MUL(ND_MUL(s, z), ND_FMA(z, 0x1.111112p-7f, 0x1.555556p-3f));
        float ceven = ND_FMA(ND_MUL(z, z), 0x1.555556p-5f, ND_FMA(ze, 0.5f, ND_MUL(s, rl)));
        float hz = ND_MUL(z, 0.5f);
        float odd = ND_ADD(rl, codd);
        float h1 = ND_ADD(1.0f, s), t1 = ND_ADD(ND_SUB(1.0f, h1), s);
        float h2 = ND_ADD(h1, hz), t2 = ND_ADD(ND_SUB(h1, h2), hz);
        float lp = ND_ADD(ND_ADD(t1, t2), ND_ADD(odd, ceven));
        return f_exp_round(k & 31, k >> 5, h2, lp, out);
    }
#undef ND_ADD
#undef ND_SUB
#undef ND_MUL
#undef ND_FMA

    // tanh.approx: halves correctly rounded; f32 as (e^x - e^-x) / (e^x + e^-x) of correctly rounded
    // exponentials, each operation rounded (ptx_math.sail)
    static inline uint64_t f_tanh(uint64_t a, uint32_t t, bool ftz)
    {
        XF x = xdec_t(a, t, ftz && t == T_F32);
        uint32_t pf = (ftz && t == T_F32) ? PF_FTZ : 0;
        if (x.c == XF_NAN)
            return canon_nan(t);
        if (t != T_F32)
        {
            if (x.c == XF_INF)
                return xenc_t(xround_t(xf_fin(x.s, 1, 0), t, RM_RN, 0, 0, false), t);
            if (x.m == 0)
                return a;
            double r = d_tanh_pos(xf_to_d(xf_abs(x)));
            return d_round_to(x.s ? -r : r, t, 0);
        }
        if (x.c == XF_INF || (x.m != 0 && xf_cmp_p2(x, 4) >= 0 && xf_cmp(xf_abs(x), xf_fin(false, 20, 0)) >= 0))
            return xpost(xround_t(xf_fin(x.s, 1, 0), T_F32, RM_RN, 0, 0, false), T_F32, pf);
        if (x.m == 0 || xf_cmp_p2(x, -12) < 0)
            return xpost(x, T_F32, pf);
#ifndef ZLUDA_NO_FAST_PATHS
        {
            // 2^-12 <= |x| < 20: the binary32 fast path for e^|x|, e^-|x| (tanh is odd: the exponentials swap)
            float ep, em;
            if (f_exp_pm(__builtin_bit_cast(float, (uint32_t)a & 0x7FFFFFFFu), &ep, &em))
            {
                float num = __zluda_nd_fsub_f32(ep, em), den = __zluda_nd_fadd_f32(ep, em);
                uint32_t r = __builtin_bit_cast(uint32_t, __zluda_nd_fdiv_f32(num, den));
                return r | ((uint32_t)a & 0x80000000u);
            }
        }
#endif
        double xd = xf_to_d(x);
        double ex = d_exp(xd);
        XF e1 = xdec_t(d_round_to(ex, T_F32, 0), T_F32, false);
        XF e2 = xdec_t(d_round_to(1.0 / ex, T_F32, 0), T_F32, false);
        XF num = xround_t(xf_add(e1, xf_neg(e2), RM_RN), T_F32, RM_RN, 0, 0, false);
        XF den = xround_t(xf_add(e1, e2, RM_RN), T_F32, RM_RN, 0, 0, false);
        return xfin(xf_div(num, den, 24), T_F32, RM_RN, pf, 0, 0, false);
    }
    // ctrl: 4-7 type, 8 packed, 12 .ftz
    uint16_t FUNC(ftanh_16)(uint16_t a, uint32_t ctrl) { return (uint16_t)f_tanh(a, (ctrl >> 4) & 15, false); }
    uint32_t FUNC(ftanh_32)(uint32_t a, uint32_t ctrl)
    {
        uint32_t t = (ctrl >> 4) & 15;
        if (ctrl & F_PK)
            return (uint32_t)f_tanh(a & 0xFFFF, t, false) | ((uint32_t)f_tanh(a >> 16, t, false) << 16);
        return (uint32_t)f_tanh(a, t, ctrl & F_FTZ);
    }

    // rcp / rsqrt .approx{.ftz}.f64 (apx64): on a[63:32], result rounded to 1.11.20
    uint64_t FUNC(fapx64)(uint64_t a, uint32_t isr)
    {
        uint64_t hi = 0xFFFFFFFF00000000ull;
        uint64_t b = flush_bits(a & hi, T_F64);
        XF x = xdec_t(b, T_F64, false);
        uint64_t nan = 0x7FFFFFFF00000000ull;
        if (x.c == XF_NAN)
            return nan;
        if (x.c == XF_INF)
            return (isr && x.s) ? nan : (x.s ? 1ull << 63 : 0);
        if (x.m == 0)
            return (x.s ? 1ull << 63 : 0) | 0x7FF0000000000000ull;
        if (isr && x.s)
            return nan;
        double d = d_from_bits(b);
        double r = isr ? 1.0 / __builtin_sqrt(d) : 1.0 / d;
        XF v = xdec_t(d_bits(r), T_F64, false);
        XF q = xround_v(v, 21, -1022, 1023, RM_RN, 0, 0, false);
        uint64_t e = xenc_ieee(xround_t(q, T_F64, RM_RN, 0, 0, false), 53, 11, 64);
        return flush_bits(e, T_F64) & hi;
    }

    // div.approx{.ftz}.f32 (divApprox): the quotient correctly rounded, except for 2^126 < |b|
    uint32_t FUNC(fdiv_approx_f32)(uint32_t a, uint32_t b, uint32_t ctrl)
    {
        bool ftz = ctrl & F_FTZ;
        XF x = xdec_t(a, T_F32, ftz), y = xdec_t(b, T_F32, false);
        if (y.c == XF_FIN && y.m != 0 && xf_cmp_p2(y, 126) > 0)
        {
            if (x.c != XF_FIN)
                return 0x7FFFFFFF;
            return (a ^ b) & 0x80000000u;
        }
        return __zluda_ptx_impl_fdivsqrt_f32(a, b, ctrl & ~3u);
    }
    // rsqrt.approx{.ftz}.f32: 1 / sqrt(x), two roundings
    uint32_t FUNC(frsqrt_f32)(uint32_t a, uint32_t ctrl)
    {
        bool ftz = ctrl & F_FTZ;
        {
            // RN(1 / RN(sqrt x)): two correctly rounded IEEE operations of the backend (subnormals kept;
            // .ftz flushes the operand and the result), canonical NaN
            auto fl = [&](uint32_t u) { return (ftz && (u & 0x7F800000u) == 0) ? (u & 0x80000000u) : u; };
#ifndef ZLUDA_NO_FAST_PATHS
            // the root of a binary32 value is never subnormal (2^-74.5 <= s < 2^64, or 0, inf, NaN), nor is 1 / s:
            // the division needs no subnormal handling, and the root none either once a subnormal operand is scaled
            float s = f_sqrt_rn(__builtin_bit_cast(float, fl(a)));
            uint32_t r = __builtin_bit_cast(uint32_t, __zluda_nd_fdiv_f32(1.0f, s));
#else
            float s = __zluda_ieee_sqrt_f32(__builtin_bit_cast(float, fl(a)));
            uint32_t r = __builtin_bit_cast(uint32_t, __zluda_ieee_fdiv_f32(1.0f, s));
#endif
            if ((r & 0x7FFFFFFFu) > 0x7F800000u)
                return 0x7FFFFFFFu;
            return fl(r);
        }
        XF x = xdec_t(a, T_F32, ftz);
        XF r = xround_t(xf_sqrt(x, 24), T_F32, RM_RN, 0, 0, false);
        return (uint32_t)xfin(xf_div(xf_fin(false, 1, 0), r, 24), T_F32, RM_RN, ftz ? PF_FTZ : 0, 0, 0, false);
    }

    // ex2.approx{.ftz}.{f16,bf16,f16x2,bf16x2}: 2^x correctly rounded
    static inline uint64_t f_ex2h(uint64_t a, uint32_t t, bool ftz)
    {
        XF x = xdec_t(a, t, ftz);
        uint32_t pf = ftz ? PF_FTZ : 0;
        if (x.c == XF_NAN)
            return canon_nan(t);
        if (x.c == XF_INF)
            return x.s ? 0 : xenc_t(x, t);
        if (x.m == 0)
            return xpost(xround_t(xf_fin(false, 1, 0), t, RM_RN, 0, 0, false), t, pf);
        double xd = xf_to_d(x);
        if (xd >= 160.0)
            return xpost(xf_inf(false), t, pf);
        if (xd <= -160.0)
            return xpost(xf_fin(false, 0, 0), t, pf);
        double n = (double)(int)xd;
        if (n > xd)
            n -= 1.0;
        double f = xd - n;
        double r = f == 0.0 ? 1.0 : d_exp(f * 0.69314718055994530942);
        int k = (int)n;
        XF v = xdec_t(d_bits(r), T_F64, false);
        v.e += k;
        return xfin(v, t, RM_RN, pf, 0, 0, false);
    }
    uint16_t FUNC(fex2h_16)(uint16_t a, uint32_t ctrl) { return (uint16_t)f_ex2h(a, (ctrl >> 4) & 15, ctrl & F_FTZ); }
    uint32_t FUNC(fex2h_32)(uint32_t a, uint32_t ctrl)
    {
        uint32_t t = (ctrl >> 4) & 15;
        return (uint32_t)f_ex2h(a & 0xFFFF, t, ctrl & F_FTZ) | ((uint32_t)f_ex2h(a >> 16, t, ctrl & F_FTZ) << 16);
    }
    // ex2.approx{.ftz}.f32: 2^x correctly rounded (exp2_r; subnormal results, .ftz flushes the operand and the
    // result). ctrl: 12 .ftz
    uint32_t FUNC(fex2_f32)(uint32_t a, uint32_t ctrl)
    {
        bool ftz = ctrl & F_FTZ;
#ifndef ZLUDA_NO_FAST_PATHS
        if ((a & 0x7FFFFFFFu) < 0x22800000u) // |x| < 2^-58 (zeros, subnormals): 2^x rounds to 1
            return 0x3F800000u;
        float x = __builtin_bit_cast(float, a), v;
        if (x >= -125.0f && x < 127.5f && f_exp2_fast(x, &v))
            return __builtin_bit_cast(uint32_t, v);
#endif
        return (uint32_t)f_ex2h(a, T_F32, ftz);
    }

    // ---------------------------------------------------------------- lg2.approx{.ftz}.f32: log2 x correctly rounded
    // (log2_r). x = 2^E m with m in [0.75, 1.5) (subnormals normalized), m = (1 + t) / inv for the slot i of m (32
    // slots of 1/128 below 1, 32 of 1/64 above; inv = 1 / the slot's center, 1 next to 1 so that t = m - 1 there),
    // log2 x = E + L_i + log1p(t) / ln2 with L_i = -log2 inv. t = m inv - 1 exactly as th + tl (fma), log1p(t) in
    // float-float (the t^2 / 2 and t^3 / 3 terms with their errors), within about 2^-44 of the result: taken when
    // both ends of v + d +- 2^-40 v round to v; otherwise the same evaluation in double with double-double sums
    // (L_i to 106 bits). Checked exhaustively against binary128 references
    // (zluda_metal/tests/stress/exhaustive/fastpath.cpp).
    static const __attribute__((address_space(4))) float F_LG2_INV[64] = {
        0x1.539094p+0f, 0x1.501502p+0f, 0x1.4cab88p+0f, 0x1.49539ep+0f, 0x1.460cbcp+0f, 0x1.42d662p+0f,
        0x1.3fb014p+0f, 0x1.3c995ap+0f, 0x1.3991c2p+0f, 0x1.3698ep+0f, 0x1.33ae46p+0f, 0x1.30d19p+0f,
        0x1.2e025cp+0f, 0x1.2b404ap+0f, 0x1.288b02p+0f, 0x1.25e228p+0f, 0x1.234568p+0f, 0x1.20b47p+0f,
        0x1.1e2ef4p+0f, 0x1.1bb4a4p+0f, 0x1.194538p+0f, 0x1.16e068p+0f, 0x1.1485fp+0f, 0x1.12358ep+0f,
        0x1.0fef02p+0f, 0x1.0db20ap+0f, 0x1.0b7e6ep+0f, 0x1.0953f4p+0f, 0x1.07326p+0f, 0x1.05198p+0f,
        0x1.03091cp+0f, 0x1p+0f, 0x1p+0f, 0x1.f4465ap-1f, 0x1.ecc07cp-1f, 0x1.e573acp-1f,
        0x1.de5d6ep-1f, 0x1.d77b66p-1f, 0x1.d0cb58p-1f, 0x1.ca4b3p-1f, 0x1.c3f8fp-1f, 0x1.bdd2b8p-1f,
        0x1.b7d6c4p-1f, 0x1.b20364p-1f, 0x1.ac5702p-1f, 0x1.a6d01ap-1f, 0x1.a16d4p-1f, 0x1.9c2d14p-1f,
        0x1.970e5p-1f, 0x1.920fb4p-1f, 0x1.8d3018p-1f, 0x1.886e6p-1f, 0x1.83c978p-1f, 0x1.7f406p-1f,
        0x1.7ad22p-1f, 0x1.767dcep-1f, 0x1.724288p-1f, 0x1.6e1f76p-1f, 0x1.6a13cep-1f, 0x1.661ec6p-1f,
        0x1.623fa8p-1f, 0x1.5e75bcp-1f, 0x1.5ac056p-1f, 0x1.571ed4p-1f};
    static const __attribute__((address_space(4))) float F_LG2_L[128] = {
        -0x1.a152eep-2f, -0x1.a6b326p-27f, -0x1.921804p-2f, 0x1.a1a556p-28f, -0x1.8304d8p-2f, 0x1.df844cp-27f,
        -0x1.7418acp-2f, 0x1.db59c4p-30f, -0x1.6552b2p-2f, -0x1.61c8b8p-28f, -0x1.56b22cp-2f, -0x1.809bcap-27f,
        -0x1.48365ep-2f, -0x1.00e582p-27f, -0x1.39de8cp-2f, -0x1.8d4ad8p-27f, -0x1.2baa08p-2f, -0x1.49eb34p-27f,
        -0x1.1d983p-2f, -0x1.c52cbap-29f, -0x1.0fa84ap-2f, 0x1.2bd5fp-27f, -0x1.01d9bcp-2f, 0x1.195ep-27f,
        -0x1.e857d4p-3f, 0x1.6b2562p-29f, -0x1.cd3c6ap-3f, 0x1.b26988p-28f, -0x1.b2602cp-3f, -0x1.f52042p-28f,
        -0x1.97c1d4p-3f, -0x1.a1840ep-28f, -0x1.7d604ep-3f, -0x1.d4e346p-31f, -0x1.633a84p-3f, -0x1.39d024p-33f,
        -0x1.494f8ap-3f, 0x1.673d24p-28f, -0x1.2f9e32p-3f, -0x1.4f2a9cp-28f, -0x1.162592p-3f, -0x1.783112p-28f,
        -0x1.f9c952p-4f, 0x1.02f9b6p-29f, -0x1.c7b516p-4f, 0x1.475dbp-33f, -0x1.960ca6p-4f, 0x1.41a4ccp-30f,
        -0x1.64ce3cp-4f, 0x1.bd9fd8p-29f, -0x1.33f7c2p-4f, -0x1.43f58ap-31f, -0x1.0387dep-4f, -0x1.ee6508p-29f,
        -0x1.a6f9d6p-5f, -0x1.e3a2d6p-30f, -0x1.47a9eap-5f, -0x1.6b76f6p-31f, -0x1.d23b2ap-6f, -0x1.ce897ap-32f,
        -0x1.16a25cp-6f, -0x1.4e837ap-33f, 0.0f, 0.0f, 0.0f, 0.0f,
        0x1.11cd1ap-5f, 0x1.95bee4p-30f, 0x1.c4df98p-5f, 0x1.6b67cap-33f, 0x1.3aa304p-4f, 0x1.59a09ep-29f,
        0x1.918a1ap-4f, -0x1.5929d6p-33f, 0x1.e72eb8p-4f, 0x1.07542p-30f, 0x1.1dcd2p-3f, -0x1.a4194p-29f,
        0x1.476aa2p-3f, -0x1.ee0eccp-30f, 0x1.70742ep-3f, 0x1.e698c8p-33f, 0x1.98edd4p-3f, 0x1.be3d5p-29f,
        0x1.c0db6cp-3f, -0x1.27fd6p-32f, 0x1.e840bep-3f, 0x1.462268p-28f, 0x1.0790acp-2f, 0x1.34f208p-27f,
        0x1.1ac05cp-2f, 0x1.4bfc42p-27f, 0x1.2db10ep-2f, 0x1.4e14d4p-28f, 0x1.406468p-2f, -0x1.f078d2p-27f,
        0x1.52dbdep-2f, -0x1.1c050cp-31f, 0x1.6519p-2f, 0x1.0f1768p-27f, 0x1.771d2ep-2f, 0x1.71866p-27f,
        0x1.88e9c4p-2f, -0x1.b52012p-28f, 0x1.9a8022p-2f, 0x1.3ae12ep-28f, 0x1.abe186p-2f, 0x1.be8f72p-27f,
        0x1.bd0f3p-2f, 0x1.90ef6ap-27f, 0x1.ce0a4ap-2f, 0x1.68d1e8p-29f, 0x1.ded3fep-2f, -0x1.d40e54p-27f,
        0x1.ef6d6ap-2f, 0x1.358d76p-31f, 0x1.ffd796p-2f, -0x1.5b318p-30f, 0x1.0809dp-1f, 0x1.f1c384p-27f,
        0x1.10113ap-1f, -0x1.6d8114p-29f, 0x1.18028cp-1f, 0x1.32824ap-31f, 0x1.1fde3ep-1f, 0x1.5003b6p-26f,
        0x1.27a4cp-1f, -0x1.34c24ap-28f};
    static const __attribute__((address_space(4))) uint64_t D_LG2_L[128] = {
        0xbfda152eed359931ull, 0x3c592f329d22b308ull, 0xbfd9218039796aabull, 0x3c74e5690ff1f06eull,
        0xbfd8304d7103dda3ull, 0x3c798833591862b5ull, 0xbfd7418abe24a63dull, 0x3c7fa37ab3d61feeull,
        0xbfd6552b258722e2ull, 0xbc72f241dca6fc3full, 0xbfd56b22cc04de54ull, 0x3c6c9bb0d879afd7ull,
        0xbfd48365e8072c0dull, 0x3c3803dc35aa1904ull, 0xbfd39de8cc6a56b9ull, 0xbc7c4556357fb55cull,
        0xbfd2baa08a4f5999ull, 0x3c7a5b286ddf549full, 0xbfd1d983038a5973ull, 0xbc59fabc38b4f1d7ull,
        0xbfd0fa8496a15078ull, 0xbc7557b896ea3a78ull, 0xbfd01d9bb7350ffbull, 0x3c5b68e9421e42b1ull,
        0xbfce857d3a536a7bull, 0xbc42ae5ac22439c5ull, 0xbfccd3c6926cb3baull, 0x3c59a0a2a08ad164ull,
        0xbfcb2602cfa90211ull, 0x3c520b55ede26811ull, 0xbfc97c1d4d0c206aull, 0x3c640e710afb9b3bull,
        0xbfc7d604e1d4e346ull, 0x3c30ce3d95e5651eull, 0xbfc633a8404e7409ull, 0x3c31c75b8f53b60full,
        0xbfc494f894c616e5ull, 0xbc59ae20a276fef6ull, 0xbfc2f9e32a7954e4ull, 0x3c411897da40f648ull,
        0xbfc162592bc18896ull, 0xbc6889ff670a2c9aull, 0xbfbf9c9517e83255ull, 0x3c435900e9f53f2eull,
        0xbfbc7b515f5c5128ull, 0x3c41f68825e6d096ull, 0xbfb960ca5af96cd0ull, 0xbc529c6aa0b462f4ull,
        0xbfb64ce3b213013cull, 0x3c485973ab0e8989ull, 0xbfb33f7c2287eb14ull, 0xbc50645f9c69c3e0ull,
        0xbfb0387def732844ull, 0xbc25eb43e530f38eull, 0xbfaa6f9d6f1d16afull, 0x3c44703c3ec0059cull,
        0xbfa47a9ea5addbd8ull, 0x3c4ffc052d725bd3ull, 0xbf9d23b2a73a25e5ull, 0xbc387ca1673278c9ull,
        0xbf916a25c29d06f6ull, 0x3c1696dc0f131e69ull, 0x0000000000000000ull, 0x0000000000000000ull,
        0x0000000000000000ull, 0x0000000000000000ull, 0x3fa11cd1acadf723ull, 0xbbfefa68b88a0ae7ull,
        0x3fac4df9816b67cbull, 0xbc2ad73136c614cbull, 0x3fb3aa304acd04f4ull, 0xbc54b3077c27b8b4ull,
        0x3fb918a19f536b15ull, 0xbc3336be543a9993ull, 0x3fbe72eb841d5082ull, 0xbc5f7c9478921bd7ull,
        0x3fc1dcd1f96f9b03ull, 0x3c69a5aba13ee9c5ull, 0x3fc476aa1c23e268ull, 0x3c58bdb8834b12ffull,
        0x3fc70742e079a632ull, 0x3c4bd6b99f1a542full, 0x3fc98edd46f8f540ull, 0x3c51d8e54f6b70ddull,
        0x3fcc0db6bf6c0150ull, 0xbc6a829cc1512df9ull, 0x3fce840bea311339ull, 0x3c6cbfe95c59cb8aull,
        0x3fd0790ac9a79044ull, 0x3c63ee2f0e46be64ull, 0x3fd1ac05ca5fe214ull, 0xbc71c2140fbe02dbull,
        0x3fd2db10e538534dull, 0x3c631cb428346b53ull, 0x3fd40646707c3973ull, 0xbc63ca60d9385e27ull,
        0x3fd52dbddf71fd7aull, 0xbc7d9b1af94a58afull, 0x3fd651900878bb3cull, 0x3c31f6461fef16c6ull,
        0x3fd771d2eb8c3300ull, 0xbc5796e600ac247eull, 0x3fd88e9c392b7fbbull, 0xbc6fb8ab87be1f10ull,
        0x3fd9a80224eb84b9ull, 0xbc5bcddc6a14952cull, 0x3fdabe186df47b97ull, 0x3c3234b98e55c106ull,
        0x3fdbd0f30c877b4full, 0xbc557b3fe5f0580cull, 0x3fdce0a4a2d1a3cfull, 0xbc1c50ac20cc5c35ull,
        0x3fdded3fd15f8d60ull, 0xbc72caa8abe0356cull, 0x3fdef6d6a09ac6bbull, 0xbc63186b98f3f398ull,
        0x3fdffd795ea4ce80ull, 0xbc716c968c39b3b9ull, 0x3fe0809d07c70e13ull, 0x3c8f7e57c5ce4173ull,
        0x3fe101139e927eecull, 0x3c670576a5aa8a36ull, 0x3fe18028c04ca092ull, 0x3c89995c81c00c2dull,
        0x3fe1fde3ea801dafull, 0xbc5369b2e485c22cull, 0x3fe27a4bfd967b6eull, 0xbc62ebcc9e150d9dull};
#define ND_ADD __zluda_nd_fadd_f32
#define ND_SUB __zluda_nd_fsub_f32
#define ND_MUL __zluda_nd_fmul_f32
#define ND_FMA __zluda_nd_fma_f32
    // TwoSum: a + b = s + e exactly
    static inline float f_two_sum(float a, float b, float *e)
    {
        float s = ND_ADD(a, b), bb = ND_SUB(s, a);
        *e = ND_ADD(ND_SUB(a, ND_SUB(s, bb)), ND_SUB(b, bb));
        return s;
    }
    static inline bool f_lg2_fast(int E, int i, float m, float *out)
    {
        float inv = F_LG2_INV[i];
        float p = ND_MUL(m, inv);
        float tl = ND_FMA(m, inv, -p);
        float th = ND_SUB(p, 1.0f); // exact (Sterbenz)
        float z = ND_MUL(th, th);
        float ze = ND_FMA(ND_ADD(th, th), tl, ND_FMA(th, th, -z));
        float c = ND_MUL(z, th);
        float ce = ND_FMA(ze, th, ND_FMA(z, tl, ND_FMA(z, th, -c)));
        float c3 = ND_MUL(c, 0x1.555556p-2f);
        float c3e = ND_FMA(ce, 0x1.555556p-2f, ND_FMA(c, -0x1.555556p-27f, ND_FMA(c, 0x1.555556p-2f, -c3)));
        // t^4 (-1/4 + t/5 - t^2/6 + t^3/7 - t^4/8)
        float q = ND_FMA(th, -0x1p-3f, 0x1.24924ap-3f);
        q = ND_FMA(th, q, -0x1.555556p-3f);
        q = ND_FMA(th, q, 0x1.99999ap-3f);
        q = ND_FMA(th, q, -0.25f);
        float t4q = ND_MUL(ND_MUL(z, z), q);
        float hz = ND_MUL(z, -0.5f);
        // th - z/2 + c3 (Fast2Sum twice: |th| > |z/2| > |c3|)
        float a = ND_ADD(th, hz), ae = ND_SUB(hz, ND_SUB(a, th));
        float b = ND_ADD(a, c3), be = ND_SUB(c3, ND_SUB(b, a));
        float lo = ND_ADD(ND_ADD(ND_ADD(ae, be), ND_ADD(tl, ND_MUL(ze, -0.5f))), ND_ADD(c3e, t4q));
        // times 1 / ln2
        float y = ND_MUL(b, 0x1.715476p+0f);
        float ye = ND_FMA(lo, 0x1.715476p+0f, ND_FMA(b, 0x1.4ae0cp-26f, ND_FMA(b, 0x1.715476p+0f, -y)));
        // E + L + y
        float e1, e2;
        float s1 = f_two_sum((float)E, F_LG2_L[2 * i], &e1);
        float s2 = f_two_sum(s1, y, &e2);
        float low = ND_ADD(ND_ADD(e1, e2), ND_ADD(F_LG2_L[2 * i + 1], ye));
        float v = ND_ADD(s2, low);
        float d = ND_ADD(ND_SUB(s2, v), low);
        float dl = ND_MUL(__builtin_fabsf(v), 0x1p-40f);
        if (ND_ADD(v, ND_ADD(d, dl)) != v || ND_ADD(v, ND_SUB(d, dl)) != v)
            return false;
        *out = v;
        return true;
    }
#undef ND_ADD
#undef ND_SUB
#undef ND_MUL
#undef ND_FMA
    static inline double d_two_sum(double a, double b, double *e)
    {
        double s = a + b, bb = s - a;
        *e = (a - (s - bb)) + (b - bb);
        return s;
    }
    // the same in double (double-double sums), correctly rounded to binary32
    static inline uint32_t f_lg2_exact(int E, int i, float m)
    {
        double inv = F_LG2_INV[i];
        double t = (double)m * inv - 1.0; // exact: 48-bit product, Sterbenz
        // log1p(t) = t + t^2 R(t), R = -1/2 + t/3 - ... - t^11/12 (|t| < 2^-6: the rest below 2^-80 t)
        double r = -1.0 / 12;
        for (int k = 11; k >= 2; k--)
            r = __builtin_fma(t, r, ((k & 1) ? 1.0 : -1.0) / k);
        double u = t * t * r;
        double ph = t + u, pl = u - (ph - t); // Fast2Sum (|t| > |u|)
        const double ih = 0x1.71547652b82fep+0, il = 0x1.777d0ffda0d24p-56;
        double yh = ph * ih;
        double yl = __builtin_fma(ph, ih, -yh) + (ph * il + pl * ih);
        double e1, e2;
        double s1 = d_two_sum((double)E, d_from_bits(D_LG2_L[2 * i]), &e1);
        double s2 = d_two_sum(s1, yh, &e2);
        double low = (e1 + e2) + (d_from_bits(D_LG2_L[2 * i + 1]) + yl);
        double sh = s2 + low, sl = low - (sh - s2);
        // round sh + sl once: sh rounded to binary32 unless it is a midpoint, which sl then decides
        uint64_t hb = d_bits(sh);
        XF v = xdec_t(hb, T_F64, false);
        if (sl != 0 && (hb & 0x1FFFFFFFull) == 0x10000000ull) // sh on a binary32 midpoint
            v = xdec_t(d_bits(sl > 0 ? sh + __builtin_fabs(sh) * 0x1p-40 : sh - __builtin_fabs(sh) * 0x1p-40), T_F64, false);
        return (uint32_t)xfin(v, T_F32, RM_RN, 0, 0, 0, false);
    }
    // ctrl: 12 .ftz
    uint32_t FUNC(flg2_f32)(uint32_t a, uint32_t ctrl)
    {
        bool ftz = ctrl & F_FTZ;
        uint32_t ex = (a >> 23) & 0xFF, mt = a & 0x7FFFFF;
        if (ftz && ex == 0)
            mt = 0;
        if (ex == 0xFF)
            return mt ? 0x7FFFFFFFu : (a >> 31) ? 0x7FFFFFFFu : 0x7F800000u;
        if (ex == 0 && mt == 0)
            return 0xFF800000u; // -inf
        if (a >> 31)
            return 0x7FFFFFFFu;
        int E;
        if (ex == 0)
        {
            int sh = __builtin_clz(mt) - 8; // leading bit to bit 23
            mt = (mt << sh) & 0x7FFFFF;
            E = -126 - sh;
        }
        else
            E = (int)ex - 127;
        if (mt == 0)
            return __builtin_bit_cast(uint32_t, (float)E); // powers of two: exact
        uint32_t mb = mt | 0x3F800000u;
        if (mb >= 0x3FC00000u) // m in [0.75, 1.5)
        {
            mb -= 1u << 23;
            E++;
        }
        int i = (int)((mb - 0x3F400000u) >> 17);
        float m = __builtin_bit_cast(float, mb), v;
#ifndef ZLUDA_NO_FAST_PATHS
        if (f_lg2_fast(E, i, m, &v))
            return __builtin_bit_cast(uint32_t, v);
#endif
        return f_lg2_exact(E, i, m);
    }

    // sin / cos .approx{.ftz}.f32: correctly rounded (sincos_r), via a Payne-Hanek reduction
    static inline uint64_t two_over_pi_word(int j)
    {
        switch (j)
        {
        case 1: return 0xA2F9836E;
        case 2: return 0x4E441529;
        case 3: return 0xFC2757D1;
        case 4: return 0xF534DDC0;
        case 5: return 0xDB629599;
        case 6: return 0x3C439041;
        case 7: return 0xFE5163AB;
        case 8: return 0xDEBBC561;
        default: return 0xB7246E3A;
        }
    }
    // m * 2^e * 2/pi modulo 4, with 62 fractional bits
    static inline uint64_t ph_reduce(uint64_t m, int e)
    {
        uint64_t y = 0;
        for (int j = 1; j <= 9; j++)
        {
            int s = e + 62 - 32 * j;
            if (s >= 64)
                continue;
            if (s <= -64)
                break;
            uint64_t p = m * two_over_pi_word(j);
            y += s >= 0 ? (p << s) : (p >> -s);
        }
        return y;
    }
    // sin r, cos r for |r| <= pi/4 (+ rounding slack): Taylor through r^17 / r^18 in Horner form, truncation below
    // 2^-60 relative (|r|^19/19! resp. |r|^20/20! over sin |r| resp. cos r >= 0.7)
    static inline double d_sin_k(double r)
    {
        double r2 = r * r;
        return r * (1.0 + r2 * (-0.16666666666666666 + r2 * (0.008333333333333333 + r2 * (-0.0001984126984126984 + r2 * (2.7557319223985893e-06 + r2 * (-2.505210838544172e-08 + r2 * (1.6059043836821613e-10 + r2 * (-7.647163731819816e-13 + r2 * (2.8114572543455206e-15)))))))));
    }
    static inline double d_cos_k(double r)
    {
        double r2 = r * r;
        return 1.0 + r2 * (-0.5 + r2 * (0.041666666666666664 + r2 * (-0.001388888888888889 + r2 * (2.48015873015873e-05 + r2 * (-2.755731922398589e-07 + r2 * (2.08767569878681e-09 + r2 * (-1.1470745597729725e-11 + r2 * (4.779477332387385e-14 + r2 * (-1.5619206968586225e-16)))))))));
    }
    // A fast path for the model's correctly rounded sin / cos .approx.f32, for 2^-12 <= |x| < 200, in binary32
    // arithmetic on values that are never subnormal (__zluda_nd_*): k = rint(32 |x| / pi) < 2^11, r = |x| - k pi/32
    // as rh + rl (pi/32 = P1 + ... + P6, P1..P5 with 13 bits: k Pi exact, |x| - k P1 exact, TwoSums), then
    // sin(|x|) (cos: k + 16) = S + S (cos r - 1) + C sin r with S, C = sin, cos(k pi/32) from F_SIN_64 (hi, lo pairs),
    // sin r = r - r^3/6 + r^5/120 - r^7/5040 and cos r - 1 = -r^2/2 + r^4/24 - r^6/720 (|r| <= pi/64) in float-float.
    // The result is within about 2^-42 (relative); it is returned only when both ends of v + d +- 2^-40 v round to
    // the same v, and when the result is not close to 0 (S = 0 with |r| < 2^-14, k > 0), otherwise false (the double
    // path). Checked exhaustively against it (zluda_metal/tests/stress/exhaustive/fastpath.cpp).
    static const __attribute__((address_space(4))) float F_SIN_64[128] = {
        0.0f, 0.0f, 0x1.917a6cp-4f, -0x1.eb25eap-31f, 0x1.8f8b84p-3f, -0x1.cb2cfap-30f,
        0x1.294062p-2f, 0x1.dab3ep-27f, 0x1.87de2ap-2f, 0x1.abaa58p-28f, 0x1.e2b5d4p-2f, -0x1.fe4272p-28f,
        0x1.1c73b4p-1f, -0x1.9465cep-27f, 0x1.44cf32p-1f, 0x1.424776p-27f, 0x1.6a09e6p-1f, 0x1.9fcef4p-27f,
        0x1.8bc806p-1f, 0x1.62a2e8p-26f, 0x1.a9b662p-1f, 0x1.21d434p-26f, 0x1.c38b3p-1f, -0x1.cfe84ap-26f,
        0x1.d906bcp-1f, 0x1.e651a8p-26f, 0x1.e9f416p-1f, -0x1.273a44p-26f, 0x1.f6297cp-1f, 0x1.feeb96p-26f,
        0x1.fd88dap-1f, 0x1.e89292p-28f, 0x1p+0f, 0.0f, 0x1.fd88dap-1f, 0x1.e89292p-28f,
        0x1.f6297cp-1f, 0x1.feeb96p-26f, 0x1.e9f416p-1f, -0x1.273a44p-26f, 0x1.d906bcp-1f, 0x1.e651a8p-26f,
        0x1.c38b3p-1f, -0x1.cfe84ap-26f, 0x1.a9b662p-1f, 0x1.21d434p-26f, 0x1.8bc806p-1f, 0x1.62a2e8p-26f,
        0x1.6a09e6p-1f, 0x1.9fcef4p-27f, 0x1.44cf32p-1f, 0x1.424776p-27f, 0x1.1c73b4p-1f, -0x1.9465cep-27f,
        0x1.e2b5d4p-2f, -0x1.fe4272p-28f, 0x1.87de2ap-2f, 0x1.abaa58p-28f, 0x1.294062p-2f, 0x1.dab3ep-27f,
        0x1.8f8b84p-3f, -0x1.cb2cfap-30f, 0x1.917a6cp-4f, -0x1.eb25eap-31f, 0.0f, 0.0f,
        -0x1.917a6cp-4f, 0x1.eb25eap-31f, -0x1.8f8b84p-3f, 0x1.cb2cfap-30f, -0x1.294062p-2f, -0x1.dab3ep-27f,
        -0x1.87de2ap-2f, -0x1.abaa58p-28f, -0x1.e2b5d4p-2f, 0x1.fe4272p-28f, -0x1.1c73b4p-1f, 0x1.9465cep-27f,
        -0x1.44cf32p-1f, -0x1.424776p-27f, -0x1.6a09e6p-1f, -0x1.9fcef4p-27f, -0x1.8bc806p-1f, -0x1.62a2e8p-26f,
        -0x1.a9b662p-1f, -0x1.21d434p-26f, -0x1.c38b3p-1f, 0x1.cfe84ap-26f, -0x1.d906bcp-1f, -0x1.e651a8p-26f,
        -0x1.e9f416p-1f, 0x1.273a44p-26f, -0x1.f6297cp-1f, -0x1.feeb96p-26f, -0x1.fd88dap-1f, -0x1.e89292p-28f,
        -0x1p+0f, 0.0f, -0x1.fd88dap-1f, -0x1.e89292p-28f, -0x1.f6297cp-1f, -0x1.feeb96p-26f,
        -0x1.e9f416p-1f, 0x1.273a44p-26f, -0x1.d906bcp-1f, -0x1.e651a8p-26f, -0x1.c38b3p-1f, 0x1.cfe84ap-26f,
        -0x1.a9b662p-1f, -0x1.21d434p-26f, -0x1.8bc806p-1f, -0x1.62a2e8p-26f, -0x1.6a09e6p-1f, -0x1.9fcef4p-27f,
        -0x1.44cf32p-1f, -0x1.424776p-27f, -0x1.1c73b4p-1f, 0x1.9465cep-27f, -0x1.e2b5d4p-2f, 0x1.fe4272p-28f,
        -0x1.87de2ap-2f, -0x1.abaa58p-28f, -0x1.294062p-2f, -0x1.dab3ep-27f, -0x1.8f8b84p-3f, 0x1.cb2cfap-30f,
        -0x1.917a6cp-4f, 0x1.eb25eap-31f};
#define ND_ADD __zluda_nd_fadd_f32
#define ND_SUB __zluda_nd_fsub_f32
#define ND_MUL __zluda_nd_fmul_f32
#define ND_FMA __zluda_nd_fma_f32
    static inline bool f_sincos_fast(float ax, bool iscos, float *out)
    {
        const float shift = 0x1.8p23f;
        float kf = ND_SUB(ND_FMA(ax, 0x1.45f306p+3f, shift), shift);
        int k = (int)kf;
        float r1 = ND_FMA(-kf, 0x1.921p-4f, ax);
        float p2 = ND_MUL(kf, 0x1.f6ap-17f), p3 = ND_MUL(kf, 0x1.11p-30f), p4 = ND_MUL(kf, 0x1.68cp-43f);
        float p5 = ND_MUL(kf, 0x1.1a6p-58f), p6 = ND_MUL(kf, 0x1.3198a2p-73f);
        // TwoSum(r1, -p2), TwoSum(s, -p3), TwoSum(s, -p4)
        float s = ND_SUB(r1, p2), bb = ND_SUB(s, r1);
        float e1 = ND_SUB(ND_SUB(r1, ND_SUB(s, bb)), ND_ADD(p2, bb));
        float s2 = ND_SUB(s, p3), b2 = ND_SUB(s2, s);
        float e2 = ND_SUB(ND_SUB(s, ND_SUB(s2, b2)), ND_ADD(p3, b2));
        float s3 = ND_SUB(s2, p4), b3 = ND_SUB(s3, s2);
        float e3 = ND_SUB(ND_SUB(s2, ND_SUB(s3, b3)), ND_ADD(p4, b3));
        float low = ND_SUB(ND_SUB(ND_ADD(ND_ADD(e1, e2), e3), p5), p6);
        float sh = ND_ADD(s3, low), sl = ND_SUB(low, ND_SUB(sh, s3));
        int js = (k + (iscos ? 16 : 0)) & 63, jc = (js + 16) & 63;
        if ((js & 31) == 0 && k != 0 && __builtin_fabsf(sh) < 0x1p-14f)
            return false;
        // z = r^2
        float zh = ND_MUL(sh, sh);
        float zl = ND_ADD(ND_FMA(sh, sh, -zh), ND_MUL(ND_ADD(sh, sh), sl));
        // sin r = r - r z / 6 + r z^2 (1/120 - z/5040)
        float p = ND_MUL(sh, zh);
        float plo = ND_ADD(ND_FMA(sh, zh, -p), ND_ADD(ND_MUL(sh, zl), ND_MUL(sl, zh)));
        float th = ND_MUL(p, 0x1.555556p-3f);
        float tl = ND_ADD(ND_FMA(p, 0x1.555556p-3f, -th), ND_ADD(ND_MUL(p, -0x1.555556p-28f), ND_MUL(plo, 0x1.555556p-3f)));
        float u = ND_MUL(ND_MUL(p, zh), ND_FMA(zh, -0x1.a01a02p-13f, 0x1.111112p-7f));
        float srh = ND_SUB(sh, th);
        float srl = ND_ADD(ND_SUB(ND_SUB(sh, srh), th), ND_ADD(ND_SUB(sl, tl), u));
        // cos r - 1 = -z/2 + z^2 (1/24 - z/720)
        float cmh = ND_MUL(zh, -0.5f);
        float cml = ND_FMA(zl, -0.5f, ND_MUL(ND_MUL(zh, zh), ND_FMA(zh, -0x1.6c16c2p-10f, 0x1.555556p-5f)));
        float Sh = F_SIN_64[2 * js], Sl = F_SIN_64[2 * js + 1], Ch = F_SIN_64[2 * jc], Cl = F_SIN_64[2 * jc + 1];
        float a1 = ND_MUL(Sh, cmh);
        float a1l = ND_ADD(ND_FMA(Sh, cmh, -a1), ND_ADD(ND_MUL(Sh, cml), ND_MUL(Sl, cmh)));
        float a2 = ND_MUL(Ch, srh);
        float a2l = ND_ADD(ND_FMA(Ch, srh, -a2), ND_ADD(ND_MUL(Ch, srl), ND_MUL(Cl, srh)));
        // Fast2Sum(Sh, a2), Fast2Sum(b1, a1): |Sh| >= sin(pi/32) > |a2| or Sh = 0, |b1| > |a1| or a1 = 0
        float b1 = ND_ADD(Sh, a2), b1e = ND_SUB(a2, ND_SUB(b1, Sh));
        float bb2 = ND_ADD(b1, a1), b2e = ND_SUB(a1, ND_SUB(bb2, b1));
        float lo = ND_ADD(ND_ADD(ND_ADD(b1e, b2e), Sl), ND_ADD(a1l, a2l));
        float v = ND_ADD(bb2, lo);
        float d = ND_ADD(ND_SUB(bb2, v), lo);
        float dl = ND_MUL(__builtin_fabsf(v), 0x1p-40f);
        if (ND_ADD(v, ND_ADD(d, dl)) != v || ND_ADD(v, ND_SUB(d, dl)) != v)
            return false;
        *out = v;
        return true;
    }
#undef ND_ADD
#undef ND_SUB
#undef ND_MUL
#undef ND_FMA
    uint32_t FUNC(fsincos_f32)(uint32_t a, uint32_t ctrl)
    {
        bool issin = !(ctrl & 1), ftz = ctrl & F_FTZ;
        uint32_t pf = ftz ? PF_FTZ : 0;
        XF x = xdec_t(a, T_F32, ftz);
        if (x.c != XF_FIN)
            return 0x7FFFFFFF;
        if (x.m == 0)
            return issin ? (uint32_t)xpost(x, T_F32, pf) : 0x3F800000u;
#ifndef ZLUDA_NO_FAST_PATHS
        {
            uint32_t ua = (uint32_t)a & 0x7FFFFFFFu;
            if (ua < 0x39800000u) // |x| < 2^-12: sin x rounds to x, cos x to 1
                return issin ? (uint32_t)xpost(x, T_F32, pf) : 0x3F800000u;
            float v;
            if (ua < 0x43480000u && f_sincos_fast(__builtin_bit_cast(float, ua), !issin, &v)) // |x| < 200
                return __builtin_bit_cast(uint32_t, v) ^ (issin ? (uint32_t)a & 0x80000000u : 0);
        }
#endif
        double ax = xf_to_d(xf_abs(x));
        int q = 0;
        double r = ax;
        if (ax > 0.78539816339744830962)
        {
            uint64_t y = ph_reduce(x.m, x.e);
            q = (int)(y >> 62);
            int64_t f = (int64_t)(y & ((1ull << 62) - 1));
            if (f >= (1ll << 61))
            {
                f -= 1ll << 62;
                q = (q + 1) & 3;
            }
            r = (double)f * d_pow2(-62) * 1.57079632679489661923;
        }
        int k = issin ? q : (q + 1) & 3;
        double v = (k & 1) ? d_cos_k(r) : d_sin_k(r);
        if (k & 2)
            v = -v;
        if (issin && x.s)
            v = -v;
        return (uint32_t)d_round_to(v, T_F32, pf);
    }

    // ---------------------------------------------------------------- packed integer lanes (gx_intx, PTX-EXEC-PRED)
    // {add,sub,min,max}{.sat}{.relu}.{u8x4,s8x4,u16x2,s16x2} d, a, b; {neg,abs}.{s8x4,s16x2} d, a;
    // set.CmpOp.{u8x4,s8x4,u16x2,s16x2} d, a, b
    // ctrl: 0-3 op (0 add 1 sub 2 min 3 max 4 neg 5 abs 6 set), 4 16-bit lanes, 5 signed, 6 .sat, 7 .relu, 8-11 cmp
    uint32_t FUNC(ipack)(uint32_t a, uint32_t b, uint32_t ctrl)
    {
        uint32_t op = ctrl & 15;
        int w = (ctrl & 16) ? 16 : 8;
        bool sg = ctrl & 32, sat = ctrl & 64, relu = ctrl & 128;
        uint32_t cmpc = (ctrl >> 8) & 15;
        uint32_t lm = (1u << w) - 1, r = 0;
        int64_t lo = sg ? -(1ll << (w - 1)) : 0, hi = sg ? (1ll << (w - 1)) - 1 : (int64_t)lm;
        for (int i = 0; i < 32 / w; i++)
        {
            uint32_t ua = (a >> (w * i)) & lm, ub = (b >> (w * i)) & lm;
            int64_t x = sg ? (int64_t)(ua ^ (1u << (w - 1))) - (1ll << (w - 1)) : ua;
            int64_t y = sg ? (int64_t)(ub ^ (1u << (w - 1))) - (1ll << (w - 1)) : ub;
            int64_t v;
            switch (op)
            {
            case 0: v = x + y; break;
            case 1: v = x - y; break;
            case 2: v = x < y ? x : y; break;
            case 3: v = x > y ? x : y; break;
            case 4: v = -x; break;
            case 5: v = x < 0 ? -x : x; break;
            default:
            {
                bool c;
                switch (cmpc)
                {
                case 0: c = x == y; break;
                case 1: c = x != y; break;
                case 2: c = x < y; break;
                case 3: c = x <= y; break;
                case 4: c = x > y; break;
                default: c = x >= y; break;
                }
                v = c ? (int64_t)lm : 0;
            }
            }
            if (sat && op < 2)
                v = v < lo ? lo : v > hi ? hi : v;
            if (relu && sg && op >= 2 && op <= 3 && v < 0)
                v = 0;
            r |= ((uint32_t)v & lm) << (w * i);
        }
        return r;
    }
    int32_t FUNC(iminmax_relu_s32)(int32_t a, int32_t b, uint32_t isMax)
    {
        int32_t v = isMax ? (a > b ? a : b) : (a < b ? a : b);
        return v < 0 ? 0 : v;
    }
