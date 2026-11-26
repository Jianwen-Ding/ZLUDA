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
    static inline uint64_t f_elem(uint64_t a, uint64_t b, uint64_t c, uint32_t ctrl)
    {
        uint32_t op = ctrl & 15, t = (ctrl >> 4) & 15;
        int rm = (ctrl >> 9) & 7;
        bool ftz = (ctrl & F_FTZ) && t != T_F64;
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
#define FSET_16(DN, DT) \
    DT FUNC(fset_##DN##_16)(uint16_t a, uint16_t b, bool c, uint32_t ctrl) { return __zluda_ptx_impl_fsetp_16(a, b, c, ctrl) ? (DT)set_true((ctrl >> 16) & 31) : 0; }
    FSET_16(16, uint16_t)
    FSET_16(32, uint32_t)

    // ---------------------------------------------------------------- f32 div / sqrt / rcp, exact
    // ctrl: 0-1 op (0 div 1 sqrt 2 rcp), 9-11 rounding, 12 .ftz, 13 .sat
    uint32_t FUNC(fdivsqrt_f32)(uint32_t a, uint32_t b, uint32_t ctrl)
    {
        uint32_t op = ctrl & 3;
        int rm = (ctrl >> 9) & 7;
        bool ftz = ctrl & F_FTZ;
        XF x = xdec_t(op == 2 ? 0x3F800000u : a, T_F32, ftz), y = xdec_t(op == 2 ? a : b, T_F32, ftz);
        XF v = op == 1 ? xf_sqrt(x, 24) : xf_div(x, y, 24);
        return (uint32_t)xfin(v, T_F32, rm, f_pf(ctrl, T_F32), 0, 0, false);
    }
