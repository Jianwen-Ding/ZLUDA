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

    // neg / abs{.ftz}: ctrl 0 abs, 4-7 type, 8 packed, 12 .ftz
    static inline uint64_t f_negabs(uint64_t a, uint32_t ctrl)
    {
        uint32_t t = (ctrl >> 4) & 15;
        if (ctrl & F_FTZ)
            a = flush_bits(a, t);
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

    // ---------------------------------------------------------------- transcendental helpers in double
    // (soft-f64 on AGX: correctly rounded IEEE operations)
    static inline double d_from_bits(uint64_t b) { return __builtin_bit_cast(double, b); }
    static inline uint64_t d_bits(double d) { return __builtin_bit_cast(uint64_t, d); }
    // 2^k for -1022 <= k <= 1023
    static inline double d_pow2(int k) { return d_from_bits((uint64_t)(k + 1023) << 52); }
    // e^y for |y| <= 60 (relative error a few units of 2^-53)
    static inline double d_exp(double y)
    {
        const double inv_ln2 = 1.4426950408889634074, ln2_hi = 6.93147180369123816490e-01, ln2_lo = 1.90821492927058770002e-10;
        double kf = y * inv_ln2;
        int k = (int)(kf >= 0 ? kf + 0.5 : kf - 0.5);
        double r = (y - k * ln2_hi) - k * ln2_lo;
        double t = 1.0, s = 1.0;
        for (int i = 1; i <= 16; i++)
        {
            t = t * r / i;
            s += t;
        }
        return s * d_pow2(k);
    }
    // e^y - 1 for |y| <= 0.5
    static inline double d_expm1_small(double y)
    {
        double t = y, s = y;
        for (int i = 2; i <= 22; i++)
        {
            t = t * y / i;
            s += t;
        }
        return s;
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
        double xd = xf_to_d(x);
        XF e1 = xdec_t(d_round_to(d_exp(xd), T_F32, 0), T_F32, false);
        XF e2 = xdec_t(d_round_to(d_exp(-xd), T_F32, 0), T_F32, false);
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
    static inline double d_sin_k(double r)
    {
        double r2 = r * r, t = r, s = r;
        for (int i = 2; i <= 26; i += 2)
        {
            t = -t * r2 / (double)(i * (i + 1));
            s += t;
        }
        return s;
    }
    static inline double d_cos_k(double r)
    {
        double r2 = r * r, t = 1.0, s = 1.0;
        for (int i = 1; i <= 26; i += 2)
        {
            t = -t * r2 / (double)(i * (i + 1));
            s += t;
        }
        return s;
    }
    uint32_t FUNC(fsincos_f32)(uint32_t a, uint32_t ctrl)
    {
        bool issin = !(ctrl & 1), ftz = ctrl & F_FTZ;
        uint32_t pf = ftz ? PF_FTZ : 0;
        XF x = xdec_t(a, T_F32, ftz);
        if (x.c != XF_FIN)
            return 0x7FFFFFFF;
        if (x.m == 0)
            return issin ? (uint32_t)xpost(x, T_F32, pf) : 0x3F800000u;
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
