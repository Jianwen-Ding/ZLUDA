// Included inside extern "C" of zluda_ptx_metal_impl.cpp: helpers of ptx_parser/src/helper_table/cvt.rs.
//
// An exact, integer-only soft-float core (no hardware float arithmetic: the AGX f32 ALU flushes
// subnormals) following PTX-formalization's sail/common/ieee.sail and sail/ptx/ptx_mini.sail:
// a value is (-1)^s * m * 2^e, rounded once to the destination format.  NaN results are the
// model's canonical NaN (positive, all payload bits set).  float.inc reuses this core.

    // ---------------------------------------------------------------- values
    enum
    {
        XF_FIN = 0,
        XF_INF = 1,
        XF_NAN = 2
    };
    struct XF
    {
        uint32_t c;
        bool s;
        uint64_t m;
        int32_t e;
    };
    static inline XF xf_fin(bool s, uint64_t m, int e) { return XF{XF_FIN, s, m, e}; }
    static inline XF xf_inf(bool s) { return XF{XF_INF, s, 0, 0}; }
    static inline XF xf_nan() { return XF{XF_NAN, false, 0, 0}; }
    static inline bool xf_zero(XF v) { return v.c == XF_FIN && v.m == 0; }
    static inline XF xf_neg(XF v)
    {
        if (v.c != XF_NAN)
            v.s = !v.s;
        return v;
    }
    static inline XF xf_abs(XF v)
    {
        v.s = false;
        return v;
    }
    static inline int xf_bitlen(uint64_t m) { return m ? 64 - __builtin_clzll(m) : 0; }

    // ---------------------------------------------------------------- type codes (helper_table/cvt.rs)
    enum
    {
        T_F64 = 0,
        T_F32 = 1,
        T_F16 = 2,
        T_BF16 = 3,
        T_TF32 = 4,
        T_E4M3 = 5,
        T_E5M2 = 6,
        T_E2M3 = 7,
        T_E3M2 = 8,
        T_E2M1 = 9,
        T_UE5M3 = 10,
        T_UE8M0 = 11,
        T_S2F6 = 12,
        T_U8 = 16,
        T_U16 = 17,
        T_U32 = 18,
        T_U64 = 19,
        T_S8 = 20,
        T_S16 = 21,
        T_S32 = 22,
        T_S64 = 23
    };
    static inline bool t_ieee(uint32_t t) { return t <= T_TF32; }
    static inline bool t_mini(uint32_t t) { return t >= T_E4M3 && t <= T_UE5M3; }
    static inline bool t_int(uint32_t t) { return t >= T_U8; }
    static inline bool t_signed(uint32_t t) { return t >= T_S8; }
    static inline int t_iw(uint32_t t) { return 8 << ((t - T_U8) & 3); }
    // IEEE formats: precision, exponent bits, container width
    static inline int t_p(uint32_t t) { return t == T_F64 ? 53 : t == T_F32 ? 24 : t == T_BF16 ? 8 : 11; }
    static inline int t_eb(uint32_t t) { return t == T_F64 ? 11 : t == T_F16 ? 5 : 8; }
    static inline int t_cw(uint32_t t) { return t == T_F64 ? 64 : (t == T_F32 || t == T_TF32) ? 32 : 16; }
    // rounding modes: signed (PTX) and magnitude
    enum
    {
        RM_RN = 0,
        RM_RZ = 1,
        RM_RM = 2,
        RM_RP = 3,
        RM_RNA = 4,
        RM_RS = 5
    };
    enum
    {
        MR_NE = 0,
        MR_Z = 1,
        MR_UP = 2,
        MR_NA = 3,
        MR_RS = 4
    };
    static inline int mag_mode(int rm, bool s)
    {
        switch (rm)
        {
        case RM_RN: return MR_NE;
        case RM_RZ: return MR_Z;
        case RM_RM: return s ? MR_UP : MR_Z;
        case RM_RP: return s ? MR_Z : MR_UP;
        case RM_RNA: return MR_NA;
        default: return MR_RS;
        }
    }

    // round(m * 2^-sh), `st` = nonzero bits below m; .rs adds rb to the nb most significant
    // discarded bits and rounds away from zero on a carry out
    static inline uint64_t shr_round(uint64_t m, int sh, bool st, int mm, uint32_t rb, int nb)
    {
        if (sh <= 0)
            return m << (-sh);
        uint64_t q, r;
        bool gt, eq;
        if (sh >= 64)
        {
            q = 0;
            r = m;
            if (sh == 64)
            {
                gt = r > (1ull << 63) || (r == (1ull << 63) && st);
                eq = r == (1ull << 63) && !st;
            }
            else
            {
                gt = false;
                eq = false;
            }
        }
        else
        {
            q = m >> sh;
            r = m & ((1ull << sh) - 1);
            uint64_t h = 1ull << (sh - 1);
            gt = r > h || (r == h && st);
            eq = r == h && !st;
        }
        bool inexact = r != 0 || st;
        bool up;
        switch (mm)
        {
        case MR_NE: up = gt || (eq && (q & 1)); break;
        case MR_Z: up = false; break;
        case MR_UP: up = inexact; break;
        case MR_NA: up = gt || eq; break;
        default:
        {
            uint64_t t;
            if (sh - nb >= 64)
                t = 0;
            else if (sh >= nb)
                t = r >> (sh - nb);
            else
                t = r << (nb - sh);
            up = m != 0 && t + rb >= (1ull << nb);
        }
        }
        return q + (up ? 1 : 0);
    }

    // fround of ieee.sail: (-1)^s (m + sticky) 2^e at precision p, normal exponents [emin, emax]
    static inline XF xround(bool s, uint64_t m, int e, bool st, int p, int emin, int emax, int rm, uint32_t rb, int nb)
    {
        if (m == 0)
            return xf_fin(s, 0, 0);
        int lead = xf_bitlen(m) - 1 + e;
        int lsb = lead < emin ? emin - (p - 1) : lead - (p - 1);
        int mm = mag_mode(rm, s);
        uint64_t q = shr_round(m, lsb - e, st, mm, rb, nb);
        if (q == (1ull << p))
        {
            q = 1ull << (p - 1);
            lsb++;
        }
        if (q >= (1ull << (p - 1)) && lsb + p - 1 > emax)
        {
            if (mm == MR_Z)
                return xf_fin(s, (1ull << p) - 1, emax - (p - 1));
            return xf_inf(s);
        }
        return xf_fin(s, q, lsb);
    }
    static inline XF xround_v(XF v, int p, int emin, int emax, int rm, uint32_t rb, int nb, bool st)
    {
        if (v.c != XF_FIN)
            return v;
        return xround(v.s, v.m, v.e, st, p, emin, emax, rm, rb, nb);
    }
    // round to an IEEE format (tf32: 11 bits, 8 exponent bits)
    static inline XF xround_t(XF v, uint32_t t, int rm, uint32_t rb, int nb, bool st)
    {
        int p = t_p(t), bias = (1 << (t_eb(t) - 1)) - 1;
        return xround_v(v, p, 1 - bias, bias, rm, rb, nb, st);
    }

    // ---------------------------------------------------------------- IEEE bits <-> values
    static inline XF xdec_ieee(uint64_t b, int p, int eb, int cw, bool ftz)
    {
        bool s = (b >> (cw - 1)) & 1;
        uint64_t E = (b >> (p - 1)) & ((1ull << eb) - 1);
        uint64_t M = b & ((1ull << (p - 1)) - 1);
        int bias = (1 << (eb - 1)) - 1;
        if (E == (1ull << eb) - 1)
            return M == 0 ? xf_inf(s) : xf_nan();
        if (E == 0)
            return xf_fin(s, ftz ? 0 : M, 1 - bias - (p - 1));
        return xf_fin(s, M | (1ull << (p - 1)), (int)E - bias - (p - 1));
    }
    // exact encoding of a value representable in the format
    static inline uint64_t xenc_ieee(XF v, int p, int eb, int cw)
    {
        uint64_t sb = v.s ? 1ull << (cw - 1) : 0;
        if (v.c == XF_NAN)
            return cw == 64 ? 0x7FFFFFFFFFFFFFFFull : ((1ull << (cw - 1)) - 1);
        int bias = (1 << (eb - 1)) - 1;
        if (v.c == XF_INF)
            return sb | (((1ull << eb) - 1) << (p - 1));
        if (v.m == 0)
            return sb;
        if (v.m < (1ull << (p - 1)))
            return sb | v.m;
        return sb | ((uint64_t)(v.e + (p - 1) + bias) << (p - 1)) | (v.m - (1ull << (p - 1)));
    }
    static inline XF xdec_t(uint64_t b, uint32_t t, bool ftz)
    {
        if (t == T_TF32)
            return xround_t(xdec_ieee(b, 24, 8, 32, false), T_TF32, RM_RN, 0, 0, false);
        return xdec_ieee(b, t_p(t), t_eb(t), t_cw(t), ftz);
    }
    // encode a value already rounded to format t
    static inline uint64_t xenc_t(XF v, uint32_t t)
    {
        if (t == T_TF32)
            return xenc_ieee(xround_t(v, T_F32, RM_RN, 0, 0, false), 24, 8, 32);
        return xenc_ieee(v, t_p(t), t_eb(t), t_cw(t));
    }
    static inline bool is_subn_bits(uint64_t b, uint32_t t)
    {
        int p = t_p(t), eb = t_eb(t);
        return ((b >> (p - 1)) & ((1ull << eb) - 1)) == 0 && (b & ((1ull << (p - 1)) - 1)) != 0;
    }
    static inline uint64_t flush_bits(uint64_t b, uint32_t t) { return is_subn_bits(b, t) ? b & (1ull << (t_cw(t) - 1)) : b; }

    // value comparisons
    static inline bool xf_gt_one(XF v)
    {
        if (v.c == XF_INF)
            return !v.s;
        if (v.c != XF_FIN || v.s || v.m == 0)
            return false;
        int lead = xf_bitlen(v.m) - 1 + v.e;
        return lead > 0 || (lead == 0 && (v.m & (v.m - 1)) != 0);
    }
    // |v| compared with 2^k: -1, 0, 1 (finite nonzero v)
    static inline int xf_cmp_p2(XF v, int k)
    {
        int lead = xf_bitlen(v.m) - 1 + v.e;
        if (lead != k)
            return lead < k ? -1 : 1;
        return (v.m & (v.m - 1)) ? 1 : 0;
    }

    // post of ieee.sail: clamps (.relu .sat .satfinite), encoding, .ftz of the result
    enum
    {
        PF_FTZ = 1,
        PF_SAT = 2,
        PF_RELU = 4,
        PF_SATF = 8
    };
    static inline uint64_t xpost(XF f, uint32_t t, uint32_t pf)
    {
        XF r = f;
        if ((pf & PF_RELU) && r.c != XF_NAN && r.s)
            r = xf_fin(false, 0, 0);
        if (pf & PF_SAT)
        {
            if (f.c == XF_NAN || (f.s && !xf_zero(f)))
                r = xf_fin(false, 0, 0);
            else if (xf_gt_one(f))
                r = xf_fin(false, 1ull << (t_p(t) - 1), 1 - t_p(t));
            else
                r = f;
        }
        else if ((pf & PF_SATF) && r.c == XF_INF)
            r = xf_fin(r.s, (1ull << t_p(t)) - 1, ((1 << (t_eb(t) - 1)) - 1) - (t_p(t) - 1));
        uint64_t b = xenc_t(r, t);
        if ((pf & PF_FTZ) && t != T_TF32)
            b = flush_bits(b, t);
        return b;
    }
    // fin of ieee.sail: round an exact value with the mode, then post
    static inline uint64_t xfin(XF v, uint32_t t, int rm, uint32_t pf, uint32_t rb, int nb, bool st)
    {
        return xpost(xround_t(v, t, rm, rb, nb, st), t, pf);
    }

    // ---------------------------------------------------------------- integers
    static inline XF xdec_int(uint64_t b, uint32_t t)
    {
        int w = t_iw(t);
        uint64_t v = w == 64 ? b : b & ((1ull << w) - 1);
        if (t_signed(t) && w < 64 && ((v >> (w - 1)) & 1))
            v |= ~0ull << w;
        if (t_signed(t) && (int64_t)v < 0)
            return xf_fin(true, (uint64_t)0 - v, 0);
        return xf_fin(false, v, 0);
    }
    // round a finite value to an integer magnitude with signed mode rm; *big if >= 2^64
    static inline uint64_t xf_rint_mag(XF f, int rm, bool *big)
    {
        *big = false;
        if (f.m == 0)
            return 0;
        if (f.e >= 0)
        {
            if (xf_bitlen(f.m) + f.e > 64)
            {
                *big = true;
                return 0;
            }
            return f.m << f.e;
        }
        return shr_round(f.m, -f.e, false, mag_mode(rm, f.s), 0, 0);
    }
    // f2i of ptx_common.sail
    static inline uint64_t xf2i(XF f, int rm, uint32_t t, bool src64)
    {
        int w = t_iw(t);
        bool sg = t_signed(t);
        uint64_t wmask = w == 64 ? ~0ull : (1ull << w) - 1;
        uint64_t smax = (1ull << (w - 1)) - 1, smin = 1ull << (w - 1);
        if (f.c == XF_NAN)
            return (src64 || w == 64) ? (1ull << (w - 1)) : 0;
        bool big = false;
        uint64_t n = 0;
        if (f.c == XF_INF)
            big = true;
        else
            n = xf_rint_mag(f, rm, &big);
        if (sg)
        {
            if (f.s)
                return (big || n > smin) ? smin : ((0 - n) & wmask);
            return (big || n > smax) ? smax : n;
        }
        if (f.s)
            return 0;
        return (big || n > wmask) ? wmask : n;
    }
    // round to an integral value (rndInt of ptx_common.sail)
    static inline XF xf_rndint(XF f, int rm)
    {
        if (f.c != XF_FIN || f.e >= 0 || f.m == 0)
            return f;
        uint64_t q = shr_round(f.m, -f.e, false, mag_mode(rm, f.s), 0, 0);
        return xf_fin(f.s, q, 0);
    }

    // ---------------------------------------------------------------- mini formats (ptx_mini.sail)
    static inline int mk_E(uint32_t k) { return k == T_E4M3 ? 4 : (k == T_E5M2 || k == T_UE5M3) ? 5 : k == T_E3M2 ? 3 : 2; }
    static inline int mk_M(uint32_t k) { return (k == T_E5M2 || k == T_E3M2) ? 2 : k == T_E2M1 ? 1 : 3; }
    static inline int mk_top(uint32_t k)
    {
        switch (k)
        {
        case T_E4M3: return 126;
        case T_E5M2: return 123;
        case T_E2M3: return 31;
        case T_E3M2: return 31;
        case T_E2M1: return 7;
        default: return 254;
        }
    }
    static inline int mk_bias(uint32_t k) { return (1 << (mk_E(k) - 1)) - 1; }
    static inline int mk_bits(uint32_t k) { return k == T_UE5M3 ? 8 : 1 + mk_E(k) + mk_M(k); }
    static inline bool mk_isnan(uint32_t k, uint32_t c) { return k == T_E4M3 ? c == 127 : k == T_E5M2 ? c > 124 : k == T_UE5M3 ? c == 255 : false; }
    static inline XF mk_mag(uint32_t k, uint32_t c)
    {
        int M = mk_M(k);
        uint32_t hi = c >> M;
        if (hi == 0)
            return xf_fin(false, c & ((1u << M) - 1), 1 - mk_bias(k) - M);
        return xf_fin(false, (c & ((1u << M) - 1)) + (1u << M), (int)hi - mk_bias(k) - M);
    }
    // miniDecF
    static inline XF mk_dec(uint32_t k, uint32_t b)
    {
        int nb = mk_bits(k);
        if (k == T_UE5M3)
        {
            b &= 255;
            return mk_isnan(k, b) ? xf_nan() : mk_mag(k, b);
        }
        uint32_t c = b & ((1u << (nb - 1)) - 1);
        bool s = (b >> (nb - 1)) & 1;
        if (mk_isnan(k, c))
            return xf_nan();
        if (k == T_E5M2 && c == 124)
            return xf_inf(s);
        XF r = mk_mag(k, c);
        r.s = s;
        return r;
    }
    // magnitude code (miniEncR), saturating at the top code
    static inline uint32_t mk_enc_mag(uint32_t k, XF a, int rm, uint32_t rb, int nb)
    {
        if (a.m == 0)
            return 0;
        int M = mk_M(k), bias = mk_bias(k);
        XF r = xround(a.s, a.m, a.e, false, M + 1, 1 - bias, 1 << 20, rm, rb, nb);
        int64_t raw;
        if (r.m < (1ull << M))
            raw = (int64_t)r.m;
        else
            raw = ((int64_t)(r.e + M + bias) << M) + (int64_t)(r.m - (1ull << M));
        return raw > mk_top(k) ? (uint32_t)mk_top(k) : (uint32_t)raw;
    }
    // miniFromRS (rm: signed PTX mode, RM_RS stochastic)
    static inline uint32_t mk_from(uint32_t k, XF x, bool relu, int rm, uint32_t rb, int nb)
    {
        if (x.c == XF_NAN)
            return (k == T_E4M3 || k == T_E5M2) ? 127 : k == T_UE5M3 ? 255 : mk_top(k);
        if ((relu || k == T_UE5M3) && x.s)
            return 0;
        uint32_t sg = (x.s && k != T_UE5M3) ? 1u << (mk_bits(k) - 1) : 0;
        if (x.c == XF_INF)
            return sg | mk_top(k);
        return sg | mk_enc_mag(k, x, rm, rb, nb);
    }
    static inline XF xf_scale(XF x, int n)
    {
        if (x.c == XF_FIN)
            x.e += n;
        return x;
    }
    // miniCvt: scale s (ue8m0 bits, 127 = none), .satfinite / .relu / .pzo
    static inline uint32_t mk_cvt(uint32_t k, XF x, uint32_t s, int rm, bool relu, bool satf, bool pzo)
    {
        XF y = s == 255 ? xf_nan() : xf_scale(x, 127 - (int)s);
        if (k == T_UE5M3 && satf && !relu && y.c != XF_NAN && y.s)
        {
            XF top = mk_mag(k, mk_top(k));
            bool over = y.c == XF_INF;
            if (!over && y.m != 0)
            {
                // |y| > top: compare at a common exponent
                int e = y.e < top.e ? y.e : top.e;
                int dy = y.e - e, dt = top.e - e;
                int ly = xf_bitlen(y.m) + dy, lt = xf_bitlen(top.m) + dt;
                if (ly != lt)
                    over = ly > lt;
                else
                    over = (dy < 64 ? (y.m << dy) : 0) > (dt < 64 ? (top.m << dt) : 0);
            }
            if (over)
                return mk_top(k);
        }
        uint32_t c = mk_from(k, y, relu, rm, 0, 0);
        if (pzo && k != T_UE5M3 && c == (1u << (mk_bits(k) - 1)))
            c = 0;
        return c;
    }
    // flushIn of the .scaled::n1::ue8m0 forms
    static inline XF xf_flush_below(XF x, int mn)
    {
        if (x.c == XF_FIN && x.m != 0 && xf_cmp_p2(x, mn) < 0)
            return xf_fin(false, 0, 0);
        return x;
    }

    // ue8m0 (ueEnc): md 0 .rz, 1 .rp, 2 .rn
    static inline uint32_t ue_enc(XF x, int md, bool satf)
    {
        if (x.c == XF_NAN)
            return 255;
        if (satf && (x.c == XF_INF || (x.m != 0 && xf_cmp_p2(x, 127) > 0)))
            return 254;
        if (x.s || xf_zero(x))
            return 0;
        if (x.c == XF_INF)
            return 254;
        int nbits = xf_bitlen(x.m);
        int e = nbits - 1 + x.e;
        bool pow2 = (x.m & (x.m - 1)) == 0;
        int r;
        if (e < -160)
            r = -160;
        else if (e >= 130)
            r = e;
        else if (md == 1)
            r = pow2 ? e : e + 1;
        else if (md == 2)
        {
            // compare with 1.5 * 2^e
            int h;
            if (nbits < 2)
                h = -1;
            else
            {
                uint64_t frac = x.m - (1ull << (nbits - 1));
                uint64_t half = 1ull << (nbits - 2);
                h = frac > half ? 1 : frac == half ? 0 : -1;
            }
            r = (h > 0 || (h == 0 && ((e + 127) & 1))) ? e + 1 : e;
        }
        else
            r = e;
        r += 127;
        return r < 0 ? 0 : r > 254 ? 254 : (uint32_t)r;
    }
    // s2f6: md magnitude-free signed mode (0 rn, 1 rz, 2 rm, 3 rp)
    static inline uint32_t s2f6_code(XF x, int rm, bool relu)
    {
        if (x.c == XF_NAN)
            return 127;
        if (relu && x.s)
            return 0;
        if (x.c == XF_INF)
            return x.s ? 128 : 127;
        bool big = false;
        uint64_t n = xf_rint_mag(xf_scale(x, 6), rm, &big);
        if (x.s)
            return (uint32_t)(0 - ((big || n > 128) ? 128 : n)) & 255;
        return (big || n > 127) ? 127 : (uint32_t)n;
    }

    // ---------------------------------------------------------------- generic element conversion
    // ctrl: 0-4 dst type, 5-9 src type, 10-12 rounding (RM_*), 13 integer rounding (.rni ...),
    // 14 .ftz, 15 .sat, 16 .relu, 17 .satfinite, 18 .pzo, 19 .scaled::n1, 20 .scaled::n2
    enum
    {
        C_IRND = 1u << 13,
        C_FTZ = 1u << 14,
        C_SAT = 1u << 15,
        C_RELU = 1u << 16,
        C_SATF = 1u << 17,
        C_PZO = 1u << 18,
        C_N1 = 1u << 19,
        C_N2 = 1u << 20
    };
    static inline XF xdec_any(uint64_t v, uint32_t t, bool ftz)
    {
        if (t_ieee(t))
            return xdec_t(v, t, ftz && t == T_F32);
        if (t_mini(t))
            return mk_dec(t, (uint32_t)v);
        if (t == T_UE8M0)
        {
            uint32_t c = v & 255;
            return c == 255 ? xf_nan() : xf_fin(false, 1, (int)c - 127);
        }
        if (t == T_S2F6)
        {
            int32_t c = (int8_t)(uint8_t)v;
            return c < 0 ? xf_fin(true, (uint64_t)(-c), -6) : xf_fin(false, (uint64_t)c, -6);
        }
        return xdec_int(v, t);
    }
    // the smallest normal exponent below which .scaled::n1 flushes a source lane
    static inline int flush_min(uint32_t t) { return t == T_F16 ? -14 : -126; }
    // convert one element; `scale` is the ue8m0 scale of the element (127 = none)
    static inline uint64_t xcvt(uint64_t v, uint32_t ctrl, uint32_t scale, uint32_t rb, int nb)
    {
        uint32_t dt = ctrl & 31, st = (ctrl >> 5) & 31;
        int rm = (ctrl >> 10) & 7;
        bool ftz = ctrl & C_FTZ;
        XF x = xdec_any(v, st, ftz);
        if (t_ieee(dt))
        {
            uint32_t pf = ((ftz && dt == T_F32) ? PF_FTZ : 0) | ((ctrl & C_SAT) ? PF_SAT : 0) |
                          ((ctrl & C_RELU) ? PF_RELU : 0) | ((ctrl & C_SATF) ? PF_SATF : 0);
            if (!t_ieee(st) && !t_int(st))
            {
                // up-conversion from a mini format (upF): round to nearest, then scale
                if (scale == 255)
                    x = xf_nan();
                else
                    x = xf_scale(x, (int)scale - 127);
                return xfin(x, dt, RM_RN, pf, 0, 0, false);
            }
            if ((ctrl & C_IRND) && !t_int(st))
                return xpost(xround_t(xf_rndint(x, rm), dt, RM_RN, 0, 0, false), dt, pf);
            uint64_t r = xfin(x, dt, rm, pf, rb, nb, false);
            if ((ctrl & C_PZO) && r == (1ull << (t_cw(dt) - 1)))
                r = 0;
            return r;
        }
        if (t_int(dt))
        {
            if (t_int(st))
            {
                // integer to integer: truncate, or clamp with .sat
                int w = t_iw(dt);
                uint64_t wmask = w == 64 ? ~0ull : (1ull << w) - 1;
                if (!(ctrl & C_SAT))
                    return (x.s ? (0 - x.m) : x.m) & wmask;
                if (t_signed(dt))
                {
                    uint64_t smax = (1ull << (w - 1)) - 1;
                    if (x.s)
                        return x.m > smax + 1 ? (smax + 1) : ((0 - x.m) & wmask);
                    return x.m > smax ? smax : x.m;
                }
                return x.s ? 0 : x.m > wmask ? wmask : x.m;
            }
            return xf2i(x, (ctrl & C_IRND) ? rm : RM_RZ, dt, st == T_F64);
        }
        if (ctrl & C_N1)
            x = xf_flush_below(x, flush_min(st));
        if (t_mini(dt))
        {
            if (rm == RM_RS)
            {
                uint32_t c = mk_from(dt, x, ctrl & C_RELU, RM_RS, rb, nb);
                if ((ctrl & C_PZO) && dt != T_UE5M3 && c == (1u << (mk_bits(dt) - 1)))
                    c = 0;
                return c;
            }
            return mk_cvt(dt, x, scale, rm, ctrl & C_RELU, ctrl & C_SATF, ctrl & C_PZO);
        }
        if (dt == T_UE8M0)
            return ue_enc(x, rm == RM_RP ? 1 : rm == RM_RN ? 2 : 0, ctrl & C_SATF);
        // s2f6
        return s2f6_code(scale == 255 ? xf_nan() : xf_scale(x, 127 - (int)scale), rm, ctrl & C_RELU);
    }
    static inline uint32_t scale_hi(uint32_t ctrl, uint32_t s) { return (ctrl & C_N2) ? (s >> 8) & 255 : s & 255; }
    static inline uint32_t elem_bits(uint32_t t) { return t == T_E2M1 ? 4 : 8; }

    // scalar cvt d, a
    // Fast paths of the common conversions, bit-exact with xcvt: ctrl is an immediate, so after
    // inlining only one path remains. The hardware's conversions are exact here (half subnormals
    // are kept; an f32 subnormal rounds to a signed zero in f16 anyway; doubles are soft-f64);
    // NaN results are the canonical NaN, saturation is explicit.
    static inline bool cvt_fast(uint64_t a, uint32_t ctrl, uint64_t *r)
    {
        uint32_t c = ctrl & ~C_FTZ;
        if (c == (T_F32 | T_F16 << 5)) // cvt.f32.f16
        {
            uint16_t h = (uint16_t)a;
            if ((h & 0x7C00) == 0x7C00 && (h & 0x3FF))
                *r = 0x7FFFFFFF;
            else
                *r = __builtin_bit_cast(uint32_t, (float)__builtin_bit_cast(_Float16, h));
            return true;
        }
        if (c == (T_F16 | T_F32 << 5)) // cvt.rn.f16.f32
        {
            uint32_t u = (uint32_t)a;
            if ((u & 0x7FFFFFFF) > 0x7F800000)
                *r = 0x7FFF;
            else
                *r = __builtin_bit_cast(uint16_t, (_Float16)__builtin_bit_cast(float, u));
            return true;
        }
        if (c == (T_F64 | T_F32 << 5) && !(ctrl & C_FTZ)) // cvt.f64.f32
        {
            uint32_t u = (uint32_t)a;
            if ((u & 0x7FFFFFFF) > 0x7F800000)
                *r = 0x7FFFFFFFFFFFFFFFull;
            else
                *r = __builtin_bit_cast(uint64_t, (double)__builtin_bit_cast(float, u));
            return true;
        }
        if (c == (T_S32 | T_F32 << 5 | RM_RZ << 10 | C_IRND) || c == (T_U32 | T_F32 << 5 | RM_RZ << 10 | C_IRND))
        {
            // cvt.rzi.{s32,u32}.f32: truncate, saturating, NaN -> 0 (subnormals truncate to 0 either way)
            float f = __builtin_bit_cast(float, (uint32_t)a);
            bool sgn = (c & 31) == T_S32;
            uint32_t v;
            if (f != f)
                v = 0;
            else if (sgn)
                v = f >= 2147483648.0f ? 0x7FFFFFFFu : f <= -2147483648.0f ? 0x80000000u : (uint32_t)(int32_t)f;
            else
                v = f >= 4294967296.0f ? 0xFFFFFFFFu : !(f > 0.0f) ? 0u : (uint32_t)f;
            *r = v;
            return true;
        }
        return false;
    }
#define CVT_SCALAR(DN, DT, SN, ST) \
    DT FUNC(cvt_##DN##_##SN)(ST a, uint32_t ctrl) { uint64_t fr; if (cvt_fast((uint64_t)a, ctrl, &fr)) return (DT)fr; return (DT)xcvt((uint64_t)a, ctrl, 127, 0, 0); }
#define CVT_SCALAR_S(SN, ST)        \
    CVT_SCALAR(8, uint8_t, SN, ST)   \
    CVT_SCALAR(16, uint16_t, SN, ST) \
    CVT_SCALAR(32, uint32_t, SN, ST) \
    CVT_SCALAR(64, uint64_t, SN, ST)
    CVT_SCALAR_S(8, uint8_t)
    CVT_SCALAR_S(16, uint16_t)
    CVT_SCALAR_S(32, uint32_t)
    CVT_SCALAR_S(64, uint64_t)

    // cvt d, a, b{, scale} with a and b scalars of the source type: a in the upper element.
    // Destinations: f16x2 / bf16x2 (32 bits), x2 of an 8-bit format (16 bits), e2m1x2 (8 bits).
#define CVT_PAIR(DN, DT, SN, ST) CVT_PAIR1(DN, DT, SN, ST, 8, uint8_t) CVT_PAIR1(DN, DT, SN, ST, 16, uint16_t)
#define CVT_PAIR1(DN, DT, SN, ST, CN, CT)                                                                       \
    DT FUNC(cvt_pair_##DN##_##SN##_##CN)(ST a, ST b, CT s, uint32_t ctrl)                                         \
    {                                                                                                            \
        uint32_t dt = ctrl & 31;                                                                                 \
        uint32_t w = t_ieee(dt) ? 16 : elem_bits(dt);                                                            \
        uint32_t hi = (uint32_t)xcvt((uint64_t)a, ctrl, scale_hi(ctrl, s), 0, 0);                                \
        uint32_t lo = (uint32_t)xcvt((uint64_t)b, ctrl, s & 255, 0, 0);                                          \
        return (DT)((hi << w) | lo);                                                                             \
    }
#define CVT_PAIR_S(SN, ST)            \
    CVT_PAIR(8, uint8_t, SN, ST)   \
    CVT_PAIR(16, uint16_t, SN, ST) \
    CVT_PAIR(32, uint32_t, SN, ST)
    CVT_PAIR_S(16, uint16_t)
    CVT_PAIR_S(32, uint32_t)
    CVT_PAIR_S(64, uint64_t)

    // cvt d, a{, scale} with a packed source pair (f16x2 / bf16x2 or x2 of a mini format) and a
    // packed destination pair: element 0 in the low bits
    static inline uint32_t cvt_x2_impl(uint32_t a, uint32_t s, uint32_t ctrl)
    {
        uint32_t dt = ctrl & 31, st = (ctrl >> 5) & 31;
        uint32_t sw = t_ieee(st) ? 16 : elem_bits(st);
        uint32_t dw = t_ieee(dt) ? 16 : elem_bits(dt);
        uint32_t smask = (1u << sw) - 1;
        // ue8m0 from a pair: the lanes are read as bf16
        uint32_t ctrl2 = (dt == T_UE8M0 && st == T_F16) ? ((ctrl & ~(31u << 5)) | (T_BF16 << 5)) : ctrl;
        uint32_t lo = (uint32_t)xcvt(a & smask, ctrl2, s & 255, 0, 0);
        uint32_t hi = (uint32_t)xcvt((a >> sw) & smask, ctrl2, scale_hi(ctrl, s), 0, 0);
        return (hi << dw) | lo;
    }
#define CVT_X2(DN, DT, SN, ST) \
    DT FUNC(cvt_x2_##DN##_##SN##_8)(ST a, uint8_t s, uint32_t ctrl) { return (DT)cvt_x2_impl(a, s, ctrl); } \
    DT FUNC(cvt_x2_##DN##_##SN##_16)(ST a, uint16_t s, uint32_t ctrl) { return (DT)cvt_x2_impl(a, s, ctrl); }
#define CVT_X2_S(SN, ST)            \
    CVT_X2(8, uint8_t, SN, ST)   \
    CVT_X2(16, uint16_t, SN, ST) \
    CVT_X2(32, uint32_t, SN, ST)
    CVT_X2_S(8, uint8_t)
    CVT_X2_S(16, uint16_t)
    CVT_X2_S(32, uint32_t)

    // cvt.rs{.relu}{.satfinite}.{f16x2,bf16x2}.f32 d, a, b, rbits
    uint32_t FUNC(cvt_rs_pair)(uint32_t a, uint32_t b, uint32_t r, uint32_t ctrl)
    {
        uint32_t dt = ctrl & 31;
        int k = dt == T_F16 ? 13 : 16;
        uint32_t km = (1u << k) - 1;
        uint32_t hi = (uint32_t)xcvt(a, ctrl, 127, (r >> 16) & km, k);
        uint32_t lo = (uint32_t)xcvt(b, ctrl, 127, r & km, k);
        return (hi << 16) | lo;
    }
    // cvt.rs{.relu}.satfinite.{e4m3x4,...,e2m1x4}.f32 d, {a, b, e, f}, rbits: a in the upper bits
    static inline uint32_t cvt_rs_x4(uint32_t a, uint32_t b, uint32_t e, uint32_t f, uint32_t r, uint32_t ctrl)
    {
        uint32_t dt = ctrl & 31;
        bool e2m1 = dt == T_E2M1;
        uint32_t sw = e2m1 ? 4 : 8;
        uint32_t v[4] = {a, b, e, f};
        uint32_t out = 0;
        for (int i = 0; i < 4; i++)
        {
            uint32_t rb = e2m1 ? (r >> (i == 0 ? 24 : i == 1 ? 8 : i == 2 ? 16 : 0)) & 255 : (r >> (i < 2 ? 16 : 0)) & 65535;
            uint32_t c = (uint32_t)xcvt(v[i], ctrl, 127, rb, e2m1 ? 8 : 16);
            out |= c << (sw * (3 - i));
        }
        return out;
    }

    uint16_t FUNC(cvt_rs_x4_16)(uint32_t a, uint32_t b, uint32_t e, uint32_t f, uint32_t r, uint32_t ctrl) { return (uint16_t)cvt_rs_x4(a, b, e, f, r, ctrl); }
    uint32_t FUNC(cvt_rs_x4_32)(uint32_t a, uint32_t b, uint32_t e, uint32_t f, uint32_t r, uint32_t ctrl) { return cvt_rs_x4(a, b, e, f, r, ctrl); }

    // ZLUDA's own lowering of cvt.rn.satfinite.{e4m3x2,e5m2x2}.f32 and cvt.rn.f16x2.{e4m3x2,e5m2x2}
    uint16_t FUNC(cvt_rn_satfinite_e4m3x2_f32)(float a, float b)
    {
        uint32_t ctrl = T_E4M3 | (T_F32 << 5) | C_SATF;
        return (uint16_t)((xcvt(__builtin_bit_cast(uint32_t, a), ctrl, 127, 0, 0) << 8) | xcvt(__builtin_bit_cast(uint32_t, b), ctrl, 127, 0, 0));
    }
    uint16_t FUNC(cvt_rn_satfinite_e5m2x2_f32)(float a, float b)
    {
        uint32_t ctrl = T_E5M2 | (T_F32 << 5) | C_SATF;
        return (uint16_t)((xcvt(__builtin_bit_cast(uint32_t, a), ctrl, 127, 0, 0) << 8) | xcvt(__builtin_bit_cast(uint32_t, b), ctrl, 127, 0, 0));
    }
    uint32_t FUNC(cvt_rn_f16x2_e4m3x2)(uint16_t a) { return cvt_x2_impl(a, 127, T_F16 | (T_E4M3 << 5)); }
    uint32_t FUNC(cvt_rn_f16x2_e5m2x2)(uint16_t a) { return cvt_x2_impl(a, 127, T_F16 | (T_E5M2 << 5)); }

    // cvt.pack.sat.{u16,s16}.s32 d, a, b and cvt.pack.sat.{u8,s8}.s32.b32 d, a, b, c
    static inline uint32_t pack_sat(int32_t v, bool sg, int w)
    {
        int64_t lo = sg ? -(1ll << (w - 1)) : 0, hi = sg ? (1ll << (w - 1)) - 1 : (1ll << w) - 1;
        int64_t x = v < lo ? lo : v > hi ? hi : v;
        return (uint32_t)x & ((1u << w) - 1);
    }
    uint32_t FUNC(cvt_pack_sat)(int32_t a, int32_t b, uint32_t c, uint32_t ctrl)
    {
        bool sg = ctrl & 1;
        int w = (ctrl & 2) ? 8 : 16;
        uint32_t r = (pack_sat(a, sg, w) << w) | pack_sat(b, sg, w);
        if (w == 8)
            r |= c << 16;
        return r;
    }
