// Included inside extern "C" of zluda_ptx_metal_impl.cpp: lop3, video, testp, slct.
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
