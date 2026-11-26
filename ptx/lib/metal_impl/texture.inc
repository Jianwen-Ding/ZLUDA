// Included inside extern "C" of zluda_ptx_metal_impl.cpp: textures, samplers and surfaces in software (tex, tld4, txq,
// suld, sust, sured, suq, istypep), lowered by ptx_parser/src/helper_table/texture.rs.  A handle is the global
// address of a descriptor object in PTX-formalization's harness encoding (16 little-endian u64 words, built by the
// .texture / .sampler / .surface directives of zluda_metal/tests/ptxconf/ptxconf.py); the texels are read from the
// texture's buffer and filtered exactly as sail/ptx/ptx_tex.sail computes (in double precision here: the model's
// 4400-bit working values are exact for the usual inputs; the result is rounded to the destination format once).
#define TX_GLOBAL __attribute__((address_space(1)))
    enum
    {
        TX_MAGIC = 1415075924u,
        SM_MAGIC = 1396920397u,
        SF_MAGIC = 1398035526u
    };
    static inline uint64_t tdW(uint64_t h, int i) { return ((const volatile TX_GLOBAL uint64_t *)(uintptr_t)h)[i]; }
    static inline uint64_t tx_rd(uint64_t a, int n)
    {
        const volatile TX_GLOBAL uint8_t *p = (const volatile TX_GLOBAL uint8_t *)(uintptr_t)a;
        uint64_t v = 0;
        for (int i = 0; i < n; i++)
            v |= (uint64_t)p[i] << (8 * i);
        return v;
    }
    static inline void tx_wr(uint64_t a, int n, uint64_t v)
    {
        volatile TX_GLOBAL uint8_t *p = (volatile TX_GLOBAL uint8_t *)(uintptr_t)a;
        for (int i = 0; i < n; i++)
            p[i] = (uint8_t)(v >> (8 * i));
    }
    static inline int64_t tx_max64(int64_t a, int64_t b) { return a > b ? a : b; }
    static inline int64_t tx_min64(int64_t a, int64_t b) { return a < b ? a : b; }

    // ---------------------------------------------------------------- doubles from / to bit patterns
    static inline double dbits(uint64_t b) { return __builtin_bit_cast(double, b); }
    static inline uint64_t bitsd(double d) { return __builtin_bit_cast(uint64_t, d); }
    // a binary format of precision p and e exponent bits (f32: 24, 8; f16: 11, 5) -> double (exact)
    static inline double tx_dec(uint64_t b, int p, int eb)
    {
        int w = p + eb;
        uint64_t s = (b >> (w - 1)) & 1, e = (b >> (p - 1)) & ((1u << eb) - 1), f = b & ((1ull << (p - 1)) - 1);
        int bias = (1 << (eb - 1)) - 1;
        if (e == (uint64_t)((1 << eb) - 1))
            return dbits((s << 63) | (0x7FFull << 52) | (f ? (1ull << 51) | (f << (53 - p)) : 0));
        if (e == 0)
        {
            if (f == 0)
                return dbits(s << 63);
            int k = 0;
            while (!((f >> (p - 1)) & 1))
            {
                f <<= 1;
                k++;
            }
            e = 1 - k;
            f &= (1ull << (p - 1)) - 1;
            return dbits((s << 63) | ((uint64_t)((int64_t)e - bias + 1023) << 52) | (f << (53 - p)));
        }
        return dbits((s << 63) | ((uint64_t)((int64_t)e - bias + 1023) << 52) | (f << (53 - p)));
    }
    static inline double f32d(uint32_t b) { return tx_dec(b, 24, 8); }
    // double -> the bits of a binary format (round to nearest even, subnormals, overflow to infinity)
    static inline uint64_t tx_enc(double x, int p, int eb)
    {
        uint64_t b = bitsd(x);
        int w = p + eb;
        uint64_t s = b >> 63;
        int de = (int)((b >> 52) & 0x7FF);
        uint64_t df = b & ((1ull << 52) - 1);
        uint64_t emax = (1u << eb) - 1;
        int bias = (1 << (eb - 1)) - 1;
        if (de == 0x7FF)
            return (s << (w - 1)) | (emax << (p - 1)) | (df ? (1ull << (p - 2)) | (df >> (53 - p)) : 0);
        if (de == 0)
            return s << (w - 1); // double subnormals are far below any target's range
        int e = de - 1023;
        uint64_t m = (1ull << 52) | df; // 53 bits
        int sh;
        if (e >= 1 - bias)
            sh = 53 - p;
        else
            sh = 53 - p + (1 - bias - e);
        uint64_t q, rem, half;
        if (sh >= 64)
        {
            q = 0;
            rem = m;
            half = ~0ull;
            if (sh == 64)
                half = 1ull << 63;
        }
        else
        {
            q = m >> sh;
            rem = m & ((1ull << sh) - 1);
            half = 1ull << (sh - 1);
        }
        if (rem > half || (rem == half && (q & 1)))
            q++;
        uint64_t r;
        if (e >= 1 - bias)
        {
            // q has p bits (or p + 1 after the carry)
            int64_t be = e + bias;
            if (q >> p)
            {
                q >>= 1;
                be++;
            }
            if (be >= (int64_t)emax)
                return (s << (w - 1)) | (emax << (p - 1));
            r = ((uint64_t)be << (p - 1)) | (q & ((1ull << (p - 1)) - 1));
        }
        else
            r = q; // subnormal (a carry into the exponent field is the smallest normal)
        return (s << (w - 1)) | r;
    }
    static inline double tx_floor(double x)
    {
        uint64_t b = bitsd(x);
        int e = (int)((b >> 52) & 0x7FF) - 1023;
        if (e >= 52)
            return x;
        if (e < 0)
            return (b >> 63) && (b << 1) ? -1.0 : ((b >> 63) ? -0.0 : 0.0);
        uint64_t mask = (1ull << (52 - e)) - 1;
        if (!(b & mask))
            return x;
        double t = dbits(b & ~mask);
        return (b >> 63) ? t - 1.0 : t;
    }
    static inline int64_t tx_flo(double x) { return (int64_t)tx_floor(x); }
    static inline int64_t tx_trunc(double x) { return (int64_t)x; }
    static inline double tx_qw(double xb) { return (double)tx_flo((xb - tx_floor(xb)) * 256.0 + 0.5) / 256.0; }
    static inline double tx_abs(double x) { return x < 0 ? -x : x; }
    // log2 of a positive finite double (exact for powers of two)
    static inline double tx_log2(double x)
    {
        uint64_t b = bitsd(x);
        int k = (int)((b >> 52) & 0x7FF) - 1023;
        double m = dbits((b & ((1ull << 52) - 1)) | (1023ull << 52));
        if (m > 1.4142135623730951)
        {
            m = m * 0.5;
            k++;
        }
        double t = (m - 1.0) / (m + 1.0), t2 = t * t, s = 0, p = t;
        for (int i = 1; i <= 25; i += 2)
        {
            s += p / (double)i;
            p *= t2;
        }
        return (double)k + 2.0 * s * 1.4426950408889634;
    }

    // ---------------------------------------------------------------- texture fields
    struct TxD
    {
        uint64_t h, s; // texture and sampler handles (s == 0: none)
    };
    static inline int64_t txCh(TxD t) { return (int64_t)tdW(t.h, 3); }
    static inline int64_t txLayers(TxD t) { return tx_max64(1, (int64_t)tdW(t.h, 8)); }
    static inline int64_t txDim(TxD t, int i) { return tx_max64(1, (int64_t)tdW(t.h, 9 + i)); }
    static inline int64_t txLevels(TxD t) { return tx_max64(1, (int64_t)tdW(t.h, 12)); }
    static inline int64_t lvD(TxD t, int64_t lv, int i) { return tx_max64(1, txDim(t, i) >> lv); }
    static inline int64_t lvTexels(TxD t, int64_t lv) { return txLayers(t) * lvD(t, lv, 0) * lvD(t, lv, 1) * lvD(t, lv, 2); }
    static inline int64_t txSamples(TxD t) { return (int64_t)tdW(t.h, 14); }
    static inline int64_t lvOff(TxD t, int64_t lv)
    {
        int64_t o = 0;
        for (int64_t l = 0; l < lv && l < 16; l++)
            o += lvTexels(t, l);
        return o;
    }
    static inline int64_t effMip(TxD t) { return (int64_t)(t.s ? tdW(t.s, 4) : tdW(t.h, 15)); }
    static inline int64_t minLodW(TxD t) { return t.s ? (int64_t)tdW(t.s, 5) : 0; }
    static inline int64_t maxLodW(TxD t) { return t.s ? (int64_t)tdW(t.s, 6) : 1048576; }
    static inline int txEbOf(uint64_t c) { return (c == 0 || c == 20) ? 1 : (c == 1 || c == 21 || c == 6) ? 2 : 4; }
    static inline int64_t effFilter(TxD t) { return (int64_t)(t.s ? tdW(t.s, 1) : tdW(t.h, 4)); }
    static inline int64_t effAddr(TxD t) { return (int64_t)(t.s ? tdW(t.s, 2) : tdW(t.h, 5)); }
    static inline int64_t effNorm(TxD t) { return (t.s && tdW(t.s, 3) == 0) ? 0 : (int64_t)tdW(t.h, 6); }

    static inline int64_t adIdx(int64_t md, int64_t i, int64_t n)
    {
        switch (md)
        {
        case 0: return ((i % n) + n) % n;
        case 3:
        {
            int64_t m = ((i % (2 * n)) + 2 * n) % (2 * n);
            return m < n ? m : 2 * n - m - 1;
        }
        case 2: return (i >= 0 && i < n) ? i : -1;
        default: return tx_min64(tx_max64(i, 0), n - 1);
        }
    }
    static inline int64_t amEff(int64_t am, int64_t nm) { return (nm != 1 && (am == 0 || am == 3)) ? 1 : am; }
    static inline int codeBits(uint64_t c) { return (c == 0 || c == 20) ? 8 : (c == 1 || c == 21) ? 16 : 32; }
    static inline int64_t codeVal(uint64_t c, uint64_t v)
    {
        int b = codeBits(c);
        if (c == 20 || c == 21 || c == 3 || c == 5)
            return (int64_t)(v << (64 - b)) >> (64 - b);
        return (int64_t)v;
    }
    static inline int64_t codeMax(uint64_t c)
    {
        int b = codeBits(c);
        return (c == 0 || c == 1 || c == 2 || c == 4) ? (int64_t)((1ull << b) - 1) : (int64_t)((1ull << (b - 1)) - 1);
    }
    static inline double convF(uint64_t c, uint64_t v, uint64_t n)
    {
        if (c == 6)
            return tx_dec(v, 11, 5);
        if (c == 7)
            return f32d((uint32_t)v);
        if (n == 0)
            return (double)codeVal(c, v);
        double r = (double)codeVal(c, v) / (double)codeMax(c);
        return r >= -1.0 ? r : -1.0;
    }
    static inline double texC(TxD t, uint64_t a, int c)
    {
        uint64_t code = tdW(t.h, 2);
        int eb = txEbOf(code);
        if (c < txCh(t))
            return convF(code, tx_rd(a + (uint64_t)c * eb, eb), tdW(t.h, 7));
        return c == 3 ? 1.0 : 0.0;
    }
    // Q: level, offsets x y z, depth compare, reference bits, sample
    struct TxQ
    {
        int64_t lv, o[3];
        bool dc;
        uint32_t ref;
        int64_t smp;
    };
    struct V4
    {
        double v[4];
    };
    static inline V4 texel(TxD t, TxQ q, int64_t i, int64_t j, int64_t k, int64_t l)
    {
        V4 r = {{0, 0, 0, 0}};
        if (!(i < 0 || j < 0 || k < 0))
        {
            int64_t ti = lvOff(t, q.lv) + ((l * lvD(t, q.lv, 2) + k) * lvD(t, q.lv, 1) + j) * lvD(t, q.lv, 0) + i;
            int eb = txEbOf(tdW(t.h, 2));
            uint64_t a = tdW(t.h, 1) + (uint64_t)((ti * tx_max64(1, txSamples(t)) + q.smp) * txCh(t) * eb);
            for (int c = 0; c < 4; c++)
                r.v[c] = texC(t, a, c);
        }
        if (!q.dc)
            return r;
        double ref = f32d(q.ref);
        V4 d = {{(ref <= r.v[0]) ? 1.0 : 0.0, 0, 0, 1.0}};
        return d;
    }
    static inline double lerpF(double a, double b, double w) { return (1.0 - w) * a + w * b; }
    static inline V4 lerp4(V4 a, V4 b, double w)
    {
        V4 r;
        for (int c = 0; c < 4; c++)
            r.v[c] = lerpF(a.v[c], b.v[c], w);
        return r;
    }
    static inline double posF(double x, int64_t n, int64_t nm) { return nm == 1 ? x * (double)n : x; }
    static inline int64_t tIdx(TxD t, TxQ q, int64_t am, int64_t nm, double x, int64_t d, int dim, int lin)
    {
        double p = posF(x, lvD(t, q.lv, dim), nm);
        return adIdx(amEff(am, nm), tx_flo(lin ? p - 0.5 : p) + d + q.o[dim], lvD(t, q.lv, dim));
    }
    static inline double tW(TxD t, TxQ q, int64_t nm, double x, int dim) { return tx_qw(posF(x, lvD(t, q.lv, dim), nm) - 0.5); }
    static inline V4 sample(TxD t, TxQ q, int64_t f, int64_t am, int64_t nm, double x, double y, double z, int64_t l, int r)
    {
        if (f == 1)
        {
            int64_t ix0 = tIdx(t, q, am, nm, x, 0, 0, 1), ix1 = tIdx(t, q, am, nm, x, 1, 0, 1);
            int64_t jy0 = r >= 2 ? tIdx(t, q, am, nm, y, 0, 1, 1) : 0, jy1 = r >= 2 ? tIdx(t, q, am, nm, y, 1, 1, 1) : 0;
            int64_t kz0 = r >= 3 ? tIdx(t, q, am, nm, z, 0, 2, 1) : 0, kz1 = r >= 3 ? tIdx(t, q, am, nm, z, 1, 2, 1) : 0;
            double wx = tW(t, q, nm, x, 0), wy = r >= 2 ? tW(t, q, nm, y, 1) : 0.0, wz = r >= 3 ? tW(t, q, nm, z, 2) : 0.0;
            V4 a = lerp4(lerp4(texel(t, q, ix0, jy0, kz0, l), texel(t, q, ix1, jy0, kz0, l), wx),
                         lerp4(texel(t, q, ix0, jy1, kz0, l), texel(t, q, ix1, jy1, kz0, l), wx), wy);
            if (r < 3)
                return lerp4(a, a, 0.0);
            V4 b = lerp4(lerp4(texel(t, q, ix0, jy0, kz1, l), texel(t, q, ix1, jy0, kz1, l), wx),
                         lerp4(texel(t, q, ix0, jy1, kz1, l), texel(t, q, ix1, jy1, kz1, l), wx), wy);
            return lerp4(a, b, wz);
        }
        return texel(t, q, tIdx(t, q, am, nm, x, 0, 0, 0), r >= 2 ? tIdx(t, q, am, nm, y, 0, 1, 0) : 0,
                     r >= 3 ? tIdx(t, q, am, nm, z, 0, 2, 0) : 0, l);
    }
    static inline double tx_minF(double a, double b) { return a <= b ? a : b; }
    static inline double tx_maxF(double a, double b) { return a >= b ? a : b; }
    static inline double lamClamp(TxD t, double lam)
    {
        return tx_minF(tx_maxF(tx_maxF(lam, (double)minLodW(t) / 256.0), 0.0),
                       tx_minF((double)maxLodW(t) / 256.0, (double)(txLevels(t) - 1)));
    }
    static inline V4 mipSample(TxD t, bool uselod, double lam, TxQ q, int64_t f, int64_t am, int64_t nm, double x, double y,
                               double z, int64_t l, int r)
    {
        if (!uselod)
            return sample(t, q, f, am, nm, x, y, z, l, r);
        double lc = lamClamp(t, lam);
        if (effMip(t) == 1)
        {
            TxQ q0 = q, q1 = q;
            q0.lv = tx_flo(lc);
            q1.lv = tx_min64(tx_flo(lc) + 1, txLevels(t) - 1);
            return lerp4(sample(t, q0, f, am, nm, x, y, z, l, r), sample(t, q1, f, am, nm, x, y, z, l, r), tx_qw(lc));
        }
        TxQ q0 = q;
        q0.lv = tx_min64(tx_flo(lc + 0.5), txLevels(t) - 1);
        return sample(t, q0, f, am, nm, x, y, z, l, r);
    }
    static inline double lamLog(double s) { return s > 0 ? tx_log2(s) / 2.0 : -1000.0; }
    static inline double gradLam(TxD t, int64_t nm, int r, const double *dx, const double *dy, double sd)
    {
        double lx = 0, ly = 0;
        for (int i = r - 1; i >= 0; i--)
        {
            double sc = nm == 1 ? (double)txDim(t, i) / sd : 1.0 / sd;
            double tx = dx[i] * sc, ty = dy[i] * sc;
            lx = tx * tx + lx;
            ly = ty * ty + ly;
        }
        return lamLog(tx_maxF(lx, ly));
    }
    static inline int cubeFace(double x, double y, double z)
    {
        double ax = tx_abs(x), ay = tx_abs(y), az = tx_abs(z);
        if (ax >= ay && ax >= az)
            return x >= 0 ? 0 : 1;
        if (ax < ay && ay >= az)
            return y >= 0 ? 2 : 3;
        return z >= 0 ? 4 : 5;
    }
    static inline double cubeMa(double x, double y, double z)
    {
        int f = cubeFace(x, y, z);
        return f < 2 ? tx_abs(x) : f < 4 ? tx_abs(y) : tx_abs(z);
    }
    static inline double cubeSC(double x, double y, double z)
    {
        switch (cubeFace(x, y, z))
        {
        case 0: return -z;
        case 1: return z;
        case 5: return -x;
        default: return x;
        }
    }
    static inline double cubeTC(double x, double y, double z)
    {
        switch (cubeFace(x, y, z))
        {
        case 2: return z;
        case 3: return -z;
        default: return -y;
        }
    }
    static inline double cubeNorm(double s, double ma) { return ma > 0 ? (s / ma + 1.0) * 0.5 : 0.5; }

    typedef uint32_t TxU4 __attribute__((ext_vector_type(4)));
    typedef uint64_t TxH2 __attribute__((ext_vector_type(2)));
    typedef uint16_t TxS4 __attribute__((ext_vector_type(4)));
    typedef uint32_t TxU2 __attribute__((ext_vector_type(2)));
    typedef uint64_t TxL4 __attribute__((ext_vector_type(4)));
    typedef uint64_t TxL2 __attribute__((ext_vector_type(2)));
    // ctrl: 0-3 geometry (1 1d, 2 2d, 3 3d, 4 a1d, 5 a2d, 6 cube, 7 acube, 8 2dms, 9 a2dms), 4-5 lod (1 level, 2 grad),
    // 6 .ctype .f32, 8 offsets, 9 depth compare, 10-12 result (0 f32, 1 s32, 2 u32, 3 f16, 4 f16x2), 13-14 tld4
    // component, 15 tld4
    static inline double coordF(uint32_t v, bool f32) { return f32 ? f32d(v) : (double)(int32_t)v; }
    static inline V4 tex_core(TxH2 ts, TxU4 c, TxU4 A, TxU4 B, TxU4 C, uint32_t ctrl)
    {
        TxD t = {ts.x, ts.y};
        uint32_t geom = ctrl & 15, nl = (ctrl >> 4) & 3;
        bool cf32 = (ctrl >> 6) & 1;
        TxQ q = {0, {0, 0, 0}, false, 0, 0};
        if ((ctrl >> 8) & 1)
        {
            q.o[0] = (int32_t)C.x;
            q.o[1] = (int32_t)C.y;
            q.o[2] = (int32_t)C.z;
        }
        if ((ctrl >> 9) & 1)
        {
            q.dc = true;
            q.ref = C.w;
        }
        V4 z4 = {{0, 0, 0, 0}};
        bool arr = geom == 4 || geom == 5 || geom == 7;
        int rank = (geom == 1 || geom == 4) ? 1 : geom == 3 ? 3 : 2;
        if ((ctrl >> 15) & 1)
        {
            // tld4: the four texels of the bilinear footprint, one component
            double x, y;
            int64_t l, nm;
            if (geom == 6 || geom == 7)
            {
                int o = geom == 7 ? 1 : 0;
                double cx = f32d(c[o]), cy = f32d(c[o + 1]), cz = f32d(c[o + 2]);
                double ma = cubeMa(cx, cy, cz);
                x = cubeNorm(cubeSC(cx, cy, cz), ma);
                y = cubeNorm(cubeTC(cx, cy, cz), ma);
                l = 6 * tx_min64(geom == 7 ? (int64_t)c.x : 0, tx_max64(txLayers(t) / 6 - 1, 0)) + cubeFace(cx, cy, cz);
                nm = 1;
            }
            else
            {
                int o = arr ? 1 : 0;
                x = f32d(c[o]);
                y = f32d(c[o + 1]);
                l = arr ? tx_min64((int64_t)c.x, txLayers(t) - 1) : 0;
                nm = effNorm(t);
            }
            int64_t am = effAddr(t);
            int cc = (ctrl >> 13) & 3;
            int64_t i0 = tIdx(t, q, am, nm, x, 0, 0, 1), i1 = tIdx(t, q, am, nm, x, 1, 0, 1);
            int64_t j0 = tIdx(t, q, am, nm, y, 0, 1, 1), j1 = tIdx(t, q, am, nm, y, 1, 1, 1);
            V4 r = {{texel(t, q, i0, j1, 0, l).v[cc], texel(t, q, i1, j1, 0, l).v[cc], texel(t, q, i1, j0, 0, l).v[cc],
                     texel(t, q, i0, j0, 0, l).v[cc]}};
            return r;
        }
        if (geom == 8 || geom == 9)
        {
            bool ar = geom == 9;
            int o = ar ? 1 : 0;
            int64_t smp = (int64_t)c[o];
            if (cf32 || nl || q.dc || txSamples(t) <= 0 || smp >= txSamples(t))
                return z4;
            TxQ q2 = {0, {q.o[0], q.o[1], q.o[2]}, false, 0, smp};
            return sample(t, q2, 0, effAddr(t), effNorm(t), coordF(c[o + 1], false), coordF(c[o + 2], false), 0.0,
                          ar ? tx_min64((int64_t)c.x, txLayers(t) - 1) : 0, 2);
        }
        double dx[4] = {f32d(A.x), f32d(A.y), f32d(A.z), f32d(A.w)};
        double dy[4] = {f32d(B.x), f32d(B.y), f32d(B.z), f32d(B.w)};
        if (geom == 6 || geom == 7)
        {
            int o = geom == 7 ? 1 : 0;
            double x = f32d(c[o]), y = f32d(c[o + 1]), z = f32d(c[o + 2]);
            double ma = cubeMa(x, y, z);
            double lam = nl == 0 ? 0.0 : nl == 1 ? coordF(A.x, cf32) : gradLam(t, 1, 3, dx, dy, ma * 2.0);
            return mipSample(t, nl > 0, lam, q, effFilter(t), 1, 1, cubeNorm(cubeSC(x, y, z), ma), cubeNorm(cubeTC(x, y, z), ma),
                             0.0, 6 * tx_min64(geom == 7 ? (int64_t)c.x : 0, tx_max64(txLayers(t) / 6 - 1, 0)) + cubeFace(x, y, z), 2);
        }
        int o = arr ? 1 : 0;
        double lam = nl == 0 ? 0.0 : nl == 1 ? coordF(A.x, cf32) : gradLam(t, effNorm(t), rank, dx, dy, 1.0);
        return mipSample(t, nl > 0, lam, q, cf32 ? effFilter(t) : 0, effAddr(t), effNorm(t), coordF(c[o], cf32),
                         o + 1 < 4 ? coordF(c[o + 1], cf32) : 0.0, o + 2 < 4 ? coordF(c[o + 2], cf32) : 0.0,
                         arr ? tx_min64((int64_t)c.x, txLayers(t) - 1) : 0, rank);
    }
    static inline uint32_t tx_out(double f, uint32_t kind)
    {
        switch (kind)
        {
        case 1:
        case 2: return (uint32_t)(int64_t)f;
        case 3:
        case 4: return (uint32_t)tx_enc(f, 11, 5);
        default: return (uint32_t)tx_enc(f, 24, 8);
        }
    }
    TxU4 FUNC(tex_v4_32)(TxH2 ts, TxU4 c, TxU4 A, TxU4 B, TxU4 C, uint32_t ctrl)
    {
        V4 r = tex_core(ts, c, A, B, C, ctrl);
        uint32_t k = (ctrl >> 10) & 7;
        return TxU4{tx_out(r.v[0], k), tx_out(r.v[1], k), tx_out(r.v[2], k), tx_out(r.v[3], k)};
    }
    TxS4 FUNC(tex_v4_16)(TxH2 ts, TxU4 c, TxU4 A, TxU4 B, TxU4 C, uint32_t ctrl)
    {
        V4 r = tex_core(ts, c, A, B, C, ctrl);
        return TxS4{(uint16_t)tx_out(r.v[0], 3), (uint16_t)tx_out(r.v[1], 3), (uint16_t)tx_out(r.v[2], 3), (uint16_t)tx_out(r.v[3], 3)};
    }
    TxU2 FUNC(tex_v2_f16x2)(TxH2 ts, TxU4 c, TxU4 A, TxU4 B, TxU4 C, uint32_t ctrl)
    {
        V4 r = tex_core(ts, c, A, B, C, ctrl);
        return TxU2{tx_out(r.v[0], 3) | (tx_out(r.v[1], 3) << 16), tx_out(r.v[2], 3) | (tx_out(r.v[3], 3) << 16)};
    }

    // ---------------------------------------------------------------- txq
    static inline uint32_t amEnum(uint64_t c) { return c == 0 ? 0 : c == 1 ? 3 : c == 2 ? 4 : c == 3 ? 1 : (uint32_t)c; }
    // attr: 0 width 1 height 2 depth 3 array_size 4 normalized_coords 5 filter_mode 6 addr_mode 7 num_mipmap_levels
    // 8 num_samples 9 force_unnormalized_coords 10 channel_data_type 11 channel_order 12 memory_layout
    uint32_t FUNC(txq)(uint64_t h, uint32_t lod, uint32_t attr)
    {
        TxD t = {h, 0};
        if (tdW(h, 0) == SM_MAGIC)
        {
            if (attr == 5)
                return (uint32_t)tdW(h, 1);
            if (attr == 6)
                return amEnum(tdW(h, 2));
            if (attr == 9)
                return tdW(h, 3) == 0;
            return 0;
        }
        int64_t lv = lod;
        switch (attr)
        {
        case 0: return (uint32_t)lvD(t, lv, 0);
        case 1: return (uint32_t)lvD(t, lv, 1);
        case 2: return (uint32_t)lvD(t, lv, 2);
        case 3:
        {
            uint64_t n = (tdW(h, 13) & 2) ? tdW(h, 8) / 6 : tdW(h, 8);
            return n > 1 ? (uint32_t)n : 0;
        }
        case 4: return (uint32_t)tdW(h, 6);
        case 5: return (uint32_t)tdW(h, 4);
        case 6: return amEnum(tdW(h, 5));
        case 7: return tdW(h, 12) > 1 ? (uint32_t)tdW(h, 12) : 0;
        case 8: return (uint32_t)txSamples(t);
        default: return 0;
        }
    }

    // ---------------------------------------------------------------- surfaces
    static inline int64_t sfTb(uint64_t b) { return tx_max64(1, (int64_t)tdW(b, 2)); }
    static inline int64_t sfLayers(uint64_t b) { return tx_max64(1, (int64_t)tdW(b, 3)); }
    static inline int64_t sfDim(uint64_t b, int i) { return tx_max64(1, (int64_t)tdW(b, 4 + i)); }
    static inline int64_t sfAddr(uint64_t b, int64_t x, int64_t y, int64_t z, int64_t l, int64_t n)
    {
        if (x < 0 || x + n > sfDim(b, 0) * sfTb(b) || y < 0 || y >= sfDim(b, 1) || z < 0 || z >= sfDim(b, 2) || l < 0 || l >= sfLayers(b))
            return -1;
        return (int64_t)tdW(b, 1) + (((l * sfDim(b, 2) + z) * sfDim(b, 1) + y) * sfDim(b, 0)) * sfTb(b) + x;
    }
    // ctrl: 0-3 geometry, 4-7 vector length, 8-11 element bytes, 12-13 mode (0 trap 1 clamp 2 zero), 14 .p, 16-19 op
    // (sured: 0 add 1 min 2 max 3 and 4 or 5 inc 6 dec 7 exch 8 xor), 20 signed type
    struct SfX
    {
        int64_t x, y, z, l;
    };
    static inline SfX sfCoords(uint32_t geom, TxU4 c)
    {
        bool arr = geom == 4 || geom == 5;
        int rank = (geom == 1 || geom == 4) ? 1 : geom == 3 ? 3 : 2;
        if (arr)
            return SfX{(int32_t)c.y, rank >= 2 ? (int64_t)(int32_t)c.z : 0, 0, (int64_t)c.x};
        return SfX{(int32_t)c.x, rank >= 2 ? (int64_t)(int32_t)c.y : 0, rank >= 3 ? (int64_t)(int32_t)c.z : 0, 0};
    }
    static inline int64_t sfAN(uint64_t b, uint32_t mode, int64_t n, SfX xs)
    {
        if (mode == 1)
            return sfAddr(b, tx_min64(tx_max64(xs.x, 0), tx_max64(sfDim(b, 0) * sfTb(b) - n, 0)), tx_min64(tx_max64(xs.y, 0), sfDim(b, 1) - 1),
                          tx_min64(tx_max64(xs.z, 0), sfDim(b, 2) - 1), tx_min64(tx_max64(xs.l, 0), sfLayers(b) - 1), n);
        return sfAddr(b, xs.x, xs.y, xs.z, xs.l, n);
    }
    TxL4 FUNC(suld)(uint64_t b, TxU4 c, uint32_t ctrl)
    {
        int n = (ctrl >> 4) & 15, eb = (ctrl >> 8) & 15;
        uint32_t mode = (ctrl >> 12) & 3;
        int64_t a = sfAN(b, mode, (int64_t)n * eb, sfCoords(ctrl & 15, c));
        TxL4 r = {0, 0, 0, 0};
        if (a >= 0)
            for (int i = 0; i < n && i < 4; i++)
                r[i] = tx_rd((uint64_t)a + (uint64_t)i * eb, eb);
        return r;
    }
    TxU4 FUNC(suld_v_32)(uint64_t b, TxU4 c, uint32_t ctrl)
    {
        TxL4 r = __zluda_ptx_impl_suld(b, c, ctrl);
        return TxU4{(uint32_t)r.x, (uint32_t)r.y, (uint32_t)r.z, (uint32_t)r.w};
    }
    TxU2 FUNC(suld_v2_32)(uint64_t b, TxU4 c, uint32_t ctrl)
    {
        TxL4 r = __zluda_ptx_impl_suld(b, c, ctrl);
        return TxU2{(uint32_t)r.x, (uint32_t)r.y};
    }
    uint32_t FUNC(suld_v1_32)(uint64_t b, TxU4 c, uint32_t ctrl) { return (uint32_t)__zluda_ptx_impl_suld(b, c, ctrl).x; }
    uint64_t FUNC(suld_v1_64)(uint64_t b, TxU4 c, uint32_t ctrl) { return __zluda_ptx_impl_suld(b, c, ctrl).x; }
    TxL2 FUNC(suld_v2_64)(uint64_t b, TxU4 c, uint32_t ctrl)
    {
        TxL4 r = __zluda_ptx_impl_suld(b, c, ctrl);
        return TxL2{r.x, r.y};
    }
    TxL4 FUNC(suld_v4_64)(uint64_t b, TxU4 c, uint32_t ctrl) { return __zluda_ptx_impl_suld(b, c, ctrl); }

    static inline int chBytesOf(uint64_t f) { return f <= 3 ? 1 : f <= 8 ? 2 : 4; }
    static inline int64_t maskPow(uint64_t f) { return (int64_t)((1ull << (8 * chBytesOf(f) - ((f == 1 || f == 5) ? 1 : 0))) - 1); }
    static inline double sfClamp(double x, double lo, double hi)
    {
        if (x != x)
            return 0;
        return x < lo ? lo : hi < x ? hi : x;
    }
    static inline int64_t rneI(double x)
    {
        double fl = tx_floor(x), fr = x - fl;
        int64_t i = (int64_t)fl;
        return (fr > 0.5 || (fr == 0.5 && (i & 1))) ? i + 1 : i;
    }
    static inline uint64_t fmtConv(uint64_t f, uint32_t v)
    {
        int cb = 8 * chBytesOf(f);
        uint64_t m = cb >= 64 ? ~0ull : (1ull << cb) - 1;
        if (f == 0 || f == 4)
            return (uint64_t)rneI(sfClamp(f32d(v), 0.0, 1.0) * (double)maskPow(f));
        if (f == 1 || f == 5)
            return (uint64_t)rneI(sfClamp(f32d(v), -1.0, 1.0) * (double)maskPow(f)) & m;
        if (f == 2 || f == 6 || f == 9)
            return v < m ? v : m;
        if (f == 3 || f == 7 || f == 10)
        {
            int64_t lo = -(int64_t)(1ull << (cb - 1)), hi = (int64_t)(1ull << (cb - 1)) - 1;
            int64_t s = (int32_t)v;
            return (uint64_t)(s < lo ? lo : s > hi ? hi : s) & m;
        }
        if (f == 8)
            return tx_enc(f32d(v), 11, 5);
        return v;
    }
    static inline void sust_core(uint64_t b, TxU4 c, TxL4 v, uint32_t ctrl)
    {
        int n = (ctrl >> 4) & 15, eb = (ctrl >> 8) & 15;
        uint32_t mode = (ctrl >> 12) & 3;
        SfX xs = sfCoords(ctrl & 15, c);
        if ((ctrl >> 14) & 1)
        {
            // formatted: the channels of the surface format
            int64_t ch = tx_max64(1, (int64_t)tdW(b, 8));
            uint64_t f = tdW(b, 7);
            int64_t k = n < ch ? n : ch;
            int cbb = chBytesOf(f);
            xs.x *= sfTb(b);
            int64_t a = sfAN(b, mode, k * cbb, xs);
            if (a >= 0)
                for (int i = 0; i < k && i < 4; i++)
                    tx_wr((uint64_t)a + (uint64_t)i * cbb, cbb, fmtConv(f, (uint32_t)v[i]));
            return;
        }
        int64_t a = sfAN(b, mode, (int64_t)n * eb, xs);
        if (a >= 0)
            for (int i = 0; i < n && i < 4; i++)
                tx_wr((uint64_t)a + (uint64_t)i * eb, eb, v[i]);
    }
    void FUNC(sust_64)(uint64_t b, TxU4 c, TxL4 v, uint32_t ctrl) { sust_core(b, c, v, ctrl); }
    void FUNC(sust_32)(uint64_t b, TxU4 c, TxU4 v, uint32_t ctrl) { sust_core(b, c, TxL4{v.x, v.y, v.z, v.w}, ctrl); }
    static inline void sured_core(uint64_t b, TxU4 c, uint64_t v, uint32_t ctrl)
    {
        int eb = (ctrl >> 8) & 15;
        uint32_t mode = (ctrl >> 12) & 3, op = (ctrl >> 16) & 15;
        bool sg = (ctrl >> 20) & 1;
        SfX xs = sfCoords(ctrl & 15, c);
        if ((ctrl >> 14) & 1)
        {
            xs.x *= eb;
            uint64_t f = tdW(b, 7);
            sg = f == 3 || f == 7 || f == 10;
        }
        int64_t a = sfAN(b, mode, eb, xs);
        if (a < 0)
            return;
        if (eb == 8)
        {
            TX_GLOBAL uint64_t *p = (TX_GLOBAL uint64_t *)(uintptr_t)a;
            switch (op)
            {
            case 0: __atomic_fetch_add(p, v, __ATOMIC_RELAXED); break;
            case 1: if (sg) __atomic_fetch_min((TX_GLOBAL int64_t *)p, (int64_t)v, __ATOMIC_RELAXED); else __atomic_fetch_min(p, v, __ATOMIC_RELAXED); break;
            case 2: if (sg) __atomic_fetch_max((TX_GLOBAL int64_t *)p, (int64_t)v, __ATOMIC_RELAXED); else __atomic_fetch_max(p, v, __ATOMIC_RELAXED); break;
            case 3: __atomic_fetch_and(p, v, __ATOMIC_RELAXED); break;
            case 4: __atomic_fetch_or(p, v, __ATOMIC_RELAXED); break;
            case 8: __atomic_fetch_xor(p, v, __ATOMIC_RELAXED); break;
            default: __atomic_exchange_n(p, v, __ATOMIC_RELAXED); break;
            }
            return;
        }
        TX_GLOBAL uint32_t *p = (TX_GLOBAL uint32_t *)(uintptr_t)a;
        uint32_t w = (uint32_t)v;
        switch (op)
        {
        case 0: __atomic_fetch_add(p, w, __ATOMIC_RELAXED); break;
        case 1: if (sg) __atomic_fetch_min((TX_GLOBAL int32_t *)p, (int32_t)w, __ATOMIC_RELAXED); else __atomic_fetch_min(p, w, __ATOMIC_RELAXED); break;
        case 2: if (sg) __atomic_fetch_max((TX_GLOBAL int32_t *)p, (int32_t)w, __ATOMIC_RELAXED); else __atomic_fetch_max(p, w, __ATOMIC_RELAXED); break;
        case 3: __atomic_fetch_and(p, w, __ATOMIC_RELAXED); break;
        case 4: __atomic_fetch_or(p, w, __ATOMIC_RELAXED); break;
        case 8: __atomic_fetch_xor(p, w, __ATOMIC_RELAXED); break;
        default: __atomic_exchange_n(p, w, __ATOMIC_RELAXED); break;
        }
    }
    void FUNC(sured_64)(uint64_t b, TxU4 c, uint64_t v, uint32_t ctrl) { sured_core(b, c, v, ctrl); }
    void FUNC(sured_32)(uint64_t b, TxU4 c, uint32_t v, uint32_t ctrl) { sured_core(b, c, v, ctrl); }
    uint32_t FUNC(suq)(uint64_t b, uint32_t attr)
    {
        switch (attr)
        {
        case 0: return (uint32_t)sfDim(b, 0);
        case 1: return (uint32_t)sfDim(b, 1);
        case 2: return (uint32_t)sfDim(b, 2);
        case 3: return tdW(b, 12) != 0 ? (uint32_t)tdW(b, 3) : 0;
        case 10: return (uint32_t)tdW(b, 9);
        case 11: return (uint32_t)tdW(b, 10);
        case 12: return (uint32_t)tdW(b, 11);
        default: return 0;
        }
    }
    // istypep: kind 0 texref 1 samplerref 2 surfref
    bool FUNC(istypep)(uint64_t h, uint32_t kind)
    {
        if (h < 4096)
            return false;
        uint64_t m = kind == 0 ? TX_MAGIC : kind == 1 ? SM_MAGIC : SF_MAGIC;
        return tdW(h, 0) == m;
    }
    bool FUNC(tx_true)() { return true; }
