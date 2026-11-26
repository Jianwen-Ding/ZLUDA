// Included inside extern "C" of zluda_ptx_metal_impl.cpp (after float.inc, texture.inc): multimem.ld_reduce / st / red
// (ptx_parser/src/helper_table/multimem.rs).  A multimem address lies in a window of the .multimem harness directive
// (zluda_metal/tests/ptxconf/ptxconf.py): the window's 4 KiB pages start with its description (magic, window address,
// size, buffer count, buffer addresses), and the address stands for the same offset in each buffer.  Sail: ptx_mm.sail.
    static const uint64_t MM_MAGIC = 0x4D4D57494E444F57ull;
    struct MmWin
    {
        uint64_t page, base, n;
    };
    static inline MmWin mm_win(uint64_t h)
    {
        uint64_t pg = h & ~0xFFFull;
        if (tdW(pg, 0) != MM_MAGIC)
            return MmWin{pg, 0, 0};
        return MmWin{pg, tdW(pg, 1), tdW(pg, 3)};
    }
    static inline uint64_t mm_dev(MmWin w, uint64_t i, uint64_t h) { return tdW(w.page, 4 + (int)i) + (h - w.base); }

    // a value of cvt.inc type t -> XF, and back (round to nearest even)
    static inline XF mm_dec(uint64_t v, uint32_t t) { return xdec_any(v, t, false); }
    static inline uint64_t mm_enc(XF x, uint32_t t)
    {
        if (t_mini(t))
            return mk_cvt(t, x, 127, RM_RN, false, false, false);
        return xfin(x, t, RM_RN, 0, 0, 0, false);
    }
    static inline bool mm_fp8(uint32_t t) { return t == T_E4M3 || t == T_E5M2; }
    // bin(op, a): op 0 add 1 min 2 max 3 and 4 or 5 xor, on integers (cvt.inc integer codes) or floats
    static inline uint64_t mm_bin(uint32_t op, uint32_t a, uint64_t x, uint64_t y)
    {
        if (t_int(a))
        {
            int w = t_iw(a);
            uint64_t m = w == 64 ? ~0ull : (1ull << w) - 1;
            bool sg = t_signed(a);
            int64_t sx = w == 64 ? (int64_t)x : (int64_t)(x << (64 - w)) >> (64 - w);
            int64_t sy = w == 64 ? (int64_t)y : (int64_t)(y << (64 - w)) >> (64 - w);
            switch (op)
            {
            case 0: return (x + y) & m;
            case 1: return sg ? (sx <= sy ? x : y) : ((x & m) <= (y & m) ? x : y);
            case 2: return sg ? (sx >= sy ? x : y) : ((x & m) >= (y & m) ? x : y);
            case 3: return x & y;
            case 4: return x | y;
            default: return x ^ y;
            }
        }
        switch (op)
        {
        case 0: return f_elem(x, y, 0, 0 | (a << 4));
        case 1: return f_elem(x, y, 0, 4 | (a << 4));
        case 2: return f_elem(x, y, 0, 5 | (a << 4));
        case 3: return x & y;
        case 4: return x | y;
        default: return x ^ y;
        }
    }
    // mmTo / mmComb / mmBack: a lane of type l in the accumulation type a
    static inline uint64_t mm_to(uint32_t l, uint32_t a, uint64_t v)
    {
        if (l == a)
            return v;
        if (mm_fp8(l))
        {
            uint64_t h = mm_enc(mm_dec(v, l), T_F16);
            return a == T_F16 ? h : mm_enc(mm_dec(h, T_F16), a);
        }
        return mm_enc(mm_dec(v, l), a);
    }
    static inline uint64_t mm_comb(uint32_t op, uint32_t a, uint64_t x, uint64_t y)
    {
        if (mm_fp8(a))
        {
            uint64_t hx = mm_enc(mm_dec(x, a), T_F16), hy = mm_enc(mm_dec(y, a), T_F16);
            return mm_enc(mm_dec(mm_bin(op, T_F16, hx, hy), T_F16), a);
        }
        return mm_bin(op, a, x, y);
    }
    static inline uint64_t mm_back(uint32_t l, uint32_t a, uint64_t v) { return l == a ? v : mm_enc(mm_dec(v, a), l); }

    // ctrl: 0-2 op, 3-7 lane type, 8-10 lanes, 11-15 accumulation type, 16-19 element bytes, 20-23 vector length, 24 red
    static inline uint64_t mm_red_elem(MmWin w, uint64_t h, uint32_t ctrl)
    {
        uint32_t op = ctrl & 7, l = (ctrl >> 3) & 31, lanes = (ctrl >> 8) & 7, a = (ctrl >> 11) & 31, eb = (ctrl >> 16) & 15;
        int lw = 8 * (int)eb / (int)lanes;
        uint64_t lm = lw == 64 ? ~0ull : (1ull << lw) - 1;
        uint64_t out = 0;
        for (uint32_t j = 0; j < lanes; j++)
        {
            uint64_t acc = 0;
            for (uint64_t i = 0; i < w.n; i++)
            {
                uint64_t v = (tx_rd(mm_dev(w, i, h), (int)eb) >> (lw * j)) & lm;
                uint64_t x = mm_to(l, a, v);
                acc = i == 0 ? x : mm_comb(op, a, acc, x);
            }
            out |= (mm_back(l, a, acc) & lm) << (lw * j);
        }
        return out;
    }
    static inline TxL4 mm_ld(uint64_t h, uint32_t ctrl)
    {
        MmWin w = mm_win(h);
        uint32_t eb = (ctrl >> 16) & 15, n = (ctrl >> 20) & 15;
        TxL4 r = {0, 0, 0, 0};
        for (uint32_t e = 0; e < n && e < 4; e++)
            r[e] = mm_red_elem(w, h + e * eb, ctrl);
        return r;
    }
    uint32_t FUNC(mm_ld1_32)(uint64_t h, uint32_t ctrl) { return (uint32_t)mm_ld(h, ctrl).x; }
    uint16_t FUNC(mm_ld1_16)(uint64_t h, uint32_t ctrl) { return (uint16_t)mm_ld(h, ctrl).x; }
    uint64_t FUNC(mm_ld1_64)(uint64_t h, uint32_t ctrl) { return mm_ld(h, ctrl).x; }
    TxU4 FUNC(mm_ld4_32)(uint64_t h, uint32_t ctrl)
    {
        TxL4 r = mm_ld(h, ctrl);
        return TxU4{(uint32_t)r.x, (uint32_t)r.y, (uint32_t)r.z, (uint32_t)r.w};
    }
    TxS4 FUNC(mm_ld4_16)(uint64_t h, uint32_t ctrl)
    {
        TxL4 r = mm_ld(h, ctrl);
        return TxS4{(uint16_t)r.x, (uint16_t)r.y, (uint16_t)r.z, (uint16_t)r.w};
    }
    TxL4 FUNC(mm_ld4_64)(uint64_t h, uint32_t ctrl) { return mm_ld(h, ctrl); }

    // multimem.st / red: st / red on each buffer (red: a read-modify-write of each element, compare-and-swap loops)
    static inline void mm_rmw(uint64_t addr, uint32_t eb, uint32_t ctrl, uint64_t v)
    {
        uint32_t op = ctrl & 7, l = (ctrl >> 3) & 31, lanes = (ctrl >> 8) & 7;
        int lw = 8 * (int)eb / (int)lanes;
        uint64_t lm = lw == 64 ? ~0ull : (1ull << lw) - 1;
        if (eb == 8)
        {
            TX_GLOBAL uint64_t *p = (TX_GLOBAL uint64_t *)(uintptr_t)addr;
            uint64_t old = __atomic_load_n(p, __ATOMIC_RELAXED), nw;
            do
                nw = mm_bin(op, l, old, v);
            while (!__atomic_compare_exchange_n(p, &old, nw, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
            return;
        }
        // 32-bit word holding the element (2-byte elements: the aligned word around it)
        uint64_t wa = addr & ~3ull;
        int sh = (int)(addr - wa) * 8;
        uint32_t em = eb >= 4 ? 0xFFFFFFFFu : ((1u << (8 * eb)) - 1) << sh;
        TX_GLOBAL uint32_t *p = (TX_GLOBAL uint32_t *)(uintptr_t)wa;
        uint32_t old = __atomic_load_n(p, __ATOMIC_RELAXED), nw;
        do
        {
            uint64_t cur = (old & em) >> sh, res = 0;
            for (uint32_t j = 0; j < lanes; j++)
                res |= (mm_bin(op, l, (cur >> (lw * j)) & lm, (v >> (lw * j)) & lm) & lm) << (lw * j);
            nw = (old & ~em) | (((uint32_t)res << sh) & em);
        } while (!__atomic_compare_exchange_n(p, &old, nw, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
    }
    static inline void mm_st(uint64_t h, TxL4 v, uint32_t ctrl)
    {
        MmWin w = mm_win(h);
        uint32_t eb = (ctrl >> 16) & 15, n = (ctrl >> 20) & 15;
        bool red = (ctrl >> 24) & 1;
        for (uint64_t i = 0; i < w.n; i++)
            for (uint32_t e = 0; e < n && e < 4; e++)
            {
                uint64_t a = mm_dev(w, i, h + e * eb);
                if (red)
                    mm_rmw(a, eb, ctrl, v[e]);
                else
                    tx_wr(a, (int)eb, v[e]);
            }
    }
    void FUNC(mm_st_32)(uint64_t h, TxU4 v, uint32_t ctrl) { mm_st(h, TxL4{v.x, v.y, v.z, v.w}, ctrl); }
    void FUNC(mm_st_16)(uint64_t h, TxS4 v, uint32_t ctrl) { mm_st(h, TxL4{v.x, v.y, v.z, v.w}, ctrl); }
    void FUNC(mm_st_64)(uint64_t h, TxL4 v, uint32_t ctrl) { mm_st(h, v, ctrl); }

    // ---------------------------------------------------------------- fabric (ptx_mm.sail exec_fabric)
    // the logical endpoints of the .fabric harness directive: 16 entries of 8 words (id + 1, size, buffer count, up to 5
    // buffer addresses), written by the harness through the module variable
    TX_GLOBAL uint64_t __zluda_fabric_tab[128];
    struct FbEp
    {
        int e;
        uint64_t size, n;
    };
    static inline FbEp fb_find(uint64_t le)
    {
        for (int i = 0; i < 16; i++)
            if (((volatile TX_GLOBAL uint64_t *)__zluda_fabric_tab)[8 * i] == le + 1)
                return FbEp{i, ((volatile TX_GLOBAL uint64_t *)__zluda_fabric_tab)[8 * i + 1],
                            ((volatile TX_GLOBAL uint64_t *)__zluda_fabric_tab)[8 * i + 2]};
        return FbEp{-1, 0, 0};
    }
    static inline uint64_t fb_dev(FbEp e, uint64_t i) { return ((volatile TX_GLOBAL uint64_t *)__zluda_fabric_tab)[8 * e.e + 3 + (int)i]; }
    // the devices of [le, off .. off + n): all (multimem) or the first
    static inline uint64_t fb_count(FbEp e, uint64_t off, uint64_t n, bool all)
    {
        if (e.e < 0 || off + n > e.size)
            return 0;
        return all ? e.n : (e.n ? 1 : 0);
    }
    typedef uint64_t FbP __attribute__((ext_vector_type(2)));
    typedef uint32_t FbQ __attribute__((ext_vector_type(2)));
    void FUNC(fabric_get)(SHARED_SPACE uint8_t *d, uint32_t lid, uint64_t loff, uint32_t n, SHARED_SPACE uint64_t *bar)
    {
        FbP le = {lid, loff};
        FbEp e = fb_find(le.x);
        if (!fb_count(e, le.y, n, false))
            return;
        uint64_t a = fb_dev(e, 0) + le.y;
        for (uint32_t i = 0; i < n; i++)
            d[i] = (uint8_t)tx_rd(a + i, 1);
        __zluda_ptx_impl_mbar_tx(bar, n, 1);
    }
    // ctrl: 0-3 atom op, 4-8 type, 9-12 element bytes, 13 multimem, 14 mask, 15 counted, 16 red, 17 tx in 16B units
    void FUNC(fabric_put)(FbQ lm, FbP oc, SHARED_SPACE uint8_t *s, uint32_t n, SHARED_SPACE uint64_t *bar, uint32_t ctrl)
    {
        FbP le = {lm.x, oc.x}, mc = {lm.y, oc.y};
        FbEp e = fb_find(le.x);
        bool all = (ctrl >> 13) & 1;
        uint64_t k = fb_count(e, le.y, n, all);
        uint32_t eb = (ctrl >> 9) & 15;
        bool red = (ctrl >> 16) & 1;
        uint32_t mctrl = (ctrl & 15) | (((ctrl >> 4) & 31) << 3) | (1u << 8);
        for (uint64_t i = 0; i < k; i++)
        {
            uint64_t a = fb_dev(e, i) + le.y;
            if (red)
            {
                for (uint32_t j = 0; j + eb <= n; j += eb)
                {
                    uint64_t v = 0;
                    for (uint32_t b = 0; b < eb; b++)
                        v |= (uint64_t)s[j + b] << (8 * b);
                    mm_rmw(a + j, eb, mctrl, v);
                }
            }
            else
                for (uint32_t j = 0; j < n; j++)
                    if (!((ctrl >> 14) & 1) || ((mc.x >> (j % 16)) & 1))
                        tx_wr(a + j, 1, s[j]);
        }
        if ((ctrl >> 15) & 1)
        {
            // counted completion: the counter at every destination device counts the bytes
            uint64_t kc = fb_count(e, mc.y, 8, all);
            for (uint64_t i = 0; i < kc; i++)
                __atomic_fetch_add((TX_GLOBAL uint64_t *)(uintptr_t)(fb_dev(e, i) + mc.y), (uint64_t)n, __ATOMIC_RELAXED);
        }
        if (k)
            __zluda_ptx_impl_mbar_tx(bar, n / 16, 1);
    }
    // try_pullred: lane 0 reduces the element of every device into shared memory (ctrl as mm_red_elem's)
    void FUNC(fabric_pullred)(SHARED_SPACE uint8_t *d, uint32_t lid, uint64_t loff, uint32_t n, SHARED_SPACE uint64_t *bar, uint32_t ctrl)
    {
        FbP le = {lid, loff};
        if (__zluda_ptx_impl_sreg_laneid() != 0)
            return;
        FbEp e = fb_find(le.x);
        uint64_t k = fb_count(e, le.y, n, true);
        if (!k)
            return;
        uint32_t eb = (ctrl >> 16) & 15, op = ctrl & 7, t = (ctrl >> 3) & 31;
        for (uint32_t j = 0; j + eb <= n; j += eb)
        {
            uint64_t acc = 0;
            for (uint64_t i = 0; i < k; i++)
            {
                uint64_t v = tx_rd(fb_dev(e, i) + le.y + j, (int)eb);
                acc = i == 0 ? v : mm_bin(op, t, acc, v);
            }
            for (uint32_t b = 0; b < eb; b++)
                d[j + b] = (uint8_t)(acc >> (8 * b));
        }
        __zluda_ptx_impl_mbar_tx(bar, n, 1);
    }
    // try_atom [le, off], [dst], [src], [mbar]: the old value to dst, at the offset within the 16-byte chunk
    void FUNC(fabric_atom)(uint32_t lid, uint64_t loff, SHARED_SPACE uint8_t *dd, SHARED_SPACE uint8_t *ss, SHARED_SPACE uint64_t *bar,
                           uint32_t ctrl)
    {
        FbP le = {lid, loff};
        uint32_t op = ctrl & 15, t = (ctrl >> 4) & 31, eb = (ctrl >> 9) & 15;
        FbEp e = fb_find(le.x);
        if (!fb_count(e, le.y, eb, false))
            return;
        uint64_t so = le.y % 16;
        uint64_t v = 0, x = 0;
        for (uint32_t b = 0; b < eb; b++)
        {
            v |= (uint64_t)ss[so + b] << (8 * b);
            x |= (uint64_t)ss[16 + so + b] << (8 * b);
        }
        uint64_t a = fb_dev(e, 0) + le.y, old;
        uint64_t m = eb == 8 ? ~0ull : (1ull << (8 * eb)) - 1;
        if (eb == 8)
        {
            TX_GLOBAL uint64_t *p = (TX_GLOBAL uint64_t *)(uintptr_t)a;
            old = __atomic_load_n(p, __ATOMIC_RELAXED);
            uint64_t nw;
            do
            {
                nw = op == 6 ? v : op == 7 ? (old == v ? x : old) : op == 8 ? (old >= v ? 0 : old + 1) : op == 9 ? ((old == 0 || old > v) ? v : old - 1) : mm_bin(op, t, old, v);
            } while (!__atomic_compare_exchange_n(p, &old, nw, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
        }
        else
        {
            TX_GLOBAL uint32_t *p = (TX_GLOBAL uint32_t *)(uintptr_t)a;
            uint32_t o = __atomic_load_n(p, __ATOMIC_RELAXED), nw;
            do
            {
                uint64_t ov = o & m;
                uint64_t r = op == 6 ? v : op == 7 ? (ov == (v & m) ? x : ov) : op == 8 ? (ov >= (v & m) ? 0 : ov + 1) : op == 9 ? ((ov == 0 || ov > (v & m)) ? v : ov - 1) : mm_bin(op, t, ov, v);
                nw = (uint32_t)(r & m);
            } while (!__atomic_compare_exchange_n(p, &o, nw, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
            old = o;
        }
        for (uint32_t b = 0; b < eb; b++)
            dd[so + b] = (uint8_t)(old >> (8 * b));
        __zluda_ptx_impl_mbar_tx(bar, 1, 1);
    }
    // fabric tensor forms: [tmap, le, off, {dims}, {coords}] with .override::global_address (the endpoint's device
    // address) and .override::global_dim (tmOv of ptx_tma.sail); v: the n coordinates, then the n dims
    static inline void fb_tm(Tm *t, uint64_t tmap, uint64_t ga, I32x8 v, int n, I32x8 *cs)
    {
        tm_read(tmap, t);
        t->w[3] = ga;
        I32x8 c = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < 5; i++)
            if (i < n)
            {
                c[i] = v[i];
                t->w[4 + i] = (uint32_t)v[n + i];
            }
        *cs = c;
    }
    static inline void fabric_tget(SHARED_SPACE uint8_t *dst, uint64_t tmap, uint32_t lid, uint64_t off, I32x8 v,
                                   SHARED_SPACE uint64_t *mb, int n)
    {
        FbEp e = fb_find(lid);
        if (e.e < 0 || off > e.size)
            return;
        uint64_t ga = fb_dev(e, 0) + off;
        MEM_SERIAL({
            Tm t;
            I32x8 cs;
            fb_tm(&t, tmap, ga, v, n, &cs);
            I32x8 inf = {0, 0, 0, 0, 0, 0, 0, 0};
            tma_load_impl(dst, &t, cs, mb, inf, 0);
        })
    }
    void FUNC(fabric_tget1)(SHARED_SPACE uint8_t *d, uint64_t t, uint32_t l, uint64_t o, I32x8 v, SHARED_SPACE uint64_t *mb) { fabric_tget(d, t, l, o, v, mb, 1); }
    void FUNC(fabric_tget2)(SHARED_SPACE uint8_t *d, uint64_t t, uint32_t l, uint64_t o, I32x8 v, SHARED_SPACE uint64_t *mb) { fabric_tget(d, t, l, o, v, mb, 2); }
    void FUNC(fabric_tget3)(SHARED_SPACE uint8_t *d, uint64_t t, uint32_t l, uint64_t o, I32x8 v, SHARED_SPACE uint64_t *mb) { fabric_tget(d, t, l, o, v, mb, 3); }
    // try_put / try_red .tensor (tile): the tensor store on each destination device; ctrl: 0-3 op (tma_store's: 0 store,
    // 1 add ...), 4-6 rank, 7 multimem
    void FUNC(fabric_tput)(uint64_t tmap, uint32_t lid, uint64_t loff, I32x8 v, SHARED_SPACE uint8_t *src, uint32_t ctrl)
    {
        FbP lo = {lid, loff};
        FbEp e = fb_find(lo.x);
        int n = (ctrl >> 4) & 7;
        uint32_t op = ctrl & 15;
        uint64_t k = (e.e < 0 || lo.y > e.size) ? 0 : ((ctrl >> 7) & 1) ? e.n : (e.n ? 1 : 0);
        for (uint64_t d = 0; d < k; d++)
        {
            uint64_t ga = fb_dev(e, d) + lo.y;
            MEM_SERIAL({
                Tm t;
                I32x8 cs;
                fb_tm(&t, tmap, ga, v, n, &cs);
                uint32_t eb = tm_eb(&t), c = tm_etc(&t), md = tm_effmode(&t);
                uint32_t base = (uint32_t)(uintptr_t)src;
                int64_t cnt = tm_count(&t, false);
                for (int64_t i = 0; i < cnt; i++)
                {
                    int64_t a = tm_addr_t(&t, cs, i);
                    if (a < 0)
                        continue;
                    uint32_t sa = tm_swizzle(md, base + (uint32_t)i * eb);
                    uint64_t val = s_rd((volatile SHARED_SPACE uint8_t *)(src + (int32_t)(sa - base)), eb);
                    if (op != 0)
                        val = red_elem(op, c, g_rd((uint64_t)a, eb), val, eb);
                    g_wr((uint64_t)a, eb, val);
                }
            })
        }
    }
    void FUNC(fabric_tx)(SHARED_SPACE uint64_t *mb, uint32_t n) { __zluda_ptx_impl_mbar_tx(mb, n, 1); }
