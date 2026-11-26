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
