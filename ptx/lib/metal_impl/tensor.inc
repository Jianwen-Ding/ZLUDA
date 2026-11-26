// Included inside extern "C" of zluda_ptx_metal_impl.cpp (after matrix.inc): helpers of
// ptx_parser/src/helper_table/tensor.rs, the warpgroup (wgmma) and tensor-memory (tcgen05)
// matrix instructions, following PTX-formalization's Sail model (sail/ptx/ptx_wg.sail,
// ptx_tc.sail). The arithmetic is the warp matrix engine's (matrix.inc: exact products, an exact
// fixed-point sum rounded once).
//
// A CTA is one AGX threadgroup and a warpgroup four consecutive SIMD-groups. The CTA's per-CTA
// global memory (the hidden kernel argument "perctamemory" of llvm-air2agx, see air_lower.cpp)
// holds a scratch area of up to 128 words per thread (wgmma accumulators) and the emulated tensor
// memory. No private array is indexed dynamically (see matrix.inc).

    uint64_t __zluda_metal_percta_memory(uint32_t bytes);

#define TN_SCRATCH_BYTES (1024u * 4u * 128u)
#define TN_GLOBAL MX_GLOBAL

    void FUNC(tensor_nop)() {}

    MX_INL uint32_t tn_tid()
    {
        return __zluda_metal_thread_position_in_threadgroup(0) +
               __zluda_metal_threads_per_threadgroup(0) *
                   (__zluda_metal_thread_position_in_threadgroup(1) +
                    __zluda_metal_threads_per_threadgroup(1) * __zluda_metal_thread_position_in_threadgroup(2));
    }
    // this thread's nw scratch words (nw <= 128: the region of a kernel with wgmma of up to nw accumulator registers is
    // 1024 * nw words; the tcgen05 regions follow the largest)
    MX_INL uint32_t TN_GLOBAL *tn_scratch(uint32_t nw)
    {
        uint64_t b = __zluda_metal_percta_memory(1024u * 4u * nw);
        return (uint32_t TN_GLOBAL *)(uintptr_t)(b + (uint64_t)(tn_tid() & 1023u) * 4u * nw);
    }

    // ---------------------------------------------------------------- shared memory matrix descriptors
    struct TnDesc
    {
        int start, lbo, sbo, swz, base;
    };
    MX_INL TnDesc tn_desc(uint64_t d)
    {
        TnDesc r;
        r.start = (int)((d & 16383u) << 4);
        r.lbo = (int)(((d >> 16) & 16383u) << 4);
        r.sbo = (int)(((d >> 32) & 16383u) << 4);
        r.swz = (int)(((d >> 62) & 3u) + 4u * ((d >> 56) & 1u));
        r.base = (int)((d >> 49) & 7u);
        return r;
    }
    MX_INL int tn_swG(int z) { return z == 1 ? 8 : z == 2 ? 4 : z == 3 ? 2 : (z >= 4 && z <= 6) ? 8 : 1; }
    MX_INL int tn_swB(int z) { return z == 1 ? 3 : z == 2 ? 2 : z == 3 ? 1 : 0; }
    MX_INL int tn_swizzle(int z, int a, int b)
    {
        int p = a - (b << 7);
        if (z == 0) return a;
        if (z <= 3) return a ^ ((p & (((1 << tn_swB(z)) - 1) << 7)) >> 3);
        if (z == 4) return a ^ (((p >> 7) & 3) << 5);
        if (z == 5) return a ^ (((p >> 7) & 3) << 5) ^ (((p >> 7) & 1) << 3);
        if (z == 6) return a ^ (((p >> 7) & 1) << 6);
        return a;
    }
    // element offset of (mn, k) in the canonical layout (wgEO); t = elements per 16 bytes
    MX_INL int tn_eo(int z, bool km, int t, int mn, int k, int l, int s)
    {
        if (km)
        {
            if (z == 0) return (mn % 8) * t + (mn / 8) * s + (k % t) + (k / t) * l;
            return (mn % 8) * tn_swG(z) * t + (mn / 8) * s + (k % t) + (k / t) * t;
        }
        if (z == 0) return (mn % t) + (mn / t) * s + (k % 8) * t + (k / 8) * l;
        return (mn % t) + ((mn / t) % tn_swG(z)) * t + (mn / (t * tn_swG(z))) * l + (k % 8) * tn_swG(z) * t + (k / 8) * s;
    }
    MX_INL uint32_t tn_rdsh(uint32_t a, int nb)
    {
        if (nb == 1) return *(const SHARED_SPACE uint8_t *)(uintptr_t)a;
        if (nb == 2) return *(const SHARED_SPACE uint16_t *)(uintptr_t)a;
        return *(const SHARED_SPACE uint32_t *)(uintptr_t)a;
    }
    // element (mn, k) of the tile of descriptor d (wgElem), w bits per element
    MX_INL uint32_t tn_elem(const TnDesc &d, bool km, int w, int mn, int k)
    {
        int bo = tn_eo(d.swz, km, 128 / w, mn, k, d.lbo * 8 / w, d.sbo * 8 / w) * w;
        int a = tn_swizzle(d.swz, d.start + (bo >> 3), d.base);
        return (tn_rdsh((uint32_t)a, w >= 8 ? w / 8 : 1) >> (bo & 7)) & mx_mask(w);
    }
    // negation of float elements (imm-scale = -1): the sign bit
    MX_INL uint32_t tn_neg(int t, bool neg, uint32_t v)
    {
        if (!neg || mx_isint(t)) return v;
        int nb = (t == MX_E2M3 || t == MX_E3M2 || t == MX_E2M1) ? 5 : mx_ew(t) - 1;
        return v ^ (1u << nb);
    }

    // ---------------------------------------------------------------- sparse metadata (ptx_mma.sail)
    MX_INL int tn_spCw(int t) { return t == MX_TF32 ? 2 : mx_ew(t) == 4 ? 8 : 4; }
    MX_INL int tn_spNz(int t) { return t == MX_TF32 ? 1 : mx_ew(t) == 4 ? 4 : 2; }
    MX_INL int tn_spPos(int t, uint32_t nib, int j)
    {
        if (t == MX_TF32) return (int)((nib & 3) >> 1);
        if (mx_ew(t) == 4) return 2 * (int)((nib >> (2 * (j >> 1))) & 3) + (j & 1);
        return (int)((nib >> (2 * j)) & 3);
    }
    // (lane in the group of four, bit offset) of the nibble of chunk c of row g + 8 rh (spNibAt)
    MX_INL void tn_spNibAt(int t, int rh, int c, int nc, int sel, int &ln, int &b)
    {
        if (nc == 4) { ln = sel; b = 16 * rh + 4 * c; }
        else if (nc == 8 && mx_ew(t) >= 16) { ln = 2 * sel + c / 4; b = 16 * rh + 4 * (c % 4); }
        else if (nc == 8) { ln = 2 * sel + rh; b = 4 * c; }
        else { ln = 2 * (c / 8) + rh; b = 4 * (c % 8); }
    }

    // ---------------------------------------------------------------- wgmma.mma_async
    // ctl: N | K << 9 | atype << 18 | btype << 22 | dtype << 26 (f16 0, f32 1, s32 2) | sp << 28 | A in registers << 29
    //      | satfinite << 30; high word: bop | negA << 2 | negB << 3 | transA << 4 | transB << 5 | sel << 6
    // x: A registers 0..3, metadata 4, scale-d 5 (immediate form)
    MX_INL void tn_wgmma_core(uint32_t TN_GLOBAL *s, mx_u32x8 x, uint64_t da, uint64_t db, bool sd, uint64_t ctl)
    {
        const uint32_t c0 = (uint32_t)ctl, c1 = (uint32_t)(ctl >> 32);
        const int N = c0 & 511, K = (c0 >> 9) & 511;
        MxCfg cf;
        cf.K = K; cf.M = 64; cf.N = N;
        cf.at = (c0 >> 18) & 15;
        cf.bt = (c0 >> 22) & 15;
        cf.dt = cf.ct = (c0 >> 26) & 3;
        const bool sparse = (c0 >> 28) & 1, aregs = (c0 >> 29) & 1;
        cf.sat = (c0 >> 30) & 1;
        cf.bop = c1 & 3;
        cf.rnd = 0; cf.kind = 0; cf.acol = cf.bcol = 0; cf.nb = 1; cf.e43 = 0;
        cf.bidA = cf.tidA = cf.bidB = cf.tidB = 0;
        cf.sel = (c1 >> 6) & 3;
        cf.na = cf.nbr = cf.nc = cf.nd = 0;
        const bool negA = (c1 >> 2) & 1, negB = (c1 >> 3) & 1;
        const bool kmA = !((c1 >> 4) & 1), kmB = !((c1 >> 5) & 1);
        const TnDesc dA = tn_desc(da), dB = tn_desc(db);
        const int ewa = mx_ew(cf.at), ewb = mx_ew(cf.bt), ewd = mx_cdew(cf.dt);
        const int pera = 32 / ewa, perd = 32 / ewd;
        const uint32_t ma = mx_mask(ewa);
        const int l = mx_lane();
        const int u = (int)(tn_tid() & 127u);
        const int wq = u >> 5;
        const int nel = N / 2;
        const uint32_t meta = x[4];
        for (int e = 0; e < nel; e++)
        {
            const int rr = (l >> 2) + 8 * ((e & 3) >> 1); // row within the warp's 16
            const int r = 16 * wq + rr;
            const int col = 8 * (e >> 2) + 2 * (l & 3) + (e & 1);
            const int rg = e / perd, sh = ewd * (e % perd);
            uint32_t cv = sd ? (s[rg] >> sh) & mx_mask(ewd) : 0u;
            MxDot dot;
            mxd_init(dot, cf, cv, ewd);
            if (!sparse)
            {
                if (aregs)
                {
                    const int nw = (K * ewa) >> 5;
                    for (int w = 0; w < nw; w++)
                    {
                        int la, ra;
                        mx_invA(16, rr, w, la, ra);
                        uint32_t aw = mx_shfl(mx_pick8(x, ra), la);
                        for (int j = 0; j < pera; j++)
                        {
                            int k = w * pera + j;
                            uint32_t av = tn_neg(cf.at, negA, (aw >> (ewa * j)) & ma);
                            uint32_t bv = tn_neg(cf.bt, negB, tn_elem(dB, kmB, ewb, col, k));
                            mxd_add(dot, cf, av, bv, false, 0, 0);
                        }
                    }
                }
                else
                {
                    for (int k = 0; k < K; k++)
                    {
                        uint32_t av = tn_neg(cf.at, negA, tn_elem(dA, kmA, ewa, r, k));
                        uint32_t bv = tn_neg(cf.bt, negB, tn_elem(dB, kmB, ewb, col, k));
                        mxd_add(dot, cf, av, bv, false, 0, 0);
                    }
                }
            }
            else
            {
                const int t = cf.at;
                const int nz = tn_spNz(t), cw = tn_spCw(t);
                const int nch = K / cw;
                const int g = rr & 7, rh = rr >> 3;
                for (int ch = 0; ch < nch; ch++)
                {
                    uint32_t nib = 0;
                    if (nch == 4 || nch == 8 || nch == 16)
                    {
                        int ln, bo;
                        tn_spNibAt(t, rh, ch, nch, cf.sel, ln, bo);
                        nib = (mx_shfl(meta, 4 * g + ln) >> bo) & 15;
                    }
                    uint32_t areg = 0;
                    int sub0 = 0;
                    if (aregs)
                    {
                        int tg, uu;
                        if (ewa >= 16) { tg = ch & 3; uu = 2 * (ch >> 2) + rh; }
                        else { tg = (ch & 7) >> 1; uu = (ch & 1) | (rh << 1) | ((ch >> 3) << 2); }
                        int i0 = uu * nz;
                        areg = mx_shfl(mx_pick8(x, i0 / pera), 4 * g + tg);
                        sub0 = i0 % pera;
                    }
                    for (int p = 0; p < cw; p++)
                    {
                        uint32_t av = 0;
                        for (int j = 0; j < nz; j++)
                            if (tn_spPos(t, nib, j) == p)
                                av = tn_neg(t, negA, aregs ? (areg >> (ewa * (sub0 + j))) & ma
                                                           : tn_elem(dA, kmA, ewa, r, ch * nz + j));
                        uint32_t bv = tn_neg(cf.bt, negB, tn_elem(dB, kmB, ewb, col, cw * ch + p));
                        mxd_add(dot, cf, av, bv, false, 0, 0);
                    }
                }
            }
            uint32_t res = mxd_result(dot, cf, false, ewd) & mx_mask(ewd);
            s[rg] = (s[rg] & ~(mx_mask(ewd) << sh)) | (res << sh);
        }
    }

extern "C++"
{
    template <typename V, int NV>
    MX_INL V tn_wgmma(V c, mx_u32x8 x, uint64_t da, uint64_t db, bool sd, uint64_t ctl)
    {
        uint32_t TN_GLOBAL *s = tn_scratch(NV);
        const int N = (uint32_t)ctl & 511;
        const int nregs = (((uint32_t)ctl >> 26) & 3) == 0 ? N / 4 : N / 2;
        MX_UNROLL for (int j = 0; j < NV; j++)
            if (j < nregs) s[j] = c[j];
        tn_wgmma_core(s, x, da, db, sd, ctl);
        V d;
        MX_UNROLL for (int j = 0; j < NV; j++) d[j] = j < nregs ? s[j] : 0u;
        return d;
    }
}
#define TN_WGMMA(NV)                                                                                   \
    typedef uint32_t tn_u32x##NV __attribute__((ext_vector_type(NV)));                               \
    tn_u32x##NV FUNC(wgmma_##NV)(tn_u32x##NV c, mx_u32x8 x, uint64_t da, uint64_t db, uint64_t ctl)   \
    {                                                                                                  \
        return tn_wgmma<tn_u32x##NV, NV>(c, x, da, db, x[5] != 0, ctl);                              \
    }                                                                                                  \
    tn_u32x##NV FUNC(wgmma_##NV##_p)(tn_u32x##NV c, mx_u32x8 x, uint64_t da, uint64_t db, bool p,     \
                                    uint64_t ctl)                                                     \
    {                                                                                                  \
        return tn_wgmma<tn_u32x##NV, NV>(c, x, da, db, p, ctl);                                       \
    }
    TN_WGMMA(8)
    TN_WGMMA(16)
    TN_WGMMA(32)
    TN_WGMMA(64)
    TN_WGMMA(128)

    // ================================================================ tcgen05
    // Per-CTA memory: the wgmma scratch [0, 512 KB), the allocation state (4 KB), tensor memory (512 lanes of 1024
    // 32-bit columns; rows >= 128 are the M > 128 rows of the model's unbounded lanes) and the mma workspace (the A
    // and B tiles, the scale factors). The model's tensor memory starts empty: its words read 0, as the zeroed region.
#define TN_STATE_OFF TN_SCRATCH_BYTES
#define TN_TM_OFF (TN_STATE_OFF + 4096u)
#define TN_WA_OFF (TN_TM_OFF + 512u * 1024u * 4u)
#define TN_WB_OFF (TN_WA_OFF + 512u * 256u * 4u)
#define TN_WS_OFF (TN_WB_OFF + 256u * 512u * 4u)
#define TN_PERCTA (TN_WS_OFF + 8192u * 4u)
    MX_INL uint8_t TN_GLOBAL *tn_base() { return (uint8_t TN_GLOBAL *)(uintptr_t)__zluda_metal_percta_memory(TN_PERCTA); }
    MX_INL volatile uint32_t TN_GLOBAL *tn_state() { return (volatile uint32_t TN_GLOBAL *)(tn_base() + TN_STATE_OFF); }
    MX_INL uint32_t TN_GLOBAL *tn_cell(uint32_t lane, uint32_t col)
    {
        return (uint32_t TN_GLOBAL *)(tn_base() + TN_TM_OFF) + ((lane & 511u) << 10) + (col & 1023u);
    }
    MX_INL uint32_t tn_word(uint32_t lane, uint32_t col) { return *tn_cell(lane, col); }
    MX_INL uint32_t TN_GLOBAL *tn_ws(uint32_t off) { return (uint32_t TN_GLOBAL *)(tn_base() + off); }
    MX_INL uint32_t tn_lead() { return (uint32_t)__builtin_ctz((uint32_t)air_simd_ballot(true)); }
    // the active lanes one after the other (operations issued by single threads)
#define TN_SERIAL(...)                                       \
    {                                                        \
        uint32_t tn_m_ = (uint32_t)air_simd_ballot(true);    \
        uint32_t tn_me_ = (uint32_t)mx_lane();               \
        while (tn_m_)                                        \
        {                                                    \
            if ((uint32_t)__builtin_ctz(tn_m_) == tn_me_)    \
            {                                                \
                __VA_ARGS__                                  \
            }                                                \
            tn_m_ &= tn_m_ - 1;                              \
        }                                                    \
    }

    // ---------------------------------------------------------------- alloc / dealloc
    // state: 32 allocations (start + 1, 0 = free; columns), word 64: an exclusive allocation is live
    MX_INL int tn_alloc_try(uint32_t n, bool ex)
    {
        volatile uint32_t TN_GLOBAL *st = tn_state();
        if (ex)
        {
            if (n < 32 || n > 576 || n % 32 != 0) return -2;
            for (int i = 0; i < 32; i++)
                if (st[2 * i]) return -1;
            st[0] = 1;
            st[1] = n;
            st[64] = 1;
            return 0;
        }
        if (n < 32 || n > 512 || (n & (n - 1)) != 0) return -2;
        if (st[64]) return -1;
        for (uint32_t b = 0; b + n <= 512; b += n)
        {
            bool free = true;
            for (int i = 0; i < 32; i++)
            {
                uint32_t s = st[2 * i];
                if (s && b < s - 1 + st[2 * i + 1] && s - 1 < b + n) free = false;
            }
            if (!free) continue;
            for (int i = 0; i < 32; i++)
                if (!st[2 * i])
                {
                    st[2 * i] = b + 1;
                    st[2 * i + 1] = n;
                    return (int)b;
                }
            return -1;
        }
        return -1;
    }
    // warp-collective: the warp's allocation is made once; it blocks (yields at threadgroup barriers) until the
    // columns are available
    MX_INL int tn_alloc(uint32_t n, uint32_t ex)
    {
        uint32_t lead = tn_lead();
        int r = -1;
        for (int round = 0;; round++)
        {
            if ((uint32_t)mx_lane() == lead) r = tn_alloc_try(n, ex != 0);
            r = (int)air_simd_shuffle((uint32_t)r, (uint16_t)lead);
            if (r != -1 || round >= 256) break;
            air_wg_barrier(3, 1);
        }
        return r;
    }
    void FUNC(tcgen05_alloc)(SHARED_SPACE uint32_t *dst, uint32_t n, uint32_t ex)
    {
        int r = tn_alloc(n, ex);
        if (r >= 0) *dst = (uint32_t)r;
    }
    void FUNC(tcgen05_alloc_g)(uint32_t *dst, uint32_t n, uint32_t ex)
    {
        int r = tn_alloc(n, ex);
        if (r >= 0) *(SHARED_SPACE uint32_t *)dst = (uint32_t)r;
    }
    void FUNC(tcgen05_dealloc)(uint32_t ta, uint32_t n, uint32_t ex)
    {
        uint32_t lead = tn_lead();
        if ((uint32_t)mx_lane() == lead)
        {
            volatile uint32_t TN_GLOBAL *st = tn_state();
            uint32_t b = ta & 65535u;
            for (int i = 0; i < 32; i++)
                if (st[2 * i] == b + 1 && st[2 * i + 1] == n)
                {
                    st[2 * i] = 0;
                    st[64] = 0;
                    break;
                }
        }
        (void)ex;
    }

    // ---------------------------------------------------------------- ld / st (tcCell, tcPkCell)
    // shape: 0 .32x32b, 1 .16x64b, 2 .16x128b, 3 .16x256b, 4 .16x32bx2
    MX_INL void tn_cellpos(int shape, uint32_t a, int t, int i, uint32_t h, uint32_t &ln, uint32_t &cl)
    {
        uint32_t l = a >> 16, c = a & 65535u;
        int g = t >> 2;
        switch (shape)
        {
        case 0: ln = l + t; cl = c + i; break;
        case 1: ln = l + g + 8 * (t & 1); cl = c + 2 * i + ((t >> 1) & 1); break;
        case 2: ln = l + g + 8 * (i & 1); cl = c + 4 * (i >> 1) + (t & 3); break;
        case 3: ln = l + g + 8 * ((i & 3) >> 1); cl = c + 8 * (i >> 2) + 2 * (t & 3) + (i & 1); break;
        default: ln = l + (t & 15); cl = c + i + (t >> 4) * h; break;
        }
    }
    MX_INL void tn_pkpos(int shape, uint32_t a, int t, int i, uint32_t h, int half, uint32_t &ln, uint32_t &cl)
    {
        if (shape >= 1 && shape <= 3)
        {
            tn_cellpos(shape, a, t, i, h, ln, cl);
            uint32_t c = a & 65535u;
            cl = c + 2 * (cl - c) + half;
        }
        else
            tn_cellpos(shape, a, t, 2 * i + half, h, ln, cl);
    }
    // fpMinMax (ieee.sail) of f32 bits, no ftz / xorsign
    MX_INL bool tn_f32nan(uint32_t a) { return (a & 0x7FFFFFFFu) > 0x7F800000u; }
    MX_INL int32_t tn_f32key(uint32_t a) { return (a >> 31) ? -(int32_t)(a & 0x7FFFFFFFu) : (int32_t)a; }
    MX_INL uint32_t tn_fminmax(bool nanf, uint32_t a, uint32_t b, bool mn)
    {
        if (tn_f32nan(a) || tn_f32nan(b))
        {
            if (nanf || (tn_f32nan(a) && tn_f32nan(b))) return 0x7FFFFFFFu;
            return tn_f32nan(a) ? b : a;
        }
        if ((a & 0x7FFFFFFFu) != 0 || (b & 0x7FFFFFFFu) != 0)
        {
            int32_t ka = tn_f32key(a), kb = tn_f32key(b);
            if (ka < kb) return mn ? a : b;
            if (kb < ka) return mn ? b : a;
        }
        return mn ? (a | b) : (a & b);
    }
    // tcRed: ctl bit 13 min, 14 abs, 15 NaN, 16-17 type (0 f32, 1 u32, 2 s32)
    MX_INL uint32_t tn_red(uint32_t acc, uint32_t a1, uint32_t ctl)
    {
        const bool mn = (ctl >> 13) & 1, ab = (ctl >> 14) & 1, nanf = (ctl >> 15) & 1;
        const int ty = (ctl >> 16) & 3;
        uint32_t a = ab ? (a1 & 0x7FFFFFFFu) : a1;
        if (ty == 0) return tn_fminmax(nanf, acc, a, mn);
        if (ty == 1) return mn ? (a1 < acc ? a1 : acc) : (a1 > acc ? a1 : acc);
        return mn ? ((int32_t)a1 < (int32_t)acc ? a1 : acc) : ((int32_t)a1 > (int32_t)acc ? a1 : acc);
    }

extern "C++"
{
    // ctl: shape | n << 3 | pack << 11 | red << 12 | ... | count of the reduced values << 19; the reduction value
    // follows the n loaded registers
    template <typename V, int NV>
    MX_INL V tn_ld(uint32_t ta, uint32_t h, uint32_t off, uint64_t ctl64)
    {
        const uint32_t ctl = (uint32_t)ctl64;
        const int shape = ctl & 7, n = (ctl >> 3) & 255;
        const bool pk = (ctl >> 11) & 1, red = (ctl >> 12) & 1;
        const int nred = (ctl >> 19) & 1023;
        const uint32_t a = ta + off;
        const int t = mx_lane();
        V d;
        uint32_t acc = 0;
        MX_UNROLL for (int i = 0; i < NV; i++)
        {
            uint32_t v = 0;
            if (i < n)
            {
                uint32_t ln, cl;
                if (pk)
                {
                    tn_pkpos(shape, a, t, i, h, 0, ln, cl);
                    uint32_t lo = tn_word(ln, cl) & 0xFFFFu;
                    tn_pkpos(shape, a, t, i, h, 1, ln, cl);
                    v = lo | (tn_word(ln, cl) << 16);
                }
                else
                {
                    tn_cellpos(shape, a, t, i, h, ln, cl);
                    v = tn_word(ln, cl);
                }
                if (red && i < nred)
                    acc = i == 0 ? (((ctl >> 14) & 1) ? (v & 0x7FFFFFFFu) : v) : tn_red(acc, v, ctl);
            }
            d[i] = v;
        }
        if (red)
            MX_UNROLL for (int i = 0; i < NV; i++)
                if (i == n) d[i] = acc;
        return d;
    }
    // tcgen05.ld.spcompress (spcKeep / spcFin of ptx_spc.sail with .f32 elements and 2-bit indices): of every group
    // of four loaded values the two that come first by value (max, or min with ctl bit 13; |x| with bit 14; NaN
    // first) in index order; the result is the nm metadata words, the nc words of kept values (and the reduction)
    template <typename V, int NV>
    MX_INL V tn_ldspc(uint32_t ta, uint32_t h, uint32_t off, uint64_t ctl64)
    {
        const uint32_t ctl = (uint32_t)ctl64;
        const int shape = ctl & 7;
        const bool red = (ctl >> 12) & 1, mn = (ctl >> 13) & 1, ab = (ctl >> 14) & 1;
        const int nload = (ctl >> 19) & 1023;
        const int nm = (int)((ctl64 >> 32) & 255), nc = (int)((ctl64 >> 40) & 255);
        const uint32_t a = ta + off;
        const int t = mx_lane();
        uint32_t mdw[4] = {0, 0, 0, 0};
        uint32_t cs[64];
        uint32_t acc = 0;
        MX_UNROLL for (int g = 0; g < 32; g++)
        {
            if (4 * g + 3 < nload)
            {
                uint32_t v[4];
                int32_t key[4];
                bool nan[4];
                MX_UNROLL for (int q = 0; q < 4; q++)
                {
                    uint32_t ln, cl;
                    tn_cellpos(shape, a, t, 4 * g + q, h, ln, cl);
                    v[q] = tn_word(ln, cl);
                    uint32_t x = ab ? (v[q] & 0x7FFFFFFFu) : v[q];
                    nan[q] = tn_f32nan(v[q]);
                    key[q] = tn_f32key(x);
                    if (red)
                        acc = (g == 0 && q == 0) ? (ab ? (v[q] & 0x7FFFFFFFu) : v[q]) : tn_red(acc, v[q], ctl);
                }
                int k0 = -1, k1 = -1;
                MX_UNROLL for (int i = 0; i < 4; i++)
                {
                    int r = 0;
                    MX_UNROLL for (int j = 0; j < 4; j++)
                    {
                        if (j == i) continue;
                        bool before;
                        if (nan[j]) before = !nan[i] || j < i;
                        else before = !nan[i] && ((mn ? key[j] < key[i] : key[i] < key[j]) || (key[j] == key[i] && j < i));
                        r += before ? 1 : 0;
                    }
                    if (r < 2)
                    {
                        if (k0 < 0) k0 = i;
                        else k1 = i;
                    }
                }
                uint32_t c0 = k0 == 0 ? v[0] : k0 == 1 ? v[1] : k0 == 2 ? v[2] : v[3];
                uint32_t c1 = k1 == 0 ? v[0] : k1 == 1 ? v[1] : k1 == 2 ? v[2] : v[3];
                if (2 * g + 1 < 64) { cs[2 * g] = c0; cs[2 * g + 1] = c1; }
                uint32_t idx = (uint32_t)(k0 & 3) | ((uint32_t)(k1 & 3) << 2);
                mdw[g / 8] |= idx << (4 * (g % 8));
            }
            else if (2 * g + 1 < 64) { cs[2 * g] = 0; cs[2 * g + 1] = 0; }
        }
        V d;
        MX_UNROLL for (int i = 0; i < NV; i++)
        {
            uint32_t x = 0;
            if (i < nm) x = i < 4 ? mdw[i] : 0u;
            else if (i < nm + nc) x = i - nm < 64 ? cs[i - nm] : 0u;
            else if (red && i == nm + nc) x = acc;
            d[i] = x;
        }
        return d;
    }
    template <typename V, int NV>
    MX_INL void tn_st(uint32_t ta, uint32_t h, uint32_t off, V v, uint64_t ctl64)
    {
        const uint32_t ctl = (uint32_t)ctl64;
        const int shape = ctl & 7, n = (ctl >> 3) & 255;
        const bool up = (ctl >> 11) & 1;
        const uint32_t a = ta + off;
        const int t = mx_lane();
        MX_UNROLL for (int i = 0; i < NV; i++)
        {
            if (i < n)
            {
                uint32_t ln, cl;
                if (up)
                {
                    tn_pkpos(shape, a, t, i, h, 0, ln, cl);
                    *tn_cell(ln, cl) = v[i] & 0xFFFFu;
                    tn_pkpos(shape, a, t, i, h, 1, ln, cl);
                    *tn_cell(ln, cl) = v[i] >> 16;
                }
                else
                {
                    tn_cellpos(shape, a, t, i, h, ln, cl);
                    *tn_cell(ln, cl) = v[i];
                }
            }
        }
    }
}
#define TN_LDST(NV)                                                                                     \
    tn_u32x##NV FUNC(tcgen05_ld_##NV)(uint32_t ta, uint32_t h, uint32_t off, uint64_t ctl)             \
    {                                                                                                   \
        return tn_ld<tn_u32x##NV, NV>(ta, h, off, ctl);                                                \
    }                                                                                                   \
    tn_u32x##NV FUNC(tcgen05_ldspc_##NV)(uint32_t ta, uint32_t h, uint32_t off, uint64_t ctl)          \
    {                                                                                                   \
        return tn_ldspc<tn_u32x##NV, NV>(ta, h, off, ctl);                                             \
    }                                                                                                   \
    void FUNC(tcgen05_st_##NV)(uint32_t ta, uint32_t h, uint32_t off, tn_u32x##NV v, uint64_t ctl)     \
    {                                                                                                   \
        tn_st<tn_u32x##NV, NV>(ta, h, off, v, ctl);                                                    \
    }
    TN_LDST(8)
    TN_LDST(16)
    TN_LDST(32)
    TN_LDST(64)
    TN_LDST(128)

    // ---------------------------------------------------------------- tcgen05.mma
#define TN_S16 14
    // kinds (tensor.rs mma_kind): 0 f16, 1 tf32, 2 f8f6f4, 3 i8, 4 ti16, 5 mxf8f6f4, 6 mxf4, 7 mxf4nvf4
    MX_INL int tn_f8(uint32_t c)
    {
        switch (c)
        {
        case 1: return MX_E5M2;
        case 3: return MX_E2M3;
        case 4: return MX_E3M2;
        case 5: return MX_E2M1;
        default: return MX_E4M3;
        }
    }
    // tcAT (i: the idesc for A, idesc >> 3 for B)
    MX_INL int tn_tcT(int kind, uint32_t i)
    {
        uint32_t f = (i >> 7) & 7;
        switch (kind)
        {
        case 6:
        case 7: return MX_E2M1P;
        case 1: return MX_TF32;
        case 3: return f == 0 ? MX_U8 : MX_S8;
        case 4: return TN_S16;
        case 2:
        case 5: return tn_f8(f);
        default: return f == 0 ? MX_F16 : MX_BF16;
        }
    }
    MX_INL int tn_ewT(int t) { return t == TN_S16 ? 16 : mx_ew(t); }
    MX_INL bool tn_sub(int t) { return t == MX_E2M1 || t == MX_E2M3 || t == MX_E3M2; }
    MX_INL int tn_isMx(int kind) { return kind >= 5; }
    // tcK
    MX_INL int tn_tcK(int kind, uint32_t i)
    {
        switch (kind)
        {
        case 5: return ((i >> 31) & 1) ? 64 : 32;
        case 6:
        case 7:
        {
            int kf = 2 * ((i >> 3) & 1) + ((i >> 31) & 1);
            return kf == 0 ? 64 : kf == 1 ? 96 : 128;
        }
        case 1: return 8;
        case 3: return 32;
        case 4: return 16;
        case 2: return ((i >> 29) & 1) ? 64 : 32;
        default: return 16;
        }
    }
    // the tcgen05 shared memory descriptor (tcDesc: swizzle codes 2, 4, 6, 1 are wgmma's 1, 2, 3 and 4)
    MX_INL TnDesc tn_tcdesc(uint64_t d)
    {
        TnDesc r = tn_desc(d);
        uint32_t c = (uint32_t)(d >> 61) & 7;
        r.swz = c == 2 ? 1 : c == 4 ? 2 : c == 6 ? 3 : c == 1 ? 4 : 0;
        return r;
    }
    // tcSE: element (r, kk) of a shared-memory tile; k is the Dense-K dimension
    MX_INL uint32_t tn_se(const TnDesc &d, bool km, int t, int r, int kk, int k)
    {
        if (km && tn_sub(t))
        {
            int nb = t == MX_E2M1 ? 4 : 6;
            int bp = k == 32 ? 128 * (kk / 16) + (kk % 16) * nb : kk * nb;
            uint32_t lo = tn_elem(d, true, 8, r, bp >> 3);
            uint32_t hi = (bp & 7) + nb > 8 ? tn_elem(d, true, 8, r, (bp >> 3) + 1) << 8 : 0u;
            uint32_t v = ((lo | hi) >> (bp & 7)) & mx_mask(nb);
            return nb == 4 ? v << 2 : v;
        }
        return tn_elem(d, km, tn_ewT(t), r, kk);
    }
    // tcNeg
    MX_INL uint32_t tn_tcneg(int t, bool neg, uint32_t v)
    {
        if (!neg) return v;
        if (t == TN_S16) return (uint32_t)(-(int32_t)(int16_t)(uint16_t)v) & 0xFFFFu;
        return tn_neg(t, true, v);
    }
    MX_INL int tn_spCwT(int t) { return t == MX_TF32 ? 2 : tn_ewT(t) == 4 ? 8 : 4; }
    MX_INL int tn_spNzT(int t) { return t == MX_TF32 ? 1 : tn_ewT(t) == 4 ? 4 : 2; }
    MX_INL int tn_spPosT(int t, uint32_t nib, int j)
    {
        if (t == MX_TF32) return (int)((nib & 3) >> 1);
        if (tn_ewT(t) == 4) return 2 * (int)((nib >> (2 * (j >> 1))) & 3) + (j & 1);
        return (int)((nib >> (2 * j)) & 3);
    }
    // tcNib
    MX_INL uint32_t tn_tcnib(int t, int r, uint32_t mc, uint32_t sel, int c, int kp)
    {
        if ((2 * kp) / tn_spCwT(t) <= 8)
            return (tn_word(16 * (r / 16) + 8 * (c / 4) + r % 8, mc + sel) >> (16 * ((r % 16) / 8) + 4 * (c % 4))) & 15;
        return (tn_word(r, mc + c / 8) >> (4 * (c % 8))) & 15;
    }
    // block scale factor (tcScaleF): st 1 UE8M0, 2 UE5M3, 0 UE4M3
    MX_INL MxF tn_scaleF(int st, uint32_t x)
    {
        if (st == 1) return x == 255 ? mx_nan() : mx_fin(false, 1, (int)x - 127);
        if (st == 2)
        {
            if (x == 255) return mx_nan();
            uint32_t hi = x >> 3, m = x & 7;
            return hi == 0 ? mx_fin(false, m, 1 - 15 - 3) : mx_fin(false, m + 8, (int)hi - 15 - 3);
        }
        return mx_scale(1, x);
    }
    MX_INL int64_t tn_elemI(int t, uint32_t b)
    {
        if (t == TN_S16) return (int64_t)(int16_t)(uint16_t)b;
        return mx_elemI(t, b);
    }
    // tcZero (.ws zero-column mask)
    MX_INL bool tn_zbit(uint64_t z, int i, int p)
    {
        if (!((z >> 39) & 1)) return false;
        int v0 = (int)((z >> (32 + i)) & 1), sc = (int)((z >> (8 * i)) & 255);
        int on = (int)((z >> 40) & 255) + 1, zr = (int)((z >> 48) & 255) + 1;
        int q = (p + sc) % (on + zr);
        return (v0 == 1 && q < on) || (v0 == 0 && q >= zr);
    }
    MX_INL bool tn_zero(bool hz, uint64_t z, int md, int n, int c)
    {
        if (!hz || md <= 0 || md > 128) return false;
        int w = n / (128 / md);
        if (w <= 0) return false;
        return tn_zbit(z, c / w, c % w);
    }

    // x: d taddr, off, A taddr, off, idesc, meta taddr, off, lut taddr, off, scale-A taddr, off, scale-B taddr, off,
    //    enable-input-d, scale-input-d, 0, disable-output-lane 16..23
    // ctl: kind | sp << 3 | lut << 4 | block_scale << 5 | ws << 6 | A in shared memory << 7 | ashift << 8
    //      | zero-column mask << 9 | disable-output-lane << 10 | enable from a predicate << 11 | scale vector << 12
    MX_INL void tn_mma(tn_u32x32 xx, uint64_t da, uint64_t db, uint64_t zm, bool en, uint32_t ctl)
    {
        const int kind = ctl & 7;
        const bool sp = (ctl >> 3) & 1, lut = (ctl >> 4) & 1, bs = (ctl >> 5) & 1, ws = (ctl >> 6) & 1;
        const bool ad = (ctl >> 7) & 1, ash = (ctl >> 8) & 1, hz = (ctl >> 9) & 1, hm = (ctl >> 10) & 1;
        const int sv = (ctl >> 12) & 3;
        const uint32_t dcol = (xx[0] + xx[1]) & 65535u;
        const uint32_t ta = xx[2] + xx[3];
        const uint32_t id = xx[4];
        const uint32_t mcol = (xx[5] + xx[6]) & 65535u, lcol = (xx[7] + xx[8]) & 65535u;
        const uint32_t sacol = (xx[9] + xx[10]) & 65535u, sbcol = (xx[11] + xx[12]) & 65535u;
        const uint32_t sc = ws ? 0u : xx[14];
        const int md = tn_isMx(kind) ? (int)((id >> 27) & 3) * 128 : (int)((id >> 24) & 31) * 16;
        const int n = (int)((id >> 17) & 63) * 8;
        const int K = tn_tcK(kind, id);
        const int ks = sp ? 2 * K : K;
        const int at = tn_tcT(kind, id), bt = tn_tcT(kind, id >> 3);
        const int ewa = tn_ewT(at);
        // the accumulator: f16 / f32 (dtype field), s32 for i8 / ti16, f32 for the block-scaled kinds
        const int dc = (bs || tn_isMx(kind)) ? 1 : (kind == 3 || kind == 4) ? 2 : (((id >> 4) & 3) == 0 ? 0 : 1);
        const int ewd = dc == 0 ? 16 : 32;
        const bool isint = kind == 3 || kind == 4;
        const bool sat = !bs && ((id >> 3) & 1);
        const bool negA = (id >> 13) & 1, negB = (id >> 14) & 1;
        const bool kmA = !((id >> 15) & 1), kmB = !((id >> 16) & 1);
        const uint32_t sel = id & 3;
        if (md > 512 || n > 512 || ks > 256 || md == 0 || n == 0) return;
        const TnDesc dA = tn_tcdesc(da), dB = tn_tcdesc(db);
        const int sh = hz ? (int)((zm >> 56) & 63) : 0;
        uint32_t TN_GLOBAL *WA = tn_ws(TN_WA_OFF);
        uint32_t TN_GLOBAL *WB = tn_ws(TN_WB_OFF);
        uint32_t TN_GLOBAL *WS = tn_ws(TN_WS_OFF);
        // B (ks x n), tcTileB
        for (int nn = 0; nn < n; nn++)
            for (int kk = 0; kk < ks; kk++)
            {
                uint32_t v = 0;
                if (!tn_zero(hz, zm, md, n, nn))
                {
                    if (lut)
                    {
                        int bp = 192 * (int)((db >> 53) & 1) + 3 * kk;
                        uint32_t lo = tn_elem(dB, true, 8, nn + sh, bp >> 3);
                        uint32_t hi = (bp & 7) + 3 > 8 ? tn_elem(dB, true, 8, nn + sh, (bp >> 3) + 1) << 8 : 0u;
                        uint32_t j = ((lo | hi) >> (bp & 7)) & 7;
                        v = (tn_word((nn + sh) / 8, lcol + j / 4) >> (8 * (j % 4))) & 255;
                    }
                    else
                        v = tn_se(dB, kmB, bt, nn + sh, kk, K);
                    v = tn_tcneg(bt, negB, v);
                }
                WB[kk * 512 + nn] = v;
            }
        // A (md x ks): the tile (tcTileA), the sparse expansion (tcSpExp), .ashift
        const int spt = ((kind == 6 || kind == 7) && ((id >> 12) & 1)) ? MX_S8 : at;
        for (int r = 0; r < md; r++)
        {
            const int r0 = ash ? (r > 0 ? r - 1 : 0) : r;
            uint32_t TN_GLOBAL *row = WA + r * 256;
            for (int kk = 0; kk < ks; kk++) row[kk] = 0;
            for (int kk = 0; kk < K; kk++)
            {
                uint32_t v;
                if (ad) v = tn_se(dA, kmA, at, r0, kk, K);
                else if (tn_sub(at) && at != MX_E2M1 && K == 64)
                {
                    int bp = 6 * kk;
                    uint64_t lo = tn_word(r0, (ta & 65535u) + bp / 32);
                    uint64_t hi = (bp % 32) + 6 > 32 ? (uint64_t)tn_word(r0, (ta & 65535u) + bp / 32 + 1) << 32 : 0;
                    v = (uint32_t)(((lo | hi) >> (bp % 32)) & 63);
                }
                else
                    v = (tn_word(r0, (ta & 65535u) + (kk * ewa) / 32) >> ((kk * ewa) % 32)) & mx_mask(ewa);
                v = tn_tcneg(at, negA, v);
                int pos = kk;
                if (sp)
                {
                    const int nz = tn_spNzT(spt), cw = tn_spCwT(spt);
                    uint32_t nib = tn_tcnib(spt, r0, mcol, sel, kk / nz, K);
                    pos = cw * (kk / nz) + tn_spPosT(spt, nib, kk % nz);
                }
                if (pos < ks) row[pos] = v;
            }
        }
        // block scales (tcSF)
        int blk = 1, nb = 1, st = 1;
        if (bs)
        {
            blk = kind == 5 ? ks : sv == 1 ? 16 : sv == 3 ? ks / 4 : sv == 2 ? ks / 2 : 32;
            nb = ks / blk;
            st = (kind == 6 || kind == 7) ? (int)((id >> 23) & 3) : 1;
            const int ida = (id >> 29) & 3, idb = (id >> 4) & 3;
            const bool l128 = (id >> 26) & 1;
            for (int r = 0; r < md; r++)
                for (int j = 0; j < nb && j < 8; j++)
                {
                    int b = ida + j;
                    uint32_t w = l128 ? tn_word(r, sacol + b / 4)
                                      : tn_word(r % 32, sacol + (b / 4) * (md > 128 ? 8 : 4) + r / 32);
                    WS[r * 8 + j] = (w >> (8 * (b % 4))) & 255;
                }
            for (int nn = 0; nn < n; nn++)
                for (int j = 0; j < nb && j < 8; j++)
                {
                    int b = idb + j;
                    uint32_t w = tn_word(nn % 32, sbcol + (b / 4) * (n > 128 ? 8 : 4) + nn / 32);
                    WS[4096 + nn * 8 + j] = (w >> (8 * (b % 4))) & 255;
                }
        }
        // the products, row by row
        for (int r = 0; r < md; r++)
        {
            const bool off = hm && r < 256 && ((xx[16 + r / 32] >> (r % 32)) & 1);
            const uint32_t TN_GLOBAL *row = WA + r * 256;
            for (int c = 0; c < n; c++)
            {
                uint32_t cv = en ? tn_word(r, dcol + c) & mx_mask(ewd) : 0u;
                uint32_t res;
                if (isint)
                {
                    int64_t acc = (int64_t)(int32_t)cv;
                    for (int kk = 0; kk < ks; kk++)
                        acc += tn_elemI(at, row[kk]) * tn_elemI(bt, WB[kk * 512 + c]);
                    if (sat) acc = acc < -2147483648ll ? -2147483648ll : acc > 2147483647ll ? 2147483647ll : acc;
                    res = (uint32_t)acc;
                }
                else
                {
                    MxAcc<20> acc;
                    mxa_init(acc, -320);
                    MxF cf = ewd == 16 ? mx_decIEEE(cv, 11, 5, 16) : mx_decIEEE(cv, 24, 8, 32);
                    if (en && sc > 0 && cf.cls == 0 && !bs)
                    {
                        // scale-input-d: C * 2^-s rounded to the accumulator format
                        MxAcc<20> t;
                        mxa_init(t, -320);
                        cf.e -= (int)sc;
                        mxa_addF(t, cf);
                        uint32_t cb = ewd == 16 ? (uint32_t)mxa_round(t, 11, 5, 16, 0, false)
                                                : (uint32_t)mxa_round(t, 24, 8, 32, 0, false);
                        cf = ewd == 16 ? mx_decIEEE(cb, 11, 5, 16) : mx_decIEEE(cb, 24, 8, 32);
                    }
                    mxa_addF(acc, cf);
                    for (int kk = 0; kk < ks; kk++)
                    {
                        MxF p = mx_mul(mx_elemF(at, row[kk]), mx_elemF(bt, WB[kk * 512 + c]));
                        if (bs)
                        {
                            int j = kk / blk;
                            p = mx_mul(p, tn_scaleF(st, WS[r * 8 + j]));
                            p = mx_mul(p, tn_scaleF(st, WS[4096 + c * 8 + j]));
                        }
                        mxa_addF(acc, p);
                    }
                    res = ewd == 16 ? (uint32_t)mxa_round(acc, 11, 5, 16, 0, sat)
                                    : (uint32_t)mxa_round(acc, 24, 8, 32, 0, sat);
                }
                if (!off) *tn_cell(r, dcol + c) = res;
            }
        }
    }
    void FUNC(tcgen05_mma)(tn_u32x32 x, uint64_t da, uint64_t db, uint64_t zm, uint64_t ctl)
    {
        TN_SERIAL(tn_mma(x, da, db, zm, x[13] != 0, (uint32_t)ctl);)
    }
    void FUNC(tcgen05_mma_p)(tn_u32x32 x, uint64_t da, uint64_t db, uint64_t zm, bool p, uint64_t ctl)
    {
        TN_SERIAL(tn_mma(x, da, db, zm, p, (uint32_t)ctl);)
    }

    // ---------------------------------------------------------------- cp / shift
    // ctl: rows | words << 8 | destination pattern << 12 (0 -, 1 warpx4, 2 warpx2::02_13, 3 warpx2::01_23)
    //      | decompression << 16 (6 b6x16_p32, 4 b4x16_p64)
    MX_INL void tn_cp(uint32_t a, uint64_t sd, uint32_t ctl)
    {
        const int rows = ctl & 255, words = (ctl >> 8) & 15, pat = (ctl >> 12) & 3, dec = (ctl >> 16) & 7;
        const TnDesc d = tn_tcdesc(sd);
        const uint32_t l = a >> 16, cl = a & 65535u;
        for (int i = 0; i < rows; i++)
        {
            uint32_t w[8];
            MX_UNROLL for (int j = 0; j < 8; j++) w[j] = j < words ? tn_elem(d, true, 32, i, j) : 0u;
            if (dec)
            {
                MX_UNROLL for (int g = 0; g < 2; g++)
                {
                    if (4 * g < words)
                    {
                        uint32_t s0 = w[4 * g], s1 = w[4 * g + 1], s2 = w[4 * g + 2];
                        uint32_t o[4] = {0, 0, 0, 0};
                        MX_UNROLL for (int e = 0; e < 16; e++)
                        {
                            int bp = dec * e;
                            uint32_t lo = bp < 32 ? s0 : bp < 64 ? s1 : s2;
                            uint32_t hi = bp < 32 ? s1 : bp < 64 ? s2 : 0u;
                            int sh = bp & 31;
                            uint32_t v = (sh ? (lo >> sh) | (hi << (32 - sh)) : lo) & ((1u << dec) - 1u);
                            if (dec == 4) v <<= 2;
                            o[e / 4] |= v << (8 * (e % 4));
                        }
                        MX_UNROLL for (int q = 0; q < 4; q++) w[4 * g + q] = o[q];
                    }
                }
            }
            uint32_t r0 = i, r1 = 0, r2 = 0, r3 = 0;
            int nd = 1;
            if (pat == 1) { r1 = i + 32; r2 = i + 64; r3 = i + 96; nd = 4; }
            else if (pat == 2) { r0 = i % 32 + 32 * (2 * (i / 32)); r1 = r0 + 32; nd = 2; }
            else if (pat == 3) { r0 = i % 32 + 32 * (i / 32); r1 = i % 32 + 32 * (i / 32 + 2); nd = 2; }
            MX_UNROLL for (int q = 0; q < 4; q++)
            {
                if (q < nd)
                {
                    uint32_t rr = q == 0 ? r0 : q == 1 ? r1 : q == 2 ? r2 : r3;
                    MX_UNROLL for (int j = 0; j < 8; j++)
                        if (j < words) *tn_cell(l + rr, cl + j) = w[j];
                }
            }
        }
    }
    void FUNC(tcgen05_cp)(uint32_t ta, uint32_t off, uint64_t sd, uint32_t ctl) { TN_SERIAL(tn_cp(ta + off, sd, ctl);) }
    MX_INL void tn_shift(uint32_t a)
    {
        const uint32_t l = a >> 16, cl = a & 65535u;
        for (int r = 30; r >= 0; r--)
            MX_UNROLL for (int w = 0; w < 8; w++) *tn_cell(l + r + 1, cl + w) = tn_word(l + r, cl + w);
    }
    void FUNC(tcgen05_shift)(uint32_t ta, uint32_t off) { TN_SERIAL(tn_shift(ta + off);) }
