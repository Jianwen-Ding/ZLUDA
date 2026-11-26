// Included inside extern "C" of zluda_ptx_metal_impl.cpp (after matrix.inc): helpers of
// ptx_parser/src/helper_table/tensor.rs, the warpgroup (wgmma) and tensor-memory (tcgen05)
// matrix instructions, following PTX-formalization's Sail model (sail/ptx/ptx_wg.sail,
// ptx_tc.sail). The arithmetic is the warp matrix engine's (matrix.inc: exact products, an exact
// fixed-point sum rounded once).
//
// A CTA is one AGX threadgroup and a warpgroup four consecutive SIMD-groups. The CTA's per-CTA
// global memory (the hidden kernel argument "perctamemory" of llvm-air2agx, see air_lower.cpp)
// holds a scratch area of 128 words per thread (wgmma accumulators) and the emulated tensor
// memory. No private array is indexed dynamically (see matrix.inc).

    uint64_t __zluda_metal_percta_memory(uint32_t bytes);

#define TN_SCRATCH_BYTES (1024u * 512u)
#define TN_GLOBAL MX_GLOBAL

    void FUNC(tensor_nop)() {}

    MX_INL uint32_t tn_tid()
    {
        return __zluda_metal_thread_position_in_threadgroup(0) +
               __zluda_metal_threads_per_threadgroup(0) *
                   (__zluda_metal_thread_position_in_threadgroup(1) +
                    __zluda_metal_threads_per_threadgroup(1) * __zluda_metal_thread_position_in_threadgroup(2));
    }
    // this thread's 128 scratch words
    MX_INL uint32_t TN_GLOBAL *tn_scratch()
    {
        uint64_t b = __zluda_metal_percta_memory(TN_SCRATCH_BYTES);
        return (uint32_t TN_GLOBAL *)(uintptr_t)(b + (uint64_t)(tn_tid() & 1023u) * 512u);
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
        uint32_t TN_GLOBAL *s = tn_scratch();
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
