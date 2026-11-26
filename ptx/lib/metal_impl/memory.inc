// Included inside extern "C" of zluda_ptx_metal_impl.cpp: helpers of ptx_parser/src/helper_table/memory.rs.
//
// Memory family: mbarrier objects, cp.async / cp.async.bulk / TMA (completed at issue: a legal schedule of the
// asynchronous proxy), isspacep, st.bulk, ... Semantics: PTX-formalization sail/ptx/ptx_async.sail, ptx_tma.sail.
//
// Execution model: a CTA is one threadgroup; its SIMD groups run until they reach a barrier (the AGX simulator
// runs them one after the other, a legal schedule). Read-modify-write sequences on the helpers' shared tables
// are serialised over the active lanes of a SIMD group (mem_serial). Blocking waits (mbarrier.try_wait) yield
// with threadgroup barriers until the phase completes, and time out (return False, as the model does when no
// thread can make progress) after MB_TIMEOUT_ROUNDS rounds without any mbarrier state change.

    // ---------------------------------------------------------------- lane serialisation
    static inline uint32_t mem_lane() { return __zluda_metal_thread_index_in_simdgroup(); }
    static inline uint32_t mem_active() { return (uint32_t)air_simd_ballot(true); }
#define MEM_SERIAL(...)                                     \
    {                                                       \
        uint32_t mem_m_ = mem_active();                     \
        uint32_t mem_me_ = mem_lane();                      \
        while (mem_m_)                                      \
        {                                                   \
            if ((uint32_t)__builtin_ctz(mem_m_) == mem_me_) \
            {                                               \
                __VA_ARGS__                                 \
            }                                               \
            mem_m_ &= mem_m_ - 1;                           \
        }                                                   \
    }

    // ---------------------------------------------------------------- mbarrier objects
    // The model keeps mbarrier objects outside memory (mb(phase, pending, expected, tx, layout) plus the
    // conditional phase and the payload reports of .layout::v1): mbarrier.init does not write the 8 bytes. The
    // helpers keep them in a table in threadgroup memory keyed by the object's threadgroup address (+1, 0 = free),
    // assumed zero at launch.
#define MB_SLOTS 16
#define MB_TIMEOUT_ROUNDS 24
    struct MbSlot
    {
        uint32_t key;
        uint32_t phase;
        int32_t pending;
        int32_t expected;
        int32_t tx;
        uint32_t layout;
        uint32_t cphase; // the conditional phase
        uint32_t report; // bit 0: report of the current primary phase, bit 1: of the last completed one
    };
    static SHARED_SPACE uint32_t mb_words[MB_SLOTS * 8];
    static SHARED_SPACE uint32_t mb_epoch; // counts mbarrier state changes (progress, for try_wait's time limit)

    static inline uint32_t mb_key(SHARED_SPACE uint64_t *a) { return (uint32_t)(uintptr_t)a + 1u; }
    static inline volatile SHARED_SPACE MbSlot *mb_find(uint32_t key)
    {
        for (int i = 0; i < MB_SLOTS; i++)
            if (((volatile SHARED_SPACE MbSlot *)&mb_words[8 * i])->key == key)
                return (volatile SHARED_SPACE MbSlot *)&mb_words[8 * i];
        return nullptr;
    }
    static inline void mb_bump() { *(volatile SHARED_SPACE uint32_t *)&mb_epoch = *(volatile SHARED_SPACE uint32_t *)&mb_epoch + 1u; }
    // mbCheck: the phase completes when the pending count and the tx-count are both zero
    static inline void mb_check(volatile SHARED_SPACE MbSlot *s)
    {
        if (s->pending == 0 && s->tx == 0)
        {
            s->phase = s->phase + 1u;
            s->pending = s->expected;
            uint32_t r = s->report;
            if ((r & 1u) == 0)
                s->cphase = s->cphase + 1u;
            s->report = (r & 1u) << 1;
        }
    }

    void FUNC(mbar_init)(SHARED_SPACE uint64_t *a, uint32_t count, uint32_t layout)
    {
        MEM_SERIAL({
            uint32_t key = mb_key(a);
            volatile SHARED_SPACE MbSlot *s = mb_find(key);
            if (!s)
                s = mb_find(0);
            if (s)
            {
                s->key = key;
                s->phase = 0;
                s->pending = (int32_t)count;
                s->expected = (int32_t)count;
                s->tx = 0;
                s->layout = layout;
                s->cphase = 0;
                s->report = 0;
            }
            mb_bump();
        })
    }
    void FUNC(mbar_inval)(SHARED_SPACE uint64_t *a)
    {
        MEM_SERIAL({
            volatile SHARED_SPACE MbSlot *s = mb_find(mb_key(a));
            if (s)
                s->key = 0;
            mb_bump();
        })
    }
    // arrive / arrive_drop with an arrival count and a tx-count increment (expect_tx): returns the state before
    // the operation, phase | pending << 32 (mbState)
    uint64_t FUNC(mbar_arrive)(SHARED_SPACE uint64_t *a, uint32_t count, uint32_t tx, uint32_t drop)
    {
        uint64_t st = 0;
        MEM_SERIAL({
            volatile SHARED_SPACE MbSlot *s = mb_find(mb_key(a));
            if (s)
            {
                st = (uint64_t)s->phase | ((uint64_t)(uint32_t)s->pending << 32);
                s->tx = s->tx + (int32_t)tx;
                s->pending = s->pending - (int32_t)count;
                if (drop)
                    s->expected = s->expected - (int32_t)count;
                mb_check(s);
            }
            mb_bump();
        })
        return st;
    }
    void FUNC(mbar_arrive_nd)(SHARED_SPACE uint64_t *a, uint32_t count, uint32_t tx, uint32_t drop)
    {
        (void)__zluda_ptx_impl_mbar_arrive(a, count, tx, drop);
    }
    // expect_tx (+n) / complete_tx (-n) without an arrive
    void FUNC(mbar_tx)(SHARED_SPACE uint64_t *a, uint32_t n, uint32_t complete)
    {
        int32_t delta = complete ? -(int32_t)n : (int32_t)n;
        MEM_SERIAL({
            volatile SHARED_SPACE MbSlot *s = mb_find(mb_key(a));
            if (s)
            {
                s->tx = s->tx + delta;
                mb_check(s);
            }
            mb_bump();
        })
    }
    // a report-on (.layout::v1 objects only)
    static inline void mb_report_on(SHARED_SPACE uint64_t *a)
    {
        MEM_SERIAL({
            volatile SHARED_SPACE MbSlot *s = mb_find(mb_key(a));
            if (s && s->layout == 1)
                s->report = s->report | 1u;
        })
    }

    // ctrl: bit 0 .parity, bit 1 try_wait (potentially blocking), bit 2 .phase_type::conditional
    static inline bool mb_done(SHARED_SPACE uint64_t *a, uint64_t v, uint32_t ctrl, uint32_t *rp)
    {
        volatile SHARED_SPACE MbSlot *s = mb_find(mb_key(a));
        *rp = 0;
        if (!s)
            return false;
        bool d;
        if (ctrl & 4)
            d = (s->cphase & 1u) != (uint32_t)(v & 1);
        else if (ctrl & 1)
            d = (s->phase & 1u) != (uint32_t)(v & 1);
        else
            d = (int32_t)(s->phase - (uint32_t)v) > 0;
        if (d)
            *rp = (s->report >> 1) & 1u;
        return d;
    }
    typedef uint32_t MbWait __attribute__((ext_vector_type(2)));
    // waitComplete and reportPredicate
    MbWait FUNC(mbar_wait2)(SHARED_SPACE uint64_t *a, uint64_t v, uint32_t ctrl)
    {
        uint32_t rp = 0;
        bool done = mb_done(a, v, ctrl, &rp);
        uint32_t rounds = 0;
        uint32_t epoch = *(volatile SHARED_SPACE uint32_t *)&mb_epoch;
        // test_wait yields once (a later observation is a legal schedule); try_wait until the time limit
        uint32_t limit = (ctrl & 2) ? MB_TIMEOUT_ROUNDS : 1;
        while ((uint32_t)air_simd_ballot(!done) != 0 && rounds < limit)
        {
            air_wg_barrier(3, 1);
            if (!done)
                done = mb_done(a, v, ctrl, &rp);
            uint32_t e = *(volatile SHARED_SPACE uint32_t *)&mb_epoch;
            if (e != epoch && (ctrl & 2))
                rounds = 0, epoch = e;
            else
                rounds++;
        }
        return MbWait{(uint32_t)done, done ? rp : 0u};
    }
    bool FUNC(mbar_wait)(SHARED_SPACE uint64_t *a, uint64_t v, uint32_t ctrl)
    {
        return __zluda_ptx_impl_mbar_wait2(a, v, ctrl).x != 0;
    }
    MbWait FUNC(mbar_wait2p)(SHARED_SPACE uint64_t *a, uint32_t v, uint32_t ctrl) { return __zluda_ptx_impl_mbar_wait2(a, v, ctrl); }
    bool FUNC(mbar_waitp)(SHARED_SPACE uint64_t *a, uint32_t v, uint32_t ctrl) { return __zluda_ptx_impl_mbar_wait(a, v, ctrl); }
    uint32_t FUNC(mbar_pending_count)(uint64_t st) { return (uint32_t)(st >> 32) & 0xFFFFFu; }
    bool FUNC(mbar_check_layout)(SHARED_SPACE uint64_t *a, uint32_t layout)
    {
        volatile SHARED_SPACE MbSlot *s = mb_find(mb_key(a));
        return s && s->layout == layout;
    }

    // generic-address forms: the address must be in the .shared window (resolved statically by the AIR lowering)
#define MB_G(p) ((SHARED_SPACE uint64_t *)(p))
    void FUNC(mbar_init_g)(uint64_t *a, uint32_t c, uint32_t l) { __zluda_ptx_impl_mbar_init(MB_G(a), c, l); }
    void FUNC(mbar_inval_g)(uint64_t *a) { __zluda_ptx_impl_mbar_inval(MB_G(a)); }
    uint64_t FUNC(mbar_arrive_g)(uint64_t *a, uint32_t c, uint32_t t, uint32_t d) { return __zluda_ptx_impl_mbar_arrive(MB_G(a), c, t, d); }
    void FUNC(mbar_arrive_nd_g)(uint64_t *a, uint32_t c, uint32_t t, uint32_t d) { __zluda_ptx_impl_mbar_arrive_nd(MB_G(a), c, t, d); }
    void FUNC(mbar_tx_g)(uint64_t *a, uint32_t n, uint32_t c) { __zluda_ptx_impl_mbar_tx(MB_G(a), n, c); }
    bool FUNC(mbar_wait_g)(uint64_t *a, uint64_t v, uint32_t c) { return __zluda_ptx_impl_mbar_wait(MB_G(a), v, c); }
    bool FUNC(mbar_waitp_g)(uint64_t *a, uint32_t v, uint32_t c) { return __zluda_ptx_impl_mbar_wait(MB_G(a), v, c); }
    MbWait FUNC(mbar_wait2_g)(uint64_t *a, uint64_t v, uint32_t c) { return __zluda_ptx_impl_mbar_wait2(MB_G(a), v, c); }
    MbWait FUNC(mbar_wait2p_g)(uint64_t *a, uint32_t v, uint32_t c) { return __zluda_ptx_impl_mbar_wait2(MB_G(a), v, c); }
    bool FUNC(mbar_check_layout_g)(uint64_t *a, uint32_t l) { return __zluda_ptx_impl_mbar_check_layout(MB_G(a), l); }

    void FUNC(mem_nop)() {}

    // ---------------------------------------------------------------- cp.async (completed at issue)
    // cp-size bytes at dst: the first src-size bytes from src, then zeros (cpSrcN: min(src-size, cp-size))
    void FUNC(cp_async)(SHARED_SPACE uint8_t *d, __attribute__((address_space(1))) uint8_t *s, uint32_t n, uint32_t k)
    {
        if (k > n)
            k = n;
        for (uint32_t i = 0; i < n; i++)
            d[i] = i < k ? s[i] : 0;
    }
    void FUNC(cp_async_ig)(SHARED_SPACE uint8_t *d, __attribute__((address_space(1))) uint8_t *s, uint32_t n, bool ig)
    {
        __zluda_ptx_impl_cp_async(d, s, n, ig ? 0u : n);
    }

    // ---------------------------------------------------------------- isspacep
    // decided from the pointer's provenance by the AIR lowering (resolveIsSpacep)
    bool __zluda_metal_isspacep(void *p, uint32_t kind);
    bool FUNC(isspacep)(uint64_t a, uint32_t kind) { return __zluda_metal_isspacep((void *)a, kind); }

    // ---------------------------------------------------------------- global byte access, element arithmetic
#define GLOBAL_SPACE __attribute__((address_space(1)))
    static inline uint64_t g_rd(uint64_t a, uint32_t n)
    {
        volatile GLOBAL_SPACE uint8_t *p = (volatile GLOBAL_SPACE uint8_t *)a;
        uint64_t v = 0;
        for (uint32_t i = 0; i < n; i++)
            v |= (uint64_t)p[i] << (8 * i);
        return v;
    }
    static inline void g_wr(uint64_t a, uint32_t n, uint64_t v)
    {
        volatile GLOBAL_SPACE uint8_t *p = (volatile GLOBAL_SPACE uint8_t *)a;
        for (uint32_t i = 0; i < n; i++)
            p[i] = (uint8_t)(v >> (8 * i));
    }
    static inline uint64_t s_rd(volatile SHARED_SPACE uint8_t *p, uint32_t n)
    {
        uint64_t v = 0;
        for (uint32_t i = 0; i < n; i++)
            v |= (uint64_t)p[i] << (8 * i);
        return v;
    }
    static inline void s_wr(volatile SHARED_SPACE uint8_t *p, uint32_t n, uint64_t v)
    {
        for (uint32_t i = 0; i < n; i++)
            p[i] = (uint8_t)(v >> (8 * i));
    }

    // float formats: 0 f32, 1 f16, 2 bf16 (exponent bits, mantissa bits)
    static inline uint32_t ff_e(int f) { return f == 1 ? 5 : 8; }
    static inline uint32_t ff_m(int f) { return f == 0 ? 23 : f == 1 ? 10 : 7; }
    static inline bool ff_nan(uint64_t b, int f)
    {
        uint32_t e = ff_e(f), m = ff_m(f);
        return ((b >> m) & ((1u << e) - 1)) == ((1u << e) - 1) && (b & ((1ull << m) - 1)) != 0;
    }
    // exact conversion to double (integer arithmetic: no flushing)
    static inline double ff_to_d(uint64_t b, int f)
    {
        uint32_t e = ff_e(f), m = ff_m(f);
        int bias = (1 << (e - 1)) - 1;
        bool s = (b >> (e + m)) & 1;
        int32_t ex = (int32_t)((b >> m) & ((1u << e) - 1));
        uint64_t mant = b & ((1ull << m) - 1);
        uint64_t r;
        if (ex == (1 << e) - 1)
            r = 0x7FF0000000000000ull | (mant << (52 - m));
        else if (ex == 0)
        {
            if (mant == 0)
                r = 0;
            else
            {
                int sh = 0;
                while (!(mant & (1ull << m)))
                    mant <<= 1, sh++;
                mant &= (1ull << m) - 1;
                r = ((uint64_t)(1 - bias - sh + 1023) << 52) | (mant << (52 - m));
            }
        }
        else
            r = ((uint64_t)(ex - bias + 1023) << 52) | (mant << (52 - m));
        return __builtin_bit_cast(double, r | ((uint64_t)s << 63));
    }
    // round-to-nearest-even conversion from double (integer arithmetic), NaN -> canonical NaN
    static inline uint64_t d_to_ff(double d, int f)
    {
        uint32_t e = ff_e(f), m = ff_m(f);
        int bias = (1 << (e - 1)) - 1;
        uint64_t u = __builtin_bit_cast(uint64_t, d);
        uint64_t s = (u >> 63) << (e + m);
        int32_t ex = (int32_t)((u >> 52) & 0x7FF);
        uint64_t mant = u & ((1ull << 52) - 1);
        uint64_t emax = (1ull << e) - 1;
        if (ex == 0x7FF)
            return mant ? ((1ull << (e + m)) - 1) : (s | (emax << m));
        if (ex == 0 && mant == 0)
            return s;
        int32_t E = ex - 1023 + bias; // biased target exponent
        uint64_t sig = mant | (1ull << 52); // 53 bits
        int shift = 52 - (int)m;          // bits to drop
        if (E <= 0)
        {
            shift += 1 - E;
            E = 0;
        }
        uint64_t q, rem, half;
        if (shift >= 64)
            q = 0, rem = sig, half = (shift == 64) ? (1ull << 63) : ~0ull;
        else
        {
            q = sig >> shift;
            rem = sig & ((1ull << shift) - 1);
            half = 1ull << (shift - 1);
        }
        if (shift < 64 && (rem > half || (rem == half && (q & 1))))
            q++;
        // q holds the significand with the implicit bit at position m (normal) or the subnormal value
        uint64_t r;
        if (E == 0)
            r = q; // a carry into bit m gives the smallest normal number, as encoded
        else
        {
            if (q >> (m + 1))
            {
                q >>= 1;
                E++;
            }
            if ((uint64_t)E >= emax)
                return s | (emax << m);
            r = ((uint64_t)E << m) | (q & ((1ull << m) - 1));
        }
        return s | r;
    }
    static inline uint64_t ff_canon(int f) { return (1ull << (ff_e(f) + ff_m(f))) - 1; }
    static inline uint64_t ff_add(uint64_t a, uint64_t b, int f)
    {
        if (ff_nan(a, f) || ff_nan(b, f))
            return ff_canon(f);
        return d_to_ff(ff_to_d(a, f) + ff_to_d(b, f), f);
    }
    // PTX min / max: one NaN gives the other operand, two give the canonical NaN; -0 < +0
    static inline uint64_t ff_minmax(uint64_t a, uint64_t b, int f, bool isMin)
    {
        bool na = ff_nan(a, f), nb = ff_nan(b, f);
        if (na && nb)
            return ff_canon(f);
        if (na)
            return b;
        if (nb)
            return a;
        double x = ff_to_d(a, f), y = ff_to_d(b, f);
        if (x < y)
            return isMin ? a : b;
        if (y < x)
            return isMin ? b : a;
        return isMin ? (a | b) : (a & b);
    }
    // atomNew of the reductions on an element of type code c (tensormap.replace.elemtype numbering:
    // 0 u8, 1 u16, 2 u32, 3 s32, 4 u64, 5 s64, 6 f16, 7 f32, 8 f32.ftz, 9 f64, 10 bf16, 11 tf32, 12 tf32.ftz);
    // op: 1 add, 2 min, 3 max, 4 inc, 5 dec, 6 and, 7 or, 8 xor
    static inline uint64_t red_elem(uint32_t op, uint32_t c, uint64_t o, uint64_t b, uint32_t eb)
    {
        uint64_t msk = eb >= 8 ? ~0ull : ((1ull << (8 * eb)) - 1);
        int f = (c == 6) ? 1 : (c == 10) ? 2 : (c == 7 || c == 8 || c == 11 || c == 12) ? 0 : -1;
        bool sgn = c == 3 || c == 5;
        if (op == 6)
            return o & b;
        if (op == 7)
            return o | b;
        if (op == 8)
            return (o ^ b) & msk;
        if (c == 9)
        {
            double x = __builtin_bit_cast(double, o), y = __builtin_bit_cast(double, b);
            if (op == 1)
                return __builtin_bit_cast(uint64_t, x + y);
            if (op == 2 || op == 3)
            {
                bool nx = x != x, ny = y != y;
                if (nx && ny)
                    return o | b | (1ull << 51);
                if (nx)
                    return b;
                if (ny)
                    return o;
                if (x < y)
                    return op == 2 ? o : b;
                if (y < x)
                    return op == 2 ? b : o;
                return op == 2 ? (o | b) : (o & b);
            }
            return o;
        }
        if (f >= 0)
        {
            if (op == 1)
                return ff_add(o, b, f);
            if (op == 2 || op == 3)
                return ff_minmax(o, b, f, op == 2);
            return o;
        }
        int64_t so = (int64_t)(o << (64 - 8 * eb)) >> (64 - 8 * eb);
        int64_t sb = (int64_t)(b << (64 - 8 * eb)) >> (64 - 8 * eb);
        switch (op)
        {
        case 1: return (o + b) & msk;
        case 2: return sgn ? (so <= sb ? o : b) : (o <= b ? o : b);
        case 3: return sgn ? (so >= sb ? o : b) : (o >= b ? o : b);
        case 4: return o >= b ? 0 : ((o + 1) & msk);
        case 5: return (o == 0 || o > b) ? b : ((o - 1) & msk);
        default: return o;
        }
    }

    // ---------------------------------------------------------------- tensor maps (sail/ptx/ptx_tma.sail)
    typedef int32_t I32x8 __attribute__((ext_vector_type(8)));
    struct Tm
    {
        uint64_t w[16];
    };
    static inline void tm_read(uint64_t a, Tm *t)
    {
        for (int i = 0; i < 16; i++)
            t->w[i] = g_rd(a + 8 * i, 8);
    }
    static inline uint32_t tm_rank(const Tm *t) { return (uint32_t)(t->w[2] & 255); }
    // (selects rather than dynamically indexed stack accesses)
    static inline int64_t tm_dim(const Tm *t, int i)
    {
        return (int64_t)(i == 0 ? t->w[4] : i == 1 ? t->w[5] : i == 2 ? t->w[6] : i == 3 ? t->w[7] : t->w[8]);
    }
    static inline int64_t tm_stride(const Tm *t, int i)
    {
        return (int64_t)(i <= 1 ? t->w[9] : i == 2 ? t->w[10] : i == 3 ? t->w[11] : t->w[12]);
    }
    static inline int64_t tm_box(const Tm *t, int i) { return i < 4 ? (int64_t)((t->w[13] >> (16 * i)) & 65535) : (int64_t)(t->w[14] & 65535); }
    static inline int64_t tm_est(const Tm *t, int i) { return (int64_t)((t->w[15] >> (8 * i)) & 255); }
    static inline uint32_t tm_swz(const Tm *t) { return (uint32_t)((t->w[14] >> 16) & 255); }
    static inline uint32_t tm_atom(const Tm *t) { return (uint32_t)((t->w[14] >> 48) & 255); }
    static inline uint32_t tm_fillmode(const Tm *t) { return (uint32_t)((t->w[14] >> 24) & 255); }
    static inline uint32_t tm_etc(const Tm *t) { return (uint32_t)(t->w[1] & 255); }
    static inline uint32_t tm_eb(const Tm *t)
    {
        switch (tm_etc(t))
        {
        case 0: return 1;
        case 1: case 6: case 10: return 2;
        case 2: case 3: case 7: case 8: case 11: case 12: return 4;
        case 4: case 5: case 9: return 8;
        default: return 1;
        }
    }
    static inline uint32_t tm_effmode(const Tm *t)
    {
        uint32_t s = tm_swz(t);
        if (s == 1 || s == 4)
            return 3;
        if (s == 2)
            return 2;
        if (s == 3)
        {
            switch (tm_atom(t))
            {
            case 0: return 1;
            case 1: return 4;
            case 2: return 5;
            case 3: return 6;
            default: return 0;
            }
        }
        return 0;
    }
    static inline uint64_t tm_fill(const Tm *t)
    {
        if (tm_fillmode(t) == 0)
            return 0;
        if (tm_eb(t) == 4)
            return 0x7FC00000u;
        if (tm_etc(t) == 6)
            return 0x7E00u;
        if (tm_etc(t) == 10)
            return 0x7FC0u;
        if (tm_etc(t) == 9)
            return 0x7FF8000000000000ull;
        return 0;
    }
    // swizzleB(z, a, 0) of ptx_wg.sail
    static inline uint32_t tm_swizzle(uint32_t z, uint32_t a)
    {
        switch (z)
        {
        case 1: return a ^ ((a & (7u << 7)) >> 3);
        case 2: return a ^ ((a & (3u << 7)) >> 3);
        case 3: return a ^ ((a & (1u << 7)) >> 3);
        case 4: return a ^ (((a >> 7) & 3u) << 5);
        case 5: return a ^ (((a >> 7) & 3u) << 5) ^ (((a >> 7) & 1u) << 3);
        case 6: return a ^ (((a >> 7) & 1u) << 6);
        default: return a;
        }
    }
    static inline int64_t tdiv(int64_t a, int64_t b) { return b == 0 ? 0 : a / b; }
    static inline int64_t tmod(int64_t a, int64_t b) { return b == 0 ? a : a % b; }
    static inline int64_t fdiv(int64_t a, int64_t b)
    {
        if (b == 0)
            return 0;
        int64_t q = a / b;
        if ((a % b != 0) && ((a < 0) != (b < 0)))
            q--;
        return q;
    }
    static inline int64_t tm_n(const Tm *t, int i)
    {
        int64_t e = tm_est(t, i);
        if (e < 1)
            e = 1;
        return tdiv(tm_box(t, i) + e - 1, e);
    }
    static inline int64_t tm_count(const Tm *t, bool g)
    {
        if (g)
            return 4 * tm_box(t, 0);
        int64_t a = 1;
        _Pragma("clang loop unroll(full)") for (uint32_t i = 0; i < 5; i++)
            if (i < tm_rank(t))
                a *= tm_n(t, i);
        return a;
    }
    // the global address of element e of the box at coordinates cs, -1 when out of bounds
    static inline int64_t tm_addr_t(const Tm *t, I32x8 cs, int64_t e)
    {
        int64_t dv = 1, acc = 0;
        _Pragma("clang loop unroll(full)") for (uint32_t i = 0; i < 5; i++)
        {
            if (i >= tm_rank(t))
                break;
            int64_t g = (int64_t)cs[i] + tmod(tdiv(e, dv), tm_n(t, i)) * tm_est(t, i);
            if (g < 0 || g >= tm_dim(t, i))
                return -1;
            acc += i == 0 ? g * tm_eb(t) : g * tm_stride(t, i);
            dv *= tm_n(t, i);
        }
        return (int64_t)t->w[3] + acc;
    }
    static inline int64_t tm_addr_g(const Tm *t, I32x8 cs, int64_t e)
    {
        int64_t b0 = tm_box(t, 0);
        int64_t c = (int64_t)cs[0] + tmod(e, b0);
        int64_t q = tdiv(e, b0);
        int64_t r = q == 0 ? cs[1] : q == 1 ? cs[2] : q == 2 ? cs[3] : cs[4];
        if (c < 0 || c >= tm_dim(t, 0) || r < 0 || r >= tm_dim(t, 1))
            return -1;
        return (int64_t)t->w[3] + c * tm_eb(t) + r * tm_stride(t, 1);
    }
    // im2col
    static inline int64_t tm_lo(const Tm *t, int k) { return (int64_t)((t->w[1] >> (8 + 16 * k)) & 65535) - 32768; }
    static inline int64_t tm_up(const Tm *t, int k) { return (int64_t)((t->w[2] >> (8 + 16 * k)) & 65535) - 32768; }
    static inline int64_t tm_nk(const Tm *t, int k)
    {
        int64_t v = tdiv(tm_dim(t, k) - 1 + tm_up(t, k - 1) - tm_lo(t, k - 1), tm_est(t, k)) + 1;
        return v < 1 ? 1 : v;
    }
    static inline int64_t tm_start(const Tm *t, I32x8 cs)
    {
        int64_t mu = 1, acc = 0;
        int r = (int)tm_rank(t);
        _Pragma("clang loop unroll(full)") for (int k = 1; k <= 3; k++)
        {
            if (k > r - 2)
                break;
            acc += fdiv((int64_t)cs[k] - tm_lo(t, k - 1), tm_est(t, k)) * mu;
            mu *= tm_nk(t, k);
        }
        return acc + (int64_t)cs[r - 1] * mu;
    }
    static inline int64_t tm_pix(const Tm *t, I32x8 inf, int md)
    {
        if (md == 1)
            return tm_box(t, 1) + inf[0];
        if (md == 2)
            return 128 + 4 * (int64_t)inf[0];
        return tm_box(t, 1);
    }
    static inline int64_t tm_addr_w(const Tm *t, I32x8 cs, int64_t sh, int64_t acc)
    {
        int r = (int)tm_rank(t);
        _Pragma("clang loop unroll(full)") for (int k = 1; k <= 3; k++)
        {
            if (k > r - 2)
                break;
            int64_t v = (int64_t)cs[k] + (k == 1 ? sh : 0);
            if (v < 0 || v >= tm_dim(t, k))
                return -1;
            acc += v * tm_stride(t, k);
        }
        int64_t n = cs[r - 1];
        if (n < 0 || n >= tm_dim(t, r - 1))
            return -1;
        return acc + n * tm_stride(t, r - 1);
    }
    static inline int64_t tm_addr_i(const Tm *t, I32x8 cs, I32x8 inf, int md, int64_t e)
    {
        int64_t c0 = (int64_t)cs[0] + tmod(e, tm_box(t, 0)) * tm_est(t, 0);
        if (c0 < 0 || c0 >= tm_dim(t, 0))
            return -1;
        int64_t p = tdiv(e, tm_box(t, 0));
        int64_t acc0 = (int64_t)t->w[3] + c0 * tm_eb(t);
        if (md != 0)
        {
            int64_t halo = inf[0];
            int64_t q = md == 2 ? 32 * tdiv(p, 32 + halo) + tmod(p, 32 + halo) : p;
            int64_t woff = md == 3 ? 0 : inf[1];
            return tm_addr_w(t, cs, q * tm_est(t, 1) + woff, acc0);
        }
        int64_t idx = tm_start(t, cs) + p;
        int64_t acc = acc0;
        int r = (int)tm_rank(t);
        _Pragma("clang loop unroll(full)") for (int k = 1; k <= 3; k++)
        {
            if (k > r - 2)
                break;
            int64_t pos = tm_lo(t, k - 1) + tmod(idx, tm_nk(t, k)) * tm_est(t, k) + inf[k - 1];
            if (pos < 0 || pos >= tm_dim(t, k))
                return -1;
            acc += pos * tm_stride(t, k);
            idx = tdiv(idx, tm_nk(t, k));
        }
        if (idx < 0 || idx >= tm_dim(t, r - 1))
            return -1;
        return acc + idx * tm_stride(t, r - 1);
    }

    // report-on patterns (.mbarrier::report::validity::*): 1 per_element::ff, 2 per_16bytes::80000000, 3 ::8000,
    // 4 ::80, 5 ::8
    static inline bool rep_chunk(uint32_t rep, uint64_t a)
    {
        uint64_t c = a - (a % 16);
        switch (rep)
        {
        case 2: return (g_rd(c, 4) & 0xFFFFFFFFull) == 0x80000000ull;
        case 3: return (g_rd(c, 2) & 0xFFFF) == 0x8000;
        case 4: return (g_rd(c, 1) & 0xFF) == 0x80;
        case 5: return (g_rd(c, 1) & 0xF) == 0x8;
        default: return false;
        }
    }

    // cp.async.bulk.tensor loads, completed at issue. ctrl: bits 0-1 mode (0 tile, 1 gather4, 2 im2col), 2-3 im2col
    // variant (0 im2col, 1 ::w, 2 ::w::128, 3 _no_offs::w), 4-7 report pattern
    static inline void tma_load_impl(SHARED_SPACE uint8_t *dst, Tm *t, I32x8 cs, SHARED_SPACE uint64_t *mb, I32x8 inf,
                                     uint32_t ctrl)
    {
        uint32_t mode = ctrl & 3, imd = (ctrl >> 2) & 3, rep = (ctrl >> 4) & 15;
        bool g = mode == 1, i2c = mode == 2;
        uint32_t eb = tm_eb(t);
        uint32_t md = tm_effmode(t);
        uint64_t fill = tm_fill(t);
        uint32_t A = (uint32_t)(uintptr_t)dst;
        if (tm_etc(t) >= 13 && !i2c)
        {
            // packed sub-byte types: groups of 16 elements, GB global bytes into SB shared bytes (zero padding)
            uint32_t etc = tm_etc(t);
            int64_t gb = etc == 15 ? 12 : 8, sb = etc == 13 ? 8 : 16, nb = etc == 15 ? 6 : 4;
            int64_t gpr = tm_box(t, 0) / 16, vol = 1;
            _Pragma("clang loop unroll(full)") for (uint32_t i = 1; i < 5; i++) if (i < tm_rank(t)) vol *= tm_n(t, i);
            int64_t groups = gpr * vol;
            for (int64_t k = 0; k * 16 < groups * sb; k++)
            {
                uint32_t a = A + 16 * (uint32_t)k;
                uint32_t a2 = md == 5 ? tm_swizzle(4, a) : tm_swizzle(md, a);
                uint32_t flip = (md == 5 && ((a >> 7) & 1)) ? 8 : 0;
                for (int64_t h = 0; h < 16 / sb; h++)
                {
                    int64_t q0 = sb == 16 ? k : 2 * k + h;
                    int64_t ga = -1;
                    if (q0 < groups && gpr > 0)
                    {
                        int64_t c0 = (int64_t)cs[0] + (q0 % gpr) * 16;
                        if (c0 >= 0 && c0 < tm_dim(t, 0))
                        {
                            int64_t q = q0 / gpr, acc = (int64_t)t->w[3] + c0 * nb / 8;
                            bool bad = false;
                            _Pragma("clang loop unroll(full)") for (uint32_t i = 1; i < 5; i++)
                            {
                                if (i >= tm_rank(t) || bad)
                                    continue;
                                int64_t v = (int64_t)cs[i] + tmod(q, tm_n(t, i)) * tm_est(t, i);
                                if (v < 0 || v >= tm_dim(t, i))
                                    bad = true;
                                else
                                {
                                    acc += v * tm_stride(t, i);
                                    q = tdiv(q, tm_n(t, i));
                                }
                            }
                            if (!bad)
                                ga = acc;
                        }
                    }
                    for (int64_t j = 0; j < sb; j++)
                    {
                        uint8_t v = (ga >= 0 && j < gb) ? ((volatile GLOBAL_SPACE uint8_t *)ga)[j] : 0;
                        uint32_t within = ((uint32_t)(h * 8 + j)) ^ flip;
                        ((volatile SHARED_SPACE uint8_t *)(dst + (int32_t)(a2 - A)))[within] = v;
                    }
                }
            }
            __zluda_ptx_impl_mbar_tx(mb, (uint32_t)(groups * sb), 1);
            return;
        }
        int64_t cnt = i2c ? tm_pix(t, inf, (int)imd) * tm_box(t, 0) : tm_count(t, g);
        int64_t per = 16 / eb;
        bool hit = false;
        for (int64_t k = 0; k * 16 < cnt * (int64_t)eb; k++)
        {
            uint32_t a = A + 16 * (uint32_t)k;
            uint32_t a2 = md == 5 ? tm_swizzle(4, a) : tm_swizzle(md, a);
            uint32_t flip = (md == 5 && ((a >> 7) & 1)) ? 8 : 0;
            for (int64_t j = 0; j < per; j++)
            {
                int64_t e = k * per + j;
                int64_t ga = i2c ? tm_addr_i(t, cs, inf, (int)imd, e) : g ? tm_addr_g(t, cs, e) : tm_addr_t(t, cs, e);
                uint64_t v = ga < 0 ? fill : g_rd((uint64_t)ga, eb);
                if (ga >= 0 && e < cnt && rep && !hit)
                {
                    if (rep == 1)
                    {
                        for (uint32_t q = 0; q < eb; q++)
                            if (((v >> (8 * q)) & 255) == 255)
                                hit = true;
                    }
                    else
                        hit = rep_chunk(rep, (uint64_t)ga);
                }
                uint32_t within = ((uint32_t)j * eb) ^ flip;
                s_wr((volatile SHARED_SPACE uint8_t *)(dst + (int32_t)(a2 - A) + within), eb, v);
            }
        }
        uint32_t n = (uint32_t)(cnt * eb);
        if (hit)
            mb_report_on(mb);
        __zluda_ptx_impl_mbar_tx(mb, n, 1);
    }
    void FUNC(tma_load)(SHARED_SPACE uint8_t *dst, uint64_t tmap, I32x8 cs, SHARED_SPACE uint64_t *mb, I32x8 inf, uint32_t ctrl)
    {
        MEM_SERIAL({
            Tm t;
            tm_read(tmap, &t);
            tma_load_impl(dst, &t, cs, mb, inf, ctrl);
        })
    }
    // .override::global_address
    void FUNC(tma_load_ga)(SHARED_SPACE uint8_t *dst, uint64_t tmap, uint64_t ga, I32x8 cs, SHARED_SPACE uint64_t *mb, uint32_t ctrl)
    {
        MEM_SERIAL({
            Tm t;
            tm_read(tmap, &t);
            t.w[3] = ga;
            I32x8 inf = {0, 0, 0, 0, 0, 0, 0, 0};
            tma_load_impl(dst, &t, cs, mb, inf, ctrl);
        })
    }

    // cp.async.bulk.tensor stores / cp.reduce.async.bulk.tensor (global <- shared::cta). ctrl: bits 0-3 op (0 store,
    // 1 add, 2 min, 3 max, 4 inc, 5 dec, 6 and, 7 or, 8 xor), bits 4-5 mode (0 tile, 1 scatter4, 2 im2col_no_offs,
    // 3 im2col_no_offs::w)
    void FUNC(tma_store)(uint64_t tmap, I32x8 cs, SHARED_SPACE uint8_t *src, uint32_t ctrl)
    {
        MEM_SERIAL({
            Tm t;
            tm_read(tmap, &t);
            uint32_t op = ctrl & 15, mode = (ctrl >> 4) & 3;
            uint32_t eb = tm_eb(&t), c = tm_etc(&t), md = tm_effmode(&t);
            uint32_t base = (uint32_t)(uintptr_t)src;
            I32x8 z = {0, 0, 0, 0, 0, 0, 0, 0};
            int64_t cnt, bad = 0;
            bool i2c = mode >= 2;
            if (i2c)
            {
                bool w = mode == 3;
                int64_t b1 = tm_box(&t, 1);
                bad = b1;
                if (w)
                {
                    int64_t q = 0;
                    while (q < b1 && (int64_t)cs[1] + q * tm_est(&t, 1) >= 0 && (int64_t)cs[1] + q * tm_est(&t, 1) < tm_dim(&t, 1))
                        q++;
                    bad = q;
                }
                cnt = b1 * tm_box(&t, 0);
            }
            else
                cnt = tm_count(&t, mode == 1);
            for (int64_t e = 0; e < cnt; e++)
            {
                int64_t ga;
                if (i2c)
                {
                    if (tdiv(e, tm_box(&t, 0)) >= bad)
                        continue;
                    ga = tm_addr_i(&t, cs, z, mode == 3 ? 3 : 0, e);
                }
                else
                    ga = mode == 1 ? tm_addr_g(&t, cs, e) : tm_addr_t(&t, cs, e);
                if (ga < 0)
                    continue;
                uint32_t sa = tm_swizzle(md, base + (uint32_t)e * eb);
                uint64_t v = s_rd((volatile SHARED_SPACE uint8_t *)(src + (int32_t)(sa - base)), eb);
                if (op != 0)
                    v = red_elem(op, c, g_rd((uint64_t)ga, eb), v, eb);
                g_wr((uint64_t)ga, eb, v);
            }
        })
    }

    // tensormap.replace.tile.field [addr], {ord,} value (tmReplace of ptx_tma.sail); field codes: 0 global_address,
    // 1 rank, 2 box_dim, 3 global_dim, 4 global_stride, 5 element_stride, 6 elemtype, 7 interleave_layout,
    // 8 swizzle_mode, 9 swizzle_atomicity, 10 fill_mode
    static inline void tm_replace(uint64_t *w, uint32_t f, uint32_t o, uint64_t v)
    {
        switch (f)
        {
        case 0: w[3] = v; break;
        case 1: w[2] = (w[2] & ~255ull) | ((v + 1) & 255); break;
        case 2:
            if (o < 4)
                w[13] = (w[13] & ~(65535ull << (16 * o))) | ((v & 65535) << (16 * o));
            else
                w[14] = (w[14] & ~65535ull) | (v & 65535);
            break;
        case 3: w[4 + (o & 7)] = v; break;
        case 4: w[9 + (o & 3)] = v; break;
        case 5: w[15] = (w[15] & ~(255ull << (8 * o))) | ((v & 255) << (8 * o)); break;
        case 6: w[1] = (w[1] & ~255ull) | (v & 255); break;
        case 7: w[14] = (w[14] & ~(255ull << 32)) | ((v & 255) << 32); break;
        case 8: w[14] = (w[14] & ~(255ull << 16)) | ((v & 255) << 16); break;
        case 9: w[14] = (w[14] & ~(255ull << 48)) | ((v & 255) << 48); break;
        case 10: w[14] = (w[14] & ~(255ull << 24)) | ((v & 255) << 24); break;
        default: break;
        }
    }
    void FUNC(tensormap_replace)(uint64_t a, uint32_t f, uint32_t o, uint64_t v)
    {
        MEM_SERIAL({
            uint32_t i = f == 0 ? 3 : f == 1 ? 2 : f == 2 ? (o < 4 ? 13 : 14) : f == 3 ? 4 + (o & 7) : f == 4 ? 9 + (o & 3)
                       : f == 5 ? 15 : f == 6 ? 1 : 14;
            uint64_t w[16];
            w[i] = g_rd(a + 8 * i, 8);
            tm_replace(w, f, o, v);
            g_wr(a + 8 * i, 8, w[i]);
        })
    }
    void FUNC(tensormap_replace_s)(SHARED_SPACE uint8_t *s, uint32_t f, uint32_t o, uint64_t v)
    {
        MEM_SERIAL({
            uint32_t i = f == 0 ? 3 : f == 1 ? 2 : f == 2 ? (o < 4 ? 13 : 14) : f == 3 ? 4 + (o & 7) : f == 4 ? 9 + (o & 3)
                       : f == 5 ? 15 : f == 6 ? 1 : 14;
            uint64_t w[16];
            w[i] = s_rd((volatile SHARED_SPACE uint8_t *)(s + 8 * i), 8);
            tm_replace(w, f, o, v);
            s_wr((volatile SHARED_SPACE uint8_t *)(s + 8 * i), 8, w[i]);
        })
    }
    // tensormap.cp_fenceproxy: 128 bytes shared -> global
    void FUNC(tensormap_cp)(uint64_t d, SHARED_SPACE uint8_t *s)
    {
        MEM_SERIAL({
            for (int i = 0; i < 16; i++)
                g_wr(d + 8 * i, 8, s_rd((volatile SHARED_SPACE uint8_t *)(s + 8 * i), 8));
        })
    }
    void FUNC(tensormap_replace32)(uint64_t a, uint32_t f, uint32_t o, uint32_t v) { __zluda_ptx_impl_tensormap_replace(a, f, o, v); }
    void FUNC(tensormap_replace32_s)(SHARED_SPACE uint8_t *s, uint32_t f, uint32_t o, uint32_t v) { __zluda_ptx_impl_tensormap_replace_s(s, f, o, v); }

    // ---------------------------------------------------------------- cp.async.bulk / cp.reduce.async.bulk
    // ctrl: bits 0-3 reduction op (red_elem), 4-7 element type code, 8-11 element bytes, 12-15 report pattern, 16 cp_mask
    void FUNC(bulk_g2s)(SHARED_SPACE uint8_t *d, GLOBAL_SPACE uint8_t *s, uint32_t n, SHARED_SPACE uint64_t *mb, uint32_t ctrl)
    {
        MEM_SERIAL({
            uint32_t rep = (ctrl >> 12) & 15;
            bool hit = false;
            uint64_t sa = (uint64_t)(uintptr_t)s;
            for (uint32_t i = 0; i < n; i++)
            {
                uint8_t v = ((volatile GLOBAL_SPACE uint8_t *)s)[i];
                ((volatile SHARED_SPACE uint8_t *)d)[i] = v;
                if (rep == 1 && v == 0xFF)
                    hit = true;
            }
            if (rep >= 2)
                for (uint32_t c = 0; c + 16 <= n; c += 16)
                    if (rep_chunk(rep, sa + c))
                        hit = true;
            if (hit)
                mb_report_on(mb);
            __zluda_ptx_impl_mbar_tx(mb, n, 1);
        })
    }
    static inline void bulk_red(volatile SHARED_SPACE uint8_t *dsh, uint64_t dg, bool glob, volatile SHARED_SPACE uint8_t *s,
                                uint32_t n, uint32_t ctrl, uint32_t mask)
    {
        uint32_t op = ctrl & 15, c = (ctrl >> 4) & 15, eb = (ctrl >> 8) & 15;
        if (op == 0)
        {
            for (uint32_t i = 0; i < n; i++)
            {
                if (!((mask >> (i % 16)) & 1))
                    continue;
                if (glob)
                    ((volatile GLOBAL_SPACE uint8_t *)dg)[i] = s[i];
                else
                    dsh[i] = s[i];
            }
            return;
        }
        for (uint32_t i = 0; i + eb <= n; i += eb)
        {
            uint64_t v = s_rd(s + i, eb);
            uint64_t o = glob ? g_rd(dg + i, eb) : s_rd(dsh + i, eb);
            uint64_t r = red_elem(op, c, o, v, eb);
            if (glob)
                g_wr(dg + i, eb, r);
            else
                s_wr(dsh + i, eb, r);
        }
    }
    void FUNC(bulk_s2s)(SHARED_SPACE uint8_t *d, SHARED_SPACE uint8_t *s, uint32_t n, SHARED_SPACE uint64_t *mb, uint32_t ctrl)
    {
        MEM_SERIAL({
            bulk_red((volatile SHARED_SPACE uint8_t *)d, 0, false, (volatile SHARED_SPACE uint8_t *)s, n, ctrl, 0xFFFF);
            __zluda_ptx_impl_mbar_tx(mb, n, 1);
        })
    }
    void FUNC(bulk_s2g)(GLOBAL_SPACE uint8_t *d, SHARED_SPACE uint8_t *s, uint32_t n, uint32_t mask, uint32_t ctrl)
    {
        MEM_SERIAL({
            bulk_red(nullptr, (uint64_t)(uintptr_t)d, true, (volatile SHARED_SPACE uint8_t *)s, n, ctrl,
                     (ctrl >> 16) & 1 ? mask : 0xFFFFu);
        })
    }
    void FUNC(mem_fence)() {}

    // .override::global_address.override::global_dim{_stride} (tmOv of ptx_tma.sail): v holds the n coordinates, the
    // n dims, then the n - 1 lower strides; stride = (lower + ((upper >> 4i) & 15) << 32) << 4
    static inline void tma_load_ov(SHARED_SPACE uint8_t *dst, uint64_t tmap, uint64_t ga, I32x8 v, SHARED_SPACE uint64_t *mb,
                                   uint16_t up, int n)
    {
        MEM_SERIAL({
            Tm t;
            tm_read(tmap, &t);
            t.w[3] = ga;
            I32x8 cs = {0, 0, 0, 0, 0, 0, 0, 0};
            _Pragma("clang loop unroll(full)") for (int i = 0; i < 3; i++) if (i < n)
            {
                cs[i] = v[i];
                t.w[4 + i] = (uint32_t)v[n + i];
            }
            _Pragma("clang loop unroll(full)") for (int i = 0; i < 2; i++) if (i < n - 1)
                t.w[9 + i] = ((uint64_t)(uint32_t)v[2 * n + i] + ((uint64_t)((up >> (4 * i)) & 15) << 32)) << 4;
            I32x8 inf = {0, 0, 0, 0, 0, 0, 0, 0};
            tma_load_impl(dst, &t, cs, mb, inf, 0);
        })
    }
    void FUNC(tma_load_ov1)(SHARED_SPACE uint8_t *d, uint64_t t, uint64_t g, I32x8 v, SHARED_SPACE uint64_t *mb, uint16_t up) { tma_load_ov(d, t, g, v, mb, up, 1); }
    void FUNC(tma_load_ov2)(SHARED_SPACE uint8_t *d, uint64_t t, uint64_t g, I32x8 v, SHARED_SPACE uint64_t *mb, uint16_t up) { tma_load_ov(d, t, g, v, mb, up, 2); }
    void FUNC(tma_load_ov3)(SHARED_SPACE uint8_t *d, uint64_t t, uint64_t g, I32x8 v, SHARED_SPACE uint64_t *mb, uint16_t up) { tma_load_ov(d, t, g, v, mb, up, 3); }

    // ---------------------------------------------------------------- atom / red outside the grammar
    // ctrl: bits 0-3 op (1 add, 2 min, 3 max, 4 inc, 5 dec, 6 and, 7 or, 8 xor, 9 exch, 10 cas), 4-7 type code
    // (red_elem codes, 13 f16x2, 14 bf16x2), 8 add.f32 flushing subnormal inputs and results (atom.global.add.f32)
    static inline uint64_t f32_ftz(uint64_t b) { return ((b >> 23) & 255) == 0 ? (b & 0x80000000u) : b; }
    static inline uint64_t rmw_elem(uint32_t ctrl, uint64_t o, uint64_t b, uint64_t c, uint32_t eb)
    {
        uint32_t op = ctrl & 15, code = (ctrl >> 4) & 15;
        if (op == 9)
            return b;
        if (op == 10)
            return o == b ? c : o;
        if (code == 13 || code == 14)
        {
            uint64_t lo = red_elem(op, code == 13 ? 6 : 10, o & 0xFFFF, b & 0xFFFF, 2);
            uint64_t hi = red_elem(op, code == 13 ? 6 : 10, (o >> 16) & 0xFFFF, (b >> 16) & 0xFFFF, 2);
            return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16);
        }
        if ((ctrl >> 8) & 1)
            return f32_ftz(red_elem(1, 7, f32_ftz(o), f32_ftz(b), 4));
        return red_elem(op, code, o, b, eb);
    }
#define ATOM_IMPL(BITS, T, SFX, SPACE)                                                                     \
    T FUNC(atom##BITS##_##SFX)(SPACE T * a, T b, T c, uint32_t ctrl)                                    \
    {                                                                                                     \
        T r = 0;                                                                                          \
        MEM_SERIAL({                                                                                      \
            volatile SPACE T *p = (volatile SPACE T *)a;                                                  \
            r = *p;                                                                                       \
            *p = (T)rmw_elem(ctrl, r, b, c, BITS / 8);                                                    \
        })                                                                                                \
        return r;                                                                                         \
    }
    ATOM_IMPL(16, uint16_t, g, GLOBAL_SPACE)
    ATOM_IMPL(32, uint32_t, g, GLOBAL_SPACE)
    ATOM_IMPL(64, uint64_t, g, GLOBAL_SPACE)
    ATOM_IMPL(16, uint16_t, s, SHARED_SPACE)
    ATOM_IMPL(32, uint32_t, s, SHARED_SPACE)
    ATOM_IMPL(64, uint64_t, s, SHARED_SPACE)
    ATOM_IMPL(16, uint16_t, f, )
    ATOM_IMPL(32, uint32_t, f, )
    ATOM_IMPL(64, uint64_t, f, )
    // vector forms: element-wise on 32-bit elements
#define ATOMV_IMPL(N, SFX, SPACE)                                                                           \
    typedef uint32_t U32x##N##_##SFX __attribute__((ext_vector_type(N)));                                 \
    U32x##N##_##SFX FUNC(atom_v##N##_##SFX)(SPACE uint32_t * a, U32x##N##_##SFX b, uint32_t ctrl)         \
    {                                                                                                       \
        U32x##N##_##SFX r;                                                                                  \
        MEM_SERIAL({                                                                                        \
            volatile SPACE uint32_t *p = (volatile SPACE uint32_t *)a;                                      \
            _Pragma("clang loop unroll(full)") for (int i = 0; i < N; i++)                                  \
            {                                                                                               \
                uint32_t o = p[i];                                                                          \
                r[i] = o;                                                                                   \
                p[i] = (uint32_t)rmw_elem(ctrl, o, b[i], 0, 4);                                             \
            }                                                                                               \
        })                                                                                                  \
        return r;                                                                                           \
    }
    ATOMV_IMPL(2, g, GLOBAL_SPACE)
    ATOMV_IMPL(4, g, GLOBAL_SPACE)
    ATOMV_IMPL(8, g, GLOBAL_SPACE)
    ATOMV_IMPL(2, s, SHARED_SPACE)
    ATOMV_IMPL(4, s, SHARED_SPACE)
    ATOMV_IMPL(8, s, SHARED_SPACE)
    ATOMV_IMPL(2, f, )
    ATOMV_IMPL(4, f, )
    ATOMV_IMPL(8, f, )
