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
#define MEM_SERIAL(BODY)                                    \
    {                                                       \
        uint32_t mem_m_ = mem_active();                     \
        uint32_t mem_me_ = mem_lane();                      \
        while (mem_m_)                                      \
        {                                                   \
            if ((uint32_t)__builtin_ctz(mem_m_) == mem_me_) \
            {                                               \
                BODY                                        \
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
