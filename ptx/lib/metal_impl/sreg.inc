// Included inside extern "C" of zluda_ptx_metal_impl.cpp: special registers beyond the core ones (ptx/src/pass/mod.rs
// PtxSpecialRegister).  Values follow PTX-formalization's launch setup (sail/ptx/ptx_sched.sail, ptx_int.sail).
    uint32_t FUNC(sreg_lanemask_eq)() { return 1u << __zluda_ptx_impl_sreg_laneid(); }
    uint32_t FUNC(sreg_lanemask_gt)() { return ~((2u << __zluda_ptx_impl_sreg_laneid()) - 1u); }

    // %clock / %clock64 / %globaltimer: monotonic counters ("tick" of the model, advancing by 25 per read), one counter
    // per threadgroup
    static SHARED_SPACE uint32_t __zluda_cluster_clock;
    static inline uint32_t clock_tick() { return __atomic_fetch_add(&__zluda_cluster_clock, 25u, __ATOMIC_RELAXED) + 25u; }
    uint64_t FUNC(sreg_clock64)() { return clock_tick(); }
    uint64_t FUNC(sreg_globaltimer)() { return clock_tick(); }
    uint32_t FUNC(sreg_globaltimer_lo)() { return clock_tick(); }
    uint32_t FUNC(sreg_globaltimer_hi)() { return 0; }

    static inline uint32_t tg_volume()
    {
        return __zluda_ptx_impl_sreg_ntid(0) * __zluda_ptx_impl_sreg_ntid(1) * __zluda_ptx_impl_sreg_ntid(2);
    }
    static inline uint32_t tg_linear_tid()
    {
        return __zluda_ptx_impl_sreg_tid(0) +
               __zluda_ptx_impl_sreg_ntid(0) * (__zluda_ptx_impl_sreg_tid(1) + __zluda_ptx_impl_sreg_ntid(1) * __zluda_ptx_impl_sreg_tid(2));
    }
    uint32_t FUNC(sreg_warpid)() { return tg_linear_tid() / 32u; }
    uint32_t FUNC(sreg_nwarpid)() { return (tg_volume() + 31u) / 32u; }
    uint32_t FUNC(sreg_warpsize)() { return 32; }
    uint32_t FUNC(sreg_smid)() { return 0; }
    uint32_t FUNC(sreg_nsmid)() { return 1; }
    uint64_t FUNC(sreg_gridid)() { return 0; }
    uint64_t FUNC(sreg_current_graph_exec)() { return 0; }

    // shared memory of the CTA (PTX ISA 10.29-10.32): the kernel's static .shared bytes (resolved per kernel by the AIR
    // lowering) and the launch's dynamic size (a hidden kernel parameter); allocated in 128-byte units; the reserved
    // region (1024 bytes) is placed after the user shared memory, as in the model
    uint32_t __zluda_metal_static_smem_size();
    uint32_t __zluda_metal_dynamic_smem_size();
    static inline uint32_t smem_total() { return (__zluda_metal_static_smem_size() + __zluda_metal_dynamic_smem_size() + 127u) & ~127u; }
    uint32_t FUNC(sreg_dynamic_smem_size)() { return __zluda_metal_dynamic_smem_size(); }
    uint32_t FUNC(sreg_total_smem_size)() { return smem_total(); }
    uint32_t FUNC(sreg_aggr_smem_size)() { return smem_total() + 1024u; }
    uint32_t FUNC(sreg_reserved_smem_offset_begin)() { return smem_total(); }
    uint32_t FUNC(sreg_reserved_smem_offset_end)() { return smem_total() + 1024u; }
    uint32_t FUNC(sreg_reserved_smem_offset_cap)() { return 1024u; }
    uint32_t FUNC(sreg_reserved_smem_offset_0)() { return smem_total(); }
    uint32_t FUNC(sreg_reserved_smem_offset_1)() { return smem_total() + 512u; }

    // per-CTA global memory (.minperctamemory): not allocated (0, as for a kernel without the directive)
    uint64_t FUNC(sreg_perctamemoryoffset)() { return 0; }
    uint32_t FUNC(sreg_perctamemorysize)() { return 0; }

#define SREG_ZERO(N) uint32_t FUNC(sreg_envreg##N)() { return 0; }
    SREG_ZERO(0) SREG_ZERO(1) SREG_ZERO(2) SREG_ZERO(3) SREG_ZERO(4) SREG_ZERO(5) SREG_ZERO(6) SREG_ZERO(7)
    SREG_ZERO(8) SREG_ZERO(9) SREG_ZERO(10) SREG_ZERO(11) SREG_ZERO(12) SREG_ZERO(13) SREG_ZERO(14) SREG_ZERO(15)
    SREG_ZERO(16) SREG_ZERO(17) SREG_ZERO(18) SREG_ZERO(19) SREG_ZERO(20) SREG_ZERO(21) SREG_ZERO(22) SREG_ZERO(23)
    SREG_ZERO(24) SREG_ZERO(25) SREG_ZERO(26) SREG_ZERO(27) SREG_ZERO(28) SREG_ZERO(29) SREG_ZERO(30) SREG_ZERO(31)
#undef SREG_ZERO
#define SREG_PM(N) uint32_t FUNC(sreg_pm##N)() { return 0; } uint64_t FUNC(sreg_pm##N##_64)() { return 0; }
    SREG_PM(0) SREG_PM(1) SREG_PM(2) SREG_PM(3) SREG_PM(4) SREG_PM(5) SREG_PM(6) SREG_PM(7)
#undef SREG_PM
    uint32_t FUNC(sreg_clock)() { return clock_tick(); }
