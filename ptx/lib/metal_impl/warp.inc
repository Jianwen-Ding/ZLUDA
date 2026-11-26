// Included inside extern "C" of zluda_ptx_metal_impl.cpp (after float.inc): warp collectives lowered by
// ptx_parser/src/helper_table/warp.rs.  Sail: ptx_coll.sail (collCompute).  The lanes taking part in an
// instance are the non-exited lanes of the membermask; on AGX a SIMD-group runs its 32 lanes in lockstep
// and exited (or diverged) lanes are inactive, so they are the active lanes of the mask.
    static inline uint32_t coll_lanes(uint32_t mask) { return ballot32(true) & mask; }

    // kind: 0 all, 1 any, 2 uni, 3 ballot; bit 4: the source predicate is negated
    static inline uint32_t warp_vote(bool p, uint32_t mask, uint32_t kind)
    {
        bool v = (kind >> 4) & 1 ? !p : p;
        uint32_t part = coll_lanes(mask);
        uint32_t b = ballot32(v) & part;
        switch (kind & 15)
        {
        case 0: return b == part;
        case 1: return b != 0;
        case 2: return b == part || b == 0;
        default: return b;
        }
    }
    bool FUNC(warp_vote_pred)(bool p, uint32_t mask, uint32_t kind) { return warp_vote(p, mask, kind) != 0; }
    uint32_t FUNC(warp_vote_ballot)(bool p, uint32_t mask, uint32_t kind) { return warp_vote(p, mask, kind); }

    // the lanes of `part` whose value equals v
    static inline uint32_t same_lanes(uint64_t v, uint32_t part, bool wide)
    {
        uint32_t r = 0;
        for (uint16_t l = 0; l < 32; l++)
        {
            uint32_t lo = air_simd_shuffle((uint32_t)v, l);
            uint32_t hi = wide ? air_simd_shuffle((uint32_t)(v >> 32), l) : 0;
            uint64_t o = wide ? (((uint64_t)hi << 32) | lo) : lo;
            if (((part >> l) & 1) && o == v)
                r |= 1u << l;
        }
        return r;
    }
    uint32_t FUNC(warp_match_any_32)(uint32_t a, uint32_t mask) { return same_lanes(a, coll_lanes(mask), false); }
    uint32_t FUNC(warp_match_any_64)(uint64_t a, uint32_t mask) { return same_lanes(a, coll_lanes(mask), true); }
    // match.all: "d is set to mask corresponding to non-exited threads in membermask" when all values agree
    static inline B32Pred match_all(uint64_t a, uint32_t mask, bool wide)
    {
        uint32_t part = coll_lanes(mask);
        bool ok = same_lanes(a, part, wide) == part;
        return B32Pred{ok ? part : 0u, (uint32_t)ok};
    }
    B32Pred FUNC(warp_match_all_32)(uint32_t a, uint32_t mask) { return match_all(a, mask, false); }
    B32Pred FUNC(warp_match_all_64)(uint64_t a, uint32_t mask) { return match_all(a, mask, true); }
    uint32_t FUNC(warp_match_all_d_32)(uint32_t a, uint32_t mask) { return match_all(a, mask, false).x; }
    uint32_t FUNC(warp_match_all_d_64)(uint64_t a, uint32_t mask) { return match_all(a, mask, true).x; }
    bool FUNC(warp_match_all_pred_32)(uint32_t a, uint32_t mask) { return match_all(a, mask, false).y != 0; }
    bool FUNC(warp_match_all_pred_64)(uint64_t a, uint32_t mask) { return match_all(a, mask, true).y != 0; }

    // elect.sync: the lowest lane of the membermask is the leader
    static inline uint32_t elect_leader(uint32_t mask)
    {
        uint32_t part = coll_lanes(mask);
        return part ? (uint32_t)__builtin_ctz(part) : 0u;
    }
    B32Pred FUNC(warp_elect)(uint32_t mask)
    {
        uint32_t l = elect_leader(mask);
        return B32Pred{l, (uint32_t)(l == __zluda_ptx_impl_sreg_laneid())};
    }
    bool FUNC(warp_elect_pred)(uint32_t mask) { return elect_leader(mask) == __zluda_ptx_impl_sreg_laneid(); }
    uint32_t FUNC(warp_elect_lane)(uint32_t mask) { return elect_leader(mask); }

    // redux.sync: ctrl op 0-3 (add min max and or xor), type 4-7 (u32 s32 b32 f32), .abs 8, .NaN 9, .ftz 10
    static inline uint32_t redux_step(uint32_t ctrl, uint32_t a, uint32_t b)
    {
        uint32_t op = ctrl & 15, ty = (ctrl >> 4) & 15;
        if (ty == 3 && (op == 1 || op == 2))
            return (uint32_t)fp_minmax((ctrl >> 10) & 1, (ctrl >> 9) & 1, false, T_F32, a, b, op == 1);
        switch (op)
        {
        case 0: return a + b;
        case 1: return ty == 1 ? ((int32_t)a <= (int32_t)b ? a : b) : (a <= b ? a : b);
        case 2: return ty == 1 ? ((int32_t)a >= (int32_t)b ? a : b) : (a >= b ? a : b);
        case 3: return a & b;
        case 4: return a | b;
        default: return a ^ b;
        }
    }
    uint32_t FUNC(warp_redux)(uint32_t a, uint32_t mask, uint32_t ctrl)
    {
        uint32_t part = coll_lanes(mask);
        if (((ctrl >> 4) & 15) == 3 && ((ctrl >> 8) & 1))
            a &= 0x7FFFFFFFu;
        uint32_t acc = 0;
        bool first = true;
        for (uint16_t l = 0; l < 32; l++)
        {
            uint32_t o = air_simd_shuffle(a, l);
            if (!((part >> l) & 1))
                continue;
            acc = first ? o : redux_step(ctrl, acc, o);
            first = false;
        }
        return acc;
    }

    // coll_test (the formal model's rendezvous test): kind 0 sum of the lanes' values, 1 the value of the lowest lane
    uint32_t FUNC(warp_coll_test)(uint32_t a, uint32_t mask, uint32_t kind)
    {
        uint32_t part = coll_lanes(mask);
        if (kind == 1)
            return air_simd_shuffle(a, (uint16_t)(part ? __builtin_ctz(part) : 0));
        uint32_t s = 0;
        for (uint16_t l = 0; l < 32; l++)
        {
            uint32_t o = air_simd_shuffle(a, l);
            if ((part >> l) & 1)
                s += o;
        }
        return s;
    }
