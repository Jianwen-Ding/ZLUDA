//! memory, asynchronous copies and barriers (mbarrier, cp.async, cp.async.bulk, isspacep, alloca, ...).
//! Helpers: ptx/lib/metal_impl/memory.inc. Semantics: PTX-formalization sail/ptx/ptx_async.sail, ptx_tma.sail.
use super::*;

fn has(m: &[&str], s: &str) -> bool {
    m.contains(&s)
}

/// The state space of an address operand from the opcode's modifiers: `.shared{::cta, ::cluster}` (a CTA is one
/// threadgroup: the executing CTA's window of `.shared::cluster` is its own shared memory) or generic.
fn shared_or_generic(m: &[&str]) -> StateSpace {
    if m.iter().any(|x| *x == "shared" || *x == "shared::cta" || *x == "shared::cluster") {
        StateSpace::Shared
    } else {
        StateSpace::Generic
    }
}

fn suffix(sp: StateSpace) -> &'static str {
    if sp == StateSpace::Shared {
        ""
    } else {
        "_g"
    }
}

fn addr<'i>(h: &HOp<'i>) -> Option<Op<'i>> {
    if h.addr {
        Some(h.op.clone())
    } else {
        None
    }
}

fn mbarrier<'i>(m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    let sp = shared_or_generic(m);
    let sfx = suffix(sp);
    if m.iter().any(|x| x.starts_with("multicast")) {
        // the operation on the object at the same offset in each CTA of ctaMask (the last operand)
        let w32 = has(m, "multicast::cluster::32b");
        let mask_of = |o: &HOp<'i>| o.op.clone();
        let _ = w32;
        if has(m, "arrive") || has(m, "arrive_drop") {
            if ops.len() < 3 {
                return err("mbarrier multicast operands");
            }
            let Some(a) = addr(&ops[1]) else { return err("mbarrier multicast address") };
            let mask = mask_of(ops.last().unwrap());
            let args = &ops[2..ops.len() - 1];
            let (count, tx) = if has(m, "expect_tx") {
                (imm(1), args.first().map(|x| x.op.clone()).unwrap_or(imm(0)))
            } else {
                (args.first().map(|x| x.op.clone()).unwrap_or(imm(1)), imm(0))
            };
            return Call::new("mbar_arrive_mc")
                .src_space(a, ScalarType::B64, StateSpace::Shared)
                .src(count, ScalarType::U32)
                .src(tx, ScalarType::U32)
                .src(mask, ScalarType::B32)
                .imm(has(m, "arrive_drop") as u64)
                .build();
        }
        if has(m, "expect_tx") || has(m, "complete_tx") {
            let [a, n, mk] = ops else { return err("mbarrier multicast tx operands") };
            let Some(a) = addr(a) else { return err("mbarrier multicast address") };
            return Call::new("mbar_tx_mc")
                .src_space(a, ScalarType::B64, StateSpace::Shared)
                .src(n.op.clone(), ScalarType::U32)
                .src(mk.op.clone(), ScalarType::B32)
                .imm(has(m, "complete_tx") as u64)
                .build();
        }
        return err("mbarrier multicast form");
    }
    if has(m, "init") {
        let [a, n] = ops else { return err("mbarrier.init operands") };
        let Some(a) = addr(a) else { return err("mbarrier.init address") };
        return Call::new(format!("mbar_init{sfx}"))
            .src_space(a, ScalarType::B64, sp)
            .src(n.op.clone(), ScalarType::U32)
            .imm(has(m, "layout::v1") as u64)
            .build();
    }
    if has(m, "inval") {
        let [a] = ops else { return err("mbarrier.inval operands") };
        let Some(a) = addr(a) else { return err("mbarrier.inval address") };
        return Call::new(format!("mbar_inval{sfx}")).src_space(a, ScalarType::B64, sp).build();
    }
    if has(m, "pending_count") {
        let [d, s] = ops else { return err("mbarrier.pending_count operands") };
        return Call::new("mbar_pending_count")
            .dst(d.op.clone(), ScalarType::U32)
            .src(s.op.clone(), ScalarType::B64)
            .build();
    }
    if has(m, "check_layout") {
        let [d, a] = ops else { return err("mbarrier.check_layout operands") };
        let Some(a) = addr(a) else { return err("mbarrier.check_layout address") };
        return Call::new(format!("mbar_check_layout{sfx}"))
            .dst(d.op.clone(), ScalarType::Pred)
            .src_space(a, ScalarType::B64, sp)
            .imm(has(m, "layout::v1") as u64)
            .build();
    }
    if has(m, "arrive") || has(m, "arrive_drop") {
        if ops.len() < 2 {
            return err("mbarrier.arrive operands");
        }
        let Some(a) = addr(&ops[1]) else { return err("mbarrier.arrive address") };
        let rest = &ops[2..];
        let (count, tx) = if has(m, "expect_tx") {
            let Some(t) = rest.first() else { return err("mbarrier.arrive.expect_tx txCount") };
            (imm(1), t.op.clone())
        } else {
            (rest.first().map(|x| x.op.clone()).unwrap_or(imm(1)), imm(0))
        };
        let drop = has(m, "arrive_drop") as u64;
        let c = if ops[0].sink {
            Call::new(format!("mbar_arrive_nd{sfx}"))
        } else {
            Call::new(format!("mbar_arrive{sfx}")).dst(ops[0].op.clone(), ScalarType::B64)
        };
        return c
            .src_space(a, ScalarType::B64, sp)
            .src(count, ScalarType::U32)
            .src(tx, ScalarType::U32)
            .imm(drop)
            .build();
    }
    if has(m, "expect_tx") || has(m, "complete_tx") {
        let [a, n] = ops else { return err("mbarrier.expect_tx operands") };
        let Some(a) = addr(a) else { return err("mbarrier.expect_tx address") };
        return Call::new(format!("mbar_tx{sfx}"))
            .src_space(a, ScalarType::B64, sp)
            .src(n.op.clone(), ScalarType::U32)
            .imm(has(m, "complete_tx") as u64)
            .build();
    }
    if has(m, "test_wait") || has(m, "try_wait") {
        let ctrl = has(m, "parity") as u64
            | ((has(m, "try_wait") as u64) << 1)
            | ((has(m, "phase_type::conditional") as u64) << 2);
        let (st_ty, p) = if has(m, "parity") { (ScalarType::B32, "p") } else { (ScalarType::B64, "") };
        // d, [a], state{, suspendTimeHint}  or  d, reportPredicate, [a], state{, hint} (.phase_type::primary)
        let ai = ops.iter().position(|o| o.addr);
        // reportValue (the operand before the address in the 4-operand form) is always 0 in the model; it is
        // not written here (one helper call has at most two destinations)
        let Some(ai) = ai else { return err("mbarrier wait operands") };
        if ai != 1 && ai != 2 {
            return err("mbarrier wait operands");
        }
        let s = ops.get(ai + 1)?;
        let Some(a) = addr(&ops[ai]) else { return err("mbarrier wait address") };
        return match &ops[0].pair {
            None => Call::new(format!("mbar_wait{p}{sfx}"))
                .dst(ops[0].op.clone(), ScalarType::Pred)
                .src_space(a, ScalarType::B64, sp)
                .src(s.op.clone(), st_ty)
                .imm(ctrl)
                .build(),
            Some(rp) => {
                let c = if ops[0].sink { Call::new(format!("mbar_wait2{p}{sfx}")) } else {
                    Call::new(format!("mbar_wait2{p}{sfx}")).dst(ops[0].op.clone(), ScalarType::Pred)
                };
                c.dst2(rp.clone(), ScalarType::Pred)
                    .src_space(a, ScalarType::B64, sp)
                    .src(s.op.clone(), st_ty)
                    .imm(ctrl)
                    .build()
            }
        };
    }
    err("mbarrier form")
}

/// cp.async.{ca,cg}.shared{::cta}.global [dst], [src], cp-size{, src-size | ignore-src}{, cache-policy} (from the
/// grammar rule of lib.rs). With .L2::cache_hint a single optional operand is the cache policy.
pub(crate) fn cp_async<'i>(
    dst: Op<'i>,
    src: Op<'i>,
    cp_size: Op<'i>,
    src_size: Option<Op<'i>>,
    hint: bool,
    cache_policy: Option<Op<'i>>,
    pred: bool,
) -> Result<ast::Instruction<ParsedOperandStr<'i>>, String> {
    let n = match &cp_size {
        ast::ParsedOperand::Imm(i) => i.as_u64().filter(|n| [4, 8, 16].contains(n)),
        _ => None,
    }
    .ok_or_else(|| "cp.async cp-size".to_string())?;
    let src_size = if hint && cache_policy.is_none() { None } else { src_size };
    let c = match src_size {
        Some(ss) if pred => Call::new("cp_async_ig")
            .src_space(dst, ScalarType::B8, StateSpace::Shared)
            .src_space(src, ScalarType::B8, StateSpace::Global)
            .imm(n)
            .src(ss, ScalarType::Pred),
        Some(ss) => Call::new("cp_async")
            .src_space(dst, ScalarType::B8, StateSpace::Shared)
            .src_space(src, ScalarType::B8, StateSpace::Global)
            .imm(n)
            .src(ss, ScalarType::U32),
        None => Call::new("cp_async")
            .src_space(dst, ScalarType::B8, StateSpace::Shared)
            .src_space(src, ScalarType::B8, StateSpace::Global)
            .imm(n)
            .imm(n),
    };
    c.build().unwrap()
}

fn cp<'i>(m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    if !has(m, "async") {
        return None;
    }
    // cp.async.mbarrier.arrive{.noinc}{.shared{::cta}}.b64 [addr]: without .noinc the pending count is
    // incremented, then the arrive-on of the (already complete) copies decrements it: no change
    if has(m, "mbarrier") && has(m, "arrive") {
        let [a] = ops else { return err("cp.async.mbarrier.arrive operands") };
        let Some(a) = addr(a) else { return err("cp.async.mbarrier.arrive address") };
        if !has(m, "noinc") {
            return Call::new("mem_nop").build();
        }
        let sp = shared_or_generic(m);
        return Call::new(format!("mbar_arrive_nd{}", suffix(sp)))
            .src_space(a, ScalarType::B64, sp)
            .imm(1)
            .imm(0)
            .imm(0)
            .build();
    }
    if has(m, "tensor") {
        return tensor(m, ops);
    }
    if has(m, "bulk") {
        return bulk(m, ops);
    }
    None
}

/// 0 or the report pattern code of .mbarrier::report::validity::* (memory.inc rep_chunk)
fn report_code(m: &[&str]) -> u64 {
    for x in m {
        if let Some(p) = x.strip_prefix("mbarrier::report::validity::") {
            return match p {
                "per_element::ff" => 1,
                "per_16bytes::80000000" => 2,
                "per_16bytes::8000" => 3,
                "per_16bytes::80" => 4,
                "per_16bytes::8" => 5,
                _ => 0,
            };
        }
    }
    0
}

fn red_op(m: &[&str]) -> Option<u64> {
    ["add", "min", "max", "inc", "dec", "and", "or", "xor"]
        .iter()
        .position(|o| m.contains(o))
        .map(|i| i as u64 + 1)
}

/// the destination comes first: `.shared::cluster.global` is a load, `.global.shared::cta` a store
fn to_shared(m: &[&str]) -> bool {
    let g = m.iter().position(|x| *x == "global");
    let s = m.iter().position(|x| x.starts_with("shared"));
    match (g, s) {
        (Some(g), Some(s)) => s < g,
        (None, Some(_)) => true,
        _ => false,
    }
}

fn vec_or_none<'i>(op: Option<&Op<'i>>) -> Op<'i> {
    op.cloned().unwrap_or(ast::ParsedOperand::VecPack(Vec::new()))
}

/// cp.async.bulk.tensor / cp.reduce.async.bulk.tensor / cp.async.bulk.prefetch.tensor
fn tensor<'i>(m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    if has(m, "prefetch") {
        return Call::new("mem_nop").build();
    }
    if m.iter().any(|x| x.starts_with("multicast")) {
        // [dst], [tmap, {coords}], [mbar], ctaMask{, cache-policy}: tile mode
        if ops.len() < 4 || !ops[0].addr || !ops[1].addr || !ops[2].addr {
            return err("cp.async.bulk.tensor multicast operands");
        }
        if m.iter().any(|x| x.starts_with("im2col") || x.starts_with("tile::gather4")) {
            return err("cp.async.bulk.tensor multicast im2col / gather4");
        }
        let [cs] = &ops[1].items[..] else { return err("cp.async.bulk.tensor coordinates") };
        let mask = if has(m, "multicast::cluster::32b") { ops[3].op.clone() } else { ops[3].op.clone() };
        return Call::new("tma_load_mc")
            .src_space(ops[0].op.clone(), ScalarType::B8, StateSpace::Shared)
            .src(ops[1].op.clone(), ScalarType::U64)
            .src_vec(cs.clone(), 8, ScalarType::S32)
            .src_space(ops[2].op.clone(), ScalarType::B64, StateSpace::Shared)
            .src(mask, if has(m, "multicast::cluster::32b") { ScalarType::B32 } else { ScalarType::B16 })
            .imm((report_code(m) << 4) | ((has(m, "cta_group::2") as u64) << 8))
            .build();
    }
    if has(m, "override::global_dim") || has(m, "override::global_dim_stride") {
        // [dst], [tmap, gaddr, {dims}, {lower strides}, upper, {coords}], [mbar]: coordinates, dims and strides go in
        // one vector (rank n: n + n + n - 1 <= 8)
        let [g, ds, rest @ ..] = &ops.get(1).map(|o| &o.items[..]).unwrap_or(&[])[..] else {
            return err("override::global_dim operands");
        };
        let (ls, up, cs) = match rest {
            [ls, up, cs] => (Some(ls), up.clone(), cs),
            [cs] => (None, imm(0), cs),
            _ => return err("override::global_dim operands"),
        };
        let elems = |o: &Op<'i>| match o {
            ast::ParsedOperand::VecPack(v) => v.clone(),
            _ => Vec::new(),
        };
        let n = elems(cs).len();
        let mut all = elems(cs);
        all.extend(elems(ds));
        if let Some(ls) = ls {
            all.extend(elems(ls));
        }
        if n == 0 || n > 3 || all.len() > 8 || ops.len() < 3 {
            return err("override::global_dim rank");
        }
        return Call::new(format!("tma_load_ov{n}"))
            .src_space(ops[0].op.clone(), ScalarType::B8, StateSpace::Shared)
            .src(ops[1].op.clone(), ScalarType::U64)
            .src(g.clone(), ScalarType::U64)
            .src_vec(ast::ParsedOperand::VecPack(all), 8, ScalarType::B32)
            .src_space(ops[2].op.clone(), ScalarType::B64, StateSpace::Shared)
            .src(up, ScalarType::B16)
            .build();
    }
    if to_shared(m) {
        // [dst], [tmap, {coords}], [mbar]{, im2colInfo}{, cache-policy}
        if ops.len() < 3 || !ops[0].addr || !ops[1].addr || !ops[2].addr {
            return err("cp.async.bulk.tensor operands");
        }
        let (mode, imd) = if has(m, "tile::gather4") {
            (1, 0)
        } else if has(m, "im2col::w") {
            (2, 1)
        } else if has(m, "im2col::w::128") {
            (2, 2)
        } else if has(m, "im2col_no_offs::w") {
            (2, 3)
        } else if has(m, "im2col") {
            (2, 0)
        } else {
            (0, 0)
        };
        let ctrl = mode | (imd << 2) | (report_code(m) << 4);
        if has(m, "override::global_address") {
            let [g, cs] = &ops[1].items[..] else { return err("override::global_address operands") };
            return Call::new("tma_load_ga")
                .src_space(ops[0].op.clone(), ScalarType::B8, StateSpace::Shared)
                .src(ops[1].op.clone(), ScalarType::U64)
                .src(g.clone(), ScalarType::U64)
                .src_vec(cs.clone(), 8, ScalarType::S32)
                .src_space(ops[2].op.clone(), ScalarType::B64, StateSpace::Shared)
                .imm(ctrl)
                .build();
        }
        let [cs] = &ops[1].items[..] else { return err("cp.async.bulk.tensor coordinates") };
        let inf = if mode == 2 { ops.get(3).filter(|o| matches!(o.op, ast::ParsedOperand::VecPack(_))).map(|o| &o.op) } else { None };
        return Call::new("tma_load")
            .src_space(ops[0].op.clone(), ScalarType::B8, StateSpace::Shared)
            .src(ops[1].op.clone(), ScalarType::U64)
            .src_vec(cs.clone(), 8, ScalarType::S32)
            .src_space(ops[2].op.clone(), ScalarType::B64, StateSpace::Shared)
            .src_vec(vec_or_none(inf), 8, ScalarType::S32)
            .imm(ctrl)
            .build();
    }
    // [tmap, {coords}], [src]{, cache-policy}
    if ops.len() < 2 || !ops[0].addr || !ops[1].addr {
        return err("cp.async.bulk.tensor store operands");
    }
    let [cs] = &ops[0].items[..] else { return err("cp.async.bulk.tensor coordinates") };
    let op = if has(m, "reduce") {
        let Some(o) = red_op(m) else { return err("cp.reduce.async.bulk.tensor operation") };
        o
    } else {
        0
    };
    let mode = if has(m, "tile::scatter4") {
        1
    } else if has(m, "im2col_no_offs::w") {
        3
    } else if has(m, "im2col_no_offs") {
        2
    } else {
        0
    };
    Call::new("tma_store")
        .src(ops[0].op.clone(), ScalarType::U64)
        .src_vec(cs.clone(), 8, ScalarType::S32)
        .src_space(ops[1].op.clone(), ScalarType::B8, StateSpace::Shared)
        .imm(op | (mode << 4))
        .build()
}

/// element type code of a bulk reduction (memory.inc red_elem) and its size
fn red_type(t: &str) -> Option<(u64, u64)> {
    Some(match t {
        "u32" | "b32" => (2, 4),
        "s32" => (3, 4),
        "u64" | "b64" => (4, 8),
        "s64" => (5, 8),
        "f16" => (6, 2),
        "f32" => (7, 4),
        "f64" => (9, 8),
        "bf16" => (10, 2),
        _ => return None,
    })
}

/// cp.async.bulk / cp.reduce.async.bulk (non-tensor), prefetch, commit_group / wait_group
fn bulk<'i>(m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    if has(m, "prefetch") || has(m, "commit_group") || has(m, "wait_group") || has(m, "wait_group.read") {
        return Call::new("mem_nop").build();
    }
    if m.iter().any(|x| x.starts_with("multicast")) {
        // [dst], [src], size, [mbar], ctaMask
        if ops.len() < 5 || !ops[0].addr || !ops[1].addr || !ops[3].addr {
            return err("cp.async.bulk multicast operands");
        }
        return Call::new("bulk_g2s_mc")
            .src_space(ops[0].op.clone(), ScalarType::B8, StateSpace::Shared)
            .src_space(ops[1].op.clone(), ScalarType::B8, StateSpace::Global)
            .src(ops[2].op.clone(), ScalarType::U32)
            .src_space(ops[3].op.clone(), ScalarType::B64, StateSpace::Shared)
            .src(ops[4].op.clone(), if has(m, "multicast::cluster::32b") { ScalarType::B32 } else { ScalarType::B16 })
            .imm(report_code(m) << 12)
            .build();
    }
    if ops.len() < 3 || !ops[0].addr || !ops[1].addr {
        return err("cp.async.bulk operands");
    }
    let src_global = has(m, "global") && to_shared(m);
    let (rop, rty) = if has(m, "reduce") {
        let Some(o) = red_op(m) else { return err("cp.reduce.async.bulk operation") };
        let Some(t) = m.iter().rev().find_map(|x| red_type(x)) else { return err("cp.reduce.async.bulk type") };
        (o, t)
    } else {
        (0, (0, 1))
    };
    let ctrl = rop | (rty.0 << 4) | (rty.1 << 8) | (report_code(m) << 12) | ((has(m, "cp_mask") as u64) << 16);
    if to_shared(m) {
        // [dst], [src], size, [mbar]: from global (or from the CTA's own shared memory)
        let Some(mb) = ops.get(3).filter(|o| o.addr) else { return err("cp.async.bulk mbarrier operand") };
        let c = Call::new(if src_global { "bulk_g2s" } else { "bulk_s2s" })
            .src_space(ops[0].op.clone(), ScalarType::B8, StateSpace::Shared);
        let c = if src_global {
            c.src_space(ops[1].op.clone(), ScalarType::B8, StateSpace::Global)
        } else {
            c.src_space(ops[1].op.clone(), ScalarType::B8, StateSpace::Shared)
        };
        return c
            .src(ops[2].op.clone(), ScalarType::U32)
            .src_space(mb.op.clone(), ScalarType::B64, StateSpace::Shared)
            .imm(ctrl)
            .build();
    }
    // global <- shared::cta: [dst], [src], size{, byteMask}{, cache-policy}
    let mask = if has(m, "cp_mask") { ops.get(3).map(|o| o.op.clone()).unwrap_or(imm(0xFFFF)) } else { imm(0xFFFF) };
    Call::new("bulk_s2g")
        .src_space(ops[0].op.clone(), ScalarType::B8, StateSpace::Global)
        .src_space(ops[1].op.clone(), ScalarType::B8, StateSpace::Shared)
        .src(ops[2].op.clone(), ScalarType::U32)
        .src(mask, ScalarType::U32)
        .imm(ctrl)
        .build()
}

fn tensormap<'i>(m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    if has(m, "cp_fenceproxy") {
        // tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.scope.sync.aligned [dst], [src], 128
        let [d, s, _] = ops else { return err("tensormap.cp_fenceproxy operands") };
        return Call::new("tensormap_cp")
            .src(d.op.clone(), ScalarType::U64)
            .src_space(s.op.clone(), ScalarType::B8, StateSpace::Shared)
            .build();
    }
    if !has(m, "replace") {
        return None;
    }
    let fields = [
        "global_address", "rank", "box_dim", "global_dim", "global_stride", "element_stride", "elemtype",
        "interleave_layout", "swizzle_mode", "swizzle_atomicity", "fill_mode",
    ];
    let Some(f) = fields.iter().position(|x| m.contains(x)) else { return err("tensormap.replace field") };
    let vt = if has(m, "b64") { ScalarType::B64 } else { ScalarType::B32 };
    let (a, ord, v) = match ops {
        [a, o, v] => (a, o.op.clone(), v),
        [a, v] => (a, imm(0), v),
        _ => return err("tensormap.replace operands"),
    };
    let v = if vt == ScalarType::B32 {
        // the helper takes the new value as 64 bits; a 32-bit register goes through a b32 slot
        return Call::new(if has(m, "shared::cta") { "tensormap_replace32_s" } else { "tensormap_replace32" })
            .src_space_or_int(a, has(m, "shared::cta"))
            .imm(f as u64)
            .src(ord, ScalarType::U32)
            .src(v.op.clone(), ScalarType::B32)
            .build();
    } else {
        v
    };
    Call::new(if has(m, "shared::cta") { "tensormap_replace_s" } else { "tensormap_replace" })
        .src_space_or_int(a, has(m, "shared::cta"))
        .imm(f as u64)
        .src(ord, ScalarType::U32)
        .src(v.op.clone(), ScalarType::B64)
        .build()
}

trait AddrSrc<'i> {
    fn src_space_or_int(self, a: &HOp<'i>, shared: bool) -> Self;
}
impl<'i> AddrSrc<'i> for Call<'i> {
    fn src_space_or_int(self, a: &HOp<'i>, shared: bool) -> Self {
        if shared {
            self.src_space(a.op.clone(), ScalarType::B8, StateSpace::Shared)
        } else {
            self.src(a.op.clone(), ScalarType::U64)
        }
    }
}

/// mbarrier.{test,try}_wait.phase_type::primary waitComplete|reportPredicate, reportValue, [a], state: the model
/// writes reportValue 0 ("only ever OR'ed with 0")
pub(super) fn extra<'i>(base: &str, m: &[&str], ops: &[HOp<'i>]) -> Vec<ast::Instruction<ParsedOperandStr<'i>>> {
    if base == "mbarrier" && (has(m, "test_wait") || has(m, "try_wait")) && ops.len() >= 3 && ops[2].addr && !ops[1].addr {
        return vec![ast::Instruction::Mov {
            data: ast::MovDetails { typ: ast::Type::Scalar(ScalarType::B32) },
            arguments: ast::MovArgs { dst: ops[1].op.clone(), src: imm(0) },
        }];
    }
    Vec::new()
}

/// element type code (memory.inc rmw_elem) and size of an atom / red type
fn atom_type(t: &str) -> Option<(u64, u64)> {
    Some(match t {
        "u16" | "b16" => (1, 2),
        "u32" | "b32" => (2, 4),
        "s32" => (3, 4),
        "u64" | "b64" => (4, 8),
        "s64" => (5, 8),
        "f16" => (6, 2),
        "f32" => (7, 4),
        "f64" => (9, 8),
        "bf16" => (10, 2),
        "f16x2" => (13, 4),
        "bf16x2" => (14, 4),
        _ => return None,
    })
}

/// atom / red forms outside the grammar: atom{.sem}{.scope}{.space}{.vec}.op{.noftz}{.hint}.type d, [a], b{, c}{, pol};
/// red ... [a], b{, pol}. Lane-serialised read-modify-write (memory.inc atom_*): a SIMD group runs until a barrier.
fn atom_red<'i>(base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    let red = base == "red";
    if red && has(m, "async") {
        // red.async{.sem}{.scope}{.ss}.completion.op.type [a], b, [mbar]
        let ops_ = ["add", "min", "max", "inc", "dec", "and", "or", "xor"];
        let op = ops_.iter().position(|o| m.contains(o))? as u64 + 1;
        let (code, size) = m.iter().rev().find_map(|x| atom_type(x))?;
        let [a, b, mb] = ops else { return err("red.async operands") };
        let (Some(a), Some(mb)) = (addr(a), addr(mb)) else { return err("red.async address") };
        let sp = shared_or_generic(m);
        let bits = if size == 8 { ScalarType::B64 } else { ScalarType::B32 };
        return Call::new(format!("red_async{}{}", size * 8, suffix(sp)))
            .src_space(a, bits, sp)
            .src(b.op.clone(), bits)
            .src_space(mb, ScalarType::B64, sp)
            .imm(op | (code << 4))
            .build();
    }
    let ops_ = ["add", "min", "max", "inc", "dec", "and", "or", "xor", "exch", "cas"];
    let op = ops_.iter().position(|o| m.contains(o))? as u64 + 1;
    let (code, size) = m.iter().rev().find_map(|x| atom_type(x))?;
    let vec = m.iter().find_map(|x| match *x {
        "v2" => Some(2u8),
        "v4" => Some(4),
        "v8" => Some(8),
        _ => None,
    });
    let space = if m.iter().any(|x| x.starts_with("shared")) {
        StateSpace::Shared
    } else if has(m, "global") {
        StateSpace::Global
    } else {
        StateSpace::Generic
    };
    let ftz = op == 1 && code == 7 && !has(m, "noftz") && space != StateSpace::Shared;
    let ctrl = op | (code << 4) | ((ftz as u64) << 8);
    let sfx = match space {
        StateSpace::Shared => "s",
        StateSpace::Global => "g",
        _ => "f",
    };
    let (d, rest) = if red { (None, ops) } else { (Some(&ops[0]), &ops[1..]) };
    let [a, b, more @ ..] = rest else { return err("atom operands") };
    let Some(a) = addr(a) else { return err("atom address") };
    let bits = match size {
        2 => ScalarType::B16,
        8 => ScalarType::B64,
        _ => ScalarType::B32,
    };
    if let Some(n) = vec {
        let c = Call::new(format!("atom_v{n}_{sfx}"));
        let c = match d {
            Some(d) if !d.sink => c.dst_vec(d.op.clone(), n, ScalarType::B32),
            _ => c,
        };
        return c
            .src_space(a, ScalarType::B32, space)
            .src_vec(b.op.clone(), n, ScalarType::B32)
            .imm(ctrl)
            .build();
    }
    let cv = if op == 10 { more.first().map(|x| x.op.clone()) } else { None };
    let c = Call::new(format!("atom{}_{sfx}", size * 8));
    let c = match d {
        Some(d) if !d.sink => c.dst(d.op.clone(), bits),
        _ => c,
    };
    c.src_space(a, bits, space)
        .src(b.op.clone(), bits)
        .src(cv.unwrap_or(imm(0)), bits)
        .imm(ctrl)
        .build()
}

#[allow(unused_variables)]
pub(super) fn lower<'i>(opcode: &str, base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    match base {
        "mbarrier" => mbarrier(m, ops),
        "cp" => cp(m, ops),
        "tensormap" => tensormap(m, ops),
        "atom" | "red" => atom_red(base, m, ops),
        // cache control: no effect on the results (discard "behaves like a weak write of an unstable indeterminate
        // value": the model leaves memory as it is)
        "prefetch" | "prefetchu" | "discard" | "applypriority" => Call::new("mem_nop").build(),
        // createpolicy: the model's cache policy is 0
        "createpolicy" => {
            let Some(d) = ops.first() else { return err("createpolicy operands") };
            Call::new("mem_zero64").dst(d.op.clone(), ScalarType::B64).build()
        }
        // ldu: ld (the model executes it as ld)
        // mov.b128 q, {lo, hi};  mov.b128 {lo, hi}, q;
        "mov" if m == ["b128"] => {
            let [d, a] = ops else { return err("mov.b128 operands") };
            let pair = |o: &Op<'i>| match o {
                ast::ParsedOperand::VecPack(v) if v.len() == 2 => match (&v[0], &v[1]) {
                    (Some(ast::RegOrImmediate::Reg(x)), Some(ast::RegOrImmediate::Reg(y))) => {
                        Some((ast::ParsedOperand::Reg(*x), ast::ParsedOperand::Reg(*y)))
                    }
                    _ => None,
                },
                _ => None,
            };
            if let Some((lo, hi)) = pair(&a.op) {
                return Call::new("mov_b128_pack")
                    .dst(d.op.clone(), ScalarType::B128)
                    .src(lo, ScalarType::B64)
                    .src(hi, ScalarType::B64)
                    .build();
            }
            if let Some((lo, hi)) = pair(&d.op) {
                return Call::new("mov_b128_unpack")
                    .dst(lo, ScalarType::B64)
                    .dst2(hi, ScalarType::B64)
                    .src(a.op.clone(), ScalarType::B128)
                    .build();
            }
            err("mov.b128 form")
        }
        // alloca.u64 ptr, size{, align}; stacksave.u64 ptr; stackrestore.u64 ptr
        "alloca" if m == ["u64"] => {
            let (d, n, al) = match ops {
                [d, n] => (d, n.op.clone(), imm(8)),
                [d, n, a] => (d, n.op.clone(), a.op.clone()),
                _ => return err("alloca operands"),
            };
            Call::new("alloca_u64")
                .dst(d.op.clone(), ScalarType::U64)
                .src(n, ScalarType::U64)
                .src(al, ScalarType::U32)
                .build()
        }
        "stacksave" if m == ["u64"] => {
            let [d] = ops else { return err("stacksave operands") };
            Call::new("stacksave_u64").dst(d.op.clone(), ScalarType::U64).build()
        }
        "stackrestore" if m == ["u64"] => {
            let [a] = ops else { return err("stackrestore operands") };
            Call::new("stackrestore_u64").src(a.op.clone(), ScalarType::U64).build()
        }
        // mapa{.shared::cluster}.type d, a, rank;  getctarank{.shared::cluster}.type d, a
        "mapa" => {
            let [d, a, r] = ops else { return err("mapa operands") };
            let (name, t) = match (has(m, "shared::cluster"), has(m, "u64")) {
                (true, false) => ("mapa_c32", ScalarType::U32),
                (true, true) => ("mapa_c64", ScalarType::U64),
                (false, true) => ("mapa_g64", ScalarType::U64),
                (false, false) => return err("mapa.u32 generic"),
            };
            Call::new(name).dst(d.op.clone(), t).src(a.op.clone(), t).src(r.op.clone(), ScalarType::U32).build()
        }
        "getctarank" => {
            let [d, a] = ops else { return err("getctarank operands") };
            let (name, t) = match (has(m, "shared::cluster"), has(m, "u64")) {
                (true, false) => ("getctarank_c32", ScalarType::U32),
                (true, true) => ("getctarank_c64", ScalarType::U64),
                (false, true) => ("getctarank_g64", ScalarType::U64),
                (false, false) => return err("getctarank.u32 generic"),
            };
            Call::new(name).dst(d.op.clone(), ScalarType::U32).src(a.op.clone(), t).build()
        }
        // barrier.cluster.arrive / .wait{.acquire}{.aligned}
        "barrier" if m.first() == Some(&"cluster") && ops.is_empty() => {
            if has(m, "wait") {
                Call::new("barrier_cluster_wait").build()
            } else {
                Call::new("mem_nop").build()
            }
        }
        // st.async{.sem}{.scope}{.ss}.completion{.vec}.type [a], b, [mbar]
        "st" if has(m, "async") => {
            let [a, b, mb] = ops else { return err("st.async operands") };
            let (Some(a), Some(mb)) = (addr(a), addr(mb)) else { return err("st.async address") };
            if m.iter().any(|x| x.starts_with('v') && x[1..].parse::<u32>().is_ok()) {
                return err("st.async vector");
            }
            let size = m.iter().rev().find_map(|x| atom_type(x)).map(|x| x.1).unwrap_or(4);
            let sp = shared_or_generic(m);
            let bits = if size == 8 { ScalarType::B64 } else { ScalarType::B32 };
            Call::new(format!("st_async{}{}", size * 8, suffix(sp)))
                .src_space(a, bits, sp)
                .src(b.op.clone(), bits)
                .src_space(mb, ScalarType::B64, sp)
                .build()
        }
        // st.bulk{.weak}{.shared::cta} [a], size, initval
        "st" if has(m, "bulk") => {
            let [a, n, ..] = ops else { return err("st.bulk operands") };
            let Some(a) = addr(a) else { return err("st.bulk address") };
            let sp = shared_or_generic(m);
            Call::new(format!("st_bulk{}", suffix(sp))).src_space(a, ScalarType::B8, sp).src(n.op.clone(), ScalarType::U64).build()
        }
        "clusterlaunchcontrol" => {
            if has(m, "try_cancel") {
                let [a, mb] = ops else { return err("clusterlaunchcontrol.try_cancel operands") };
                let (Some(a), Some(mb)) = (addr(a), addr(mb)) else { return err("clusterlaunchcontrol address") };
                let sp = shared_or_generic(m);
                return Call::new(format!("clc_try_cancel{}", suffix(sp)))
                    .src_space(a, ScalarType::B8, sp)
                    .src_space(mb, ScalarType::B64, sp)
                    .imm(has(m, "multicast::cluster::all") as u64)
                    .build();
            }
            // query_cancel: the response is always the failure response (0)
            let Some(d) = ops.first() else { return err("clusterlaunchcontrol.query_cancel operands") };
            let t = if has(m, "pred") { ScalarType::Pred } else { ScalarType::B32 };
            if let ast::ParsedOperand::VecPack(_) = d.op {
                return err("clusterlaunchcontrol.query_cancel vector");
            }
            Call::new(if t == ScalarType::Pred { "mem_false" } else { "mem_zero32" }).dst(d.op.clone(), t).build()
        }
        "griddepcontrol" | "setmaxnreg" => Call::new("mem_nop").build(),
        "ldu" => {
            let [d, a] = ops else { return err("ldu operands") };
            let Some(a) = addr(a) else { return err("ldu address") };
            let t = m.iter().rev().find_map(|x| scalar(x))?;
            let vec = m.iter().find_map(|x| match *x {
                "v2" => Some(2u8),
                "v4" => Some(4),
                _ => None,
            });
            let space = if has(m, "global") { StateSpace::Global } else { StateSpace::Generic };
            Some(Ok(ast::Instruction::Ld {
                data: ast::LdDetails {
                    qualifier: ast::LdStQualifier::Weak,
                    state_space: space,
                    caching: ast::LdCacheOperator::Cached,
                    typ: match vec {
                        Some(n) => ast::Type::Vector(n, t),
                        None => ast::Type::Scalar(t),
                    },
                    non_coherent: false,
                },
                arguments: ast::LdArgs { dst: d.op.clone(), src: a },
            }))
        }
        // fence.proxy.* (asynchronous operations complete at issue), and the fence forms the grammar lacks: a
        // SIMD group runs until a barrier, so no reordering is observable
        "fence" if ops.len() <= 2 => Call::new("mem_fence").build(),
        "isspacep" => {
            // isspacep.space p, a;  a is a generic address (decided statically, air_lower.cpp resolveIsSpacep)
            let [p, a] = ops else { return err("isspacep operands") };
            let kind = match m {
                ["global"] => 0,
                ["shared"] | ["shared::cta"] => 1,
                ["local"] => 2,
                ["const"] => 3,
                ["param"] | ["param::entry"] => 4,
                ["shared::cluster"] => 5,
                _ => return err("isspacep space"),
            };
            Call::new("isspacep")
                .dst(p.op.clone(), ScalarType::Pred)
                .src(a.op.clone(), ScalarType::B64)
                .imm(kind)
                .build()
        }
        _ => None,
    }
}
