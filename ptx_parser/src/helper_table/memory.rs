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
        return err("mbarrier .multicast::cluster needs thread-block clusters");
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
        match ai {
            Some(1) => {
                let s = ops.get(2)?;
                let Some(a) = addr(&ops[1]) else { return err("mbarrier wait address") };
                return Call::new(format!("mbar_wait{p}{sfx}"))
                    .dst(ops[0].op.clone(), ScalarType::Pred)
                    .src_space(a, ScalarType::B64, sp)
                    .src(s.op.clone(), st_ty)
                    .imm(ctrl)
                    .build();
            }
            Some(2) => {
                let s = ops.get(3)?;
                let Some(a) = addr(&ops[2]) else { return err("mbarrier wait address") };
                return Call::new(format!("mbar_wait2{p}{sfx}"))
                    .dst(ops[0].op.clone(), ScalarType::Pred)
                    .dst2(ops[1].op.clone(), ScalarType::Pred)
                    .src_space(a, ScalarType::B64, sp)
                    .src(s.op.clone(), st_ty)
                    .imm(ctrl)
                    .build();
            }
            _ => return err("mbarrier wait operands"),
        }
    }
    err("mbarrier form")
}

#[allow(unused_variables)]
pub(super) fn lower<'i>(opcode: &str, base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    match base {
        "mbarrier" => mbarrier(m, ops),
        _ => None,
    }
}
