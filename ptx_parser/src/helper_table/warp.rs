//! Warp collectives outside the grammar: vote.sync.uni (and every vote form with a mask), match.sync,
//! elect.sync, redux.sync (all operations and types, .abs, .NaN), coll_test (the formal model's internal
//! rendezvous test instruction).  Sail: ptx_coll.sail (collCompute): the lanes taking part are the
//! non-exited lanes of the membermask, which on AGX (32-lane SIMD-groups in lockstep) are the active lanes
//! of the mask.  Helpers: ptx/lib/metal_impl/warp.inc.
use super::*;

fn pick(m: &[&str], names: &[&str]) -> Option<u64> {
    names.iter().position(|n| m.contains(n)).map(|i| i as u64)
}

pub(super) fn lower<'i>(_opcode: &str, base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    match base {
        "vote" => {
            // vote{.sync}.mode.pred d, {!}a{, membermask};  vote{.sync}.ballot.b32 d, {!}a{, membermask};
            let kind = pick(m, &["all", "any", "uni", "ballot"])?;
            let (d, a, mask) = match ops {
                [d, a, mm] => (d, a, mm.op.clone()),
                [d, a] => (d, a, imm(0xFFFF_FFFF)),
                _ => return err("vote operands"),
            };
            let call = if kind == 3 {
                Call::new("warp_vote_ballot").dst(d.op.clone(), ScalarType::B32)
            } else {
                Call::new("warp_vote_pred").dst(d.op.clone(), ScalarType::Pred)
            };
            call.src(a.op.clone(), ScalarType::Pred)
                .src(mask, ScalarType::B32)
                .imm(kind | ((a.neg as u64) << 4))
                .build()
        }
        "match" => {
            // match.any.sync.b32 d, a, membermask;  match.all.sync.b64 d{|p}, a, membermask;  (either order of .sync)
            let kind = pick(m, &["any", "all"])?;
            let wide = m.contains(&"b64");
            let [d, a, mm] = ops else { return err("match operands") };
            let t = if wide { ScalarType::B64 } else { ScalarType::B32 };
            let w = if wide { 64 } else { 32 };
            let mut call = Call::new(if kind == 0 { format!("warp_match_any_{w}") } else { format!("warp_match_all_{w}") });
            if kind == 1 {
                if !d.sink {
                    call = call.dst(d.op.clone(), ScalarType::B32);
                }
                match &d.pair {
                    Some(ast::ParsedOperand::Reg(r)) if *r != "_" => {
                        call = call.dst2(ast::ParsedOperand::Reg(*r), ScalarType::Pred);
                    }
                    _ => call = Call::new(format!("warp_match_all_d_{w}")).dst(d.op.clone(), ScalarType::B32),
                }
                if d.sink {
                    // only the predicate
                    let Some(p) = d.pair.clone() else { return err("match.all operands") };
                    return Call::new(format!("warp_match_all_pred_{w}"))
                        .dst(p, ScalarType::Pred)
                        .src(a.op.clone(), t)
                        .src(mm.op.clone(), ScalarType::B32)
                        .build();
                }
            } else {
                call = call.dst(d.op.clone(), ScalarType::B32);
            }
            call.src(a.op.clone(), t).src(mm.op.clone(), ScalarType::B32).build()
        }
        "elect" => {
            // elect.sync d|p, membermask;  elect.sync _|p, membermask;
            let [d, mm] = ops else { return err("elect operands") };
            let p = d.pair.clone();
            match (d.sink, p) {
                (false, Some(p)) => Call::new("warp_elect")
                    .dst(d.op.clone(), ScalarType::B32)
                    .dst2(p, ScalarType::Pred)
                    .src(mm.op.clone(), ScalarType::B32)
                    .build(),
                (true, Some(p)) => Call::new("warp_elect_pred")
                    .dst(p, ScalarType::Pred)
                    .src(mm.op.clone(), ScalarType::B32)
                    .build(),
                (false, None) => Call::new("warp_elect_lane")
                    .dst(d.op.clone(), ScalarType::B32)
                    .src(mm.op.clone(), ScalarType::B32)
                    .build(),
                _ => err("elect operands"),
            }
        }
        "redux" => {
            // redux.sync.op{.abs}{.NaN}.type d, a, membermask;
            let op = pick(m, &["add", "min", "max", "and", "or", "xor"])?;
            let ty = pick(m, &["u32", "s32", "b32", "f32"])?;
            let [d, a, mm] = ops else { return err("redux operands") };
            let ctrl = op
                | (ty << 4)
                | ((m.contains(&"abs") as u64) << 8)
                | ((m.contains(&"NaN") as u64) << 9)
                | ((m.contains(&"ftz") as u64) << 10);
            Call::new("warp_redux")
                .dst(d.op.clone(), ScalarType::B32)
                .src(a.op.clone(), ScalarType::B32)
                .src(mm.op.clone(), ScalarType::B32)
                .imm(ctrl)
                .build()
        }
        "coll_test" => {
            let kind = pick(m, &["sum", "first"])?;
            let [d, a, mm] = ops else { return err("coll_test operands") };
            Call::new("warp_coll_test")
                .dst(d.op.clone(), ScalarType::B32)
                .src(a.op.clone(), ScalarType::B32)
                .src(mm.op.clone(), ScalarType::B32)
                .imm(kind)
                .build()
        }
        _ => None,
    }
}
