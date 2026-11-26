//! warpgroup and tensor-memory matrix instructions: wgmma.*, tcgen05.*.
//!
//! The helpers (ptx/lib/metal_impl/tensor.inc) follow PTX-formalization's Sail model
//! (sail/ptx/ptx_wg.sail, ptx_tc.sail) and reuse the exact arithmetic of the warp matrix engine
//! (metal_impl/matrix.inc). Asynchronous operations complete at issue, so the fences, commits
//! and waits are no-ops.
//!
//! wgmma.mma_async: every thread of the warpgroup computes its own accumulator elements; the
//! A / B tiles of the shared-memory matrix descriptors are read from threadgroup memory, A
//! fragments in registers and the sparsity metadata come from the lanes of the thread's own
//! warp (simd shuffles). The accumulator is a vector of 8 .. 128 b32 (padded), passed through a
//! per-thread scratch area of the CTA's per-CTA global memory.
use super::*;

type VecElems<'i> = Vec<Option<ast::RegOrImmediate<&'i str>>>;

fn elems<'i>(op: &Op<'i>) -> Option<VecElems<'i>> {
    Some(match op {
        ast::ParsedOperand::VecPack(v) => v.clone(),
        ast::ParsedOperand::Reg(r) => vec![Some(ast::RegOrImmediate::Reg(*r))],
        ast::ParsedOperand::Imm(i) => vec![Some(ast::RegOrImmediate::Imm(*i))],
        _ => return None,
    })
}

fn op_imm(op: &Op) -> Option<i64> {
    match op {
        ast::ParsedOperand::Imm(ast::ImmediateValue::U64(v)) => Some(*v as i64),
        ast::ParsedOperand::Imm(ast::ImmediateValue::S64(v)) => Some(*v),
        _ => None,
    }
}

fn zero<'i>() -> Option<ast::RegOrImmediate<&'i str>> {
    Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::U64(0)))
}

fn immv<'i>(v: u64) -> Option<ast::RegOrImmediate<&'i str>> {
    Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::U64(v)))
}

/// a vector operand padded to `n` elements: zeros for sources, sinks for destinations
fn pad<'i>(mut v: VecElems<'i>, n: usize, dst: bool) -> Result<Op<'i>, String> {
    if v.len() > n {
        return Err(format!("vector operand with {} elements (max {n})", v.len()));
    }
    while v.len() < n {
        v.push(if dst { None } else { zero() });
    }
    Ok(ast::ParsedOperand::VecPack(v))
}

impl<'i> Call<'i> {
    fn tvdst(mut self, op: Op<'i>, n: u8, t: ScalarType) -> Self {
        self.data = self.data.with_type(0, ast::Type::Vector(n, t), StateSpace::Reg);
        self.slots[0] = Some(op);
        self
    }
    fn tvsrc(mut self, op: Op<'i>, n: u8, t: ScalarType) -> Self {
        let i = self.next_src;
        self.data = self.data.with_type(i, ast::Type::Vector(n, t), StateSpace::Reg);
        self.slots[i] = Some(op);
        self.next_src += 1;
        self
    }
}

fn nop<'i>() -> R<'i> {
    Call::new("tensor_nop").build()
}

/// A/B element type code (matrix.inc MX_*)
fn ab_code(t: &str) -> Option<u32> {
    Some(match t {
        "f16" => 0,
        "bf16" => 1,
        "tf32" => 2,
        "e4m3" => 3,
        "e5m2" => 4,
        "e2m3" => 5,
        "e3m2" => 6,
        "e2m1" => 7,
        "s8" => 9,
        "u8" => 10,
        "b1" => 13,
        _ => return None,
    })
}

fn vec_size(n: usize) -> usize {
    [8, 16, 32, 64, 128].into_iter().find(|&s| s >= n).unwrap_or(0)
}

fn lower_wgmma<'i>(m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    if m.first() == Some(&"fence") || m.first() == Some(&"commit_group") || m.first() == Some(&"wait_group") {
        return nop();
    }
    if m.first() != Some(&"mma_async") {
        return None;
    }
    let sp = m.contains(&"sp");
    // wgShape: m64nNkK
    let Some((n, k)) = m.iter().find_map(|s| {
        let r = s.strip_prefix("m64n")?;
        let (nn, kk) = r.split_once('k')?;
        Some((nn.parse::<u32>().ok()?, kk.parse::<u32>().ok()?))
    }) else {
        return err("wgmma.mma_async shape");
    };
    if !(8..=256).contains(&n) || n % 8 != 0 || ![8, 16, 32, 64, 256].contains(&k) {
        return err("wgmma.mma_async: invalid shape");
    }
    let tys: Vec<&str> = m
        .iter()
        .copied()
        .filter(|x| ["f16", "bf16", "tf32", "f32", "e4m3", "e5m2", "s8", "u8", "s32", "b1"].contains(x))
        .collect();
    if tys.len() != 3 {
        return err("wgmma.mma_async types");
    }
    let (dt, at, bt) = (tys[0], tys[1], tys[2]);
    let dc: u32 = match dt {
        "f16" => 0,
        "f32" => 1,
        "s32" => 2,
        _ => return err("wgmma accumulator type"),
    };
    let (Some(ac), Some(bc)) = (ab_code(at), ab_code(bt)) else { return err("wgmma element type") };
    let nfix = if sp { 5 } else { 3 };
    if ops.len() < nfix {
        return err("wgmma.mma_async operands");
    }
    let Some(d) = elems(&ops[0].op) else { return err("wgmma accumulator") };
    let nd = if dc == 0 { n / 4 } else { n / 2 } as usize;
    if d.len() != nd {
        return err(format!("wgmma accumulator of {} registers, expected {nd}", d.len()));
    }
    let aregs = matches!(ops[1].op, ast::ParsedOperand::VecPack(_));
    let rest = &ops[nfix..];
    let nr = rest.len();
    let arg = |i: Option<usize>, dflt: i64| -> Result<i64, String> {
        match i {
            Some(i) if i < nr => op_imm(&rest[i].op).ok_or_else(|| "wgmma immediate operand".to_string()),
            _ => Ok(dflt),
        }
    };
    let r = (|| -> Result<_, String> {
        let sca = arg(Some(1), 1)?;
        let scb = arg(Some(2), 1)?;
        let tra = arg(Some(3), 0)?;
        let trb = arg(if nr >= 4 { Some(nr - 1) } else { None }, 0)?;
        // scale-d: an immediate or a predicate
        let mut x: VecElems = vec![zero(); 8];
        let sdreg = if nr == 0 {
            x[5] = immv(1);
            None
        } else if let Some(v) = op_imm(&rest[0].op) {
            x[5] = immv((v != 0) as u64);
            None
        } else {
            Some(rest[0].op.clone())
        };
        let mut sel = 0u64;
        if sp {
            let e = match elems(&ops[3].op) {
                Some(v) if v.len() == 1 => v[0].clone(),
                _ => return Err("wgmma.sp metadata".into()),
            };
            x[4] = e;
            sel = op_imm(&ops[4].op).ok_or("wgmma.sp selector must be an immediate")? as u64;
        }
        let desc_a = if aregs {
            let a = elems(&ops[1].op).ok_or("wgmma A")?;
            if a.len() > 4 {
                return Err("wgmma A fragment".into());
            }
            for (i, e) in a.into_iter().enumerate() {
                x[i] = e;
            }
            imm(0)
        } else {
            ops[1].op.clone()
        };
        let bop: u32 = if m.contains(&"xor") {
            1
        } else if m.contains(&"and") {
            2
        } else {
            0
        };
        let ctl0 = n
            | (k << 9)
            | (ac << 18)
            | (bc << 22)
            | (dc << 26)
            | ((sp as u32) << 28)
            | ((aregs as u32) << 29)
            | ((m.contains(&"satfinite") as u32) << 30);
        let ctl1 = bop
            | (((sca < 0) as u32) << 2)
            | (((scb < 0) as u32) << 3)
            | (((tra != 0) as u32) << 4)
            | (((trb != 0) as u32) << 5)
            | ((sel as u32 & 3) << 6)
            | ((sdreg.is_some() as u32) << 8);
        let nv = vec_size(nd);
        let mut c = Call::new(format!("wgmma_{nv}{}", if sdreg.is_some() { "_p" } else { "" }))
            .tvdst(pad(d.clone(), nv, true)?, nv as u8, ScalarType::B32)
            .tvsrc(pad(d, nv, false)?, nv as u8, ScalarType::B32)
            .tvsrc(ast::ParsedOperand::VecPack(x), 8, ScalarType::B32)
            .src(desc_a, ScalarType::B64)
            .src(ops[2].op.clone(), ScalarType::B64);
        if let Some(p) = sdreg {
            c = c.src(p, ScalarType::Pred);
        }
        Ok(c.src(imm((ctl1 as u64) << 32 | ctl0 as u64), ScalarType::U64))
    })();
    match r {
        Ok(c) => c.build(),
        Err(e) => err(e),
    }
}

pub(super) fn lower<'i>(_opcode: &str, base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    match base {
        "wgmma" => lower_wgmma(m, ops),
        _ => None,
    }
}
