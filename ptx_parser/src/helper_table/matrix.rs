//! warp and warpgroup matrix instructions (mma, ldmatrix, stmatrix, movmatrix, wmma, wgmma, tcgen05).
//!
//! The warp-level matrix instructions are collective: every helper (ptx/lib/metal_impl/matrix.inc)
//! gathers the fragments of the whole warp with simd shuffles and computes its own lane's result
//! exactly as PTX-formalization's Sail model does (sail/ptx/ptx_mma.sail, ptx_ldm.sail).
//! Fragments are passed as fixed-width vectors (8 x b32, or b64 for .f64), padded with zero
//! immediates (sources) or sinks (destinations), so a few helper signatures serve every shape;
//! the shape, types and modifiers are packed into control words that fold after inlining.
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

fn imm_val(e: &Option<ast::RegOrImmediate<&str>>) -> Option<u64> {
    match e {
        Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::U64(v))) => Some(*v),
        Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::S64(v))) => Some(*v as u64),
        _ => None,
    }
}

fn op_imm(op: &Op) -> Option<u64> {
    match op {
        ast::ParsedOperand::Imm(ast::ImmediateValue::U64(v)) => Some(*v),
        ast::ParsedOperand::Imm(ast::ImmediateValue::S64(v)) => Some(*v as u64),
        _ => None,
    }
}

/// a vector operand padded to `n` elements: zeros for sources, sinks for destinations
fn pad<'i>(mut v: VecElems<'i>, n: usize, dst: bool) -> Result<Op<'i>, String> {
    if v.len() > n {
        return Err(format!("vector operand with {} elements (max {n})", v.len()));
    }
    while v.len() < n {
        v.push(if dst { None } else { Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::U64(0))) });
    }
    Ok(ast::ParsedOperand::VecPack(v))
}

impl<'i> Call<'i> {
    fn vdst(mut self, op: Op<'i>, n: u8, t: ScalarType) -> Self {
        self.data = self.data.with_type(0, ast::Type::Vector(n, t), StateSpace::Reg);
        self.slots[0] = Some(op);
        self
    }
    fn vsrc(mut self, op: Op<'i>, n: u8, t: ScalarType) -> Self {
        let i = self.next_src;
        self.data = self.data.with_type(i, ast::Type::Vector(n, t), StateSpace::Reg);
        self.slots[i] = Some(op);
        self.next_src += 1;
        self
    }
}

fn shape(m: &[&str]) -> Option<(u32, u32, u32)> {
    for s in m {
        let Some(r) = s.strip_prefix('m') else { continue };
        let Some((mm, r)) = r.split_once('n') else { continue };
        let Some((nn, kk)) = r.split_once('k') else { continue };
        if let (Ok(a), Ok(b), Ok(c)) = (mm.parse(), nn.parse(), kk.parse()) {
            return Some((a, b, c));
        }
    }
    None
}

const ELEM_TYPES: &[&str] = &[
    "f16", "bf16", "tf32", "f32", "f64", "e4m3", "e5m2", "e2m3", "e3m2", "e2m1", "s8", "u8", "s4", "u4",
    "b1", "s32", "ue8m0", "ue4m3",
];

/// A/B element type code (matrix.inc: MX_*)
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
        "e2m1p" => 8,
        "s8" => 9,
        "u8" => 10,
        "s4" => 11,
        "u4" => 12,
        "b1" => 13,
        _ => return None,
    })
}

/// accumulator type code: f16 0, f32 1, s32 2, f64 3
fn cd_code(t: &str) -> Option<u32> {
    Some(match t {
        "f16" => 0,
        "f32" => 1,
        "s32" => 2,
        "f64" => 3,
        _ => return None,
    })
}

fn ew(t: &str) -> u32 {
    match t {
        "f16" | "bf16" => 16,
        "tf32" | "f32" | "s32" => 32,
        "f64" => 64,
        "s4" | "u4" | "e2m1p" => 4,
        "b1" => 1,
        _ => 8,
    }
}

fn rnd_code(m: &[&str]) -> u32 {
    if m.contains(&"rz") {
        1
    } else if m.contains(&"rm") {
        2
    } else if m.contains(&"rp") {
        3
    } else {
        0
    }
}

fn layouts(m: &[&str]) -> (bool, bool) {
    let l: Vec<bool> = m.iter().filter(|x| **x == "row" || **x == "col").map(|x| *x == "col").collect();
    (l.first().copied().unwrap_or(false), l.get(1).copied().unwrap_or(false))
}

/// wmRegs of ptx_mma.sail: registers of a wmma fragment
fn wm_regs(mat: &str, s: (u32, u32, u32), t: &str) -> u32 {
    let (sm, sn, _) = s;
    let shape = format!("m{}n{}k{}", s.0, s.1, s.2);
    let mat = if mat == "d" { "c" } else { mat };
    if mat == "c" {
        if t == "f16" {
            4
        } else if sm == 8 && sn == 8 {
            2
        } else {
            8
        }
    } else if t == "f16" {
        8
    } else if t == "bf16" {
        if mat == "a" {
            match shape.as_str() {
                "m16n16k16" => 4,
                "m8n32k16" => 2,
                _ => 8,
            }
        } else {
            match shape.as_str() {
                "m16n16k16" => 4,
                "m8n32k16" => 8,
                _ => 2,
            }
        }
    } else if t == "tf32" {
        4
    } else if t == "f64" {
        1
    } else if t == "s8" || t == "u8" {
        if mat == "a" {
            match shape.as_str() {
                "m16n16k16" => 2,
                "m8n32k16" => 1,
                _ => 4,
            }
        } else {
            match shape.as_str() {
                "m16n16k16" => 2,
                "m8n32k16" => 4,
                _ => 1,
            }
        }
    } else {
        1
    }
}

fn lower_mma<'i>(m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    let Some((sm, sn, sk)) = shape(m) else { return err("mma shape") };
    let tys: Vec<&str> = m.iter().copied().filter(|x| ELEM_TYPES.contains(x)).collect();
    let sparse = m.iter().any(|x| *x == "sp" || *x == "sp::ordered_metadata");
    let block = m.contains(&"block_scale");
    let mx4 = m.contains(&"kind::mxf4") || m.contains(&"kind::mxf4nvf4");
    if (block && tys.len() != 5) || (!block && tys.len() != 4) {
        return err("mma types");
    }
    let (dt, mut at, mut bt, ct) = (tys[0], tys[1], tys[2], tys[3]);
    if mx4 {
        at = "e2m1p";
        bt = "e2m1p";
    }
    let nfixed = 4 + if sparse { 2 } else { 0 } + if block { 4 } else { 0 };
    if ops.len() != nfixed {
        return err("mma operand count");
    }
    let (acol, bcol) = layouts(m);
    let get = |i: usize| elems(&ops[i].op).ok_or_else(|| "mma operand".to_string());
    let (d, a, b, c) = match (get(0), get(1), get(2), get(3)) {
        (Ok(d), Ok(a), Ok(b), Ok(c)) => (d, a, b, c),
        _ => return err("mma operands"),
    };
    let mut idx = 4;
    let mut x: VecElems = vec![None; 3];
    let zero = || Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::U64(0)));
    x[0] = zero();
    x[1] = zero();
    x[2] = zero();
    let mut sel = 0u64;
    if sparse {
        let e = match elems(&ops[4].op) {
            Some(v) if v.len() == 1 => v[0].clone(),
            _ => return err("mma.sp metadata"),
        };
        x[0] = e;
        sel = match op_imm(&ops[5].op) {
            Some(v) => v,
            None => return err("mma.sp selector must be an immediate"),
        };
        idx = 6;
    }
    let (mut bida, mut tida, mut bidb, mut tidb) = (0u64, 0u64, 0u64, 0u64);
    if block {
        let sa = match elems(&ops[idx].op) {
            Some(v) if v.len() == 1 => v[0].clone(),
            _ => return err("mma scale a"),
        };
        let sb = match elems(&ops[idx + 2].op) {
            Some(v) if v.len() == 1 => v[0].clone(),
            _ => return err("mma scale b"),
        };
        let sela = elems(&ops[idx + 1].op).unwrap_or_default();
        let selb = elems(&ops[idx + 3].op).unwrap_or_default();
        if sela.len() != 2 || selb.len() != 2 {
            return err("mma scale selectors");
        }
        match (imm_val(&sela[0]), imm_val(&sela[1]), imm_val(&selb[0]), imm_val(&selb[1])) {
            (Some(a0), Some(a1), Some(b0), Some(b1)) => {
                bida = a0;
                tida = a1;
                bidb = b0;
                tidb = b1;
            }
            _ => return err("mma scale selectors must be immediates"),
        }
        x[1] = sa;
        x[2] = sb;
    }
    let kind: u32 = if at == "f16" && (sm, sn, sk) == (8, 8, 4) {
        1
    } else if sparse && block {
        5
    } else if sparse {
        3
    } else if block {
        4
    } else {
        0
    };
    let nb: u32 = if m.contains(&"scale_vec::4X") {
        4
    } else if m.contains(&"scale_vec::2X") || (m.contains(&"kind::mxf4") && !m.contains(&"scale_vec::1X")) {
        2
    } else {
        1
    };
    let bop: u32 = if m.contains(&"xor") {
        1
    } else if m.contains(&"and") {
        2
    } else {
        0
    };
    let (Some(dc), Some(cc)) = (cd_code(dt), cd_code(ct)) else { return err("mma accumulator type") };
    let f64_ = at == "f64";
    let (ac, bc) = if f64_ {
        (15, 15)
    } else {
        match (ab_code(at), ab_code(bt)) {
            (Some(a), Some(b)) => (a, b),
            _ => return err("mma element type"),
        }
    };
    let e43 = block && tys[4] == "ue4m3";
    let ctl0 = sk | (sm << 9) | (sn << 15) | (ac << 21) | (bc << 25) | (kind << 29);
    let ctl1 = cc
        | (dc << 2)
        | (rnd_code(m) << 4)
        | ((m.contains(&"satfinite") as u32) << 6)
        | (bop << 7)
        | ((acol as u32) << 9)
        | ((bcol as u32) << 10)
        | (nb << 11)
        | ((e43 as u32) << 14)
        | ((bida as u32 & 3) << 15)
        | ((tida as u32 & 3) << 17)
        | ((bidb as u32 & 3) << 19)
        | ((tidb as u32 & 3) << 21)
        | ((sel as u32 & 3) << 23);
    let na = a.len() as u64;
    let nbr = b.len() as u64;
    let nc = c.len() as u64;
    let nd = d.len() as u64;
    let ctl2 = na | (nbr << 4) | (nc << 8) | (nd << 12);
    x.push(Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::U64(ctl2))));
    let r = (|| -> Result<_, String> {
        if f64_ {
            Ok(Call::new("mma_f64")
                .vdst(pad(d, 4, true)?, 4, ScalarType::B64)
                .vsrc(pad(a, 8, false)?, 8, ScalarType::B64)
                .vsrc(pad(b, 4, false)?, 4, ScalarType::B64)
                .vsrc(pad(c, 4, false)?, 4, ScalarType::B64)
                .imm(ctl0 as u64)
                .imm(ctl1 as u64))
        } else {
            Ok(Call::new("mma_gen")
                .vdst(pad(d, 8, true)?, 8, ScalarType::B32)
                .vsrc(pad(a, 8, false)?, 8, ScalarType::B32)
                .vsrc(pad(b, 8, false)?, 8, ScalarType::B32)
                .vsrc(pad(c, 8, false)?, 8, ScalarType::B32)
                .vsrc(ast::ParsedOperand::VecPack(x), 4, ScalarType::B32)
                .imm(ctl0 as u64)
                .imm(ctl1 as u64))
        }
    })();
    match r {
        Ok(c) => c.build(),
        Err(e) => err(e),
    }
}

fn space_of(m: &[&str]) -> (StateSpace, &'static str) {
    if m.iter().any(|x| x.starts_with("shared")) {
        (StateSpace::Shared, "shared")
    } else if m.contains(&"global") {
        (StateSpace::Global, "global")
    } else {
        (StateSpace::Generic, "generic")
    }
}

fn lower_wmma<'i>(m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    let Some(s) = shape(m) else { return err("wmma shape") };
    let (sm, sn, sk) = s;
    let tys: Vec<&str> = m.iter().copied().filter(|x| ELEM_TYPES.contains(x)).collect();
    let col = m.contains(&"col");
    if m.first() == Some(&"load") || m.first() == Some(&"store") {
        let load = m[0] == "load";
        let mat = m.get(1).copied().unwrap_or("a");
        let mat = if mat == "d" { "c" } else { mat };
        if tys.len() != 1 {
            return err("wmma type");
        }
        let t = tys[0];
        let regs = wm_regs(mat, s, t);
        let e = ew(t);
        let per = if e == 64 { 1 } else { 32 / e };
        let nel = regs * per;
        let matc = match mat {
            "a" => 0,
            "b" => 1,
            _ => 2,
        };
        let (pi, vi) = if load { (1, 0) } else { (0, 1) };
        if ops.len() < 2 || ops.len() > 3 || !ops[pi].addr {
            return err("wmma operands");
        }
        let stride = if ops.len() == 3 {
            ops[2].op.clone()
        } else {
            let st = match mat {
                "a" => if col { sm } else { sk },
                "b" => if col { sk } else { sn },
                _ => if col { sm } else { sn },
            };
            imm(st as u64)
        };
        let ctl = matc | ((col as u32) << 2) | (sm << 3) | (sn << 9) | (sk << 15) | (e << 24);
        let (space, sname) = space_of(m);
        let Some(v) = elems(&ops[vi].op) else { return err("wmma fragment") };
        if v.len() != regs as usize {
            return err(format!("wmma fragment of {} registers, expected {regs}", v.len()));
        }
        let (n, bt, suffix) = if e == 64 { (4u8, ScalarType::B64, "_f64") } else { (8u8, ScalarType::B32, "") };
        let r = (|| -> Result<_, String> {
            let mut c = Call::new(format!("wmma_{}{}_{}", if load { "load" } else { "store" }, suffix, sname));
            if load {
                c = c.vdst(pad(v, n as usize, true)?, n, bt).src_space(ops[pi].op.clone(), ScalarType::B8, space);
            } else {
                c = c.src_space(ops[pi].op.clone(), ScalarType::B8, space).vsrc(pad(v, n as usize, false)?, n, bt);
            }
            Ok(c.src(stride, ScalarType::U32).imm(ctl as u64).imm(nel as u64))
        })();
        return match r {
            Ok(c) => c.build(),
            Err(e) => err(e),
        };
    }
    if m.first() != Some(&"mma") {
        return err("wmma form");
    }
    // wmma.mma: dtype.ctype (f16 A/B) or dtype.atype.btype.ctype
    let (dt, at, bt, ct) = match tys.len() {
        2 => (tys[0], "f16", "f16", tys[1]),
        4 => (tys[0], tys[1], tys[2], tys[3]),
        _ => return err("wmma.mma types"),
    };
    if ops.len() != 4 {
        return err("wmma.mma operands");
    }
    let get = |i: usize| elems(&ops[i].op).ok_or_else(|| "wmma operand".to_string());
    let (d, a, b, c) = match (get(0), get(1), get(2), get(3)) {
        (Ok(d), Ok(a), Ok(b), Ok(c)) => (d, a, b, c),
        _ => return err("wmma operands"),
    };
    let bop: u32 = if m.contains(&"xor") {
        1
    } else if m.contains(&"and") {
        2
    } else {
        0
    };
    let (Some(dc), Some(cc)) = (cd_code(dt), cd_code(ct)) else { return err("wmma accumulator type") };
    let f64_ = at == "f64";
    let (ac, bc) = if f64_ {
        (15, 15)
    } else {
        match (ab_code(at), ab_code(bt)) {
            (Some(a), Some(b)) => (a, b),
            _ => return err("wmma element type"),
        }
    };
    let kind = 2u32;
    let ctl0 = sk | (sm << 9) | (sn << 15) | (ac << 21) | (bc << 25) | (kind << 29);
    let ctl1 = cc | (dc << 2) | (rnd_code(m) << 4) | ((m.contains(&"satfinite") as u32) << 6) | (bop << 7);
    let ctl2 = a.len() as u64 | ((b.len() as u64) << 4) | ((c.len() as u64) << 8) | ((d.len() as u64) << 12);
    let zero = || Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::U64(0)));
    let x: VecElems = vec![zero(), zero(), zero(), Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::U64(ctl2)))];
    let r = (|| -> Result<_, String> {
        if f64_ {
            Ok(Call::new("mma_f64")
                .vdst(pad(d, 4, true)?, 4, ScalarType::B64)
                .vsrc(pad(a, 8, false)?, 8, ScalarType::B64)
                .vsrc(pad(b, 4, false)?, 4, ScalarType::B64)
                .vsrc(pad(c, 4, false)?, 4, ScalarType::B64)
                .imm(ctl0 as u64)
                .imm(ctl1 as u64))
        } else {
            Ok(Call::new("mma_gen")
                .vdst(pad(d, 8, true)?, 8, ScalarType::B32)
                .vsrc(pad(a, 8, false)?, 8, ScalarType::B32)
                .vsrc(pad(b, 8, false)?, 8, ScalarType::B32)
                .vsrc(pad(c, 8, false)?, 8, ScalarType::B32)
                .vsrc(ast::ParsedOperand::VecPack(x), 4, ScalarType::B32)
                .imm(ctl0 as u64)
                .imm(ctl1 as u64))
        }
    })();
    match r {
        Ok(c) => c.build(),
        Err(e) => err(e),
    }
}

fn num_of(m: &[&str]) -> u32 {
    if m.contains(&"x4") {
        4
    } else if m.contains(&"x2") {
        2
    } else {
        1
    }
}

fn lower_ldst<'i>(base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    let n = num_of(m);
    let trans = m.contains(&"trans");
    let (space, sname) = space_of(m);
    let sname = if sname == "global" { "generic" } else { sname };
    let space = if space == StateSpace::Global { StateSpace::Generic } else { space };
    match base {
        "ldmatrix" => {
            if ops.len() != 2 || !ops[1].addr {
                return err("ldmatrix operands");
            }
            let Some(d) = elems(&ops[0].op) else { return err("ldmatrix destination") };
            // kind: 0 m8n8 b16, 1 m8n16 (k = 0), 2 m16n16 (k = 1)
            let kind = if m.contains(&"m16n16") {
                2
            } else if m.contains(&"m8n16") {
                1
            } else {
                0
            };
            let fmt = if m.contains(&"b6x16_p32") {
                6
            } else if m.contains(&"b4x16_p64") {
                4
            } else if m.contains(&"s4") {
                5
            } else {
                0
            };
            let ctl = n | ((trans as u32) << 4) | (kind << 5) | (fmt << 8);
            let r = (|| -> Result<_, String> {
                Ok(Call::new(format!("ldmatrix_gen_{sname}"))
                    .vdst(pad(d, 8, true)?, 8, ScalarType::B32)
                    .src_space(ops[1].op.clone(), ScalarType::B8, space)
                    .imm(ctl as u64))
            })();
            match r {
                Ok(c) => c.build(),
                Err(e) => err(e),
            }
        }
        "stmatrix" => {
            if ops.len() != 2 || !ops[0].addr {
                return err("stmatrix operands");
            }
            let Some(v) = elems(&ops[1].op) else { return err("stmatrix source") };
            let kind = if m.contains(&"m16n8") { 1 } else { 0 };
            let ctl = n | ((trans as u32) << 4) | (kind << 5);
            let r = (|| -> Result<_, String> {
                Ok(Call::new(format!("stmatrix_gen_{sname}"))
                    .src_space(ops[0].op.clone(), ScalarType::B8, space)
                    .vsrc(pad(v, 4, false)?, 4, ScalarType::B32)
                    .imm(ctl as u64))
            })();
            match r {
                Ok(c) => c.build(),
                Err(e) => err(e),
            }
        }
        _ => {
            // movmatrix.sync.aligned.m8n8.trans.b16 d, a
            if ops.len() != 2 {
                return err("movmatrix operands");
            }
            Call::new("movmatrix_m8n8_trans_b16")
                .dst(ops[0].op.clone(), ScalarType::B32)
                .src(ops[1].op.clone(), ScalarType::B32)
                .build()
        }
    }
}

#[allow(unused_variables)]
pub(super) fn lower<'i>(opcode: &str, base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    match base {
        "mma" => lower_mma(m, ops),
        "wmma" => lower_wmma(m, ops),
        "ldmatrix" | "stmatrix" | "movmatrix" => lower_ldst(base, m, ops),
        _ => None,
    }
}
