//! multimem.ld_reduce / st / red over the windows of the harness directive .multimem (an address of the window stands
//! for the same offset in each member buffer).  Sail: ptx_mm.sail (exec_mm, mmRed).  Helpers:
//! ptx/lib/metal_impl/multimem.inc.
use super::*;

/// cvt.inc type codes of a multimem type: (lane code, lanes per element, element bytes)
fn mm_type(t: &str) -> Option<(u64, u64, u64)> {
    Some(match t {
        "f64" => (0, 1, 8),
        "f32" => (1, 1, 4),
        "f16" => (2, 1, 2),
        "bf16" => (3, 1, 2),
        "f16x2" => (2, 2, 4),
        "bf16x2" => (3, 2, 4),
        "e4m3" => (5, 1, 1),
        "e5m2" => (6, 1, 1),
        "e4m3x2" => (5, 2, 2),
        "e5m2x2" => (6, 2, 2),
        "e4m3x4" => (5, 4, 4),
        "e5m2x4" => (6, 4, 4),
        "u32" | "b32" => (18, 1, 4),
        "u64" | "b64" => (19, 1, 8),
        "s32" => (22, 1, 4),
        "s64" => (23, 1, 8),
        _ => return None,
    })
}

fn bits_of(eb: u64) -> ScalarType {
    match eb {
        8 => ScalarType::B64,
        2 => ScalarType::B16,
        1 => ScalarType::B8,
        _ => ScalarType::B32,
    }
}

/// the address operand `[a]` of a shared-memory object, `[a+o]` included
fn shared_addr<'i>(h: &HOp<'i>) -> Option<Op<'i>> {
    if h.addr && h.items.is_empty() {
        Some(h.op.clone())
    } else {
        None
    }
}

fn pair<'i>(a: &Op<'i>, b: &Op<'i>) -> Option<Op<'i>> {
    let item = |o: &Op<'i>| match o {
        ast::ParsedOperand::Reg(r) => Some(Some(ast::RegOrImmediate::Reg(*r))),
        ast::ParsedOperand::Imm(i) => Some(Some(ast::RegOrImmediate::Imm(*i))),
        _ => None,
    };
    Some(ast::ParsedOperand::VecPack(vec![item(a)?, item(b)?]))
}

/// fabric.try_get / try_put / try_red .tensor: [tmap, le, off{, counter}, {dims}, {coords}] (.override::global_address
/// with the endpoint's device address, .override::global_dim)
fn fabric_tensor<'i>(m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    let elems = |o: &Op<'i>| match o {
        ast::ParsedOperand::VecPack(v) => v.clone(),
        _ => Vec::new(),
    };
    let tmv = |h: &HOp<'i>| -> Option<(Op<'i>, Op<'i>, Op<'i>, usize, Op<'i>)> {
        let (le, off, ds, cs) = match &h.items[..] {
            [le, off, ds, cs] => (le, off, ds, cs),
            _ => return None,
        };
        let n = elems(cs).len();
        let mut all = elems(cs);
        all.extend(elems(ds));
        if n == 0 || n > 4 || all.len() > 8 {
            return None;
        }
        Some((h.op.clone(), le.clone(), off.clone(), n, ast::ParsedOperand::VecPack(all)))
    };
    if m.contains(&"try_get") {
        let [d, t, bar, ..] = ops else { return err("fabric.try_get.tensor operands") };
        let (Some(d), Some(b), Some((tm, le, off, n, v))) = (shared_addr(d), shared_addr(bar), tmv(t)) else {
            return err("fabric.try_get.tensor operands");
        };
        if n > 3 {
            return err("fabric.try_get.tensor rank");
        }
        return Call::new(format!("fabric_tget{n}"))
            .src_space(d, ScalarType::B8, StateSpace::Shared)
            .src(tm, ScalarType::U64)
            .src(le, ScalarType::U32)
            .src(off, ScalarType::B64)
            .src_vec(v, 8, ScalarType::B32)
            .src_space(b, ScalarType::B64, StateSpace::Shared)
            .build();
    }
    // [tmap, le, off, {dims}, {coords}], [src], size, [mbar]: the store, then the mbarrier's complete-tx (16B units)
    let [t, src, _sz, _bar, ..] = ops else { return err("fabric.try_put.tensor operands") };
    let (Some(s), Some((tm, le, off, n, v))) = (shared_addr(src), tmv(t)) else { return err("fabric.try_put.tensor operands") };
    let op = if m.contains(&"try_red") {
        match ["add", "min", "max", "inc", "dec", "and", "or", "xor"].iter().position(|x| m.contains(x)) {
            Some(i) => i as u64 + 1,
            None => return err("fabric.try_red.tensor operation"),
        }
    } else {
        0
    };
    Call::new("fabric_tput")
        .src(tm, ScalarType::U64)
        .src(le, ScalarType::U32)
        .src(off, ScalarType::B64)
        .src_vec(v, 8, ScalarType::B32)
        .src_space(s, ScalarType::B8, StateSpace::Shared)
        .imm(op | (n as u64) << 4 | (m.contains(&"multimem") as u64) << 7)
        .build()
}

/// fabric.* on the logical endpoints of the harness directive .fabric (Sail: ptx_mm.sail exec_fabric)
fn fabric<'i>(m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    if m.contains(&"submit") || m.contains(&"wait") {
        return Call::new("mem_nop").build();
    }
    if m.contains(&"tensor") {
        return fabric_tensor(m, ops);
    }
    let u16b = m.iter().any(|x| x.ends_with("complete_tx::16B"));
    let mmc = m.contains(&"multimem");
    let sh = |h: &HOp<'i>| shared_addr(h);
    let t = m.iter().rev().copied().find(|x| mm_type(x).is_some()).unwrap_or("b32");
    let (code, _, eb) = mm_type(t).unwrap_or((18, 1, 4));
    let aop = ["add", "min", "max", "and", "or", "xor", "exch", "cas", "inc", "dec"].iter().position(|x| m.contains(x)).unwrap_or(0) as u64;
    if m.contains(&"try_get") {
        let [d, le, n, bar, ..] = ops else { return err("fabric.try_get operands") };
        let (Some(d), Some(b), [off]) = (sh(d), sh(bar), &le.items[..]) else { return err("fabric.try_get addresses") };
        return Call::new("fabric_get")
            .src_space(d, ScalarType::B8, StateSpace::Shared)
            .src(le.op.clone(), ScalarType::U32)
            .src(off.clone(), ScalarType::B64)
            .src(n.op.clone(), ScalarType::U32)
            .src_space(b, ScalarType::B64, StateSpace::Shared)
            .build();
    }
    if m.contains(&"try_pullred") {
        let [d, le, n, bar, ..] = ops else { return err("fabric.try_pullred operands") };
        let (Some(d), Some(b), [off]) = (sh(d), sh(bar), &le.items[..]) else { return err("fabric.try_pullred addresses") };
        let op = ["add", "min", "max", "and", "or", "xor"].iter().position(|x| m.contains(x)).unwrap_or(0) as u64;
        let ctrl = op | code << 3 | 1 << 8 | code << 11 | eb << 16 | 1 << 20;
        return Call::new("fabric_pullred")
            .src_space(d, ScalarType::B8, StateSpace::Shared)
            .src(le.op.clone(), ScalarType::U32)
            .src(off.clone(), ScalarType::B64)
            .src(n.op.clone(), ScalarType::U32)
            .src_space(b, ScalarType::B64, StateSpace::Shared)
            .imm(ctrl)
            .build();
    }
    if m.contains(&"try_put") || m.contains(&"try_red") {
        let [le, src, n, bar, rest @ ..] = ops else { return err("fabric.try_put operands") };
        let (Some(s), Some(b)) = (sh(src), sh(bar)) else { return err("fabric.try_put addresses") };
        let (off, cn) = match &le.items[..] {
            [off] => (off.clone(), None),
            [off, cn] => (off.clone(), Some(cn.clone())),
            _ => return err("fabric endpoint operand"),
        };
        let mask = if m.contains(&"cp_mask") { rest.first().map(|x| x.op.clone()) } else { None };
        // ctrl: 0-3 atom op, 4-8 type, 9-12 element bytes, 13 multimem, 14 mask, 15 counted, 16 red, 17 tx in 16B
        let ctrl = aop | code << 4 | eb << 9 | (mmc as u64) << 13 | (mask.is_some() as u64) << 14 | (cn.is_some() as u64) << 15
            | (m.contains(&"try_red") as u64) << 16 | (u16b as u64) << 17;
        return Call::new("fabric_put")
            .src_vec(pair(&le.op, &mask.unwrap_or(imm(0)))?, 2, ScalarType::B32)
            .src_vec(pair(&off, &cn.unwrap_or(imm(0)))?, 2, ScalarType::B64)
            .src_space(s, ScalarType::B8, StateSpace::Shared)
            .src(n.op.clone(), ScalarType::U32)
            .src_space(b, ScalarType::B64, StateSpace::Shared)
            .imm(ctrl)
            .build();
    }
    if m.contains(&"try_atom") {
        let [le, dd, ss, bar, ..] = ops else { return err("fabric.try_atom operands") };
        let (Some(d), Some(s), Some(b), [off]) = (sh(dd), sh(ss), sh(bar), &le.items[..]) else { return err("fabric.try_atom addresses") };
        let ctrl = aop | code << 4 | eb << 9;
        return Call::new("fabric_atom")
            .src(le.op.clone(), ScalarType::U32)
            .src(off.clone(), ScalarType::B64)
            .src_space(d, ScalarType::B8, StateSpace::Shared)
            .src_space(s, ScalarType::B8, StateSpace::Shared)
            .src_space(b, ScalarType::B64, StateSpace::Shared)
            .imm(ctrl)
            .build();
    }
    None
}

pub(super) fn lower<'i>(_opcode: &str, base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    if base == "fabric" {
        return fabric(m, ops);
    }
    if base != "multimem" || m.contains(&"cp") {
        return None;
    }
    let t = *m.iter().rev().find(|x| mm_type(x).is_some())?;
    let (code, lanes, eb) = mm_type(t)?;
    let op = ["add", "min", "max", "and", "or", "xor"].iter().position(|x| m.contains(x)).unwrap_or(0) as u64;
    let acc = if m.contains(&"acc::f32") {
        1
    } else if m.contains(&"acc::f16") {
        2
    } else {
        code
    };
    let n: u64 = if m.contains(&"v8") {
        8
    } else if m.contains(&"v4") {
        4
    } else if m.contains(&"v2") {
        2
    } else {
        1
    };
    if n > 4 {
        return err("multimem .v8");
    }
    // ctrl: 0-2 op, 3-7 lane type, 8-10 lanes, 11-15 accumulation type, 16-19 element bytes, 20-23 vector length
    let ctrl = op | code << 3 | lanes << 8 | acc << 11 | eb << 16 | n << 20;
    // element registers: 8-bit elements live in 16-bit registers or wider
    let et = bits_of(eb.max(2));
    let wide = eb == 8;
    if m.contains(&"ld_reduce") {
        let [d, a] = ops else { return err("multimem.ld_reduce operands") };
        let call = if n == 1 {
            let dop = match &d.op {
                ast::ParsedOperand::VecPack(v) if v.len() == 1 => match &v[0] {
                    Some(ast::RegOrImmediate::Reg(r)) => ast::ParsedOperand::Reg(*r),
                    _ => return err("multimem destination"),
                },
                o => o.clone(),
            };
            Call::new(format!("mm_ld1_{}", if wide { 64 } else if eb == 2 { 16 } else { 32 })).dst(dop, bits_of(eb.max(2)))
        } else {
            let mut o = d.op.clone();
            if let ast::ParsedOperand::VecPack(v) = &mut o {
                while v.len() < 4 {
                    v.push(None);
                }
            }
            Call::new(format!("mm_ld4_{}", if wide { 64 } else if eb == 2 { 16 } else { 32 })).dst_vec(o, 4, et)
        };
        return call.src(a.op.clone(), ScalarType::B64).imm(ctrl).build();
    }
    if m.contains(&"st") || m.contains(&"red") {
        let [a, v] = ops else { return err("multimem.st / red operands") };
        let vv = match &v.op {
            ast::ParsedOperand::VecPack(_) => v.op.clone(),
            ast::ParsedOperand::Reg(r) => ast::ParsedOperand::VecPack(vec![Some(ast::RegOrImmediate::Reg(*r))]),
            ast::ParsedOperand::Imm(i) => ast::ParsedOperand::VecPack(vec![Some(ast::RegOrImmediate::Imm(*i))]),
            o => o.clone(),
        };
        let red = m.contains(&"red") as u64;
        return Call::new(format!("mm_st_{}", if wide { 64 } else if eb == 2 { 16 } else { 32 }))
            .src(a.op.clone(), ScalarType::B64)
            .src_vec(vv, 4, et)
            .imm(ctrl | red << 24)
            .build();
    }
    None
}

/// fabric.try_put / try_red .tensor: the mbarrier's complete-tx after the store (size / 16)
pub(super) fn extra<'i>(base: &str, m: &[&str], ops: &[HOp<'i>]) -> Vec<ast::Instruction<ParsedOperandStr<'i>>> {
    if base != "fabric" || !m.contains(&"tensor") || m.contains(&"try_get") {
        return Vec::new();
    }
    let [_, _, sz, bar, ..] = ops else { return Vec::new() };
    let Some(b) = shared_addr(bar) else { return Vec::new() };
    let n = match &sz.op {
        ast::ParsedOperand::Imm(ast::ImmediateValue::U64(v)) => imm(v / 16),
        ast::ParsedOperand::Imm(ast::ImmediateValue::S64(v)) => imm((*v as u64) / 16),
        _ => return Vec::new(),
    };
    match Call::new("fabric_tx").src_space(b, ScalarType::B64, StateSpace::Shared).src(n, ScalarType::U32).build() {
        Some(Ok(i)) => vec![i],
        _ => Vec::new(),
    }
}
