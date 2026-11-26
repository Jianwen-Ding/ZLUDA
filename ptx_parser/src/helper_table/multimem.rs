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

pub(super) fn lower<'i>(_opcode: &str, base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
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
