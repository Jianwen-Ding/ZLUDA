//! Floating-point arithmetic variants: f16 / bf16 / packed / f32x2 / mixed precision add, sub, mul,
//! fma (any rounding, .ftz .sat .relu .satfinite .oob), min / max (.NaN .xorsign .abs, three
//! inputs), alternate floating point x4, half comparisons, f32 div / sqrt with .sat.  Sail:
//! ptx_alu.sail (bin, tern, minmax3), ptx_half.sail (gx_half, exec_half), ptx_mini.sail (exec_altfp),
//! common/ieee.sail.  Helpers: ptx/lib/metal_impl/float.inc.
use super::*;

const FTYPES: [&str; 7] = ["f16", "bf16", "f32", "f64", "f16x2", "bf16x2", "f32x2"];
const ALT4: [&str; 7] = ["e5m2x4", "e4m3x4", "e3m2x4", "e2m3x4", "e2m1x4", "e2m1p4x4", "ue8m0x4"];

/// element type code (cvt.inc T_*), packed flag, container width
fn ftype(t: &str) -> Option<(u64, bool, u32)> {
    Some(match t {
        "f64" => (0, false, 64),
        "f32" => (1, false, 32),
        "f16" => (2, false, 16),
        "bf16" => (3, false, 16),
        "f16x2" => (2, true, 32),
        "bf16x2" => (3, true, 32),
        "f32x2" => (1, true, 64),
        _ => return None,
    })
}

fn bits(w: u32) -> ScalarType {
    match w {
        16 => ScalarType::B16,
        64 => ScalarType::B64,
        _ => ScalarType::B32,
    }
}

/// modifier bits (float.inc): rounding 9-11, .ftz 12, .sat 13, .relu 14, .oob 15, .NaN 16,
/// .xorsign 17, .abs 18, .satfinite 19
fn fmods(mods: &[&str]) -> Result<u64, String> {
    let mut c = 0u64;
    for m in mods {
        c |= match *m {
            "rn" => 0,
            "rz" => 1 << 9,
            "rm" => 2 << 9,
            "rp" => 3 << 9,
            "ftz" => 1 << 12,
            "sat" => 1 << 13,
            "relu" => 1 << 14,
            "oob" => 1 << 15,
            "NaN" => 1 << 16,
            "xorsign" => 1 << 17,
            "abs" => 1 << 18,
            "satfinite" => 1 << 19,
            x => return Err(format!("float modifier .{x}")),
        };
    }
    Ok(c)
}

fn arith<'i>(base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    let opc: u64 = match base {
        "add" => 0,
        "sub" => 1,
        "mul" => 2,
        "fma" => 3,
        "min" => 4,
        "max" => 5,
        _ => return None,
    };
    let tys: Vec<&str> = m.iter().copied().filter(|x| FTYPES.contains(x) || ALT4.contains(x)).collect();
    if tys.is_empty() {
        return None;
    }
    let mods: Vec<&str> = m.iter().copied().filter(|x| !tys.contains(x)).collect();
    let w0 = tys.first().and_then(|t| ftype(t)).map_or(0, |(c, pk, w)| if pk { 0 } else { let _ = c; w });
    let op = |i: usize| cvt::fimm(ops[i].op.clone(), w0);
    if ALT4.contains(&tys[0]) {
        return alt4(opc, &tys, &mods, ops);
    }
    let mut ctrl = match fmods(&mods) {
        Ok(c) => c,
        Err(e) => return err(e),
    };
    match tys.as_slice() {
        [t] => {
            let (code, pk, w) = ftype(t)?;
            let three = opc >= 4 && ops.len() == 4;
            if ops.len() != if opc == 3 || three { 4 } else { 3 } {
                return err("float operands");
            }
            ctrl |= if three { opc + 2 } else { opc } | code << 4 | (pk as u64) << 8;
            let c = if ops.len() == 4 { op(3) } else { imm(0) };
            Call::new(format!("fop_{w}"))
                .dst(op(0), bits(w))
                .src(op(1), bits(w))
                .src(op(2), bits(w))
                .src(c, bits(w))
                .imm(ctrl)
                .build()
        }
        ["f32", h @ ("f16" | "bf16")] if opc < 4 => {
            ctrl |= opc | ftype(h)?.0 << 20;
            let (b, c) = match (opc, ops.len()) {
                (3, 4) => (op(2), op(3)),
                (0..=2, 3) => (imm(0), op(2)),
                _ => return err("mixed precision operands"),
            };
            Call::new("fmix_f32")
                .dst(op(0), ScalarType::B32)
                .src(op(1), ScalarType::B16)
                .src(b, ScalarType::B16)
                .src(c, ScalarType::B32)
                .imm(ctrl)
                .build()
        }
        [h @ ("f16x2" | "bf16x2"), "f32x2", "f32x2"] if opc < 3 && ops.len() == 3 => {
            ctrl |= opc | ftype(h)?.0 << 20;
            Call::new("fpair_f32x2")
                .dst(op(0), ScalarType::B32)
                .src(op(1), ScalarType::B64)
                .src(op(2), ScalarType::B64)
                .imm(ctrl)
                .build()
        }
        ["f32x2", h @ ("f16x2" | "bf16x2"), _, "f32x2"] if opc == 3 && ops.len() == 4 => {
            ctrl |= opc | ftype(h)?.0 << 20;
            Call::new("ffma_f32x2_pair")
                .dst(op(0), ScalarType::B64)
                .src(op(1), ScalarType::B32)
                .src(op(2), ScalarType::B32)
                .src(op(3), ScalarType::B64)
                .imm(ctrl)
                .build()
        }
        _ => None,
    }
}

/// add / sub{.rn}{.satfinite}.cdtype.atype d, a, c;  mul{.rn}{.satfinite}.dtype.atype.ctype d, a, c;
/// fma{.rn}{.satfinite}.cdtype.atype.btype d, a, b, c
fn alt4<'i>(opc: u64, tys: &[&str], mods: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    let k = |t: &str| ALT4.iter().position(|x| *x == t).map(|p| p as u64);
    let dk = match tys[0] {
        "e5m2x4" => 0,
        "e4m3x4" => 1,
        _ => return err("altfp destination"),
    };
    let satf = mods.contains(&"satfinite") as u64;
    if mods.iter().any(|x| *x != "rn" && *x != "satfinite") {
        return err("altfp modifier");
    }
    let (ka, kb, kc, b, c) = match (opc, tys.len(), ops.len()) {
        (0 | 1, 2, 3) => (k(tys[1])?, dk, 0, ops[2].op.clone(), imm(0)),
        (2, 3, 3) => (k(tys[1])?, k(tys[2])?, 0, ops[2].op.clone(), imm(0)),
        (3, 3, 4) => (k(tys[1])?, k(tys[2])?, dk, ops[2].op.clone(), ops[3].op.clone()),
        _ => return err("altfp form"),
    };
    let w = |k: u64| if k == 4 { 16 } else { 32 };
    let ctrl = opc | (dk == 1) as u64 * 4 | satf << 3 | ka << 4 | kb << 8 | kc << 12;
    Call::new(format!("alt4_{}_{}", w(ka), w(kb)))
        .dst(ops[0].op.clone(), ScalarType::B32)
        .src(ops[1].op.clone(), bits(w(ka)))
        .src(b, bits(w(kb)))
        .src(c, ScalarType::B32)
        .imm(ctrl)
        .build()
}

const CMPS: [&str; 14] = ["eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "leu", "gtu", "geu", "num", "nan"];

fn is_sink(o: &Op) -> bool {
    matches!(o, ast::ParsedOperand::Reg("_"))
}

/// set / setp on f16 / bf16 / f16x2 / bf16x2 sources
fn compare<'i>(base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    let tys: Vec<&str> = m.iter().copied().filter(|x| cvt::cvt_ty(x).is_some()).collect();
    let st = *tys.last()?;
    let (code, pk) = match st {
        "f16" => (2u64, false),
        "bf16" => (3, false),
        "f16x2" => (2, true),
        "bf16x2" => (3, true),
        _ => return None,
    };
    let cmpc = m.iter().find_map(|x| CMPS.iter().position(|c| c == x))? as u64;
    let bop = if m.contains(&"and") {
        1
    } else if m.contains(&"or") {
        2
    } else if m.contains(&"xor") {
        3
    } else {
        0
    };
    let ftz = m.contains(&"ftz") as u64;
    let (c, negc) = match ops.get(3) {
        Some(h) => (h.op.clone(), h.neg as u64),
        None => (imm(1), 0),
    };
    if bop != 0 && ops.len() != 4 {
        return err("compare needs c");
    }
    let mut ctrl = cmpc | code << 4 | (pk as u64) << 8 | bop << 9 | ftz << 11 | negc << 12;
    let w = if pk { ScalarType::B32 } else { ScalarType::B16 };
    let call = |name: String| {
        Call::new(name)
    };
    if base == "setp" {
        if tys.len() != 1 {
            return err("setp types");
        }
        if !pk {
            return call("fsetp_16".into())
                .dst(ops[0].op.clone(), ScalarType::Pred)
                .src(ops[1].op.clone(), w)
                .src(ops[2].op.clone(), w)
                .src(c, ScalarType::Pred)
                .imm(ctrl)
                .build();
        }
        let d0 = if ops[0].sink { None } else { Some(ops[0].op.clone()) };
        let d1 = ops[0].pair.clone().filter(|p| !is_sink(p));
        let base_call = |name: &str| call(name.into());
        let cl = match (d0, d1) {
            (Some(p), Some(q)) => base_call("fsetp_pair").dst(p, ScalarType::Pred).dst2(q, ScalarType::Pred),
            (Some(p), None) => base_call("fsetp_pair_lo").dst(p, ScalarType::Pred),
            (None, Some(q)) => base_call("fsetp_pair_hi").dst(q, ScalarType::Pred),
            (None, None) => return err("setp without destination"),
        };
        return cl.src(ops[1].op.clone(), w).src(ops[2].op.clone(), w).src(c, ScalarType::Pred).imm(ctrl).build();
    }
    // set.CmpOp{.BoolOp}{.ftz}.dtype.stype d, a, b{, {!}c}
    if tys.len() != 2 {
        return err("set types");
    }
    let dt = cvt::cvt_ty(tys[0])?;
    ctrl |= dt.code << 16;
    if pk {
        return call("fset_pair".into())
            .dst(ops[0].op.clone(), ScalarType::B32)
            .src(ops[1].op.clone(), w)
            .src(ops[2].op.clone(), w)
            .src(c, ScalarType::Pred)
            .imm(ctrl)
            .build();
    }
    let (dn, dty) = match tys[0] {
        "f16" | "bf16" | "u16" | "s16" | "b16" => (16, ScalarType::B16),
        _ => (32, ScalarType::B32),
    };
    call(format!("fset_{dn}_16"))
        .dst(ops[0].op.clone(), dty)
        .src(ops[1].op.clone(), w)
        .src(ops[2].op.clone(), w)
        .src(c, ScalarType::Pred)
        .imm(ctrl)
        .build()
}

/// div / sqrt{.rnd}{.ftz}.sat.f32 (exact, then clamped)
fn divsqrt<'i>(base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    if m.last() != Some(&"f32") || m.contains(&"approx") || m.contains(&"full") {
        return None;
    }
    let opc = match base {
        "div" => 0,
        "sqrt" => 1,
        "rcp" => 2,
        _ => return None,
    };
    let ctrl = match fmods(&m[..m.len() - 1]) {
        Ok(c) => c | opc,
        Err(e) => return err(e),
    };
    let b = if opc == 0 { cvt::fimm(ops.get(2)?.op.clone(), 32) } else { imm(0) };
    Call::new("fdivsqrt_f32")
        .dst(ops[0].op.clone(), ScalarType::B32)
        .src(cvt::fimm(ops[1].op.clone(), 32), ScalarType::B32)
        .src(b, ScalarType::B32)
        .imm(ctrl)
        .build()
}

/// neg / abs{.ftz}.{f32,f16,bf16,f16x2,bf16x2}: flush, then the sign bit
fn negabs<'i>(base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    let (code, pk, w) = ftype(m.last()?)?;
    if code == 0 || ops.len() != 2 || m.iter().any(|x| *x != "ftz" && ftype(x).is_none()) {
        return None;
    }
    let ctrl = (base == "abs") as u64 | code << 4 | (pk as u64) << 8 | (m.contains(&"ftz") as u64) << 12;
    Call::new(format!("fnegabs_{w}"))
        .dst(ops[0].op.clone(), bits(w))
        .src(cvt::fimm(ops[1].op.clone(), if pk { 0 } else { w }), bits(w))
        .imm(ctrl)
        .build()
}

pub(super) fn lower<'i>(_opcode: &str, base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    match base {
        "add" | "sub" | "mul" | "fma" | "min" | "max" => arith(base, m, ops),
        "set" | "setp" => compare(base, m, ops),
        "div" | "sqrt" | "rcp" => divsqrt(base, m, ops),
        "neg" | "abs" => negabs(base, m, ops),
        _ => None,
    }
}

/// Opcodes this family lowers even though the grammar accepts them (ZLUDA's lowering differs
/// from the ISA there): see helper_table::preempts.
pub(super) fn preempts(_opcode: &str, base: &str, m: &[&str]) -> bool {
    let half = |t: &&str| matches!(*t, "f16" | "bf16" | "f16x2" | "bf16x2");
    let any = |l: &[&str]| l.iter().any(|x| m.contains(x));
    match base {
        // half comparisons (.ftz, NaN, the pair destinations)
        "set" | "setp" => m.last().map_or(false, |t| half(t)),
        // half arithmetic: NaN results, subnormals, directed roundings, .relu / .sat / .oob
        "add" | "sub" | "mul" | "fma" => m.iter().any(half),
        // min / max: .ftz .NaN .xorsign .abs (llvm.canonicalize is not available), halves, f64 NaN payloads
        "min" | "max" => m.iter().any(half) || m.contains(&"f64") || any(&["ftz", "NaN", "xorsign", "abs"]),
        // neg / abs .ftz (llvm.canonicalize is not available)
        "neg" | "abs" => m.contains(&"ftz"),
        // f32 division and square root with a directed rounding or .ftz (exact, with subnormals)
        "div" | "sqrt" | "rcp" => {
            m.last() == Some(&"f32") && !any(&["approx", "full"]) && any(&["rz", "rm", "rp", "ftz", "sat"])
        }
        _ => false,
    }
}
