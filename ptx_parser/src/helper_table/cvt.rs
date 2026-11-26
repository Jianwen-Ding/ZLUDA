//! cvt variants (satfinite, relu, rs, fp8/fp6/fp4/ue8m0, packing): PTX ISA 9.7.9.21-23, Sail
//! ptx_alu.sail (cvt), ptx_half.sail (pairs, cvt.pack), ptx_mini.sail (mini formats).
//! Helpers: ptx/lib/metal_impl/cvt.inc (an exact soft-float core).
use super::*;

/// type codes shared with cvt.inc
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct CvtTy {
    /// element type code
    pub code: u64,
    /// elements (1, 2 or 4)
    pub n: u32,
    /// bit width of one element in its container (16 for f16 lanes, 8 or 4 for mini)
    pub ew: u32,
}

pub(crate) fn cvt_ty(s: &str) -> Option<CvtTy> {
    let t = |code, n, ew| Some(CvtTy { code, n, ew });
    match s {
        "f64" => t(0, 1, 64),
        "f32" => t(1, 1, 32),
        "f16" => t(2, 1, 16),
        "bf16" => t(3, 1, 16),
        "tf32" => t(4, 1, 32),
        "u8" => t(16, 1, 8),
        "u16" => t(17, 1, 16),
        "u32" => t(18, 1, 32),
        "u64" => t(19, 1, 64),
        "s8" => t(20, 1, 8),
        "s16" => t(21, 1, 16),
        "s32" => t(22, 1, 32),
        "s64" => t(23, 1, 64),
        "f16x2" => t(2, 2, 16),
        "bf16x2" => t(3, 2, 16),
        "e4m3x2" => t(5, 2, 8),
        "e5m2x2" => t(6, 2, 8),
        "e2m3x2" => t(7, 2, 8),
        "e3m2x2" => t(8, 2, 8),
        "e2m1x2" => t(9, 2, 4),
        "ue5m3x2" => t(10, 2, 8),
        "ue8m0x2" => t(11, 2, 8),
        "s2f6x2" => t(12, 2, 8),
        "e4m3x4" => t(5, 4, 8),
        "e5m2x4" => t(6, 4, 8),
        "e2m3x4" => t(7, 4, 8),
        "e3m2x4" => t(8, 4, 8),
        "e2m1x4" => t(9, 4, 4),
        _ => None,
    }
}

impl CvtTy {
    fn bits(&self) -> u32 {
        self.n * self.ew
    }
    fn is_int(&self) -> bool {
        self.code >= 16
    }
    fn is_float_scalar(&self) -> bool {
        self.n == 1 && self.code <= 4
    }
    /// the helper slot type: integers keep their type (relaxed operand rules), the rest are bits
    fn slot(&self) -> ScalarType {
        if self.is_int() {
            return scalar(match self.code {
                16 => "u8",
                17 => "u16",
                18 => "u32",
                19 => "u64",
                20 => "s8",
                21 => "s16",
                22 => "s32",
                _ => "s64",
            })
            .unwrap();
        }
        bits_ty(self.bits())
    }
}

/// A float literal operand as the bits of a `w`-bit float (helpers take float operands as bits)
pub(crate) fn fimm<'i>(op: Op<'i>, w: u32) -> Op<'i> {
    let v = match op {
        ast::ParsedOperand::Imm(ast::ImmediateValue::F32(x)) => x as f64,
        ast::ParsedOperand::Imm(ast::ImmediateValue::F64(x)) => x,
        o => return o,
    };
    imm(match w {
        64 => v.to_bits(),
        32 => (v as f32).to_bits() as u64,
        16 => f32_to_f16_bits(v as f32) as u64,
        _ => v.to_bits(),
    })
}

/// round-to-nearest-even f32 -> f16 bits (literals only)
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7fffff;
    if e == 255 {
        return s | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let ue = e - 127 + 15;
    if ue >= 31 {
        return s | 0x7c00;
    }
    let (mant, sh) = if ue <= 0 { (m | 0x800000, (14 - ue) as u32) } else { (m, 13u32) };
    if sh > 24 {
        return s;
    }
    let q = mant >> sh;
    let r = mant & ((1 << sh) - 1);
    let h = 1 << (sh - 1);
    let q = if r > h || (r == h && q & 1 == 1) { q + 1 } else { q };
    if ue <= 0 {
        s | q as u16
    } else {
        s | ((((ue as u32) << 10) + q) as u16)
    }
}

pub(crate) fn bits_ty(w: u32) -> ScalarType {
    match w {
        8 | 4 => ScalarType::B8,
        16 => ScalarType::B16,
        64 => ScalarType::B64,
        _ => ScalarType::B32,
    }
}
fn cw(w: u32) -> u32 {
    if w <= 8 {
        8
    } else {
        w
    }
}

/// The modifier part of the control word (cvt.inc: bits 10..20)
fn cvt_mods(mods: &[&str]) -> Result<u64, String> {
    let mut c = 0u64;
    for m in mods {
        c |= match *m {
            "rn" => 0,
            "rz" => 1 << 10,
            "rm" => 2 << 10,
            "rp" => 3 << 10,
            "rna" => 4 << 10,
            "rs" => 5 << 10,
            "rni" => 1 << 13,
            "rzi" => 1 << 10 | 1 << 13,
            "rmi" => 2 << 10 | 1 << 13,
            "rpi" => 3 << 10 | 1 << 13,
            "ftz" => 1 << 14,
            "sat" => 1 << 15,
            "relu" => 1 << 16,
            "satfinite" => 1 << 17,
            "pzo" => 1 << 18,
            "scaled::n1::ue8m0" => 1 << 19,
            "scaled::n2::ue8m0" => 1 << 20,
            x => return Err(format!("cvt modifier .{x}")),
        };
    }
    Ok(c)
}

/// Lowers `cvt` forms: also used by the grammar's cvt rules for the forms they do not cover.
pub(crate) fn lower_cvt<'i>(m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    let ntys = m.iter().rev().take_while(|x| cvt_ty(x).is_some()).count();
    if m.contains(&"pack") {
        // cvt.pack.sat.{u16,s16}.s32 d, a, b;  cvt.pack.sat.{u8,s8}.s32.b32 d, a, b, c
        let t = m.iter().copied().find(|x| ["u16", "s16", "u8", "s8"].contains(x))?;
        let ctrl = (t.starts_with('s') as u64) | ((t.ends_with('8') as u64) << 1);
        if !(ops.len() == 3 || ops.len() == 4) {
            return err("cvt.pack operands");
        }
        let c = ops.get(3).map(|o| o.op.clone()).unwrap_or(imm(0));
        return Call::new("cvt_pack_sat")
            .dst(ops[0].op.clone(), ScalarType::B32)
            .src(ops[1].op.clone(), ScalarType::S32)
            .src(ops[2].op.clone(), ScalarType::S32)
            .src(c, ScalarType::B32)
            .imm(ctrl)
            .build();
    }
    if ntys != 2 {
        return None;
    }
    let dt = cvt_ty(m[m.len() - 2])?;
    let st = cvt_ty(m[m.len() - 1])?;
    let mods = match cvt_mods(&m[..m.len() - 2]) {
        Ok(c) => c,
        Err(e) => return err(e),
    };
    let mut ctrl = dt.code | st.code << 5 | mods;
    // float -> int without a rounding modifier: .rzi
    if dt.is_int() && !st.is_int() && mods & (1 << 13) == 0 {
        ctrl |= 1 << 10 | 1 << 13;
    }
    let op = |i: usize| {
        let w = if i == 0 { dt.bits() } else if st.n == 1 && !st.is_int() { st.bits() } else { 0 };
        fimm(ops[i].op.clone(), w)
    };
    // the scale operand: .ue8m0x2 (16 bits) for .scaled::n2, else one .ue8m0
    let (scw, sct) = if mods & (1 << 20) != 0 { (16, ScalarType::B16) } else { (8, ScalarType::B8) };
    match (dt.n, st.n, ops.len()) {
        (1, 1, 2) => Call::new(format!("cvt_{}_{}", cw(dt.bits()), cw(st.bits())))
            .dst(op(0), dt.slot())
            .src(op(1), st.slot())
            .imm(ctrl)
            .build(),
        (2, 1, 4) if mods >> 10 & 7 == 5 => Call::new("cvt_rs_pair")
            .dst(op(0), ScalarType::B32)
            .src(op(1), ScalarType::B32)
            .src(op(2), ScalarType::B32)
            .src(op(3), ScalarType::B32)
            .imm(ctrl)
            .build(),
        (2, 1, 3 | 4) if st.is_float_scalar() => Call::new(format!("cvt_pair_{}_{}_{}", cw(dt.bits()), cw(st.bits()), scw))
            .dst(op(0), bits_ty(dt.bits()))
            .src(op(1), st.slot())
            .src(op(2), st.slot())
            .src(if ops.len() == 4 { op(3) } else { imm(127) }, sct)
            .imm(ctrl)
            .build(),
        (2, 2, 2 | 3) => Call::new(format!("cvt_x2_{}_{}_{}", cw(dt.bits()), cw(st.bits()), scw))
            .dst(op(0), bits_ty(dt.bits()))
            .src(op(1), bits_ty(st.bits()))
            .src(if ops.len() == 3 { op(2) } else { imm(127) }, sct)
            .imm(ctrl)
            .build(),
        (4, 1, 3) => {
            let ast::ParsedOperand::VecPack(items) = &ops[1].op else {
                return err("cvt x4 needs {a, b, e, f}");
            };
            if items.len() != 4 {
                return err("cvt x4 needs 4 sources");
            }
            let mut c = Call::new(format!("cvt_rs_x4_{}", cw(dt.bits()))).dst(op(0), bits_ty(dt.bits()));
            for it in items {
                let o = match it {
                    Some(ast::RegOrImmediate::Reg(r)) => ast::ParsedOperand::Reg(*r),
                    Some(ast::RegOrImmediate::Imm(i)) => ast::ParsedOperand::Imm(*i),
                    None | Some(ast::RegOrImmediate::SymOffset(..)) => return err("cvt x4 sink source"),
                };
                c = c.src(o, ScalarType::B32);
            }
            c.src(op(2), ScalarType::B32).imm(ctrl).build()
        }
        _ => err("cvt operand form"),
    }
}

pub(super) fn lower<'i>(_opcode: &str, base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    match base {
        "cvt" => lower_cvt(m, ops),
        _ => None,
    }
}

fn hop<'i>(op: Op<'i>) -> HOp<'i> {
    HOp { op, addr: false, neg: false, sel: None, pair: None, sink: false, minus: false, items: Default::default() }
}

/// Should a grammar-parsed `cvt` go through the soft-float helper?  ZLUDA's own lowering is
/// kept for the forms it computes exactly.
fn use_helper(rnd: Option<&str>, ftz: bool, sat: bool, dt: &str, st: &str) -> bool {
    let float = |t: &str| !matches!(t, "u8" | "u16" | "u32" | "u64" | "s8" | "s16" | "s32" | "s64");
    let _ = ftz;
    // integer -> f32 / f64 to nearest: exact on the hardware (and soft-f64)
    if !float(st) && matches!(dt, "f32" | "f64") && matches!(rnd, None | Some("rn")) && !sat {
        return false;
    }
    // everything else with a float side: directed roundings, NaN results (the model's canonical
    // NaN), subnormals (the AGX f32 ALU flushes them), saturation of float -> integer (and the
    // ISA's NaN results; llvm.fpto*i.sat is not available on Metal)
    float(st) || float(dt)
}

/// Called by the cvt rules of the grammar (lib.rs) with the modifiers as text (`.rn`, `.f32`):
/// returns the helper instruction for the forms that ZLUDA's lowering does not cover.
/// `.satfinite` forms always lower here.
pub(crate) fn grammar_override<'i, 'e>(
    errors: &mut Vec<PtxError<'e>>,
    rnd: Option<String>,
    ftz: bool,
    sat: bool,
    relu: bool,
    satfinite: bool,
    dt: String,
    st: String,
    ops: Vec<Op<'i>>,
) -> Option<ast::Instruction<ParsedOperandStr<'i>>> {
    let strip = |s: &str| s.trim_start_matches('.').to_string();
    let (dt, st, rnd) = (strip(&dt), strip(&st), rnd.map(|r| strip(&r)));
    if !satfinite && !use_helper(rnd.as_deref(), ftz, sat, &dt, &st) {
        return None;
    }
    let mut m: Vec<String> = Vec::new();
    m.extend(rnd);
    if ftz {
        m.push("ftz".into());
    }
    if sat {
        m.push("sat".into());
    }
    if relu {
        m.push("relu".into());
    }
    if satfinite {
        m.push("satfinite".into());
    }
    m.push(dt);
    m.push(st);
    let mref: Vec<&str> = m.iter().map(|x| x.as_str()).collect();
    let hops: Vec<HOp<'i>> = ops.into_iter().map(hop).collect();
    match lower_cvt(&mref, &hops) {
        Some(Ok(i)) => Some(i),
        r => {
            let e = match r {
                Some(Err(e)) => e,
                _ => "unsupported form".into(),
            };
            errors.push(PtxError::Todo(format!("cvt.{}: {e}", m.join("."))));
            Some(ast::Instruction::Helper {
                data: ast::HelperData::new("unsupported"),
                arguments: ast::HelperArgs { dst: None, dst2: None, src1: None, src2: None, src3: None, src4: None, src5: None, src6: None },
            })
        }
    }
}
