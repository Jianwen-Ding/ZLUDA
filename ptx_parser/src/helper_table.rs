//! Lowering of fallback-parsed instructions (helpers.rs) to `Instruction::Helper` calls.
//!
//! Each entry decodes the opcode's modifiers into immediates (control words) passed to one
//! generic helper function in ptx/lib/zluda_ptx_metal_impl.cpp; after inlining LLVM folds
//! the control words away. Semantics follow PTX-formalization's Sail model (sail/ptx).
use super::*;
use crate::helpers::HOp;

type Op<'i> = ast::ParsedOperand<&'i str>;
type R<'i> = Option<Result<ast::Instruction<ParsedOperandStr<'i>>, String>>;

fn imm<'i>(v: u64) -> Op<'i> {
    ast::ParsedOperand::Imm(ast::ImmediateValue::U64(v))
}

/// A helper call under construction: slot 0 = dst, 1 = dst2, 2.. = sources.
pub(crate) struct Call<'i> {
    data: ast::HelperData,
    slots: [Option<Op<'i>>; 8],
    next_src: usize,
}

impl<'i> Call<'i> {
    pub fn new(name: impl Into<String>) -> Self {
        Call {
            data: ast::HelperData::new(name),
            slots: Default::default(),
            next_src: 2,
        }
    }
    pub fn dst(mut self, op: Op<'i>, t: ScalarType) -> Self {
        self.data = self.data.with(0, t);
        self.slots[0] = Some(op);
        self
    }
    pub fn dst2(mut self, op: Op<'i>, t: ScalarType) -> Self {
        self.data = self.data.with(1, t);
        self.slots[1] = Some(op);
        self
    }
    pub fn src(mut self, op: Op<'i>, t: ScalarType) -> Self {
        let i = self.next_src;
        self.data = self.data.with(i, t);
        self.slots[i] = Some(op);
        self.next_src += 1;
        self
    }
    pub fn src_space(mut self, op: Op<'i>, t: ScalarType, space: StateSpace) -> Self {
        let i = self.next_src;
        self.data = self.data.with_type(i, ast::Type::Scalar(t), space);
        self.slots[i] = Some(op);
        self.next_src += 1;
        self
    }
    pub fn imm(self, v: u64) -> Self {
        self.src(imm(v), ScalarType::U32)
    }
    pub fn build(self) -> R<'i> {
        let [dst, dst2, src1, src2, src3, src4, src5, src6] = self.slots;
        Some(Ok(ast::Instruction::Helper {
            data: self.data,
            arguments: ast::HelperArgs {
                dst,
                dst2,
                src1,
                src2,
                src3,
                src4,
                src5,
                src6,
            },
        }))
    }
}

fn err<'i>(e: impl Into<String>) -> R<'i> {
    Some(Err(e.into()))
}

/// opcode split at '.': base, then modifiers (`shared::cta` stays one modifier)
fn split(opcode: &str) -> (&str, Vec<&str>) {
    let mut it = opcode.split('.');
    let base = it.next().unwrap_or("");
    (base, it.collect())
}

fn scalar(s: &str) -> Option<ScalarType> {
    Some(match s {
        "u8" => ScalarType::U8,
        "s8" => ScalarType::S8,
        "b8" => ScalarType::B8,
        "u16" => ScalarType::U16,
        "s16" => ScalarType::S16,
        "b16" => ScalarType::B16,
        "u32" => ScalarType::U32,
        "s32" => ScalarType::S32,
        "b32" => ScalarType::B32,
        "u64" => ScalarType::U64,
        "s64" => ScalarType::S64,
        "b64" => ScalarType::B64,
        "f16" => ScalarType::F16,
        "bf16" => ScalarType::BF16,
        "f32" => ScalarType::F32,
        "f64" => ScalarType::F64,
        "pred" => ScalarType::Pred,
        "f16x2" => ScalarType::F16x2,
        "bf16x2" => ScalarType::BF16x2,
        _ => return None,
    })
}

fn is_signed(s: &str) -> bool {
    s.starts_with('s')
}

/// byte / half selector code: 0 none, 1-4 b0..b3, 5 h0, 6 h1
fn sel_code(s: Option<&str>) -> Result<u64, String> {
    Ok(match s {
        None => 0,
        Some(".b0") => 1,
        Some(".b1") => 2,
        Some(".b2") => 3,
        Some(".b3") => 4,
        Some(".h0") => 5,
        Some(".h1") => 6,
        Some(x) => return Err(format!("selector {x}")),
    })
}

fn cmp_code(m: &[&str]) -> u64 {
    for (i, c) in ["eq", "ne", "lt", "le", "gt", "ge"].iter().enumerate() {
        if m.contains(c) {
            return i as u64;
        }
    }
    0
}

/// Lane selector digits (`.b3210`, `.h10`) packed as nibbles, lane 0 in the low nibble; with
/// `n` lanes. `None` gives the default.
fn lane_sel(s: Option<&str>, n: usize, default: &str) -> Result<u64, String> {
    let s = s.map(|x| &x[1..]).unwrap_or(default);
    let digits = &s[1..];
    if digits.len() != n || !(s.starts_with('b') || s.starts_with('h')) {
        return Err(format!("lane selector .{s}"));
    }
    let mut v = 0u64;
    for (l, c) in digits.chars().rev().enumerate() {
        v |= (c.to_digit(16).ok_or("selector digit")? as u64) << (4 * l);
    }
    Ok(v)
}

/// lanes present in a destination mask `.h10` / `.b31`
fn lane_mask(s: Option<&str>, n: usize) -> u64 {
    match s {
        None => (1 << n) - 1,
        Some(x) => x[2..]
            .chars()
            .filter_map(|c| c.to_digit(10))
            .fold(0, |m, d| m | (1 << d)),
    }
}

fn video<'i>(base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    let types: Vec<&str> = m.iter().copied().filter(|x| scalar(x).is_some()).collect();
    let mods: Vec<&str> = m.iter().copied().filter(|x| scalar(x).is_none()).collect();
    let scalar_ops = ["vadd", "vsub", "vabsdiff", "vmin", "vmax", "vshl", "vshr"];
    let sec = if mods.contains(&"add") {
        1
    } else if mods.contains(&"min") {
        2
    } else if mods.contains(&"max") {
        3
    } else {
        0
    };
    let sat = mods.contains(&"sat") as u64;
    if let Some(opi) = scalar_ops.iter().position(|x| *x == base) {
        // vop.dtype.atype.btype{.sat}{.op2} d{.dsel}, a{.asel}, b{.bsel}{, c}
        if types.len() != 3 || ops.len() < 3 {
            return err("video operands");
        }
        let (asel, bsel, dsel) = match (sel_code(ops[1].sel), sel_code(ops[2].sel), sel_code(ops[0].sel)) {
            (Ok(a), Ok(b), Ok(d)) => (a, b, d),
            _ => return err("video selector"),
        };
        let merge = (ops[0].sel.is_some()) as u64;
        let ctrl = opi as u64
            | (is_signed(types[0]) as u64) << 4
            | (is_signed(types[1]) as u64) << 5
            | (is_signed(types[2]) as u64) << 6
            | sat << 7
            | (if merge == 1 { 0 } else { sec }) << 8
            | asel << 10
            | bsel << 14
            | ((!mods.contains(&"wrap")) as u64) << 18
            | merge << 19
            | dsel << 20;
        let c = ops.get(3).map(|o| o.op.clone()).unwrap_or(imm(0));
        return Call::new("video_scalar")
            .dst(ops[0].op.clone(), ScalarType::B32)
            .src(ops[1].op.clone(), ScalarType::B32)
            .src(ops[2].op.clone(), ScalarType::B32)
            .src(c, ScalarType::B32)
            .imm(ctrl)
            .build();
    }
    if base == "vset" {
        // vset.atype.btype.cmp d{.dsel}, a{.asel}, b{.bsel}{.op2 c}
        if types.len() != 2 || ops.len() < 3 {
            return err("vset operands");
        }
        let (asel, bsel, dsel) = match (sel_code(ops[1].sel), sel_code(ops[2].sel), sel_code(ops[0].sel)) {
            (Ok(a), Ok(b), Ok(d)) => (a, b, d),
            _ => return err("video selector"),
        };
        let merge = (ops[0].sel.is_some() && ops.len() == 4) as u64;
        let ctrl = 7
            | (is_signed(types[0]) as u64) << 5
            | (is_signed(types[1]) as u64) << 6
            | (if merge == 1 { 0 } else { sec }) << 8
            | asel << 10
            | bsel << 14
            | merge << 19
            | dsel << 20
            | cmp_code(&mods) << 24;
        let c = ops.get(3).map(|o| o.op.clone()).unwrap_or(imm(0));
        return Call::new("video_scalar")
            .dst(ops[0].op.clone(), ScalarType::B32)
            .src(ops[1].op.clone(), ScalarType::B32)
            .src(ops[2].op.clone(), ScalarType::B32)
            .src(c, ScalarType::B32)
            .imm(ctrl)
            .build();
    }
    if base == "vmad" {
        // vmad.dtype.atype.btype{.sat}{.scale} d, {-}a{.asel}, {-}b{.bsel}, {-}c;  (.po: plus one)
        if types.len() != 3 || ops.len() != 4 {
            return err("vmad operands");
        }
        let (asel, bsel) = match (sel_code(ops[1].sel), sel_code(ops[2].sel)) {
            (Ok(a), Ok(b)) => (a, b),
            _ => return err("video selector"),
        };
        let neg = |o: &HOp<'i>| -> Result<(Op<'i>, bool), String> {
            match &o.op {
                ast::ParsedOperand::Imm(ast::ImmediateValue::S64(x)) if *x < 0 => {
                    Ok((imm((-*x) as u64), true))
                }
                op => Ok((op.clone(), false)),
            }
        };
        let (a, na) = match neg(&ops[1]) { Ok(x) => x, Err(e) => return err(e) };
        let (b, nb) = match neg(&ops[2]) { Ok(x) => x, Err(e) => return err(e) };
        let (c, nc) = match neg(&ops[3]) { Ok(x) => x, Err(e) => return err(e) };
        let (na, nb, nc) = (na || ops[1].minus, nb || ops[2].minus, nc || ops[3].minus);
        let ctrl = (is_signed(types[1]) as u64) << 5
            | (is_signed(types[2]) as u64) << 6
            | sat << 7
            | asel << 10
            | bsel << 14
            | (mods.contains(&"po") as u64) << 18
            | (na as u64) << 19
            | (nb as u64) << 20
            | (nc as u64) << 21
            | (if mods.contains(&"shr7") { 1 } else if mods.contains(&"shr15") { 2 } else { 0 }) << 22;
        return Call::new("vmad")
            .dst(ops[0].op.clone(), ScalarType::B32)
            .src(a, ScalarType::B32)
            .src(b, ScalarType::B32)
            .src(c, ScalarType::B32)
            .imm(ctrl)
            .build();
    }
    // SIMD video: vop2 / vop4
    let simd_ops = ["vadd", "vsub", "vavrg", "vabsdiff", "vmin", "vmax", "vset"];
    let (name, n) = match base.strip_suffix('2') {
        Some(b) => (b, 2),
        None => match base.strip_suffix('4') {
            Some(b) => (b, 4),
            None => return None,
        },
    };
    let opi = simd_ops.iter().position(|x| *x == name)? as u64;
    let is_set = name == "vset";
    let (dt, at, bt) = if is_set {
        if types.len() != 2 {
            return err("vset2/4 types");
        }
        ("u32", types[0], types[1])
    } else {
        if types.len() != 3 {
            return err("simd video types");
        }
        (types[0], types[1], types[2])
    };
    if ops.len() != 4 {
        return err("simd video operands");
    }
    let (dsel_def, asel_def, bsel_def) = if n == 2 { ("h10", "h10", "h32") } else { ("b3210", "b3210", "b7654") };
    let _ = dsel_def;
    let asel = match lane_sel(ops[1].sel, n, asel_def) { Ok(v) => v, Err(e) => return err(e) };
    let bsel = match lane_sel(ops[2].sel, n, bsel_def) { Ok(v) => v, Err(e) => return err(e) };
    let mask = lane_mask(ops[0].sel, n);
    let ctrl = opi
        | ((n == 4) as u64) << 3
        | (is_signed(dt) as u64) << 4
        | (is_signed(at) as u64) << 5
        | (is_signed(bt) as u64) << 6
        | (sat & (!is_set) as u64) << 7
        | (mods.contains(&"add") as u64) << 8
        | mask << 10
        | cmp_code(&mods) << 24;
    Call::new("video_simd")
        .dst(ops[0].op.clone(), ScalarType::B32)
        .src(ops[1].op.clone(), ScalarType::B32)
        .src(ops[2].op.clone(), ScalarType::B32)
        .src(ops[3].op.clone(), ScalarType::B32)
        .imm(ctrl)
        .imm(asel | bsel << 16)
        .build()
}

pub(crate) fn lower<'i>(opcode: &str, ops: Vec<HOp<'i>>) -> R<'i> {
    let (base, m) = split(opcode);
    let m = &m[..];
    match base {
        "vadd" | "vsub" | "vabsdiff" | "vmin" | "vmax" | "vshl" | "vshr" | "vset" | "vmad" | "vadd2"
        | "vsub2" | "vavrg2" | "vabsdiff2" | "vmin2" | "vmax2" | "vset2" | "vadd4" | "vsub4"
        | "vavrg4" | "vabsdiff4" | "vmin4" | "vmax4" | "vset4" => video(base, m, &ops),
        "lop3" => {
            // lop3.b32 d, a, b, c, immLut;  lop3.BoolOp.b32 d|p, a, b, c, immLut, q;
            if ops.len() == 5 && m == ["b32"] {
                return Call::new("lop3_b32")
                    .dst(ops[0].op.clone(), ScalarType::B32)
                    .src(ops[1].op.clone(), ScalarType::B32)
                    .src(ops[2].op.clone(), ScalarType::B32)
                    .src(ops[3].op.clone(), ScalarType::B32)
                    .src(ops[4].op.clone(), ScalarType::B32)
                    .build();
            }
            if ops.len() == 6 && m.len() == 2 && (m[0] == "and" || m[0] == "or") {
                let Some(p) = ops[0].pair.clone() else { return err("lop3 predicate form needs d|p") };
                if ops[0].sink {
                    return Call::new(format!("lop3_{}_b32_pred", m[0]))
                        .dst(p, ScalarType::Pred)
                        .src(ops[1].op.clone(), ScalarType::B32)
                        .src(ops[2].op.clone(), ScalarType::B32)
                        .src(ops[3].op.clone(), ScalarType::B32)
                        .src(ops[4].op.clone(), ScalarType::B32)
                        .src(ops[5].op.clone(), ScalarType::Pred)
                        .imm(ops[5].neg as u64)
                        .build();
                }
                return Call::new(format!("lop3_{}_b32", m[0]))
                    .dst(ops[0].op.clone(), ScalarType::B32)
                    .dst2(p, ScalarType::Pred)
                    .src(ops[1].op.clone(), ScalarType::B32)
                    .src(ops[2].op.clone(), ScalarType::B32)
                    .src(ops[3].op.clone(), ScalarType::B32)
                    .src(ops[4].op.clone(), ScalarType::B32)
                    .src(ops[5].op.clone(), ScalarType::Pred)
                    .imm(ops[5].neg as u64)
                    .build();
            }
            err("lop3 form")
        }
        "testp" => {
            // testp.op.type p, a;  op = finite, infinite, number, notanumber, normal, subnormal
            let ops_ = ["finite", "infinite", "number", "notanumber", "normal", "subnormal"];
            if m.len() != 2 || ops.len() != 2 {
                return err("testp form");
            }
            let k = ops_.iter().position(|x| *x == m[0])? as u64;
            let t = match m[1] {
                "f32" => ScalarType::F32,
                "f64" => ScalarType::F64,
                _ => return err("testp type"),
            };
            Call::new(format!("testp_{}", m[1]))
                .dst(ops[0].op.clone(), ScalarType::Pred)
                .src(ops[1].op.clone(), t)
                .imm(k)
                .build()
        }
        "slct" => {
            // slct.dtype.s32 d, a, b, c;  slct{.ftz}.dtype.f32 d, a, b, c;   d = c >= 0 ? a : b
            let ftz = m.first() == Some(&"ftz");
            let m2: Vec<&str> = m.iter().copied().filter(|x| *x != "ftz").collect();
            if m2.len() != 2 || ops.len() != 4 {
                return err("slct form");
            }
            let dt = scalar(m2[0])?;
            let size = match m2[0] {
                "b16" | "u16" | "s16" | "f16" => 16,
                "b64" | "u64" | "s64" | "f64" => 64,
                _ => 32,
            };
            let ct = scalar(m2[1])?;
            let bits = match size {
                16 => ScalarType::B16,
                64 => ScalarType::B64,
                _ => ScalarType::B32,
            };
            Call::new(format!("slct_b{size}_{}{}", m2[1], if ftz { "_ftz" } else { "" }))
                .dst(ops[0].op.clone(), bits)
                .src(ops[1].op.clone(), bits)
                .src(ops[2].op.clone(), bits)
                .src(ops[3].op.clone(), ct)
                .build()
                .map(|r| {
                    let _ = dt;
                    r
                })
        }
        _ => None,
    }
}
