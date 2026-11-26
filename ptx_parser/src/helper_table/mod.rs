//! Lowering of fallback-parsed instructions (helpers.rs) to `Instruction::Helper` calls.
//!
//! Each entry decodes the opcode's modifiers into immediates (control words) passed to one
//! generic helper function in ptx/lib/zluda_ptx_metal_impl.cpp; after inlining LLVM folds
//! the control words away. Semantics follow PTX-formalization's Sail model (sail/ptx).
use super::*;
pub(crate) use crate::helpers::HOp;

mod cvt;
mod float;
mod matrix;
mod memory;
mod misc;
mod video;

pub(crate) use memory::cp_async;

pub(crate) type Op<'i> = ast::ParsedOperand<&'i str>;
pub(crate) type R<'i> = Option<Result<ast::Instruction<ParsedOperandStr<'i>>, String>>;

pub(crate) fn imm<'i>(v: u64) -> Op<'i> {
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
    /// a vector source (`{a, b}`), padded with zeros to `n` elements
    pub fn src_vec(mut self, op: Op<'i>, n: u8, t: ScalarType) -> Self {
        let i = self.next_src;
        let op = match op {
            ast::ParsedOperand::VecPack(mut v) => {
                while v.len() < n as usize {
                    v.push(Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::U64(0))));
                }
                ast::ParsedOperand::VecPack(v)
            }
            other => other,
        };
        self.data = self.data.with_type(i, ast::Type::Vector(n, t), StateSpace::Reg);
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

pub(crate) fn err<'i>(e: impl Into<String>) -> R<'i> {
    Some(Err(e.into()))
}

/// opcode split at '.': base, then modifiers (`shared::cta` stays one modifier)
pub(crate) fn split(opcode: &str) -> (&str, Vec<&str>) {
    let mut it = opcode.split('.');
    let base = it.next().unwrap_or("");
    (base, it.collect())
}

pub(crate) fn scalar(s: &str) -> Option<ScalarType> {
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

pub(crate) fn is_signed(s: &str) -> bool {
    s.starts_with('s')
}

/// byte / half selector code: 0 none, 1-4 b0..b3, 5 h0, 6 h1
pub(crate) fn sel_code(s: Option<&str>) -> Result<u64, String> {
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

pub(crate) fn cmp_code(m: &[&str]) -> u64 {
    for (i, c) in ["eq", "ne", "lt", "le", "gt", "ge"].iter().enumerate() {
        if m.contains(c) {
            return i as u64;
        }
    }
    0
}

/// Lane selector digits (`.b3210`, `.h10`) packed as nibbles, lane 0 in the low nibble; with
/// `n` lanes. `None` gives the default.
pub(crate) fn lane_sel(s: Option<&str>, n: usize, default: &str) -> Result<u64, String> {
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
pub(crate) fn lane_mask(s: Option<&str>, n: usize) -> u64 {
    match s {
        None => (1 << n) - 1,
        Some(x) => x[2..]
            .chars()
            .filter_map(|c| c.to_digit(10))
            .fold(0, |m, d| m | (1 << d)),
    }
}


/// Instructions that follow a fallback instruction (under its predicate): a third destination, ...
pub(crate) fn extra<'i>(opcode: &str, ops: &[HOp<'i>]) -> Vec<ast::Instruction<ParsedOperandStr<'i>>> {
    let (base, m) = split(opcode);
    memory::extra(base, &m, ops)
}

/// Lowers a fallback-parsed instruction: each family module returns None for opcodes it
/// does not know.
pub(crate) fn lower<'i>(opcode: &str, ops: Vec<HOp<'i>>) -> R<'i> {
    let (base, m) = split(opcode);
    let m = &m[..];
    for family in [video::lower, misc::lower, cvt::lower, float::lower, matrix::lower, memory::lower] {
        if let Some(r) = family(opcode, base, m, &ops) {
            return Some(r);
        }
    }
    None
}
