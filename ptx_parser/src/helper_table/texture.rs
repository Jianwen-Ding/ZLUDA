//! Textures, samplers and surfaces (tex, tld4, txq, suld, sust, sured, suq, istypep) as software helpers over the
//! descriptor objects of PTX-formalization's harness (a handle is the u64 global address of the object).  Sail:
//! ptx_tex.sail (exec_tex).  Helpers: ptx/lib/metal_impl/texture.inc.
use super::*;

fn geom(m: &[&str]) -> Option<u64> {
    for (i, g) in ["1d", "2d", "3d", "a1d", "a2d", "cube", "acube", "2dms", "a2dms"].iter().enumerate() {
        if m.contains(g) {
            return Some(i as u64 + 1);
        }
    }
    None
}

fn vec_len(op: &Op<'_>) -> usize {
    match op {
        ast::ParsedOperand::VecPack(v) => v.len(),
        _ => 1,
    }
}

/// `{a}` / `a`: a vector operand padded to 4 elements
fn vec4<'i>(op: &Op<'i>) -> Op<'i> {
    match op {
        ast::ParsedOperand::VecPack(_) => op.clone(),
        ast::ParsedOperand::Reg(r) => ast::ParsedOperand::VecPack(vec![Some(ast::RegOrImmediate::Reg(*r))]),
        ast::ParsedOperand::Imm(i) => ast::ParsedOperand::VecPack(vec![Some(ast::RegOrImmediate::Imm(*i))]),
        other => other.clone(),
    }
}

fn zero4<'i>() -> Op<'i> {
    ast::ParsedOperand::VecPack(vec![])
}

/// a vector of two u64 handles: texture and sampler (0 without)
fn handles<'i>(t: &Op<'i>, s: Option<&Op<'i>>) -> Option<Op<'i>> {
    let item = |o: &Op<'i>| match o {
        ast::ParsedOperand::Reg(r) => Some(Some(ast::RegOrImmediate::Reg(*r))),
        ast::ParsedOperand::Imm(i) => Some(Some(ast::RegOrImmediate::Imm(*i))),
        _ => None,
    };
    let mut v = vec![item(t)?];
    v.push(match s {
        Some(s) => item(s)?,
        None => Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::U64(0))),
    });
    Some(ast::ParsedOperand::VecPack(v))
}

fn tex<'i>(base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    let g = geom(m)?;
    let gather = base == "tld4";
    let nl: u64 = if m.contains(&"level") {
        1
    } else if m.contains(&"grad") {
        2
    } else {
        0
    };
    // the two types: .dtype .ctype
    let tys: Vec<&str> = m.iter().copied().filter(|x| matches!(*x, "f32" | "s32" | "u32" | "f16" | "f16x2")).collect();
    let (dt, ct) = match tys[..] {
        [d, c] => (d, c),
        _ => return err("tex types"),
    };
    let kind: u64 = match dt {
        "f32" => 0,
        "s32" => 1,
        "u32" => 2,
        "f16" => 3,
        _ => 4,
    };
    let Some((d, a)) = ops.split_first().and_then(|(d, r)| r.first().map(|a| (d, a))) else { return err("tex operands") };
    if !a.addr {
        return err("tex address");
    }
    let (s, coords) = match &a.items[..] {
        [c] => (None, c),
        [s, c] => (Some(s), c),
        _ => return err("tex address operands"),
    };
    let Some(ts) = handles(&a.op, s) else { return err("tex handle") };
    let mut rest = &ops[2..];
    let (mut av, mut bv) = (zero4(), zero4());
    if !gather {
        if nl == 1 {
            let Some(l) = rest.first() else { return err("tex.level operand") };
            av = vec4(&l.op);
            rest = &rest[1..];
        } else if nl == 2 {
            let [x, y, r2 @ ..] = rest else { return err("tex.grad operands") };
            av = vec4(&x.op);
            bv = vec4(&y.op);
            rest = r2;
        }
    }
    // {offsets} then the depth reference
    let mut cv: Vec<Option<ast::RegOrImmediate<&str>>> = vec![
        Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::U64(0))),
        Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::U64(0))),
        Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::U64(0))),
        Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::U64(0))),
    ];
    let mut ctrl = g | (nl << 4) | ((ct == "f32") as u64) << 6 | ((s.is_some()) as u64) << 7 | kind << 10;
    if let Some(o) = rest.first() {
        if let ast::ParsedOperand::VecPack(v) = &o.op {
            for (i, x) in v.iter().take(3).enumerate() {
                cv[i] = x.clone();
            }
            ctrl |= 1 << 8;
            rest = &rest[1..];
        }
    }
    if let Some(r) = rest.first() {
        match &r.op {
            ast::ParsedOperand::Reg(x) => cv[3] = Some(ast::RegOrImmediate::Reg(*x)),
            ast::ParsedOperand::Imm(x) => cv[3] = Some(ast::RegOrImmediate::Imm(*x)),
            _ => return err("tex depth reference"),
        }
        ctrl |= 1 << 9;
    }
    if gather {
        let comp = if m.contains(&"g") {
            1
        } else if m.contains(&"b") {
            2
        } else if m.contains(&"a") {
            3
        } else {
            0
        };
        ctrl |= comp << 13 | 1 << 15;
    }
    let (name, n, et) = match kind {
        3 => ("tex_v4_16", 4, ScalarType::B16),
        4 => ("tex_v2_f16x2", 2, ScalarType::B32),
        _ => ("tex_v4_32", 4, ScalarType::B32),
    };
    let dst = match &d.op {
        ast::ParsedOperand::VecPack(v) => {
            let mut v = v.clone();
            // fewer destinations than the helper's vector: the rest are sinks
            while v.len() < n as usize {
                v.push(None);
            }
            ast::ParsedOperand::VecPack(v)
        }
        o => o.clone(),
    };
    let _ = vec_len;
    Call::new(name)
        .dst_vec(dst, n, et)
        .src_vec(ts, 2, ScalarType::B64)
        .src_vec(vec4(coords), 4, ScalarType::B32)
        .src_vec(av, 4, ScalarType::B32)
        .src_vec(bv, 4, ScalarType::B32)
        .src_vec(ast::ParsedOperand::VecPack(cv), 4, ScalarType::B32)
        .imm(ctrl)
        .build()
}

fn attr_code(m: &[&str]) -> Option<u64> {
    let names = [
        "width",
        "height",
        "depth",
        "array_size",
        "normalized_coords",
        "filter_mode",
        "addr_mode_0",
        "num_mipmap_levels",
        "num_samples",
        "force_unnormalized_coords",
        "channel_data_type",
        "channel_order",
        "memory_layout",
    ];
    for (i, n) in names.iter().enumerate() {
        if m.contains(n) {
            return Some(i as u64);
        }
    }
    if m.contains(&"addr_mode_1") || m.contains(&"addr_mode_2") {
        return Some(6);
    }
    None
}

/// surface instructions: ctrl 0-3 geometry, 4-7 vector length, 8-11 element bytes, 12-13 mode, 14 .p, 16-19 op,
/// 20 signed
fn surface<'i>(base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    let g = geom(m)?;
    let n: u64 = if m.contains(&"v4") {
        4
    } else if m.contains(&"v2") {
        2
    } else {
        1
    };
    let t = m.iter().rev().copied().find(|x| scalar(x).is_some())?;
    let eb: u64 = match t {
        "b8" | "u8" | "s8" => 1,
        "b16" | "u16" | "s16" => 2,
        "b64" | "u64" | "s64" => 8,
        _ => 4,
    };
    let mode: u64 = if m.contains(&"clamp") {
        1
    } else if m.contains(&"zero") {
        2
    } else {
        0
    };
    let p = m.contains(&"p");
    let mut ctrl = g | n << 4 | eb << 8 | mode << 12 | (p as u64) << 14 | (is_signed(t) as u64) << 20;
    let wide = eb == 8;
    let et = if wide { ScalarType::B64 } else { ScalarType::B32 };
    match base {
        "suld" => {
            let [d, a] = ops else { return err("suld operands") };
            let [c] = &a.items[..] else { return err("suld address") };
            let name = match (n, wide) {
                (1, false) => "suld_v1_32",
                (2, false) => "suld_v2_32",
                (_, false) => "suld_v_32",
                (1, true) => "suld_v1_64",
                (2, true) => "suld_v2_64",
                (_, true) => "suld_v4_64",
            };
            let call = Call::new(name);
            let call = match (&d.op, n) {
                (ast::ParsedOperand::VecPack(v), 1) => {
                    let Some(Some(ast::RegOrImmediate::Reg(r))) = v.first() else { return err("suld destination") };
                    call.dst(ast::ParsedOperand::Reg(*r), et)
                }
                (o, 1) => call.dst(o.clone(), et),
                (o, k) => {
                    let mut o = o.clone();
                    if let ast::ParsedOperand::VecPack(v) = &mut o {
                        while v.len() < k as usize {
                            v.push(None);
                        }
                    }
                    call.dst_vec(o, if wide { k as u8 } else if k == 2 { 2 } else { 4 }, et)
                }
            };
            call.src(a.op.clone(), ScalarType::B64).src_vec(vec4(c), 4, ScalarType::B32).imm(ctrl).build()
        }
        "sust" => {
            let [a, v] = ops else { return err("sust operands") };
            let [c] = &a.items[..] else { return err("sust address") };
            if p {
                ctrl = (ctrl & !(0xF << 4)) | (vec_len(&v.op) as u64) << 4;
            }
            Call::new(if wide { "sust_64" } else { "sust_32" })
                .src(a.op.clone(), ScalarType::B64)
                .src_vec(vec4(c), 4, ScalarType::B32)
                .src_vec(vec4(&v.op), 4, if p { ScalarType::B32 } else { et })
                .imm(ctrl)
                .build()
        }
        "sured" => {
            let [a, v] = ops else { return err("sured operands") };
            let [c] = &a.items[..] else { return err("sured address") };
            let op = ["add", "min", "max", "and", "or", "inc", "dec", "exch", "xor"].iter().position(|x| m.contains(x))? as u64;
            ctrl |= op << 16;
            Call::new(if wide { "sured_64" } else { "sured_32" })
                .src(a.op.clone(), ScalarType::B64)
                .src_vec(vec4(c), 4, ScalarType::B32)
                .src(v.op.clone(), et)
                .imm(ctrl)
                .build()
        }
        _ => None,
    }
}

pub(super) fn lower<'i>(_opcode: &str, base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    match base {
        "tex" | "tld4" => tex(base, m, ops),
        "txq" | "suq" => {
            let attr = attr_code(m)?;
            let (d, a, lod) = match ops {
                [d, a] => (d, a, imm(0)),
                [d, a, l] => (d, a, l.op.clone()),
                _ => return err("txq operands"),
            };
            if base == "suq" {
                return Call::new("suq").dst(d.op.clone(), ScalarType::B32).src(a.op.clone(), ScalarType::B64).imm(attr).build();
            }
            Call::new("txq")
                .dst(d.op.clone(), ScalarType::B32)
                .src(a.op.clone(), ScalarType::B64)
                .src(lod, ScalarType::B32)
                .imm(attr)
                .build()
        }
        "suld" | "sust" | "sured" => surface(base, m, ops),
        "istypep" => {
            let k = ["texref", "samplerref", "surfref"].iter().position(|x| m.contains(x))? as u64;
            let [d, a] = ops else { return err("istypep operands") };
            Call::new("istypep").dst(d.op.clone(), ScalarType::Pred).src(a.op.clone(), ScalarType::B64).imm(k).build()
        }
        _ => None,
    }
}

/// `tex ... {d}|p, ...`: p is set when the texture is resident (always: the instruction that follows)
pub(super) fn extra<'i>(base: &str, ops: &[HOp<'i>]) -> Vec<ast::Instruction<ParsedOperandStr<'i>>> {
    if !matches!(base, "tex" | "tld4") {
        return Vec::new();
    }
    match ops.first().and_then(|d| d.pair.clone()) {
        Some(p) => match Call::new("tx_true").dst(p, ScalarType::Pred).build() {
            Some(Ok(i)) => vec![i],
            _ => Vec::new(),
        },
        None => Vec::new(),
    }
}
