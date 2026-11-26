//! lop3, testp, slct and other single instructions.
use super::*;

/// tried before the grammar: `mov` of special registers the grammar's types reject
pub(super) fn preempts(_opcode: &str, base: &str, m: &[&str]) -> bool {
    base == "bmsk"
        // packed integer lanes (float.rs ipack): the grammar's lowering is a 32-bit operation
        || matches!(base, "add" | "sub" | "min" | "max" | "neg" | "abs")
            && m.iter().any(|x| matches!(*x, "u8x4" | "s8x4" | "u16x2" | "s16x2"))
        || base == "dp4a" && m.len() == 2 && m[0] != m[1]
        || (base == "bar" || base == "barrier") && m.contains(&"red")
        || base == "mov" && (m.contains(&"v4") || m.contains(&"v2") || m == ["u32"] || m == ["b32"] || m == ["s32"])
}

const VEC_SREGS: [&str; 8] =
    ["%tid", "%ntid", "%ctaid", "%nctaid", "%clusterid", "%nclusterid", "%cluster_ctaid", "%cluster_nctaid"];
const PRED_SREGS: [&str; 1] = ["%is_explicit_cluster"];

/// mov.v4.u32 {a, b, c, d}, %tid;  (the members .x .y .z, then 0);  mov.u32 d, %is_explicit_cluster;  (0 / 1)
fn mov_sreg<'i>(m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    let [d, a] = ops else { return None };
    let ast::ParsedOperand::Reg(src) = a.op else { return None };
    if PRED_SREGS.contains(&src) && !d.addr && matches!(d.op, ast::ParsedOperand::Reg(_)) && m.len() == 1 {
        return Some(Ok(ast::Instruction::Selp {
            data: scalar(m[0])?,
            arguments: ast::SelpArgs { dst: d.op.clone(), src1: imm(1), src2: imm(0), src3: a.op.clone() },
        }));
    }
    None
}

/// the movs of a vector special register read (mov_sreg): the first member is the instruction itself
pub(super) fn mov_sreg_members<'i>(m: &[&str], ops: &[HOp<'i>]) -> Option<Vec<ast::Instruction<ParsedOperandStr<'i>>>> {
    let [d, a] = ops else { return None };
    let ast::ParsedOperand::Reg(src) = a.op else { return None };
    if !VEC_SREGS.contains(&src) || !(m.contains(&"v4") || m.contains(&"v2")) {
        return None;
    }
    let t = scalar(m.last()?)?;
    let ast::ParsedOperand::VecPack(ds) = &d.op else { return None };
    let mut out = Vec::new();
    for (i, x) in ds.iter().enumerate() {
        let Some(ast::RegOrImmediate::Reg(r)) = x else { continue };
        let src = if i < 3 { ast::ParsedOperand::VecMember(src, i as u8) } else { imm(0) };
        out.push(ast::Instruction::Mov {
            data: ast::MovDetails { typ: ast::Type::Scalar(t) },
            arguments: ast::MovArgs { dst: ast::ParsedOperand::Reg(*r), src },
        });
    }
    if out.is_empty() {
        return None;
    }
    Some(out)
}

pub(super) fn lower<'i>(_opcode: &str, base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    match base {
        "bfind" => {
            // bfind{.shiftamt}.type d, a;
            let t = *m.last()?;
            let [d, a] = ops else { return err("bfind operands") };
            let wide = t.ends_with("64");
            let ctrl = (is_signed(t) as u64) | ((m.contains(&"shiftamt") as u64) << 1);
            Call::new(if wide { "bfind_64" } else { "bfind_32" })
                .dst(d.op.clone(), ScalarType::U32)
                .src(a.op.clone(), if wide { ScalarType::B64 } else { ScalarType::B32 })
                .imm(ctrl)
                .build()
        }
        "fns" => {
            let [d, mk, b, o] = ops else { return err("fns operands") };
            Call::new("fns_b32")
                .dst(d.op.clone(), ScalarType::B32)
                .src(mk.op.clone(), ScalarType::B32)
                .src(b.op.clone(), ScalarType::U32)
                .src(o.op.clone(), ScalarType::S32)
                .build()
        }
        "bmsk" => {
            let [d, a, b] = ops else { return err("bmsk operands") };
            Call::new("bmsk_b32")
                .dst(d.op.clone(), ScalarType::B32)
                .src(a.op.clone(), ScalarType::U32)
                .src(b.op.clone(), ScalarType::U32)
                .imm(m.contains(&"clamp") as u64)
                .build()
        }
        "szext" => {
            let t = *m.last()?;
            let [d, a, b] = ops else { return err("szext operands") };
            Call::new("szext_32")
                .dst(d.op.clone(), ScalarType::B32)
                .src(a.op.clone(), ScalarType::B32)
                .src(b.op.clone(), ScalarType::U32)
                .imm((m.contains(&"clamp") as u64) | ((is_signed(t) as u64) << 1))
                .build()
        }
        "prmt" => {
            let mode = ["f4e", "b4e", "rc8", "ecl", "ecr", "rc16"].iter().position(|x| m.contains(x))? as u64 + 1;
            let [d, a, b, c] = ops else { return err("prmt operands") };
            Call::new("prmt_mode")
                .dst(d.op.clone(), ScalarType::B32)
                .src(a.op.clone(), ScalarType::B32)
                .src(b.op.clone(), ScalarType::B32)
                .src(c.op.clone(), ScalarType::B32)
                .imm(mode)
                .build()
        }
        "mad24" | "mul24" => {
            // mad24.mode{.sat}.type d, a, b, c;  mul24.mode.type d, a, b;
            let t = *m.last()?;
            let (d, a, b, c) = match ops {
                [d, a, b, c] if base == "mad24" => (d, a, b, Some(c)),
                [d, a, b] if base == "mul24" => (d, a, b, None),
                _ => return err("mad24 operands"),
            };
            let ctrl = (is_signed(t) as u64) | ((m.contains(&"hi") as u64) << 1) | ((m.contains(&"sat") as u64) << 2);
            Call::new("mad24")
                .dst(d.op.clone(), ScalarType::B32)
                .src(a.op.clone(), ScalarType::B32)
                .src(b.op.clone(), ScalarType::B32)
                .src(c.map_or(imm(0), |c| c.op.clone()), ScalarType::B32)
                .imm(ctrl)
                .build()
        }
        "dp4a" => {
            // dp4a.atype.btype d, a, b, c;
            let [d, a, b, c] = ops else { return err("dp4a operands") };
            if m.len() != 2 {
                return None;
            }
            let ctrl = (is_signed(m[0]) as u64) | ((is_signed(m[1]) as u64) << 1);
            Call::new("dp4a_mixed")
                .dst(d.op.clone(), ScalarType::B32)
                .src(a.op.clone(), ScalarType::B32)
                .src(b.op.clone(), ScalarType::B32)
                .src(c.op.clone(), ScalarType::B32)
                .imm(ctrl)
                .build()
        }
        "clmad" => {
            let [d, a, b, c] = ops else { return err("clmad operands") };
            Call::new("clmad")
                .dst(d.op.clone(), ScalarType::B64)
                .src(a.op.clone(), ScalarType::B64)
                .src(b.op.clone(), ScalarType::B64)
                .src(c.op.clone(), ScalarType::B64)
                .imm(m.contains(&"hi") as u64)
                .build()
        }
        "bar" | "barrier" if m.contains(&"red") => {
            // bar{.cta}.red.popc{.aligned}.u32 d, a{, b}, {!}c;  bar{.cta}.red.op{.aligned}.pred p, a{, b}, {!}c;
            let kind = ["popc", "and", "or"].iter().position(|k| m.contains(k))? as u64;
            let (d, a, b, c) = match ops {
                [d, a, c] => (d, a, None, c),
                [d, a, b, c] => (d, a, Some(b), c),
                _ => return err("bar.red operands"),
            };
            let call = if kind == 0 {
                Call::new("nbar_red_popc").dst(d.op.clone(), ScalarType::U32)
            } else {
                Call::new("nbar_red_pred").dst(d.op.clone(), ScalarType::Pred)
            };
            call.src(a.op.clone(), ScalarType::U32)
                .src(b.map_or(imm(0), |b| b.op.clone()), ScalarType::U32)
                .src(c.op.clone(), ScalarType::Pred)
                .imm(kind | ((c.neg as u64) << 4))
                .build()
        }
        // ld{.ss}.type.proxy::readonly d, [a];  (PTX ISA 9.4: a hint without effect on the result)
        "ld" if m.contains(&"proxy::readonly") => {
            let [d, a] = ops else { return err("ld operands") };
            if !a.addr {
                return err("ld address");
            }
            let t = m.iter().rev().find_map(|x| scalar(x))?;
            let ss = if m.contains(&"global") {
                StateSpace::Global
            } else if m.contains(&"shared") || m.contains(&"shared::cta") {
                StateSpace::Shared
            } else if m.contains(&"local") {
                StateSpace::Local
            } else if m.contains(&"const") {
                StateSpace::Const
            } else {
                StateSpace::Generic
            };
            Some(Ok(ast::Instruction::Ld {
                data: ast::LdDetails {
                    qualifier: ast::LdStQualifier::Weak,
                    state_space: ss,
                    caching: ast::LdCacheOperator::Cached,
                    typ: ast::Type::Scalar(t),
                    non_coherent: false,
                },
                arguments: ast::LdArgs { dst: d.op.clone(), src: a.op.clone() },
            }))
        }
        "mov" => {
            if let Some(mut v) = mov_sreg_members(m, ops) {
                return Some(Ok(v.remove(0)));
            }
            mov_sreg(m, ops)
        }
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
