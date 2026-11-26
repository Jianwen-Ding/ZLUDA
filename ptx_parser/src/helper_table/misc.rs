//! lop3, testp, slct and other single instructions.
use super::*;

/// tried before the grammar: `mov` of special registers the grammar's types reject
pub(super) fn preempts(_opcode: &str, base: &str, m: &[&str]) -> bool {
    base == "mov" && (m.contains(&"v4") || m.contains(&"v2") || m == ["u32"] || m == ["b32"] || m == ["s32"])
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
