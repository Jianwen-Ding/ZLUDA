//! lop3, testp, slct and other single instructions.
use super::*;

pub(super) fn lower<'i>(_opcode: &str, base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    match base {
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
