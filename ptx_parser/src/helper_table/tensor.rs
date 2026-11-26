//! warpgroup and tensor-memory matrix instructions: wgmma.*, tcgen05.*.
//!
//! The helpers (ptx/lib/metal_impl/tensor.inc) follow PTX-formalization's Sail model
//! (sail/ptx/ptx_wg.sail, ptx_tc.sail) and reuse the exact arithmetic of the warp matrix engine
//! (metal_impl/matrix.inc). Asynchronous operations complete at issue, so the fences, commits
//! and waits are no-ops.
//!
//! wgmma.mma_async: every thread of the warpgroup computes its own accumulator elements; the
//! A / B tiles of the shared-memory matrix descriptors are read from threadgroup memory, A
//! fragments in registers and the sparsity metadata come from the lanes of the thread's own
//! warp (simd shuffles). The accumulator is a vector of 8 .. 128 b32 (padded), passed through a
//! per-thread scratch area of the CTA's per-CTA global memory.
use super::*;

type VecElems<'i> = Vec<Option<ast::RegOrImmediate<&'i str>>>;

fn elems<'i>(op: &Op<'i>) -> Option<VecElems<'i>> {
    Some(match op {
        ast::ParsedOperand::VecPack(v) => v.clone(),
        ast::ParsedOperand::Reg(r) => vec![Some(ast::RegOrImmediate::Reg(*r))],
        ast::ParsedOperand::Imm(i) => vec![Some(ast::RegOrImmediate::Imm(*i))],
        _ => return None,
    })
}

fn op_imm(op: &Op) -> Option<i64> {
    match op {
        ast::ParsedOperand::Imm(ast::ImmediateValue::U64(v)) => Some(*v as i64),
        ast::ParsedOperand::Imm(ast::ImmediateValue::S64(v)) => Some(*v),
        _ => None,
    }
}

fn zero<'i>() -> Option<ast::RegOrImmediate<&'i str>> {
    Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::U64(0)))
}

fn immv<'i>(v: u64) -> Option<ast::RegOrImmediate<&'i str>> {
    Some(ast::RegOrImmediate::Imm(ast::ImmediateValue::U64(v)))
}

/// a vector operand padded to `n` elements: zeros for sources, sinks for destinations
fn pad<'i>(mut v: VecElems<'i>, n: usize, dst: bool) -> Result<Op<'i>, String> {
    if v.len() > n {
        return Err(format!("vector operand with {} elements (max {n})", v.len()));
    }
    while v.len() < n {
        v.push(if dst { None } else { zero() });
    }
    Ok(ast::ParsedOperand::VecPack(v))
}

impl<'i> Call<'i> {
    fn tvdst(mut self, op: Op<'i>, n: u8, t: ScalarType) -> Self {
        self.data = self.data.with_type(0, ast::Type::Vector(n, t), StateSpace::Reg);
        self.slots[0] = Some(op);
        self
    }
    fn tvsrc(mut self, op: Op<'i>, n: u8, t: ScalarType) -> Self {
        let i = self.next_src;
        self.data = self.data.with_type(i, ast::Type::Vector(n, t), StateSpace::Reg);
        self.slots[i] = Some(op);
        self.next_src += 1;
        self
    }
}

fn nop<'i>() -> R<'i> {
    Call::new("tensor_nop").build()
}

/// A/B element type code (matrix.inc MX_*)
fn ab_code(t: &str) -> Option<u32> {
    Some(match t {
        "f16" => 0,
        "bf16" => 1,
        "tf32" => 2,
        "e4m3" => 3,
        "e5m2" => 4,
        "e2m3" => 5,
        "e3m2" => 6,
        "e2m1" => 7,
        "s8" => 9,
        "u8" => 10,
        "b1" => 13,
        _ => return None,
    })
}

fn vec_size(n: usize) -> usize {
    [8, 16, 32, 64, 128].into_iter().find(|&s| s >= n).unwrap_or(0)
}

fn lower_wgmma<'i>(m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    if m.first() == Some(&"fence") || m.first() == Some(&"commit_group") || m.first() == Some(&"wait_group") {
        return nop();
    }
    if m.first() != Some(&"mma_async") {
        return None;
    }
    let sp = m.contains(&"sp");
    // wgShape: m64nNkK
    let Some((n, k)) = m.iter().find_map(|s| {
        let r = s.strip_prefix("m64n")?;
        let (nn, kk) = r.split_once('k')?;
        Some((nn.parse::<u32>().ok()?, kk.parse::<u32>().ok()?))
    }) else {
        return err("wgmma.mma_async shape");
    };
    if !(8..=256).contains(&n) || n % 8 != 0 || ![8, 16, 32, 64, 256].contains(&k) {
        return err("wgmma.mma_async: invalid shape");
    }
    let tys: Vec<&str> = m
        .iter()
        .copied()
        .filter(|x| ["f16", "bf16", "tf32", "f32", "e4m3", "e5m2", "s8", "u8", "s32", "b1"].contains(x))
        .collect();
    if tys.len() != 3 {
        return err("wgmma.mma_async types");
    }
    let (dt, at, bt) = (tys[0], tys[1], tys[2]);
    let dc: u32 = match dt {
        "f16" => 0,
        "f32" => 1,
        "s32" => 2,
        _ => return err("wgmma accumulator type"),
    };
    let (Some(ac), Some(bc)) = (ab_code(at), ab_code(bt)) else { return err("wgmma element type") };
    let nfix = if sp { 5 } else { 3 };
    if ops.len() < nfix {
        return err("wgmma.mma_async operands");
    }
    let Some(d) = elems(&ops[0].op) else { return err("wgmma accumulator") };
    let nd = if dc == 0 { n / 4 } else { n / 2 } as usize;
    if d.len() != nd {
        return err(format!("wgmma accumulator of {} registers, expected {nd}", d.len()));
    }
    let aregs = matches!(ops[1].op, ast::ParsedOperand::VecPack(_));
    let rest = &ops[nfix..];
    let nr = rest.len();
    let arg = |i: Option<usize>, dflt: i64| -> Result<i64, String> {
        match i {
            Some(i) if i < nr => op_imm(&rest[i].op).ok_or_else(|| "wgmma immediate operand".to_string()),
            _ => Ok(dflt),
        }
    };
    let r = (|| -> Result<_, String> {
        let sca = arg(Some(1), 1)?;
        let scb = arg(Some(2), 1)?;
        let tra = arg(Some(3), 0)?;
        let trb = arg(if nr >= 4 { Some(nr - 1) } else { None }, 0)?;
        // scale-d: an immediate or a predicate
        let mut x: VecElems = vec![zero(); 8];
        let sdreg = if nr == 0 {
            x[5] = immv(1);
            None
        } else if let Some(v) = op_imm(&rest[0].op) {
            x[5] = immv((v != 0) as u64);
            None
        } else {
            Some(rest[0].op.clone())
        };
        let mut sel = 0u64;
        if sp {
            let e = match elems(&ops[3].op) {
                Some(v) if v.len() == 1 => v[0].clone(),
                _ => return Err("wgmma.sp metadata".into()),
            };
            x[4] = e;
            sel = op_imm(&ops[4].op).ok_or("wgmma.sp selector must be an immediate")? as u64;
        }
        let desc_a = if aregs {
            let a = elems(&ops[1].op).ok_or("wgmma A")?;
            if a.len() > 4 {
                return Err("wgmma A fragment".into());
            }
            for (i, e) in a.into_iter().enumerate() {
                x[i] = e;
            }
            imm(0)
        } else {
            ops[1].op.clone()
        };
        let bop: u32 = if m.contains(&"xor") {
            1
        } else if m.contains(&"and") {
            2
        } else {
            0
        };
        let ctl0 = n
            | (k << 9)
            | (ac << 18)
            | (bc << 22)
            | (dc << 26)
            | ((sp as u32) << 28)
            | ((aregs as u32) << 29)
            | ((m.contains(&"satfinite") as u32) << 30);
        let ctl1 = bop
            | (((sca < 0) as u32) << 2)
            | (((scb < 0) as u32) << 3)
            | (((tra != 0) as u32) << 4)
            | (((trb != 0) as u32) << 5)
            | ((sel as u32 & 3) << 6)
            | ((sdreg.is_some() as u32) << 8);
        let nv = vec_size(nd);
        let mut c = Call::new(format!("wgmma_{nv}{}", if sdreg.is_some() { "_p" } else { "" }))
            .tvdst(pad(d.clone(), nv, true)?, nv as u8, ScalarType::B32)
            .tvsrc(pad(d, nv, false)?, nv as u8, ScalarType::B32)
            .tvsrc(ast::ParsedOperand::VecPack(x), 8, ScalarType::B32)
            .src(desc_a, ScalarType::B64)
            .src(ops[2].op.clone(), ScalarType::B64);
        if let Some(p) = sdreg {
            c = c.src(p, ScalarType::Pred);
        }
        Ok(c.src(imm((ctl1 as u64) << 32 | ctl0 as u64), ScalarType::U64))
    })();
    match r {
        Ok(c) => c.build(),
        Err(e) => err(e),
    }
}

/// a tensor memory address operand `[taddr]`, `[taddr+off]`: (register or immediate, offset)
fn taddr<'i>(h: &HOp<'i>) -> Result<(Op<'i>, u64), String> {
    match &h.op {
        ast::ParsedOperand::Reg(r) => Ok((ast::ParsedOperand::Reg(*r), 0)),
        ast::ParsedOperand::RegOffset(r, o) => Ok((ast::ParsedOperand::Reg(*r), *o as u64 & 0xFFFF_FFFF)),
        ast::ParsedOperand::Imm(_) => Ok((h.op.clone(), 0)),
        _ => Err("tensor memory address".into()),
    }
}

/// a 32-bit value operand as a vector element (register or immediate)
fn elem<'i>(op: &Op<'i>) -> Result<Option<ast::RegOrImmediate<&'i str>>, String> {
    match op {
        ast::ParsedOperand::Reg(r) => Ok(Some(ast::RegOrImmediate::Reg(*r))),
        ast::ParsedOperand::Imm(i) => Ok(Some(ast::RegOrImmediate::Imm(*i))),
        _ => Err("tcgen05 operand".into()),
    }
}

fn has(m: &[&str], s: &str) -> bool {
    m.contains(&s)
}

/// ld / st shape code: 0 .32x32b, 1 .16x64b, 2 .16x128b, 3 .16x256b, 4 .16x32bx2
fn ldst_shape(m: &[&str]) -> Option<u64> {
    Some(if has(m, "32x32b") {
        0
    } else if has(m, "16x64b") {
        1
    } else if has(m, "16x128b") {
        2
    } else if has(m, "16x256b") {
        3
    } else if has(m, "16x32bx2") {
        4
    } else {
        return None;
    })
}

fn tc_num(m: &[&str]) -> u64 {
    for (s, n) in [("x128", 128), ("x64", 64), ("x32", 32), ("x16", 16), ("x8", 8), ("x4", 4), ("x2", 2)] {
        if has(m, s) {
            return n;
        }
    }
    1
}

fn tc_per(m: &[&str]) -> u64 {
    if has(m, "16x256b") {
        4
    } else if has(m, "16x128b") {
        2
    } else {
        1
    }
}

fn lower_tc_ldst<'i>(ld: bool, m: &[&str], ops: &[HOp<'i>]) -> Result<ast::Instruction<ParsedOperandStr<'i>>, String> {
    let shape = ldst_shape(m).ok_or("tcgen05.ld/st shape")?;
    let red = has(m, "red");
    let spc = has(m, "spcompress");
    // ctl: shape | n << 3 | pack << 11 | red << 12 | min << 13 | abs << 14 | NaN << 15 | red type << 16 (0 f32,
    //      1 u32, 2 s32) | spcompress << 18 | count of the reduced values << 19
    let rty = if has(m, "u32") {
        1
    } else if has(m, "s32") {
        2
    } else {
        0
    };
    let nred = tc_num(m) * tc_per(m);
    let mut ctl = shape
        | ((has(m, "pack::16b") || has(m, "unpack::16b")) as u64) << 11
        | (red as u64) << 12
        | (has(m, "min") as u64) << 13
        | (has(m, "abs") as u64) << 14
        | (has(m, "NaN") as u64) << 15
        | rty << 16
        | (spc as u64) << 18
        | nred << 19;
    if ld {
        // d, [a] (, off) | d, rv, [a] (, off) | md, cd (, rv), [a]
        let ai = ops.iter().position(|h| h.addr).ok_or("tcgen05.ld address")?;
        let (a, aoff) = taddr(&ops[ai])?;
        let h = match ops.get(ai + 1) {
            Some(x) => x.op.clone(),
            None => imm(0),
        };
        let mut out: VecElems = vec![];
        let mut nv = 0usize;
        if spc {
            // md, cd (, rv), [a]: the metadata words, the kept values, the reduction
            if has(m, "b4") || !has(m, "f32") || !has(m, "sp::2:4") {
                return Err("tcgen05.ld.spcompress form".into());
            }
            let md = elems(&ops[0].op).ok_or("tcgen05.ld.spcompress metadata")?;
            let cd = elems(&ops.get(1).ok_or("tcgen05.ld.spcompress values")?.op).ok_or("tcgen05.ld.spcompress values")?;
            if ai != if red { 3 } else { 2 } {
                return Err("tcgen05.ld.spcompress operands".into());
            }
            ctl |= (md.len() as u64) << 32 | (cd.len() as u64) << 40;
            out.extend(md);
            out.extend(cd);
            if red {
                out.extend(elems(&ops[2].op).ok_or("tcgen05.ld.red destination")?);
            }
            let n = vec_size(out.len());
            return Call::new(format!("tcgen05_ldspc_{n}"))
                .tvdst(pad(out, n, true)?, n as u8, ScalarType::B32)
                .src(a, ScalarType::B32)
                .src(h, ScalarType::B32)
                .src(imm(aoff), ScalarType::U32)
                .src(imm(ctl), ScalarType::U64)
                .build()
                .unwrap();
        }
        let d = elems(&ops[0].op).ok_or("tcgen05.ld destination")?;
        nv = nv.max(d.len());
        ctl |= (d.len() as u64) << 3;
        out.extend(d);
        if red {
            if ai != 2 {
                return Err("tcgen05.ld.red operands".into());
            }
            let rv = elems(&ops[1].op).ok_or("tcgen05.ld.red destination")?;
            out.extend(rv);
        } else if ai != 1 {
            return Err("tcgen05.ld operands".into());
        }
        let n = vec_size(out.len());
        if n == 0 {
            return Err("tcgen05.ld: too many registers".into());
        }
        let _ = nv;
        Call::new(format!("tcgen05_ld_{n}"))
            .tvdst(pad(out, n, true)?, n as u8, ScalarType::B32)
            .src(a, ScalarType::B32)
            .src(h, ScalarType::B32)
            .src(imm(aoff), ScalarType::U32)
            .src(imm(ctl), ScalarType::U64)
            .build()
            .unwrap()
    } else {
        // [a], (off,) v
        if !ops[0].addr || ops.len() < 2 {
            return Err("tcgen05.st operands".into());
        }
        let (a, aoff) = taddr(&ops[0])?;
        let h = if ops.len() == 3 { ops[1].op.clone() } else { imm(0) };
        let v = elems(&ops[ops.len() - 1].op).ok_or("tcgen05.st source")?;
        ctl |= (v.len() as u64) << 3;
        let n = vec_size(v.len());
        if n == 0 {
            return Err("tcgen05.st: too many registers".into());
        }
        Call::new(format!("tcgen05_st_{n}"))
            .src(a, ScalarType::B32)
            .src(h, ScalarType::B32)
            .src(imm(aoff), ScalarType::U32)
            .tvsrc(pad(v, n, false)?, n as u8, ScalarType::B32)
            .src(imm(ctl), ScalarType::U64)
            .build()
            .unwrap()
    }
}

/// mma kind code: 0 f16, 1 tf32, 2 f8f6f4, 3 i8, 4 ti16, 5 mxf8f6f4, 6 mxf4, 7 mxf4nvf4
fn mma_kind(m: &[&str]) -> Option<u64> {
    Some(if has(m, "kind::f16") {
        0
    } else if has(m, "kind::tf32") {
        1
    } else if has(m, "kind::f8f6f4") {
        2
    } else if has(m, "kind::i8") {
        3
    } else if has(m, "kind::ti16") {
        4
    } else if has(m, "kind::mxf8f6f4") {
        5
    } else if has(m, "kind::mxf4") {
        6
    } else if has(m, "kind::mxf4nvf4") {
        7
    } else {
        return None;
    })
}

fn lower_tc_mma<'i>(m: &[&str], ops: &[HOp<'i>]) -> Result<ast::Instruction<ParsedOperandStr<'i>>, String> {
    if has(m, "cta_group::2") {
        return Err("tcgen05.mma.cta_group::2 needs thread-block clusters".into());
    }
    let kind = mma_kind(m).ok_or("tcgen05.mma kind")?;
    let sp = has(m, "sp");
    let lut = has(m, "decompress::lut::b");
    let bs = has(m, "block_scale");
    let ws = has(m, "ws");
    // x: d taddr, d off, A taddr, A off, idesc, meta taddr, meta off, lut taddr, lut off, scale-A taddr, off,
    //    scale-B taddr, off, enable-input-d, scale-input-d, 0, disable-output-lane mask 16..23
    let mut x: VecElems = vec![zero(); 32];
    if ops.len() < 4 || !ops[0].addr {
        return Err("tcgen05.mma operands".into());
    }
    let (d, doff) = taddr(&ops[0])?;
    x[0] = elem(&d)?;
    x[1] = immv(doff);
    let ad = !ops[1].addr;
    let desc_a = if ad {
        ops[1].op.clone()
    } else {
        let (a, aoff) = taddr(&ops[1])?;
        x[2] = elem(&a)?;
        x[3] = immv(aoff);
        imm(0)
    };
    let desc_b = ops[2].op.clone();
    let mut i = 3;
    if sp || lut {
        if !ops.get(i).map_or(false, |h| h.addr) {
            return Err("tcgen05.mma metadata / lookup table address".into());
        }
        let (t, toff) = taddr(&ops[i])?;
        let k = if sp { 5 } else { 7 };
        x[k] = elem(&t)?;
        x[k + 1] = immv(toff);
        i += 1;
    }
    x[4] = elem(&ops.get(i).ok_or("tcgen05.mma idesc")?.op)?;
    i += 1;
    let mut en: Option<Op> = None;
    let mut zm = imm(0);
    let mut has_zm = false;
    let mut has_mask = false;
    if bs {
        for k in [9usize, 11] {
            let h = ops.get(i).ok_or("tcgen05.mma scale address")?;
            if !h.addr {
                return Err("tcgen05.mma scale address".into());
            }
            let (t, toff) = taddr(h)?;
            x[k] = elem(&t)?;
            x[k + 1] = immv(toff);
            i += 1;
        }
        en = ops.get(i).map(|h| h.op.clone());
    } else {
        if let Some(h) = ops.get(i) {
            if let ast::ParsedOperand::VecPack(v) = &h.op {
                if v.len() > 8 {
                    return Err("tcgen05.mma disable-output-lane".into());
                }
                for (j, e) in v.iter().enumerate() {
                    x[16 + j] = e.clone();
                }
                has_mask = true;
                i += 1;
            }
        }
        en = ops.get(i).map(|h| h.op.clone());
        if let Some(h) = ops.get(i + 1) {
            if ws {
                zm = h.op.clone();
                has_zm = true;
            } else {
                x[14] = elem(&h.op)?;
            }
        }
    }
    // enable-input-d: an immediate or a predicate
    let en_reg = match en {
        None => {
            x[13] = immv(1);
            None
        }
        Some(ast::ParsedOperand::Imm(v)) => {
            let nz = match v {
                ast::ImmediateValue::U64(v) => v != 0,
                ast::ImmediateValue::S64(v) => v != 0,
                _ => true,
            };
            x[13] = immv(nz as u64);
            None
        }
        Some(p) => Some(p),
    };
    let mut ctl = kind
        | (sp as u64) << 3
        | (lut as u64) << 4
        | (bs as u64) << 5
        | (ws as u64) << 6
        | (ad as u64) << 7
        | (has(m, "ashift") as u64) << 8
        | (has_zm as u64) << 9
        | (has_mask as u64) << 10
        | (en_reg.is_some() as u64) << 11;
    // block scale vector size: 0 default, 1 block16, 2 scale_vec::2X, 3 scale_vec::4X
    ctl |= (if has(m, "block16") {
        1
    } else if has(m, "scale_vec::2X") {
        2
    } else if has(m, "scale_vec::4X") {
        3
    } else {
        0
    }) << 12;
    let mut c = Call::new(if en_reg.is_some() { "tcgen05_mma_p" } else { "tcgen05_mma" })
        .tvsrc(ast::ParsedOperand::VecPack(x), 32, ScalarType::B32)
        .src(desc_a, ScalarType::B64)
        .src(desc_b, ScalarType::B64)
        .src(zm, ScalarType::B64);
    if let Some(p) = en_reg {
        c = c.src(p, ScalarType::Pred);
    }
    Ok(c.src(imm(ctl), ScalarType::U64).build().unwrap()?)
}

fn lower_tcgen05<'i>(m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    let r = (|| -> Result<ast::Instruction<ParsedOperandStr<'i>>, String> {
        let first = m.first().copied().unwrap_or("");
        if first.starts_with("fence::") || first.starts_with("wait::") || first == "relinquish_alloc_permit" {
            return Ok(nop().unwrap()?);
        }
        if first == "alloc" || first == "dealloc" {
            let ex = has(m, "exclusive");
            if has(m, "cta_group::2") {
                return Err("tcgen05.alloc.cta_group::2 needs thread-block clusters".into());
            }
            if ops.len() != 2 {
                return Err("tcgen05.alloc operands".into());
            }
            if first == "alloc" {
                if !ops[0].addr {
                    return Err("tcgen05.alloc destination".into());
                }
                let shared = m.iter().any(|x| x.starts_with("shared"));
                let (sp, sfx) = if shared { (StateSpace::Shared, "") } else { (StateSpace::Generic, "_g") };
                return Ok(Call::new(format!("tcgen05_alloc{sfx}"))
                    .src_space(ops[0].op.clone(), ScalarType::B32, sp)
                    .src(ops[1].op.clone(), ScalarType::U32)
                    .imm(ex as u64)
                    .build()
                    .unwrap()?);
            }
            return Ok(Call::new("tcgen05_dealloc")
                .src(ops[0].op.clone(), ScalarType::B32)
                .src(ops[1].op.clone(), ScalarType::U32)
                .imm(ex as u64)
                .build()
                .unwrap()?);
        }
        if first == "ld" || first == "st" {
            return lower_tc_ldst(first == "ld", m, ops);
        }
        if first == "mma" {
            return lower_tc_mma(m, ops);
        }
        if first == "commit" {
            // an arrive-on operation on the mbarrier (the memory family's mbarrier objects)
            if has(m, "cta_group::2") || m.iter().any(|x| x.starts_with("multicast")) {
                return Err("tcgen05.commit .multicast / cta_group::2 need thread-block clusters".into());
            }
            if ops.len() != 1 || !ops[0].addr {
                return Err("tcgen05.commit operands".into());
            }
            let shared = m.iter().any(|x| x.starts_with("shared"));
            let (sp, sfx) = if shared { (StateSpace::Shared, "") } else { (StateSpace::Generic, "_g") };
            return Ok(Call::new(format!("mbar_arrive_nd{sfx}"))
                .src_space(ops[0].op.clone(), ScalarType::B64, sp)
                .imm(1)
                .imm(0)
                .imm(0)
                .build()
                .unwrap()?);
        }
        if first == "cp" || first == "shift" || m.contains(&"shift") {
            if has(m, "cta_group::2") {
                return Err("tcgen05.cp / shift .cta_group::2 need thread-block clusters".into());
            }
            if ops.is_empty() || !ops[0].addr {
                return Err("tcgen05.cp / shift operands".into());
            }
            let (a, aoff) = taddr(&ops[0])?;
            if first == "cp" {
                if ops.len() != 2 {
                    return Err("tcgen05.cp operands".into());
                }
                // ctl: rows code | words << 8 | dst pattern << 12 | decompression << 16
                let rows: u64 = if has(m, "128x256b") || has(m, "128x128b") {
                    128
                } else if has(m, "4x256b") {
                    4
                } else if has(m, "64x128b") {
                    64
                } else if has(m, "32x128b") {
                    32
                } else {
                    return Err("tcgen05.cp shape".into());
                };
                let words: u64 = if has(m, "128x256b") || has(m, "4x256b") { 8 } else { 4 };
                let pat: u64 = if has(m, "warpx4") {
                    1
                } else if has(m, "warpx2::02_13") {
                    2
                } else if has(m, "warpx2::01_23") {
                    3
                } else {
                    0
                };
                let dec: u64 = if has(m, "b6x16_p32") {
                    6
                } else if has(m, "b4x16_p64") {
                    4
                } else {
                    0
                };
                return Ok(Call::new("tcgen05_cp")
                    .src(a, ScalarType::B32)
                    .src(imm(aoff), ScalarType::U32)
                    .src(ops[1].op.clone(), ScalarType::B64)
                    .src(imm(rows | words << 8 | pat << 12 | dec << 16), ScalarType::U32)
                    .build()
                    .unwrap()?);
            }
            return Ok(Call::new("tcgen05_shift")
                .src(a, ScalarType::B32)
                .src(imm(aoff), ScalarType::U32)
                .build()
                .unwrap()?);
        }
        Err(format!("tcgen05.{first}"))
    })();
    Some(r)
}

pub(super) fn lower<'i>(_opcode: &str, base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    match base {
        "wgmma" => lower_wgmma(m, ops),
        "tcgen05" => lower_tcgen05(m, ops),
        _ => None,
    }
}
