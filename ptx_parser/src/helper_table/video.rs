//! Video instructions (scalar, SIMD, vset, vmad): PTX ISA 9.7.10, Sail ptx_half.sail.
use super::*;

fn video_impl<'i>(base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
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


pub(super) fn lower<'i>(_opcode: &str, base: &str, m: &[&str], ops: &[HOp<'i>]) -> R<'i> {
    match base {
        "vadd" | "vsub" | "vabsdiff" | "vmin" | "vmax" | "vshl" | "vshr" | "vset" | "vmad" | "vadd2"
        | "vsub2" | "vavrg2" | "vabsdiff2" | "vmin2" | "vmax2" | "vset2" | "vadd4" | "vsub4"
        | "vavrg4" | "vabsdiff4" | "vmin4" | "vmax4" | "vset4" => video_impl(base, m, ops),
        _ => None,
    }
}
