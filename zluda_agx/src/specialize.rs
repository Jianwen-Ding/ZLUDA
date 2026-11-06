//! Launch-time specialization: a kernel launched again and again with the same scalar arguments and block size is
//! compiled once more with them as constants (ZLUDA_AGX_SPECIALIZE, see the AGX docs). The PTX of the kernel is
//! rewritten: `ld.param` of a 4-byte scalar parameter (`.u32`, `.s32`, `.b32`, `.f32`) becomes a `mov` of the
//! argument's value; an 8-byte parameter equal to an earlier parameter of the same type reads that earlier parameter;
//! and `mov.u32 %r, %ntid.{x,y,z}` becomes a `mov` of the block's size. Everything else, the other kernels
//! and the parameter list included, stays as it is, so the specialized kernel takes the same arguments.

use std::collections::HashMap;

/// Canonical equality classes of 8-byte launch arguments, appended to the specialization key. Each duplicate is
/// its argument index followed by the first equal earlier index. Values are omitted, so different pointers with
/// the same equality pattern share one compiled kernel.
pub fn equal_eight_byte_args(args: &[&[u8]]) -> Vec<u8> {
    let mut pairs = Vec::new();
    for (i, arg) in args.iter().enumerate() {
        if arg.len() != 8 {
            continue;
        }
        if let Some(j) = args[..i].iter().position(|earlier| earlier.len() == 8 && *earlier == *arg) {
            pairs.extend_from_slice(&(i as u32).to_le_bytes());
            pairs.extend_from_slice(&(j as u32).to_le_bytes());
        }
    }
    pairs
}

/// The module with `kernel` specialized for `args` (the launch's parameter bytes) and `block`, and the number of
/// instructions rewritten; `None` if the kernel is not found or nothing in it depends on them.
pub fn specialize_ptx(src: &str, kernel: &str, args: &[&[u8]], block: [u32; 3]) -> Option<(String, usize)> {
    let entry = find_entry(src, kernel)?;
    let open = entry + src[entry..].find('(')?;
    let close = open + src[open..].find(')')?;
    // index -> (type, name) of scalar parameters we can specialize
    let mut params: Vec<Option<(&str, &str)>> = Vec::new();
    for p in src[open + 1..close].split(',') {
        let words: Vec<&str> = p.split_whitespace().collect();
        params.push(match words.as_slice() {
            [".param", ty @ (".u32" | ".s32" | ".b32" | ".f32" | ".u64" | ".s64" | ".b64"), name]
                if !name.contains('[') => Some((&ty[1..], *name)),
            _ => None,
        });
    }
    let body_start = close + src[close..].find('{')?;
    let body_end = body_start + matching_brace(&src[body_start..])?;
    let mut out = String::with_capacity(src.len());
    out.push_str(&src[..body_start]);
    let mut rewritten = 0;
    for line in src[body_start..body_end].split_inclusive('\n') {
        match rewrite(line, &params, args, block) {
            Some(new) => {
                out.push_str(&new);
                rewritten += 1;
            }
            None => out.push_str(line),
        }
    }
    out.push_str(&src[body_end..]);
    if rewritten == 0 {
        return None;
    }
    let out = without_other_entries(&out, kernel);
    // This exact Geekbench 7 tile shape has been checked against captured GPU data.
    // Keep the optimization behind a complete PTX fingerprint until a general
    // shared-memory layout pass can prove the same transformation for other code.
    let out = crate::gemm_tile::transpose_b(&out, block).unwrap_or(out);
    Some((out, rewritten))
}

/// `src` without the kernels other than `kernel` (they would only add compile time); functions and globals stay,
/// and module-scope `.global` and `.const` variables become `.visible`, so that LLVM keeps all of them: the
/// specialized module then lays out its global data as the module does and uses the module's buffer.
pub(crate) fn without_other_entries(src: &str, kernel: &str) -> String {
    let src = &visible_globals(src);
    let mut out = String::with_capacity(src.len());
    let mut at = 0;
    while let Some(i) = src[at..].find(".entry") {
        let entry = at + i;
        let name_start = entry + ".entry".len();
        let name: String = src[name_start..].trim_start().chars().take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$').collect();
        // the whole declaration from the start of its line (`.visible .entry`, `.weak .entry`, ...) to its body's end
        let line = src[..entry].rfind('\n').map_or(0, |n| n + 1);
        let body = src[entry..].find('{').map(|b| entry + b);
        let end = body.and_then(|b| matching_brace(&src[b..]).map(|e| b + e));
        match end {
            Some(end) if name != kernel && line >= at => {
                out.push_str(&src[at..line]);
                at = end;
            }
            _ => {
                out.push_str(&src[at..name_start]);
                at = name_start;
            }
        }
    }
    out.push_str(&src[at..]);
    out
}

/// Kernel names in a PTX image, in source order. This is used to defer AGX
/// code generation until a CUDA module asks for a particular kernel.
pub(crate) fn entry_names(src: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut at = 0;
    while let Some(i) = src[at..].find(".entry") {
        let start = at + i + ".entry".len();
        let rest = src[start..].trim_start();
        let name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$').collect();
        if !name.is_empty() && rest[name.len()..].trim_start().starts_with('(') {
            names.push(name);
        }
        at = start;
    }
    names
}

/// `src` with `.visible` before the module-scope `.global` / `.const` declarations that have no linkage directive.
fn visible_globals(src: &str) -> String {
    let mut out = String::with_capacity(src.len() + 64);
    let mut depth = 0i32;
    for line in src.split_inclusive('\n') {
        let t = line.trim_start();
        if depth == 0 && (t.starts_with(".global ") || t.starts_with(".const ")) {
            out.push_str(&line[..line.len() - t.len()]);
            out.push_str(".visible ");
            out.push_str(t);
        } else {
            out.push_str(line);
        }
        depth += line.matches('{').count() as i32 - line.matches('}').count() as i32;
    }
    out
}

/// The byte offset of `.entry <kernel>` followed by `(`.
fn find_entry(src: &str, kernel: &str) -> Option<usize> {
    let mut at = 0;
    while let Some(i) = src[at..].find(".entry") {
        let start = at + i;
        let rest = src[start + ".entry".len()..].trim_start();
        if let Some(after) = rest.strip_prefix(kernel) {
            if after.trim_start().starts_with('(') {
                return Some(start);
            }
        }
        at = start + ".entry".len();
    }
    None
}

/// The offset just past the `}` that closes the `{` at offset 0.
fn matching_brace(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// The replacement of one line of the kernel body, if it reads a specialized value.
fn rewrite(line: &str, params: &[Option<(&str, &str)>], args: &[&[u8]], block: [u32; 3]) -> Option<String> {
    let indent = &line[..line.len() - line.trim_start().len()];
    let text = line.trim();
    let newline = if line.ends_with('\n') { "\n" } else { "" };
    // mov.u32 %r, %ntid.x;
    if let Some(rest) = text.strip_prefix("mov.u32") {
        let (dst, src) = operands(rest)?;
        let dim = match src {
            "%ntid.x" => 0,
            "%ntid.y" => 1,
            "%ntid.z" => 2,
            _ => return None,
        };
        return Some(format!("{indent}mov.u32 \t{dst}, {};{newline}", block[dim]));
    }
    // ld.param.<ty> %r, [name] / [name+0]
    let rest = text.strip_prefix("ld.param.")?;
    let (ty, rest) = rest.split_at(rest.find(|c: char| c.is_whitespace())?);
    let (dst, addr) = operands(rest)?;
    let name = addr.strip_prefix('[')?.strip_suffix(']')?;
    let name = name.strip_suffix("+0").unwrap_or(name).trim();
    let index = params.iter().position(|s| s.is_some_and(|(_, n)| n == name))?;
    let (pty, _) = params[index]?;
    if ty != pty {
        return None;
    }
    if matches!(ty, "u64" | "s64" | "b64") {
        let value = args.get(index)?;
        if value.len() != 8 {
            return None;
        }
        let (_, earlier) = params[..index].iter().enumerate().find_map(|(j, p)| {
            let (earlier_ty, earlier) = (*p)?;
            (earlier_ty == ty && args.get(j).is_some_and(|a| *a == *value)).then_some((j, earlier))
        })?;
        return Some(format!("{indent}ld.param.{ty} \t{dst}, [{earlier}];{newline}"));
    }
    let bytes: [u8; 4] = args.get(index)?.get(..4)?.try_into().ok()?;
    if args[index].len() != 4 {
        return None;
    }
    let v = u32::from_le_bytes(bytes);
    let value = match ty {
        "f32" => format!("0f{v:08X}"),
        "s32" => format!("{}", v as i32),
        _ => format!("{v}"),
    };
    Some(format!("{indent}mov.{ty} \t{dst}, {value};{newline}"))
}

/// Whether kernel `kernel` of `src`, or a function kept beside it, names one of `vars` outside a module-scope
/// declaration. A kernel that does not cannot gain from constant-memory specialization, and the folded copy would
/// take the place of its launch-time specialization of scalars and block size (Geekbench's image kernels, whose
/// module has a host-written table only its `_wide` kernels read, ran 1.7x slower that way).
pub fn entry_names_any(src: &str, kernel: &str, vars: &[&str]) -> bool {
    let text = without_other_entries(src, kernel);
    let ident = |c: char| c.is_alphanumeric() || c == '_' || c == '$';
    text.lines().filter(|l| !l.trim_start().starts_with('.')).any(|line| {
        vars.iter().any(|v| {
            line.match_indices(v).any(|(i, _)| {
                !line[..i].chars().next_back().is_some_and(ident) && !line[i + v.len()..].chars().next().is_some_and(ident)
            })
        })
    })
}

/// The module-scope `.const` variables without an initializer, which only the host writes (through
/// `cuModuleGetGlobal`'s address): name and size in bytes, in source order. Constant-memory specialization
/// (`fold_const_loads`) folds their current contents into a copy of the module.
pub fn host_written_consts(src: &str) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    for line in src.lines() {
        let t = line.trim();
        if depth == 0 && !t.contains('=') {
            let t = t.strip_prefix(".visible ").unwrap_or(t);
            if let Some(decl) = t.strip_prefix(".const ") {
                if let Some(v) = const_decl(decl) {
                    out.push(v);
                }
            }
        }
        depth += line.matches('{').count() as i32 - line.matches('}').count() as i32;
    }
    out
}

/// `(name, bytes)` of a `.const` declaration's text after `.const`: `.align 16 .b8 name[4944];`, `.u64 name;`.
fn const_decl(decl: &str) -> Option<(String, usize)> {
    let words: Vec<&str> = decl.trim().strip_suffix(';')?.split_whitespace().collect();
    let words: &[&str] = if words.first() == Some(&".align") { words.get(2..)? } else { &words };
    let [ty, name] = words else { return None };
    let elem = scalar_bytes(ty.strip_prefix('.')?)?;
    let (name, count) = match name.split_once('[') {
        Some((n, rest)) => (n, rest.strip_suffix(']')?.parse::<usize>().ok()?),
        None => (*name, 1),
    };
    (!name.is_empty()).then(|| (name.to_string(), elem * count))
}

fn scalar_bytes(ty: &str) -> Option<usize> {
    Some(match ty {
        "b8" | "u8" | "s8" => 1,
        "b16" | "u16" | "s16" | "f16" => 2,
        "b32" | "u32" | "s32" | "f32" => 4,
        "b64" | "u64" | "s64" | "f64" => 8,
        _ => return None,
    })
}

/// Constant-memory specialization: `src` with every `ld.const` of a variable in `values` (name, its current bytes)
/// at a constant offset replaced by a `mov` of the value it reads, and the module-scope variables made `.visible`
/// so that the copy lays out its global data as the module does (and keeps the variables, which loads at computed
/// addresses still read). Valid while the variables hold these bytes: the driver compares them at every launch.
/// Loads wider than `max_bytes` stay loads. Returns the text and the number of loads replaced.
pub fn fold_const_loads(src: &str, values: &[(&str, &[u8])], max_bytes: usize) -> (String, usize) {
    let mut out = String::with_capacity(src.len());
    let mut regs: HashMap<String, String> = HashMap::new();
    let mut folded = 0;
    for line in src.split_inclusive('\n') {
        let t = line.trim_start();
        if t.starts_with(".reg ") {
            // .reg .b32 %r<12>;  .reg .f32 %f<3>, %g;
            let mut words = t[5..].trim_end().trim_end_matches(';').splitn(2, char::is_whitespace);
            if let (Some(ty), Some(names)) = (words.next(), words.next()) {
                for n in names.split(',') {
                    let n = n.trim();
                    let prefix = n.split('<').next().unwrap_or(n);
                    regs.insert(prefix.to_string(), ty.trim_start_matches('.').to_string());
                }
            }
        } else if t.contains(".entry") || t.contains(".func") {
            regs.clear();
        }
        match t.contains("ld.const").then(|| fold_const_load(line, values, &regs, max_bytes)).flatten() {
            Some(new) => {
                out.push_str(&new);
                folded += 1;
            }
            None => out.push_str(line),
        }
    }
    (visible_globals(&out), folded)
}

/// The register type (`b32`, `f64`, ...) of a register by its declaration (`%r12` from `.reg .b32 %r<20>`).
fn reg_type<'a>(reg: &str, regs: &'a HashMap<String, String>) -> Option<&'a str> {
    regs.get(reg).or_else(|| regs.get(reg.trim_end_matches(|c: char| c.is_ascii_digit()))).map(|s| s.as_str())
}

fn fold_const_load(line: &str, values: &[(&str, &[u8])], regs: &HashMap<String, String>, max_bytes: usize) -> Option<String> {
    let indent = &line[..line.len() - line.trim_start().len()];
    let newline = if line.ends_with('\n') { "\n" } else { "" };
    let mut text = line.trim();
    let mut guard = "";
    if text.starts_with('@') {
        let sp = text.find(char::is_whitespace)?;
        guard = &text[..sp];
        text = text[sp..].trim_start();
    }
    let rest = text.strip_prefix("ld.const.")?;
    let (ty, rest) = rest.split_at(rest.find(char::is_whitespace)?);
    let (lanes, ty) = match ty.split_once('.') {
        Some(("v2", t)) => (2, t),
        Some(("v4", t)) => (4, t),
        None => (1, ty),
        _ => return None,
    };
    let size = scalar_bytes(ty).filter(|&n| n <= max_bytes)?;
    let rest = rest.trim().strip_suffix(';')?;
    let (dst, addr) = rest.rsplit_once(',')?;
    let addr = addr.trim().strip_prefix('[')?.strip_suffix(']')?;
    let (name, offset) = match addr.split_once('+') {
        Some((n, o)) => (n.trim(), o.trim().parse::<usize>().ok()?),
        None => (addr.trim(), 0),
    };
    let bytes = values.iter().find(|(n, _)| *n == name)?.1;
    let dsts: Vec<&str> = match lanes {
        1 => vec![dst.trim()],
        _ => dst.trim().strip_prefix('{')?.strip_suffix('}')?.split(',').map(str::trim).collect(),
    };
    if dsts.len() != lanes || offset % size != 0 || offset + size * lanes > bytes.len() {
        return None;
    }
    let mut new = String::new();
    for (i, d) in dsts.iter().enumerate() {
        if *d == "_" {
            continue;
        }
        let at = offset + i * size;
        let mut raw = [0u8; 8];
        raw[..size].copy_from_slice(&bytes[at..at + size]);
        let mut v = u64::from_le_bytes(raw);
        if ty.starts_with('s') && size < 8 {
            let shift = 64 - 8 * size as u32;
            v = (((v << shift) as i64) >> shift) as u64;
        }
        let rty = reg_type(d, regs)?;
        let value = match rty {
            "f32" if matches!(ty, "f32" | "b32") => format!("mov.f32 \t{d}, 0f{:08X};", v as u32),
            "f64" if matches!(ty, "f64" | "b64") => format!("mov.f64 \t{d}, 0d{v:016X};"),
            "b16" | "u16" | "s16" | "b32" | "u32" | "s32" | "b64" | "u64" | "s64" => {
                let width = scalar_bytes(rty)?;
                let float = matches!(ty, "f32" | "f64");
                if width < size || ty == "f16" || (float && width != size) {
                    return None;
                }
                let v = if width < 8 { v & ((1u64 << (8 * width)) - 1) } else { v };
                format!("mov.b{} \t{d}, 0x{v:0w$X};", 8 * width, w = 2 * width)
            }
            _ => return None,
        };
        if !new.is_empty() {
            new.push_str(newline);
            new.push_str(indent);
        }
        if !guard.is_empty() {
            new.push_str(guard);
            new.push(' ');
        }
        new.push_str(&value);
    }
    Some(format!("{indent}{new}{newline}"))
}

/// `dst, src;` of an instruction's operand text.
fn operands(rest: &str) -> Option<(&str, &str)> {
    let rest = rest.trim().strip_suffix(';')?;
    let (dst, src) = rest.split_once(',')?;
    Some((dst.trim(), src.trim()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PTX: &str = ".version 7.0
.target sm_75
.address_size 64
.const .align 4 .b8 table[8] = {1, 0, 0, 0, 2};
.visible .global .align 4 .u32 counter;
.visible .entry other(.param .u32 other_param_0) { ld.param.u32 %r1, [other_param_0]; ret; }
.visible .entry k(
	.param .u64 k_param_0,
	.param .u32 k_param_1,
	.param .f32 k_param_2,
	.param .s32 k_param_3
)
{
	.reg .b32 %r<5>;
	ld.param.u64 	%rd1, [k_param_0];
	ld.param.u32 	%r1, [k_param_1];
	ld.param.f32 	%f1, [k_param_2+0];
	ld.param.s32 	%r2, [k_param_3];
	mov.u32 	%r3, %ntid.x;
	mov.u32 	%r4, %ntid.y;
	{
	mov.u32 	%r4, %tid.x;
	}
	ret;
}
";

    #[test]
    fn scalars_and_block_become_constants() {
        let args: [&[u8]; 4] = [&7u64.to_le_bytes(), &42u32.to_le_bytes(), &1.5f32.to_le_bytes(), &(-3i32).to_le_bytes()];
        let (out, n) = specialize_ptx(PTX, "k", &args, [64, 2, 1]).unwrap();
        assert_eq!(n, 5);
        assert!(out.contains("ld.param.u64 \t%rd1, [k_param_0];"), "pointers stay");
        assert!(out.contains("mov.u32 \t%r1, 42;"));
        assert!(out.contains("mov.f32 \t%f1, 0f3FC00000;"));
        assert!(out.contains("mov.s32 \t%r2, -3;"));
        assert!(out.contains("mov.u32 \t%r3, 64;") && out.contains("mov.u32 \t%r4, 2;"));
        assert!(out.contains("mov.u32 \t%r4, %tid.x;"));
        // the other kernel is left out
        assert!(!out.contains("other") && out.contains(".visible .entry k("));
        // the module's variables are all kept
        assert!(out.contains("\n.visible .const .align 4 .b8 table[8] = {1, 0, 0, 0, 2};\n"));
        assert!(out.contains("\n.visible .global .align 4 .u32 counter;\n"));
        assert!(specialize_ptx(PTX, "missing", &args, [1, 1, 1]).is_none());
    }

    #[test]
    fn generic_kernel_images_keep_globals_and_one_entry() {
        assert_eq!(entry_names(PTX), ["other", "k"]);
        let only_k = without_other_entries(PTX, "k");
        assert_eq!(entry_names(&only_k), ["k"]);
        assert!(only_k.contains(".visible .const .align 4 .b8 table[8]"));
        assert!(only_k.contains(".visible .global .align 4 .u32 counter"));
    }

    #[test]
    fn equal_64_bit_arguments_share_a_parameter_only_for_that_equality_class() {
        let ptx = ".version 7.0\n.target sm_75\n.address_size 64\n.visible .entry copy(\n\
            \t.param .u64 src,\n\
            \t.param .u64 dst,\n\
            \t.param .u32 count\n\
            ) {\n\
            \t.reg .b64 %rd<3>;\n\
            \tld.param.u64 %rd1, [src];\n\
            \tld.param.u64 %rd2, [dst+0];\n\
            \tld.param.u32 %r1, [count];\n\
            \tret;\n}\n";
        let a = 0x1100_4000u64.to_le_bytes();
        let b = 0x1100_8000u64.to_le_bytes();
        let n = 128u32.to_le_bytes();
        let equal: [&[u8]; 3] = [&a, &a, &n];
        let distinct: [&[u8]; 3] = [&a, &b, &n];
        assert_ne!(equal_eight_byte_args(&equal), equal_eight_byte_args(&distinct));
        let (same, _) = specialize_ptx(ptx, "copy", &equal, [64, 1, 1]).unwrap();
        assert!(same.contains("ld.param.u64 \t%rd2, [src];"));
        let (different, _) = specialize_ptx(ptx, "copy", &distinct, [64, 1, 1]).unwrap();
        assert!(different.contains("ld.param.u64 %rd2, [dst+0];"));
    }

    #[test]
    fn host_written_constants_only_for_kernels_naming_them() {
        let ptx = ".version 7.0\n.target sm_75\n.address_size 64\n\
            .visible .const .align 4 .b8 table[64];\n\
            .visible .entry plain(.param .u32 n) {\n\t.reg .b32 %r<2>;\n\tld.param.u32 %r1, [n];\n\tret;\n}\n\
            .visible .entry wide() {\n\t.reg .b64 %rd<2>;\n\tmov.u64 %rd1, table;\n\tret;\n}\n\
            .visible .entry direct() {\n\t.reg .f32 %f<2>;\n\tld.const.f32 %f1, [table+4];\n\tret;\n}\n\
            .visible .entry other() {\n\t.reg .f32 %f<2>;\n\tld.const.f32 %f1, [table_b];\n\tret;\n}\n";
        assert!(!entry_names_any(ptx, "plain", &["table"]));
        assert!(entry_names_any(ptx, "wide", &["table"]));
        assert!(entry_names_any(ptx, "direct", &["table"]));
        assert!(!entry_names_any(ptx, "other", &["table"]));
    }

    #[test]
    fn host_written_constants_fold_into_moves() {
        let ptx = ".version 7.0\n.target sm_75\n.address_size 64\n\
            .const .align 16 .b8 params[24];\n\
            .const .align 4 .b8 table[4] = {1, 2, 3, 4};\n\
            .extern .const .align 4 .b8 other[8];\n\
            .visible .entry k() {\n\
            \t.reg .pred %p<2>;\n\
            \t.reg .b16 %rs<2>;\n\
            \t.reg .b32 %r<4>;\n\
            \t.reg .f32 %f<2>;\n\
            \t.reg .b64 %rd<4>;\n\
            \tld.const.u64 %rd1, [params];\n\
            \tld.const.s32 %rd2, [params+8];\n\
            \tld.const.f32 %f1, [params+12];\n\
            \tld.const.u8 %rs1, [params+16];\n\
            \t@%p1 ld.const.v2.u32 {%r1, %r2}, [params+16];\n\
            \tld.const.u32 %r3, [params+%rd1];\n\
            \tld.const.u32 %r3, [table];\n\
            \tret;\n}\n";
        assert_eq!(host_written_consts(ptx), [("params".to_string(), 24)]);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
        bytes.extend_from_slice(&(-2i32).to_le_bytes());
        bytes.extend_from_slice(&1.5f32.to_le_bytes());
        bytes.extend_from_slice(&[0xab, 0, 0, 0, 7, 0, 0, 0]);
        let (out, n) = fold_const_loads(ptx, &[("params", &bytes)], 8);
        assert_eq!(n, 5, "{out}");
        let (narrow, n4) = fold_const_loads(ptx, &[("params", &bytes)], 4);
        assert_eq!(n4, 4, "{narrow}");
        assert!(narrow.contains("ld.const.u64 %rd1, [params];"));
        assert!(out.contains("mov.b64 \t%rd1, 0x1122334455667788;"));
        assert!(out.contains("mov.b64 \t%rd2, 0xFFFFFFFFFFFFFFFE;"), "sign-extended into the wider register");
        assert!(out.contains("mov.f32 \t%f1, 0f3FC00000;"));
        assert!(out.contains("mov.b16 \t%rs1, 0x00AB;"));
        assert!(out.contains("@%p1 mov.b32 \t%r1, 0x000000AB;\n\t@%p1 mov.b32 \t%r2, 0x00000007;"));
        assert!(out.contains("ld.const.u32 %r3, [params+%rd1];"), "computed addresses stay loads");
        assert!(out.contains("ld.const.u32 %r3, [table];"), "initialized constants are not host written");
        assert!(out.contains("\n.visible .const .align 16 .b8 params[24];"));
    }
}
