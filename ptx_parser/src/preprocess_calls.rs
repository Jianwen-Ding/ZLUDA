//! Calls, function addresses and branch tables before lexing (preprocess.rs runs this pass):
//!
//! * a call's arguments and results are adapted to the callee's formals (PTX ISA 7.1 "For a caller"): a register or
//!   constant passed to a .param formal goes through a fresh .param variable, a .param variable passed to a .reg
//!   formal is loaded, results likewise; results the call leaves out go to fresh variables; `()` is no argument;
//! * function addresses (`mov.u64 %fp, f;`, `.global .u64 tab[] = { f, g };`) are function numbers, and an indirect
//!   call (`call (r), %fp, (a), Proto;` with a .callprototype or .calltargets) compares the number with each
//!   candidate (the .calltargets, or the module's functions of the prototype's shape) and calls it directly: AGX
//!   has no indirect calls;
//! * `brx.idx i, Tab;` is a comparison chain over the .branchtargets (label ranges `N<3>` expanded);
//! * a function that calls itself is unrolled into copies of bounded depth (AGX has no recursion; the depth bound
//!   is RECURSION_DEPTH);
//! * weak definitions: the first weak definition of a name is kept unless a strong one exists, extern declarations
//!   of defined names are dropped (linking, 11.6).
use std::collections::{HashMap, HashSet};

const FN_ID_BASE: u64 = 0xF_0000;
const RECURSION_DEPTH: usize = 8;

#[derive(Clone, Debug)]
struct Formal {
    param: bool,
    ty: String,
    array: Option<String>,
    align: Option<String>,
}

#[derive(Clone, Debug)]
struct Sig {
    rets: Vec<Formal>,
    params: Vec<Formal>,
}

fn is_ident_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'$' || c == b'%'
}

/// splits at top-level commas
fn split_top(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }
        if c == ',' && depth == 0 {
            out.push(cur.trim().to_string());
            cur.clear();
        } else {
            cur.push(c);
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// `.param.u64 x` -> `.param .u64 x`
fn spaced(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 8);
    let mut prev = ' ';
    for c in s.chars() {
        if c == '.' && !prev.is_whitespace() && !prev.is_ascii_digit() && prev != '(' && prev != ',' {
            o.push(' ');
        }
        o.push(c);
        prev = c;
    }
    o
}

fn parse_formal(s: &str) -> Option<(Formal, String)> {
    let s = spaced(s);
    let words: Vec<&str> = s.split_whitespace().collect();
    let param = words.contains(&".param");
    if !param && !words.contains(&".reg") {
        return None;
    }
    let mut ty = String::new();
    let mut align = None;
    let mut i = 0;
    let mut name = String::new();
    while i < words.len() {
        let w = words[i];
        if w == ".align" {
            align = words.get(i + 1).map(|x| x.to_string());
            i += 2;
            continue;
        }
        if w.starts_with('.') {
            if w != ".param" && w != ".reg" && w != ".ptr" && !w.starts_with(".global") && !w.starts_with(".shared")
                && !w.starts_with(".const") && !w.starts_with(".local")
            {
                ty = w.to_string();
            }
        } else {
            name = w.to_string();
        }
        i += 1;
    }
    let mut array = None;
    if let Some(p) = name.find('[') {
        array = Some(name[p..].to_string());
        name.truncate(p);
    }
    Some((Formal { param, ty, array, align }, name))
}

/// the parenthesized list starting at s[0] == '(': (contents, rest)
fn paren_list(s: &str) -> Option<(&str, &str)> {
    let s = s.trim_start();
    if !s.starts_with('(') {
        return None;
    }
    let mut depth = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some((&s[1..i], &s[i + 1..]));
                }
            }
            _ => {}
        }
    }
    None
}

/// `(rets) name (params)` / `name (params)` after `.func`: (name, sig)
fn parse_header(h: &str) -> Option<(String, Sig, Vec<String>)> {
    let mut rest = h.trim_start();
    let mut rets = Vec::new();
    if let Some((r, after)) = paren_list(rest) {
        for f in split_top(r) {
            rets.push(parse_formal(&f)?.0);
        }
        rest = after.trim_start();
    }
    let name: String = rest.bytes().take_while(|c| is_ident_byte(*c)).map(|c| c as char).collect();
    if name.is_empty() {
        return None;
    }
    rest = &rest[name.len()..];
    let mut params = Vec::new();
    let mut pnames = Vec::new();
    if let Some((p, _)) = paren_list(rest) {
        for f in split_top(p) {
            let (fm, n) = parse_formal(&f)?;
            params.push(fm);
            pnames.push(n);
        }
    }
    Some((name, Sig { rets, params }, pnames))
}

struct Func {
    name: String,
    sig: Sig,
    formals: Vec<String>,
    entry: bool,
    /// byte range of the header start (the linkage words) .. end of body (or of the declaration's ';')
    start: usize,
    end: usize,
    /// body range (inside the braces), None for a declaration
    body: Option<(usize, usize)>,
    weak: bool,
    ext: bool,
}

fn find_word(t: &str, w: &str, from: usize) -> Option<usize> {
    let b = t.as_bytes();
    let mut s = from;
    while let Some(p) = t[s..].find(w) {
        let p = s + p;
        let e = p + w.len();
        if (p == 0 || !is_ident_byte(b[p - 1]) && b[p - 1] != b'.') && (e >= b.len() || !is_ident_byte(b[e])) {
            return Some(p);
        }
        s = e;
    }
    None
}

/// the functions of the module (text without comments: callers blank them first)
fn functions(t: &str) -> Vec<Func> {
    let b = t.as_bytes();
    let mut out = Vec::new();
    let mut from = 0;
    loop {
        let pf = find_word(t, ".func", from);
        let pe = find_word(t, ".entry", from);
        let (p, entry) = match (pf, pe) {
            (Some(a), Some(c)) => {
                if a < c {
                    (a, false)
                } else {
                    (c, true)
                }
            }
            (Some(a), None) => (a, false),
            (None, Some(c)) => (c, true),
            (None, None) => break,
        };
        // linkage words before it on the same statement
        let mut start = p;
        while start > 0 && !matches!(b[start - 1], b';' | b'}' | b'\n') {
            start -= 1;
        }
        let pre = &t[start..p];
        let weak = pre.split_whitespace().any(|w| w == ".weak");
        let ext = pre.split_whitespace().any(|w| w == ".extern");
        let kw = if entry { ".entry".len() } else { ".func".len() };
        let hstart = p + kw;
        // header ends at '{' or ';' at depth 0
        let mut depth = 0;
        let mut i = hstart;
        while i < b.len() {
            match b[i] {
                b'(' => depth += 1,
                b')' => depth -= 1,
                b'{' | b';' if depth == 0 => break,
                _ => {}
            }
            i += 1;
        }
        let header = &t[hstart..i];
        let parsed = parse_header(header);
        if i >= b.len() {
            break;
        }
        let (end, body) = if b[i] == b'{' {
            let mut d = 0;
            let mut j = i;
            while j < b.len() {
                match b[j] {
                    b'{' => d += 1,
                    b'}' => {
                        d -= 1;
                        if d == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            (j + 1, Some((i + 1, j)))
        } else {
            (i + 1, None)
        };
        if let Some((name, sig, formals)) = parsed {
            out.push(Func { name, sig, formals, entry, start, end, body, weak, ext });
        }
        from = end.min(b.len());
    }
    out
}

fn blank_comments(t: &str) -> String {
    let mut b = t.as_bytes().to_vec();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'"' {
            i += 1;
            while i < b.len() && b[i] != b'"' {
                i += 1;
            }
            i += 1;
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                b[i] = b' ';
                i += 1;
            }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            while i < b.len() && !(b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/') {
                if b[i] != b'\n' {
                    b[i] = b' ';
                }
                i += 1;
            }
            if i + 1 < b.len() {
                b[i] = b' ';
                b[i + 1] = b' ';
                i += 2;
            }
        } else {
            i += 1;
        }
    }
    String::from_utf8(b).unwrap_or_else(|_| t.to_string())
}

pub(crate) fn needs_work(t: &str) -> bool {
    t.contains("call") || t.contains(".weak") || t.contains("brx") || t.contains(".extern")
}

pub(crate) fn rewrite(text: &str) -> String {
    let t = blank_comments(text);
    let t = link_weak(&t);
    let t = unroll_recursion(&t);
    let t = size_unsized_formals(&t);
    let funcs = functions(&t);
    if funcs.is_empty() {
        return t;
    }
    let mut ids: HashMap<String, u64> = HashMap::new();
    let mut sigs: HashMap<String, Sig> = HashMap::new();
    let mut callable: Vec<String> = Vec::new();
    for f in &funcs {
        if !ids.contains_key(&f.name) {
            ids.insert(f.name.clone(), FN_ID_BASE + 16 * ids.len() as u64);
        }
        if !f.entry {
            sigs.insert(f.name.clone(), f.sig.clone());
            if f.body.is_some() && !callable.contains(&f.name) {
                callable.push(f.name.clone());
            }
        }
    }
    let mut out = String::with_capacity(t.len());
    let mut last = 0;
    let mut ctr = 0usize;
    for f in &funcs {
        let Some((bs, be)) = f.body else { continue };
        out.push_str(&replace_fn_names(&t[last..bs], &ids, false));
        out.push_str(&rewrite_body(&t[bs..be], &ids, &sigs, &callable, &f.formals, &mut ctr));
        last = be;
    }
    out.push_str(&replace_fn_names(&t[last..], &ids, false));
    out
}

/// an unsized last .param array formal of a function (`.param .b8 va[]`, PTX ISA 11.2.2) gets the size of the
/// argument a call may omit, `[1]` (decl_of)
fn size_unsized_formals(t: &str) -> String {
    let mut out = t.to_string();
    for f in functions(t).iter().rev() {
        if f.entry || !f.sig.params.iter().any(|p| p.array.as_deref() == Some("[]")) {
            continue;
        }
        let hend = f.body.map(|(bs, _)| bs).unwrap_or(f.end);
        let header = out[f.start..hend].replace("[]", "[1]");
        out.replace_range(f.start..hend, &header);
    }
    out
}

/// function names used as values become their numbers (outside headers and calls)
fn replace_fn_names(s: &str, ids: &HashMap<String, u64>, _in_body: bool) -> String {
    // only in initializers: `= { f1, f2 }` / `= f1`
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(p) = rest.find('=') {
        out.push_str(&rest[..p + 1]);
        rest = &rest[p + 1..];
        let e = rest.find(';').unwrap_or(rest.len());
        out.push_str(&subst_idents(&rest[..e], ids));
        rest = &rest[e..];
    }
    out.push_str(rest);
    out
}

fn subst_idents(s: &str, ids: &HashMap<String, u64>) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        if is_ident_byte(b[i]) && (i == 0 || (!is_ident_byte(b[i - 1]) && b[i - 1] != b'.')) {
            let st = i;
            while i < b.len() && is_ident_byte(b[i]) {
                i += 1;
            }
            let w = &s[st..i];
            match ids.get(w) {
                Some(id) if !(i < b.len() && b[i] == b'[') => out.push_str(&id.to_string()),
                _ => out.push_str(w),
            }
        } else {
            let c = s[i..].chars().next().unwrap();
            out.push(c);
            i += c.len_utf8();
        }
    }
    out
}

/// statements of a body: (start, end) byte ranges excluding the `;`; block braces `{` `}` at statement starts are
/// boundaries, braces inside a statement (vector operands, initializers) are part of it
fn statements(b: &str) -> Vec<(usize, usize)> {
    let bb = b.as_bytes();
    let mut out = Vec::new();
    let mut s = 0;
    let mut i = 0;
    let mut inner = 0i32;
    while i < bb.len() {
        let at_start = b[s..i].trim().is_empty() || b[s..i].trim_end().ends_with(':') && !b[s..i].trim_end().ends_with("::");
        match bb[i] {
            b'"' => {
                i += 1;
                while i < bb.len() && bb[i] != b'"' {
                    i += 1;
                }
            }
            b';' if inner == 0 => {
                out.push((s, i));
                s = i + 1;
            }
            b'{' if inner == 0 && at_start => {
                if !b[s..i].trim().is_empty() {
                    // a label before a block
                    out.push((s, i));
                }
                s = i + 1;
            }
            b'{' => inner += 1,
            b'}' if inner > 0 => inner -= 1,
            b'}' => {
                if !b[s..i].trim().is_empty() {
                    out.push((s, i));
                }
                s = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    out
}

/// label prefix and predicate of a statement: (prefix text, guard e.g. "@%p" / "@!%p" / "", instruction text)
fn split_stmt(s: &str) -> (String, String, String) {
    let mut rest = s.trim_start();
    let mut prefix = String::new();
    loop {
        // `L: ` (not `a::b`)
        let w: String = rest.bytes().take_while(|c| is_ident_byte(*c)).map(|c| c as char).collect();
        if !w.is_empty() && rest[w.len()..].trim_start().starts_with(':') && !rest[w.len()..].trim_start().starts_with("::") {
            let after = rest[w.len()..].trim_start();
            prefix.push_str(&w);
            prefix.push_str(": ");
            rest = after[1..].trim_start();
            continue;
        }
        break;
    }
    let mut guard = String::new();
    if rest.starts_with('@') {
        let e = rest.find(char::is_whitespace).unwrap_or(rest.len());
        guard = rest[..e].to_string();
        rest = rest[e..].trim_start();
    }
    (prefix, guard, rest.to_string())
}

fn width_of(ty: &str) -> u32 {
    let d: String = ty.chars().filter(|c| c.is_ascii_digit()).collect();
    d.parse().unwrap_or(0)
}

fn decl_of(f: &Formal, name: &str) -> String {
    let al = f.align.as_ref().map(|a| format!(".align {a} ")).unwrap_or_default();
    let arr = match &f.array {
        Some(a) if a == "[]" => "[1]".to_string(),
        Some(a) => a.clone(),
        None => String::new(),
    };
    if f.param {
        format!(".param {al}{} {name}{arr};", f.ty)
    } else {
        format!(".reg {} {name};", f.ty)
    }
}

fn sig_matches(a: &Sig, b: &Sig) -> bool {
    let eq = |x: &[Formal], y: &[Formal]| {
        x.len() == y.len() && x.iter().zip(y).all(|(p, q)| width_of(&p.ty) == width_of(&q.ty) && p.array.is_some() == q.array.is_some())
    };
    eq(&a.rets, &b.rets) && eq(&a.params, &b.params)
}

fn rewrite_body(
    body: &str,
    ids: &HashMap<String, u64>,
    sigs: &HashMap<String, Sig>,
    callable: &[String],
    formals: &[String],
    ctr: &mut usize,
) -> String {
    let stmts = statements(body);
    // names declared in the function shadow function names
    let mut ids = ids.clone();
    for n in formals {
        ids.remove(n);
    }
    for &(s, e) in &stmts {
        let (_, _, inst) = split_stmt(&body[s..e]);
        let inst = spaced(&inst);
        let mut ws = inst.split_whitespace();
        if matches!(ws.next(), Some(".reg" | ".param" | ".local" | ".shared" | ".global" | ".const")) {
            let decl = &inst[inst.find(char::is_whitespace).unwrap_or(0)..];
            for part in split_top(decl) {
                if let Some(n) = part.split_whitespace().last() {
                    let n: String = n.bytes().take_while(|c| is_ident_byte(*c)).map(|c| c as char).collect();
                    ids.remove(&n);
                }
            }
        }
    }
    let ids = &ids;
    // declarations of this body
    let mut param_vars: HashSet<String> = HashSet::new();
    let mut protos: HashMap<String, Sig> = HashMap::new();
    let mut targets: HashMap<String, Vec<String>> = HashMap::new();
    let mut btargets: HashMap<String, Vec<String>> = HashMap::new();
    for &(s, e) in &stmts {
        let (prefix, _, inst) = split_stmt(&body[s..e]);
        let label = prefix.trim().trim_end_matches(':').trim().to_string();
        let inst = if inst.starts_with(".param") { spaced(&inst) } else { inst };
        let ws: Vec<&str> = inst.split_whitespace().collect();
        if ws.first() == Some(&".param") {
            for part in split_top(&inst[".param".len()..]) {
                if let Some(n) = part.split_whitespace().last() {
                    let n = n.split('[').next().unwrap_or(n);
                    param_vars.insert(n.to_string());
                }
            }
        } else if ws.first() == Some(&".callprototype") {
            if let Some((_, sig, _)) = parse_header(&inst[".callprototype".len()..]) {
                protos.insert(label, sig);
            }
        } else if ws.first() == Some(&".calltargets") {
            targets.insert(label, split_top(&inst[".calltargets".len()..]));
        } else if ws.first() == Some(&".branchtargets") {
            let mut l = Vec::new();
            for x in split_top(&inst[".branchtargets".len()..]) {
                if let (Some(lt), true) = (x.find('<'), x.ends_with('>')) {
                    if let Ok(n) = x[lt + 1..x.len() - 1].trim().parse::<usize>() {
                        for k in 0..n {
                            l.push(format!("{}{k}", &x[..lt]));
                        }
                        continue;
                    }
                }
                l.push(x);
            }
            btargets.insert(label, l);
        }
    }
    let mut out = String::with_capacity(body.len());
    let mut last = 0;
    for &(s, e) in &stmts {
        let (prefix, guard, inst) = split_stmt(&body[s..e]);
        let ws0 = inst.split_whitespace().next().unwrap_or("");
        let base = ws0.split('.').next().unwrap_or("");
        let repl: Option<String> = match base {
            _ if ws0 == ".callprototype" || ws0 == ".calltargets" || ws0 == ".branchtargets" => Some(String::new()),
            "call" => rewrite_call(&inst, &guard, ids, sigs, callable, &param_vars, &protos, &targets, ctr)
                .map(|r| format!("{prefix}{r}")),
            "brx" => rewrite_brx(&inst, &guard, &btargets, ctr).map(|r| format!("{prefix}{r}")),
            _ if !ws0.starts_with('.') && inst.contains(|c: char| c.is_ascii_alphabetic()) => {
                // function names as operands: their numbers
                let ops = &inst[ws0.len()..];
                let sub = subst_idents(ops, ids);
                if sub != ops {
                    Some(format!("{prefix}{guard}{}{ws0}{sub}", if guard.is_empty() { "" } else { " " }))
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(r) = repl {
            out.push_str(&body[last..s]);
            out.push(' ');
            out.push_str(&r);
            last = e;
            // a rewritten statement that is a block (or nothing) takes the `;`
            if (r.trim().is_empty() || r.trim_end().ends_with('}')) && body.as_bytes().get(e) == Some(&b';') {
                last = e + 1;
            }
        }
    }
    out.push_str(&body[last..]);
    out
}

#[allow(clippy::too_many_arguments)]
fn rewrite_call(
    inst: &str,
    guard: &str,
    ids: &HashMap<String, u64>,
    sigs: &HashMap<String, Sig>,
    callable: &[String],
    param_vars: &HashSet<String>,
    protos: &HashMap<String, Sig>,
    targets: &HashMap<String, Vec<String>>,
    ctr: &mut usize,
) -> Option<String> {
    let op = inst.split_whitespace().next()?;
    let mut rest = inst[op.len()..].trim_start();
    // (rets),
    let mut rets: Vec<String> = Vec::new();
    if rest.starts_with('(') {
        let (r, after) = paren_list(rest)?;
        rets = split_top(r);
        rest = after.trim_start().strip_prefix(',')?.trim_start();
    }
    let parts = split_top(rest);
    let callee = parts.first()?.clone();
    let mut args: Vec<String> = Vec::new();
    let mut proto: Option<String> = None;
    for p in &parts[1..] {
        if p.starts_with('(') {
            args = split_top(paren_list(p)?.0);
        } else {
            proto = Some(p.clone());
        }
    }
    let direct = sigs.contains_key(&callee);
    // an unknown callee name (not a register): left as it is
    if !direct && !callee.starts_with('%') && proto.is_none() {
        return None;
    }
    let candidates: Vec<String> = if direct {
        vec![callee.clone()]
    } else {
        match proto.as_ref() {
            Some(pn) if targets.contains_key(pn) => targets[pn].clone(),
            Some(pn) if protos.contains_key(pn) => {
                callable.iter().filter(|f| sig_matches(&sigs[*f], &protos[pn])).cloned().collect()
            }
            _ => callable.iter().filter(|f| sigs[*f].params.len() == args.len() && sigs[*f].rets.len() == rets.len()).cloned().collect(),
        }
    };
    let mut blocks = Vec::new();
    let mut needs_rewrite = !direct;
    for f in &candidates {
        let sig = sigs.get(f)?;
        let mut decls = Vec::new();
        let mut pre = Vec::new();
        let mut post = Vec::new();
        let mut a_out = Vec::new();
        let mut r_out = Vec::new();
        for (i, fm) in sig.params.iter().enumerate() {
            let Some(a) = args.get(i) else {
                // an omitted (unsized array) argument
                let n = format!("__zluda_pa{}", *ctr);
                *ctr += 1;
                decls.push(decl_of(fm, &n));
                a_out.push(n);
                needs_rewrite = true;
                continue;
            };
            let is_pv = param_vars.contains(a.as_str());
            if fm.param && !is_pv {
                let n = format!("__zluda_pa{}", *ctr);
                *ctr += 1;
                decls.push(decl_of(fm, &n));
                pre.push(format!("st.param{} [{n}], {a};", fm.ty));
                a_out.push(n);
                needs_rewrite = true;
            } else if !fm.param && is_pv {
                let n = format!("%__zluda_ra{}", *ctr);
                *ctr += 1;
                decls.push(decl_of(fm, &n));
                pre.push(format!("ld.param{} {n}, [{a}];", fm.ty));
                a_out.push(n);
                needs_rewrite = true;
            } else {
                a_out.push(a.clone());
            }
        }
        for (i, fm) in sig.rets.iter().enumerate() {
            let r = rets.get(i).filter(|r| r.as_str() != "_");
            let Some(r) = r else {
                let n = if fm.param { format!("__zluda_pr{}", *ctr) } else { format!("%__zluda_rr{}", *ctr) };
                *ctr += 1;
                decls.push(decl_of(fm, &n));
                r_out.push(n);
                needs_rewrite = true;
                continue;
            };
            let is_pv = param_vars.contains(r.as_str());
            if fm.param && !is_pv {
                let n = format!("__zluda_pr{}", *ctr);
                *ctr += 1;
                decls.push(decl_of(fm, &n));
                post.push(format!("ld.param{} {r}, [{n}];", fm.ty));
                r_out.push(n);
                needs_rewrite = true;
            } else if !fm.param && is_pv {
                let n = format!("%__zluda_rr{}", *ctr);
                *ctr += 1;
                decls.push(decl_of(fm, &n));
                post.push(format!("st.param{} [{r}], {n};", fm.ty));
                r_out.push(n);
                needs_rewrite = true;
            } else {
                r_out.push(r.clone());
            }
        }
        if args.is_empty() && parts.len() > 1 && parts[1..].iter().any(|p| p.replace(' ', "") == "()") {
            needs_rewrite = true;
        }
        let call = format!(
            "{op} {}{f}{};",
            if r_out.is_empty() { String::new() } else { format!("({}), ", r_out.join(", ")) },
            if a_out.is_empty() { String::new() } else { format!(", ({})", a_out.join(", ")) }
        );
        blocks.push((f.clone(), decls, pre, call, post));
    }
    if !needs_rewrite {
        return None;
    }
    let mut s = String::from("{ ");
    if direct {
        let (_, decls, pre, call, post) = &blocks[0];
        let g = if guard.is_empty() { String::new() } else { format!("{guard} ") };
        s.push_str(&decls.join(" "));
        s.push(' ');
        s.push_str(&pre.join(" "));
        s.push_str(&format!(" {g}{call} "));
        for p in post {
            s.push_str(&format!("{g}{p} "));
        }
    } else {
        let k = *ctr;
        *ctr += 1;
        let c = format!("%__zluda_ic{k}");
        s.push_str(&format!(".reg .pred {c}; "));
        let g = if guard.is_empty() {
            None
        } else {
            let gn = format!("%__zluda_ig{k}");
            let src = guard.trim_start_matches('@');
            s.push_str(&format!(".reg .pred {gn}; "));
            if let Some(x) = src.strip_prefix('!') {
                s.push_str(&format!("not.pred {gn}, {x}; "));
            } else {
                s.push_str(&format!("mov.pred {gn}, {src}; "));
            }
            Some(gn)
        };
        for (f, decls, pre, call, post) in &blocks {
            let id = ids.get(f)?;
            s.push_str(&format!("{{ {} {} setp.eq.u64 {c}, {callee}, {id}; ", decls.join(" "), pre.join(" ")));
            if let Some(gn) = &g {
                s.push_str(&format!("and.pred {c}, {c}, {gn}; "));
            }
            s.push_str(&format!("@{c} {call} "));
            for p in post {
                s.push_str(&format!("@{c} {p} "));
            }
            s.push_str("} ");
        }
    }
    s.push('}');
    Some(s)
}

fn rewrite_brx(inst: &str, guard: &str, btargets: &HashMap<String, Vec<String>>, ctr: &mut usize) -> Option<String> {
    let op = inst.split_whitespace().next()?;
    if !op.starts_with("brx.idx") {
        return None;
    }
    let parts = split_top(&inst[op.len()..]);
    let [idx, tab] = &parts[..] else { return None };
    let labels = btargets.get(tab)?;
    let k = *ctr;
    *ctr += 1;
    let c = format!("%__zluda_bx{k}");
    let mut s = format!("{{ .reg .pred {c}; ");
    let g = if guard.is_empty() {
        None
    } else {
        let gn = format!("%__zluda_bg{k}");
        let src = guard.trim_start_matches('@');
        s.push_str(&format!(".reg .pred {gn}; "));
        if let Some(x) = src.strip_prefix('!') {
            s.push_str(&format!("not.pred {gn}, {x}; "));
        } else {
            s.push_str(&format!("mov.pred {gn}, {src}; "));
        }
        Some(gn)
    };
    for (i, l) in labels.iter().enumerate() {
        s.push_str(&format!("setp.eq.u32 {c}, {idx}, {i}; "));
        if let Some(gn) = &g {
            s.push_str(&format!("and.pred {c}, {c}, {gn}; "));
        }
        s.push_str(&format!("@{c} bra {l}; "));
    }
    s.push('}');
    Some(s)
}

/// weak definitions (11.6): the first weak definition of a name wins unless there is a strong one; an extern
/// declaration of a name defined in the module is dropped
fn link_weak(t: &str) -> String {
    if !t.contains(".weak") && !t.contains(".extern") {
        return t.to_string();
    }
    // module-scope statements: functions (header + body) and variable declarations
    let b = t.as_bytes();
    let mut items: Vec<(usize, usize, String, bool, bool, bool)> = Vec::new(); // start, end, name, weak, extern, is_def
    let mut i = 0;
    let mut depth = 0i32;
    let mut s = 0;
    while i < b.len() {
        match b[i] {
            b'"' => {
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    i += 1;
                }
            }
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    let st = &t[s..=i];
                    if let Some(it) = module_item(st) {
                        // a function definition, unless it is an initializer `= { ... }`
                        if !st[..st.find('{').unwrap_or(0)].contains('=') {
                            items.push((s, i + 1, it.0, it.1, it.2, true));
                            s = i + 1;
                        }
                    } else if !st.contains('=') {
                        s = i + 1;
                    }
                }
            }
            b';' if depth == 0 => {
                let st = &t[s..i];
                if let Some(it) = module_item(st) {
                    let is_fn = st.split_whitespace().any(|w| w == ".func" || w == ".entry");
                    // function declarations stay; a variable without .extern is a definition
                    if !is_fn {
                        items.push((s, i + 1, it.0, it.1, it.2, !it.2));
                    }
                }
                s = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    // an item starts at the last line that begins with a linkage / state-space / function keyword (directives
    // without `;` such as .version may precede it in the same scanned statement)
    for it in items.iter_mut() {
        let st = &t[it.0..it.1];
        let lim = st.find('{').unwrap_or(st.len());
        let mut best = None;
        let mut off = 0;
        for line in st[..lim].split_inclusive('\n') {
            let w = line.split_whitespace().next().unwrap_or("");
            if [".extern", ".weak", ".visible", ".global", ".const", ".shared", ".func", ".entry"].contains(&w) {
                best = Some(off + (line.len() - line.trim_start().len()));
            }
            off += line.len();
        }
        if let Some(b) = best {
            it.0 += b;
        }
    }
    let mut drop: Vec<(usize, usize)> = Vec::new();
    let mut names: Vec<&String> = items.iter().map(|x| &x.2).collect();
    names.sort();
    names.dedup();
    for n in names {
        let group: Vec<_> = items.iter().filter(|x| &x.2 == n).collect();
        if group.len() < 2 {
            continue;
        }
        let strong = group.iter().find(|x| !x.3 && !x.4 && x.5);
        let keep = strong.or_else(|| group.iter().find(|x| x.5 && !x.4));
        let Some(keep) = keep else { continue };
        for x in &group {
            if !std::ptr::eq(*x, *keep) {
                drop.push((x.0, x.1));
            }
        }
    }
    if drop.is_empty() {
        return t.to_string();
    }
    let mut out = b.to_vec();
    for (a, e) in drop {
        for c in &mut out[a..e] {
            if *c != b'\n' {
                *c = b' ';
            }
        }
    }
    String::from_utf8(out).unwrap_or_else(|_| t.to_string())
}

/// (name, weak, extern) of a module-scope variable or function statement
fn module_item(st: &str) -> Option<(String, bool, bool)> {
    let ws: Vec<&str> = st.split_whitespace().collect();
    let weak = ws.contains(&".weak");
    let ext = ws.contains(&".extern");
    if ws.contains(&".func") || ws.contains(&".entry") {
        let p = st.find(".func").or_else(|| st.find(".entry"))?;
        let kw = if st[p..].starts_with(".func") { 5 } else { 6 };
        let (name, _, _) = parse_header(&st[p + kw..])?;
        return Some((name, weak, ext));
    }
    if !(ws.contains(&".global") || ws.contains(&".const") || ws.contains(&".shared")) {
        return None;
    }
    let head = st.split('=').next()?;
    let last = head.split_whitespace().last()?;
    let name: String = last.bytes().take_while(|c| is_ident_byte(*c)).map(|c| c as char).collect();
    if name.is_empty() {
        return None;
    }
    Some((name, weak, ext))
}

/// a function that calls itself: copies f, f__zluda_r1 .. f__zluda_rN where copy k calls copy k + 1 and the last
/// copy's recursive calls are removed (the depth bound)
fn unroll_recursion(t: &str) -> String {
    let funcs = functions(t);
    let mut out = t.to_string();
    for f in funcs.iter().rev() {
        let (Some((bs, be)), false) = (f.body, f.entry) else { continue };
        let body = &t[bs..be];
        if !calls_name(body, &f.name) {
            continue;
        }
        let def = &t[f.start..f.end];
        let mut copies = String::new();
        for k in 1..=RECURSION_DEPTH {
            let name_k = format!("{}__zluda_r{k}", f.name);
            let next = format!("{}__zluda_r{}", f.name, k + 1);
            let mut d = rename_def(def, &f.name, &name_k);
            if k < RECURSION_DEPTH {
                d = rename_calls(&d, &f.name, &next);
            } else {
                d = rename_calls(&d, &f.name, "__zluda_recursion_limit");
            }
            copies.push('\n');
            copies.push_str(&d);
        }
        let first = rename_calls(def, &f.name, &format!("{}__zluda_r1", f.name));
        // declarations of the copies before the function, the copies after it (in reverse dependency order: a copy
        // calls the next one, which must be declared first)
        let mut decls = String::new();
        if let Some(h) = def.find('{') {
            for k in 1..=RECURSION_DEPTH {
                let hd = rename_def(&def[..h], &f.name, &format!("{}__zluda_r{k}", f.name));
                let hd = hd.replace(".visible", "").replace(".weak", "");
                decls.push_str(&format!("{};\n", hd.trim()));
            }
        }
        out.replace_range(f.start..f.end, &format!("{decls}{first}{copies}"));
    }
    if out.contains("__zluda_recursion_limit") {
        // calls beyond the depth bound are removed
        let mut r = String::new();
        for st in out.split_inclusive(';') {
            if st.contains("__zluda_recursion_limit") && st.contains("call") {
                // keep everything before the call statement's start (labels, braces)
                let p = st.rfind(|c| c == '{' || c == '}' || c == '\n').map_or(0, |p| p + 1);
                r.push_str(&st[..p]);
                r.push_str(" trap;");
            } else {
                r.push_str(st);
            }
        }
        out = r;
    }
    out
}

fn calls_name(body: &str, name: &str) -> bool {
    body.split(';').any(|st| {
        let s = st.trim_start();
        let s = s.trim_start_matches(|c: char| c == '{' || c == '}' || c.is_whitespace());
        (s.starts_with("call") || s.contains(" call") || s.contains("\tcall")) && call_target(s) == Some(name)
    })
}

fn call_target(st: &str) -> Option<&str> {
    let p = st.find("call")?;
    let mut rest = &st[p..];
    rest = rest[rest.find(char::is_whitespace)?..].trim_start();
    if rest.starts_with('(') {
        let (_, after) = paren_list(rest)?;
        rest = after.trim_start().strip_prefix(',')?.trim_start();
    }
    let e = rest.find(|c: char| c == ',' || c == ';' || c.is_whitespace()).unwrap_or(rest.len());
    Some(&rest[..e])
}

fn rename_def(def: &str, from: &str, to: &str) -> String {
    // the name in the header (first occurrence after .func)
    let p = def.find(".func").unwrap_or(0);
    let head_end = def[p..].find(|c| c == '{' || c == ';').map_or(def.len(), |e| p + e);
    let head = &def[..head_end];
    let q = find_word(head, from, p).unwrap_or(head.len());
    if q >= head.len() {
        return def.to_string();
    }
    format!("{}{}{}", &def[..q], to, &def[q + from.len()..])
}

fn rename_calls(def: &str, from: &str, to: &str) -> String {
    let mut out = String::new();
    for st in def.split_inclusive(';') {
        if call_target(st) == Some(from) {
            let p = st.find("call").unwrap();
            let rest = &st[p..];
            let q = find_word(rest, from, 0).unwrap();
            out.push_str(&st[..p]);
            out.push_str(&rest[..q]);
            out.push_str(to);
            out.push_str(&rest[q + from.len()..]);
        } else {
            out.push_str(st);
        }
    }
    out
}
