//! Source-level normalisation run before lexing (PTX ISA 9.3 / 9.4 syntax the grammar does not cover):
//!
//! * constant expressions (4.5.4 - 4.5.6) in operands, address offsets, initializers and array sizes are evaluated
//!   with C precedence and 64-bit .s64 / .u64 / .f64 arithmetic and replaced by their value; binary literals become
//!   decimal; a mask initializer on constants `0xff00(expr)` (5.4.4) is evaluated;
//! * `generic(var)` initializers (5.4.4) become `var` (the generic address of a .global variable is its address);
//! * directives without effect on execution are dropped: `.pragma` (any position, string lists), `.attribute(...)`,
//!   `.loc_intermediate`, `.nv_intermediate_source_section { ... }`, `.section NAME { ... }` debug sections, `.alias`,
//!   `.target` options such as `texmode_independent`;
//! * parameterized names of .global / .shared / .local / .const variables (`g<3>`, 5.4.6) are expanded.
use std::borrow::Cow;

pub fn preprocess(text: &str) -> Cow<'_, str> {
    if !needs_work(text) && !super::preprocess_calls::needs_work(text) {
        return Cow::Borrowed(text);
    }
    // a rewrite that fails leaves the text as it is (the parser reports what it does not accept)
    let r = std::panic::catch_unwind(|| rewrite_all(text));
    match r {
        Ok(s) => Cow::Owned(s),
        Err(_) => Cow::Borrowed(text),
    }
}

fn rewrite_all(text: &str) -> String {
    let s = drop_directives(text);
    let s = fold_expressions(&s);
    let s = expand_param_names(&s);
    let s = array_operands(&s);
    let s = colour_members(&s);
    let s = vector_members_in_lists(&s);
    let s = per_cta_memory(&s);
    if super::preprocess_calls::needs_work(&s) {
        super::preprocess_calls::rewrite(&s)
    } else {
        s
    }
}

fn needs_work(text: &str) -> bool {
    let b = text.as_bytes();
    // any operator character outside the plain grammar, a directive handled here, or a binary literal
    text.contains(".pragma")
        || text.contains(".attribute")
        || text.contains("_intermediate")
        || text.contains(".section")
        || text.contains(".alias")
        || text.contains("generic(")
        || text.contains("texmode_")
        || text.contains("compute_")
        || text.contains(".minperctamemory")
        || text.contains("0b")
        || text.contains('[')
        || b.iter().any(|c| matches!(c, b'*' | b'/' | b'%' | b'&' | b'^' | b'~' | b'?' | b'<' | b'>' | b'(' | b'+' | b'-'))
}

// ------------------------------------------------------------------------------------------------ lexical helpers

/// byte ranges of comments and string literals (left untouched)
fn opaque_ranges(t: &str) -> Vec<(usize, usize)> {
    let b = t.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            let s = i;
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            out.push((s, i));
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            let s = i;
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(b.len());
            out.push((s, i));
        } else if b[i] == b'"' {
            let s = i;
            i += 1;
            while i < b.len() && b[i] != b'"' {
                i += 1;
            }
            i = (i + 1).min(b.len());
            out.push((s, i));
        } else {
            i += 1;
        }
    }
    out
}

fn is_word_start(c: u8) -> bool {
    c.is_ascii_alphabetic() || c == b'_' || c == b'$' || c == b'%' || c == b'.'
}
fn is_word(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'$' || c == b'%' || c == b'.' || c == b':'
}

/// position of the next occurrence of `pat` at or after `from` outside comments / strings, as a whole directive word
fn find_directive(t: &str, opaque: &[(usize, usize)], pat: &str, from: usize) -> Option<usize> {
    let mut start = from;
    while let Some(p) = t[start..].find(pat) {
        let p = start + p;
        let end = p + pat.len();
        let inside = opaque.iter().any(|&(a, b)| p >= a && p < b);
        let before_ok = p == 0 || !is_word(t.as_bytes()[p - 1]);
        let after_ok = end >= t.len() || !is_word(t.as_bytes()[end]) || pat.ends_with('(');
        if !inside && before_ok && after_ok {
            return Some(p);
        }
        start = end;
    }
    None
}

/// the end of the statement starting at `p` (after its ';'), outside strings
fn statement_end(t: &str, p: usize) -> usize {
    let b = t.as_bytes();
    let mut i = p;
    while i < b.len() {
        if b[i] == b'"' {
            i += 1;
            while i < b.len() && b[i] != b'"' {
                i += 1;
            }
        } else if b[i] == b';' {
            return i + 1;
        }
        i += 1;
    }
    b.len()
}

/// the end of `.pragma "a", "b"{;}` (the `;` is absent between a kernel's header and its body)
fn pragma_end(t: &str, mut i: usize) -> usize {
    let b = t.as_bytes();
    loop {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < b.len() && b[i] == b'"' {
            i += 1;
            while i < b.len() && b[i] != b'"' {
                i += 1;
            }
            i = (i + 1).min(b.len());
        }
        let mut j = i;
        while j < b.len() && b[j].is_ascii_whitespace() {
            j += 1;
        }
        if j < b.len() && b[j] == b',' {
            i = j + 1;
            continue;
        }
        if j < b.len() && b[j] == b';' {
            return j + 1;
        }
        return i;
    }
}

/// the end of a brace block whose `{` is the first one at or after `p` (balanced), outside comments / strings
fn block_end(t: &str, p: usize) -> usize {
    let b = t.as_bytes();
    let mut i = p;
    while i < b.len() && b[i] != b'{' {
        i += 1;
    }
    let mut depth = 0i32;
    while i < b.len() {
        match b[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return i + 1;
                }
            }
            b'"' => {
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    b.len()
}

/// replaces [a, b) by spaces, keeping newlines (line numbers of later errors stay meaningful)
fn blank(out: &mut [u8], a: usize, b: usize) {
    for c in &mut out[a..b] {
        if *c != b'\n' {
            *c = b' ';
        }
    }
}

// ------------------------------------------------------------------------------------------------ directives

fn drop_directives(text: &str) -> String {
    let mut t = text.to_string();
    // .nv_intermediate_source_section: arbitrary text between .source_begin and .source_end
    loop {
        let op = opaque_ranges(&t);
        let Some(p) = find_directive(&t, &op, ".nv_intermediate_source_section", 0) else { break };
        let mut b = t.clone().into_bytes();
        // the sources first (unbalanced braces), then the section's block
        let mut q = p;
        while let Some(s) = t[q..].find(".source_begin") {
            let s = q + s;
            let e = t[s..].find(".source_end").map(|e| s + e + ".source_end".len()).unwrap_or(t.len());
            blank(&mut b, s, e);
            q = e;
        }
        let t2 = String::from_utf8(b).unwrap();
        let e = block_end(&t2, p);
        let mut b = t2.into_bytes();
        blank(&mut b, p, e);
        t = String::from_utf8(b).unwrap();
    }
    let op = opaque_ranges(&t);
    let mut b = t.clone().into_bytes();
    for d in [".pragma", ".loc_intermediate", ".alias"] {
        let mut from = 0;
        while let Some(p) = find_directive(&t, &op, d, from) {
            let e = match d {
                ".loc_intermediate" => t[p..].find('\n').map_or(t.len(), |e| p + e),
                ".pragma" => pragma_end(&t, p + d.len()),
                _ => statement_end(&t, p),
            };
            blank(&mut b, p, e);
            from = e;
        }
    }
    // .section NAME { ... }
    let mut from = 0;
    while let Some(p) = find_directive(&t, &op, ".section", from) {
        let e = block_end(&t, p);
        blank(&mut b, p, e);
        from = e;
    }
    // .attribute( ... )
    let mut from = 0;
    while let Some(p) = find_directive(&t, &op, ".attribute", from) {
        let bb = t.as_bytes();
        let mut i = p + ".attribute".len();
        while i < bb.len() && bb[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < bb.len() && bb[i] == b'(' {
            let mut depth = 0;
            while i < bb.len() {
                if bb[i] == b'(' {
                    depth += 1;
                } else if bb[i] == b')' {
                    depth -= 1;
                    if depth == 0 {
                        i += 1;
                        break;
                    }
                }
                i += 1;
            }
            blank(&mut b, p, i);
        }
        from = i.max(p + 1);
    }
    // .target sm_90a, texmode_independent: the options
    let mut from = 0;
    while let Some(p) = find_directive(&t, &op, ".target", from) {
        let e = t[p..].find('\n').map_or(t.len(), |e| p + e);
        let line = &t[p..e];
        let mut keep = Vec::new();
        for part in line[".target".len()..].split(',') {
            let w = part.trim();
            if !(w.starts_with("texmode_") || w == "map_f64_to_f32" || w == "debug") {
                // compute_NN is a synonym of sm_NN (11.1.2)
                keep.push(match w.strip_prefix("compute_") {
                    Some(v) => format!("sm_{v}"),
                    None => w.to_string(),
                });
            }
        }
        let repl = format!(".target {}", keep.join(", "));
        if repl.len() <= line.len() {
            let mut r = repl.into_bytes();
            r.resize(line.len(), b' ');
            b[p..e].copy_from_slice(&r);
        }
        from = e;
    }
    String::from_utf8(b).unwrap()
}

// ------------------------------------------------------------------------------------------------ expressions

#[derive(Clone, Copy, Debug, PartialEq)]
enum Val {
    I(u64, bool), // bits, signed
    F(f64),
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(Val),
    Op(&'static str),
    Cast(bool), // (.s64) / (.u64): signed
    LParen,
    RParen,
}

const OPS: [&str; 22] = [
    "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "+", "-", "*", "/", "%", "&", "|", "^", "~", "!", "<", ">", "?",
    ":",
];

fn parse_num(s: &str) -> Option<Val> {
    let (body, uns) = match s.strip_suffix('U') {
        Some(b) => (b, true),
        None => (s, false),
    };
    let v = if let Some(h) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        u64::from_str_radix(h, 16).ok()?
    } else if let Some(h) = body.strip_prefix("0b").or_else(|| body.strip_prefix("0B")) {
        u64::from_str_radix(h, 2).ok()?
    } else if body.bytes().all(|c| c.is_ascii_digit()) && !body.is_empty() {
        if body.len() > 1 && body.starts_with('0') {
            u64::from_str_radix(&body[1..], 8).ok()?
        } else {
            body.parse::<u64>().ok()?
        }
    } else {
        if uns {
            return None;
        }
        return body.parse::<f64>().ok().map(Val::F);
    };
    // "literals are signed unless U or too large"
    Some(Val::I(v, !uns && v <= i64::MAX as u64))
}

/// one lexical item of the folding pass
#[derive(Debug)]
enum Item<'a> {
    Ws(&'a str),
    Opaque(&'a str),
    Word(&'a str),
    Num(&'a str),
    Op(&'static str),
    Cast(bool, &'a str),
    LParen,
    RParen,
    Other(&'a str),
}

fn lex(t: &str) -> Vec<Item<'_>> {
    let b = t.as_bytes();
    let op = opaque_ranges(t);
    let mut oi = 0;
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if oi < op.len() && op[oi].0 == i {
            out.push(Item::Opaque(&t[op[oi].0..op[oi].1]));
            i = op[oi].1;
            oi += 1;
            continue;
        }
        let c = b[i];
        if c.is_ascii_whitespace() {
            let s = i;
            while i < b.len() && b[i].is_ascii_whitespace() && !(oi < op.len() && op[oi].0 == i) {
                i += 1;
            }
            out.push(Item::Ws(&t[s..i]));
        } else if c.is_ascii_digit() || (c == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
            let s = i;
            if c == b'0' && i + 1 < b.len() && matches!(b[i + 1], b'f' | b'F' | b'd' | b'D') {
                // hex float bit patterns: opaque words
                i += 2;
                while i < b.len() && b[i].is_ascii_alphanumeric() {
                    i += 1;
                }
                out.push(Item::Word(&t[s..i]));
                continue;
            }
            if c == b'0' && i + 1 < b.len() && matches!(b[i + 1], b'x' | b'X' | b'b' | b'B') {
                i += 2;
                while i < b.len() && b[i].is_ascii_hexdigit() {
                    i += 1;
                }
            } else {
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                if i < b.len() && b[i] == b'.' {
                    i += 1;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < b.len() && matches!(b[i], b'e' | b'E') {
                    let mut j = i + 1;
                    if j < b.len() && matches!(b[j], b'+' | b'-') {
                        j += 1;
                    }
                    if j < b.len() && b[j].is_ascii_digit() {
                        i = j;
                        while i < b.len() && b[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                }
            }
            if i < b.len() && b[i] == b'U' {
                i += 1;
            }
            if i < b.len() && is_word(b[i]) {
                // part of an identifier-like word (e.g. 1d)
                while i < b.len() && is_word(b[i]) {
                    i += 1;
                }
                out.push(Item::Word(&t[s..i]));
            } else {
                out.push(Item::Num(&t[s..i]));
            }
        } else if is_word_start(c)
            && !(matches!(c, b'%' | b'.') && !(i + 1 < b.len() && (b[i + 1].is_ascii_alphabetic() || matches!(b[i + 1], b'_' | b'$' | b'%'))))
        {
            let s = i;
            i += 1;
            while i < b.len() && is_word(b[i]) {
                i += 1;
            }
            // `::` belongs to words; a single ':' after a word is a label / ternary colon
            let mut e = i;
            while e > s + 1 && b[e - 1] == b':' && !(e >= 2 && b[e - 2] == b':') {
                e -= 1;
            }
            i = e;
            out.push(Item::Word(&t[s..i]));
        } else if c == b'(' {
            // a cast (.s64) / (.u64)
            let rest = &t[i..];
            let cast = ["(.s64)", "(.u64)"].iter().find(|p| rest.starts_with(**p));
            if let Some(p) = cast {
                out.push(Item::Cast(p.contains('s'), &t[i..i + p.len()]));
                i += p.len();
            } else {
                out.push(Item::LParen);
                i += 1;
            }
        } else if c == b')' {
            out.push(Item::RParen);
            i += 1;
        } else if let Some(o) = OPS.iter().find(|o| t[i..].starts_with(**o)) {
            out.push(Item::Op(o));
            i += o.len();
        } else {
            let s = i;
            i += 1;
            while i < b.len() && !t.is_char_boundary(i) {
                i += 1;
            }
            out.push(Item::Other(&t[s..i]));
        }
    }
    out
}

struct Eval<'a> {
    t: &'a [Tok],
    p: usize,
}

fn to_i(v: Val) -> (u64, bool) {
    match v {
        Val::I(x, s) => (x, s),
        Val::F(f) => (f as i64 as u64, true),
    }
}
fn to_f(v: Val) -> f64 {
    match v {
        Val::I(x, true) => x as i64 as f64,
        Val::I(x, false) => x as f64,
        Val::F(f) => f,
    }
}
fn truthy(v: Val) -> bool {
    match v {
        Val::I(x, _) => x != 0,
        Val::F(f) => f != 0.0,
    }
}

impl<'a> Eval<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.p)
    }
    fn eat_op(&mut self, o: &str) -> bool {
        if let Some(Tok::Op(x)) = self.peek() {
            if *x == o {
                self.p += 1;
                return true;
            }
        }
        false
    }
    fn ternary(&mut self) -> Option<Val> {
        let c = self.binary(0)?;
        if self.eat_op("?") {
            let a = self.ternary()?;
            if !self.eat_op(":") {
                return None;
            }
            let b = self.ternary()?;
            return Some(if truthy(c) { a } else { b });
        }
        Some(c)
    }
    fn prec(o: &str) -> Option<u8> {
        Some(match o {
            "||" => 1,
            "&&" => 2,
            "|" => 3,
            "^" => 4,
            "&" => 5,
            "==" | "!=" => 6,
            "<" | ">" | "<=" | ">=" => 7,
            "<<" | ">>" => 8,
            "+" | "-" => 9,
            "*" | "/" | "%" => 10,
            _ => return None,
        })
    }
    fn binary(&mut self, min: u8) -> Option<Val> {
        let mut l = self.unary()?;
        loop {
            let Some(Tok::Op(o)) = self.peek() else { break };
            let o = *o;
            let Some(pr) = Self::prec(o) else { break };
            if pr <= min {
                break;
            }
            self.p += 1;
            let r = self.binary(pr)?;
            l = apply(o, l, r)?;
        }
        Some(l)
    }
    fn unary(&mut self) -> Option<Val> {
        match self.peek()?.clone() {
            Tok::Op("-") => {
                self.p += 1;
                Some(match self.unary()? {
                    Val::I(x, s) => Val::I(x.wrapping_neg(), s),
                    Val::F(f) => Val::F(-f),
                })
            }
            Tok::Op("+") => {
                self.p += 1;
                self.unary()
            }
            Tok::Op("~") => {
                self.p += 1;
                let (x, s) = to_i(self.unary()?);
                Some(Val::I(!x, s))
            }
            Tok::Op("!") => {
                self.p += 1;
                Some(Val::I(!truthy(self.unary()?) as u64, true))
            }
            Tok::Cast(s) => {
                self.p += 1;
                let v = self.unary()?;
                Some(Val::I(to_i(v).0, s))
            }
            Tok::LParen => {
                self.p += 1;
                let v = self.ternary()?;
                if self.peek()? != &Tok::RParen {
                    return None;
                }
                self.p += 1;
                Some(v)
            }
            Tok::Num(v) => {
                self.p += 1;
                // mask(expr): a constant followed by a parenthesized expression (5.4.4): the masked bits moved
                // to the lowest position
                if let (Val::I(m, _), Some(Tok::LParen)) = (v, self.peek()) {
                    self.p += 1;
                    let e = self.ternary()?;
                    if self.peek()? != &Tok::RParen {
                        return None;
                    }
                    self.p += 1;
                    let (x, _) = to_i(e);
                    let sh = if m == 0 { 0 } else { m.trailing_zeros() };
                    return Some(Val::I((x & m) >> sh, true));
                }
                Some(v)
            }
            _ => None,
        }
    }
}

fn apply(o: &str, a: Val, b: Val) -> Option<Val> {
    let float = matches!(a, Val::F(_)) || matches!(b, Val::F(_));
    if float {
        let (x, y) = (to_f(a), to_f(b));
        return Some(match o {
            "+" => Val::F(x + y),
            "-" => Val::F(x - y),
            "*" => Val::F(x * y),
            "/" => Val::F(x / y),
            "<" => Val::I((x < y) as u64, true),
            ">" => Val::I((x > y) as u64, true),
            "<=" => Val::I((x <= y) as u64, true),
            ">=" => Val::I((x >= y) as u64, true),
            "==" => Val::I((x == y) as u64, true),
            "!=" => Val::I((x != y) as u64, true),
            "&&" => Val::I((x != 0.0 && y != 0.0) as u64, true),
            "||" => Val::I((x != 0.0 || y != 0.0) as u64, true),
            _ => return None,
        });
    }
    let (x, sa) = to_i(a);
    let (y, sb) = to_i(b);
    let s = sa && sb;
    Some(match o {
        "+" => Val::I(x.wrapping_add(y), s),
        "-" => Val::I(x.wrapping_sub(y), s),
        "*" => Val::I(x.wrapping_mul(y), s),
        "/" => {
            if y == 0 {
                return None;
            }
            Val::I(if s { (x as i64).wrapping_div(y as i64) as u64 } else { x / y }, s)
        }
        "%" => {
            if y == 0 {
                return None;
            }
            // "% on .u64 operands" (4.5.4)
            Val::I(x % y, false)
        }
        // shifts keep the first operand's type
        "<<" => Val::I(if y >= 64 { 0 } else { x << y }, sa),
        ">>" => Val::I(
            if sa {
                (x as i64 >> y.min(63)) as u64
            } else if y >= 64 {
                0
            } else {
                x >> y
            },
            sa,
        ),
        "&" => Val::I(x & y, s),
        "|" => Val::I(x | y, s),
        "^" => Val::I(x ^ y, s),
        "<" | ">" | "<=" | ">=" => {
            let r = if s {
                let (x, y) = (x as i64, y as i64);
                match o {
                    "<" => x < y,
                    ">" => x > y,
                    "<=" => x <= y,
                    _ => x >= y,
                }
            } else {
                match o {
                    "<" => x < y,
                    ">" => x > y,
                    "<=" => x <= y,
                    _ => x >= y,
                }
            };
            Val::I(r as u64, true)
        }
        "==" => Val::I((x == y) as u64, true),
        "!=" => Val::I((x != y) as u64, true),
        "&&" => Val::I((x != 0 && y != 0) as u64, true),
        "||" => Val::I((x != 0 || y != 0) as u64, true),
        _ => return None,
    })
}

fn render(v: Val) -> String {
    match v {
        Val::I(x, true) => format!("{}", x as i64),
        Val::I(x, false) if x <= i64::MAX as u64 => format!("{x}"),
        Val::I(x, false) => format!("0x{x:X}U"),
        Val::F(f) => {
            let s = format!("{f:?}");
            if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") {
                s
            } else {
                format!("{s}.0")
            }
        }
    }
}

fn fold_expressions(t: &str) -> String {
    let items = lex(t);
    let mut out = String::with_capacity(t.len());
    let mut i = 0;
    // the first word of the current statement and the last significant item before position i
    let mut stmt_word: Option<&str> = None;
    let mut prev_sig: Option<usize> = None;
    let is_run_item = |it: &Item| match it {
        Item::Num(_) | Item::Op(_) | Item::Cast(..) | Item::LParen | Item::RParen => true,
        Item::Word(w) => *w == "WARP_SZ",
        _ => false,
    };
    while i < items.len() {
        if !is_run_item(&items[i]) {
            match &items[i] {
                Item::Ws(s) | Item::Opaque(s) | Item::Other(s) => out.push_str(s),
                Item::Word(w) => {
                    out.push_str(w);
                    if stmt_word.is_none() && !w.starts_with('@') && !w.starts_with('%') {
                        stmt_word = Some(w);
                    }
                }
                _ => {}
            }
            if let Item::Other(";") | Item::Other("{") | Item::Other("}") = &items[i] {
                stmt_word = None;
            }
            if !matches!(items[i], Item::Ws(_) | Item::Opaque(_)) {
                prev_sig = Some(i);
            }
            i += 1;
            continue;
        }
        // a run: run items and the whitespace between them
        let s = i;
        let mut e = i;
        let mut j = i;
        let mut has_q = false;
        while j < items.len() {
            match &items[j] {
                Item::Ws(_) => {}
                Item::Op(":") if !has_q => break,
                it if is_run_item(it) => {
                    if let Item::Op("?") = it {
                        has_q = true;
                    }
                    e = j + 1;
                }
                _ => break,
            }
            j += 1;
        }
        if e == s {
            // a lone ':' (a label)
            if let Item::Op(o) = &items[i] {
                out.push_str(o);
            }
            prev_sig = Some(i);
            i += 1;
            continue;
        }
        let run: Vec<usize> = (s..e).filter(|&k| !matches!(items[k], Item::Ws(_))).collect();
        let emit_raw = |out: &mut String, a: usize, b: usize| {
            for it in &items[a..b] {
                match it {
                    Item::Ws(x) | Item::Opaque(x) | Item::Word(x) | Item::Num(x) | Item::Other(x) => out.push_str(x),
                    Item::Op(o) => out.push_str(o),
                    Item::Cast(_, x) => out.push_str(x),
                    Item::LParen => out.push('('),
                    Item::RParen => out.push(')'),
                }
            }
        };
        match fold_run(&items, &run, prev_sig.map(|p| &items[p]), stmt_word) {
            Some((lead, text, last)) => {
                // tokens before `lead` stay as they are
                emit_raw(&mut out, s, lead);
                out.push_str(&text);
                emit_raw(&mut out, last + 1, e);
            }
            None => {
                // binary literals at least
                for k in s..e {
                    match &items[k] {
                        Item::Num(x) if x.starts_with("0b") || x.starts_with("0B") => {
                            out.push_str(&parse_num(x).map(render).unwrap_or_else(|| x.to_string()))
                        }
                        _ => emit_raw(&mut out, k, k + 1),
                    }
                }
            }
        }
        prev_sig = run.last().copied().or(prev_sig);
        i = e;
    }
    // generic(var) -> var
    let mut r = out;
    while let Some(p) = r.find("generic(") {
        let before_ok = p == 0 || !is_word(r.as_bytes()[p - 1]);
        let Some(q) = r[p..].find(')') else { break };
        let inner = r[p + 8..p + q].trim().to_string();
        if !before_ok || inner.is_empty() || !inner.bytes().all(is_word) {
            break;
        }
        r.replace_range(p..p + q + 1, &inner);
    }
    r
}

/// evaluates the run `run` (indices of significant items) if it is a constant expression with an operator:
/// (index of the first replaced item, replacement, index of the last replaced item)
fn fold_run(items: &[Item], run: &[usize], prev: Option<&Item>, stmt: Option<&str>) -> Option<(usize, String, usize)> {
    let mut a = 0;
    let mut b = run.len();
    // a leading ')' closes an outer group; a trailing '(' opens one
    while a < b && matches!(items[run[a]], Item::RParen) {
        a += 1;
    }
    while b > a && matches!(items[run[b - 1]], Item::LParen) {
        b -= 1;
    }
    // operators that cannot start / end an expression
    while a < b && matches!(items[run[a]], Item::Op(o) if !matches!(o, "-" | "+" | "~" | "!")) {
        a += 1;
    }
    while b > a && matches!(items[run[b - 1]], Item::Op(_) | Item::Cast(..)) {
        b -= 1;
    }
    if a >= b {
        return None;
    }
    // balance: drop unmatched parentheses at the ends
    loop {
        let mut depth = 0i32;
        let mut bad_close = None;
        for k in a..b {
            match items[run[k]] {
                Item::LParen => depth += 1,
                Item::RParen => {
                    depth -= 1;
                    if depth < 0 && bad_close.is_none() {
                        bad_close = Some(k);
                    }
                }
                _ => {}
            }
        }
        if let Some(k) = bad_close {
            b = k;
        } else if depth > 0 {
            if matches!(items[run[a]], Item::LParen) {
                a += 1;
            } else {
                return None;
            }
        } else {
            break;
        }
        while b > a && matches!(items[run[b - 1]], Item::Op(_) | Item::Cast(..) | Item::LParen) {
            b -= 1;
        }
        if a >= b {
            return None;
        }
    }
    let toks: Vec<Tok> = run[a..b]
        .iter()
        .map(|&k| match &items[k] {
            Item::Num(x) => parse_num(x).map(Tok::Num),
            Item::Word(_) => Some(Tok::Num(Val::I(32, true))),
            Item::Op(o) => Some(Tok::Op(o)),
            Item::Cast(s, _) => Some(Tok::Cast(*s)),
            Item::LParen => Some(Tok::LParen),
            Item::RParen => Some(Tok::RParen),
            _ => None,
        })
        .collect::<Option<_>>()?;
    let nums = toks.iter().filter(|t| matches!(t, Tok::Num(_))).count();
    let ops = toks.iter().filter(|t| matches!(t, Tok::Op(_) | Tok::Cast(_))).count();
    let has_warp = run[a..b].iter().any(|&k| matches!(items[k], Item::Word(_)));
    let has_bin = run[a..b].iter().any(|&k| matches!(&items[k], Item::Num(x) if x.starts_with("0b") || x.starts_with("0B")));
    if nums == 0 {
        return None;
    }
    // after a register / address / closing bracket, a leading + or - is binary: the value is emitted as `+ v` / `- v`
    let binary_lead = matches!(toks[0], Tok::Op("+") | Tok::Op("-"))
        && matches!(prev, Some(Item::Word(_)) | Some(Item::Other("]")))
        && !matches!(prev, Some(Item::Word(w)) if *w == "WARP_SZ");
    if binary_lead && toks[1..].iter().all(|t| matches!(t, Tok::Op("+") | Tok::Op("-") | Tok::Num(_))) && nums == 1 {
        // `[a+-4]`: the grammar's own offset form
        return None;
    }
    let has_mask = toks.windows(2).any(|w| matches!(w, [Tok::Num(Val::I(..)), Tok::LParen]));
    let simple = ops == 0 || (ops == 1 && toks.len() == 2 && matches!(toks[0], Tok::Op("-") | Tok::Op("+")));
    let parens_only = toks.iter().all(|t| matches!(t, Tok::Num(_) | Tok::LParen | Tok::RParen));
    if (simple || parens_only) && !has_warp && !has_bin && !has_mask {
        if !(simple && ops == 1 && !binary_lead) {
            return None;
        }
    }
    let mut ev = Eval { t: &toks, p: 0 };
    let v = ev.ternary()?;
    if ev.p != toks.len() {
        return None;
    }
    let text = if binary_lead {
        match v {
            Val::I(x, true) if (x as i64) < 0 => format!("- {}", (x as i64).unsigned_abs()),
            Val::F(f) if f < 0.0 => format!("- {}", render(Val::F(-f))),
            _ => format!("+ {}", render(v)),
        }
    } else {
        let r = render(v);
        // keep the parentheses of a call argument list: call (d), f, (1 + 2)
        let wrapped = matches!(toks.first(), Some(Tok::LParen)) && matches!(toks.last(), Some(Tok::RParen));
        if wrapped && stmt.map_or(false, |w| w.starts_with("call")) {
            format!("({r})")
        } else {
            r
        }
    };
    Some((run[a], text, run[b - 1]))
}

// ------------------------------------------------------------------------------------------------ declarations

/// `.global .u32 g<3>;` -> `.global .u32 g0; .global .u32 g1; .global .u32 g2;` (not .reg / .param)
fn expand_param_names(t: &str) -> String {
    let mut out = String::with_capacity(t.len());
    let mut rest = t;
    loop {
        let Some(semi) = rest.find(';') else {
            out.push_str(rest);
            break;
        };
        let stmt = &rest[..semi];
        let start = stmt.rfind(|c| c == ';' || c == '{' || c == '}' || c == '\n').map_or(0, |p| p + 1);
        let s = stmt[start..].trim();
        let spaces = [".global", ".shared", ".local", ".const"];
        let mut done = false;
        if spaces.iter().any(|sp| s.split_whitespace().any(|w| w == *sp))
            && !s.contains('=')
            && !s.contains(',')
            && s.ends_with('>')
        {
            if let Some(lt) = s.rfind('<') {
                let n = s[lt + 1..s.len() - 1].trim();
                let head = &s[..lt];
                if let (Ok(n), Some(sp)) = (n.parse::<usize>(), head.rfind(char::is_whitespace)) {
                    let name = &head[sp + 1..];
                    let decl = &head[..sp + 1];
                    if !name.is_empty() && name.bytes().all(is_word) && n > 0 && n <= 1024 {
                        out.push_str(&rest[..start]);
                        let lead = &stmt[start..].len() - stmt[start..].trim_start().len();
                        out.push_str(&stmt[start..start + lead]);
                        let parts: Vec<String> = (0..n).map(|i| format!("{decl}{name}{i}")).collect();
                        out.push_str(&parts.join("; "));
                        out.push(';');
                        done = true;
                    }
                }
            }
        }
        if !done {
            out.push_str(&rest[..semi + 1]);
        }
        rest = &rest[semi + 1..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn folds() {
        let f = |s: &str| fold_expressions(s);
        assert_eq!(f("mov.u64 %d0, 1 + 2 * 3;"), "mov.u64 %d0, 7;");
        assert_eq!(f("mov.u64 %d2, - 5;"), "mov.u64 %d2, -5;");
        assert_eq!(f("mov.u64 %d6, -1U >> 1;"), "mov.u64 %d6, 9223372036854775807;");
        assert_eq!(f("mov.u64 %d10, (.u64) -1 >> 60;"), "mov.u64 %d10, 15;");
        assert_eq!(f("st.global.v2.u64 [%o + 72 - 8], {%d8, %d9};"), "st.global.v2.u64 [%o + 64], {%d8, %d9};");
        assert_eq!(f(".reg .b32 %r<4>;"), ".reg .b32 %r<4>;");
        assert_eq!(f("ld.global.u32 %r1, [g + 8];"), "ld.global.u32 %r1, [g + 8];");
        assert_eq!(f("mov.u64 %d8, 1 < 2 ? 5 : 6;"), "mov.u64 %d8, 5;");
        assert_eq!(f("mov.f32 %f1, 1.5 * 2.0;"), "mov.f32 %f1, 3.0;");
        assert_eq!(f("$L1: bra $L1;"), "$L1: bra $L1;");
        assert_eq!(f("call (%r2), f, (1 + 2);"), "call (%r2), f, (3);");
        assert_eq!(f(".global .u32 g[2 * 2] = { 1 + 1, WARP_SZ / 4 };"), ".global .u32 g[4] = { 2, 8 };");
        assert_eq!(f("mov.u32 %r3, 0b1111;"), "mov.u32 %r3, 15;");
    }
}

// ------------------------------------------------------------------------------------------------ array operands

/// array elements as operands (6.4.1, 6.4.2): `ld.global.u32 s, a[8];` -> `ld.global.u32 s, [a+8];`,
/// `mov.u64 p, a[4];` -> `mov.u64 p, a+4;`; a register index `a[r+4]` of a memory operand computes the address in a
/// scoped register: `{ .reg .u64 %__zluda_ai; mov.u64 %__zluda_ai, a; add.u64 %__zluda_ai, %__zluda_ai, r; ld ... [%__zluda_ai+4]; }`
fn array_operands(t: &str) -> String {
    let wide = !t.contains(".address_size 32");
    let at = if wide { "u64" } else { "u32" };
    let op = opaque_ranges(t);
    let b = t.as_bytes();
    let mut out = String::with_capacity(t.len());
    let mut last = 0;
    let mut i = 0;
    // statement boundaries
    while i < b.len() {
        let start = i;
        while i < b.len() && !matches!(b[i], b';' | b'{' | b'}') {
            if let Some(&(_, e)) = op.iter().find(|&&(a, _)| a == i) {
                i = e;
            } else {
                i += 1;
            }
        }
        let end = i;
        if i < b.len() {
            i += 1;
        }
        // the statement starts after the comments that precede it
        let start = op.iter().filter(|&&(a, e)| a >= start && e <= end).map(|&(_, e)| e).max().unwrap_or(start);
        let stmt = &t[start..end];
        if !stmt.contains('[') {
            continue;
        }
        if let Some(repl) = rewrite_array_stmt(stmt, at) {
            out.push_str(&t[last..start]);
            out.push_str(&repl);
            last = end;
            if repl.ends_with('}') && end < b.len() && b[end] == b';' {
                last = end + 1;
            }
        }
    }
    out.push_str(&t[last..]);
    out
}

fn rewrite_array_stmt(stmt: &str, at: &str) -> Option<String> {
    // leading label(s) and predicate
    let lead_len = stmt.len() - stmt.trim_start().len();
    let mut body = &stmt[lead_len..];
    let mut prefix = String::new();
    loop {
        let w = body.split_whitespace().next()?;
        if w.ends_with(':') && !w.contains("::") {
            let p = body.find(':')? + 1;
            prefix.push_str(&body[..p]);
            body = &body[p..];
            let l = body.len() - body.trim_start().len();
            prefix.push_str(&body[..l]);
            body = &body[l..];
            continue;
        }
        break;
    }
    let mut pred = "";
    if body.starts_with('@') {
        let e = body.find(char::is_whitespace)?;
        pred = &body[..e];
        body = body[e..].trim_start();
    }
    let opcode = body.split_whitespace().next()?;
    if opcode.starts_with('.') || !opcode.bytes().next()?.is_ascii_alphabetic() {
        return None;
    }
    let base = opcode.split('.').next()?;
    let memop = matches!(base, "ld" | "st" | "ldu" | "atom" | "red" | "prefetch" | "prefetchu" | "cp" | "mbarrier");
    let bb = body.as_bytes();
    let mut out = String::new();
    let mut k = 0;
    let mut pre: Vec<String> = Vec::new();
    let mut changed = false;
    let mut idx = 0;
    let mut post: Option<String> = None;
    while k < bb.len() {
        // an identifier directly followed by `[` (after optional spaces), not inside `[...]`
        if (bb[k].is_ascii_alphabetic() || bb[k] == b'_' || bb[k] == b'$') && (k == 0 || !is_word(bb[k - 1])) && k >= opcode.len() {
            let s = k;
            while k < bb.len() && (bb[k].is_ascii_alphanumeric() || bb[k] == b'_' || bb[k] == b'$') {
                k += 1;
            }
            let name = &body[s..k];
            let mut j = k;
            while j < bb.len() && bb[j] == b' ' {
                j += 1;
            }
            // not an operand of a bracketed list `[name, ...]` nor a declaration
            let in_brackets = body[..s].matches('[').count() > body[..s].matches(']').count();
            if j < bb.len() && bb[j] == b'[' && !in_brackets {
                let close = body[j..].find(']')? + j;
                let inner = body[j + 1..close].trim();
                if inner.starts_with('%') {
                    // register index (+ / - a constant offset)
                    let split = inner.find(|c| c == '+' || c == '-');
                    let (reg, off) = match split {
                        Some(p) => (inner[..p].trim(), inner[p..].replace(' ', "")),
                        None => (inner, String::new()),
                    };
                    if !memop || !reg[1..].bytes().all(is_word) {
                        return None;
                    }
                    let tmp = format!("%__zluda_ai{idx}");
                    idx += 1;
                    pre.push(format!(".reg .{at} {tmp}; mov.{at} {tmp}, {name}; add.{at} {tmp}, {tmp}, {reg};"));
                    out.push_str(&format!("[{tmp}{}]", if off.is_empty() { String::new() } else { format!("+{}", off.trim_start_matches('+')) }));
                } else {
                    let off = if inner.starts_with('-') { inner.to_string() } else { format!("+{inner}") };
                    if memop {
                        out.push_str(&format!("[{name}{off}]"));
                    } else if base == "mov" {
                        // mov.T d, a[N]: d = a, then d += N
                        let ty = opcode.rsplit('.').next()?;
                        let dst = body[opcode.len()..].split(',').next()?.trim();
                        if !dst.starts_with('%') && !dst.bytes().all(is_word) {
                            return None;
                        }
                        let pfx = if pred.is_empty() { String::new() } else { format!("{pred} ") };
                        post = Some(format!("; {pfx}add.{ty} {dst}, {dst}, {}", inner));
                        out.push_str(name);
                    } else {
                        return None;
                    }
                }
                k = close + 1;
                changed = true;
                continue;
            }
            out.push_str(name);
            continue;
        }
        let ch = body[k..].chars().next()?;
        out.push(ch);
        k += ch.len_utf8();
    }
    if !changed {
        return None;
    }
    let lead = &stmt[..lead_len];
    if let Some(p) = post {
        out.push_str(&p);
    }
    let inst = if pred.is_empty() { out } else { format!("{pred} {out}") };
    if pre.is_empty() {
        Some(format!("{lead}{prefix}{inst}"))
    } else {
        // the statement's `;` follows: close the scope after it
        Some(format!("{lead}{prefix}{{ {} {inst}; }}", pre.join(" ")))
    }
}

// ------------------------------------------------------------------------------------------------ vector members

/// `v.r v.g v.b v.a` of a vector register are `v.x v.y v.z v.w` (6.4.2)
fn colour_members(t: &str) -> String {
    if !t.contains(".v2") && !t.contains(".v4") {
        return t.to_string();
    }
    // names declared `.reg .v2 / .v4 ...`
    let mut names: Vec<String> = Vec::new();
    for st in t.split(|c| c == ';' || c == '{' || c == '}' || c == '(' || c == ')') {
        for part in st.split(',') {
            let ws: Vec<&str> = part.split_whitespace().collect();
            if let Some(p) = ws.iter().position(|w| *w == ".v2" || *w == ".v4") {
                if ws.contains(&".reg") || p > 0 || st.contains(".reg") {
                    if let Some(n) = ws.last() {
                        let n: String = n.bytes().take_while(|c| is_word(*c) && *c != b'.' && *c != b'<').map(|c| c as char).collect();
                        if !n.is_empty() && !n.starts_with('.') {
                            names.push(n);
                        }
                    }
                }
            }
        }
        // further names of the same declaration: `.reg .v2 .u32 v, w;`
        let ws: Vec<&str> = st.split_whitespace().collect();
        if ws.first() == Some(&".reg") && (ws.contains(&".v2") || ws.contains(&".v4")) {
            for part in st.split(',').skip(1) {
                let n: String = part.trim().bytes().take_while(|c| is_word(*c) && *c != b'.' && *c != b'<').map(|c| c as char).collect();
                if !n.is_empty() {
                    names.push(n);
                }
            }
        }
    }
    if names.is_empty() {
        return t.to_string();
    }
    let b = t.as_bytes();
    let mut out = String::with_capacity(t.len());
    let mut i = 0;
    while i < b.len() {
        if (b[i].is_ascii_alphabetic() || b[i] == b'%' || b[i] == b'_') && (i == 0 || !is_word(b[i - 1])) {
            let s = i;
            while i < b.len() && is_word(b[i]) && b[i] != b'.' {
                i += 1;
            }
            let name = &t[s..i];
            out.push_str(name);
            if i + 2 <= b.len() && b[i] == b'.' && names.iter().any(|n| n == name) {
                let m = b.get(i + 1).copied();
                let after = b.get(i + 2).copied().map_or(true, |c| !is_word(c));
                if let (Some(m), true) = (m, after) {
                    let x = match m {
                        b'r' => Some('x'),
                        b'g' => Some('y'),
                        b'b' => Some('z'),
                        b'a' => Some('w'),
                        _ => None,
                    };
                    if let Some(x) = x {
                        out.push('.');
                        out.push(x);
                        i += 2;
                    }
                }
            }
            continue;
        }
        let c = t[i..].chars().next().unwrap();
        out.push(c);
        i += c.len_utf8();
    }
    out
}

/// a vector operand `{a, v.z, b}` holding members of vector registers: the members go through scoped scalar registers
/// (moved before a source list, after a destination list)
fn vector_members_in_lists(t: &str) -> String {
    if !t.contains(".v2") && !t.contains(".v4") {
        return t.to_string();
    }
    // element types of the vector registers
    let mut ty: Vec<(String, String)> = Vec::new();
    for st in t.split(|c| c == ';' || c == '{' || c == '}') {
        let ws: Vec<&str> = st.split_whitespace().collect();
        if ws.first() == Some(&".reg") {
            if let Some(p) = ws.iter().position(|w| *w == ".v2" || *w == ".v4") {
                if let Some(e) = ws.get(p + 1) {
                    let decl = st.splitn(2, e).nth(1).unwrap_or("");
                    for n in decl.split(',') {
                        let n: String = n.trim().bytes().take_while(|c| is_word(*c) && *c != b'.' && *c != b'<').map(|c| c as char).collect();
                        if !n.is_empty() {
                            ty.push((n, e.to_string()));
                        }
                    }
                }
            }
        }
    }
    if ty.is_empty() {
        return t.to_string();
    }
    let mut out = String::with_capacity(t.len());
    let mut ctr = 0;
    for st in t.split_inclusive(';') {
        let body = &st[..st.len() - if st.ends_with(';') { 1 } else { 0 }];
        let (Some(lb), Some(rb)) = (body.find('{'), body.rfind('}')) else {
            out.push_str(st);
            continue;
        };
        if rb <= lb {
            out.push_str(st);
            continue;
        }
        // a statement start: the text after the last block brace or label before the instruction
        let head = &body[..lb];
        let inst_start = head.rfind(|c| c == '\n').map_or(0, |p| p + 1);
        let inst = &body[inst_start..];
        let mut ws_it = inst.split_whitespace();
        let mut opcode = ws_it.next().unwrap_or("");
        let guard = if opcode.starts_with('@') {
            let g = opcode;
            opcode = ws_it.next().unwrap_or("");
            format!("{g} ")
        } else {
            String::new()
        };
        if opcode.starts_with('.') || lb < inst_start || !st.ends_with(';') || body[lb + 1..rb].contains('{') {
            out.push_str(st);
            continue;
        }
        let items: Vec<String> = body[lb + 1..rb].split(',').map(|x| x.trim().to_string()).collect();
        let mut new_items = Vec::new();
        let mut decls = Vec::new();
        let mut moves = Vec::new();
        let dst = !opcode.starts_with("st") && body[inst_start..lb].trim_end().ends_with(opcode.trim());
        for it in &items {
            let mut done = false;
            if let Some(d) = it.rfind('.') {
                let (n, m) = (&it[..d], &it[d + 1..]);
                if matches!(m, "x" | "y" | "z" | "w") {
                    if let Some((_, e)) = ty.iter().find(|(x, _)| x == n) {
                        let tmp = format!("%__zluda_vm{ctr}");
                        ctr += 1;
                        decls.push(format!(".reg {e} {tmp};"));
                        let bt = format!(".b{}", e.trim_start_matches(|c: char| !c.is_ascii_digit()));
                        if dst {
                            moves.push(format!("{guard}mov{bt} {it}, {tmp};"));
                        } else {
                            moves.push(format!("mov{bt} {tmp}, {it};"));
                        }
                        new_items.push(tmp);
                        done = true;
                    }
                }
            }
            if !done {
                new_items.push(it.clone());
            }
        }
        if decls.is_empty() {
            out.push_str(st);
            continue;
        }
        let inst_new = format!("{}{{{}}}{}", &body[inst_start..lb], new_items.join(", "), &body[rb + 1..]);
        out.push_str(&body[..inst_start]);
        if dst {
            out.push_str(&format!("{{ {} {}; {} }}", decls.join(" "), inst_new.trim(), moves.join(" ")));
        } else {
            out.push_str(&format!("{{ {} {} {}; }}", decls.join(" "), moves.join(" "), inst_new.trim()));
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------ per-CTA memory

/// CTAs of a launch that `.minperctamemory` provides slices for
const PCM_CTAS: u64 = 256;

/// `.minperctamemory N` (PTX ISA 9.4 11.4.10, 10.34-10.35): every CTA gets a slice of %perctamemorysize bytes (N rounded
/// up to a multiple of 1024) at %perctamemoryoffset, the slices in CTA order: a module variable of PCM_CTAS slices per
/// kernel, %perctamemorysize its constant, `mov.u64 d, %perctamemoryoffset` the address of the CTA's slice
fn per_cta_memory(t: &str) -> String {
    let Some(mut p) = t.find(".minperctamemory") else { return t.to_string() };
    let mut out = t.to_string();
    let mut n = 0;
    loop {
        // the kernel: the last `.entry NAME` before the directive, its body the next `{ ... }`
        let Some(ep) = out[..p].rfind(".entry") else { break };
        let name: String = out[ep + 6..].trim_start().bytes().take_while(|c| is_word(*c) && *c != b'.').map(|c| c as char).collect();
        let num_s = out[p + ".minperctamemory".len()..].trim_start();
        let digits: String = num_s.bytes().take_while(|c| c.is_ascii_alphanumeric()).map(|c| c as char).collect();
        let Some(v) = parse_num(&digits).and_then(|v| if let Val::I(x, _) = v { Some(x) } else { None }) else { break };
        let size = v.max(1).div_ceil(1024) * 1024;
        let dir_end = p + ".minperctamemory".len() + (out[p + ".minperctamemory".len()..].len() - num_s.len()) + digits.len();
        out.replace_range(p..dir_end, "");
        let line_start = out[..ep].rfind('\n').map_or(0, |x| x + 1);
        let var = format!("__zluda_pcm_{name}");
        let decl = format!(".global .align 1024 .b8 {var}[{}];\n", size * PCM_CTAS);
        out.insert_str(line_start, &decl);
        let body_start = out[line_start + decl.len()..].find('{').map(|x| x + line_start + decl.len());
        if let Some(bs) = body_start {
            let be = block_end(&out, bs);
            let body = out[bs..be].to_string();
            let mut nb = String::new();
            for st in body.split_inclusive(';') {
                let trimmed = st.trim_start_matches(|c: char| c != 'm' && c.is_whitespace());
                let movp = st.find("mov.");
                if let (Some(mp), true) = (movp, st.contains("%perctamemoryoffset")) {
                    let ins = &st[mp..];
                    let parts: Vec<&str> = ins.trim_end_matches(';').splitn(2, char::is_whitespace).collect();
                    if parts.len() == 2 && parts[0].ends_with("64") {
                        let ops: Vec<&str> = parts[1].split(',').map(|x| x.trim()).collect();
                        if ops.len() == 2 && ops[1] == "%perctamemoryoffset" {
                            let d = ops[0];
                            let a = format!("%__zluda_pcm_a{n}");
                            let b = format!("%__zluda_pcm_b{n}");
                            n += 1;
                            nb.push_str(&st[..mp]);
                            nb.push_str(&format!(
                                "{{ .reg .u32 {a}, {b}; mov.u32 {a}, %ctaid.z; mov.u32 {b}, %nctaid.y; mul.lo.u32 {a}, {a}, {b}; \
                                 mov.u32 {b}, %ctaid.y; add.u32 {a}, {a}, {b}; mov.u32 {b}, %nctaid.x; mul.lo.u32 {a}, {a}, {b}; \
                                 mov.u32 {b}, %ctaid.x; add.u32 {a}, {a}, {b}; mov.u64 {d}, {var}; mad.wide.u32 {d}, {a}, {size}, {d}; }}"
                            ));
                            continue;
                        }
                    }
                }
                let _ = trimmed;
                nb.push_str(&st.replace("%perctamemorysize", &size.to_string()));
            }
            // the `;` after a rewritten statement was consumed with it
            out.replace_range(bs..be, &nb);
        }
        match out.find(".minperctamemory") {
            Some(q) => p = q,
            None => break,
        }
    }
    out
}
