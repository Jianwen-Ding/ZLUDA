//! Fallback parser for instructions that the `derive_parser!` grammar does not cover.
//!
//! A statement that no grammar rule accepts is split into its opcode text (the contiguous
//! characters at its start, `vadd.u32.u32.s32.sat`, `mbarrier.init.shared::cta.b64`) and its
//! operands, and `lower` maps it to an `Instruction::Helper`: a call of
//! `__zluda_ptx_impl_<name>` (ptx/lib/zluda_ptx_metal_impl.cpp) with typed operands.
//! This keeps every new instruction a table entry plus a helper function.
use super::*;

/// One operand of a fallback instruction.
#[derive(Clone)]
pub(crate) struct HOp<'i> {
    pub op: ast::ParsedOperand<&'i str>,
    /// `[...]`
    pub addr: bool,
    /// `!p`
    pub neg: bool,
    /// a video byte / half selector or other suffix: `a.b0`, `a.h1`, `a.b3210`
    pub sel: Option<&'i str>,
    /// the second register of `d|p`
    pub pair: Option<ast::ParsedOperand<&'i str>>,
    /// `_` (a sink destination)
    pub sink: bool,
    /// `-x` on a register
    pub minus: bool,
}

fn statement_end(toks: &[(Token, logos::Span)]) -> Option<usize> {
    let mut depth = 0i32;
    for (i, (t, _)) in toks.iter().enumerate() {
        match t {
            Token::LBrace => depth += 1,
            Token::RBrace => {
                depth -= 1;
                if depth < 0 {
                    return None;
                }
            }
            Token::Semicolon if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

fn sub_operand<'a, 'input>(
    text: &'input str,
    toks: &'a [(Token<'input>, logos::Span)],
) -> Option<ast::ParsedOperand<&'input str>> {
    let mut errors = Vec::new();
    let mut s = PtxParser {
        input: toks,
        state: PtxParserState::new(text, &mut errors),
    };
    // `-x` negative immediates
    if let [(Token::Minus, _), ..] = toks {
        let v = ast::ParsedOperand::<&str>::parse(&mut PtxParser {
            input: &toks[1..],
            state: PtxParserState::new(text, &mut Vec::new()),
        })
        .ok()?;
        return match v {
            ast::ParsedOperand::Imm(ast::ImmediateValue::U64(x)) => {
                Some(ast::ParsedOperand::Imm(ast::ImmediateValue::S64(-(x as i64))))
            }
            ast::ParsedOperand::Imm(ast::ImmediateValue::S64(x)) => {
                Some(ast::ParsedOperand::Imm(ast::ImmediateValue::S64(-x)))
            }
            ast::ParsedOperand::Imm(ast::ImmediateValue::F32(x)) => {
                Some(ast::ParsedOperand::Imm(ast::ImmediateValue::F32(-x)))
            }
            ast::ParsedOperand::Imm(ast::ImmediateValue::F64(x)) => {
                Some(ast::ParsedOperand::Imm(ast::ImmediateValue::F64(-x)))
            }
            _ => None,
        };
    }
    let r = ast::ParsedOperand::<&str>::parse(&mut s).ok()?;
    if !s.input.is_empty() || !errors.is_empty() {
        return None;
    }
    Some(r)
}

fn parse_hop<'a, 'input>(
    text: &'input str,
    mut toks: &'a [(Token<'input>, logos::Span)],
) -> Option<HOp<'input>> {
    let mut h = HOp {
        op: ast::ParsedOperand::Imm(ast::ImmediateValue::U64(0)),
        addr: false,
        neg: false,
        sel: None,
        pair: None,
        sink: false,
        minus: false,
    };
    if let [(Token::Sink, _)] = toks {
        h.sink = true;
        return Some(h);
    }
    if let [(Token::Minus, _), rest @ ..] = toks {
        if let [(Token::Ident(_), _), ..] = rest {
            h.minus = true;
            toks = rest;
        }
    }
    if let [(Token::Exclamation, _), rest @ ..] = toks {
        h.neg = true;
        toks = rest;
    }
    if let [(Token::LBracket, _), inner @ .., (Token::RBracket, _)] = toks {
        h.addr = true;
        h.op = sub_operand(text, inner)?;
        return Some(h);
    }
    if let Some(p) = toks.iter().position(|(t, _)| matches!(t, Token::Pipe)) {
        if let [(Token::Sink, _)] = &toks[..p] {
            h.sink = true;
        } else {
            h.op = sub_operand(text, &toks[..p])?;
        }
        h.pair = Some(sub_operand(text, &toks[p + 1..])?);
        return Some(h);
    }
    if let [(Token::Ident(_), _), (Token::DotIdent(sel), _)] = toks {
        if !matches!(*sel, ".x" | ".y" | ".z" | ".w" | ".r" | ".g" | ".b" | ".a") {
            h.sel = Some(sel);
            h.op = sub_operand(text, &toks[..1])?;
            return Some(h);
        }
    }
    h.op = sub_operand(text, toks)?;
    Some(h)
}

/// Parses one fallback instruction (without its `;`) and lowers it.
pub(crate) fn helper_instruction<'a, 'input>(
    stream: &mut PtxParser<'a, 'input>,
) -> PResult<ast::Instruction<ParsedOperandStr<'input>>> {
    let text = stream.state.text;
    let toks = stream.input;
    let end = statement_end(toks).ok_or_else(|| ErrMode::from_error_kind(stream, ErrorKind::Verify))?;
    if end == 0 {
        return Err(ErrMode::from_error_kind(stream, ErrorKind::Verify));
    }
    // the opcode: tokens without whitespace between them
    let mut k = 1;
    while k < end && toks[k].1.start == toks[k - 1].1.end {
        k += 1;
    }
    let opcode = &text[toks[0].1.start..toks[k - 1].1.end];
    if !opcode.chars().next().map_or(false, |c| c.is_ascii_alphabetic()) {
        return Err(ErrMode::from_error_kind(stream, ErrorKind::Verify));
    }
    let mut ops = Vec::new();
    let mut depth = 0i32;
    let mut start = k;
    for i in k..=end {
        let t = &toks[i].0;
        match t {
            Token::LBrace | Token::LBracket => depth += 1,
            Token::RBrace | Token::RBracket => depth -= 1,
            _ => {}
        }
        if (i == end || (matches!(t, Token::Comma) && depth == 0)) && i > start {
            match parse_hop(text, &toks[start..i]) {
                Some(h) => ops.push(h),
                None => return Err(ErrMode::from_error_kind(stream, ErrorKind::Verify)),
            }
            start = i + 1;
        }
    }
    match crate::helper_table::lower(opcode, ops) {
        Some(Ok(inst)) => {
            stream.input = &toks[end..];
            Ok(inst)
        }
        Some(Err(e)) => {
            stream.input = &toks[end..];
            stream.state.errors.push(PtxError::Todo(format!("{opcode}: {e}")));
            Ok(ast::Instruction::Helper {
                data: ast::HelperData::new("unsupported"),
                arguments: ast::HelperArgs {
                    dst: None,
                    dst2: None,
                    src1: None,
                    src2: None,
                    src3: None,
                    src4: None,
                    src5: None,
                    src6: None,
                },
            })
        }
        None => Err(ErrMode::from_error_kind(stream, ErrorKind::Verify)),
    }
}
