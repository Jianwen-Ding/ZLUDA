use super::*;
use ptx_parser as ast;

pub(crate) fn run<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    directives: Vec<ast::Directive<'input, ast::ParsedOperand<&'input str>>>,
) -> Result<Vec<NormalizedDirective2>, TranslateError> {
    resolver.start_scope();
    let result = directives
        .into_iter()
        .map(|directive| run_directive(resolver, directive))
        .collect::<Result<Vec<_>, _>>()?;
    resolver.end_scope();
    Ok(result)
}

fn run_directive<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    directive: ast::Directive<'input, ast::ParsedOperand<&'input str>>,
) -> Result<NormalizedDirective2, TranslateError> {
    Ok(match directive {
        ast::Directive::Variable(linking, var) => {
            NormalizedDirective2::Variable(linking, run_variable(resolver, var)?)
        }
        ast::Directive::Method(linking, directive) => {
            NormalizedDirective2::Method(run_method(resolver, linking, directive)?)
        }
    })
}

fn run_method<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    linkage: ast::LinkingDirective,
    method: ast::Function<'input, &'input str, ast::Statement<ast::ParsedOperand<&'input str>>>,
) -> Result<NormalizedFunction2, TranslateError> {
    let is_kernel = method.func_directive.name.is_kernel();
    let name = resolver.add_or_get_in_current_scope_untyped(method.func_directive.name.text())?;
    resolver.start_scope();
    let (return_arguments, input_arguments) = run_function_decl(resolver, method.func_directive)?;
    let body = method
        .body
        .map(|statements| {
            let mut result = Vec::with_capacity(statements.len());
            run_statements(resolver, &mut result, statements)?;
            Ok::<_, TranslateError>(result)
        })
        .transpose()?;
    resolver.end_scope();
    Ok(Function {
        return_arguments,
        name,
        input_arguments,
        body,
        import_as: None,
        linkage,
        kernel_attributes: is_kernel.then(|| KernelAttributes {
            flush_to_zero_f32: false,
            flush_to_zero_f16f64: false,
            rounding_mode_f32: ptx_parser::RoundingMode::NearestEven,
            rounding_mode_f16f64: ptx_parser::RoundingMode::NearestEven,
        }),
        tuning: method.tuning,
        kernel_meta32: None,
    })
}

fn run_function_decl<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    func_directive: ast::MethodDeclaration<'input, &'input str>,
) -> Result<(Vec<ast::Variable<SpirvWord>>, Vec<ast::Variable<SpirvWord>>), TranslateError> {
    assert!(func_directive.shared_mem.is_none());
    let return_arguments = func_directive
        .return_arguments
        .into_iter()
        .map(|var| run_variable(resolver, var))
        .collect::<Result<Vec<_>, _>>()?;
    let input_arguments = func_directive
        .input_arguments
        .into_iter()
        .map(|var| run_variable(resolver, var))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((return_arguments, input_arguments))
}

fn run_variable_info<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    info: ast::VariableInfo<&'input str>,
) -> Result<ast::VariableInfo<SpirvWord>, TranslateError> {
    Ok(ast::VariableInfo {
        align: info.align,
        v_type: info.v_type,
        state_space: info.state_space,
        array_init: run_array_init(resolver, &info.array_init)?,
    })
}

fn run_variable<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    variable: ast::Variable<&'input str>,
) -> Result<ast::Variable<SpirvWord>, TranslateError> {
    Ok(ast::Variable {
        info: run_variable_info(resolver, variable.info.clone())?,
        name: resolver.add(
            Cow::Borrowed(variable.name),
            Some((variable.info.v_type.clone(), variable.info.state_space)),
        )?,
    })
}

fn run_statements<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    result: &mut Vec<NormalizedStatement>,
    statements: Vec<ast::Statement<ast::ParsedOperand<&'input str>>>,
) -> Result<(), TranslateError> {
    for statement in statements.iter() {
        match statement {
            ast::Statement::Label(label) => {
                resolver.add(Cow::Borrowed(*label), None)?;
            }
            _ => {}
        }
    }
    for statement in statements {
        match statement {
            ast::Statement::Label(label) => {
                result.push(Statement::Label(resolver.get_in_current_scope(label)?))
            }
            ast::Statement::Variable(variable) => run_multivariable(resolver, result, variable)?,
            ast::Statement::Instruction(predicate, instruction) => {
                result.push(Statement::Instruction((
                    predicate
                        .map(|pred| {
                            Ok::<_, TranslateError>(ast::PredAt {
                                not: pred.not,
                                label: resolver.get(pred.label)?,
                            })
                        })
                        .transpose()?,
                    run_instruction(resolver, instruction)?,
                )))
            }
            ast::Statement::Block(block) => {
                resolver.start_scope();
                run_statements(resolver, result, block)?;
                resolver.end_scope();
            }
        }
    }
    Ok(())
}

fn run_instruction<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    instruction: ast::Instruction<ast::ParsedOperand<&'input str>>,
) -> Result<ast::Instruction<ast::ParsedOperand<SpirvWord>>, TranslateError> {
    ast::visit_map(instruction, &mut ResolveOperands(resolver))
}

/// Resolves operand names. `%tid.x` where a scalar register named `%tid` shadows the special
/// register (PTX-formalization's tests declare `.reg .b32 %tid`) names the special register.
struct ResolveOperands<'r, 'input, 'b>(&'r mut ScopedResolver<'input, 'b>);

impl<'r, 'input, 'b> ResolveOperands<'r, 'input, 'b> {
    fn shadowed_sreg(&self, name: &str) -> Option<SpirvWord> {
        if self.0.scopes.len() < 2 {
            return None;
        }
        let inner = self
            .0
            .scopes
            .iter()
            .rev()
            .find_map(|scope| scope.name_to_ident.get(name).map(|id| (scope, *id)))?;
        let outer = *self.0.scopes[0].name_to_ident.get(name)?;
        if inner.1 == outer {
            return None;
        }
        let inner_type = &inner.0.ident_map.get(&inner.1)?.type_space;
        let outer_type = &self.0.scopes[0].ident_map.get(&outer)?.type_space;
        match (inner_type, outer_type) {
            (Some((ast::Type::Scalar(_), _)), Some((ast::Type::Vector(..), _))) => Some(outer),
            _ => None,
        }
    }
}

impl<'r, 'input, 'b> ast::VisitorMap<ast::ParsedOperand<&'input str>, ast::ParsedOperand<SpirvWord>, TranslateError>
    for ResolveOperands<'r, 'input, 'b>
{
    fn visit(
        &mut self,
        args: ast::ParsedOperand<&'input str>,
        type_space: Option<(&ast::Type, ast::StateSpace)>,
        is_dst: bool,
        relaxed_type_check: bool,
    ) -> Result<ast::ParsedOperand<SpirvWord>, TranslateError> {
        if let ast::ParsedOperand::VecMember(name, index) = args {
            if let Some(sreg) = self.shadowed_sreg(name) {
                return Ok(ast::ParsedOperand::VecMember(sreg, index));
            }
        }
        let resolver = &mut *self.0;
        let mut f = |name: &'input str, _: Option<(&ast::Type, ast::StateSpace)>, _, _| resolver.get(name);
        ast::VisitorMap::visit(&mut f, args, type_space, is_dst, relaxed_type_check)
    }

    fn visit_ident(
        &mut self,
        args: &'input str,
        _type_space: Option<(&ast::Type, ast::StateSpace)>,
        _is_dst: bool,
        _relaxed_type_check: bool,
    ) -> Result<SpirvWord, TranslateError> {
        self.0.get(args)
    }
}

fn run_multivariable<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    result: &mut Vec<NormalizedStatement>,
    variable: ast::MultiVariable<&'input str>,
) -> Result<(), TranslateError> {
    match variable {
        ptx_parser::MultiVariable::List(items) => {
            for item in items {
                run_multivariable(resolver, result, item)?;
            }
        }
        ptx_parser::MultiVariable::Parameterized { info, name, count } => {
            for i in 0..count {
                let name = Cow::Owned(format!("{}{}", name, i));
                let ident = resolver.add(name, Some((info.v_type.clone(), info.state_space)))?;
                result.push(Statement::Variable(ast::Variable {
                    info: run_variable_info(resolver, info.clone())?,
                    name: ident,
                }));
            }
        }
        ptx_parser::MultiVariable::Names { info, names } => {
            for name in names {
                let name = Cow::Borrowed(name);
                let ident = resolver.add(name, Some((info.v_type.clone(), info.state_space)))?;
                result.push(Statement::Variable(ast::Variable {
                    info: run_variable_info(resolver, info.clone())?,
                    name: ident,
                }));
            }
        }
    }
    Ok(())
}

fn run_array_init<'input, 'b>(
    resolver: &mut ScopedResolver<'input, 'b>,
    array_init: &[ast::RegOrImmediate<&'input str>],
) -> Result<Vec<ast::RegOrImmediate<SpirvWord>>, TranslateError> {
    Ok(array_init
        .iter()
        .map(|elem| match elem {
            ast::RegOrImmediate::Reg(name) => Ok(ast::RegOrImmediate::Reg(resolver.get(name)?)),
            ast::RegOrImmediate::Imm(imm) => Ok(ast::RegOrImmediate::Imm(*imm)),
            ast::RegOrImmediate::SymOffset(name, off, mask) => {
                Ok(ast::RegOrImmediate::SymOffset(resolver.get(name)?, *off, *mask))
            }
        })
        .collect::<Result<Vec<_>, _>>()?)
}
