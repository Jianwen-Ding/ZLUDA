use ptx_parser as ast;
use quick_error::quick_error;
use rustc_hash::FxHashMap;
use std::hash::Hash;
use std::{
    borrow::Cow,
    collections::{hash_map, HashMap},
    ffi::CString,
};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

mod convert_32bit_to_64bit;
mod deparamize_functions;
mod expand_operands;
mod fix_special_registers;
mod hoist_globals;
mod insert_explicit_load_store;
mod insert_implicit_conversions;
mod insert_post_saturation;
mod instruction_mode_to_global_mode;
pub mod llvm;
mod normalize_basic_blocks;
mod normalize_identifiers;
mod normalize_predicates;
mod optimize_function_arguments;
mod rcp_f64_into_div;
mod remove_unreachable_basic_blocks;
mod replace_instructions_with_functions;
mod replace_instructions_with_functions_fp_required;
mod replace_known_functions;
mod resolve_function_pointers;

#[cfg(test)]
mod test;

static ZLUDA_PTX_IMPL: &'static [u8] = include_bytes!("../../lib/zluda_ptx_impl.bc");
static ZLUDA_PTX_IMPL_CONSTRAINED: &'static [u8] =
    include_bytes!("../../lib/zluda_ptx_impl_constrained.bc");
const ZLUDA_PTX_PREFIX: &'static str = "__zluda_ptx_impl_";

quick_error! {
    #[derive(Debug, strum_macros::AsRefStr)]
    pub enum TranslateError {
        UnknownSymbol(symbol: String) {
            display("Unknown symbol: \"{}\"", symbol)
        }
        UntypedSymbol {}
        MismatchedType {}
        Unreachable {}
        Todo(msg: String) {
            display("TODO: {}", msg)
        }
    }
}

/// GPU attributes needed at compile time.
#[derive(Copy, Clone)]
pub struct Attributes {
    /// Clock frequency in kHz.
    pub clock_rate: u32,
}

pub fn to_llvm_module<'input>(
    ast: ast::Module<'input>,
    attributes: Attributes,
    mut on_pass_end: impl FnMut(&str),
) -> Result<Module, TranslateError> {
    let sm_version = ast.sm_version;
    let mut flat_resolver = GlobalStringIdentResolver2::<'input>::new(SpirvWord(1));
    let mut scoped_resolver = ScopedResolver::new(&mut flat_resolver);
    let sreg_map = SpecialRegistersMap::new(&mut scoped_resolver)?;
    let directives = normalize_identifiers::run(&mut scoped_resolver, ast.directives)?;
    on_pass_end("normalize_identifiers");
    let directives = replace_known_functions::run(&mut flat_resolver, directives);
    on_pass_end("replace_known_functions");
    let directives = normalize_predicates::run(&mut flat_resolver, directives)?;
    on_pass_end("normalize_predicates");
    let directives = optimize_function_arguments::run(&mut flat_resolver, directives)?;
    on_pass_end("optimize_function_arguments");
    let directives = resolve_function_pointers::run(directives)?;
    on_pass_end("resolve_function_pointers");
    let directives = fix_special_registers::run(&mut flat_resolver, &sreg_map, directives)?;
    on_pass_end("fix_special_registers");
    let directives = expand_operands::run(&mut flat_resolver, directives)?;
    on_pass_end("expand_operands");
    let directives = insert_post_saturation::run(&mut flat_resolver, directives)?;
    on_pass_end("insert_post_saturation");
    let directives = deparamize_functions::run(&mut flat_resolver, directives)?;
    on_pass_end("deparamize_functions");
    let directives = rcp_f64_into_div::run(&mut flat_resolver, directives)?;
    on_pass_end("rcp_f64_into_div");
    let directives =
        replace_instructions_with_functions_fp_required::run(&mut flat_resolver, directives)?;
    on_pass_end("replace_instructions_with_functions_fp_required");
    let directives = normalize_basic_blocks::run(&mut flat_resolver, directives)?;
    on_pass_end("normalize_basic_blocks");
    let directives = remove_unreachable_basic_blocks::run(directives)?;
    on_pass_end("remove_unreachable_basic_blocks");
    let directives = instruction_mode_to_global_mode::run(&mut flat_resolver, directives)?;
    on_pass_end("instruction_mode_to_global_mode");
    let mut directives = insert_explicit_load_store::run(&mut flat_resolver, directives)?;
    on_pass_end("insert_explicit_load_store");
    let mut metadata32 = None;
    if ast.address_size == 32 {
        let (new_directives, new_module32) =
            convert_32bit_to_64bit::run(&mut flat_resolver, directives)?;
        directives = new_directives;
        metadata32 = Some(new_module32);
        on_pass_end("convert_32bit_to_64bit");
    }
    let directives =
        insert_implicit_conversions::run(ast.address_size == 32, &mut flat_resolver, directives)?;
    on_pass_end("insert_implicit_conversions");
    let directives = replace_instructions_with_functions::run(&mut flat_resolver, directives)?;
    on_pass_end("replace_instructions_with_functions");
    let directives = hoist_globals::run(directives)?;
    on_pass_end("hoist_globals");
    let fp_mode = get_fp_mode(&directives[..]);
    on_pass_end("get_fp_mode");
    let context = llvm_zluda::utils::Context::new();
    let llvm_ir = llvm::emit::run(&context, flat_resolver, directives, fp_mode)?;
    let attributes_ir = llvm::attributes::run(&context, attributes)?;
    on_pass_end("emit_llvm");
    Ok(Module {
        llvm_ir,
        attributes_ir,
        context,
        metadata: kernel_metadata::ModuleMetadataV1::new(sm_version),
        metadata32,
        constrained_fp: fp_mode == llvm::emit::FloatingPointMode::Constrained,
    })
}

fn get_fp_mode(
    directives: &[Directive2<ast::Instruction<SpirvWord>, SpirvWord>],
) -> llvm::emit::FloatingPointMode {
    for d in directives {
        if let Directive2::Method(method) = d {
            for s in method.body.iter().flatten() {
                if let Statement::SetMode(_) = s {
                    return llvm::emit::FloatingPointMode::Constrained;
                }
            }
        }
    }
    llvm::emit::FloatingPointMode::Normal
}

pub struct Module {
    pub llvm_ir: llvm_zluda::utils::Module,
    pub attributes_ir: llvm_zluda::utils::Module,
    pub context: llvm_zluda::utils::Context,
    pub metadata: kernel_metadata::ModuleMetadataV1,
    pub metadata32: Option<kernel_metadata::ModuleMetadata32Bit>,
    pub constrained_fp: bool,
}

impl Module {
    pub fn linked_bitcode(&self) -> &'static [u8] {
        if self.constrained_fp {
            ZLUDA_PTX_IMPL_CONSTRAINED
        } else {
            ZLUDA_PTX_IMPL
        }
    }
}

#[derive(Ord, PartialOrd, Eq, PartialEq, Hash, Copy, Clone, EnumIter)]
enum PtxSpecialRegister {
    Tid,
    Ntid,
    Ctaid,
    Nctaid,
    Clock,
    LanemaskLe,
    LanemaskLt,
    LanemaskGe,
    Laneid,
    ClusterCtarank,
    ClusterNctarank,
    Clusterid,
    Nclusterid,
    ClusterCtaid,
    ClusterNctaid,
    IsExplicitCluster,
    LanemaskEq,
    LanemaskGt,
    Clock64,
    Globaltimer,
    GlobaltimerLo,
    GlobaltimerHi,
    Warpid,
    Nwarpid,
    Warpsize,
    Smid,
    Nsmid,
    Gridid,
    DynamicSmemSize,
    TotalSmemSize,
    AggrSmemSize,
    ReservedSmemOffsetBegin,
    ReservedSmemOffsetEnd,
    ReservedSmemOffsetCap,
    ReservedSmemOffset0,
    ReservedSmemOffset1,
    Perctamemoryoffset,
    Perctamemorysize,
    CurrentGraphExec,
    Envreg0,
    Envreg1,
    Envreg2,
    Envreg3,
    Envreg4,
    Envreg5,
    Envreg6,
    Envreg7,
    Envreg8,
    Envreg9,
    Envreg10,
    Envreg11,
    Envreg12,
    Envreg13,
    Envreg14,
    Envreg15,
    Envreg16,
    Envreg17,
    Envreg18,
    Envreg19,
    Envreg20,
    Envreg21,
    Envreg22,
    Envreg23,
    Envreg24,
    Envreg25,
    Envreg26,
    Envreg27,
    Envreg28,
    Envreg29,
    Envreg30,
    Envreg31,
    Pm0,
    Pm1,
    Pm2,
    Pm3,
    Pm4,
    Pm5,
    Pm6,
    Pm7,
    Pm0_64,
    Pm1_64,
    Pm2_64,
    Pm3_64,
    Pm4_64,
    Pm5_64,
    Pm6_64,
    Pm7_64,
}

impl PtxSpecialRegister {
    fn as_str(self) -> &'static str {
        match self {
            Self::Tid => "%tid",
            Self::Ntid => "%ntid",
            Self::Ctaid => "%ctaid",
            Self::Nctaid => "%nctaid",
            Self::Clock => "%clock",
            Self::LanemaskLe => "%lanemask_le",
            Self::LanemaskLt => "%lanemask_lt",
            Self::LanemaskGe => "%lanemask_ge",
            Self::Laneid => "%laneid",
            Self::ClusterCtarank => "%cluster_ctarank",
            Self::ClusterNctarank => "%cluster_nctarank",
            Self::Clusterid => "%clusterid",
            Self::Nclusterid => "%nclusterid",
            Self::ClusterCtaid => "%cluster_ctaid",
            Self::ClusterNctaid => "%cluster_nctaid",
            Self::IsExplicitCluster => "%is_explicit_cluster",
            Self::LanemaskEq => "%lanemask_eq",
            Self::LanemaskGt => "%lanemask_gt",
            Self::Clock64 => "%clock64",
            Self::Globaltimer => "%globaltimer",
            Self::GlobaltimerLo => "%globaltimer_lo",
            Self::GlobaltimerHi => "%globaltimer_hi",
            Self::Warpid => "%warpid",
            Self::Nwarpid => "%nwarpid",
            Self::Warpsize => "%warpsize",
            Self::Smid => "%smid",
            Self::Nsmid => "%nsmid",
            Self::Gridid => "%gridid",
            Self::DynamicSmemSize => "%dynamic_smem_size",
            Self::TotalSmemSize => "%total_smem_size",
            Self::AggrSmemSize => "%aggr_smem_size",
            Self::ReservedSmemOffsetBegin => "%reserved_smem_offset_begin",
            Self::ReservedSmemOffsetEnd => "%reserved_smem_offset_end",
            Self::ReservedSmemOffsetCap => "%reserved_smem_offset_cap",
            Self::ReservedSmemOffset0 => "%reserved_smem_offset_0",
            Self::ReservedSmemOffset1 => "%reserved_smem_offset_1",
            Self::Perctamemoryoffset => "%perctamemoryoffset",
            Self::Perctamemorysize => "%perctamemorysize",
            Self::CurrentGraphExec => "%current_graph_exec",
            Self::Envreg0 => "%envreg0",
            Self::Envreg1 => "%envreg1",
            Self::Envreg2 => "%envreg2",
            Self::Envreg3 => "%envreg3",
            Self::Envreg4 => "%envreg4",
            Self::Envreg5 => "%envreg5",
            Self::Envreg6 => "%envreg6",
            Self::Envreg7 => "%envreg7",
            Self::Envreg8 => "%envreg8",
            Self::Envreg9 => "%envreg9",
            Self::Envreg10 => "%envreg10",
            Self::Envreg11 => "%envreg11",
            Self::Envreg12 => "%envreg12",
            Self::Envreg13 => "%envreg13",
            Self::Envreg14 => "%envreg14",
            Self::Envreg15 => "%envreg15",
            Self::Envreg16 => "%envreg16",
            Self::Envreg17 => "%envreg17",
            Self::Envreg18 => "%envreg18",
            Self::Envreg19 => "%envreg19",
            Self::Envreg20 => "%envreg20",
            Self::Envreg21 => "%envreg21",
            Self::Envreg22 => "%envreg22",
            Self::Envreg23 => "%envreg23",
            Self::Envreg24 => "%envreg24",
            Self::Envreg25 => "%envreg25",
            Self::Envreg26 => "%envreg26",
            Self::Envreg27 => "%envreg27",
            Self::Envreg28 => "%envreg28",
            Self::Envreg29 => "%envreg29",
            Self::Envreg30 => "%envreg30",
            Self::Envreg31 => "%envreg31",
            Self::Pm0 => "%pm0",
            Self::Pm1 => "%pm1",
            Self::Pm2 => "%pm2",
            Self::Pm3 => "%pm3",
            Self::Pm4 => "%pm4",
            Self::Pm5 => "%pm5",
            Self::Pm6 => "%pm6",
            Self::Pm7 => "%pm7",
            Self::Pm0_64 => "%pm0_64",
            Self::Pm1_64 => "%pm1_64",
            Self::Pm2_64 => "%pm2_64",
            Self::Pm3_64 => "%pm3_64",
            Self::Pm4_64 => "%pm4_64",
            Self::Pm5_64 => "%pm5_64",
            Self::Pm6_64 => "%pm6_64",
            Self::Pm7_64 => "%pm7_64",
        }
    }

    fn get_type(self) -> ast::Type {
        match self {
            PtxSpecialRegister::Tid
            | PtxSpecialRegister::Ntid
            | PtxSpecialRegister::Ctaid
            | PtxSpecialRegister::Nctaid
            | PtxSpecialRegister::Clusterid
            | PtxSpecialRegister::Nclusterid
            | PtxSpecialRegister::ClusterCtaid
            | PtxSpecialRegister::ClusterNctaid => ast::Type::Vector(4, self.get_function_return_type()),
            _ => ast::Type::Scalar(self.get_function_return_type()),
        }
    }

    fn get_function_return_type(self) -> ast::ScalarType {
        match self {
            PtxSpecialRegister::Tid => ast::ScalarType::U32,
            PtxSpecialRegister::Ntid => ast::ScalarType::U32,
            PtxSpecialRegister::Ctaid => ast::ScalarType::U32,
            PtxSpecialRegister::Nctaid => ast::ScalarType::U32,
            PtxSpecialRegister::Clock => ast::ScalarType::U32,
            PtxSpecialRegister::LanemaskLe => ast::ScalarType::U32,
            PtxSpecialRegister::LanemaskLt => ast::ScalarType::U32,
            PtxSpecialRegister::LanemaskGe => ast::ScalarType::U32,
            PtxSpecialRegister::Laneid => ast::ScalarType::U32,
            PtxSpecialRegister::IsExplicitCluster => ast::ScalarType::Pred,
            PtxSpecialRegister::LanemaskEq => ast::ScalarType::U32,
            PtxSpecialRegister::LanemaskGt => ast::ScalarType::U32,
            PtxSpecialRegister::Clock64 => ast::ScalarType::U64,
            PtxSpecialRegister::Globaltimer => ast::ScalarType::U64,
            PtxSpecialRegister::GlobaltimerLo => ast::ScalarType::U32,
            PtxSpecialRegister::GlobaltimerHi => ast::ScalarType::U32,
            PtxSpecialRegister::Warpid => ast::ScalarType::U32,
            PtxSpecialRegister::Nwarpid => ast::ScalarType::U32,
            PtxSpecialRegister::Warpsize => ast::ScalarType::U32,
            PtxSpecialRegister::Smid => ast::ScalarType::U32,
            PtxSpecialRegister::Nsmid => ast::ScalarType::U32,
            PtxSpecialRegister::Gridid => ast::ScalarType::U64,
            PtxSpecialRegister::DynamicSmemSize => ast::ScalarType::U32,
            PtxSpecialRegister::TotalSmemSize => ast::ScalarType::U32,
            PtxSpecialRegister::AggrSmemSize => ast::ScalarType::U32,
            PtxSpecialRegister::ReservedSmemOffsetBegin => ast::ScalarType::U32,
            PtxSpecialRegister::ReservedSmemOffsetEnd => ast::ScalarType::U32,
            PtxSpecialRegister::ReservedSmemOffsetCap => ast::ScalarType::U32,
            PtxSpecialRegister::ReservedSmemOffset0 => ast::ScalarType::U32,
            PtxSpecialRegister::ReservedSmemOffset1 => ast::ScalarType::U32,
            PtxSpecialRegister::Perctamemoryoffset => ast::ScalarType::U64,
            PtxSpecialRegister::Perctamemorysize => ast::ScalarType::U32,
            PtxSpecialRegister::CurrentGraphExec => ast::ScalarType::U64,
            PtxSpecialRegister::Envreg0 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg1 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg2 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg3 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg4 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg5 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg6 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg7 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg8 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg9 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg10 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg11 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg12 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg13 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg14 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg15 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg16 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg17 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg18 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg19 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg20 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg21 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg22 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg23 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg24 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg25 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg26 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg27 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg28 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg29 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg30 => ast::ScalarType::U32,
            PtxSpecialRegister::Envreg31 => ast::ScalarType::U32,
            PtxSpecialRegister::Pm0 => ast::ScalarType::U32,
            PtxSpecialRegister::Pm1 => ast::ScalarType::U32,
            PtxSpecialRegister::Pm2 => ast::ScalarType::U32,
            PtxSpecialRegister::Pm3 => ast::ScalarType::U32,
            PtxSpecialRegister::Pm4 => ast::ScalarType::U32,
            PtxSpecialRegister::Pm5 => ast::ScalarType::U32,
            PtxSpecialRegister::Pm6 => ast::ScalarType::U32,
            PtxSpecialRegister::Pm7 => ast::ScalarType::U32,
            PtxSpecialRegister::Pm0_64 => ast::ScalarType::U64,
            PtxSpecialRegister::Pm1_64 => ast::ScalarType::U64,
            PtxSpecialRegister::Pm2_64 => ast::ScalarType::U64,
            PtxSpecialRegister::Pm3_64 => ast::ScalarType::U64,
            PtxSpecialRegister::Pm4_64 => ast::ScalarType::U64,
            PtxSpecialRegister::Pm5_64 => ast::ScalarType::U64,
            PtxSpecialRegister::Pm6_64 => ast::ScalarType::U64,
            PtxSpecialRegister::Pm7_64 => ast::ScalarType::U64,
            PtxSpecialRegister::ClusterCtarank
            | PtxSpecialRegister::ClusterNctarank
            | PtxSpecialRegister::Clusterid
            | PtxSpecialRegister::Nclusterid
            | PtxSpecialRegister::ClusterCtaid
            | PtxSpecialRegister::ClusterNctaid => ast::ScalarType::U32,
        }
    }

    fn get_function_input_type(self) -> Option<ast::ScalarType> {
        match self {
            PtxSpecialRegister::Tid
            | PtxSpecialRegister::Ntid
            | PtxSpecialRegister::Ctaid
            | PtxSpecialRegister::Nctaid
            | PtxSpecialRegister::Clusterid
            | PtxSpecialRegister::Nclusterid
            | PtxSpecialRegister::ClusterCtaid
            | PtxSpecialRegister::ClusterNctaid => Some(ast::ScalarType::U8),
            PtxSpecialRegister::ClusterCtarank
            | PtxSpecialRegister::ClusterNctarank
            | PtxSpecialRegister::IsExplicitCluster
            | PtxSpecialRegister::Clock
            | PtxSpecialRegister::LanemaskLe
            | PtxSpecialRegister::LanemaskLt
            | PtxSpecialRegister::LanemaskGe
            | PtxSpecialRegister::Laneid => None,
            _ => None,
        }
    }

    fn get_unprefixed_function_name(self) -> &'static str {
        match self {
            PtxSpecialRegister::Tid => "sreg_tid",
            PtxSpecialRegister::Ntid => "sreg_ntid",
            PtxSpecialRegister::Ctaid => "sreg_ctaid",
            PtxSpecialRegister::Nctaid => "sreg_nctaid",
            PtxSpecialRegister::Clock => "sreg_clock",
            PtxSpecialRegister::LanemaskLt => "sreg_lanemask_lt",
            PtxSpecialRegister::LanemaskLe => "sreg_lanemask_le",
            PtxSpecialRegister::LanemaskGe => "sreg_lanemask_ge",
            PtxSpecialRegister::Laneid => "sreg_laneid",
            PtxSpecialRegister::ClusterCtarank => "sreg_cluster_ctarank",
            PtxSpecialRegister::ClusterNctarank => "sreg_cluster_nctarank",
            PtxSpecialRegister::Clusterid => "sreg_clusterid",
            PtxSpecialRegister::Nclusterid => "sreg_nclusterid",
            PtxSpecialRegister::ClusterCtaid => "sreg_cluster_ctaid",
            PtxSpecialRegister::ClusterNctaid => "sreg_cluster_nctaid",
            PtxSpecialRegister::IsExplicitCluster => "sreg_is_explicit_cluster",
            PtxSpecialRegister::LanemaskEq => "sreg_lanemask_eq",
            PtxSpecialRegister::LanemaskGt => "sreg_lanemask_gt",
            PtxSpecialRegister::Clock64 => "sreg_clock64",
            PtxSpecialRegister::Globaltimer => "sreg_globaltimer",
            PtxSpecialRegister::GlobaltimerLo => "sreg_globaltimer_lo",
            PtxSpecialRegister::GlobaltimerHi => "sreg_globaltimer_hi",
            PtxSpecialRegister::Warpid => "sreg_warpid",
            PtxSpecialRegister::Nwarpid => "sreg_nwarpid",
            PtxSpecialRegister::Warpsize => "sreg_warpsize",
            PtxSpecialRegister::Smid => "sreg_smid",
            PtxSpecialRegister::Nsmid => "sreg_nsmid",
            PtxSpecialRegister::Gridid => "sreg_gridid",
            PtxSpecialRegister::DynamicSmemSize => "sreg_dynamic_smem_size",
            PtxSpecialRegister::TotalSmemSize => "sreg_total_smem_size",
            PtxSpecialRegister::AggrSmemSize => "sreg_aggr_smem_size",
            PtxSpecialRegister::ReservedSmemOffsetBegin => "sreg_reserved_smem_offset_begin",
            PtxSpecialRegister::ReservedSmemOffsetEnd => "sreg_reserved_smem_offset_end",
            PtxSpecialRegister::ReservedSmemOffsetCap => "sreg_reserved_smem_offset_cap",
            PtxSpecialRegister::ReservedSmemOffset0 => "sreg_reserved_smem_offset_0",
            PtxSpecialRegister::ReservedSmemOffset1 => "sreg_reserved_smem_offset_1",
            PtxSpecialRegister::Perctamemoryoffset => "sreg_perctamemoryoffset",
            PtxSpecialRegister::Perctamemorysize => "sreg_perctamemorysize",
            PtxSpecialRegister::CurrentGraphExec => "sreg_current_graph_exec",
            PtxSpecialRegister::Envreg0 => "sreg_envreg0",
            PtxSpecialRegister::Envreg1 => "sreg_envreg1",
            PtxSpecialRegister::Envreg2 => "sreg_envreg2",
            PtxSpecialRegister::Envreg3 => "sreg_envreg3",
            PtxSpecialRegister::Envreg4 => "sreg_envreg4",
            PtxSpecialRegister::Envreg5 => "sreg_envreg5",
            PtxSpecialRegister::Envreg6 => "sreg_envreg6",
            PtxSpecialRegister::Envreg7 => "sreg_envreg7",
            PtxSpecialRegister::Envreg8 => "sreg_envreg8",
            PtxSpecialRegister::Envreg9 => "sreg_envreg9",
            PtxSpecialRegister::Envreg10 => "sreg_envreg10",
            PtxSpecialRegister::Envreg11 => "sreg_envreg11",
            PtxSpecialRegister::Envreg12 => "sreg_envreg12",
            PtxSpecialRegister::Envreg13 => "sreg_envreg13",
            PtxSpecialRegister::Envreg14 => "sreg_envreg14",
            PtxSpecialRegister::Envreg15 => "sreg_envreg15",
            PtxSpecialRegister::Envreg16 => "sreg_envreg16",
            PtxSpecialRegister::Envreg17 => "sreg_envreg17",
            PtxSpecialRegister::Envreg18 => "sreg_envreg18",
            PtxSpecialRegister::Envreg19 => "sreg_envreg19",
            PtxSpecialRegister::Envreg20 => "sreg_envreg20",
            PtxSpecialRegister::Envreg21 => "sreg_envreg21",
            PtxSpecialRegister::Envreg22 => "sreg_envreg22",
            PtxSpecialRegister::Envreg23 => "sreg_envreg23",
            PtxSpecialRegister::Envreg24 => "sreg_envreg24",
            PtxSpecialRegister::Envreg25 => "sreg_envreg25",
            PtxSpecialRegister::Envreg26 => "sreg_envreg26",
            PtxSpecialRegister::Envreg27 => "sreg_envreg27",
            PtxSpecialRegister::Envreg28 => "sreg_envreg28",
            PtxSpecialRegister::Envreg29 => "sreg_envreg29",
            PtxSpecialRegister::Envreg30 => "sreg_envreg30",
            PtxSpecialRegister::Envreg31 => "sreg_envreg31",
            PtxSpecialRegister::Pm0 => "sreg_pm0",
            PtxSpecialRegister::Pm1 => "sreg_pm1",
            PtxSpecialRegister::Pm2 => "sreg_pm2",
            PtxSpecialRegister::Pm3 => "sreg_pm3",
            PtxSpecialRegister::Pm4 => "sreg_pm4",
            PtxSpecialRegister::Pm5 => "sreg_pm5",
            PtxSpecialRegister::Pm6 => "sreg_pm6",
            PtxSpecialRegister::Pm7 => "sreg_pm7",
            PtxSpecialRegister::Pm0_64 => "sreg_pm0_64",
            PtxSpecialRegister::Pm1_64 => "sreg_pm1_64",
            PtxSpecialRegister::Pm2_64 => "sreg_pm2_64",
            PtxSpecialRegister::Pm3_64 => "sreg_pm3_64",
            PtxSpecialRegister::Pm4_64 => "sreg_pm4_64",
            PtxSpecialRegister::Pm5_64 => "sreg_pm5_64",
            PtxSpecialRegister::Pm6_64 => "sreg_pm6_64",
            PtxSpecialRegister::Pm7_64 => "sreg_pm7_64",
        }
    }
}

#[cfg(debug_assertions)]
fn error_unreachable() -> TranslateError {
    unreachable!()
}

#[cfg(not(debug_assertions))]
fn error_unreachable() -> TranslateError {
    TranslateError::Unreachable
}

#[cfg(debug_assertions)]
fn error_todo_msg<T: Into<String>>(msg: T) -> TranslateError {
    unreachable!("{}", msg.into())
}

#[cfg(not(debug_assertions))]
fn error_todo_msg<T: Into<String>>(msg: T) -> TranslateError {
    TranslateError::Todo(msg.into())
}

#[cfg(debug_assertions)]
fn error_todo() -> TranslateError {
    unreachable!()
}

#[cfg(not(debug_assertions))]
fn error_todo() -> TranslateError {
    TranslateError::Todo("".to_string())
}

#[cfg(debug_assertions)]
fn error_unknown_symbol<T: Into<String>>(symbol: T) -> TranslateError {
    panic!("Unknown symbol: \"{}\"", symbol.into())
}

#[cfg(not(debug_assertions))]
fn error_unknown_symbol<T: Into<String>>(symbol: T) -> TranslateError {
    TranslateError::UnknownSymbol(symbol.into())
}

#[cfg(debug_assertions)]
fn error_mismatched_type() -> TranslateError {
    panic!("Mismatched type")
}

#[cfg(not(debug_assertions))]
fn error_mismatched_type() -> TranslateError {
    TranslateError::MismatchedType
}

enum Statement<I, P: ast::Operand> {
    Label(SpirvWord),
    Variable(ast::Variable<P::Ident>),
    Instruction(I),
    // SPIR-V compatible replacement for PTX predicates
    Conditional(BranchCondition),
    Conversion(ImplicitConversion),
    Constant(ConstantDefinition),
    RetValue(ast::RetData, Vec<(SpirvWord, ast::Type)>),
    PtrAccess(PtrAccess<P>),
    RepackVector(RepackVectorDetails),
    FunctionPointer(FunctionPointerDetails),
    VectorRead(VectorRead),
    VectorWrite(VectorWrite),
    SetMode(ModeRegister),
    // This instruction is a nop, it serves as a marker to indicate that the
    // next instruction requires certain floating-point modes to be set.
    // Some transcendentals compile to a sequence of instructions that
    // require certain modes to be set _mid-function_.
    // See replace_instructions_with_functions_fp_required pass for details
    FpModeRequired {
        ftz_f32: Option<bool>,
        rnd_f32: Option<ast::RoundingMode>,
        ftz_f16f64: Option<bool>,
        rnd_f16f64: Option<ast::RoundingMode>,
    },
    FpSaturate {
        dst: SpirvWord,
        src: SpirvWord,
        type_: ast::ScalarType,
    },
}

#[derive(Eq, PartialEq, Clone, Copy)]
#[cfg_attr(test, derive(Debug))]
enum ModeRegister {
    Denormal {
        f32: bool,
        f16f64: bool,
    },
    Rounding {
        f32: ast::RoundingMode,
        f16f64: ast::RoundingMode,
    },
}

impl std::fmt::Display for ModeRegister {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModeRegister::Denormal { f32, f16f64 } => write!(
                f,
                "zluda.set_mode.denormal.f32::{}.f16f64::{}",
                if *f32 { "on" } else { "off" },
                if *f16f64 { "on" } else { "off" },
            ),
            ModeRegister::Rounding { f32, f16f64 } => {
                write!(f, "zluda.set_mode.rounding.f32{}.f16f64{}", f32, f16f64)
            }
        }
    }
}

impl<T: ast::Operand<Ident = SpirvWord>> Statement<ast::Instruction<T>, T> {
    fn visit_map<To: ast::Operand<Ident = SpirvWord>, Err>(
        self,
        visitor: &mut impl ast::VisitorMap<T, To, Err>,
    ) -> std::result::Result<Statement<ast::Instruction<To>, To>, Err> {
        Ok(match self {
            Statement::Instruction(i) => {
                return ast::visit_map(i, visitor).map(Statement::Instruction)
            }
            Statement::Label(label) => {
                Statement::Label(visitor.visit_ident(label, None, false, false)?)
            }
            Statement::Variable(var) => {
                let name = visitor.visit_ident(
                    var.name,
                    Some((&var.info.v_type, var.info.state_space)),
                    true,
                    false,
                )?;
                Statement::Variable(ast::Variable {
                    info: ast::VariableInfo {
                        align: var.info.align,
                        v_type: var.info.v_type,
                        state_space: var.info.state_space,
                        array_init: var.info.array_init,
                    },
                    name,
                })
            }
            Statement::Conditional(conditional) => {
                let predicate = visitor.visit_ident(
                    conditional.predicate,
                    Some((&ast::ScalarType::Pred.into(), ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                let if_true = visitor.visit_ident(conditional.if_true, None, false, false)?;
                let if_false = visitor.visit_ident(conditional.if_false, None, false, false)?;
                Statement::Conditional(BranchCondition {
                    predicate,
                    if_true,
                    if_false,
                })
            }
            Statement::Conversion(ImplicitConversion {
                src,
                dst,
                from_type,
                to_type,
                from_space,
                to_space,
                kind,
            }) => {
                let dst = visitor.visit_ident(
                    dst,
                    Some((&to_type, ast::StateSpace::Reg)),
                    true,
                    false,
                )?;
                let src = visitor.visit_ident(
                    src,
                    Some((&from_type, ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                Statement::Conversion(ImplicitConversion {
                    src,
                    dst,
                    from_type,
                    to_type,
                    from_space,
                    to_space,
                    kind,
                })
            }
            Statement::Constant(ConstantDefinition { dst, typ, value }) => {
                let dst = visitor.visit_ident(
                    dst,
                    Some((&typ.into(), ast::StateSpace::Reg)),
                    true,
                    false,
                )?;
                Statement::Constant(ConstantDefinition { dst, typ, value })
            }
            Statement::RetValue(data, value) => {
                let value = value
                    .into_iter()
                    .map(|(ident, type_)| {
                        Ok((
                            visitor.visit_ident(
                                ident,
                                Some((&type_, ast::StateSpace::Local)),
                                false,
                                false,
                            )?,
                            type_,
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Statement::RetValue(data, value)
            }
            Statement::PtrAccess(PtrAccess {
                underlying_type,
                state_space,
                dst,
                ptr_src,
                offset_src,
            }) => {
                let dst =
                    visitor.visit_ident(dst, Some((&underlying_type, state_space)), true, false)?;
                let ptr_src = visitor.visit_ident(
                    ptr_src,
                    Some((&underlying_type, state_space)),
                    false,
                    false,
                )?;
                let offset_src = visitor.visit(
                    offset_src,
                    Some((
                        &ast::Type::Scalar(ast::ScalarType::S64),
                        ast::StateSpace::Reg,
                    )),
                    false,
                    false,
                )?;
                Statement::PtrAccess(PtrAccess {
                    underlying_type,
                    state_space,
                    dst,
                    ptr_src,
                    offset_src,
                })
            }
            Statement::VectorRead(VectorRead {
                scalar_type,
                vector_width,
                scalar_dst: dst,
                vector_src,
                member,
            }) => {
                let scalar_t = scalar_type.into();
                let vector_t = ast::Type::Vector(vector_width, scalar_type);
                let dst: SpirvWord = visitor.visit_ident(
                    dst,
                    Some((&scalar_t, ast::StateSpace::Reg)),
                    true,
                    false,
                )?;
                let src = visitor.visit_ident(
                    vector_src,
                    Some((&vector_t, ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                Statement::VectorRead(VectorRead {
                    scalar_type,
                    vector_width,
                    scalar_dst: dst,
                    vector_src: src,
                    member,
                })
            }
            Statement::VectorWrite(VectorWrite {
                scalar_type,
                vector_width,
                vector_dst,
                vector_src,
                scalar_src,
                member,
            }) => {
                let scalar_t = scalar_type.into();
                let vector_t = ast::Type::Vector(vector_width, scalar_type);
                let vector_dst = visitor.visit_ident(
                    vector_dst,
                    Some((&vector_t, ast::StateSpace::Reg)),
                    true,
                    false,
                )?;
                let vector_src = visitor.visit_ident(
                    vector_src,
                    Some((&vector_t, ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                let scalar_src = visitor.visit_ident(
                    scalar_src,
                    Some((&scalar_t, ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                Statement::VectorWrite(VectorWrite {
                    vector_dst,
                    vector_src,
                    scalar_src,
                    scalar_type,
                    vector_width,
                    member,
                })
            }
            Statement::RepackVector(RepackVectorDetails {
                is_extract,
                typ,
                packed,
                mut unpacked,
                relaxed_type_check,
            }) => {
                let (packed, unpacked) = if is_extract {
                    for ident in unpacked.iter_mut().flatten() {
                        *ident = visitor.visit_ident(
                            *ident,
                            Some((&typ.into(), ast::StateSpace::Reg)),
                            true,
                            relaxed_type_check,
                        )?;
                    }
                    let packed = visitor.visit_ident(
                        packed,
                        Some((
                            &ast::Type::Vector(unpacked.len() as u8, typ),
                            ast::StateSpace::Reg,
                        )),
                        false,
                        false,
                    )?;
                    (packed, unpacked)
                } else {
                    let packed = visitor.visit_ident(
                        packed,
                        Some((
                            &ast::Type::Vector(unpacked.len() as u8, typ),
                            ast::StateSpace::Reg,
                        )),
                        true,
                        false,
                    )?;
                    for ident in unpacked.iter_mut().flatten() {
                        *ident = visitor.visit_ident(
                            *ident,
                            Some((&typ.into(), ast::StateSpace::Reg)),
                            false,
                            relaxed_type_check,
                        )?;
                    }
                    (packed, unpacked)
                };
                Statement::RepackVector(RepackVectorDetails {
                    is_extract,
                    typ,
                    packed,
                    unpacked,
                    relaxed_type_check,
                })
            }
            Statement::FunctionPointer(FunctionPointerDetails { dst, src }) => {
                let dst = visitor.visit_ident(
                    dst,
                    Some((
                        &ast::Type::Scalar(ast::ScalarType::U64),
                        ast::StateSpace::Reg,
                    )),
                    true,
                    false,
                )?;
                let src = visitor.visit_ident(src, None, false, false)?;
                Statement::FunctionPointer(FunctionPointerDetails { dst, src })
            }
            Statement::SetMode(mode_register) => Statement::SetMode(mode_register),
            Statement::FpSaturate { dst, src, type_ } => {
                let dst = visitor.visit_ident(
                    dst,
                    Some((&type_.into(), ast::StateSpace::Reg)),
                    true,
                    false,
                )?;
                let src = visitor.visit_ident(
                    src,
                    Some((&type_.into(), ast::StateSpace::Reg)),
                    false,
                    false,
                )?;
                Statement::FpSaturate { dst, src, type_ }
            }
            Statement::FpModeRequired {
                ftz_f32,
                rnd_f32,
                ftz_f16f64,
                rnd_f16f64,
            } => Statement::FpModeRequired {
                ftz_f32,
                rnd_f32,
                ftz_f16f64,
                rnd_f16f64,
            },
        })
    }
}

struct BranchCondition {
    predicate: SpirvWord,
    if_true: SpirvWord,
    if_false: SpirvWord,
}

impl std::fmt::Display for BranchCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "zluda.branch")
    }
}

#[derive(Clone)]
struct ImplicitConversion {
    src: SpirvWord,
    dst: SpirvWord,
    from_type: ast::Type,
    to_type: ast::Type,
    from_space: ast::StateSpace,
    to_space: ast::StateSpace,
    kind: ConversionKind,
}

impl std::fmt::Display for ImplicitConversion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "zluda.convert_implicit{}{}{}{}{}",
            self.kind, self.to_space, self.to_type, self.from_space, self.from_type
        )
    }
}

#[derive(PartialEq, Clone, strum_macros::Display)]
#[strum(serialize_all = "snake_case", prefix = ".")]
enum ConversionKind {
    Default,
    // zero-extend/chop/bitcast depending on types
    SignExtend,
    BitToPtr,
    PtrToPtr,
    AddressOf,
}

struct ConstantDefinition {
    pub dst: SpirvWord,
    pub typ: ast::ScalarType,
    pub value: ast::ImmediateValue,
}

impl std::fmt::Display for ConstantDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "zluda.constant{} {}", self.typ, self.value)
    }
}

pub struct PtrAccess<T> {
    underlying_type: ast::Type,
    state_space: ast::StateSpace,
    dst: SpirvWord,
    ptr_src: SpirvWord,
    offset_src: T,
}

struct RepackVectorDetails {
    is_extract: bool,
    typ: ast::ScalarType,
    packed: SpirvWord,
    // None discards an extracted lane without allocating a destination register.
    unpacked: Vec<Option<SpirvWord>>,
    relaxed_type_check: bool,
}

impl std::fmt::Display for RepackVectorDetails {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let extract = if self.is_extract {
            ".extract"
        } else {
            ".composite"
        };
        let relaxed = if self.relaxed_type_check {
            ".relaxed"
        } else {
            ""
        };
        write!(f, "zluda.repack_vector{}{}{}", extract, relaxed, self.typ)
    }
}

struct FunctionPointerDetails {
    dst: SpirvWord,
    src: SpirvWord,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct SpirvWord(u32);

impl std::fmt::Display for SpirvWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl From<u32> for SpirvWord {
    fn from(value: u32) -> Self {
        Self(value)
    }
}
impl From<SpirvWord> for u32 {
    fn from(value: SpirvWord) -> Self {
        value.0
    }
}

impl ast::Operand for SpirvWord {
    type Ident = Self;

    fn from_ident(ident: Self::Ident) -> Self {
        ident
    }
}

type ExpandedStatement = Statement<ast::Instruction<SpirvWord>, SpirvWord>;

type NormalizedStatement = Statement<
    (
        Option<ast::PredAt<SpirvWord>>,
        ast::Instruction<ast::ParsedOperand<SpirvWord>>,
    ),
    ast::ParsedOperand<SpirvWord>,
>;

enum Directive2<Instruction, Operand: ast::Operand> {
    Variable(ast::LinkingDirective, ast::Variable<SpirvWord>),
    Method(Function<Instruction, Operand>),
}

struct KernelAttributes {
    flush_to_zero_f32: bool,
    flush_to_zero_f16f64: bool,
    rounding_mode_f32: ast::RoundingMode,
    rounding_mode_f16f64: ast::RoundingMode,
}

struct Function<Instruction, Operand: ast::Operand> {
    pub return_arguments: Vec<ast::Variable<Operand::Ident>>,
    pub name: Operand::Ident,
    pub input_arguments: Vec<ast::Variable<Operand::Ident>>,
    pub body: Option<Vec<Statement<Instruction, Operand>>>,
    kernel_attributes: Option<KernelAttributes>,
    import_as: Option<String>,
    tuning: Vec<ast::TuningDirective>,
    linkage: ast::LinkingDirective,
    kernel_meta32: Option<Function32>,
}

// We transform the method so that it takes an implicit pointer to memory as an
// additional argument, and all explicit memory accesses are done through that
// pointer
struct Function32 {
    implicit_memory_ptr: SpirvWord,
}

impl<I, O: ast::Operand> Function<I, O> {
    fn is_kernel(&self) -> bool {
        self.kernel_attributes.is_some()
    }
}

type NormalizedDirective2 = Directive2<
    (
        Option<ast::PredAt<SpirvWord>>,
        ast::Instruction<ast::ParsedOperand<SpirvWord>>,
    ),
    ast::ParsedOperand<SpirvWord>,
>;

type NormalizedFunction2 = Function<
    (
        Option<ast::PredAt<SpirvWord>>,
        ast::Instruction<ast::ParsedOperand<SpirvWord>>,
    ),
    ast::ParsedOperand<SpirvWord>,
>;

type UnconditionalDirective =
    Directive2<ast::Instruction<ast::ParsedOperand<SpirvWord>>, ast::ParsedOperand<SpirvWord>>;

type UnconditionalFunction =
    Function<ast::Instruction<ast::ParsedOperand<SpirvWord>>, ast::ParsedOperand<SpirvWord>>;

struct GlobalStringIdentResolver2<'input> {
    pub(crate) current_id: SpirvWord,
    pub(crate) ident_map: FxHashMap<SpirvWord, IdentEntry<'input>>,
}

impl<'input> GlobalStringIdentResolver2<'input> {
    fn new(spirv_word: SpirvWord) -> Self {
        Self {
            current_id: spirv_word,
            ident_map: FxHashMap::default(),
        }
    }

    fn register_named(
        &mut self,
        name: Cow<'input, str>,
        type_space: Option<(ast::Type, ast::StateSpace)>,
    ) -> SpirvWord {
        let new_id = self.current_id;
        self.ident_map.insert(
            new_id,
            IdentEntry {
                name: Some(name),
                type_space,
            },
        );
        self.current_id.0 += 1;
        new_id
    }

    fn register_unnamed(&mut self, type_space: Option<(ast::Type, ast::StateSpace)>) -> SpirvWord {
        let new_id = self.current_id;
        self.ident_map.insert(
            new_id,
            IdentEntry {
                name: None,
                type_space,
            },
        );
        self.current_id.0 += 1;
        new_id
    }

    fn get_typed(&self, id: SpirvWord) -> Result<&(ast::Type, ast::StateSpace), TranslateError> {
        match self.ident_map.get(&id) {
            Some(IdentEntry {
                type_space: Some(type_space),
                ..
            }) => Ok(type_space),
            _ => Err(error_unknown_symbol(format!("{:?}", id))),
        }
    }
}

struct IdentEntry<'input> {
    name: Option<Cow<'input, str>>,
    type_space: Option<(ast::Type, ast::StateSpace)>,
}

struct ScopedResolver<'input, 'b> {
    flat_resolver: &'b mut GlobalStringIdentResolver2<'input>,
    scopes: Vec<ScopeMarker<'input>>,
}

impl<'input, 'b> ScopedResolver<'input, 'b> {
    fn new(flat_resolver: &'b mut GlobalStringIdentResolver2<'input>) -> Self {
        Self {
            flat_resolver,
            scopes: vec![ScopeMarker::new()],
        }
    }

    fn start_scope(&mut self) {
        self.scopes.push(ScopeMarker::new());
    }

    fn end_scope(&mut self) {
        let scope = self.scopes.pop().unwrap();
        scope.flush(self.flat_resolver);
    }

    fn add_or_get_in_current_scope_untyped(
        &mut self,
        name: &'input str,
    ) -> Result<SpirvWord, TranslateError> {
        let current_scope = self.scopes.last_mut().unwrap();
        Ok(
            match current_scope.name_to_ident.entry(Cow::Borrowed(name)) {
                hash_map::Entry::Occupied(occupied_entry) => {
                    let ident = *occupied_entry.get();
                    let entry = current_scope
                        .ident_map
                        .get(&ident)
                        .ok_or_else(|| error_unreachable())?;
                    if entry.type_space.is_some() {
                        return Err(error_unknown_symbol(name));
                    }
                    ident
                }
                hash_map::Entry::Vacant(vacant_entry) => {
                    let new_id = self.flat_resolver.current_id;
                    self.flat_resolver.current_id.0 += 1;
                    vacant_entry.insert(new_id);
                    current_scope.ident_map.insert(
                        new_id,
                        IdentEntry {
                            name: Some(Cow::Borrowed(name)),
                            type_space: None,
                        },
                    );
                    new_id
                }
            },
        )
    }

    fn add(
        &mut self,
        name: Cow<'input, str>,
        type_space: Option<(ast::Type, ast::StateSpace)>,
    ) -> Result<SpirvWord, TranslateError> {
        let result = self.flat_resolver.current_id;
        self.flat_resolver.current_id.0 += 1;
        let current_scope = self.scopes.last_mut().unwrap();
        if current_scope
            .name_to_ident
            .insert(name.clone(), result)
            .is_some()
        {
            return Err(error_unknown_symbol(name));
        }
        current_scope.ident_map.insert(
            result,
            IdentEntry {
                name: Some(name),
                type_space,
            },
        );
        Ok(result)
    }

    fn get(&mut self, name: &str) -> Result<SpirvWord, TranslateError> {
        self.scopes
            .iter()
            .rev()
            .find_map(|resolver| resolver.name_to_ident.get(name).copied())
            .ok_or_else(|| error_unknown_symbol(name))
    }

    fn get_in_current_scope(&self, label: &'input str) -> Result<SpirvWord, TranslateError> {
        let current_scope = self.scopes.last().unwrap();
        current_scope
            .name_to_ident
            .get(label)
            .copied()
            .ok_or_else(|| error_unreachable())
    }
}

struct ScopeMarker<'input> {
    ident_map: FxHashMap<SpirvWord, IdentEntry<'input>>,
    name_to_ident: FxHashMap<Cow<'input, str>, SpirvWord>,
}

impl<'input> ScopeMarker<'input> {
    fn new() -> Self {
        Self {
            ident_map: FxHashMap::default(),
            name_to_ident: FxHashMap::default(),
        }
    }

    fn flush(self, resolver: &mut GlobalStringIdentResolver2<'input>) {
        resolver.ident_map.extend(self.ident_map);
    }
}

struct SpecialRegistersMap {
    reg_to_id: FxHashMap<PtxSpecialRegister, SpirvWord>,
    id_to_reg: FxHashMap<SpirvWord, PtxSpecialRegister>,
}

impl SpecialRegistersMap {
    fn new(resolver: &mut ScopedResolver) -> Result<Self, TranslateError> {
        let mut result = SpecialRegistersMap {
            reg_to_id: FxHashMap::default(),
            id_to_reg: FxHashMap::default(),
        };
        for sreg in PtxSpecialRegister::iter() {
            let text = sreg.as_str();
            let id = resolver.add(
                Cow::Borrowed(text),
                Some((sreg.get_type(), ast::StateSpace::Reg)),
            )?;
            result.reg_to_id.insert(sreg, id);
            result.id_to_reg.insert(id, sreg);
        }
        Ok(result)
    }

    fn get(&self, id: SpirvWord) -> Option<PtxSpecialRegister> {
        self.id_to_reg.get(&id).copied()
    }

    fn len() -> usize {
        PtxSpecialRegister::iter().len()
    }

    fn foreach_declaration<'a, 'input>(
        resolver: &'a mut GlobalStringIdentResolver2<'input>,
        mut fn_: impl FnMut(
            PtxSpecialRegister,
            (
                Vec<ast::Variable<SpirvWord>>,
                SpirvWord,
                Vec<ast::Variable<SpirvWord>>,
            ),
        ),
    ) {
        for sreg in PtxSpecialRegister::iter() {
            let external_fn_name = [ZLUDA_PTX_PREFIX, sreg.get_unprefixed_function_name()].concat();
            let name = resolver.register_named(Cow::Owned(external_fn_name), None);
            let return_type = sreg.get_function_return_type();
            let input_type = sreg.get_function_input_type();
            let return_arguments = vec![ast::Variable {
                info: ast::VariableInfo {
                    align: None,
                    v_type: return_type.into(),
                    state_space: ast::StateSpace::Reg,
                    array_init: Vec::new(),
                },
                name: resolver.register_unnamed(Some((return_type.into(), ast::StateSpace::Reg))),
            }];
            let input_arguments = input_type
                .into_iter()
                .map(|type_| ast::Variable {
                    info: ast::VariableInfo {
                        align: None,
                        v_type: type_.into(),
                        state_space: ast::StateSpace::Reg,
                        array_init: Vec::new(),
                    },
                    name: resolver.register_unnamed(Some((type_.into(), ast::StateSpace::Reg))),
                })
                .collect::<Vec<_>>();
            fn_(sreg, (return_arguments, name, input_arguments));
        }
    }
}

pub struct VectorRead {
    scalar_type: ast::ScalarType,
    vector_width: u8,
    scalar_dst: SpirvWord,
    vector_src: SpirvWord,
    member: u8,
}

pub struct VectorWrite {
    scalar_type: ast::ScalarType,
    vector_width: u8,
    vector_dst: SpirvWord,
    vector_src: SpirvWord,
    scalar_src: SpirvWord,
    member: u8,
}

fn scalar_to_ptx_name(this: ast::ScalarType) -> &'static str {
    match this {
        ast::ScalarType::B8 => "b8",
        ast::ScalarType::B16 => "b16",
        ast::ScalarType::B32 => "b32",
        ast::ScalarType::B64 => "b64",
        ast::ScalarType::B128 => "b128",
        ast::ScalarType::U8 => "u8",
        ast::ScalarType::U16 => "u16",
        ast::ScalarType::U16x2 => "u16x2",
        ast::ScalarType::U32 => "u32",
        ast::ScalarType::U64 => "u64",
        ast::ScalarType::S8 => "s8",
        ast::ScalarType::S16 => "s16",
        ast::ScalarType::S16x2 => "s16x2",
        ast::ScalarType::S32 => "s32",
        ast::ScalarType::S64 => "s64",
        ast::ScalarType::F16 => "f16",
        ast::ScalarType::F16x2 => "f16x2",
        ast::ScalarType::F32 => "f32",
        ast::ScalarType::F64 => "f64",
        ast::ScalarType::BF16 => "bf16",
        ast::ScalarType::BF16x2 => "bf16x2",
        ast::ScalarType::Pred => "pred",
        ast::ScalarType::E4m3x2 => "e4m3x2",
        ast::ScalarType::E5m2x2 => "e5m2x2",
    }
}

type UnconditionalStatement =
    Statement<ast::Instruction<ast::ParsedOperand<SpirvWord>>, ast::ParsedOperand<SpirvWord>>;
