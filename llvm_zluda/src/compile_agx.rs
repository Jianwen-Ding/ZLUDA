//! What compile_air shares with the AGX backend: its LLVM set-up, diagnostics and
//! the loading and internalizing of modules. See docs/src/agx.md for the kernel ABI.

use crate::utils::{Context, Message, Module};
use crate::{
    LLVMInitializeAGXAsmPrinter, LLVMInitializeAGXTarget, LLVMInitializeAGXTargetInfo,
    LLVMInitializeAGXTargetMC, LLVMZludaParseCommandLineOptions,
};
use llvm_sys::core::{
    LLVMContextSetDiagnosticHandler, LLVMDisposeMessage, LLVMGetDiagInfoDescription,
    LLVMGetDiagInfoSeverity, LLVMGetFirstFunction, LLVMGetNextFunction, LLVMGetValueName2,
    LLVMGetVisibility, LLVMIsDeclaration, LLVMSetLinkage, LLVMSetVisibility,
};
use llvm_sys::{LLVMLinkage, LLVMVisibility};
use llvm_sys::prelude::LLVMDiagnosticInfoRef;
use llvm_sys::LLVMDiagnosticSeverity;
use std::ffi::{c_void, CStr};
use std::sync::OnceLock;

pub const AGX_TRIPLE: &CStr = c"agx-apple-macosx";

/// The default processor: the M2 family, whose latencies the scheduling model
/// is based on.
pub const AGX_DEFAULT_CPU: &str = "agx-g14g";

/// Collects the errors the backend reports through the context's diagnostic
/// handler (unsupported operations, missing encodings ...), which would
/// otherwise only be printed.
pub(crate) struct Diagnostics<'a> {
    ctx: &'a Context,
    errors: Box<Vec<String>>,
}

extern "C" fn collect_diagnostic(info: LLVMDiagnosticInfoRef, errors: *mut c_void) {
    let severity = unsafe { LLVMGetDiagInfoSeverity(info) };
    if severity != LLVMDiagnosticSeverity::LLVMDSError {
        return;
    }
    let description = unsafe { LLVMGetDiagInfoDescription(info) };
    let message = unsafe { CStr::from_ptr(description) }
        .to_string_lossy()
        .into_owned();
    unsafe { LLVMDisposeMessage(description) };
    unsafe { (*(errors as *mut Vec<String>)).push(message) };
}

impl<'a> Diagnostics<'a> {
    pub(crate) fn new(ctx: &'a Context) -> Self {
        let mut errors = Box::new(Vec::new());
        unsafe {
            LLVMContextSetDiagnosticHandler(
                ctx.get(),
                Some(collect_diagnostic),
                (&mut *errors as *mut Vec<String>).cast(),
            )
        };
        Self { ctx, errors }
    }

    pub(crate) fn check(&self) -> Result<(), String> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.join("\n"))
        }
    }
}

impl Drop for Diagnostics<'_> {
    fn drop(&mut self) {
        unsafe { LLVMContextSetDiagnosticHandler(self.ctx.get(), None, std::ptr::null_mut()) };
    }
}

pub(crate) fn load_module(ctx: &Context, ir: &[u8], name: &CStr) -> Result<Module, String> {
    let module = Module::try_from_ir(ctx, ir, name)?;
    module.verify()?;
    Ok(module)
}

/// The helper functions are an implementation detail of the module: make
/// them internal, so that the unused ones are removed instead of being
/// compiled as entry points (for example when every kernel of the module was
/// dropped by `--ignore-errors`).
pub(crate) fn internalize_helpers(module: &Module) {
    unsafe {
        let mut function = LLVMGetFirstFunction(module.get());
        while !function.is_null() {
            let mut len = 0;
            let name = LLVMGetValueName2(function, &mut len);
            let name = std::slice::from_raw_parts(name as *const u8, len);
            if LLVMIsDeclaration(function) == 0 && name.starts_with(b"__zluda_ptx_impl_") {
                LLVMSetLinkage(function, LLVMLinkage::LLVMInternalLinkage);
                LLVMSetVisibility(function, LLVMVisibility::LLVMDefaultVisibility);
            }
            function = LLVMGetNextFunction(function);
        }
    }
}

/// A linked image is complete: its functions (hidden, as the PTX translation emits every non-kernel function)
/// become internal, so that they are inlined into the kernels and the unused ones disappear.
pub(crate) fn internalize_functions(module: &Module) {
    unsafe {
        let mut function = LLVMGetFirstFunction(module.get());
        while !function.is_null() {
            if LLVMIsDeclaration(function) == 0 && LLVMGetVisibility(function) == LLVMVisibility::LLVMHiddenVisibility {
                LLVMSetLinkage(function, LLVMLinkage::LLVMInternalLinkage);
                LLVMSetVisibility(function, LLVMVisibility::LLVMDefaultVisibility);
            }
            function = LLVMGetNextFunction(function);
        }
    }
}

pub(crate) fn init_globals() -> Result<(), String> {
    static INIT_AGX: OnceLock<Result<(), Message>> = OnceLock::new();
    INIT_AGX
        .get_or_init(|| {
            // ZLUDA_AGX_LLVM_ARGS: extra LLVM options, separated by whitespace (backend experiments, for example
            // -agx-threadgroup-unroll=4)
            let mut options = vec![std::ffi::CString::new("llvm_zluda").unwrap()];
            // Wide constants of kernels in uniforms (.agx.kernels extension 6, which agx_runtime sets at launch):
            // fewer registers, more resident SIMD-groups. ZLUDA_AGX_UNIFORM_CONSTANTS=0 turns it off.
            if std::env::var("ZLUDA_AGX_UNIFORM_CONSTANTS").map_or(true, |v| v != "0") {
                options.push(std::ffi::CString::new("-agx-uniform-constants").unwrap());
            }
            // Division by a 32-bit kernel argument with a multiplier and shifts agx_runtime computes at launch
            // (.agx.kernels extension 7). ZLUDA_AGX_LAUNCH_DIVISORS=0 turns it off.
            if std::env::var("ZLUDA_AGX_LAUNCH_DIVISORS").map_or(true, |v| v != "0") {
                options.push(std::ffi::CString::new("-agx-launch-divisors").unwrap());
            }
            if let Ok(extra) = std::env::var("ZLUDA_AGX_LLVM_ARGS") {
                options.extend(extra.split_whitespace().filter_map(|a| std::ffi::CString::new(a).ok()));
            }
            let ptrs: Vec<*const std::ffi::c_char> = options.iter().map(|s| s.as_ptr()).collect();
            let mut err_msg = std::ptr::null_mut();
            let success = unsafe {
                LLVMZludaParseCommandLineOptions(ptrs.len() as i32, ptrs.as_ptr(), &mut err_msg)
            };
            if !success {
                return Err(Message::new(unsafe { CStr::from_ptr(err_msg) }));
            }
            unsafe { LLVMInitializeAGXTargetInfo() };
            unsafe { LLVMInitializeAGXTarget() };
            unsafe { LLVMInitializeAGXTargetMC() };
            unsafe { LLVMInitializeAGXAsmPrinter() };
            #[cfg(zluda_air)]
            unsafe {
                crate::ffi::LLVMInitializeAIRTargetInfo();
                crate::ffi::LLVMInitializeAIRTarget();
                crate::ffi::LLVMInitializeAIRTargetMC();
            }
            Ok(())
        })
        .as_ref()
        .map(|()| ())
        .map_err(|e| e.to_str().to_string())
}
