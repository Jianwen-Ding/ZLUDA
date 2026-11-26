//! PTX -> AIR -> metallib compilation pipeline
//!
//! 1. `ptx_parser` + `ptx::to_llvm_module` (the regular ZLUDA translator)
//! 2. `zluda_air_lower` (cpp/air_lower.cpp): link the Metal helper library,
//!    retarget to AIR, build Metal kernel entry points
//! 3. `zluda_metallib_write` (cpp/metallib.cpp): pack the per-kernel `.air`
//!    modules into a `.metallib`
use crate::ffi;
use std::ffi::{CStr, CString};
use std::ptr;

/// Metal helper library, built from ptx/lib/zluda_ptx_metal_impl.cpp by build.rs
pub static ZLUDA_PTX_METAL_IMPL: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/zluda_ptx_metal_impl.bc"));

#[derive(Clone, Debug)]
pub struct KernelParam {
    pub offset: u64,
    pub size: u64,
}

#[derive(Clone, Debug)]
pub struct Kernel {
    pub name: String,
    pub params: Vec<KernelParam>,
    pub params_size: u64,
    pub dynamic_shared: bool,
    pub static_shared: u64,
    /// AIR bitcode module holding just this kernel
    pub air: Vec<u8>,
    /// Where the address of the module's device-globals buffer goes in the
    /// parameter buffer (if the kernel uses device globals)
    pub globals_param_offset: Option<u64>,
}

/// Writable module-scope device variable, a slice of the globals buffer
#[derive(Clone, Debug)]
pub struct DeviceGlobal {
    pub name: String,
    pub offset: u64,
    pub size: u64,
}

#[derive(Clone, Debug)]
pub struct CompiledModule {
    pub kernels: Vec<Kernel>,
    pub globals: Vec<DeviceGlobal>,
    /// Initial contents of the device-globals buffer
    pub globals_init: Vec<u8>,
    pub metallib: Vec<u8>,
    /// Textual AIR of the whole module (only when requested)
    pub air_text: Option<String>,
    /// ZLUDA's LLVM IR before AIR lowering (only when requested)
    pub zluda_ir_text: Option<String>,
}

#[derive(Default, Clone, Copy)]
pub struct Options {
    pub keep_text: bool,
    /// keep doubles and directed roundings in the AIR (for llvm-air2agx, which emulates them)
    pub portable_air: bool,
}

pub fn compile_ptx(ptx: &str, opts: Options) -> Result<CompiledModule, String> {
    let ast = ptx_parser::parse_module_checked(ptx).map_err(|errs| {
        errs.iter()
            .map(|e| format!("{e:?}"))
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    let module = ptx::to_llvm_module(
        ast,
        ptx::Attributes {
            clock_rate: 1_398_000,
        },
        |_| {},
    )
    .map_err(|e| format!("PTX translation failed: {e}"))?;
    let zluda_ir_text = opts
        .keep_text
        .then(|| module.llvm_ir.print_module_to_string().to_str().to_string());
    let main_bc = module.llvm_ir.write_bitcode_to_memory();
    let attributes_bc = module.attributes_ir.write_bitcode_to_memory();
    let lowered = lower_to_air(&main_bc, &attributes_bc, opts.keep_text, opts.portable_air)?;
    let metallib = write_metallib(&lowered.kernels)?;
    Ok(CompiledModule {
        kernels: lowered.kernels,
        globals: lowered.globals,
        globals_init: lowered.globals_init,
        air_text: lowered.text,
        metallib,
        zluda_ir_text,
    })
}

struct Lowered {
    kernels: Vec<Kernel>,
    globals: Vec<DeviceGlobal>,
    globals_init: Vec<u8>,
    text: Option<String>,
}

fn lower_to_air(main_bc: &[u8], attributes_bc: &[u8], want_text: bool, portable_air: bool) -> Result<Lowered, String> {
    let input = ffi::ZludaAirInput {
        main_bc: main_bc.as_ptr(),
        main_len: main_bc.len(),
        attributes_bc: attributes_bc.as_ptr(),
        attributes_len: attributes_bc.len(),
        impl_bc: ZLUDA_PTX_METAL_IMPL.as_ptr(),
        impl_len: ZLUDA_PTX_METAL_IMPL.len(),
        want_text: want_text as i32,
        portable_air: portable_air as i32,
    };
    let mut output = ffi::ZludaAirOutput {
        kernels: ptr::null_mut(),
        kernel_count: 0,
        globals: ptr::null_mut(),
        global_count: 0,
        globals_init: ptr::null_mut(),
        globals_size: 0,
        module_text: ptr::null_mut(),
    };
    let mut error = ptr::null_mut();
    let result = unsafe { ffi::zluda_air_lower(&input, &mut output, &mut error) };
    if result != 0 {
        let msg = unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned();
        unsafe { ffi::zluda_air_free_string(error) };
        return Err(msg);
    }
    let mut kernels = Vec::new();
    for i in 0..output.kernel_count {
        let k = unsafe { &*output.kernels.add(i) };
        let params = (0..k.param_count)
            .map(|p| unsafe {
                KernelParam {
                    offset: *k.param_offsets.add(p),
                    size: *k.param_sizes.add(p),
                }
            })
            .collect();
        kernels.push(Kernel {
            name: unsafe { CStr::from_ptr(k.name) }
                .to_string_lossy()
                .into_owned(),
            params,
            params_size: k.params_size,
            dynamic_shared: k.dynamic_shared != 0,
            static_shared: k.static_shared,
            air: unsafe { std::slice::from_raw_parts(k.air, k.air_len) }.to_vec(),
            globals_param_offset: (k.globals_param_offset != u64::MAX)
                .then_some(k.globals_param_offset),
        });
    }
    let globals = (0..output.global_count)
        .map(|i| unsafe {
            let g = &*output.globals.add(i);
            DeviceGlobal {
                name: CStr::from_ptr(g.name).to_string_lossy().into_owned(),
                offset: g.offset,
                size: g.size,
            }
        })
        .collect();
    let globals_init =
        unsafe { std::slice::from_raw_parts(output.globals_init, output.globals_size as usize) }
            .to_vec();
    let text = (!output.module_text.is_null()).then(|| {
        unsafe { CStr::from_ptr(output.module_text) }
            .to_string_lossy()
            .into_owned()
    });
    unsafe { ffi::zluda_air_free(&mut output) };
    Ok(Lowered {
        kernels,
        globals,
        globals_init,
        text,
    })
}

pub fn write_metallib(kernels: &[Kernel]) -> Result<Vec<u8>, String> {
    let names: Vec<CString> = kernels
        .iter()
        .map(|k| CString::new(k.name.clone()).unwrap())
        .collect();
    let functions: Vec<ffi::ZludaMetallibFunction> = kernels
        .iter()
        .zip(names.iter())
        .map(|(k, n)| ffi::ZludaMetallibFunction {
            name: n.as_ptr(),
            air: k.air.as_ptr(),
            air_len: k.air.len(),
        })
        .collect();
    let mut out = ptr::null_mut();
    let mut len = 0usize;
    if unsafe { ffi::zluda_metallib_write(functions.as_ptr(), functions.len(), &mut out, &mut len) }
        != 0
    {
        return Err("failed to write metallib".to_string());
    }
    let data = unsafe { std::slice::from_raw_parts(out, len) }.to_vec();
    unsafe { ffi::zluda_metallib_free(out) };
    Ok(data)
}

/// A function stored in a metallib
#[derive(Clone, Debug)]
pub struct MetallibFunction {
    pub name: String,
    /// 2 for kernels
    pub function_type: u8,
    /// AIR bitcode (without the bitcode wrapper)
    pub air: Vec<u8>,
}

/// Parses a metallib produced by [`write_metallib`] (checks the function hashes)
pub fn read_metallib(data: &[u8]) -> Result<Vec<MetallibFunction>, String> {
    let mut entries = ptr::null_mut();
    let mut count = 0usize;
    let mut error = ptr::null_mut();
    if unsafe {
        ffi::zluda_metallib_read(
            data.as_ptr(),
            data.len(),
            &mut entries,
            &mut count,
            &mut error,
        )
    } != 0
    {
        let msg = unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned();
        unsafe { ffi::zluda_air_free_string(error) };
        return Err(msg);
    }
    let result = (0..count)
        .map(|i| unsafe {
            let e = &*entries.add(i);
            MetallibFunction {
                name: CStr::from_ptr(e.name).to_string_lossy().into_owned(),
                function_type: e.type_,
                air: std::slice::from_raw_parts(e.air, e.air_len).to_vec(),
            }
        })
        .collect();
    unsafe { ffi::zluda_metallib_free_entries(entries, count) };
    Ok(result)
}
