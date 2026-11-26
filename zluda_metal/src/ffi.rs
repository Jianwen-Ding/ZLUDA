//! Bindings to the C++ parts (cpp/air_lower.h, cpp/metallib.h)
#![allow(non_camel_case_types, dead_code)]
use std::os::raw::{c_char, c_int};

#[repr(C)]
pub struct ZludaAirInput {
    pub main_bc: *const u8,
    pub main_len: usize,
    pub attributes_bc: *const u8,
    pub attributes_len: usize,
    pub impl_bc: *const u8,
    pub impl_len: usize,
    pub want_text: c_int,
    pub portable_air: c_int,
}

#[repr(C)]
pub struct ZludaAirKernel {
    pub name: *mut c_char,
    pub param_count: usize,
    pub param_offsets: *mut u64,
    pub param_sizes: *mut u64,
    pub params_size: u64,
    pub dynamic_shared: c_int,
    pub static_shared: u64,
    pub air: *mut u8,
    pub air_len: usize,
    pub globals_param_offset: u64,
}

#[repr(C)]
pub struct ZludaAirGlobal {
    pub name: *mut c_char,
    pub offset: u64,
    pub size: u64,
}

#[repr(C)]
pub struct ZludaAirOutput {
    pub kernels: *mut ZludaAirKernel,
    pub kernel_count: usize,
    pub globals: *mut ZludaAirGlobal,
    pub global_count: usize,
    pub globals_init: *mut u8,
    pub globals_size: u64,
    pub module_text: *mut c_char,
}

#[repr(C)]
pub struct ZludaMetallibFunction {
    pub name: *const c_char,
    pub air: *const u8,
    pub air_len: usize,
}

#[repr(C)]
pub struct ZludaMetallibEntry {
    pub name: *mut c_char,
    pub type_: u8,
    pub air: *const u8,
    pub air_len: usize,
    pub hash: [u8; 32],
}

extern "C" {
    pub fn zluda_metallib_read(
        data: *const u8,
        len: usize,
        entries: *mut *mut ZludaMetallibEntry,
        count: *mut usize,
        error: *mut *mut c_char,
    ) -> c_int;
    pub fn zluda_metallib_free_entries(entries: *mut ZludaMetallibEntry, count: usize);
    pub fn zluda_air_lower(
        input: *const ZludaAirInput,
        output: *mut ZludaAirOutput,
        error: *mut *mut c_char,
    ) -> c_int;
    pub fn zluda_air_free(output: *mut ZludaAirOutput);
    pub fn zluda_air_free_string(s: *mut c_char);
    pub fn zluda_metallib_write(
        functions: *const ZludaMetallibFunction,
        count: usize,
        out: *mut *mut u8,
        out_len: *mut usize,
    ) -> c_int;
    pub fn zluda_metallib_free(data: *mut u8);
}
