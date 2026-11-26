//! ZLUDA on Metal: a PTX -> AIR -> metallib compiler and a CUDA driver API
//! implementation on top of metal-cpp.
//!
//! Build with `ZLUDA_SYSTEM_LLVM_CONFIG=llvm-config-18 cargo build -p zluda_metal`;
//! the resulting `libzluda_metal.so` (or `.dylib`) exports the CUDA driver API
//! and can be used as `libcuda.so.1`.
pub mod agx;
mod backend;
pub mod compile;
pub mod cuda;
mod ffi;
