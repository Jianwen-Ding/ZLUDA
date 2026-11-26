//! CUDA driver API (the libcuda.so surface) implemented on Metal.
//!
//! One device (the default Metal device), one in-order queue: all CUDA streams
//! map onto it, which is a valid (if conservative) implementation of stream
//! semantics. Device pointers are Metal GPU addresses of shared-storage
//! buffers, so kernels dereference them directly.
#![allow(non_snake_case, non_camel_case_types, clippy::missing_safety_doc)]

use crate::backend;
use crate::compile;
use std::collections::HashMap;
use std::ffi::{c_void, CStr};
use std::os::raw::{c_char, c_int, c_uint};
use std::ptr;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub type CUresult = c_uint;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
pub const CUDA_ERROR_NO_DEVICE: CUresult = 100;
pub const CUDA_ERROR_INVALID_DEVICE: CUresult = 101;
pub const CUDA_ERROR_INVALID_IMAGE: CUresult = 200;
pub const CUDA_ERROR_INVALID_CONTEXT: CUresult = 201;
pub const CUDA_ERROR_NO_BINARY_FOR_GPU: CUresult = 209;
pub const CUDA_ERROR_INVALID_PTX: CUresult = 218;
pub const CUDA_ERROR_FILE_NOT_FOUND: CUresult = 301;
pub const CUDA_ERROR_INVALID_HANDLE: CUresult = 400;
pub const CUDA_ERROR_NOT_FOUND: CUresult = 500;
pub const CUDA_ERROR_NOT_READY: CUresult = 600;
pub const CUDA_ERROR_ILLEGAL_ADDRESS: CUresult = 700;
pub const CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: CUresult = 701;
pub const CUDA_ERROR_LAUNCH_FAILED: CUresult = 719;
pub const CUDA_ERROR_NOT_SUPPORTED: CUresult = 801;
pub const CUDA_ERROR_UNKNOWN: CUresult = 999;

/// Compute capability reported to applications
const CC_MAJOR: c_int = 8;
const CC_MINOR: c_int = 0;
const DRIVER_VERSION: c_int = 12080;

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
pub struct Context {
    _private: u8,
}
pub struct Stream {
    _private: u8,
}
pub struct Event {
    time: Mutex<Option<Instant>>,
}

pub struct Module {
    /// `.address_size 32`: each module variable's allocation (name, address, size)
    globals32: Vec<(String, CUdeviceptr, usize)>,
    compiled: compile::CompiledModule,
    /// device-globals buffer (0 if the module has no writable device variables)
    globals: CUdeviceptr,
    library: backend::Library,
    functions: Mutex<HashMap<String, Box<Function>>>,
}

pub struct Function {
    /// `.address_size 32`: the source parameters and the hidden arguments' values
    /// (64-bit base, then each module variable's 32-bit address)
    explicit_params: Option<usize>,
    hidden32: Vec<u64>,
    info: compile::Kernel,
    kernel: backend::Kernel,
    globals: CUdeviceptr,
}

struct Global {
    device: backend::DeviceInfo,
    primary: Box<Context>,
    last_error: Mutex<String>,
}

static GLOBAL: OnceLock<Result<Global, String>> = OnceLock::new();

thread_local! {
    static CURRENT: std::cell::Cell<*mut Context> = const { std::cell::Cell::new(ptr::null_mut()) };
}

fn global() -> Result<&'static Global, CUresult> {
    match GLOBAL.get() {
        Some(Ok(g)) => Ok(g),
        Some(Err(_)) => Err(CUDA_ERROR_NO_DEVICE),
        None => Err(CUDA_ERROR_NOT_INITIALIZED),
    }
}

fn report(err: String, code: CUresult) -> CUresult {
    if std::env::var_os("ZLUDA_METAL_LOG").is_some() {
        eprintln!("[zluda metal] {err}");
    }
    if let Ok(g) = global() {
        *g.last_error.lock().unwrap() = err;
    }
    code
}

macro_rules! tri {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(code) => return code,
        }
    };
}

fn ensure_current() -> Result<&'static Global, CUresult> {
    let g = global()?;
    CURRENT.with(|c| {
        if c.get().is_null() {
            // the runtime API makes the primary context current implicitly
            c.set(&*g.primary as *const Context as *mut Context);
        }
    });
    Ok(g)
}

// ---------------------------------------------------------------------------
// Initialization and device management
// ---------------------------------------------------------------------------
#[no_mangle]
pub unsafe extern "system" fn cuInit(_flags: c_uint) -> CUresult {
    let g = GLOBAL.get_or_init(|| {
        backend::init().map(|device| Global {
            device,
            primary: Box::new(Context { _private: 0 }),
            last_error: Mutex::new(String::new()),
        })
    });
    match g {
        Ok(_) => CUDA_SUCCESS,
        Err(e) => {
            if std::env::var_os("ZLUDA_METAL_LOG").is_some() {
                eprintln!("[zluda metal] cuInit failed: {e}");
            }
            CUDA_ERROR_NO_DEVICE
        }
    }
}

#[no_mangle]
pub unsafe extern "system" fn cuDriverGetVersion(version: *mut c_int) -> CUresult {
    if version.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *version = DRIVER_VERSION;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDeviceGetCount(count: *mut c_int) -> CUresult {
    tri!(global());
    if count.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *count = 1;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult {
    tri!(global());
    if device.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if ordinal != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *device = 0;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDeviceGetName(
    name: *mut c_char,
    len: c_int,
    dev: CUdevice,
) -> CUresult {
    let g = tri!(global());
    if dev != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if name.is_null() || len <= 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let bytes = g.device.name.as_bytes();
    let n = bytes.len().min(len as usize - 1);
    ptr::copy_nonoverlapping(bytes.as_ptr(), name as *mut u8, n);
    *name.add(n) = 0;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult {
    let g = tri!(global());
    if dev != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *bytes = g.device.total_memory as usize;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDeviceComputeCapability(
    major: *mut c_int,
    minor: *mut c_int,
    dev: CUdevice,
) -> CUresult {
    tri!(global());
    if dev != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *major = CC_MAJOR;
    *minor = CC_MINOR;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDeviceGetAttribute(
    value: *mut c_int,
    attrib: c_int,
    dev: CUdevice,
) -> CUresult {
    let g = tri!(global());
    if dev != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if value.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let max_threads = g.device.max_threads_per_threadgroup.min(1024) as c_int;
    let shared = g.device.max_threadgroup_memory.min(i32::MAX as u64) as c_int;
    *value = match attrib {
        1 => max_threads,                       // MAX_THREADS_PER_BLOCK
        2 | 3 => max_threads,                   // MAX_BLOCK_DIM_X/Y
        4 => 64,                                // MAX_BLOCK_DIM_Z
        5 => i32::MAX,                          // MAX_GRID_DIM_X
        6 | 7 => 65535,                         // MAX_GRID_DIM_Y/Z
        8 => shared,                            // MAX_SHARED_MEMORY_PER_BLOCK
        9 => 65536,                             // TOTAL_CONSTANT_MEMORY
        10 => g.device.simd_width as c_int,     // WARP_SIZE
        11 => i32::MAX,                         // MAX_PITCH
        12 => 65536,                            // MAX_REGISTERS_PER_BLOCK
        13 => 1_398_000,                        // CLOCK_RATE (kHz)
        14 => 256,                              // TEXTURE_ALIGNMENT
        15 => 0,                                // GPU_OVERLAP
        16 => 8,                                // MULTIPROCESSOR_COUNT
        17 => 0,                                // KERNEL_EXEC_TIMEOUT
        18 => g.device.unified_memory as c_int, // INTEGRATED
        19 => 1,                                // CAN_MAP_HOST_MEMORY
        20 => 0,                                // COMPUTE_MODE (default)
        31 => 0,                                // CONCURRENT_KERNELS
        32 => 0,                                // ECC_ENABLED
        33..=35 => 0,                           // PCI bus/device, TCC driver
        36 => 3_200_000,                        // MEMORY_CLOCK_RATE
        37 => 128,                              // GLOBAL_MEMORY_BUS_WIDTH
        38 => 8 << 20,                          // L2_CACHE_SIZE
        39 => max_threads,                      // MAX_THREADS_PER_MULTIPROCESSOR
        40 => 1,                                // ASYNC_ENGINE_COUNT
        41 => 1,                                // UNIFIED_ADDRESSING
        75 => CC_MAJOR,                         // COMPUTE_CAPABILITY_MAJOR
        76 => CC_MINOR,                         // COMPUTE_CAPABILITY_MINOR
        81 => shared,                           // MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        82 => 65536,                            // MAX_REGISTERS_PER_MULTIPROCESSOR
        83 => 0,                                // MANAGED_MEMORY
        84 => 0,                                // MULTI_GPU_BOARD
        97 => shared,                           // MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        106 => 32,                              // MAX_BLOCKS_PER_MULTIPROCESSOR
        0 | 200.. => return CUDA_ERROR_INVALID_VALUE,
        _ => 0,
    };
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDeviceGetUuid(uuid: *mut [u8; 16], dev: CUdevice) -> CUresult {
    tri!(global());
    if dev != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *uuid = *b"ZLUDA-METAL-0000";
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Contexts
// ---------------------------------------------------------------------------
#[no_mangle]
pub unsafe extern "system" fn cuDevicePrimaryCtxRetain(
    ctx: *mut *mut Context,
    dev: CUdevice,
) -> CUresult {
    let g = tri!(global());
    if dev != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *ctx = &*g.primary as *const Context as *mut Context;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDevicePrimaryCtxRelease_v2(_dev: CUdevice) -> CUresult {
    tri!(global());
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDevicePrimaryCtxRelease(dev: CUdevice) -> CUresult {
    cuDevicePrimaryCtxRelease_v2(dev)
}

#[no_mangle]
pub unsafe extern "system" fn cuCtxCreate_v2(
    ctx: *mut *mut Context,
    _flags: c_uint,
    dev: CUdevice,
) -> CUresult {
    tri!(global());
    if dev != 0 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    let c = Box::into_raw(Box::new(Context { _private: 0 }));
    *ctx = c;
    CURRENT.with(|cur| cur.set(c));
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuCtxDestroy_v2(ctx: *mut Context) -> CUresult {
    let g = tri!(global());
    if ctx.is_null() {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    let _ = backend::synchronize();
    CURRENT.with(|cur| {
        if cur.get() == ctx {
            cur.set(ptr::null_mut());
        }
    });
    if ctx != &*g.primary as *const Context as *mut Context {
        drop(Box::from_raw(ctx));
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuCtxSetCurrent(ctx: *mut Context) -> CUresult {
    tri!(global());
    CURRENT.with(|cur| cur.set(ctx));
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuCtxGetCurrent(ctx: *mut *mut Context) -> CUresult {
    tri!(global());
    *ctx = CURRENT.with(|cur| cur.get());
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuCtxPushCurrent_v2(ctx: *mut Context) -> CUresult {
    cuCtxSetCurrent(ctx)
}

#[no_mangle]
pub unsafe extern "system" fn cuCtxPopCurrent_v2(ctx: *mut *mut Context) -> CUresult {
    tri!(global());
    if !ctx.is_null() {
        *ctx = CURRENT.with(|cur| cur.get());
    }
    CURRENT.with(|cur| cur.set(ptr::null_mut()));
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuCtxGetDevice(dev: *mut CUdevice) -> CUresult {
    tri!(ensure_current());
    *dev = 0;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuCtxSynchronize() -> CUresult {
    tri!(ensure_current());
    match backend::synchronize() {
        Ok(()) => CUDA_SUCCESS,
        Err(e) => report(e, CUDA_ERROR_LAUNCH_FAILED),
    }
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------
#[no_mangle]
pub unsafe extern "system" fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, size: usize) -> CUresult {
    tri!(ensure_current());
    if dptr.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    match backend::alloc(size as u64) {
        Ok(a) => {
            *dptr = a;
            CUDA_SUCCESS
        }
        Err(e) => report(e, CUDA_ERROR_OUT_OF_MEMORY),
    }
}

#[no_mangle]
pub unsafe extern "system" fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult {
    tri!(ensure_current());
    if dptr == 0 {
        return CUDA_SUCCESS;
    }
    if backend::free(dptr) {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_VALUE
    }
}

#[no_mangle]
pub unsafe extern "system" fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult {
    let g = tri!(ensure_current());
    *free = g.device.total_memory as usize;
    *total = g.device.total_memory as usize;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuMemGetAddressRange_v2(
    base: *mut CUdeviceptr,
    size: *mut usize,
    dptr: CUdeviceptr,
) -> CUresult {
    tri!(ensure_current());
    match backend::range(dptr) {
        Some((b, len)) => {
            if !base.is_null() {
                *base = b;
            }
            if !size.is_null() {
                *size = len as usize;
            }
            CUDA_SUCCESS
        }
        None => CUDA_ERROR_NOT_FOUND,
    }
}

#[no_mangle]
pub unsafe extern "system" fn cuMemAllocHost_v2(pp: *mut *mut c_void, size: usize) -> CUresult {
    let layout = std::alloc::Layout::from_size_align(size.max(1) + 64, 64).unwrap();
    let p = std::alloc::alloc(layout);
    if p.is_null() {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    // remember the size in front of the block
    *(p as *mut usize) = size.max(1) + 64;
    *pp = p.add(64) as *mut c_void;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuMemHostAlloc(
    pp: *mut *mut c_void,
    size: usize,
    _flags: c_uint,
) -> CUresult {
    cuMemAllocHost_v2(pp, size)
}

#[no_mangle]
pub unsafe extern "system" fn cuMemFreeHost(p: *mut c_void) -> CUresult {
    if p.is_null() {
        return CUDA_SUCCESS;
    }
    let base = (p as *mut u8).sub(64);
    let size = *(base as *mut usize);
    std::alloc::dealloc(base, std::alloc::Layout::from_size_align(size, 64).unwrap());
    CUDA_SUCCESS
}

unsafe fn device_slice(ptr: CUdeviceptr, len: usize) -> Result<*mut u8, CUresult> {
    match backend::host_pointer(ptr) {
        Some((p, left)) if left as usize >= len => Ok(p),
        _ if len == 0 => Ok(ptr::null_mut()),
        _ => Err(report(
            format!("invalid device range 0x{ptr:x} + {len}"),
            CUDA_ERROR_INVALID_VALUE,
        )),
    }
}

fn sync_for_host_access() -> Result<(), CUresult> {
    backend::synchronize().map_err(|e| report(e, CUDA_ERROR_LAUNCH_FAILED))
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpyHtoD_v2(
    dst: CUdeviceptr,
    src: *const c_void,
    n: usize,
) -> CUresult {
    tri!(ensure_current());
    tri!(sync_for_host_access());
    let d = tri!(device_slice(dst, n));
    if n > 0 {
        ptr::copy(src as *const u8, d, n);
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpyDtoH_v2(
    dst: *mut c_void,
    src: CUdeviceptr,
    n: usize,
) -> CUresult {
    tri!(ensure_current());
    tri!(sync_for_host_access());
    let s = tri!(device_slice(src, n));
    if n > 0 {
        ptr::copy(s, dst as *mut u8, n);
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpyDtoD_v2(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    n: usize,
) -> CUresult {
    tri!(ensure_current());
    tri!(sync_for_host_access());
    let s = tri!(device_slice(src, n));
    let d = tri!(device_slice(dst, n));
    if n > 0 {
        ptr::copy(s, d, n);
    }
    CUDA_SUCCESS
}

/// Unified addressing copy: either side may be host or device memory
#[no_mangle]
pub unsafe extern "system" fn cuMemcpy(dst: CUdeviceptr, src: CUdeviceptr, n: usize) -> CUresult {
    tri!(ensure_current());
    tri!(sync_for_host_access());
    let resolve = |p: CUdeviceptr| match backend::host_pointer(p) {
        Some((h, _)) => h,
        None => p as *mut u8,
    };
    if n > 0 {
        ptr::copy(resolve(src), resolve(dst), n);
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpyHtoDAsync_v2(
    dst: CUdeviceptr,
    src: *const c_void,
    n: usize,
    _stream: *mut Stream,
) -> CUresult {
    cuMemcpyHtoD_v2(dst, src, n)
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpyDtoHAsync_v2(
    dst: *mut c_void,
    src: CUdeviceptr,
    n: usize,
    _stream: *mut Stream,
) -> CUresult {
    cuMemcpyDtoH_v2(dst, src, n)
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpyDtoDAsync_v2(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    n: usize,
    _stream: *mut Stream,
) -> CUresult {
    cuMemcpyDtoD_v2(dst, src, n)
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpyAsync(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    n: usize,
    _stream: *mut Stream,
) -> CUresult {
    cuMemcpy(dst, src, n)
}

unsafe fn memset_impl(dst: CUdeviceptr, elem: &[u8], count: usize) -> CUresult {
    tri!(ensure_current());
    tri!(sync_for_host_access());
    let len = elem.len() * count;
    let d = tri!(device_slice(dst, len));
    for i in 0..count {
        ptr::copy_nonoverlapping(elem.as_ptr(), d.add(i * elem.len()), elem.len());
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuMemsetD8_v2(dst: CUdeviceptr, v: u8, n: usize) -> CUresult {
    memset_impl(dst, &[v], n)
}

#[no_mangle]
pub unsafe extern "system" fn cuMemsetD16_v2(dst: CUdeviceptr, v: u16, n: usize) -> CUresult {
    memset_impl(dst, &v.to_ne_bytes(), n)
}

#[no_mangle]
pub unsafe extern "system" fn cuMemsetD32_v2(dst: CUdeviceptr, v: u32, n: usize) -> CUresult {
    memset_impl(dst, &v.to_ne_bytes(), n)
}

#[no_mangle]
pub unsafe extern "system" fn cuMemsetD8Async(
    dst: CUdeviceptr,
    v: u8,
    n: usize,
    _s: *mut Stream,
) -> CUresult {
    cuMemsetD8_v2(dst, v, n)
}

#[no_mangle]
pub unsafe extern "system" fn cuMemsetD32Async(
    dst: CUdeviceptr,
    v: u32,
    n: usize,
    _s: *mut Stream,
) -> CUresult {
    cuMemsetD32_v2(dst, v, n)
}

// ---------------------------------------------------------------------------
// Modules and functions
// ---------------------------------------------------------------------------
fn dump(name: &str, data: &[u8]) {
    if let Some(dir) = std::env::var_os("ZLUDA_METAL_DUMP_DIR") {
        let dir = std::path::PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join(name), data);
    }
}

unsafe fn load_module_from_ptx(module: *mut *mut Module, ptx: &str) -> CUresult {
    tri!(ensure_current());
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    dump(&format!("module{id}.ptx"), ptx.as_bytes());
    let compiled = match compile::compile_ptx(
        ptx,
        compile::FpMode::from_env().apply(compile::Options {
            portable_air: backend::use_agx(),
            ..Default::default()
        }),
    ) {
        Ok(c) => c,
        Err(e) => {
            return report(
                format!("PTX compilation failed: {e}"),
                CUDA_ERROR_INVALID_PTX,
            )
        }
    };
    dump(&format!("module{id}.metallib"), &compiled.metallib);
    for k in &compiled.kernels {
        dump(&format!("module{id}.{}.air", k.name), &k.air);
    }
    let library = match backend::Library::new(&compiled.metallib) {
        Ok(l) => l,
        Err(e) => return report(e, CUDA_ERROR_INVALID_IMAGE),
    };
    let mut globals = 0;
    if !compiled.globals.is_empty() {
        globals = match backend::alloc(compiled.globals_init.len() as u64) {
            Ok(a) => a,
            Err(e) => return report(e, CUDA_ERROR_OUT_OF_MEMORY),
        };
        let (host, _) = backend::host_pointer(globals).unwrap();
        ptr::copy_nonoverlapping(
            compiled.globals_init.as_ptr(),
            host,
            compiled.globals_init.len(),
        );
    }
    let mut globals32 = vec![];
    if let Some(m32) = &compiled.module32 {
        for (name, init, _align) in &m32.globals {
            let a = match backend::alloc(init.len().max(1) as u64) {
                Ok(a) => a,
                Err(e) => return report(e, CUDA_ERROR_OUT_OF_MEMORY),
            };
            if a + init.len() as u64 > u32::MAX as u64 {
                return report(
                    "a .address_size 32 module needs its memory below 4 GiB".into(),
                    CUDA_ERROR_OUT_OF_MEMORY,
                );
            }
            if let Some((host, _)) = backend::host_pointer(a) {
                ptr::copy_nonoverlapping(init.as_ptr(), host, init.len());
            }
            globals32.push((name.clone(), a, init.len()));
        }
    }
    *module = Box::into_raw(Box::new(Module {
        globals32,
        compiled,
        globals,
        library,
        functions: Mutex::new(HashMap::new()),
    }));
    CUDA_SUCCESS
}

unsafe fn image_to_ptx<'a>(image: *const c_void) -> Result<&'a str, CUresult> {
    if image.is_null() {
        return Err(CUDA_ERROR_INVALID_VALUE);
    }
    let bytes = image as *const u8;
    // ELF cubins and fatbins carry machine code for NVIDIA GPUs only
    let head = std::slice::from_raw_parts(bytes, 4);
    if head == b"\x7fELF" || head == [0x50, 0xed, 0x55, 0xba] || head == [0x46, 0x62, 0x31, 0x45] {
        return Err(report(
            "binary (cubin/fatbin) modules are not supported, pass PTX text".to_string(),
            CUDA_ERROR_NO_BINARY_FOR_GPU,
        ));
    }
    CStr::from_ptr(image as *const c_char)
        .to_str()
        .map_err(|_| CUDA_ERROR_INVALID_IMAGE)
}

#[no_mangle]
pub unsafe extern "system" fn cuModuleLoadData(
    module: *mut *mut Module,
    image: *const c_void,
) -> CUresult {
    if module.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let ptx = tri!(image_to_ptx(image));
    load_module_from_ptx(module, ptx)
}

#[no_mangle]
pub unsafe extern "system" fn cuModuleLoadDataEx(
    module: *mut *mut Module,
    image: *const c_void,
    _num_options: c_uint,
    _options: *mut c_int,
    _values: *mut *mut c_void,
) -> CUresult {
    cuModuleLoadData(module, image)
}

#[no_mangle]
pub unsafe extern "system" fn cuModuleLoad(
    module: *mut *mut Module,
    fname: *const c_char,
) -> CUresult {
    if module.is_null() || fname.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let path = CStr::from_ptr(fname).to_string_lossy().into_owned();
    let mut data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(_) => return CUDA_ERROR_FILE_NOT_FOUND,
    };
    data.push(0);
    cuModuleLoadData(module, data.as_ptr().cast())
}

#[no_mangle]
pub unsafe extern "system" fn cuModuleUnload(module: *mut Module) -> CUresult {
    if module.is_null() {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    let _ = backend::synchronize();
    let m = Box::from_raw(module);
    if m.globals != 0 {
        backend::free(m.globals);
    }
    for g in &m.globals32 {
        backend::free(g.1);
    }
    drop(m);
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuModuleGetFunction(
    func: *mut *mut Function,
    module: *mut Module,
    name: *const c_char,
) -> CUresult {
    if func.is_null() || module.is_null() || name.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let m = &*module;
    let name = CStr::from_ptr(name).to_string_lossy().into_owned();
    let mut functions = m.functions.lock().unwrap();
    if let Some(f) = functions.get(&name) {
        *func = &**f as *const Function as *mut Function;
        return CUDA_SUCCESS;
    }
    let Some(info) = m.compiled.kernels.iter().find(|k| k.name == name) else {
        return CUDA_ERROR_NOT_FOUND;
    };
    let kernel = match m.library.kernel(&name) {
        Ok(k) => k,
        Err(e) => return report(e, CUDA_ERROR_INVALID_IMAGE),
    };
    let explicit_params = m
        .compiled
        .module32
        .as_ref()
        .and_then(|m32| m32.explicit_params.get(&name).copied());
    // 32-bit addresses are the allocations' own (the AGX backend allocates below 4 GiB): base 0
    let hidden32 = if explicit_params.is_some() {
        std::iter::once(0u64).chain(m.globals32.iter().map(|g| g.1)).collect()
    } else {
        vec![]
    };
    let f = Box::new(Function {
        explicit_params,
        hidden32,
        info: info.clone(),
        kernel,
        globals: m.globals,
    });
    *func = &*f as *const Function as *mut Function;
    functions.insert(name, f);
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuModuleGetGlobal_v2(
    dptr: *mut CUdeviceptr,
    bytes: *mut usize,
    module: *mut Module,
    name: *const c_char,
) -> CUresult {
    if module.is_null() || name.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let m = &*module;
    let name = CStr::from_ptr(name).to_string_lossy();
    if let Some(g) = m.globals32.iter().find(|g| g.0 == name) {
        if !dptr.is_null() {
            *dptr = g.1;
        }
        if !bytes.is_null() {
            *bytes = g.2;
        }
        return CUDA_SUCCESS;
    }
    match m.compiled.globals.iter().find(|g| g.name == name) {
        Some(g) => {
            if !dptr.is_null() {
                *dptr = m.globals + g.offset;
            }
            if !bytes.is_null() {
                *bytes = g.size as usize;
            }
            CUDA_SUCCESS
        }
        None => report(
            format!("no writable device variable named {name} (constant data is not addressable on Metal)"),
            CUDA_ERROR_NOT_FOUND,
        ),
    }
}

#[no_mangle]
pub unsafe extern "system" fn cuFuncGetAttribute(
    value: *mut c_int,
    attrib: c_int,
    func: *mut Function,
) -> CUresult {
    if value.is_null() || func.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let f = &*func;
    *value = match attrib {
        0 => f.kernel.max_threads.min(1024) as c_int, // MAX_THREADS_PER_BLOCK
        1 => (f.kernel.static_shared.max(f.info.static_shared)) as c_int, // SHARED_SIZE_BYTES
        2 => 0,                                       // CONST_SIZE_BYTES
        3 => 0,                                       // LOCAL_SIZE_BYTES
        4 => 32,                                      // NUM_REGS
        5 | 6 => CC_MAJOR * 10 + CC_MINOR,            // PTX_VERSION / BINARY_VERSION
        7 => 0,                                       // CACHE_MODE_CA
        8 => 32768,                                   // MAX_DYNAMIC_SHARED_SIZE_BYTES
        9 => 0,                                       // PREFERRED_SHARED_MEMORY_CARVEOUT
        _ => return CUDA_ERROR_INVALID_VALUE,
    };
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuFuncSetAttribute(
    _func: *mut Function,
    _attrib: c_int,
    _value: c_int,
) -> CUresult {
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuFuncSetCacheConfig(
    _func: *mut Function,
    _config: c_int,
) -> CUresult {
    CUDA_SUCCESS
}

const CU_LAUNCH_PARAM_END: usize = 0;
const CU_LAUNCH_PARAM_BUFFER_POINTER: usize = 1;
const CU_LAUNCH_PARAM_BUFFER_SIZE: usize = 2;

#[no_mangle]
pub unsafe extern "system" fn cuLaunchKernel(
    func: *mut Function,
    grid_x: c_uint,
    grid_y: c_uint,
    grid_z: c_uint,
    block_x: c_uint,
    block_y: c_uint,
    block_z: c_uint,
    shared_mem_bytes: c_uint,
    _stream: *mut Stream,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    let g = tri!(ensure_current());
    if func.is_null() {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    let f = &*func;
    let threads = block_x as u64 * block_y as u64 * block_z as u64;
    if threads == 0 || grid_x == 0 || grid_y == 0 || grid_z == 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if threads > f.kernel.max_threads || threads > g.device.max_threads_per_threadgroup {
        return report(
            format!(
                "block of {threads} threads exceeds the kernel limit {}",
                f.kernel.max_threads
            ),
            CUDA_ERROR_INVALID_VALUE,
        );
    }
    if f.info.static_shared + shared_mem_bytes as u64 > g.device.max_threadgroup_memory {
        return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    }
    let mut params = vec![0u8; f.info.params_size as usize];
    let explicit = f.explicit_params.unwrap_or(f.info.params.len());
    for (k, v) in f.hidden32.iter().enumerate() {
        if let Some(p) = f.info.params.get(explicit + k) {
            let b = v.to_le_bytes();
            params[p.offset as usize..p.offset as usize + p.size as usize]
                .copy_from_slice(&b[..p.size as usize]);
        }
    }
    if !kernel_params.is_null() {
        for (i, p) in f.info.params.iter().enumerate().take(explicit) {
            let src = *kernel_params.add(i) as *const u8;
            if src.is_null() {
                return CUDA_ERROR_INVALID_VALUE;
            }
            ptr::copy_nonoverlapping(
                src,
                params.as_mut_ptr().add(p.offset as usize),
                p.size as usize,
            );
        }
    } else if !extra.is_null() {
        let mut buffer: *const u8 = ptr::null();
        let mut size: usize = 0;
        let mut i = 0;
        loop {
            let key = *extra.add(i) as usize;
            if key == CU_LAUNCH_PARAM_END {
                break;
            }
            let val = *extra.add(i + 1);
            match key {
                CU_LAUNCH_PARAM_BUFFER_POINTER => buffer = val as *const u8,
                CU_LAUNCH_PARAM_BUFFER_SIZE => size = *(val as *const usize),
                _ => return CUDA_ERROR_INVALID_VALUE,
            }
            i += 2;
        }
        if buffer.is_null() || size < params.len() {
            return CUDA_ERROR_INVALID_VALUE;
        }
        ptr::copy_nonoverlapping(buffer, params.as_mut_ptr(), params.len());
    } else if !f.info.params.is_empty() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if let Some(off) = f.info.globals_param_offset {
        params[off as usize..off as usize + 8].copy_from_slice(&f.globals.to_ne_bytes());
    }
    if let Some(off) = f.info.dyn_smem_param_offset {
        params[off as usize..off as usize + 4].copy_from_slice(&shared_mem_bytes.to_ne_bytes());
    }
    let dynamic_shared = if f.info.dynamic_shared {
        shared_mem_bytes.max(16)
    } else {
        0
    };
    match f.kernel.launch(
        [grid_x, grid_y, grid_z],
        [block_x, block_y, block_z],
        &params,
        dynamic_shared,
    ) {
        Ok(()) => CUDA_SUCCESS,
        Err(e) => report(e, CUDA_ERROR_LAUNCH_FAILED),
    }
}

// ---------------------------------------------------------------------------
// Streams and events (single in-order queue)
// ---------------------------------------------------------------------------
#[no_mangle]
pub unsafe extern "system" fn cuStreamCreate(stream: *mut *mut Stream, _flags: c_uint) -> CUresult {
    tri!(ensure_current());
    *stream = Box::into_raw(Box::new(Stream { _private: 0 }));
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuStreamCreateWithPriority(
    stream: *mut *mut Stream,
    flags: c_uint,
    _priority: c_int,
) -> CUresult {
    cuStreamCreate(stream, flags)
}

#[no_mangle]
pub unsafe extern "system" fn cuStreamDestroy_v2(stream: *mut Stream) -> CUresult {
    if !stream.is_null() {
        let _ = backend::synchronize();
        drop(Box::from_raw(stream));
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuStreamSynchronize(_stream: *mut Stream) -> CUresult {
    cuCtxSynchronize()
}

#[no_mangle]
pub unsafe extern "system" fn cuStreamQuery(_stream: *mut Stream) -> CUresult {
    cuCtxSynchronize()
}

#[no_mangle]
pub unsafe extern "system" fn cuStreamWaitEvent(
    _s: *mut Stream,
    _e: *mut Event,
    _flags: c_uint,
) -> CUresult {
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuEventCreate(event: *mut *mut Event, _flags: c_uint) -> CUresult {
    *event = Box::into_raw(Box::new(Event {
        time: Mutex::new(None),
    }));
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuEventDestroy_v2(event: *mut Event) -> CUresult {
    if !event.is_null() {
        drop(Box::from_raw(event));
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuEventRecord(event: *mut Event, _stream: *mut Stream) -> CUresult {
    if event.is_null() {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    // Events complete when all prior work completed
    let r = cuCtxSynchronize();
    *(*event).time.lock().unwrap() = Some(Instant::now());
    r
}

#[no_mangle]
pub unsafe extern "system" fn cuEventSynchronize(event: *mut Event) -> CUresult {
    if event.is_null() {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuEventQuery(event: *mut Event) -> CUresult {
    cuEventSynchronize(event)
}

#[no_mangle]
pub unsafe extern "system" fn cuEventElapsedTime(
    ms: *mut f32,
    start: *mut Event,
    end: *mut Event,
) -> CUresult {
    if ms.is_null() || start.is_null() || end.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let s = *(*start).time.lock().unwrap();
    let e = *(*end).time.lock().unwrap();
    match (s, e) {
        (Some(s), Some(e)) => {
            *ms = e.saturating_duration_since(s).as_secs_f32() * 1000.0;
            CUDA_SUCCESS
        }
        _ => CUDA_ERROR_NOT_READY,
    }
}

// ---------------------------------------------------------------------------
// Errors and entry point lookup
// ---------------------------------------------------------------------------
fn error_name(e: CUresult) -> Option<&'static CStr> {
    Some(match e {
        CUDA_SUCCESS => c"CUDA_SUCCESS",
        CUDA_ERROR_INVALID_VALUE => c"CUDA_ERROR_INVALID_VALUE",
        CUDA_ERROR_OUT_OF_MEMORY => c"CUDA_ERROR_OUT_OF_MEMORY",
        CUDA_ERROR_NOT_INITIALIZED => c"CUDA_ERROR_NOT_INITIALIZED",
        CUDA_ERROR_NO_DEVICE => c"CUDA_ERROR_NO_DEVICE",
        CUDA_ERROR_INVALID_DEVICE => c"CUDA_ERROR_INVALID_DEVICE",
        CUDA_ERROR_INVALID_IMAGE => c"CUDA_ERROR_INVALID_IMAGE",
        CUDA_ERROR_INVALID_CONTEXT => c"CUDA_ERROR_INVALID_CONTEXT",
        CUDA_ERROR_NO_BINARY_FOR_GPU => c"CUDA_ERROR_NO_BINARY_FOR_GPU",
        CUDA_ERROR_INVALID_PTX => c"CUDA_ERROR_INVALID_PTX",
        CUDA_ERROR_FILE_NOT_FOUND => c"CUDA_ERROR_FILE_NOT_FOUND",
        CUDA_ERROR_INVALID_HANDLE => c"CUDA_ERROR_INVALID_HANDLE",
        CUDA_ERROR_NOT_FOUND => c"CUDA_ERROR_NOT_FOUND",
        CUDA_ERROR_NOT_READY => c"CUDA_ERROR_NOT_READY",
        CUDA_ERROR_ILLEGAL_ADDRESS => c"CUDA_ERROR_ILLEGAL_ADDRESS",
        CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => c"CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
        CUDA_ERROR_LAUNCH_FAILED => c"CUDA_ERROR_LAUNCH_FAILED",
        CUDA_ERROR_NOT_SUPPORTED => c"CUDA_ERROR_NOT_SUPPORTED",
        CUDA_ERROR_UNKNOWN => c"CUDA_ERROR_UNKNOWN",
        _ => return None,
    })
}

#[no_mangle]
pub unsafe extern "system" fn cuGetErrorName(e: CUresult, out: *mut *const c_char) -> CUresult {
    match error_name(e) {
        Some(n) => {
            *out = n.as_ptr();
            CUDA_SUCCESS
        }
        None => CUDA_ERROR_INVALID_VALUE,
    }
}

#[no_mangle]
pub unsafe extern "system" fn cuGetErrorString(e: CUresult, out: *mut *const c_char) -> CUresult {
    cuGetErrorName(e, out)
}

/// Last detailed error message of the Metal backend (ZLUDA extension)
#[no_mangle]
pub unsafe extern "system" fn zludaMetalGetLastError(buf: *mut c_char, len: usize) -> usize {
    let Ok(g) = global() else { return 0 };
    let msg = g.last_error.lock().unwrap().clone();
    if !buf.is_null() && len > 0 {
        let n = msg.len().min(len - 1);
        ptr::copy_nonoverlapping(msg.as_ptr(), buf as *mut u8, n);
        *buf.add(n) = 0;
    }
    msg.len()
}

extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

#[cfg(target_os = "macos")]
const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;
#[cfg(not(target_os = "macos"))]
const RTLD_DEFAULT: *mut c_void = ptr::null_mut();

/// Used by the CUDA runtime (cudart) to find driver entry points
#[no_mangle]
pub unsafe extern "system" fn cuGetProcAddress_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    _cuda_version: c_int,
    _flags: u64,
    status: *mut c_int,
) -> CUresult {
    if symbol.is_null() || pfn.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let base = CStr::from_ptr(symbol).to_string_lossy().into_owned();
    for suffix in ["_v3", "_v2", ""] {
        let name = std::ffi::CString::new(format!("{base}{suffix}")).unwrap();
        let p = dlsym(RTLD_DEFAULT, name.as_ptr());
        if !p.is_null() {
            *pfn = p;
            if !status.is_null() {
                *status = 0;
            }
            return CUDA_SUCCESS;
        }
    }
    *pfn = ptr::null_mut();
    if !status.is_null() {
        *status = 1; // CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND
    }
    CUDA_ERROR_NOT_FOUND
}

#[no_mangle]
pub unsafe extern "system" fn cuGetProcAddress(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cuda_version: c_int,
    flags: u64,
) -> CUresult {
    cuGetProcAddress_v2(symbol, pfn, cuda_version, flags, ptr::null_mut())
}
