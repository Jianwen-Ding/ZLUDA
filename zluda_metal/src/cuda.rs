//! CUDA driver API (the libcuda.so surface) implemented on Metal / the AGX simulator.
//!
//! One device. Contexts form a per-thread stack (`cuda/context.rs`), allocations, modules, streams and events
//! belong to the context that made them and go away with it. Device work of a stream runs in the calling thread
//! when the stream is idle and otherwise on that stream's worker thread, in stream order (`cuda/stream.rs`), so
//! host callbacks, events and cross-stream waits behave as in CUDA. Device pointers are Metal GPU addresses of
//! shared-storage buffers (or the AGX simulator's heap), so kernels dereference them directly.
//!
//! Every declaration here is written from the CUDA Driver API reference's documented semantics; the numeric
//! values of enums and error codes are the documented ABI.
#![allow(non_snake_case, non_camel_case_types, clippy::missing_safety_doc)]

use crate::backend;
use crate::compile;
use std::collections::{HashMap, HashSet};
use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int, c_uint};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

mod context;
mod device;
mod errors;
mod graph;
mod library;
mod memory;
mod occupancy;
mod proc_address;
mod stream;

pub use context::Context;
#[allow(unused_imports)]
pub(crate) use context::current_limit;
pub use stream::{Event, Stream};

pub type CUresult = c_uint;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;

pub use errors::*;

/// Compute capability reported to applications
const CC_MAJOR: c_int = 8;
const CC_MINOR: c_int = 0;
const DRIVER_VERSION: c_int = 12080;

// ---------------------------------------------------------------------------
// Handle registries: a handle is valid while it is registered (CUDA_ERROR_INVALID_HANDLE otherwise)
// ---------------------------------------------------------------------------
pub(crate) struct Handles<T> {
    map: OnceLock<Mutex<HashMap<usize, Arc<T>>>>,
}

impl<T> Handles<T> {
    pub(crate) const fn new() -> Self {
        Handles { map: OnceLock::new() }
    }
    fn m(&self) -> std::sync::MutexGuard<'_, HashMap<usize, Arc<T>>> {
        self.map.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap_or_else(|e| e.into_inner())
    }
    pub(crate) fn insert(&self, v: Arc<T>) -> *mut T {
        let p = Arc::as_ptr(&v) as *mut T;
        self.m().insert(p as usize, v);
        p
    }
    pub(crate) fn get(&self, p: *const T) -> Option<Arc<T>> {
        self.m().get(&(p as usize)).cloned()
    }
    pub(crate) fn remove(&self, p: *const T) -> Option<Arc<T>> {
        self.m().remove(&(p as usize))
    }
    pub(crate) fn all(&self) -> Vec<Arc<T>> {
        self.m().values().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
pub struct Module {
    /// `.address_size 32`: each module variable's allocation (name, address, size)
    globals32: Vec<(String, CUdeviceptr, usize)>,
    compiled: compile::CompiledModule,
    /// device-globals buffer (0 if the module has no writable device variables)
    globals: CUdeviceptr,
    library: backend::Library,
    functions: Mutex<HashMap<String, Box<Function>>>,
    /// the context that loaded it (0: a context-independent library's module, `cuda/library.rs`)
    #[allow(dead_code)]
    ctx: usize,
    /// `.target sm_XY`: XY (CU_FUNC_ATTRIBUTE_PTX_VERSION)
    target: c_int,
}

pub struct Function {
    /// `.address_size 32`: the source parameters and the hidden arguments' values
    /// (64-bit base, then each module variable's 32-bit address)
    explicit_params: Option<usize>,
    hidden32: Vec<u64>,
    info: compile::Kernel,
    kernel: backend::Kernel,
    globals: CUdeviceptr,
    /// the module's trap / assert record (__zluda_fault), if a kernel can trap
    fault: Option<CUdeviceptr>,
    module: *mut Module,
    name: CString,
    /// the library it came from (CUkernel), 0 for a module's function
    library: AtomicU64,
    attrs: Mutex<FuncAttrs>,
}
unsafe impl Send for Function {}
unsafe impl Sync for Function {}

/// The settable function attributes (cuFuncSetAttribute)
#[derive(Clone, Copy)]
pub(crate) struct FuncAttrs {
    max_dynamic_shared: u32,
    carveout: i32,
    cache_config: u32,
    cluster: [u32; 3],
    non_portable_cluster: u32,
    cluster_policy: u32,
}

/// live modules (pointer -> owning context) and functions
static MODULES: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
static FUNCTIONS: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

pub(crate) fn module_valid(m: *mut Module) -> bool {
    MODULES.lock().unwrap().as_ref().is_some_and(|s| s.contains_key(&(m as usize)))
}
pub(crate) fn function_valid(f: *mut Function) -> bool {
    !f.is_null() && FUNCTIONS.lock().unwrap().as_ref().is_some_and(|s| s.contains(&(f as usize)))
}

struct Global {
    device: backend::DeviceInfo,
    primary: Box<Context>,
    last_error: Mutex<String>,
}

static GLOBAL: OnceLock<Result<Global, String>> = OnceLock::new();
/// set by an atexit handler: calls during process teardown return CUDA_ERROR_DEINITIALIZED
static DEINITIALIZED: AtomicBool = AtomicBool::new(false);

fn global() -> Result<&'static Global, CUresult> {
    if DEINITIALIZED.load(Ordering::Relaxed) {
        return Err(CUDA_ERROR_DEINITIALIZED);
    }
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
pub(crate) use tri;

/// The current context must exist and be usable (no sticky fault): CUDA_ERROR_INVALID_CONTEXT without one
fn ensure_current() -> Result<&'static Global, CUresult> {
    context::current()?;
    global()
}

fn check_device(dev: CUdevice) -> Result<&'static Global, CUresult> {
    let g = global()?;
    if dev != 0 {
        return Err(CUDA_ERROR_INVALID_DEVICE);
    }
    Ok(g)
}

/// A fault of a kernel: the current context's sticky error (the first fault's code stays), and the message
fn device_fault(msg: String, code: CUresult) {
    if let Some(ctx) = context::top() {
        let _ = ctx.sticky.compare_exchange(0, code, Ordering::AcqRel, Ordering::Acquire);
    }
    eprintln!("[zluda metal] {msg}");
    report(msg, code);
}

/// The CUDA error of a fault message of the functional model (agx::DEVICE_FAULT)
fn fault_code(msg: &str) -> CUresult {
    if msg.contains("unmapped") || msg.contains("outside the declared memory") || msg.contains("wraps the address") {
        CUDA_ERROR_ILLEGAL_ADDRESS
    } else if msg.contains("step limit") {
        CUDA_ERROR_LAUNCH_TIMEOUT
    } else if msg.contains("stack access beyond") {
        CUDA_ERROR_HARDWARE_STACK_ERROR
    } else if msg.contains("outside the program") || msg.contains("unaligned instruction") {
        CUDA_ERROR_INVALID_PC
    } else if msg.contains("misaligned") {
        CUDA_ERROR_MISALIGNED_ADDRESS
    } else {
        CUDA_ERROR_LAUNCH_FAILED
    }
}

/// A launch's result: a fault of the kernel becomes the context's sticky error (the launch itself succeeds, as an
/// asynchronous CUDA launch does), any other error is the launch's own
pub(crate) fn launch_result(r: Result<(), String>) -> CUresult {
    match r {
        Ok(()) => CUDA_SUCCESS,
        Err(e) => match e.strip_prefix(crate::agx::DEVICE_FAULT) {
            Some(f) => {
                device_fault(format!("kernel fault: {f}"), fault_code(f));
                CUDA_SUCCESS
            }
            None => report(e, CUDA_ERROR_LAUNCH_FAILED),
        },
    }
}

/// Whether the current context has a sticky error (cuBLAS fails its calls then)
pub(crate) fn current_sticky() -> CUresult {
    context::top().map_or(0, |c| c.sticky.load(Ordering::Acquire))
}

/// A NUL-terminated string at a device (generic) address, at most 512 bytes
fn device_string(addr: u64) -> String {
    match backend::host_pointer(addr) {
        Some((p, n)) => {
            let b = unsafe { std::slice::from_raw_parts(p, (n as usize).min(512)) };
            let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
            String::from_utf8_lossy(&b[..end]).into_owned()
        }
        None => format!("<0x{addr:x}>"),
    }
}

/// After a launch: the kernel's trap / assert record (__zluda_fault, air_lower.cpp faultReporting), read and cleared
fn check_fault_record(addr: Option<CUdeviceptr>, kernel: &str) {
    let Some(a) = addr else { return };
    let Some((p, n)) = backend::host_pointer(a) else { return };
    if n < 32 {
        return;
    }
    let w = unsafe { std::slice::from_raw_parts_mut(p as *mut u32, 8) };
    let flags = w[0];
    if flags == 0 {
        return;
    }
    let q = |i: usize| w[i] as u64 | (w[i + 1] as u64) << 32;
    let (line, msg, file, func) = (w[1], q(2), q(4), q(6));
    w.fill(0);
    if flags & 2 != 0 {
        // as CUDA's device assert prints it
        device_fault(format!("{}:{}: {}: Assertion `{}` failed. (kernel {kernel})", device_string(file), line,
                             device_string(func), device_string(msg)), CUDA_ERROR_ASSERT);
    } else {
        device_fault(format!("kernel {kernel} executed trap"), CUDA_ERROR_LAUNCH_FAILED);
    }
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
extern "C" {
    fn atexit(cb: extern "C" fn()) -> c_int;
}

extern "C" fn mark_deinitialized() {
    DEINITIALIZED.store(true, Ordering::Relaxed);
}

/// cuInit: "Currently, the Flags parameter must be 0" (CUDA_ERROR_INVALID_VALUE otherwise)
#[no_mangle]
pub unsafe extern "system" fn cuInit(flags: c_uint) -> CUresult {
    if flags != 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if DEINITIALIZED.load(Ordering::Relaxed) {
        return CUDA_ERROR_DEINITIALIZED;
    }
    let mut first = false;
    let g = GLOBAL.get_or_init(|| {
        first = true;
        backend::init().map(|device| Global {
            device,
            primary: Box::new(Context::new(true, 0)),
            last_error: Mutex::new(String::new()),
        })
    });
    match g {
        Ok(g) => {
            if first {
                context::register(&*g.primary);
                // the driver is torn down at exit: later calls (from other atexit handlers or destructors)
                // return CUDA_ERROR_DEINITIALIZED
                atexit(mark_deinitialized);
            }
            CUDA_SUCCESS
        }
        Err(e) => {
            if std::env::var_os("ZLUDA_METAL_LOG").is_some() {
                eprintln!("[zluda metal] cuInit failed: {e}");
            }
            CUDA_ERROR_NO_DEVICE
        }
    }
}

/// cuDriverGetVersion works before cuInit
#[no_mangle]
pub unsafe extern "system" fn cuDriverGetVersion(version: *mut c_int) -> CUresult {
    if version.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *version = DRIVER_VERSION;
    CUDA_SUCCESS
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

/// `.target sm_XY[a|f], ...` -> XY
fn ptx_target(ptx: &str) -> c_int {
    ptx.lines()
        .find_map(|l| l.trim().strip_prefix(".target"))
        .and_then(|t| t.split(|c: char| c == ',' || c.is_whitespace()).find_map(|w| w.strip_prefix("sm_")))
        .and_then(|v| v.trim_end_matches(|c: char| !c.is_ascii_digit()).parse().ok())
        .unwrap_or(CC_MAJOR * 10 + CC_MINOR)
}

unsafe fn load_module_from_ptx(module: *mut *mut Module, ptx: &str) -> CUresult {
    let ctx = tri!(context::current()) as *const Context as usize;
    load_module(module, ptx, ctx)
}

/// Load a module owned by context `ctx` (0: a library's, context independent)
unsafe fn load_module(module: *mut *mut Module, ptx: &str, ctx: usize) -> CUresult {
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
        // addresses of variables in initializers (PTX ISA 5.4.4)
        for r in &compiled.globals_relocs {
            let mut v = globals.wrapping_add(r.addend);
            if r.mask != 0 {
                v = (v & r.mask) >> r.mask.trailing_zeros();
            }
            let bytes = v.to_le_bytes();
            ptr::copy_nonoverlapping(bytes.as_ptr(), host.add(r.at as usize), (r.size as usize).min(8));
        }
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
    let m = Box::into_raw(Box::new(Module {
        globals32,
        compiled,
        globals,
        library,
        functions: Mutex::new(HashMap::new()),
        ctx,
        target: ptx_target(ptx),
    }));
    MODULES.lock().unwrap().get_or_insert_with(HashMap::new).insert(m as usize, ctx);
    *module = m;
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
    tri!(ensure_current());
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
    num_options: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> CUresult {
    if num_options > 0 && (options.is_null() || values.is_null()) {
        tri!(global());
        return CUDA_ERROR_INVALID_VALUE;
    }
    cuModuleLoadData(module, image)
}

/// cuModuleLoadFatBinary: a fat binary holds machine code for NVIDIA GPUs (CUDA_ERROR_NO_BINARY_FOR_GPU); PTX
/// text is accepted as by cuModuleLoadData
#[no_mangle]
pub unsafe extern "system" fn cuModuleLoadFatBinary(module: *mut *mut Module, image: *const c_void) -> CUresult {
    cuModuleLoadData(module, image)
}

#[no_mangle]
pub unsafe extern "system" fn cuModuleLoad(
    module: *mut *mut Module,
    fname: *const c_char,
) -> CUresult {
    tri!(ensure_current());
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

/// Free a module and its functions (cuModuleUnload, a context's or library's teardown)
pub(crate) unsafe fn unload_module(module: *mut Module) -> bool {
    if MODULES.lock().unwrap().as_mut().and_then(|s| s.remove(&(module as usize))).is_none() {
        return false;
    }
    stream::wait_all_idle();
    let m = Box::from_raw(module);
    if let Some(fs) = FUNCTIONS.lock().unwrap().as_mut() {
        for f in m.functions.lock().unwrap().values() {
            fs.remove(&(&**f as *const Function as usize));
        }
    }
    if m.globals != 0 {
        backend::free(m.globals);
    }
    for g in &m.globals32 {
        backend::free(g.1);
    }
    drop(m);
    true
}

/// the modules a context loaded
pub(crate) fn modules_of(ctx: usize) -> Vec<*mut Module> {
    MODULES.lock().unwrap().as_ref().map_or(vec![], |s| {
        s.iter().filter(|(_, &c)| c == ctx).map(|(&m, _)| m as *mut Module).collect()
    })
}

#[no_mangle]
pub unsafe extern "system" fn cuModuleUnload(module: *mut Module) -> CUresult {
    tri!(global());
    if module.is_null() || !unload_module(module) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    CUDA_SUCCESS
}

/// The function named `name` of module `m`, created on first use
pub(crate) unsafe fn module_function(m: *mut Module, name: &str) -> Result<*mut Function, CUresult> {
    let mp = m;
    let m = &*m;
    let mut functions = m.functions.lock().unwrap();
    if let Some(f) = functions.get(name) {
        return Ok(&**f as *const Function as *mut Function);
    }
    let Some(info) = m.compiled.kernels.iter().find(|k| k.name == name) else {
        return Err(CUDA_ERROR_NOT_FOUND);
    };
    let kernel = match m.library.kernel(name) {
        Ok(k) => k,
        Err(e) => return Err(report(e, CUDA_ERROR_INVALID_IMAGE)),
    };
    let explicit_params = m
        .compiled
        .module32
        .as_ref()
        .and_then(|m32| m32.explicit_params.get(name).copied());
    // 32-bit addresses are the allocations' own (the AGX backend allocates below 4 GiB): base 0
    let hidden32 = if explicit_params.is_some() {
        std::iter::once(0u64).chain(m.globals32.iter().map(|g| g.1)).collect()
    } else {
        vec![]
    };
    let static_shared = kernel.static_shared.max(info.static_shared) as u32;
    let f = Box::new(Function {
        explicit_params,
        hidden32,
        info: info.clone(),
        kernel,
        globals: m.globals,
        fault: m.compiled.globals.iter().find(|g| g.name == "__zluda_fault").map(|g| m.globals + g.offset),
        module: mp,
        name: CString::new(name).unwrap_or_default(),
        library: AtomicU64::new(0),
        attrs: Mutex::new(FuncAttrs {
            max_dynamic_shared: device::SHARED_PER_BLOCK.saturating_sub(static_shared),
            carveout: -1,
            cache_config: 0,
            cluster: [0; 3],
            non_portable_cluster: 0,
            cluster_policy: 0,
        }),
    });
    let p = &*f as *const Function as *mut Function;
    FUNCTIONS.lock().unwrap().get_or_insert_with(HashSet::new).insert(p as usize);
    functions.insert(name.to_string(), f);
    Ok(p)
}

/// cuModuleGetFunction: CUDA_ERROR_INVALID_VALUE for a NULL output or name, CUDA_ERROR_INVALID_HANDLE for an
/// invalid module, CUDA_ERROR_NOT_FOUND for an unknown name
#[no_mangle]
pub unsafe extern "system" fn cuModuleGetFunction(
    func: *mut *mut Function,
    module: *mut Module,
    name: *const c_char,
) -> CUresult {
    tri!(global());
    if func.is_null() || name.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if !module_valid(module) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    let name = CStr::from_ptr(name).to_string_lossy().into_owned();
    *func = tri!(module_function(module, &name));
    CUDA_SUCCESS
}

/// The kernels of a module, in PTX order
pub(crate) unsafe fn module_kernel_names(m: *mut Module) -> Vec<String> {
    (*m).compiled.kernels.iter().map(|k| k.name.clone()).collect()
}

/// cuModuleGetFunctionCount (CUDA 12.4)
#[no_mangle]
pub unsafe extern "system" fn cuModuleGetFunctionCount(count: *mut c_uint, module: *mut Module) -> CUresult {
    tri!(global());
    if count.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if !module_valid(module) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    *count = (*module).compiled.kernels.len() as c_uint;
    CUDA_SUCCESS
}

/// cuModuleEnumerateFunctions (CUDA 12.4): the first `n` functions
#[no_mangle]
pub unsafe extern "system" fn cuModuleEnumerateFunctions(
    functions: *mut *mut Function,
    n: c_uint,
    module: *mut Module,
) -> CUresult {
    tri!(global());
    if functions.is_null() && n > 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if !module_valid(module) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    let names = module_kernel_names(module);
    if (n as usize) > names.len() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (i, name) in names.iter().take(n as usize).enumerate() {
        *functions.add(i) = tri!(module_function(module, name));
    }
    CUDA_SUCCESS
}

/// cuModuleGetLoadingMode (CUDA 11.7): modules are loaded eagerly
#[no_mangle]
pub unsafe extern "system" fn cuModuleGetLoadingMode(mode: *mut c_int) -> CUresult {
    tri!(global());
    if mode.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *mode = 1; // CU_MODULE_EAGER_LOADING
    CUDA_SUCCESS
}

pub(crate) unsafe fn module_global(
    m: &Module,
    name: &str,
) -> Result<(CUdeviceptr, usize), CUresult> {
    if let Some(g) = m.globals32.iter().find(|g| g.0 == name) {
        return Ok((g.1, g.2));
    }
    match m.compiled.globals.iter().find(|g| g.name == name) {
        Some(g) => Ok((m.globals + g.offset, g.size as usize)),
        None => Err(report(
            format!("no writable device variable named {name} (constant data is not addressable on Metal)"),
            CUDA_ERROR_NOT_FOUND,
        )),
    }
}

#[no_mangle]
pub unsafe extern "system" fn cuModuleGetGlobal_v2(
    dptr: *mut CUdeviceptr,
    bytes: *mut usize,
    module: *mut Module,
    name: *const c_char,
) -> CUresult {
    tri!(global());
    if name.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if !module_valid(module) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    let name = CStr::from_ptr(name).to_string_lossy();
    let (a, n) = tri!(module_global(&*module, &name));
    if !dptr.is_null() {
        *dptr = a;
    }
    if !bytes.is_null() {
        *bytes = n;
    }
    CUDA_SUCCESS
}

/// A function handle (CUfunction, or a CUkernel, which cuLaunchKernel accepts too)
pub(crate) unsafe fn function<'a>(f: *mut Function) -> Result<&'a Function, CUresult> {
    if function_valid(f) {
        Ok(&*f)
    } else {
        Err(CUDA_ERROR_INVALID_HANDLE)
    }
}

impl Function {
    pub(crate) fn static_shared(&self) -> u32 {
        self.kernel.static_shared.max(self.info.static_shared) as u32
    }
    /// registers (32-bit) per thread
    pub(crate) fn registers(&self) -> u32 {
        self.kernel.agx().map_or(32, |k| k.record().gprs.max(1))
    }
    /// the largest block the kernel can be launched with
    pub(crate) fn max_threads(&self) -> u32 {
        let lim = self.kernel.max_threads.min(1024) as u32;
        let mut t = lim;
        while t > 32 && occupancy::blocks_per_sm(self, t, 0) == 0 {
            t -= 32;
        }
        t
    }
}

/// cuFuncGetAttribute: every CUfunction_attribute (0 ..= 15); CUDA_ERROR_INVALID_VALUE for others
#[no_mangle]
pub unsafe extern "system" fn cuFuncGetAttribute(
    value: *mut c_int,
    attrib: c_int,
    func: *mut Function,
) -> CUresult {
    tri!(global());
    if value.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let f = tri!(function(func));
    let a = *f.attrs.lock().unwrap();
    let record = f.kernel.agx().map(|k| k.record());
    *value = match attrib {
        0 => f.max_threads() as c_int,                      // MAX_THREADS_PER_BLOCK
        1 => f.static_shared() as c_int,                    // SHARED_SIZE_BYTES
        2 => 0,                                             // CONST_SIZE_BYTES
        3 => record.map_or(0, |r| r.stack) as c_int,        // LOCAL_SIZE_BYTES
        4 => f.registers() as c_int,                        // NUM_REGS
        5 => (*f.module).target,                            // PTX_VERSION
        6 => CC_MAJOR * 10 + CC_MINOR,                      // BINARY_VERSION
        7 => 0,                                             // CACHE_MODE_CA
        8 => a.max_dynamic_shared as c_int,                 // MAX_DYNAMIC_SHARED_SIZE_BYTES
        9 => a.carveout,                                    // PREFERRED_SHARED_MEMORY_CARVEOUT
        10 => record.is_some_and(|r| r.cluster.is_some()) as c_int, // CLUSTER_SIZE_MUST_BE_SET
        11..=13 => {
            // REQUIRED_CLUSTER_WIDTH / HEIGHT / DEPTH
            let i = (attrib - 11) as usize;
            record.and_then(|r| r.cluster).map_or(a.cluster[i], |c| c[i]) as c_int
        }
        14 => a.non_portable_cluster as c_int, // NON_PORTABLE_CLUSTER_SIZE_ALLOWED
        15 => a.cluster_policy as c_int,       // CLUSTER_SCHEDULING_POLICY_PREFERENCE
        _ => return CUDA_ERROR_INVALID_VALUE,
    };
    CUDA_SUCCESS
}

/// cuFuncSetAttribute: MAX_DYNAMIC_SHARED_SIZE_BYTES (up to the opt-in limit minus the static shared memory),
/// PREFERRED_SHARED_MEMORY_CARVEOUT (-1 ..= 100) and the cluster attributes are settable; the others are read-only
/// (CUDA_ERROR_INVALID_VALUE)
#[no_mangle]
pub unsafe extern "system" fn cuFuncSetAttribute(
    func: *mut Function,
    attrib: c_int,
    value: c_int,
) -> CUresult {
    tri!(global());
    let f = tri!(function(func));
    let mut a = f.attrs.lock().unwrap();
    match attrib {
        8 => {
            if value < 0 || value as u32 > device::SHARED_PER_BLOCK_OPTIN.saturating_sub(f.static_shared()) {
                return CUDA_ERROR_INVALID_VALUE;
            }
            a.max_dynamic_shared = value as u32;
        }
        9 => {
            if !(-1..=100).contains(&value) {
                return CUDA_ERROR_INVALID_VALUE;
            }
            a.carveout = value;
        }
        11..=13 if value >= 0 => a.cluster[(attrib - 11) as usize] = value as u32,
        14 if value == 0 || value == 1 => a.non_portable_cluster = value as u32,
        15 if (0..=2).contains(&value) => a.cluster_policy = value as u32,
        _ => return CUDA_ERROR_INVALID_VALUE,
    }
    CUDA_SUCCESS
}

/// cuFuncSetCacheConfig: CU_FUNC_CACHE_PREFER_NONE / SHARED / L1 / EQUAL (a hint)
#[no_mangle]
pub unsafe extern "system" fn cuFuncSetCacheConfig(func: *mut Function, config: c_int) -> CUresult {
    tri!(global());
    let f = tri!(function(func));
    if !(0..=3).contains(&config) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    f.attrs.lock().unwrap().cache_config = config as u32;
    CUDA_SUCCESS
}

/// cuFuncSetSharedMemConfig (deprecated): a hint
#[no_mangle]
pub unsafe extern "system" fn cuFuncSetSharedMemConfig(func: *mut Function, config: c_int) -> CUresult {
    tri!(global());
    tri!(function(func));
    if !(0..=2).contains(&config) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

/// cuFuncGetModule (CUDA 11.0)
#[no_mangle]
pub unsafe extern "system" fn cuFuncGetModule(module: *mut *mut Module, func: *mut Function) -> CUresult {
    tri!(global());
    if module.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *module = tri!(function(func)).module;
    CUDA_SUCCESS
}

/// cuFuncGetName (CUDA 12.3): the kernel's (mangled) name, valid as long as the function
#[no_mangle]
pub unsafe extern "system" fn cuFuncGetName(name: *mut *const c_char, func: *mut Function) -> CUresult {
    tri!(global());
    if name.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *name = tri!(function(func)).name.as_ptr();
    CUDA_SUCCESS
}

/// cuFuncGetParamInfo (CUDA 12.4): offset and size of parameter `index`; CUDA_ERROR_INVALID_VALUE past the last
#[no_mangle]
pub unsafe extern "system" fn cuFuncGetParamInfo(
    func: *mut Function,
    index: usize,
    offset: *mut usize,
    size: *mut usize,
) -> CUresult {
    tri!(global());
    let f = tri!(function(func));
    let explicit = f.explicit_params.unwrap_or(f.info.params.len());
    if index >= explicit {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let p = &f.info.params[index];
    if !offset.is_null() {
        *offset = p.offset as usize;
    }
    if !size.is_null() {
        *size = p.size as usize;
    }
    CUDA_SUCCESS
}

/// cuFuncIsLoaded / cuFuncLoad (CUDA 12.4): functions are loaded with their module
#[no_mangle]
pub unsafe extern "system" fn cuFuncIsLoaded(state: *mut c_int, func: *mut Function) -> CUresult {
    tri!(global());
    if state.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    tri!(function(func));
    *state = 1; // CU_FUNCTION_LOADING_STATE_LOADED
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuFuncLoad(func: *mut Function) -> CUresult {
    tri!(global());
    tri!(function(func));
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Launches
// ---------------------------------------------------------------------------
const CU_LAUNCH_PARAM_END: usize = 0;
const CU_LAUNCH_PARAM_BUFFER_POINTER: usize = 1;
const CU_LAUNCH_PARAM_BUFFER_SIZE: usize = 2;

/// A validated launch, its parameter buffer filled: what a stream (or a graph node) runs
#[derive(Clone)]
pub(crate) struct KernelLaunch {
    f: usize,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    params: Vec<u8>,
}

/// Validate a launch configuration and pack its arguments (cuLaunchKernel's rules): CUDA_ERROR_INVALID_VALUE for
/// a zero or too large grid / block dimension, more dynamic shared memory than the function's
/// MAX_DYNAMIC_SHARED_SIZE_BYTES, or missing arguments; CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES when the block does not
/// fit a multiprocessor (registers)
pub(crate) unsafe fn prepare_launch(
    func: *mut Function,
    grid: [c_uint; 3],
    block: [c_uint; 3],
    shared_mem_bytes: c_uint,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> Result<KernelLaunch, CUresult> {
    let g = global()?;
    let f = function(func)?;
    let threads = block[0] as u64 * block[1] as u64 * block[2] as u64;
    if threads == 0 || grid.contains(&0) {
        return Err(CUDA_ERROR_INVALID_VALUE);
    }
    if block[0] > device::MAX_BLOCK[0] || block[1] > device::MAX_BLOCK[1] || block[2] > device::MAX_BLOCK[2]
        || grid[0] > device::MAX_GRID[0] || grid[1] > device::MAX_GRID[1] || grid[2] > device::MAX_GRID[2]
    {
        return Err(report(format!("grid {grid:?} / block {block:?} beyond the device limits"), CUDA_ERROR_INVALID_VALUE));
    }
    if threads > f.kernel.max_threads || threads > g.device.max_threads_per_threadgroup {
        return Err(report(
            format!(
                "block of {threads} threads exceeds the kernel limit {}",
                f.kernel.max_threads
            ),
            CUDA_ERROR_INVALID_VALUE,
        ));
    }
    if shared_mem_bytes > f.attrs.lock().unwrap().max_dynamic_shared {
        return Err(report(
            format!("{shared_mem_bytes} bytes of dynamic shared memory exceed the function's \
                     CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES"),
            CUDA_ERROR_INVALID_VALUE,
        ));
    }
    if f.info.static_shared + shared_mem_bytes as u64 > g.device.max_threadgroup_memory {
        return Err(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
    }
    if f.kernel.agx().is_some() && occupancy::blocks_per_sm(f, threads as u32, shared_mem_bytes) == 0 {
        return Err(report(format!("a block of {threads} threads needs more registers than a core has"),
                          CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES));
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
                return Err(CUDA_ERROR_INVALID_VALUE);
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
                _ => return Err(CUDA_ERROR_INVALID_VALUE),
            }
            i += 2;
        }
        if buffer.is_null() || size < params.len() {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        ptr::copy_nonoverlapping(buffer, params.as_mut_ptr(), params.len());
    } else if explicit > 0 {
        return Err(CUDA_ERROR_INVALID_VALUE);
    }
    if let Some(off) = f.info.globals_param_offset {
        params[off as usize..off as usize + 8].copy_from_slice(&f.globals.to_ne_bytes());
    }
    if let Some(off) = f.info.dyn_smem_param_offset {
        params[off as usize..off as usize + 4].copy_from_slice(&shared_mem_bytes.to_ne_bytes());
    }
    Ok(KernelLaunch { f: func as usize, grid, block, shared: shared_mem_bytes, params })
}

/// Run a prepared launch on the device (in stream order, `cuda/stream.rs`): a fault of the kernel becomes the
/// context's sticky error
pub(crate) unsafe fn run_kernel(l: &KernelLaunch) -> CUresult {
    let f = &*(l.f as *const Function);
    let dynamic_shared = if f.info.dynamic_shared {
        l.shared.max(16)
    } else {
        0
    };
    let r = launch_result(f.kernel.launch(l.grid, l.block, &l.params, dynamic_shared));
    check_fault_record(f.fault, &f.info.name);
    r
}

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
    stream: *mut Stream,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    tri!(ensure_current());
    let l = tri!(prepare_launch(func, [grid_x, grid_y, grid_z], [block_x, block_y, block_z], shared_mem_bytes,
                                kernel_params, extra));
    stream::submit_to(stream, stream::Op::Kernel(l))
}

/// cuLaunchCooperativeKernel: as cuLaunchKernel, and the grid must be co-resident on the device
/// (CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE otherwise, the bound cuOccupancyMaxActiveBlocksPerMultiprocessor x
/// multiprocessors). The functional model runs blocks one after another: grid-wide barriers are not supported.
#[no_mangle]
pub unsafe extern "system" fn cuLaunchCooperativeKernel(
    func: *mut Function,
    grid_x: c_uint,
    grid_y: c_uint,
    grid_z: c_uint,
    block_x: c_uint,
    block_y: c_uint,
    block_z: c_uint,
    shared_mem_bytes: c_uint,
    stream: *mut Stream,
    kernel_params: *mut *mut c_void,
) -> CUresult {
    tri!(ensure_current());
    let l = tri!(prepare_launch(func, [grid_x, grid_y, grid_z], [block_x, block_y, block_z], shared_mem_bytes,
                                kernel_params, ptr::null_mut()));
    tri!(occupancy::check_cooperative(&l));
    stream::submit_to(stream, stream::Op::Kernel(l))
}

/// CUlaunchConfig (cuLaunchKernelEx, CUDA 11.8)
#[repr(C)]
pub struct CUlaunchConfig {
    grid: [c_uint; 3],
    block: [c_uint; 3],
    shared_mem_bytes: c_uint,
    stream: *mut Stream,
    attrs: *mut CUlaunchAttribute,
    num_attrs: c_uint,
}

/// CUlaunchAttribute: an id and a 64-byte value union
#[repr(C)]
pub struct CUlaunchAttribute {
    id: c_uint,
    pad: [u8; 4],
    value: [u64; 8],
}

/// cuLaunchKernelEx: the launch attributes COOPERATIVE (2, checked as cuLaunchCooperativeKernel) and
/// CLUSTER_DIMENSION (4, only 1 x 1 x 1 or the kernel's own `.reqnctapercluster`) change the launch; the
/// scheduling hints (access policy window, priority, synchronization policy, ...) are accepted and ignored
#[no_mangle]
pub unsafe extern "system" fn cuLaunchKernelEx(
    config: *const CUlaunchConfig,
    func: *mut Function,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    tri!(ensure_current());
    if config.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let c = &*config;
    if c.num_attrs > 0 && c.attrs.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let l = tri!(prepare_launch(func, c.grid, c.block, c.shared_mem_bytes, kernel_params, extra));
    for i in 0..c.num_attrs as usize {
        let a = &*c.attrs.add(i);
        match a.id {
            2 => {
                if a.value[0] as u32 != 0 {
                    tri!(occupancy::check_cooperative(&l));
                }
            }
            4 => {
                let d = [a.value[0] as u32, (a.value[0] >> 32) as u32, a.value[1] as u32];
                let own = (*(l.f as *const Function)).kernel.agx().and_then(|k| k.record().cluster);
                if d.iter().product::<u32>() > 1 && own != Some(d) {
                    return report("launch-time thread block clusters are not supported".into(),
                                  CUDA_ERROR_NOT_SUPPORTED);
                }
            }
            0..=20 => {}
            _ => return CUDA_ERROR_INVALID_VALUE,
        }
    }
    stream::submit_to(c.stream, stream::Op::Kernel(l))
}

// ---------------------------------------------------------------------------
// Profiler (no-ops)
// ---------------------------------------------------------------------------
#[no_mangle]
pub unsafe extern "system" fn cuProfilerStart() -> CUresult {
    tri!(ensure_current());
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuProfilerStop() -> CUresult {
    tri!(ensure_current());
    CUDA_SUCCESS
}

/// Last detailed error message of the Metal backend (ZLUDA extension)
#[no_mangle]
pub unsafe extern "system" fn zludaMetalGetLastError(buf: *mut c_char, len: usize) -> usize {
    let Some(Ok(g)) = GLOBAL.get() else { return 0 };
    let msg = g.last_error.lock().unwrap().clone();
    if !buf.is_null() && len > 0 {
        let n = msg.len().min(len - 1);
        ptr::copy_nonoverlapping(msg.as_ptr(), buf as *mut u8, n);
        *buf.add(n) = 0;
    }
    msg.len()
}

