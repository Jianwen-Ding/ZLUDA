//! Safe-ish wrapper over the metal-cpp backend (cpp/metal_backend.h)
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

#[repr(C)]
pub struct ZmtlDeviceInfo {
    pub name: [c_char; 256],
    pub total_memory: u64,
    pub max_threads_per_threadgroup: u64,
    pub max_threadgroup_memory: u64,
    pub simd_width: u64,
    pub unified_memory: c_int,
}

#[repr(C)]
pub struct ZmtlLibrary {
    _private: [u8; 0],
}
#[repr(C)]
pub struct ZmtlKernel {
    _private: [u8; 0],
}

extern "C" {
    fn zmtl_init(info: *mut ZmtlDeviceInfo, error: *mut *mut c_char) -> c_int;
    fn zmtl_free_string(s: *mut c_char);
    fn zmtl_mem_alloc(size: u64, gpu_address: *mut u64, error: *mut *mut c_char) -> c_int;
    fn zmtl_mem_free(gpu_address: u64) -> c_int;
    fn zmtl_host_pointer(gpu_address: u64, bytes_left: *mut u64) -> *mut c_void;
    fn zmtl_mem_range(gpu_address: u64, base: *mut u64, size: *mut u64) -> c_int;
    fn zmtl_synchronize(error: *mut *mut c_char) -> c_int;
    fn zmtl_library_create(
        metallib: *const u8,
        len: usize,
        lib: *mut *mut ZmtlLibrary,
        error: *mut *mut c_char,
    ) -> c_int;
    fn zmtl_library_destroy(lib: *mut ZmtlLibrary);
    fn zmtl_kernel_create(
        lib: *mut ZmtlLibrary,
        name: *const c_char,
        kernel: *mut *mut ZmtlKernel,
        max_threads: *mut u64,
        static_shared: *mut u64,
        error: *mut *mut c_char,
    ) -> c_int;
    fn zmtl_kernel_destroy(kernel: *mut ZmtlKernel);
    fn zmtl_launch(
        kernel: *mut ZmtlKernel,
        grid: *const u32,
        block: *const u32,
        params: *const c_void,
        params_size: usize,
        dynamic_shared: u32,
        error: *mut *mut c_char,
    ) -> c_int;
}

fn take_error(e: *mut c_char) -> String {
    if e.is_null() {
        return "unknown Metal backend error".to_string();
    }
    let s = unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned();
    unsafe { zmtl_free_string(e) };
    s
}

fn check(r: c_int, e: *mut c_char) -> Result<(), String> {
    if r == 0 {
        Ok(())
    } else {
        Err(take_error(e))
    }
}


/// `ZLUDA_METAL_BACKEND=agx`: run kernels on the AGX simulator (src/agx.rs)
/// instead of the metal-cpp device.
pub fn use_agx() -> bool {
    static AGX: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AGX.get_or_init(|| std::env::var("ZLUDA_METAL_BACKEND").map(|v| v == "agx").unwrap_or(false))
}

pub fn init() -> Result<DeviceInfo, String> {
    if use_agx() {
        return Ok(DeviceInfo {
            name: format!("Apple AGX {} (AGX Performance Simulator)", crate::agx::mcpu()),
            total_memory: 32 << 30,
            max_threads_per_threadgroup: 1024,
            max_threadgroup_memory: 65536,
            simd_width: 32,
            unified_memory: true,
        });
    }
    metal_init()
}
pub fn alloc(size: u64) -> Result<u64, String> {
    if use_agx() { crate::agx::alloc(size) } else { metal_alloc(size) }
}
pub fn free(addr: u64) -> bool {
    if use_agx() { crate::agx::free(addr) } else { metal_free(addr) }
}
pub fn host_pointer(addr: u64) -> Option<(*mut u8, u64)> {
    if use_agx() { crate::agx::host_pointer(addr) } else { metal_host_pointer(addr) }
}
pub fn range(addr: u64) -> Option<(u64, u64)> {
    if use_agx() { crate::agx::range(addr) } else { metal_range(addr) }
}
/// Host memory the device sees at its own address (AGX backend only: Metal buffers would need
/// `newBufferWithBytesNoCopy` and their own GPU addresses)
pub fn register_host(p: *mut u8, len: u64) -> Result<u64, String> {
    if use_agx() { crate::agx::register_host(p, len) } else { Err("host memory registration needs ZLUDA_METAL_BACKEND=agx".into()) }
}
pub fn unregister_host(addr: u64) -> bool {
    use_agx() && crate::agx::unregister_host(addr)
}
pub fn synchronize() -> Result<(), String> {
    if use_agx() { Ok(()) } else { metal_synchronize() }
}

pub enum Library {
    Metal(MetalLibrary),
    Agx(crate::agx::Library),
}
impl Library {
    pub fn new(metallib: &[u8]) -> Result<Self, String> {
        if use_agx() {
            crate::agx::Library::new(metallib).map(Library::Agx)
        } else {
            MetalLibrary::new(metallib).map(Library::Metal)
        }
    }
    pub fn kernel(&self, name: &str) -> Result<Kernel, String> {
        match self {
            Library::Metal(l) => l.kernel(name).map(|k| Kernel { max_threads: k.max_threads, static_shared: k.static_shared, imp: KernelImpl::Metal(k) }),
            Library::Agx(l) => l.kernel(name).map(|k| Kernel { max_threads: k.max_threads(), static_shared: k.static_shared(), imp: KernelImpl::Agx(k) }),
        }
    }
}
enum KernelImpl {
    Metal(MetalKernel),
    Agx(crate::agx::Kernel),
}
pub struct Kernel {
    imp: KernelImpl,
    pub max_threads: u64,
    pub static_shared: u64,
}
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}
impl Kernel {
    /// The AGX kernel (its record: registers, stack), on the AGX backend
    pub fn agx(&self) -> Option<&crate::agx::Kernel> {
        match &self.imp {
            KernelImpl::Agx(k) => Some(k),
            KernelImpl::Metal(_) => None,
        }
    }
    pub fn launch(&self, grid: [u32; 3], block: [u32; 3], params: &[u8], dynamic_shared: u32) -> Result<(), String> {
        match &self.imp {
            KernelImpl::Metal(k) => k.launch(grid, block, params, dynamic_shared),
            KernelImpl::Agx(k) => k.launch(grid, block, params, dynamic_shared),
        }
    }
}

pub struct DeviceInfo {
    pub name: String,
    pub total_memory: u64,
    pub max_threads_per_threadgroup: u64,
    pub max_threadgroup_memory: u64,
    pub simd_width: u64,
    pub unified_memory: bool,
}

fn metal_init() -> Result<DeviceInfo, String> {
    let mut info: ZmtlDeviceInfo = unsafe { std::mem::zeroed() };
    let mut e = ptr::null_mut();
    check(unsafe { zmtl_init(&mut info, &mut e) }, e)?;
    Ok(DeviceInfo {
        name: unsafe { CStr::from_ptr(info.name.as_ptr()) }
            .to_string_lossy()
            .into_owned(),
        total_memory: info.total_memory,
        max_threads_per_threadgroup: info.max_threads_per_threadgroup,
        max_threadgroup_memory: info.max_threadgroup_memory,
        simd_width: info.simd_width,
        unified_memory: info.unified_memory != 0,
    })
}

fn metal_alloc(size: u64) -> Result<u64, String> {
    let mut addr = 0;
    let mut e = ptr::null_mut();
    check(unsafe { zmtl_mem_alloc(size, &mut addr, &mut e) }, e)?;
    Ok(addr)
}

fn metal_free(addr: u64) -> bool {
    unsafe { zmtl_mem_free(addr) == 0 }
}

/// Host view of device memory: (pointer, bytes until the end of the allocation)
fn metal_host_pointer(addr: u64) -> Option<(*mut u8, u64)> {
    let mut left = 0;
    let p = unsafe { zmtl_host_pointer(addr, &mut left) };
    (!p.is_null()).then_some((p as *mut u8, left))
}

/// Allocation containing `addr`: (base address, size)
fn metal_range(addr: u64) -> Option<(u64, u64)> {
    let (mut base, mut size) = (0, 0);
    (unsafe { zmtl_mem_range(addr, &mut base, &mut size) } == 0).then_some((base, size))
}

fn metal_synchronize() -> Result<(), String> {
    let mut e = ptr::null_mut();
    check(unsafe { zmtl_synchronize(&mut e) }, e)
}

pub struct MetalLibrary(*mut ZmtlLibrary);
unsafe impl Send for MetalLibrary {}
unsafe impl Sync for MetalLibrary {}

impl MetalLibrary {
    pub fn new(metallib: &[u8]) -> Result<Self, String> {
        let mut lib = ptr::null_mut();
        let mut e = ptr::null_mut();
        check(
            unsafe { zmtl_library_create(metallib.as_ptr(), metallib.len(), &mut lib, &mut e) },
            e,
        )?;
        Ok(MetalLibrary(lib))
    }

    pub fn kernel(&self, name: &str) -> Result<MetalKernel, String> {
        let cname = CString::new(name).map_err(|_| "invalid kernel name".to_string())?;
        let mut k = ptr::null_mut();
        let mut max_threads = 0;
        let mut static_shared = 0;
        let mut e = ptr::null_mut();
        check(
            unsafe {
                zmtl_kernel_create(
                    self.0,
                    cname.as_ptr(),
                    &mut k,
                    &mut max_threads,
                    &mut static_shared,
                    &mut e,
                )
            },
            e,
        )?;
        Ok(MetalKernel {
            raw: k,
            max_threads,
            static_shared,
        })
    }
}

impl Drop for MetalLibrary {
    fn drop(&mut self) {
        unsafe { zmtl_library_destroy(self.0) }
    }
}

pub struct MetalKernel {
    raw: *mut ZmtlKernel,
    pub max_threads: u64,
    pub static_shared: u64,
}
unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

impl MetalKernel {
    pub fn launch(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        params: &[u8],
        dynamic_shared: u32,
    ) -> Result<(), String> {
        let mut e = ptr::null_mut();
        check(
            unsafe {
                zmtl_launch(
                    self.raw,
                    grid.as_ptr(),
                    block.as_ptr(),
                    params.as_ptr().cast(),
                    params.len(),
                    dynamic_shared,
                    &mut e,
                )
            },
            e,
        )
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe { zmtl_kernel_destroy(self.raw) }
    }
}
