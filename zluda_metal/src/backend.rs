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

pub struct DeviceInfo {
    pub name: String,
    pub total_memory: u64,
    pub max_threads_per_threadgroup: u64,
    pub max_threadgroup_memory: u64,
    pub simd_width: u64,
    pub unified_memory: bool,
}

pub fn init() -> Result<DeviceInfo, String> {
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

pub fn alloc(size: u64) -> Result<u64, String> {
    let mut addr = 0;
    let mut e = ptr::null_mut();
    check(unsafe { zmtl_mem_alloc(size, &mut addr, &mut e) }, e)?;
    Ok(addr)
}

pub fn free(addr: u64) -> bool {
    unsafe { zmtl_mem_free(addr) == 0 }
}

/// Host view of device memory: (pointer, bytes until the end of the allocation)
pub fn host_pointer(addr: u64) -> Option<(*mut u8, u64)> {
    let mut left = 0;
    let p = unsafe { zmtl_host_pointer(addr, &mut left) };
    (!p.is_null()).then_some((p as *mut u8, left))
}

/// Allocation containing `addr`: (base address, size)
pub fn range(addr: u64) -> Option<(u64, u64)> {
    let (mut base, mut size) = (0, 0);
    (unsafe { zmtl_mem_range(addr, &mut base, &mut size) } == 0).then_some((base, size))
}

pub fn synchronize() -> Result<(), String> {
    let mut e = ptr::null_mut();
    check(unsafe { zmtl_synchronize(&mut e) }, e)
}

pub struct Library(*mut ZmtlLibrary);
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

impl Library {
    pub fn new(metallib: &[u8]) -> Result<Self, String> {
        let mut lib = ptr::null_mut();
        let mut e = ptr::null_mut();
        check(
            unsafe { zmtl_library_create(metallib.as_ptr(), metallib.len(), &mut lib, &mut e) },
            e,
        )?;
        Ok(Library(lib))
    }

    pub fn kernel(&self, name: &str) -> Result<Kernel, String> {
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
        Ok(Kernel {
            raw: k,
            max_threads,
            static_shared,
        })
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        unsafe { zmtl_library_destroy(self.0) }
    }
}

pub struct Kernel {
    raw: *mut ZmtlKernel,
    pub max_threads: u64,
    pub static_shared: u64,
}
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
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

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { zmtl_kernel_destroy(self.raw) }
    }
}
