//! cuGetProcAddress (CUDA Driver API reference, "Driver Entry Point Access"): the entry points this library
//! exports, by base name and requested CUDA version. A function whose ABI changed has several exported versions
//! (`_v2`, `_v3`, ...); the table below gives the CUDA version each appeared in, and the newest one not newer
//! than the requested version is returned, as the driver does.
use super::*;

/// (base name, [(first CUDA version, exported symbol)]), oldest first
const VERSIONED: &[(&str, &[(c_int, &str)])] = &[
    ("cuCtxCreate", &[(2000, "cuCtxCreate_v2"), (11040, "cuCtxCreate_v3"), (12050, "cuCtxCreate_v4")]),
    ("cuCtxDestroy", &[(4000, "cuCtxDestroy_v2")]),
    ("cuCtxPushCurrent", &[(4000, "cuCtxPushCurrent_v2")]),
    ("cuCtxPopCurrent", &[(4000, "cuCtxPopCurrent_v2")]),
    ("cuStreamDestroy", &[(4000, "cuStreamDestroy_v2")]),
    ("cuEventDestroy", &[(4000, "cuEventDestroy_v2")]),
    ("cuDevicePrimaryCtxRelease", &[(7000, "cuDevicePrimaryCtxRelease"), (11000, "cuDevicePrimaryCtxRelease_v2")]),
    ("cuDevicePrimaryCtxReset", &[(7000, "cuDevicePrimaryCtxReset"), (11000, "cuDevicePrimaryCtxReset_v2")]),
    ("cuDevicePrimaryCtxSetFlags", &[(7000, "cuDevicePrimaryCtxSetFlags"), (11000, "cuDevicePrimaryCtxSetFlags_v2")]),
    ("cuDeviceTotalMem", &[(3020, "cuDeviceTotalMem_v2")]),
    ("cuMemGetInfo", &[(3020, "cuMemGetInfo_v2")]),
    ("cuMemAlloc", &[(3020, "cuMemAlloc_v2")]),
    ("cuMemAllocPitch", &[(3020, "cuMemAllocPitch_v2")]),
    ("cuMemFree", &[(3020, "cuMemFree_v2")]),
    ("cuMemGetAddressRange", &[(3020, "cuMemGetAddressRange_v2")]),
    ("cuMemAllocHost", &[(3020, "cuMemAllocHost_v2")]),
    ("cuMemHostGetDevicePointer", &[(3020, "cuMemHostGetDevicePointer_v2")]),
    ("cuMemHostRegister", &[(4000, "cuMemHostRegister"), (6050, "cuMemHostRegister_v2")]),
    ("cuMemcpyHtoD", &[(3020, "cuMemcpyHtoD_v2")]),
    ("cuMemcpyDtoH", &[(3020, "cuMemcpyDtoH_v2")]),
    ("cuMemcpyDtoD", &[(3020, "cuMemcpyDtoD_v2")]),
    ("cuMemcpyHtoDAsync", &[(3020, "cuMemcpyHtoDAsync_v2")]),
    ("cuMemcpyDtoHAsync", &[(3020, "cuMemcpyDtoHAsync_v2")]),
    ("cuMemcpyDtoDAsync", &[(3020, "cuMemcpyDtoDAsync_v2")]),
    ("cuMemcpy2D", &[(3020, "cuMemcpy2D_v2")]),
    ("cuMemcpy2DUnaligned", &[(3020, "cuMemcpy2DUnaligned_v2")]),
    ("cuMemcpy2DAsync", &[(3020, "cuMemcpy2DAsync_v2")]),
    ("cuMemcpy3D", &[(3020, "cuMemcpy3D_v2")]),
    ("cuMemcpy3DAsync", &[(3020, "cuMemcpy3DAsync_v2")]),
    ("cuMemsetD8", &[(3020, "cuMemsetD8_v2")]),
    ("cuMemsetD16", &[(3020, "cuMemsetD16_v2")]),
    ("cuMemsetD32", &[(3020, "cuMemsetD32_v2")]),
    ("cuMemsetD2D8", &[(3020, "cuMemsetD2D8_v2")]),
    ("cuMemsetD2D16", &[(3020, "cuMemsetD2D16_v2")]),
    ("cuMemsetD2D32", &[(3020, "cuMemsetD2D32_v2")]),
    ("cuModuleGetGlobal", &[(3020, "cuModuleGetGlobal_v2")]),
    ("cuLinkCreate", &[(6050, "cuLinkCreate_v2")]),
    ("cuLinkAddData", &[(6050, "cuLinkAddData_v2")]),
    ("cuLinkAddFile", &[(6050, "cuLinkAddFile_v2")]),
    ("cuDeviceGetUuid", &[(9020, "cuDeviceGetUuid"), (11040, "cuDeviceGetUuid_v2")]),
    ("cuStreamBeginCapture", &[(10000, "cuStreamBeginCapture"), (10010, "cuStreamBeginCapture_v2")]),
    ("cuStreamGetCaptureInfo", &[(10010, "cuStreamGetCaptureInfo"), (11030, "cuStreamGetCaptureInfo_v2"),
                                 (12030, "cuStreamGetCaptureInfo_v3")]),
    ("cuGraphInstantiate", &[(10000, "cuGraphInstantiate"), (11000, "cuGraphInstantiate_v2"),
                             (12000, "cuGraphInstantiateWithFlags")]),
    ("cuGraphAddKernelNode", &[(10000, "cuGraphAddKernelNode"), (12000, "cuGraphAddKernelNode_v2")]),
    ("cuEventElapsedTime", &[(2000, "cuEventElapsedTime"), (12080, "cuEventElapsedTime_v2")]),
    ("cuMemAdvise", &[(8000, "cuMemAdvise"), (12020, "cuMemAdvise_v2")]),
    ("cuMemPrefetchAsync", &[(8000, "cuMemPrefetchAsync"), (12020, "cuMemPrefetchAsync_v2")]),
    ("cuGetProcAddress", &[(11030, "cuGetProcAddress"), (12000, "cuGetProcAddress_v2")]),
];

/// CUdriverProcAddressQueryResult
const SYMBOL_FOUND: c_int = 0;
const SYMBOL_NOT_FOUND: c_int = 1;
const VERSION_NOT_SUFFICIENT: c_int = 2;

#[repr(C)]
struct DlInfo {
    fname: *const c_char,
    fbase: *mut c_void,
    sname: *const c_char,
    saddr: *mut c_void,
}

extern "C" {
    fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
    fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

#[cfg(target_os = "macos")]
const RTLD_NOLOAD: c_int = 0x10;
#[cfg(not(target_os = "macos"))]
const RTLD_NOLOAD: c_int = 4;
const RTLD_LAZY: c_int = 1;

/// This library's own handle: lookups never return another library's symbols
fn own_handle() -> *mut c_void {
    static H: OnceLock<usize> = OnceLock::new();
    *H.get_or_init(|| unsafe {
        let mut info: DlInfo = std::mem::zeroed();
        if dladdr(cuGetProcAddress_v2 as *const c_void, &mut info) == 0 || info.fname.is_null() {
            return 0;
        }
        dlopen(info.fname, RTLD_LAZY | RTLD_NOLOAD) as usize
    }) as *mut c_void
}

/// A driver entry point this library exports
fn own_symbol(name: &str) -> *mut c_void {
    if !name.starts_with("cu") || name.starts_with("cublas") {
        return ptr::null_mut();
    }
    let h = own_handle();
    let Ok(c) = CString::new(name) else { return ptr::null_mut() };
    if h.is_null() {
        return ptr::null_mut();
    }
    let p = unsafe { dlsym(h, c.as_ptr()) };
    // dlsym on a library handle also searches its dependencies: only this library's own symbols count
    let mut info: DlInfo = unsafe { std::mem::zeroed() };
    let mut me: DlInfo = unsafe { std::mem::zeroed() };
    unsafe {
        if p.is_null() || dladdr(p, &mut info) == 0 || dladdr(cuGetProcAddress_v2 as *const c_void, &mut me) == 0
            || info.fbase != me.fbase
        {
            return ptr::null_mut();
        }
    }
    p
}

/// (pointer, CUdriverProcAddressQueryResult)
fn lookup(name: &str, version: c_int) -> (*mut c_void, c_int) {
    if let Some((_, versions)) = VERSIONED.iter().find(|(b, _)| *b == name) {
        return match versions.iter().rev().find(|(v, _)| *v <= version) {
            Some((_, sym)) => {
                let p = own_symbol(sym);
                (p, if p.is_null() { SYMBOL_NOT_FOUND } else { SYMBOL_FOUND })
            }
            None => (ptr::null_mut(), VERSION_NOT_SUFFICIENT),
        };
    }
    let p = own_symbol(name);
    (p, if p.is_null() { SYMBOL_NOT_FOUND } else { SYMBOL_FOUND })
}

/// cuGetProcAddress_v2 (CUDA 12.0): CUDA_ERROR_INVALID_VALUE for NULL symbol / pfn or unknown flags
/// (CU_GET_PROC_ADDRESS_DEFAULT 0, LEGACY_STREAM 1, PER_THREAD_DEFAULT_STREAM 2); CUDA_ERROR_NOT_FOUND (and
/// *pfn = NULL) for an unknown symbol or one newer than the requested version, with the reason in symbolStatus
#[no_mangle]
pub unsafe extern "system" fn cuGetProcAddress_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cuda_version: c_int,
    flags: u64,
    status: *mut c_int,
) -> CUresult {
    if symbol.is_null() || pfn.is_null() || flags > 2 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let name = CStr::from_ptr(symbol).to_string_lossy().into_owned();
    // per-thread default stream variants (_ptsz / _ptds): the per-thread stream behaves as the legacy one here
    let (p, st) = lookup(&name, cuda_version);
    *pfn = p;
    if !status.is_null() {
        *status = st;
    }
    if p.is_null() {
        CUDA_ERROR_NOT_FOUND
    } else {
        CUDA_SUCCESS
    }
}

/// cuGetProcAddress (CUDA 11.3)
#[no_mangle]
pub unsafe extern "system" fn cuGetProcAddress(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cuda_version: c_int,
    flags: u64,
) -> CUresult {
    cuGetProcAddress_v2(symbol, pfn, cuda_version, flags, ptr::null_mut())
}
