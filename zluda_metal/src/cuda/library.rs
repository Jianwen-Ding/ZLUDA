//! The context-independent library API of CUDA 12 (CUDA Driver API reference, "Library Management": cuLibrary*,
//! cuKernel*) and the JIT linker (cuLink*). A library holds one module that belongs to no context; its kernels
//! (CUkernel) are its functions, which cuLaunchKernel accepts directly, and cuKernelGetFunction returns the same
//! handle in every context.
use super::*;

pub struct Library {
    module: *mut Module,
}
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

static LIBRARIES: Handles<Library> = Handles::new();

fn library(l: *mut Library) -> Result<Arc<Library>, CUresult> {
    LIBRARIES.get(l).ok_or(CUDA_ERROR_INVALID_HANDLE)
}

/// cuLibraryLoadData: PTX text (as cuModuleLoadData); no context is needed
#[no_mangle]
pub unsafe extern "system" fn cuLibraryLoadData(
    lib: *mut *mut Library,
    code: *const c_void,
    jit_options: *mut c_int,
    jit_values: *mut *mut c_void,
    num_jit: c_uint,
    lib_options: *mut c_int,
    lib_values: *mut *mut c_void,
    num_lib: c_uint,
) -> CUresult {
    tri!(global());
    if lib.is_null() || (num_jit > 0 && (jit_options.is_null() || jit_values.is_null()))
        || (num_lib > 0 && (lib_options.is_null() || lib_values.is_null()))
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let ptx = tri!(image_to_ptx(code));
    let mut m = ptr::null_mut();
    let r = load_module(&mut m, ptx, 0);
    if r != CUDA_SUCCESS {
        return r;
    }
    *lib = LIBRARIES.insert(Arc::new(Library { module: m }));
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuLibraryLoadFromFile(
    lib: *mut *mut Library,
    fname: *const c_char,
    jit_options: *mut c_int,
    jit_values: *mut *mut c_void,
    num_jit: c_uint,
    lib_options: *mut c_int,
    lib_values: *mut *mut c_void,
    num_lib: c_uint,
) -> CUresult {
    tri!(global());
    if fname.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let mut data = match std::fs::read(CStr::from_ptr(fname).to_string_lossy().as_ref()) {
        Ok(d) => d,
        Err(_) => return CUDA_ERROR_FILE_NOT_FOUND,
    };
    data.push(0);
    cuLibraryLoadData(lib, data.as_ptr().cast(), jit_options, jit_values, num_jit, lib_options, lib_values, num_lib)
}

#[no_mangle]
pub unsafe extern "system" fn cuLibraryUnload(lib: *mut Library) -> CUresult {
    tri!(global());
    match LIBRARIES.remove(lib) {
        Some(l) => {
            unload_module(l.module);
            CUDA_SUCCESS
        }
        None => CUDA_ERROR_INVALID_HANDLE,
    }
}

/// cuLibraryGetKernel: CUDA_ERROR_NOT_FOUND for an unknown name
#[no_mangle]
pub unsafe extern "system" fn cuLibraryGetKernel(kernel: *mut *mut Function, lib: *mut Library, name: *const c_char) -> CUresult {
    tri!(global());
    if kernel.is_null() || name.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let l = tri!(library(lib));
    let f = tri!(module_function(l.module, &CStr::from_ptr(name).to_string_lossy()));
    (*f).library.store(lib as u64, Ordering::Relaxed);
    *kernel = f;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuLibraryGetModule(module: *mut *mut Module, lib: *mut Library) -> CUresult {
    tri!(context::current());
    if module.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *module = tri!(library(lib)).module;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuLibraryGetGlobal(
    dptr: *mut CUdeviceptr,
    bytes: *mut usize,
    lib: *mut Library,
    name: *const c_char,
) -> CUresult {
    tri!(context::current());
    if name.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let l = tri!(library(lib));
    let (a, n) = tri!(module_global(&*l.module, &CStr::from_ptr(name).to_string_lossy()));
    if !dptr.is_null() {
        *dptr = a;
    }
    if !bytes.is_null() {
        *bytes = n;
    }
    CUDA_SUCCESS
}

/// cuLibraryGetManaged: no `__managed__` variables (CUDA_ERROR_NOT_FOUND)
#[no_mangle]
pub unsafe extern "system" fn cuLibraryGetManaged(
    _dptr: *mut CUdeviceptr,
    _bytes: *mut usize,
    lib: *mut Library,
    name: *const c_char,
) -> CUresult {
    tri!(global());
    tri!(library(lib));
    if name.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_ERROR_NOT_FOUND
}

#[no_mangle]
pub unsafe extern "system" fn cuLibraryGetKernelCount(count: *mut c_uint, lib: *mut Library) -> CUresult {
    tri!(global());
    if count.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *count = module_kernel_names(tri!(library(lib)).module).len() as c_uint;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuLibraryEnumerateKernels(kernels: *mut *mut Function, n: c_uint, lib: *mut Library) -> CUresult {
    tri!(global());
    let l = tri!(library(lib));
    let names = module_kernel_names(l.module);
    if (kernels.is_null() && n > 0) || n as usize > names.len() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    for (i, name) in names.iter().take(n as usize).enumerate() {
        let f = tri!(module_function(l.module, name));
        (*f).library.store(lib as u64, Ordering::Relaxed);
        *kernels.add(i) = f;
    }
    CUDA_SUCCESS
}

/// cuKernelGetFunction: the kernel's function in the current context (CUDA_ERROR_INVALID_CONTEXT without one)
#[no_mangle]
pub unsafe extern "system" fn cuKernelGetFunction(func: *mut *mut Function, kernel: *mut Function) -> CUresult {
    tri!(context::current());
    if func.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    tri!(function(kernel));
    *func = kernel;
    CUDA_SUCCESS
}

/// cuKernelGetAttribute: as cuFuncGetAttribute (one device)
#[no_mangle]
pub unsafe extern "system" fn cuKernelGetAttribute(value: *mut c_int, attrib: c_int, kernel: *mut Function, dev: CUdevice) -> CUresult {
    tri!(check_device(dev));
    cuFuncGetAttribute(value, attrib, kernel)
}

#[no_mangle]
pub unsafe extern "system" fn cuKernelSetAttribute(attrib: c_int, value: c_int, kernel: *mut Function, dev: CUdevice) -> CUresult {
    tri!(check_device(dev));
    cuFuncSetAttribute(kernel, attrib, value)
}

#[no_mangle]
pub unsafe extern "system" fn cuKernelSetCacheConfig(kernel: *mut Function, config: c_int, dev: CUdevice) -> CUresult {
    tri!(check_device(dev));
    cuFuncSetCacheConfig(kernel, config)
}

#[no_mangle]
pub unsafe extern "system" fn cuKernelGetName(name: *mut *const c_char, kernel: *mut Function) -> CUresult {
    cuFuncGetName(name, kernel)
}

#[no_mangle]
pub unsafe extern "system" fn cuKernelGetLibrary(lib: *mut *mut Library, kernel: *mut Function) -> CUresult {
    tri!(global());
    if lib.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let f = tri!(function(kernel));
    match f.library.load(Ordering::Relaxed) {
        0 => CUDA_ERROR_INVALID_HANDLE,
        l => {
            *lib = l as *mut Library;
            CUDA_SUCCESS
        }
    }
}

#[no_mangle]
pub unsafe extern "system" fn cuKernelGetParamInfo(kernel: *mut Function, index: usize, offset: *mut usize, size: *mut usize) -> CUresult {
    cuFuncGetParamInfo(kernel, index, offset, size)
}

// ---------------------------------------------------------------------------
// Linker: one PTX input is the "image" (cuModuleLoadData takes PTX)
// ---------------------------------------------------------------------------
pub struct LinkState {
    inputs: Mutex<Vec<Vec<u8>>>,
    out: Mutex<Vec<u8>>,
}

static LINKS: Handles<LinkState> = Handles::new();

/// CUjitInputType: CUBIN 0, PTX 1, FATBINARY 2, OBJECT 3, LIBRARY 4, NVVM 5
const CU_JIT_INPUT_PTX: c_int = 1;

#[no_mangle]
pub unsafe extern "system" fn cuLinkCreate_v2(
    num_options: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
    state: *mut *mut LinkState,
) -> CUresult {
    tri!(context::current());
    if state.is_null() || (num_options > 0 && (options.is_null() || values.is_null())) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *state = LINKS.insert(Arc::new(LinkState { inputs: Mutex::new(vec![]), out: Mutex::new(vec![]) }));
    CUDA_SUCCESS
}

/// cuLinkAddData: PTX inputs; machine code inputs (cubin, fatbin, objects, libraries) are for NVIDIA GPUs only
/// (CUDA_ERROR_NO_BINARY_FOR_GPU), NVVM IR is not supported
#[no_mangle]
pub unsafe extern "system" fn cuLinkAddData_v2(
    state: *mut LinkState,
    ty: c_int,
    data: *mut c_void,
    size: usize,
    _name: *const c_char,
    num_options: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> CUresult {
    tri!(global());
    let Some(l) = LINKS.get(state) else { return CUDA_ERROR_INVALID_HANDLE };
    if data.is_null() || size == 0 || (num_options > 0 && (options.is_null() || values.is_null())) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    match ty {
        CU_JIT_INPUT_PTX => {
            let b = std::slice::from_raw_parts(data as *const u8, size);
            let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
            l.inputs.lock().unwrap().push(b[..end].to_vec());
            CUDA_SUCCESS
        }
        0 | 2..=4 => CUDA_ERROR_NO_BINARY_FOR_GPU,
        5 => CUDA_ERROR_NOT_SUPPORTED,
        _ => CUDA_ERROR_INVALID_VALUE,
    }
}

#[no_mangle]
pub unsafe extern "system" fn cuLinkAddFile_v2(
    state: *mut LinkState,
    ty: c_int,
    path: *const c_char,
    num_options: c_uint,
    options: *mut c_int,
    values: *mut *mut c_void,
) -> CUresult {
    tri!(global());
    if path.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let data = match std::fs::read(CStr::from_ptr(path).to_string_lossy().as_ref()) {
        Ok(d) => d,
        Err(_) => return CUDA_ERROR_FILE_NOT_FOUND,
    };
    cuLinkAddData_v2(state, ty, data.as_ptr() as *mut c_void, data.len(), path, num_options, options, values)
}

/// cuLinkComplete: the linked image, valid until cuLinkDestroy. Linking several PTX modules is not supported.
#[no_mangle]
pub unsafe extern "system" fn cuLinkComplete(state: *mut LinkState, image: *mut *mut c_void, size: *mut usize) -> CUresult {
    tri!(global());
    let Some(l) = LINKS.get(state) else { return CUDA_ERROR_INVALID_HANDLE };
    if image.is_null() || size.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let inputs = l.inputs.lock().unwrap();
    match inputs.len() {
        0 => CUDA_ERROR_INVALID_VALUE,
        1 => {
            let mut out = l.out.lock().unwrap();
            *out = inputs[0].clone();
            out.push(0);
            *image = out.as_mut_ptr() as *mut c_void;
            *size = out.len();
            CUDA_SUCCESS
        }
        _ => report("linking several PTX modules is not supported".into(), CUDA_ERROR_NOT_SUPPORTED),
    }
}

#[no_mangle]
pub unsafe extern "system" fn cuLinkDestroy(state: *mut LinkState) -> CUresult {
    tri!(global());
    match LINKS.remove(state) {
        Some(_) => CUDA_SUCCESS,
        None => CUDA_ERROR_INVALID_HANDLE,
    }
}
