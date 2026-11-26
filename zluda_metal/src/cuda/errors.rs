//! CUresult values (the documented ABI), cuGetErrorName / cuGetErrorString
use super::CUresult;
use std::ffi::CStr;
use std::os::raw::c_char;

macro_rules! errors {
    ($($name:ident = $v:expr, $desc:literal;)*) => {
        $(pub const $name: CUresult = $v;)*
        fn error_info(e: CUresult) -> Option<(&'static CStr, &'static CStr)> {
            match e {
                $($v => Some((
                    // the enumerator's name, NUL-terminated
                    unsafe { CStr::from_bytes_with_nul_unchecked(concat!(stringify!($name), "\0").as_bytes()) },
                    unsafe { CStr::from_bytes_with_nul_unchecked(concat!($desc, "\0").as_bytes()) },
                )),)*
                _ => None,
            }
        }
    };
}

errors! {
    CUDA_SUCCESS = 0, "no error";
    CUDA_ERROR_INVALID_VALUE = 1, "invalid argument";
    CUDA_ERROR_OUT_OF_MEMORY = 2, "out of memory";
    CUDA_ERROR_NOT_INITIALIZED = 3, "initialization error (cuInit has not been called)";
    CUDA_ERROR_DEINITIALIZED = 4, "driver shutting down";
    CUDA_ERROR_PROFILER_DISABLED = 5, "profiler disabled";
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6, "profiler not initialized";
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7, "profiler already started";
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8, "profiler already stopped";
    CUDA_ERROR_STUB_LIBRARY = 34, "stub library";
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46, "device unavailable";
    CUDA_ERROR_NO_DEVICE = 100, "no CUDA-capable device is detected";
    CUDA_ERROR_INVALID_DEVICE = 101, "invalid device ordinal";
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102, "device not licensed";
    CUDA_ERROR_INVALID_IMAGE = 200, "device kernel image is invalid";
    CUDA_ERROR_INVALID_CONTEXT = 201, "invalid device context";
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202, "context already current";
    CUDA_ERROR_MAP_FAILED = 205, "mapping of buffer object failed";
    CUDA_ERROR_UNMAP_FAILED = 206, "unmapping of buffer object failed";
    CUDA_ERROR_ARRAY_IS_MAPPED = 207, "array is mapped";
    CUDA_ERROR_ALREADY_MAPPED = 208, "resource already mapped";
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209, "no kernel image is available for execution on the device";
    CUDA_ERROR_ALREADY_ACQUIRED = 210, "resource already acquired";
    CUDA_ERROR_NOT_MAPPED = 211, "resource not mapped";
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212, "resource not mapped as array";
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213, "resource not mapped as pointer";
    CUDA_ERROR_ECC_UNCORRECTABLE = 214, "uncorrectable ECC error encountered";
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215, "limit is not supported on this architecture";
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216, "exclusive-thread device already in use by a different thread";
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217, "peer access is not supported between these two devices";
    CUDA_ERROR_INVALID_PTX = 218, "a PTX JIT compilation failed";
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219, "invalid OpenGL or DirectX context";
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220, "uncorrectable NVLink error detected during the execution";
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221, "PTX JIT compiler library not found";
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222, "the provided PTX was compiled with an unsupported toolchain";
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223, "PTX JIT compilation was disabled";
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224, "the provided execution affinity is not supported";
    CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC = 225, "the provided PTX contains unsupported call to cudaDeviceSynchronize";
    CUDA_ERROR_INVALID_SOURCE = 300, "device kernel image is invalid";
    CUDA_ERROR_FILE_NOT_FOUND = 301, "file not found";
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302, "shared object symbol not found";
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303, "shared object initialization failed";
    CUDA_ERROR_OPERATING_SYSTEM = 304, "OS call failed or operation not supported on this OS";
    CUDA_ERROR_INVALID_HANDLE = 400, "invalid resource handle";
    CUDA_ERROR_ILLEGAL_STATE = 401, "the operation is not permitted in the current state";
    CUDA_ERROR_LOSSY_QUERY = 402, "the attempted query is lossy";
    CUDA_ERROR_NOT_FOUND = 500, "named symbol not found";
    CUDA_ERROR_NOT_READY = 600, "device not ready";
    CUDA_ERROR_ILLEGAL_ADDRESS = 700, "an illegal memory access was encountered";
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701, "too many resources requested for launch";
    CUDA_ERROR_LAUNCH_TIMEOUT = 702, "the launch timed out and was terminated";
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703, "launch uses incompatible texturing mode";
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704, "peer access is already enabled";
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705, "peer access has not been enabled";
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708, "cannot set while device is active in this process";
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709, "context is destroyed";
    CUDA_ERROR_ASSERT = 710, "device-side assert triggered";
    CUDA_ERROR_TOO_MANY_PEERS = 711, "peer mapping resources exhausted";
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712, "part or all of the requested memory range is already mapped";
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713, "pointer does not correspond to a registered memory region";
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714, "hardware stack error";
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715, "an illegal instruction was encountered";
    CUDA_ERROR_MISALIGNED_ADDRESS = 716, "misaligned address";
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717, "operation not supported on global/shared address space";
    CUDA_ERROR_INVALID_PC = 718, "invalid program counter";
    CUDA_ERROR_LAUNCH_FAILED = 719, "unspecified launch failure";
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720, "too many blocks in cooperative launch";
    CUDA_ERROR_NOT_PERMITTED = 800, "operation not permitted";
    CUDA_ERROR_NOT_SUPPORTED = 801, "operation not supported";
    CUDA_ERROR_SYSTEM_NOT_READY = 802, "system not yet initialized";
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803, "system has unsupported display driver / cuda driver combination";
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804, "forward compatibility was attempted on non supported HW";
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900, "operation not permitted when stream is capturing";
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901, "operation failed due to a previous error during capture";
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902, "operation would result in a merge of separate capture sequences";
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903, "capture was not ended in the same stream as it began";
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904, "capturing stream has unjoined work";
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905, "dependency created on uncaptured work in another stream";
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906, "operation would make the legacy stream depend on a capturing blocking stream";
    CUDA_ERROR_CAPTURED_EVENT = 907, "operation not permitted on an event last recorded in a capturing stream";
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908, "attempt to terminate a thread-local capture sequence from another thread";
    CUDA_ERROR_TIMEOUT = 909, "wait operation timed out";
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910, "the graph update was not performed because it included changes which violated constraints specific to instantiated graph update";
    CUDA_ERROR_EXTERNAL_DEVICE = 911, "an async error has occurred in a external entity outside of CUDA";
    CUDA_ERROR_INVALID_CLUSTER_SIZE = 912, "a kernel launch error has occurred due to cluster misconfiguration";
    CUDA_ERROR_FUNCTION_NOT_LOADED = 913, "the function handle is not loaded when calling an API that requires a loaded function";
    CUDA_ERROR_INVALID_RESOURCE_TYPE = 914, "one or more resources passed in are not valid resource types for the operation";
    CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION = 915, "one or more resources are insufficient or non-applicable for the operation";
    CUDA_ERROR_UNKNOWN = 999, "unknown error";
}

/// cuGetErrorName: CUDA_ERROR_INVALID_VALUE (and *pStr = NULL) for an unknown code; works before cuInit
#[no_mangle]
pub unsafe extern "system" fn cuGetErrorName(e: CUresult, out: *mut *const c_char) -> CUresult {
    if out.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    match error_info(e) {
        Some((n, _)) => {
            *out = n.as_ptr();
            CUDA_SUCCESS
        }
        None => {
            *out = std::ptr::null();
            CUDA_ERROR_INVALID_VALUE
        }
    }
}

/// cuGetErrorString: a description of the code, as cuGetErrorName for unknown codes
#[no_mangle]
pub unsafe extern "system" fn cuGetErrorString(e: CUresult, out: *mut *const c_char) -> CUresult {
    if out.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    match error_info(e) {
        Some((_, d)) => {
            *out = d.as_ptr();
            CUDA_SUCCESS
        }
        None => {
            *out = std::ptr::null();
            CUDA_ERROR_INVALID_VALUE
        }
    }
}

