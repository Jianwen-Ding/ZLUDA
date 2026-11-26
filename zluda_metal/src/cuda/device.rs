//! Device management (CUDA Driver API reference, "Device Management"): one device, ordinal 0.
//! Its limits describe the AGX simulator's device (M2 Max, `agx_sim::dispatch` occupancy model).
use super::*;

pub(crate) const MAX_BLOCK: [u32; 3] = [1024, 1024, 64];
pub(crate) const MAX_GRID: [u32; 3] = [i32::MAX as u32, 65535, 65535];
/// shared memory per block without / with opt-in (cuFuncSetAttribute MAX_DYNAMIC_SHARED_SIZE_BYTES)
pub(crate) const SHARED_PER_BLOCK: u32 = 48 << 10;
pub(crate) const SHARED_PER_BLOCK_OPTIN: u32 = 64 << 10;
pub(crate) const SHARED_PER_SM: u32 = 64 << 10;
/// resident SIMD groups per core (96) x 32 lanes
pub(crate) const THREADS_PER_SM: u32 = 96 * 32;
/// a core's register file in 32-bit registers (3328 16-bit halves per lane slot x 32 lanes / 2)
pub(crate) const REGISTERS_PER_SM: u32 = 1664 * 32;
/// resident workgroups per core (32-thread workgroups cap at 55 resident SIMD groups)
pub(crate) const BLOCKS_PER_SM: u32 = 55;
pub(crate) const MAX_PERSISTING_L2: u64 = 0;
/// the last CUdevice_attribute value + 1 (attributes past the ones below report 0)
const ATTRIBUTE_MAX: c_int = 150;

pub(crate) fn multiprocessors() -> u32 {
    crate::agx::CORES
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
pub unsafe extern "system" fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult {
    let g = tri!(check_device(dev));
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
    let g = tri!(check_device(dev));
    if bytes.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *bytes = g.device.total_memory as usize;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDeviceComputeCapability(major: *mut c_int, minor: *mut c_int, dev: CUdevice) -> CUresult {
    tri!(check_device(dev));
    if major.is_null() || minor.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *major = CC_MAJOR;
    *minor = CC_MINOR;
    CUDA_SUCCESS
}

/// cuDeviceGetAttribute: every CUdevice_attribute; CUDA_ERROR_INVALID_VALUE for values outside the enumeration
#[no_mangle]
pub unsafe extern "system" fn cuDeviceGetAttribute(value: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult {
    let g = tri!(check_device(dev));
    if value.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let agx = backend::use_agx() as c_int;
    let max_threads = g.device.max_threads_per_threadgroup.min(1024) as c_int;
    *value = match attrib {
        1 => max_threads,                                 // MAX_THREADS_PER_BLOCK
        2 => MAX_BLOCK[0].min(max_threads as u32) as c_int, // MAX_BLOCK_DIM_X
        3 => MAX_BLOCK[1].min(max_threads as u32) as c_int, // MAX_BLOCK_DIM_Y
        4 => MAX_BLOCK[2] as c_int,                       // MAX_BLOCK_DIM_Z
        5 => MAX_GRID[0] as c_int,                        // MAX_GRID_DIM_X
        6 => MAX_GRID[1] as c_int,                        // MAX_GRID_DIM_Y
        7 => MAX_GRID[2] as c_int,                        // MAX_GRID_DIM_Z
        8 => SHARED_PER_BLOCK as c_int,                   // MAX_SHARED_MEMORY_PER_BLOCK
        9 => 65536,                                       // TOTAL_CONSTANT_MEMORY
        10 => g.device.simd_width as c_int,               // WARP_SIZE
        11 => i32::MAX,                                   // MAX_PITCH
        12 => REGISTERS_PER_SM as c_int,                  // MAX_REGISTERS_PER_BLOCK
        13 => 1_398_000,                                  // CLOCK_RATE (kHz)
        14 => 512,                                        // TEXTURE_ALIGNMENT
        15 => 0,                                          // GPU_OVERLAP (copies and kernels do not overlap)
        16 => multiprocessors() as c_int,                 // MULTIPROCESSOR_COUNT
        17 => 0,                                          // KERNEL_EXEC_TIMEOUT
        18 => g.device.unified_memory as c_int,           // INTEGRATED
        19 => agx,                                        // CAN_MAP_HOST_MEMORY
        20 => 0,                                          // COMPUTE_MODE (default)
        21..=29 => 0,                                     // texture limits (no textures)
        30 => 0,                                          // SURFACE_ALIGNMENT
        31 => 0,                                          // CONCURRENT_KERNELS (one kernel at a time)
        32 => 0,                                          // ECC_ENABLED
        33 | 34 => 0,                                     // PCI_BUS_ID, PCI_DEVICE_ID
        35 => 0,                                          // TCC_DRIVER
        36 => 3_200_000,                                  // MEMORY_CLOCK_RATE (kHz)
        37 => 512,                                        // GLOBAL_MEMORY_BUS_WIDTH
        38 => 8 << 20,                                    // L2_CACHE_SIZE
        39 => THREADS_PER_SM as c_int,                    // MAX_THREADS_PER_MULTIPROCESSOR
        40 => 0,                                          // ASYNC_ENGINE_COUNT
        41 => 1,                                          // UNIFIED_ADDRESSING
        42..=49 => 0,                                     // texture limits
        50 => 0,                                          // PCI_DOMAIN_ID
        51 => 32,                                         // TEXTURE_PITCH_ALIGNMENT
        52..=74 => 0,                                     // texture / surface limits
        75 => CC_MAJOR,                                   // COMPUTE_CAPABILITY_MAJOR
        76 => CC_MINOR,                                   // COMPUTE_CAPABILITY_MINOR
        77 => 0,                                          // MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH
        78 => 1,                                          // STREAM_PRIORITIES_SUPPORTED
        79 | 80 => 1,                                     // GLOBAL / LOCAL_L1_CACHE_SUPPORTED
        81 => SHARED_PER_SM as c_int,                     // MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        82 => REGISTERS_PER_SM as c_int,                  // MAX_REGISTERS_PER_MULTIPROCESSOR
        83 => agx,                                        // MANAGED_MEMORY
        84 | 85 => 0,                                     // MULTI_GPU_BOARD(_GROUP_ID)
        86 => 0,                                          // HOST_NATIVE_ATOMIC_SUPPORTED
        87 => 64,                                         // SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO (software f64)
        88 => 0,                                          // PAGEABLE_MEMORY_ACCESS
        89 => agx,                                        // CONCURRENT_MANAGED_ACCESS
        90 => 0,                                          // COMPUTE_PREEMPTION_SUPPORTED
        91 => agx,                                        // CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
        92..=94 => 0,                                     // stream memory operations (v1)
        95 | 96 => 0,                                     // COOPERATIVE_(MULTI_DEVICE_)LAUNCH
        97 => SHARED_PER_BLOCK_OPTIN as c_int,            // MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        98 => 0,                                          // CAN_FLUSH_REMOTE_WRITES
        99 => agx,                                        // HOST_REGISTER_SUPPORTED
        100 => 0,                                         // PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
        101 => agx,                                       // DIRECT_MANAGED_MEM_ACCESS_FROM_HOST
        102 => 0,                                         // VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
        103..=105 => 0,                                   // HANDLE_TYPE_*_SUPPORTED
        106 => BLOCKS_PER_SM as c_int,                    // MAX_BLOCKS_PER_MULTIPROCESSOR
        107 => 0,                                         // GENERIC_COMPRESSION_SUPPORTED
        108 => MAX_PERSISTING_L2 as c_int,                // MAX_PERSISTING_L2_CACHE_SIZE
        109 => 0,                                         // MAX_ACCESS_POLICY_WINDOW_SIZE
        110 => 0,                                         // GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
        111 => 0,                                         // RESERVED_SHARED_MEMORY_PER_BLOCK
        112 => 0,                                         // SPARSE_CUDA_ARRAY_SUPPORTED
        113 => agx,                                       // READ_ONLY_HOST_REGISTER_SUPPORTED
        114 => 0,                                         // TIMELINE_SEMAPHORE_INTEROP_SUPPORTED
        115 => 1,                                         // MEMORY_POOLS_SUPPORTED
        116..=119 => 0,                                   // GPU_DIRECT_RDMA_*, MEMPOOL_SUPPORTED_HANDLE_TYPES
        120 => 0,                                         // CLUSTER_LAUNCH (launch-time clusters)
        121..=125 => 0,                                   // DEFERRED_MAPPING, 64-bit stream mem ops, DMA_BUF, IPC events
        126 => 1,                                         // MEM_SYNC_DOMAIN_COUNT
        127..=130 => 0,                                   // TENSOR_MAP_ACCESS, FABRIC, UNIFIED_FUNCTION_POINTERS, NUMA_CONFIG
        131 => -1,                                        // NUMA_ID
        132 | 133 => 0,                                   // MULTICAST_SUPPORTED, MPS_ENABLED
        134 => -1,                                        // HOST_NUMA_ID
        135..=149 => 0,
        _ => return CUDA_ERROR_INVALID_VALUE,
    };
    debug_assert!(attrib < ATTRIBUTE_MAX);
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDeviceGetUuid(uuid: *mut [u8; 16], dev: CUdevice) -> CUresult {
    tri!(check_device(dev));
    if uuid.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *uuid = *b"ZLUDA-METAL-0000";
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDeviceGetUuid_v2(uuid: *mut [u8; 16], dev: CUdevice) -> CUresult {
    cuDeviceGetUuid(uuid, dev)
}

/// cuDeviceGetPCIBusId: "[domain]:[bus]:[device].[function]", at most len - 1 characters
#[no_mangle]
pub unsafe extern "system" fn cuDeviceGetPCIBusId(bus_id: *mut c_char, len: c_int, dev: CUdevice) -> CUresult {
    tri!(check_device(dev));
    if bus_id.is_null() || len <= 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let s = b"0000:00:00.0";
    let n = s.len().min(len as usize - 1);
    ptr::copy_nonoverlapping(s.as_ptr(), bus_id as *mut u8, n);
    *bus_id.add(n) = 0;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDeviceGetByPCIBusId(dev: *mut CUdevice, bus_id: *const c_char) -> CUresult {
    tri!(global());
    if dev.is_null() || bus_id.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let s = CStr::from_ptr(bus_id).to_string_lossy().to_ascii_lowercase();
    if s == "0000:00:00.0" || s == "00:00.0" || s == "0000:00:00" {
        *dev = 0;
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_DEVICE
    }
}

/// CUdevprop (cuDeviceGetProperties, deprecated)
#[repr(C)]
pub struct CUdevprop {
    max_threads_per_block: c_int,
    max_threads_dim: [c_int; 3],
    max_grid_size: [c_int; 3],
    shared_mem_per_block: c_int,
    total_constant_memory: c_int,
    simd_width: c_int,
    mem_pitch: c_int,
    regs_per_block: c_int,
    clock_rate: c_int,
    texture_align: c_int,
}

#[no_mangle]
pub unsafe extern "system" fn cuDeviceGetProperties(prop: *mut CUdevprop, dev: CUdevice) -> CUresult {
    tri!(check_device(dev));
    if prop.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let a = |i| {
        let mut v = 0;
        cuDeviceGetAttribute(&mut v, i, dev);
        v
    };
    *prop = CUdevprop {
        max_threads_per_block: a(1),
        max_threads_dim: [a(2), a(3), a(4)],
        max_grid_size: [a(5), a(6), a(7)],
        shared_mem_per_block: a(8),
        total_constant_memory: a(9),
        simd_width: a(10),
        mem_pitch: a(11),
        regs_per_block: a(12),
        clock_rate: a(13),
        texture_align: a(14),
    };
    CUDA_SUCCESS
}

/// cuDeviceCanAccessPeer: 0, there is no other device
#[no_mangle]
pub unsafe extern "system" fn cuDeviceCanAccessPeer(can: *mut c_int, dev: CUdevice, peer: CUdevice) -> CUresult {
    tri!(check_device(dev));
    tri!(check_device(peer));
    if can.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *can = 0;
    CUDA_SUCCESS
}

/// cuDeviceGetP2PAttribute: the source and destination must be different devices (CUDA_ERROR_INVALID_DEVICE)
#[no_mangle]
pub unsafe extern "system" fn cuDeviceGetP2PAttribute(
    value: *mut c_int,
    _attrib: c_int,
    src: CUdevice,
    dst: CUdevice,
) -> CUresult {
    tri!(check_device(src));
    tri!(check_device(dst));
    if value.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_ERROR_INVALID_DEVICE
}

/// cuDeviceGetExecAffinitySupport: no execution affinity types
#[no_mangle]
pub unsafe extern "system" fn cuDeviceGetExecAffinitySupport(pi: *mut c_int, ty: c_int, dev: CUdevice) -> CUresult {
    tri!(check_device(dev));
    if pi.is_null() || ty != 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pi = 0;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuFlushGPUDirectRDMAWrites(_target: c_int, _scope: c_int) -> CUresult {
    tri!(context::current());
    CUDA_SUCCESS
}
