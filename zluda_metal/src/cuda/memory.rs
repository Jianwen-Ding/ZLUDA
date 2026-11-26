//! Memory management (CUDA Driver API reference, "Memory Management", "Unified Addressing", "Stream Ordered
//! Memory Allocator"): device, pinned, registered and managed allocations, tracked per context; copies and sets in
//! 1, 2 and 3 dimensions; pointer attributes.
//!
//! With unified virtual addressing every host allocation the driver knows (cuMemAllocHost / cuMemHostAlloc,
//! cuMemHostRegister, cuMemAllocManaged) is visible to the device at its host address (the AGX simulator maps it),
//! so its device pointer is the host pointer.
use super::stream::{self, Op};
use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    Device,
    /// cuMemAllocHost / cuMemHostAlloc
    Pinned,
    /// cuMemHostRegister
    Registered,
    Managed,
}

#[derive(Clone, Copy)]
pub(crate) struct Alloc {
    pub size: u64,
    pub kind: Kind,
    pub flags: u32,
    pub ctx: usize,
    pub id: u64,
    pub sync_memops: bool,
    /// the host allocation's size (pinned / managed memory the driver allocated)
    host_layout: usize,
    /// made by cuMemAllocAsync / cuMemAllocFromPoolAsync
    pooled: bool,
}

static ALLOCS: Mutex<BTreeMap<u64, Alloc>> = Mutex::new(BTreeMap::new());

fn allocs() -> std::sync::MutexGuard<'static, BTreeMap<u64, Alloc>> {
    ALLOCS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The allocation containing `addr`: (base, allocation)
pub(crate) fn lookup(addr: u64) -> Option<(u64, Alloc)> {
    let a = allocs();
    let (b, al) = a.range(..=addr).next_back()?;
    (addr < b + al.size.max(1)).then_some((*b, *al))
}

fn insert(base: u64, size: u64, kind: Kind, flags: u32, ctx: usize, host_layout: usize, pooled: bool) {
    static IDS: AtomicU64 = AtomicU64::new(1);
    allocs().insert(base, Alloc {
        size,
        kind,
        flags,
        ctx,
        id: IDS.fetch_add(1, Ordering::Relaxed),
        sync_memops: false,
        host_layout,
        pooled,
    });
}

fn host_layout(size: usize) -> std::alloc::Layout {
    std::alloc::Layout::from_size_align((size.max(1) + 4095) & !4095, 4096).unwrap()
}

/// Page-aligned, zeroed host memory
unsafe fn host_alloc(size: usize) -> Option<(*mut u8, usize)> {
    let l = host_layout(size);
    let p = std::alloc::alloc_zeroed(l);
    (!p.is_null()).then_some((p, l.size()))
}

/// Release an allocation (its registry entry is already removed)
unsafe fn release(base: u64, a: &Alloc) {
    match a.kind {
        Kind::Device => {
            backend::free(base);
        }
        Kind::Registered => {
            backend::unregister_host(base);
        }
        Kind::Pinned | Kind::Managed => {
            backend::unregister_host(base);
            std::alloc::dealloc(base as *mut u8, std::alloc::Layout::from_size_align(a.host_layout, 4096).unwrap());
        }
    }
}

/// A context's teardown: its device and managed allocations are freed (pinned and registered host memory stays
/// the application's, unmapped from the device)
pub(crate) unsafe fn free_context(ctx: usize) {
    let gone: Vec<(u64, Alloc)> = {
        let mut a = allocs();
        let keys: Vec<u64> = a.iter().filter(|(_, v)| v.ctx == ctx).map(|(k, _)| *k).collect();
        keys.into_iter().map(|k| (k, a.remove(&k).unwrap())).collect()
    };
    for (b, a) in gone {
        match a.kind {
            Kind::Device | Kind::Managed => release(b, &a),
            Kind::Pinned | Kind::Registered => {
                backend::unregister_host(b);
                // the host memory stays valid; the driver forgets it
                if a.kind == Kind::Pinned {
                    LEAKED_PINNED.lock().unwrap().push(b);
                }
            }
        }
    }
}

/// pinned allocations of destroyed contexts (still the application's memory)
static LEAKED_PINNED: Mutex<Vec<u64>> = Mutex::new(Vec::new());

fn device_used() -> u64 {
    allocs().values().filter(|a| matches!(a.kind, Kind::Device | Kind::Managed)).map(|a| a.size).sum()
}

// ---------------------------------------------------------------------------
// Device memory
// ---------------------------------------------------------------------------
unsafe fn alloc_device(dptr: *mut CUdeviceptr, size: usize, pooled: bool) -> CUresult {
    let c = tri!(context::current());
    if dptr.is_null() || size == 0 {
        // "If bytesize is 0, cuMemAlloc() returns CUDA_ERROR_INVALID_VALUE"
        return CUDA_ERROR_INVALID_VALUE;
    }
    let total = global().map(|g| g.device.total_memory).unwrap_or(0);
    if device_used() + size as u64 > total {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    match backend::alloc(size as u64) {
        Ok(a) => {
            insert(a, size as u64, Kind::Device, 0, c.key(), 0, pooled);
            *dptr = a;
            CUDA_SUCCESS
        }
        Err(e) => report(e, CUDA_ERROR_OUT_OF_MEMORY),
    }
}

#[no_mangle]
pub unsafe extern "system" fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, size: usize) -> CUresult {
    tri!(graph::unsafe_during_capture());
    alloc_device(dptr, size, false)
}

/// Free a device or managed allocation now (cuMemFree, a stream-ordered free when it runs)
pub(crate) unsafe fn free_now(dptr: CUdeviceptr) -> CUresult {
    let a = {
        let mut m = allocs();
        match m.get(&dptr) {
            Some(a) if matches!(a.kind, Kind::Device | Kind::Managed) => m.remove(&dptr).unwrap(),
            _ => return CUDA_ERROR_INVALID_VALUE,
        }
    };
    release(dptr, &a);
    CUDA_SUCCESS
}

/// cuMemFree: the base of a device (or managed) allocation; any other pointer: CUDA_ERROR_INVALID_VALUE. Freeing
/// waits for the device to finish its work.
#[no_mangle]
pub unsafe extern "system" fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult {
    tri!(context::current());
    tri!(graph::unsafe_during_capture());
    if dptr == 0 {
        return CUDA_SUCCESS;
    }
    if !matches!(lookup(dptr), Some((b, a)) if b == dptr && matches!(a.kind, Kind::Device | Kind::Managed)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    stream::wait_all_idle();
    free_now(dptr)
}

/// cuMemGetInfo: free and total device memory (free decreases with the allocations made)
#[no_mangle]
pub unsafe extern "system" fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult {
    let g = tri!(ensure_current());
    if free.is_null() || total.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let t = g.device.total_memory;
    *free = t.saturating_sub(device_used()) as usize;
    *total = t as usize;
    CUDA_SUCCESS
}

/// cuMemGetAddressRange: base and size of the allocation containing dptr (either output may be NULL);
/// CUDA_ERROR_NOT_FOUND for an address no allocation contains
#[no_mangle]
pub unsafe extern "system" fn cuMemGetAddressRange_v2(
    base: *mut CUdeviceptr,
    size: *mut usize,
    dptr: CUdeviceptr,
) -> CUresult {
    tri!(ensure_current());
    let r = lookup(dptr).map(|(b, a)| (b, a.size)).or_else(|| backend::range(dptr));
    match r {
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

/// pitch alignment of cuMemAllocPitch
const PITCH_ALIGN: usize = 512;

/// cuMemAllocPitch: rows of WidthInBytes padded to a pitch (a multiple of 512 bytes); ElementSizeBytes must be
/// 4, 8 or 16 (CUDA_ERROR_INVALID_VALUE otherwise), as must a non-zero width and height
#[no_mangle]
pub unsafe extern "system" fn cuMemAllocPitch_v2(
    dptr: *mut CUdeviceptr,
    pitch: *mut usize,
    width: usize,
    height: usize,
    element_size: c_uint,
) -> CUresult {
    tri!(context::current());
    if dptr.is_null() || pitch.is_null() || width == 0 || height == 0 || ![4, 8, 16].contains(&element_size) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let p = width.div_ceil(PITCH_ALIGN) * PITCH_ALIGN;
    let Some(total) = p.checked_mul(height) else { return CUDA_ERROR_OUT_OF_MEMORY };
    let r = alloc_device(dptr, total, false);
    if r == CUDA_SUCCESS {
        *pitch = p;
    }
    r
}

// ---------------------------------------------------------------------------
// Host memory: pinned, registered
// ---------------------------------------------------------------------------
/// CU_MEMHOSTALLOC_PORTABLE 1, DEVICEMAP 2, WRITECOMBINED 4
const HOSTALLOC_FLAGS: c_uint = 7;
/// CU_MEMHOSTREGISTER_PORTABLE 1, DEVICEMAP 2, IOMEMORY 4, READ_ONLY 8
const HOSTREGISTER_FLAGS: c_uint = 15;

unsafe fn alloc_pinned(pp: *mut *mut c_void, size: usize, flags: c_uint) -> CUresult {
    let c = tri!(context::current());
    if pp.is_null() || flags & !HOSTALLOC_FLAGS != 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let Some((p, layout)) = host_alloc(size) else { return CUDA_ERROR_OUT_OF_MEMORY };
    // visible to the device at the same address (UVA); the Metal backend keeps it host only
    if backend::use_agx() {
        if let Err(e) = backend::register_host(p, size.max(1) as u64) {
            std::alloc::dealloc(p, std::alloc::Layout::from_size_align(layout, 4096).unwrap());
            return report(e, CUDA_ERROR_OUT_OF_MEMORY);
        }
    }
    insert(p as u64, size.max(1) as u64, Kind::Pinned, flags, c.key(), layout, false);
    *pp = p as *mut c_void;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuMemAllocHost_v2(pp: *mut *mut c_void, size: usize) -> CUresult {
    alloc_pinned(pp, size, 0)
}

/// cuMemHostAlloc: flags PORTABLE, DEVICEMAP, WRITECOMBINED; others CUDA_ERROR_INVALID_VALUE
#[no_mangle]
pub unsafe extern "system" fn cuMemHostAlloc(pp: *mut *mut c_void, size: usize, flags: c_uint) -> CUresult {
    alloc_pinned(pp, size, flags)
}

/// cuMemFreeHost: memory from cuMemAllocHost / cuMemHostAlloc only (CUDA_ERROR_INVALID_VALUE otherwise)
#[no_mangle]
pub unsafe extern "system" fn cuMemFreeHost(p: *mut c_void) -> CUresult {
    tri!(global());
    let b = p as u64;
    let a = {
        let mut m = allocs();
        match m.get(&b) {
            Some(a) if a.kind == Kind::Pinned => m.remove(&b).unwrap(),
            _ => {
                drop(m);
                // pinned memory of a destroyed context
                let mut l = LEAKED_PINNED.lock().unwrap();
                return match l.iter().position(|&x| x == b) {
                    Some(i) => {
                        l.remove(i);
                        CUDA_SUCCESS
                    }
                    None => CUDA_ERROR_INVALID_VALUE,
                };
            }
        }
    };
    stream::wait_all_idle();
    release(b, &a);
    CUDA_SUCCESS
}

/// cuMemHostGetDevicePointer: the device address of pinned / registered / managed memory (the same address, UVA;
/// a pointer inside the range maps to the same offset). Flags must be 0; other memory: CUDA_ERROR_INVALID_VALUE.
#[no_mangle]
pub unsafe extern "system" fn cuMemHostGetDevicePointer_v2(
    dptr: *mut CUdeviceptr,
    p: *mut c_void,
    flags: c_uint,
) -> CUresult {
    tri!(context::current());
    if dptr.is_null() || flags != 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    match lookup(p as u64) {
        Some((_, a)) if a.kind != Kind::Device => {
            if !backend::use_agx() {
                return report("mapped host memory needs ZLUDA_METAL_BACKEND=agx".into(), CUDA_ERROR_NOT_SUPPORTED);
            }
            *dptr = p as u64;
            CUDA_SUCCESS
        }
        _ => CUDA_ERROR_INVALID_VALUE,
    }
}

/// cuMemHostGetFlags: the flags pinned memory was allocated with
#[no_mangle]
pub unsafe extern "system" fn cuMemHostGetFlags(flags: *mut c_uint, p: *mut c_void) -> CUresult {
    tri!(context::current());
    if flags.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    match lookup(p as u64) {
        Some((_, a)) if matches!(a.kind, Kind::Pinned | Kind::Registered) => {
            *flags = a.flags;
            CUDA_SUCCESS
        }
        _ => CUDA_ERROR_INVALID_VALUE,
    }
}

/// cuMemHostRegister: page-locks (and maps for the device, at the same address) an existing host range.
/// CUDA_ERROR_INVALID_VALUE for NULL / size 0 / unknown flags, CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED if any
/// part of the range already is, CUDA_ERROR_NOT_SUPPORTED where the device cannot map host memory (Metal backend).
#[no_mangle]
pub unsafe extern "system" fn cuMemHostRegister_v2(p: *mut c_void, size: usize, flags: c_uint) -> CUresult {
    let c = tri!(context::current());
    if p.is_null() || size == 0 || flags & !HOSTREGISTER_FLAGS != 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let b = p as u64;
    {
        let m = allocs();
        let before = m.range(..=b).next_back().is_some_and(|(k, a)| b < k + a.size);
        if before || m.range(b..b + size as u64).next().is_some() {
            return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
        }
    }
    if let Err(e) = backend::register_host(p as *mut u8, size as u64) {
        let code = if backend::use_agx() { CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED } else { CUDA_ERROR_NOT_SUPPORTED };
        return report(e, code);
    }
    insert(b, size as u64, Kind::Registered, flags, c.key(), 0, false);
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuMemHostRegister(p: *mut c_void, size: usize, flags: c_uint) -> CUresult {
    cuMemHostRegister_v2(p, size, flags)
}

/// cuMemHostUnregister: the base address of a registered range; otherwise CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED
#[no_mangle]
pub unsafe extern "system" fn cuMemHostUnregister(p: *mut c_void) -> CUresult {
    tri!(context::current());
    if p.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let b = p as u64;
    let a = {
        let mut m = allocs();
        match m.get(&b) {
            Some(a) if a.kind == Kind::Registered => m.remove(&b).unwrap(),
            _ => return CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
        }
    };
    stream::wait_all_idle();
    release(b, &a);
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Managed memory
// ---------------------------------------------------------------------------
/// cuMemAllocManaged: memory the host and the device access at the same address. Flags must be
/// CU_MEM_ATTACH_GLOBAL (1) or CU_MEM_ATTACH_HOST (2), the size non-zero (CUDA_ERROR_INVALID_VALUE otherwise);
/// CUDA_ERROR_NOT_SUPPORTED on the Metal backend (CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY 0).
#[no_mangle]
pub unsafe extern "system" fn cuMemAllocManaged(dptr: *mut CUdeviceptr, size: usize, flags: c_uint) -> CUresult {
    let c = tri!(context::current());
    if dptr.is_null() || size == 0 || (flags != 1 && flags != 2) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if !backend::use_agx() {
        return report("managed memory needs ZLUDA_METAL_BACKEND=agx".into(), CUDA_ERROR_NOT_SUPPORTED);
    }
    let total = global().map(|g| g.device.total_memory).unwrap_or(0);
    if device_used() + size as u64 > total {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    let Some((p, layout)) = host_alloc(size) else { return CUDA_ERROR_OUT_OF_MEMORY };
    if let Err(e) = backend::register_host(p, size as u64) {
        std::alloc::dealloc(p, std::alloc::Layout::from_size_align(layout, 4096).unwrap());
        return report(e, CUDA_ERROR_OUT_OF_MEMORY);
    }
    insert(p as u64, size as u64, Kind::Managed, flags, c.key(), layout, false);
    *dptr = p as u64;
    CUDA_SUCCESS
}

fn managed_range(p: u64, n: usize) -> bool {
    matches!(lookup(p), Some((b, a)) if a.kind == Kind::Managed && p + n as u64 <= b + a.size)
}

/// cuMemPrefetchAsync: a managed range (CUDA_ERROR_INVALID_VALUE otherwise) to the device or the host
/// (CU_DEVICE_CPU = -1); the memory is already where both see it
#[no_mangle]
pub unsafe extern "system" fn cuMemPrefetchAsync(p: CUdeviceptr, n: usize, dst: CUdevice, s: *mut Stream) -> CUresult {
    tri!(context::current());
    if dst != 0 && dst != -1 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    if !managed_range(p, n) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    stream::submit_to(s, Op::Empty)
}

/// CUmemLocation (cuMemPrefetchAsync_v2 / cuMemAdvise_v2)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemLocation {
    ty: c_int,
    id: c_int,
}

#[no_mangle]
pub unsafe extern "system" fn cuMemPrefetchAsync_v2(
    p: CUdeviceptr,
    n: usize,
    loc: CUmemLocation,
    flags: c_uint,
    s: *mut Stream,
) -> CUresult {
    tri!(context::current());
    if flags != 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    // CU_MEM_LOCATION_TYPE_DEVICE 1, HOST 2, HOST_NUMA 3, HOST_NUMA_CURRENT 4
    match loc.ty {
        1 => cuMemPrefetchAsync(p, n, loc.id, s),
        2..=4 => cuMemPrefetchAsync(p, n, -1, s),
        _ => CUDA_ERROR_INVALID_VALUE,
    }
}

/// cuMemAdvise: CU_MEM_ADVISE_SET/UNSET_READ_MOSTLY, PREFERRED_LOCATION, ACCESSED_BY (1 ..= 6) on a managed range:
/// hints
#[no_mangle]
pub unsafe extern "system" fn cuMemAdvise(p: CUdeviceptr, n: usize, advice: c_int, dev: CUdevice) -> CUresult {
    tri!(context::current());
    if !(1..=6).contains(&advice) || !managed_range(p, n) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if dev != 0 && dev != -1 && advice != 1 && advice != 2 {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuMemAdvise_v2(p: CUdeviceptr, n: usize, advice: c_int, loc: CUmemLocation) -> CUresult {
    cuMemAdvise(p, n, advice, if loc.ty == 1 { loc.id } else { -1 })
}

// ---------------------------------------------------------------------------
// Stream ordered allocator: the device's default pool
// ---------------------------------------------------------------------------
pub struct MemPool {
    /// CU_MEMPOOL_ATTR_*: REUSE_FOLLOW_EVENT_DEPENDENCIES 1, REUSE_ALLOW_OPPORTUNISTIC 2,
    /// REUSE_ALLOW_INTERNAL_DEPENDENCIES 3, RELEASE_THRESHOLD 4 (u64), RESERVED_MEM_CURRENT 5, RESERVED_MEM_HIGH 6,
    /// USED_MEM_CURRENT 7, USED_MEM_HIGH 8
    attrs: Mutex<[u64; 9]>,
}

fn default_pool() -> &'static MemPool {
    static P: OnceLock<MemPool> = OnceLock::new();
    P.get_or_init(|| MemPool { attrs: Mutex::new([0, 1, 1, 1, 0, 0, 0, 0, 0]) })
}

fn pool_used() -> u64 {
    allocs().values().filter(|a| a.pooled).map(|a| a.size).sum()
}

#[no_mangle]
pub unsafe extern "system" fn cuDeviceGetDefaultMemPool(pool: *mut *mut MemPool, dev: CUdevice) -> CUresult {
    tri!(check_device(dev));
    if pool.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool = default_pool() as *const MemPool as *mut MemPool;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDeviceGetMemPool(pool: *mut *mut MemPool, dev: CUdevice) -> CUresult {
    cuDeviceGetDefaultMemPool(pool, dev)
}

/// cuDeviceSetMemPool: only the default pool exists
#[no_mangle]
pub unsafe extern "system" fn cuDeviceSetMemPool(dev: CUdevice, pool: *mut MemPool) -> CUresult {
    tri!(check_device(dev));
    if pool as *const MemPool != default_pool() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_SUCCESS
}

fn check_pool(pool: *mut MemPool) -> Result<&'static MemPool, CUresult> {
    if pool as *const MemPool == default_pool() {
        Ok(default_pool())
    } else {
        Err(CUDA_ERROR_INVALID_VALUE)
    }
}

/// cuMemPoolGetAttribute: the reuse flags (int), the release threshold and the memory counters (cuuint64_t)
#[no_mangle]
pub unsafe extern "system" fn cuMemPoolGetAttribute(pool: *mut MemPool, attr: c_int, value: *mut c_void) -> CUresult {
    tri!(global());
    let p = tri!(check_pool(pool));
    if value.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let mut a = p.attrs.lock().unwrap();
    let used = pool_used();
    a[5] = used;
    a[7] = used;
    a[6] = a[6].max(used);
    a[8] = a[8].max(used);
    match attr {
        1..=3 => *(value as *mut c_int) = a[attr as usize] as c_int,
        4..=8 => *(value as *mut u64) = a[attr as usize],
        _ => return CUDA_ERROR_INVALID_VALUE,
    }
    CUDA_SUCCESS
}

/// cuMemPoolSetAttribute: the reuse flags and the release threshold; the high watermarks only reset to 0
#[no_mangle]
pub unsafe extern "system" fn cuMemPoolSetAttribute(pool: *mut MemPool, attr: c_int, value: *mut c_void) -> CUresult {
    tri!(global());
    let p = tri!(check_pool(pool));
    if value.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let mut a = p.attrs.lock().unwrap();
    match attr {
        1..=3 => a[attr as usize] = (*(value as *const c_int) != 0) as u64,
        4 => a[4] = *(value as *const u64),
        6 | 8 => {
            if *(value as *const u64) != 0 {
                return CUDA_ERROR_INVALID_VALUE;
            }
            a[attr as usize] = 0;
        }
        _ => return CUDA_ERROR_INVALID_VALUE,
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuMemPoolTrimTo(pool: *mut MemPool, _keep: usize) -> CUresult {
    tri!(global());
    tri!(check_pool(pool));
    CUDA_SUCCESS
}

/// cuMemAllocAsync: allocated at once (in stream order trivially), usable by the stream's later work
#[no_mangle]
pub unsafe extern "system" fn cuMemAllocAsync(dptr: *mut CUdeviceptr, size: usize, s: *mut Stream) -> CUresult {
    tri!(context::current());
    tri!(stream::resolve(s));
    alloc_device(dptr, size, true)
}

#[no_mangle]
pub unsafe extern "system" fn cuMemAllocFromPoolAsync(
    dptr: *mut CUdeviceptr,
    size: usize,
    pool: *mut MemPool,
    s: *mut Stream,
) -> CUresult {
    tri!(global());
    tri!(check_pool(pool));
    cuMemAllocAsync(dptr, size, s)
}

/// cuMemFreeAsync: freed when the stream's earlier work is done
#[no_mangle]
pub unsafe extern "system" fn cuMemFreeAsync(dptr: CUdeviceptr, s: *mut Stream) -> CUresult {
    tri!(context::current());
    if !matches!(lookup(dptr), Some((b, a)) if b == dptr && a.kind == Kind::Device) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    stream::submit_to(s, Op::Free(dptr))
}

// ---------------------------------------------------------------------------
// Pointer attributes
// ---------------------------------------------------------------------------
/// CUmemorytype
const CU_MEMORYTYPE_HOST: c_uint = 1;
const CU_MEMORYTYPE_DEVICE: c_uint = 2;
const CU_MEMORYTYPE_ARRAY: c_uint = 3;
const CU_MEMORYTYPE_UNIFIED: c_uint = 4;

/// What the driver knows about a pointer: the registry, or other device memory (module variables) as device
/// memory of the current context
fn pointer_info(p: u64) -> Option<(u64, Alloc)> {
    lookup(p).or_else(|| {
        backend::range(p).map(|(b, n)| {
            (b, Alloc {
                size: n,
                kind: Kind::Device,
                flags: 0,
                ctx: context::top().map_or(0, |c| c.key()),
                id: b,
                sync_memops: false,
                host_layout: 0,
                pooled: false,
            })
        })
    })
}

/// One attribute of a known pointer (or the default of an unknown one, `info` None); the size written follows
/// the attribute's documented type
unsafe fn write_attribute(data: *mut c_void, attr: c_int, p: u64, info: Option<(u64, Alloc)>) -> CUresult {
    let u32_ = |v: u32| *(data as *mut u32) = v;
    let u64_ = |v: u64| *(data as *mut u64) = v;
    let host = info.is_some_and(|(_, a)| a.kind != Kind::Device);
    match attr {
        1 => *(data as *mut usize) = info.map_or(0, |(_, a)| a.ctx), // CONTEXT
        2 => u32_(match info {
            // MEMORY_TYPE
            None => 0,
            Some((_, a)) if matches!(a.kind, Kind::Pinned | Kind::Registered) => CU_MEMORYTYPE_HOST,
            Some(_) => CU_MEMORYTYPE_DEVICE,
        }),
        3 => u64_(info.map_or(0, |_| p)), // DEVICE_POINTER
        4 => {
            // HOST_POINTER: CUDA_ERROR_INVALID_VALUE for memory the host cannot access
            if info.is_some() && !host {
                return CUDA_ERROR_INVALID_VALUE;
            }
            *(data as *mut usize) = if host { p as usize } else { 0 };
        }
        6 => u32_(info.is_some_and(|(_, a)| a.sync_memops) as u32), // SYNC_MEMOPS
        7 => u64_(info.map_or(0, |(_, a)| a.id)),                   // BUFFER_ID
        8 => u32_(info.is_some_and(|(_, a)| a.kind == Kind::Managed) as u32), // IS_MANAGED
        9 => *(data as *mut c_int) = if info.is_some() { 0 } else { -2 }, // DEVICE_ORDINAL
        10 => u32_(0),                                              // IS_LEGACY_CUDA_IPC_CAPABLE
        11 => u64_(info.map_or(0, |(b, _)| b)),                     // RANGE_START_ADDR
        12 => *(data as *mut usize) = info.map_or(0, |(_, a)| a.size as usize), // RANGE_SIZE
        13 => u32_(info.is_some() as u32),                          // MAPPED
        14 => u64_(0),                                              // ALLOWED_HANDLE_TYPES
        15 => u32_(0),                                              // IS_GPU_DIRECT_RDMA_CAPABLE
        16 => u32_(match info {
            // ACCESS_FLAGS: NONE 0, READ 1, READWRITE 3
            None => 0,
            Some((_, a)) if a.kind == Kind::Registered && a.flags & 8 != 0 => 1,
            Some(_) => 3,
        }),
        17 => {
            // MEMPOOL_HANDLE
            let pooled = info.is_some_and(|(_, a)| a.pooled);
            *(data as *mut usize) = if pooled { default_pool() as *const MemPool as usize } else { 0 };
        }
        18 => *(data as *mut usize) = info.map_or(0, |(_, a)| a.size as usize), // MAPPING_SIZE
        19 => u64_(info.map_or(0, |(b, _)| b)),                                  // MAPPING_BASE_ADDR
        20 => u64_(info.map_or(0, |(_, a)| a.id)),                               // MEMORY_BLOCK_ID
        21 => u32_(0),                                                            // IS_HW_DECOMPRESS_CAPABLE
        _ => return CUDA_ERROR_INVALID_VALUE,
    }
    CUDA_SUCCESS
}

/// cuPointerGetAttribute: CUDA_ERROR_INVALID_VALUE for a pointer the driver does not know ("if ptr was not
/// allocated by, mapped by, or registered with a CUcontext which uses unified virtual addressing"), an unknown
/// attribute, or HOST_POINTER of memory the host cannot access
#[no_mangle]
pub unsafe extern "system" fn cuPointerGetAttribute(data: *mut c_void, attr: c_int, p: CUdeviceptr) -> CUresult {
    tri!(global());
    if data.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let Some(info) = pointer_info(p) else { return CUDA_ERROR_INVALID_VALUE };
    write_attribute(data, attr, p, Some(info))
}

/// cuPointerGetAttributes: "Unlike cuPointerGetAttribute, this function will not return an error when the ptr
/// encountered is not a valid CUDA pointer. Instead, the attributes are assigned default NULL values"; an
/// attribute the memory does not have (HOST_POINTER of device memory) is NULL too
#[no_mangle]
pub unsafe extern "system" fn cuPointerGetAttributes(
    n: c_uint,
    attrs: *const c_int,
    data: *mut *mut c_void,
    p: CUdeviceptr,
) -> CUresult {
    tri!(global());
    if n > 0 && (attrs.is_null() || data.is_null()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let info = pointer_info(p);
    for i in 0..n as usize {
        let (a, d) = (*attrs.add(i), *data.add(i));
        if d.is_null() {
            return CUDA_ERROR_INVALID_VALUE;
        }
        match write_attribute(d, a, p, info) {
            CUDA_SUCCESS => {}
            _ if a == 4 => *(d as *mut usize) = 0,
            e => return e,
        }
    }
    CUDA_SUCCESS
}

/// cuPointerSetAttribute: only CU_POINTER_ATTRIBUTE_SYNC_MEMOPS is settable
#[no_mangle]
pub unsafe extern "system" fn cuPointerSetAttribute(value: *const c_void, attr: c_int, p: CUdeviceptr) -> CUresult {
    tri!(context::current());
    if value.is_null() || attr != 6 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let Some((b, _)) = lookup(p) else { return CUDA_ERROR_INVALID_VALUE };
    if let Some(a) = allocs().get_mut(&b) {
        a.sync_memops = *(value as *const u32) != 0;
    }
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Copies and sets
// ---------------------------------------------------------------------------
/// One side of a copy: a device address, or host memory
#[derive(Clone, Copy, Debug)]
pub(crate) struct End {
    pub addr: u64,
    pub device: bool,
}

/// A copy of `depth` slices of `height` rows of `width` bytes (pitches in bytes, slice heights in rows)
#[derive(Clone, Debug)]
pub(crate) struct Copy3D {
    pub dst: End,
    pub src: End,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub src_pitch: usize,
    pub src_height: usize,
    pub dst_pitch: usize,
    pub dst_height: usize,
}

/// A memset of `height` rows of `width` elements of `size` bytes
#[derive(Clone, Debug)]
pub(crate) struct Set2D {
    pub dst: u64,
    pub pitch: usize,
    pub value: u32,
    pub size: usize,
    pub width: usize,
    pub height: usize,
}

fn extent(width: usize, height: usize, depth: usize, pitch: usize, slice_rows: usize) -> usize {
    if width == 0 || height == 0 || depth == 0 {
        return 0;
    }
    ((depth - 1) * slice_rows + (height - 1)) * pitch + width
}

/// A unified address: device memory the driver knows, host memory otherwise
pub(crate) fn unified(addr: u64) -> End {
    End { addr, device: backend::host_pointer(addr).is_some() }
}

/// Check that `n` bytes at a device address are one allocation
fn check_device_range(addr: u64, n: usize) -> Result<(), CUresult> {
    if n == 0 {
        return Ok(());
    }
    match backend::host_pointer(addr) {
        Some((_, left)) if left as usize >= n => Ok(()),
        _ => Err(report(format!("invalid device range 0x{addr:x} + {n}"), CUDA_ERROR_INVALID_VALUE)),
    }
}

fn check_end(e: End, n: usize) -> Result<(), CUresult> {
    if n == 0 {
        return Ok(());
    }
    if e.device {
        check_device_range(e.addr, n)
    } else if e.addr == 0 {
        Err(CUDA_ERROR_INVALID_VALUE)
    } else {
        Ok(())
    }
}

impl Copy3D {
    pub(crate) fn linear(dst: End, src: End, n: usize) -> Copy3D {
        Copy3D { dst, src, width: n, height: 1, depth: 1, src_pitch: n, src_height: 1, dst_pitch: n, dst_height: 1 }
    }
    fn validate(&self) -> Result<(), CUresult> {
        if (self.height > 1 || self.depth > 1) && (self.src_pitch < self.width || self.dst_pitch < self.width) {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        if self.depth > 1 && (self.src_height < self.height || self.dst_height < self.height) {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        if self.src_pitch > i32::MAX as usize || self.dst_pitch > i32::MAX as usize {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        check_end(self.src, extent(self.width, self.height, self.depth, self.src_pitch, self.src_height))?;
        check_end(self.dst, extent(self.width, self.height, self.depth, self.dst_pitch, self.dst_height))
    }
    /// Run it (stream order, the device idle)
    pub(crate) unsafe fn run(&self) -> CUresult {
        let resolve = |e: End| -> Option<*mut u8> {
            if e.device {
                backend::host_pointer(e.addr).map(|(p, _)| p)
            } else {
                Some(e.addr as *mut u8)
            }
        };
        let (Some(s), Some(d)) = (resolve(self.src), resolve(self.dst)) else {
            return CUDA_ERROR_INVALID_VALUE;
        };
        if self.width == 0 {
            return CUDA_SUCCESS;
        }
        let _device = crate::agx::device_lock();
        for z in 0..self.depth {
            for y in 0..self.height {
                let so = (z * self.src_height + y) * self.src_pitch;
                let dof = (z * self.dst_height + y) * self.dst_pitch;
                ptr::copy(s.add(so), d.add(dof), self.width);
            }
        }
        CUDA_SUCCESS
    }
}

impl Set2D {
    fn validate(&self) -> Result<(), CUresult> {
        if self.height > 1 && self.pitch < self.width * self.size {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        if self.dst % self.size as u64 != 0 {
            // cuMemsetD16 / D32: "dstDevice must be 2 / 4 byte aligned"
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        check_device_range(self.dst, extent(self.width * self.size, self.height, 1, self.pitch, 1))
    }
    pub(crate) unsafe fn run(&self) -> CUresult {
        let Some((d, _)) = backend::host_pointer(self.dst) else { return CUDA_ERROR_INVALID_VALUE };
        let bytes = self.value.to_ne_bytes();
        let _device = crate::agx::device_lock();
        for y in 0..self.height {
            let row = d.add(y * self.pitch);
            match self.size {
                1 => ptr::write_bytes(row, bytes[0], self.width),
                _ => {
                    for i in 0..self.width {
                        ptr::copy_nonoverlapping(bytes.as_ptr(), row.add(i * self.size), self.size);
                    }
                }
            }
        }
        CUDA_SUCCESS
    }
}

/// Submit a copy: on `stream` (async), or on the legacy stream, waiting for it (the synchronous forms)
unsafe fn copy(c: Copy3D, stream: Option<*mut Stream>) -> CUresult {
    tri!(context::current());
    tri!(c.validate());
    match stream {
        Some(s) => stream::submit_to(s, Op::Copy(c)),
        None => stream::submit_sync(Op::Copy(c)),
    }
}

unsafe fn set(m: Set2D, stream: Option<*mut Stream>) -> CUresult {
    tri!(context::current());
    tri!(m.validate());
    match stream {
        Some(s) => stream::submit_to(s, Op::Set(m)),
        None => stream::submit_sync(Op::Set(m)),
    }
}

const DEV: bool = true;
const HOST: bool = false;
fn end(addr: u64, device: bool) -> End {
    End { addr, device }
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, n: usize) -> CUresult {
    copy(Copy3D::linear(end(dst, DEV), end(src as u64, HOST), n), None)
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, n: usize) -> CUresult {
    copy(Copy3D::linear(end(dst as u64, HOST), end(src, DEV), n), None)
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpyDtoD_v2(dst: CUdeviceptr, src: CUdeviceptr, n: usize) -> CUresult {
    copy(Copy3D::linear(end(dst, DEV), end(src, DEV), n), None)
}

/// cuMemcpy: unified addressing, either side may be host or device memory
#[no_mangle]
pub unsafe extern "system" fn cuMemcpy(dst: CUdeviceptr, src: CUdeviceptr, n: usize) -> CUresult {
    copy(Copy3D::linear(unified(dst), unified(src), n), None)
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpyHtoDAsync_v2(dst: CUdeviceptr, src: *const c_void, n: usize, s: *mut Stream) -> CUresult {
    copy(Copy3D::linear(end(dst, DEV), end(src as u64, HOST), n), Some(s))
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: CUdeviceptr, n: usize, s: *mut Stream) -> CUresult {
    copy(Copy3D::linear(end(dst as u64, HOST), end(src, DEV), n), Some(s))
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpyDtoDAsync_v2(dst: CUdeviceptr, src: CUdeviceptr, n: usize, s: *mut Stream) -> CUresult {
    copy(Copy3D::linear(end(dst, DEV), end(src, DEV), n), Some(s))
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpyAsync(dst: CUdeviceptr, src: CUdeviceptr, n: usize, s: *mut Stream) -> CUresult {
    copy(Copy3D::linear(unified(dst), unified(src), n), Some(s))
}

/// cuMemcpyPeer: one device, so a copy between its contexts' memory
#[no_mangle]
pub unsafe extern "system" fn cuMemcpyPeer(
    dst: CUdeviceptr,
    _dst_ctx: *mut Context,
    src: CUdeviceptr,
    _src_ctx: *mut Context,
    n: usize,
) -> CUresult {
    copy(Copy3D::linear(end(dst, DEV), end(src, DEV), n), None)
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpyPeerAsync(
    dst: CUdeviceptr,
    _dst_ctx: *mut Context,
    src: CUdeviceptr,
    _src_ctx: *mut Context,
    n: usize,
    s: *mut Stream,
) -> CUresult {
    copy(Copy3D::linear(end(dst, DEV), end(src, DEV), n), Some(s))
}

/// CUDA_MEMCPY2D
#[repr(C)]
pub struct CUDA_MEMCPY2D {
    src_x: usize,
    src_y: usize,
    src_type: c_uint,
    src_host: *const c_void,
    src_device: CUdeviceptr,
    src_array: *mut c_void,
    src_pitch: usize,
    dst_x: usize,
    dst_y: usize,
    dst_type: c_uint,
    dst_host: *mut c_void,
    dst_device: CUdeviceptr,
    dst_array: *mut c_void,
    dst_pitch: usize,
    width: usize,
    height: usize,
}

/// CUDA_MEMCPY3D
#[repr(C)]
pub struct CUDA_MEMCPY3D {
    src_x: usize,
    src_y: usize,
    src_z: usize,
    src_lod: usize,
    src_type: c_uint,
    src_host: *const c_void,
    src_device: CUdeviceptr,
    src_array: *mut c_void,
    reserved0: *mut c_void,
    src_pitch: usize,
    src_height: usize,
    dst_x: usize,
    dst_y: usize,
    dst_z: usize,
    dst_lod: usize,
    dst_type: c_uint,
    dst_host: *mut c_void,
    dst_device: CUdeviceptr,
    dst_array: *mut c_void,
    reserved1: *mut c_void,
    dst_pitch: usize,
    dst_height: usize,
    width: usize,
    height: usize,
    depth: usize,
}

/// One side of a 2D / 3D copy: its memory type selects the host pointer or the device pointer (UNIFIED: the
/// pointer's own kind); arrays are not supported (CUDA_ERROR_NOT_SUPPORTED: no array can be created)
fn copy_end(ty: c_uint, host: u64, device: CUdeviceptr, offset: usize) -> Result<End, CUresult> {
    match ty {
        CU_MEMORYTYPE_HOST => Ok(end(host + offset as u64, HOST)),
        CU_MEMORYTYPE_DEVICE => Ok(end(device + offset as u64, DEV)),
        CU_MEMORYTYPE_UNIFIED => Ok(unified(device + offset as u64)),
        CU_MEMORYTYPE_ARRAY => Err(CUDA_ERROR_NOT_SUPPORTED),
        _ => Err(CUDA_ERROR_INVALID_VALUE),
    }
}

pub(crate) unsafe fn copy3d_from(p: &CUDA_MEMCPY3D) -> Result<Copy3D, CUresult> {
    if p.src_lod != 0 || p.dst_lod != 0 {
        return Err(CUDA_ERROR_INVALID_VALUE);
    }
    let so = (p.src_z * p.src_height + p.src_y) * p.src_pitch + p.src_x;
    let dof = (p.dst_z * p.dst_height + p.dst_y) * p.dst_pitch + p.dst_x;
    Ok(Copy3D {
        src: copy_end(p.src_type, p.src_host as u64, p.src_device, so)?,
        dst: copy_end(p.dst_type, p.dst_host as u64, p.dst_device, dof)?,
        width: p.width,
        height: p.height,
        depth: p.depth,
        src_pitch: if p.height > 1 || p.depth > 1 { p.src_pitch } else { p.width },
        src_height: p.src_height.max(1),
        dst_pitch: if p.height > 1 || p.depth > 1 { p.dst_pitch } else { p.width },
        dst_height: p.dst_height.max(1),
    })
}

unsafe fn copy2d_from(p: &CUDA_MEMCPY2D) -> Result<Copy3D, CUresult> {
    Ok(Copy3D {
        src: copy_end(p.src_type, p.src_host as u64, p.src_device, p.src_y * p.src_pitch + p.src_x)?,
        dst: copy_end(p.dst_type, p.dst_host as u64, p.dst_device, p.dst_y * p.dst_pitch + p.dst_x)?,
        width: p.width,
        height: p.height,
        depth: 1,
        src_pitch: if p.height > 1 { p.src_pitch } else { p.width },
        src_height: p.height,
        dst_pitch: if p.height > 1 { p.dst_pitch } else { p.width },
        dst_height: p.height,
    })
}

/// cuMemcpy2D: CUDA_ERROR_INVALID_VALUE for a NULL description, a pitch below WidthInBytes, or a range outside
/// the memory
#[no_mangle]
pub unsafe extern "system" fn cuMemcpy2D_v2(p: *const CUDA_MEMCPY2D) -> CUresult {
    tri!(context::current());
    if p.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    copy(tri!(copy2d_from(&*p)), None)
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpy2DUnaligned_v2(p: *const CUDA_MEMCPY2D) -> CUresult {
    cuMemcpy2D_v2(p)
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpy2DAsync_v2(p: *const CUDA_MEMCPY2D, s: *mut Stream) -> CUresult {
    tri!(context::current());
    if p.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    copy(tri!(copy2d_from(&*p)), Some(s))
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpy3D_v2(p: *const CUDA_MEMCPY3D) -> CUresult {
    tri!(context::current());
    if p.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    copy(tri!(copy3d_from(&*p)), None)
}

#[no_mangle]
pub unsafe extern "system" fn cuMemcpy3DAsync_v2(p: *const CUDA_MEMCPY3D, s: *mut Stream) -> CUresult {
    tri!(context::current());
    if p.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    copy(tri!(copy3d_from(&*p)), Some(s))
}

fn set1(dst: CUdeviceptr, value: u32, size: usize, n: usize) -> Set2D {
    Set2D { dst, pitch: n * size, value, size, width: n, height: 1 }
}
fn set2(dst: CUdeviceptr, pitch: usize, value: u32, size: usize, width: usize, height: usize) -> Set2D {
    Set2D { dst, pitch, value, size, width, height }
}

#[no_mangle]
pub unsafe extern "system" fn cuMemsetD8_v2(dst: CUdeviceptr, v: u8, n: usize) -> CUresult {
    set(set1(dst, v as u32, 1, n), None)
}
#[no_mangle]
pub unsafe extern "system" fn cuMemsetD16_v2(dst: CUdeviceptr, v: u16, n: usize) -> CUresult {
    set(set1(dst, v as u32, 2, n), None)
}
#[no_mangle]
pub unsafe extern "system" fn cuMemsetD32_v2(dst: CUdeviceptr, v: u32, n: usize) -> CUresult {
    set(set1(dst, v, 4, n), None)
}
#[no_mangle]
pub unsafe extern "system" fn cuMemsetD8Async(dst: CUdeviceptr, v: u8, n: usize, s: *mut Stream) -> CUresult {
    set(set1(dst, v as u32, 1, n), Some(s))
}
#[no_mangle]
pub unsafe extern "system" fn cuMemsetD16Async(dst: CUdeviceptr, v: u16, n: usize, s: *mut Stream) -> CUresult {
    set(set1(dst, v as u32, 2, n), Some(s))
}
#[no_mangle]
pub unsafe extern "system" fn cuMemsetD32Async(dst: CUdeviceptr, v: u32, n: usize, s: *mut Stream) -> CUresult {
    set(set1(dst, v, 4, n), Some(s))
}
#[no_mangle]
pub unsafe extern "system" fn cuMemsetD2D8_v2(dst: CUdeviceptr, pitch: usize, v: u8, w: usize, h: usize) -> CUresult {
    set(set2(dst, pitch, v as u32, 1, w, h), None)
}
#[no_mangle]
pub unsafe extern "system" fn cuMemsetD2D16_v2(dst: CUdeviceptr, pitch: usize, v: u16, w: usize, h: usize) -> CUresult {
    set(set2(dst, pitch, v as u32, 2, w, h), None)
}
#[no_mangle]
pub unsafe extern "system" fn cuMemsetD2D32_v2(dst: CUdeviceptr, pitch: usize, v: u32, w: usize, h: usize) -> CUresult {
    set(set2(dst, pitch, v, 4, w, h), None)
}
#[no_mangle]
pub unsafe extern "system" fn cuMemsetD2D8Async(dst: CUdeviceptr, pitch: usize, v: u8, w: usize, h: usize, s: *mut Stream) -> CUresult {
    set(set2(dst, pitch, v as u32, 1, w, h), Some(s))
}
#[no_mangle]
pub unsafe extern "system" fn cuMemsetD2D16Async(dst: CUdeviceptr, pitch: usize, v: u16, w: usize, h: usize, s: *mut Stream) -> CUresult {
    set(set2(dst, pitch, v as u32, 2, w, h), Some(s))
}
#[no_mangle]
pub unsafe extern "system" fn cuMemsetD2D32Async(dst: CUdeviceptr, pitch: usize, v: u32, w: usize, h: usize, s: *mut Stream) -> CUresult {
    set(set2(dst, pitch, v, 4, w, h), Some(s))
}

/// CUDA_MEMSET_NODE_PARAMS (graph memset nodes)
#[repr(C)]
pub struct CUDA_MEMSET_NODE_PARAMS {
    pub dst: CUdeviceptr,
    pub pitch: usize,
    pub value: c_uint,
    pub element_size: c_uint,
    pub width: usize,
    pub height: usize,
}

pub(crate) fn set_from(p: &CUDA_MEMSET_NODE_PARAMS) -> Result<Set2D, CUresult> {
    if ![1, 2, 4].contains(&p.element_size) {
        return Err(CUDA_ERROR_INVALID_VALUE);
    }
    let m = Set2D {
        dst: p.dst,
        pitch: if p.height > 1 { p.pitch } else { p.width * p.element_size as usize },
        value: p.value,
        size: p.element_size as usize,
        width: p.width,
        height: p.height,
    };
    m.validate()?;
    Ok(m)
}

pub(crate) fn validate_copy(c: &Copy3D) -> Result<(), CUresult> {
    c.validate()
}
