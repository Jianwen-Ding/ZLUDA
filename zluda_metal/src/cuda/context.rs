//! Contexts (CUDA Driver API reference, "Context Management" and "Primary Context Management"):
//! a per-thread stack of current contexts, primary context reference counting, limits and configuration.
//! A context owns the allocations, modules, streams and events made in it; destroying it (or releasing / resetting
//! the primary context) frees them.
use super::*;
use std::cell::RefCell;

/// cuCtxGetApiVersion of every context
const API_VERSION: c_uint = 3020;


pub struct Context {
    pub(crate) sticky: AtomicU32,
    pub(crate) primary: bool,
    pub(crate) id: u64,
    pub(crate) flags: AtomicU32,
    pub(crate) state: Mutex<CtxState>,
    /// the legacy default stream (NULL, CU_STREAM_LEGACY, CU_STREAM_PER_THREAD)
    pub(crate) null_stream: OnceLock<Arc<Stream>>,
}

pub(crate) struct CtxState {
    /// cuCtxDestroy'd (never for the primary context)
    pub destroyed: bool,
    /// primary context: cuDevicePrimaryCtxRetain count, and whether it is initialized (retained and used since
    /// its last reset)
    pub refs: u32,
    pub active: bool,
    pub cache_config: u32,
    pub shmem_config: u32,
}

impl Context {
    pub(crate) fn new(primary: bool, flags: u32) -> Context {
        static IDS: AtomicU64 = AtomicU64::new(1);
        Context {
            sticky: AtomicU32::new(0),
            primary,
            id: IDS.fetch_add(1, Ordering::Relaxed),
            flags: AtomicU32::new(flags),
            state: Mutex::new(CtxState {
                destroyed: false,
                refs: 0,
                active: false,
                cache_config: 0,
                shmem_config: 0,
            }),
            null_stream: OnceLock::new(),
        }
    }
    pub(crate) fn key(&self) -> usize {
        self as *const Context as usize
    }
}

/// Context structures are never freed (a destroyed context's handle stays recognizable: other threads that still
/// have it current get CUDA_ERROR_CONTEXT_IS_DESTROYED); `LIVE` holds the usable ones
static LIVE: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

pub(crate) fn register(c: *const Context) {
    LIVE.lock().unwrap().get_or_insert_with(HashSet::new).insert(c as usize);
}

fn live(c: *mut Context) -> Option<&'static Context> {
    let ok = !c.is_null() && LIVE.lock().unwrap().as_ref().is_some_and(|s| s.contains(&(c as usize)));
    ok.then(|| unsafe { &*c })
}

thread_local! {
    static STACK: RefCell<Vec<*mut Context>> = const { RefCell::new(Vec::new()) };
}

/// The top of this thread's context stack, whatever its state
pub(crate) fn top() -> Option<&'static Context> {
    STACK.with(|s| s.borrow().last().map(|&c| unsafe { &*c }))
}

/// A stream worker thread runs its jobs with the stream's context current
pub(crate) fn set_thread_context(c: *const Context) {
    STACK.with(|s| *s.borrow_mut() = vec![c as *mut Context]);
}

/// The current context, usable: CUDA_ERROR_INVALID_CONTEXT without one, CUDA_ERROR_CONTEXT_IS_DESTROYED for a
/// destroyed context or a primary context that is not retained ("a primary context which has not yet been
/// initialized"), the context's sticky error after a kernel fault
pub(crate) fn current() -> Result<&'static Context, CUresult> {
    current_unchecked().and_then(|c| match c.sticky.load(Ordering::Acquire) {
        0 => Ok(c),
        e => Err(e),
    })
}

/// The current context, without its sticky error
pub(crate) fn current_unchecked() -> Result<&'static Context, CUresult> {
    global()?;
    let c = top().ok_or(CUDA_ERROR_INVALID_CONTEXT)?;
    let mut st = c.state.lock().unwrap();
    if st.destroyed || (c.primary && st.refs == 0) {
        return Err(CUDA_ERROR_CONTEXT_IS_DESTROYED);
    }
    // a retained primary context that was reset is initialized again by its next use
    st.active = true;
    Ok(c)
}

/// CU_CTX_SCHED_* (one of AUTO 0, SPIN 1, YIELD 2, BLOCKING_SYNC 4), MAP_HOST 8, LMEM_RESIZE_TO_MAX 0x10,
/// COREDUMP_ENABLE 0x20, USER_COREDUMP_ENABLE 0x40, SYNC_MEMOPS 0x80
fn valid_flags(flags: c_uint) -> bool {
    flags & !0xFF == 0 && matches!(flags & 7, 0 | 1 | 2 | 4)
}

/// Free everything a context owns: wait for its streams, then its streams, events, modules and allocations
unsafe fn teardown(c: &Context) {
    stream::destroy_context_streams(c);
    // printf output, the device heap and FIFO, the limits (src/device_rt.rs)
    crate::device_rt::context_destroyed(c.key());
    for m in modules_of(c.key()) {
        unload_module(m);
    }
    memory::free_context(c.key());
    c.sticky.store(0, Ordering::Release);
}

// ---------------------------------------------------------------------------
// Primary context
// ---------------------------------------------------------------------------
/// cuDevicePrimaryCtxRetain: increments the primary context's reference count (initializing it)
#[no_mangle]
pub unsafe extern "system" fn cuDevicePrimaryCtxRetain(ctx: *mut *mut Context, dev: CUdevice) -> CUresult {
    let g = tri!(check_device(dev));
    if ctx.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let mut st = g.primary.state.lock().unwrap();
    st.refs += 1;
    st.active = true;
    *ctx = &*g.primary as *const Context as *mut Context;
    CUDA_SUCCESS
}

/// cuDevicePrimaryCtxRelease: decrements the count; at zero the primary context is destroyed (its resources
/// freed). It is not popped from any context stack. Releasing a context that is not retained:
/// CUDA_ERROR_INVALID_CONTEXT.
#[no_mangle]
pub unsafe extern "system" fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult {
    let g = tri!(check_device(dev));
    {
        let mut st = g.primary.state.lock().unwrap();
        if st.refs == 0 {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        st.refs -= 1;
        if st.refs > 0 {
            return CUDA_SUCCESS;
        }
        st.active = false;
    }
    teardown(&g.primary);
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDevicePrimaryCtxRelease(dev: CUdevice) -> CUresult {
    cuDevicePrimaryCtxRelease_v2(dev)
}

/// cuDevicePrimaryCtxReset: "Destroy all allocations and reset all state on the primary context" (its sticky
/// error too). "Resetting the primary context does not release it": the reference count stays and the next use
/// initializes it again.
#[no_mangle]
pub unsafe extern "system" fn cuDevicePrimaryCtxReset_v2(dev: CUdevice) -> CUresult {
    let g = tri!(check_device(dev));
    teardown(&g.primary);
    let mut st = g.primary.state.lock().unwrap();
    st.active = false;
    st.cache_config = 0;
    st.shmem_config = 0;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDevicePrimaryCtxReset(dev: CUdevice) -> CUresult {
    cuDevicePrimaryCtxReset_v2(dev)
}

/// cuDevicePrimaryCtxGetState: the flags and whether the primary context is active
#[no_mangle]
pub unsafe extern "system" fn cuDevicePrimaryCtxGetState(
    dev: CUdevice,
    flags: *mut c_uint,
    active: *mut c_int,
) -> CUresult {
    let g = tri!(check_device(dev));
    if flags.is_null() || active.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let st = g.primary.state.lock().unwrap();
    *flags = g.primary.flags.load(Ordering::Relaxed);
    *active = (st.refs > 0 && st.active) as c_int;
    CUDA_SUCCESS
}

/// cuDevicePrimaryCtxSetFlags: the flags the primary context uses (since CUDA 11 also while it is active);
/// invalid flags: CUDA_ERROR_INVALID_VALUE
#[no_mangle]
pub unsafe extern "system" fn cuDevicePrimaryCtxSetFlags_v2(dev: CUdevice, flags: c_uint) -> CUresult {
    let g = tri!(check_device(dev));
    if !valid_flags(flags) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    g.primary.flags.store(flags, Ordering::Relaxed);
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuDevicePrimaryCtxSetFlags(dev: CUdevice, flags: c_uint) -> CUresult {
    cuDevicePrimaryCtxSetFlags_v2(dev, flags)
}

// ---------------------------------------------------------------------------
// Context creation and the context stack
// ---------------------------------------------------------------------------
/// cuCtxCreate: a new context, pushed onto the calling thread's stack. CUDA_ERROR_INVALID_VALUE for a NULL output
/// or invalid flags, CUDA_ERROR_INVALID_DEVICE for an invalid device.
#[no_mangle]
pub unsafe extern "system" fn cuCtxCreate_v2(ctx: *mut *mut Context, flags: c_uint, dev: CUdevice) -> CUresult {
    tri!(check_device(dev));
    if ctx.is_null() || !valid_flags(flags) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let c = Box::into_raw(Box::new(Context::new(false, flags)));
    register(c);
    STACK.with(|s| s.borrow_mut().push(c));
    *ctx = c;
    CUDA_SUCCESS
}

/// cuCtxCreate_v3 (CUDA 11.4): execution affinity (SM count partitions) is not supported:
/// CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY when given
#[no_mangle]
pub unsafe extern "system" fn cuCtxCreate_v3(
    ctx: *mut *mut Context,
    params: *mut c_void,
    num_params: c_int,
    flags: c_uint,
    dev: CUdevice,
) -> CUresult {
    tri!(check_device(dev));
    if num_params < 0 || (num_params > 0 && params.is_null()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if num_params > 0 {
        return CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY;
    }
    cuCtxCreate_v2(ctx, flags, dev)
}

/// CUctxCreateParams (cuCtxCreate_v4, CUDA 12.5)
#[repr(C)]
pub struct CUctxCreateParams {
    exec_affinity: *mut c_void,
    num_exec_affinity: c_int,
    cig: *mut c_void,
}

#[no_mangle]
pub unsafe extern "system" fn cuCtxCreate_v4(
    ctx: *mut *mut Context,
    params: *const CUctxCreateParams,
    flags: c_uint,
    dev: CUdevice,
) -> CUresult {
    tri!(check_device(dev));
    if !params.is_null() {
        let p = &*params;
        if !p.cig.is_null() {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        return cuCtxCreate_v3(ctx, p.exec_affinity, p.num_exec_affinity, flags, dev);
    }
    cuCtxCreate_v2(ctx, flags, dev)
}

/// cuCtxDestroy: frees the context's resources; "If ctx is current to the calling thread then ctx will also be
/// popped from the current thread's context stack"; other threads that have it current get
/// CUDA_ERROR_CONTEXT_IS_DESTROYED. An invalid (or already destroyed) context: CUDA_ERROR_INVALID_CONTEXT. A
/// primary context is released with cuDevicePrimaryCtxRelease, not destroyed: CUDA_ERROR_INVALID_CONTEXT.
#[no_mangle]
pub unsafe extern "system" fn cuCtxDestroy_v2(ctx: *mut Context) -> CUresult {
    tri!(global());
    let Some(c) = live(ctx) else { return CUDA_ERROR_INVALID_CONTEXT };
    if c.primary {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    LIVE.lock().unwrap().as_mut().unwrap().remove(&(ctx as usize));
    STACK.with(|s| {
        let mut s = s.borrow_mut();
        if s.last() == Some(&ctx) {
            s.pop();
        }
    });
    teardown(c);
    c.state.lock().unwrap().destroyed = true;
    CUDA_SUCCESS
}

/// cuCtxPushCurrent: pushes a valid context (CUDA_ERROR_INVALID_CONTEXT otherwise)
#[no_mangle]
pub unsafe extern "system" fn cuCtxPushCurrent_v2(ctx: *mut Context) -> CUresult {
    tri!(global());
    if live(ctx).is_none() {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    STACK.with(|s| s.borrow_mut().push(ctx));
    CUDA_SUCCESS
}

/// cuCtxPopCurrent: pops the current context and returns it (pctx may be NULL); an empty stack:
/// CUDA_ERROR_INVALID_CONTEXT
#[no_mangle]
pub unsafe extern "system" fn cuCtxPopCurrent_v2(pctx: *mut *mut Context) -> CUresult {
    tri!(global());
    match STACK.with(|s| s.borrow_mut().pop()) {
        Some(c) => {
            if !pctx.is_null() {
                *pctx = c;
            }
            CUDA_SUCCESS
        }
        None => CUDA_ERROR_INVALID_CONTEXT,
    }
}

/// cuCtxSetCurrent: replaces the top of the stack (pushes onto an empty one); NULL pops the top ("or a no-op if
/// the calling CPU thread's CUDA context stack is empty")
#[no_mangle]
pub unsafe extern "system" fn cuCtxSetCurrent(ctx: *mut Context) -> CUresult {
    tri!(global());
    if ctx.is_null() {
        STACK.with(|s| s.borrow_mut().pop());
        return CUDA_SUCCESS;
    }
    if live(ctx).is_none() {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    STACK.with(|s| {
        let mut s = s.borrow_mut();
        s.pop();
        s.push(ctx);
    });
    CUDA_SUCCESS
}

/// cuCtxGetCurrent: the top of the stack, NULL when there is none
#[no_mangle]
pub unsafe extern "system" fn cuCtxGetCurrent(ctx: *mut *mut Context) -> CUresult {
    tri!(global());
    if ctx.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *ctx = STACK.with(|s| s.borrow().last().copied().unwrap_or(ptr::null_mut()));
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuCtxGetDevice(dev: *mut CUdevice) -> CUresult {
    tri!(current());
    if dev.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *dev = 0;
    CUDA_SUCCESS
}

/// The context `ctx` names, or the current one for NULL
fn ctx_or_current(ctx: *mut Context) -> Result<&'static Context, CUresult> {
    global()?;
    if ctx.is_null() {
        current_unchecked()
    } else {
        live(ctx).ok_or(CUDA_ERROR_INVALID_CONTEXT)
    }
}

/// cuCtxGetApiVersion: 3020 for every context (NULL: the current one)
#[no_mangle]
pub unsafe extern "system" fn cuCtxGetApiVersion(ctx: *mut Context, version: *mut c_uint) -> CUresult {
    tri!(ctx_or_current(ctx));
    if version.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *version = API_VERSION;
    CUDA_SUCCESS
}

/// cuCtxGetId (CUDA 12.0): a unique id of the context (NULL: the current one)
#[no_mangle]
pub unsafe extern "system" fn cuCtxGetId(ctx: *mut Context, id: *mut u64) -> CUresult {
    let c = tri!(ctx_or_current(ctx));
    if id.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *id = c.id;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuCtxGetFlags(flags: *mut c_uint) -> CUresult {
    let c = tri!(current_unchecked());
    if flags.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *flags = c.flags.load(Ordering::Relaxed);
    CUDA_SUCCESS
}

/// cuCtxSetFlags (CUDA 12.1)
#[no_mangle]
pub unsafe extern "system" fn cuCtxSetFlags(flags: c_uint) -> CUresult {
    let c = tri!(current_unchecked());
    if !valid_flags(flags) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    c.flags.store(flags, Ordering::Relaxed);
    CUDA_SUCCESS
}

/// cuCtxSynchronize: waits for all work of the current context; returns its sticky error after a kernel fault
#[no_mangle]
pub unsafe extern "system" fn cuCtxSynchronize() -> CUresult {
    let c = tri!(current());
    stream::synchronize_context(c)
}

// cuCtxGetLimit / cuCtxSetLimit: src/device_rt.rs (the limits are the device runtime's)

/// cuCtxGetCacheConfig / cuCtxSetCacheConfig: CU_FUNC_CACHE_PREFER_NONE / SHARED / L1 / EQUAL (a hint)
#[no_mangle]
pub unsafe extern "system" fn cuCtxGetCacheConfig(config: *mut c_int) -> CUresult {
    let c = tri!(current());
    if config.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *config = c.state.lock().unwrap().cache_config as c_int;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuCtxSetCacheConfig(config: c_int) -> CUresult {
    let c = tri!(current());
    if !(0..=3).contains(&config) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    c.state.lock().unwrap().cache_config = config as u32;
    CUDA_SUCCESS
}

/// cuCtxGetSharedMemConfig / cuCtxSetSharedMemConfig (deprecated): bank size DEFAULT / FOUR / EIGHT_BYTE
#[no_mangle]
pub unsafe extern "system" fn cuCtxGetSharedMemConfig(config: *mut c_int) -> CUresult {
    let c = tri!(current());
    if config.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *config = c.state.lock().unwrap().shmem_config as c_int;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuCtxSetSharedMemConfig(config: c_int) -> CUresult {
    let c = tri!(current());
    if !(0..=2).contains(&config) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    c.state.lock().unwrap().shmem_config = config as u32;
    CUDA_SUCCESS
}

/// cuCtxGetStreamPriorityRange: least 0, greatest -5 (lower numbers are higher priorities); either pointer may
/// be NULL
#[no_mangle]
pub unsafe extern "system" fn cuCtxGetStreamPriorityRange(least: *mut c_int, greatest: *mut c_int) -> CUresult {
    tri!(current());
    if !least.is_null() {
        *least = stream::PRIORITY_LEAST;
    }
    if !greatest.is_null() {
        *greatest = stream::PRIORITY_GREATEST;
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuCtxResetPersistingL2Cache() -> CUresult {
    tri!(current());
    CUDA_SUCCESS
}

/// cuCtxGetExecAffinity: execution affinity is not supported
#[no_mangle]
pub unsafe extern "system" fn cuCtxGetExecAffinity(affinity: *mut c_void, _type: c_int) -> CUresult {
    tri!(current());
    if affinity.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY
}

// ---------------------------------------------------------------------------
// Peer access: one device, so no peer is ever accessible
// ---------------------------------------------------------------------------
/// cuCtxEnablePeerAccess: CUDA_ERROR_INVALID_VALUE for flags != 0, CUDA_ERROR_INVALID_CONTEXT for an invalid peer,
/// and CUDA_ERROR_INVALID_DEVICE since cuDeviceCanAccessPeer reports that the device cannot access the peer's
/// (it is the same device: its memory is already accessible)
#[no_mangle]
pub unsafe extern "system" fn cuCtxEnablePeerAccess(peer: *mut Context, flags: c_uint) -> CUresult {
    tri!(current());
    if flags != 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if live(peer).is_none() {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    CUDA_ERROR_INVALID_DEVICE
}

/// cuCtxDisablePeerAccess: "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED if direct peer access has not yet been enabled"
#[no_mangle]
pub unsafe extern "system" fn cuCtxDisablePeerAccess(peer: *mut Context) -> CUresult {
    tri!(current());
    if live(peer).is_none() {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED
}
