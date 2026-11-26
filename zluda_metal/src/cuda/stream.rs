//! Streams and events (CUDA Driver API reference, "Stream Management", "Event Management", and the stream
//! synchronization behavior notes).
//!
//! Each stream runs its work in order. Work submitted to an idle stream whose dependencies are complete runs at
//! once in the calling thread (kernels are simulated synchronously, so this is the common path); otherwise it is
//! queued for the stream's worker thread, which runs it after the stream's earlier work and its dependencies.
//! Host functions (cuLaunchHostFunc, cuStreamAddCallback) always run on the worker, never in the caller's thread.
//! The legacy default stream (NULL / CU_STREAM_LEGACY; CU_STREAM_PER_THREAD is treated as it) waits for all
//! blocking streams of its context, and blocking streams wait for it; CU_STREAM_NON_BLOCKING streams do not.
//! cuStreamQuery / cuEventQuery therefore return CUDA_ERROR_NOT_READY while work is queued behind a host function
//! or an event that has not completed.
//!
//! Event times: on the AGX backend an event records the simulated GPU clock (the AGX Performance Simulator's
//! predicted time of every kernel; kernels are timed once a timing event exists), so cuEventElapsedTime measures
//! simulated device time. `ZLUDA_EVENT_TIMING=wall` (and the Metal backend) use the host clock.
use super::memory::{Copy3D, Set2D};
use super::*;
use std::collections::VecDeque;
use std::sync::Condvar;
use std::time::Instant;

pub(crate) const PRIORITY_LEAST: c_int = 0;
pub(crate) const PRIORITY_GREATEST: c_int = -5;
const CU_STREAM_NON_BLOCKING: c_uint = 1;
const CU_STREAM_LEGACY: usize = 1;
const CU_STREAM_PER_THREAD: usize = 2;

/// A host function of cuLaunchHostFunc / a graph host node
pub(crate) type HostFn = unsafe extern "system" fn(*mut c_void);
/// cuStreamAddCallback's callback: (stream, status, user data)
pub(crate) type StreamCallback = unsafe extern "system" fn(*mut Stream, CUresult, *mut c_void);

/// One unit of stream work (also a graph node's work)
#[derive(Clone)]
pub(crate) enum Op {
    Kernel(KernelLaunch),
    Copy(Copy3D),
    Set(Set2D),
    Host(HostFn, usize),
    Callback(StreamCallback, usize, usize),
    /// an event record: the event's generation
    Record(Arc<Event>, u64),
    /// wait for (stream, sequence number)
    Wait(Arc<Stream>, u64),
    Free(CUdeviceptr),
    Empty,
    /// graph nodes recording / waiting for an event (turned into Record / Wait when submitted)
    RecordNode(Arc<Event>),
    WaitNode(Arc<Event>),
}

impl Op {
    /// always run on the stream's worker: host functions, and waits (which would block the submitting thread)
    fn deferred(&self) -> bool {
        matches!(self, Op::Host(..) | Op::Callback(..) | Op::Wait(..))
    }
}

pub struct Stream {
    pub(crate) ctx: usize,
    pub(crate) flags: c_uint,
    pub(crate) priority: c_int,
    pub(crate) id: u64,
    pub(crate) legacy: bool,
    q: Mutex<Queue>,
    cv: Condvar,
    /// stream capture (`cuda/graph.rs`)
    pub(crate) capture: Mutex<Option<graph::StreamCapture>>,
}
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

struct Queue {
    jobs: VecDeque<Job>,
    submitted: u64,
    done: u64,
    worker: bool,
}

struct Job {
    seq: u64,
    deps: Vec<(Arc<Stream>, u64)>,
    op: Op,
}

static STREAMS: Handles<Stream> = Handles::new();

impl Stream {
    fn new(ctx: usize, flags: c_uint, priority: c_int, legacy: bool) -> Stream {
        static IDS: AtomicU64 = AtomicU64::new(1);
        Stream {
            ctx,
            flags,
            priority,
            id: IDS.fetch_add(1, Ordering::Relaxed),
            legacy,
            q: Mutex::new(Queue { jobs: VecDeque::new(), submitted: 0, done: 0, worker: false }),
            cv: Condvar::new(),
            capture: Mutex::new(None),
        }
    }
    fn lock(&self) -> std::sync::MutexGuard<'_, Queue> {
        self.q.lock().unwrap_or_else(|e| e.into_inner())
    }
    pub(crate) fn submitted(&self) -> u64 {
        self.lock().submitted
    }
    pub(crate) fn done(&self) -> u64 {
        self.lock().done
    }
    pub(crate) fn idle(&self) -> bool {
        let q = self.lock();
        q.done == q.submitted
    }
    pub(crate) fn wait(&self, seq: u64) {
        let mut q = self.lock();
        while q.done < seq {
            q = self.cv.wait(q).unwrap_or_else(|e| e.into_inner());
        }
    }
    fn complete(&self, seq: u64) {
        let mut q = self.lock();
        q.done = q.done.max(seq);
        self.cv.notify_all();
    }
    pub(crate) fn blocking(&self) -> bool {
        !self.legacy && self.flags & CU_STREAM_NON_BLOCKING == 0
    }
    fn context(&self) -> &'static Context {
        unsafe { &*(self.ctx as *const Context) }
    }
    /// The handle an application sees (NULL for the legacy stream)
    pub(crate) fn handle(self: &Arc<Self>) -> *mut Stream {
        if self.legacy {
            ptr::null_mut()
        } else {
            Arc::as_ptr(self) as *mut Stream
        }
    }
}

/// A context's legacy default stream
pub(crate) fn legacy(c: &Context) -> Arc<Stream> {
    c.null_stream.get_or_init(|| Arc::new(Stream::new(c.key(), 0, 0, true))).clone()
}

/// The streams of a context (the legacy one included)
fn context_streams(c: &Context) -> Vec<Arc<Stream>> {
    let mut v: Vec<Arc<Stream>> = STREAMS.all().into_iter().filter(|s| s.ctx == c.key()).collect();
    v.push(legacy(c));
    v
}

/// A stream handle: NULL / CU_STREAM_LEGACY / CU_STREAM_PER_THREAD name the current context's legacy stream;
/// others must be live streams (CUDA_ERROR_INVALID_HANDLE)
pub(crate) fn resolve(h: *mut Stream) -> Result<Arc<Stream>, CUresult> {
    match h as usize {
        0 | CU_STREAM_LEGACY | CU_STREAM_PER_THREAD => Ok(legacy(context::current_unchecked()?)),
        _ => {
            global()?;
            STREAMS.get(h).ok_or(CUDA_ERROR_INVALID_HANDLE)
        }
    }
}

/// What work submitted to `s` now must wait for (the legacy stream's implicit synchronization)
fn implicit_deps(s: &Arc<Stream>) -> Vec<(Arc<Stream>, u64)> {
    let c = s.context();
    let mut deps = vec![];
    if s.legacy {
        for o in STREAMS.all() {
            if o.ctx == s.ctx && o.blocking() {
                let n = o.submitted();
                if o.done() < n {
                    deps.push((o, n));
                }
            }
        }
    } else if s.blocking() {
        let l = legacy(c);
        let n = l.submitted();
        if l.done() < n {
            deps.push((l, n));
        }
    }
    deps
}

/// Submit work to a stream handle of the current context: (result, sequence number of the work in its stream)
pub(crate) unsafe fn submit_to(h: *mut Stream, op: Op) -> CUresult {
    let c = tri!(context::current());
    let s = tri!(resolve(h));
    if s.ctx != c.key() {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    submit(&s, op).0
}

/// A synchronous operation: on the legacy stream, waited for
pub(crate) unsafe fn submit_sync(op: Op) -> CUresult {
    let c = tri!(context::current());
    let s = legacy(c);
    let (r, seq) = submit(&s, op);
    if r != CUDA_SUCCESS {
        return r;
    }
    s.wait(seq);
    c.sticky.load(Ordering::Acquire)
}

pub(crate) unsafe fn submit(s: &Arc<Stream>, op: Op) -> (CUresult, u64) {
    // graph nodes that act on events
    let op = match op {
        Op::RecordNode(e) => return (record(&e, s), s.submitted()),
        Op::WaitNode(e) => return (wait_event(s, &e), s.submitted()),
        op => op,
    };
    if let Some(r) = graph::capture_op(s, &op) {
        return (r, 0);
    }
    if s.legacy {
        if let Err(e) = graph::legacy_use(s.ctx) {
            return (e, 0);
        }
    }
    let deps = implicit_deps(s);
    let deps_done = deps.iter().all(|(d, n)| d.done() >= *n);
    let mut q = s.lock();
    q.submitted += 1;
    let seq = q.submitted;
    if !op.deferred() && deps_done && q.jobs.is_empty() && q.done + 1 == seq {
        drop(q);
        let r = exec(s, &op);
        s.complete(seq);
        return (r, seq);
    }
    q.jobs.push_back(Job { seq, deps, op });
    if !q.worker {
        q.worker = true;
        let s2 = s.clone();
        std::thread::Builder::new()
            .name("zluda-stream".into())
            .spawn(move || worker(s2))
            .expect("stream worker thread");
    }
    (CUDA_SUCCESS, seq)
}

fn worker(s: Arc<Stream>) {
    context::set_thread_context(s.ctx as *const Context);
    loop {
        let job = {
            let mut q = s.lock();
            match q.jobs.pop_front() {
                Some(j) => j,
                None => {
                    q.worker = false;
                    return;
                }
            }
        };
        // in order: an earlier piece of work may still run in the thread that submitted it
        s.wait(job.seq - 1);
        for (d, n) in &job.deps {
            d.wait(*n);
        }
        let r = unsafe { exec(&s, &job.op) };
        if r != CUDA_SUCCESS && std::env::var_os("ZLUDA_METAL_LOG").is_some() {
            eprintln!("[zluda metal] asynchronous stream work failed: {r}");
        }
        s.complete(job.seq);
    }
}

/// Run one piece of work. A context with a sticky error runs no more device work; callbacks get the error.
unsafe fn exec(s: &Arc<Stream>, op: &Op) -> CUresult {
    let sticky = s.context().sticky.load(Ordering::Acquire);
    match op {
        Op::Callback(f, h, data) => {
            f(*h as *mut Stream, sticky, *data as *mut c_void);
            return CUDA_SUCCESS;
        }
        Op::Host(f, data) => {
            if sticky == 0 {
                f(*data as *mut c_void);
            }
            return CUDA_SUCCESS;
        }
        Op::Record(e, gen) => {
            e.stamp(*gen);
            return CUDA_SUCCESS;
        }
        Op::Wait(o, seq) => {
            o.wait(*seq);
            return CUDA_SUCCESS;
        }
        _ => {}
    }
    if sticky != 0 {
        return sticky;
    }
    match op {
        Op::Kernel(l) => run_kernel(l),
        Op::Copy(c) => match host_access() {
            Ok(()) => {
                crate::device_rt::flush();
                c.run()
            }
            Err(e) => e,
        },
        Op::Set(m) => match host_access() {
            Ok(()) => m.run(),
            Err(e) => e,
        },
        Op::Free(p) => memory::free_now(*p),
        _ => CUDA_SUCCESS,
    }
}

/// The Metal backend runs kernels asynchronously: host access to device memory waits for them
fn host_access() -> Result<(), CUresult> {
    backend::synchronize().map_err(|e| {
        device_fault(e, CUDA_ERROR_LAUNCH_FAILED);
        CUDA_ERROR_LAUNCH_FAILED
    })
}

/// Wait for every stream of a context; its sticky error afterwards
pub(crate) fn synchronize_context(c: &Context) -> CUresult {
    if let Err(e) = graph::sync_during_capture(c.key(), None) {
        return e;
    }
    for s in context_streams(c) {
        s.wait(s.submitted());
    }
    crate::device_rt::flush();
    if let Err(e) = host_access() {
        return e;
    }
    c.sticky.load(Ordering::Acquire)
}

/// Wait until all streams of all contexts are idle (before freeing memory or unloading modules)
pub(crate) fn wait_all_idle() {
    // a host function waiting here for its own stream would never return
    if std::thread::current().name() == Some("zluda-stream") {
        return;
    }
    for s in STREAMS.all() {
        s.wait(s.submitted());
    }
    if let Some(c) = context::top() {
        let l = legacy(c);
        l.wait(l.submitted());
    }
    let _ = backend::synchronize();
}

/// A context's teardown: wait for its streams and forget them and its events
pub(crate) fn destroy_context_streams(c: &Context) {
    for s in context_streams(c) {
        s.wait(s.submitted());
        STREAMS.remove(Arc::as_ptr(&s));
        *s.capture.lock().unwrap() = None;
    }
    for e in EVENTS.all() {
        if e.ctx == c.key() {
            EVENTS.remove(Arc::as_ptr(&e));
        }
    }
    let _ = backend::synchronize();
}

// ---------------------------------------------------------------------------
// Stream API
// ---------------------------------------------------------------------------
/// cuStreamCreate: flags CU_STREAM_DEFAULT (0) or CU_STREAM_NON_BLOCKING (1); others CUDA_ERROR_INVALID_VALUE
#[no_mangle]
pub unsafe extern "system" fn cuStreamCreate(stream: *mut *mut Stream, flags: c_uint) -> CUresult {
    cuStreamCreateWithPriority(stream, flags, PRIORITY_LEAST)
}

/// cuStreamCreateWithPriority: "priorities outside the range [greatest, least] are clamped"
#[no_mangle]
pub unsafe extern "system" fn cuStreamCreateWithPriority(stream: *mut *mut Stream, flags: c_uint, priority: c_int) -> CUresult {
    let c = tri!(context::current());
    if stream.is_null() || flags & !CU_STREAM_NON_BLOCKING != 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let p = priority.clamp(PRIORITY_GREATEST, PRIORITY_LEAST);
    *stream = STREAMS.insert(Arc::new(Stream::new(c.key(), flags, p, false)));
    CUDA_SUCCESS
}

/// cuStreamDestroy: returns at once; work still queued completes first. The legacy stream cannot be destroyed
/// (CUDA_ERROR_INVALID_HANDLE).
#[no_mangle]
pub unsafe extern "system" fn cuStreamDestroy_v2(stream: *mut Stream) -> CUresult {
    tri!(global());
    match STREAMS.remove(stream) {
        Some(s) => {
            *s.capture.lock().unwrap() = None;
            CUDA_SUCCESS
        }
        None => CUDA_ERROR_INVALID_HANDLE,
    }
}

#[no_mangle]
pub unsafe extern "system" fn cuStreamGetFlags(stream: *mut Stream, flags: *mut c_uint) -> CUresult {
    let s = tri!(resolve(stream));
    if flags.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *flags = s.flags;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuStreamGetPriority(stream: *mut Stream, priority: *mut c_int) -> CUresult {
    let s = tri!(resolve(stream));
    if priority.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *priority = s.priority;
    CUDA_SUCCESS
}

/// cuStreamGetCtx: the stream's context (the current one for the legacy stream)
#[no_mangle]
pub unsafe extern "system" fn cuStreamGetCtx(stream: *mut Stream, ctx: *mut *mut Context) -> CUresult {
    let s = tri!(resolve(stream));
    if ctx.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *ctx = s.ctx as *mut Context;
    CUDA_SUCCESS
}

/// cuStreamGetId (CUDA 12.0): a unique id per stream
#[no_mangle]
pub unsafe extern "system" fn cuStreamGetId(stream: *mut Stream, id: *mut u64) -> CUresult {
    let s = tri!(resolve(stream));
    if id.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *id = s.id;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuStreamGetDevice(stream: *mut Stream, dev: *mut CUdevice) -> CUresult {
    tri!(resolve(stream));
    if dev.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *dev = 0;
    CUDA_SUCCESS
}

/// cuStreamSynchronize: waits for the stream's work (the legacy stream: also every blocking stream's); the
/// context's sticky error after a kernel fault. Not permitted on a capturing stream
/// (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, which invalidates the capture).
#[no_mangle]
pub unsafe extern "system" fn cuStreamSynchronize(stream: *mut Stream) -> CUresult {
    tri!(context::current_unchecked());
    let s = tri!(resolve(stream));
    tri!(graph::sync_during_capture(s.ctx, Some(&s)));
    s.wait(s.submitted());
    if s.legacy {
        for o in STREAMS.all() {
            if o.ctx == s.ctx && o.blocking() {
                o.wait(o.submitted());
            }
        }
    }
    crate::device_rt::flush();
    if let Err(e) = host_access() {
        return e;
    }
    s.context().sticky.load(Ordering::Acquire)
}

/// cuStreamQuery: CUDA_SUCCESS when all the stream's work is complete, CUDA_ERROR_NOT_READY otherwise
#[no_mangle]
pub unsafe extern "system" fn cuStreamQuery(stream: *mut Stream) -> CUresult {
    tri!(context::current_unchecked());
    let s = tri!(resolve(stream));
    tri!(graph::sync_during_capture(s.ctx, Some(&s)));
    let mut ready = s.idle();
    if s.legacy {
        ready &= STREAMS.all().iter().all(|o| o.ctx != s.ctx || !o.blocking() || o.idle());
    }
    if !ready {
        return CUDA_ERROR_NOT_READY;
    }
    s.context().sticky.load(Ordering::Acquire)
}

/// cuStreamWaitEvent: the stream's later work waits for the event's last record (a no-op for an event that was
/// never recorded). Flags 0 or CU_EVENT_WAIT_EXTERNAL (1).
#[no_mangle]
pub unsafe extern "system" fn cuStreamWaitEvent(stream: *mut Stream, event: *mut Event, flags: c_uint) -> CUresult {
    tri!(context::current());
    let s = tri!(resolve(stream));
    let Some(e) = EVENTS.get(event) else { return CUDA_ERROR_INVALID_HANDLE };
    if flags & !1 != 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    wait_event(&s, &e)
}

pub(crate) unsafe fn wait_event(s: &Arc<Stream>, e: &Arc<Event>) -> CUresult {
    if let Some(r) = graph::capture_wait_event(s, e) {
        return r;
    }
    let rec = e.st.lock().unwrap().rec.clone();
    match rec {
        Some((o, seq)) if !Arc::ptr_eq(&o, s) && o.done() < seq => submit(s, Op::Wait(o, seq)).0,
        _ => CUDA_SUCCESS,
    }
}

/// cuStreamAddCallback: flags must be 0; the callback runs after the stream's earlier work, with its status
#[no_mangle]
pub unsafe extern "system" fn cuStreamAddCallback(
    stream: *mut Stream,
    cb: Option<StreamCallback>,
    data: *mut c_void,
    flags: c_uint,
) -> CUresult {
    tri!(context::current());
    let Some(cb) = cb else { return CUDA_ERROR_INVALID_VALUE };
    if flags != 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let s = tri!(resolve(stream));
    let h = s.handle();
    submit(&s, Op::Callback(cb, h as usize, data as usize)).0
}

/// cuLaunchHostFunc: the function runs on a driver thread after the stream's earlier work; later work of the
/// stream waits for it
#[no_mangle]
pub unsafe extern "system" fn cuLaunchHostFunc(stream: *mut Stream, f: Option<HostFn>, data: *mut c_void) -> CUresult {
    tri!(context::current());
    let Some(f) = f else { return CUDA_ERROR_INVALID_VALUE };
    submit_to(stream, Op::Host(f, data as usize))
}

/// cuStreamAttachMemAsync: managed memory is visible to every stream already
#[no_mangle]
pub unsafe extern "system" fn cuStreamAttachMemAsync(stream: *mut Stream, p: CUdeviceptr, _len: usize, flags: c_uint) -> CUresult {
    tri!(context::current());
    tri!(resolve(stream));
    // CU_MEM_ATTACH_GLOBAL 1, HOST 2, SINGLE 4
    if ![1, 2, 4].contains(&flags) || !matches!(memory::lookup(p), Some((_, a)) if a.kind == memory::Kind::Managed) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    submit_to(stream, Op::Empty)
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
/// CU_EVENT_BLOCKING_SYNC 1, DISABLE_TIMING 2, INTERPROCESS 4
const CU_EVENT_DISABLE_TIMING: c_uint = 2;
const CU_EVENT_INTERPROCESS: c_uint = 4;

pub struct Event {
    ctx: usize,
    flags: c_uint,
    pub(crate) st: Mutex<EventState>,
}
unsafe impl Send for Event {}
unsafe impl Sync for Event {}

pub(crate) struct EventState {
    /// incremented by every record
    pub gen: u64,
    /// the last record: (stream, sequence number), None before the first and for a record during capture
    pub rec: Option<(Arc<Stream>, u64)>,
    /// when the last record ran: host time, simulated GPU time (ns)
    time: Option<(Instant, f64)>,
    /// the last record was captured: the capture and the nodes it depends on
    pub captured: Option<(Arc<graph::Session>, Vec<usize>)>,
}

pub(crate) static EVENTS: Handles<Event> = Handles::new();

fn sim_timing() -> bool {
    backend::use_agx() && std::env::var("ZLUDA_EVENT_TIMING").map_or(true, |v| v != "wall")
}

impl Event {
    fn stamp(&self, gen: u64) {
        let mut st = self.st.lock().unwrap();
        if st.gen == gen {
            st.time = Some((Instant::now(), if sim_timing() { crate::agx::sim_clock_ns() } else { 0.0 }));
        }
    }
    /// complete: never recorded, or its last record ran
    fn complete(&self) -> bool {
        let st = self.st.lock().unwrap();
        match &st.rec {
            None => true,
            Some((s, seq)) => s.done() >= *seq,
        }
    }
}

/// cuEventCreate: CUDA_ERROR_INVALID_VALUE for unknown flags or CU_EVENT_INTERPROCESS without
/// CU_EVENT_DISABLE_TIMING
#[no_mangle]
pub unsafe extern "system" fn cuEventCreate(event: *mut *mut Event, flags: c_uint) -> CUresult {
    let c = tri!(context::current());
    if event.is_null() || flags & !7 != 0 || (flags & CU_EVENT_INTERPROCESS != 0 && flags & CU_EVENT_DISABLE_TIMING == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if flags & CU_EVENT_DISABLE_TIMING == 0 && sim_timing() {
        crate::agx::want_timing();
    }
    *event = EVENTS.insert(Arc::new(Event {
        ctx: c.key(),
        flags,
        st: Mutex::new(EventState { gen: 0, rec: None, time: None, captured: None }),
    }));
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuEventDestroy_v2(event: *mut Event) -> CUresult {
    tri!(global());
    match EVENTS.remove(event) {
        Some(_) => CUDA_SUCCESS,
        None => CUDA_ERROR_INVALID_HANDLE,
    }
}

pub(crate) unsafe fn record(e: &Arc<Event>, s: &Arc<Stream>) -> CUresult {
    if let Some(r) = graph::capture_record(s, e) {
        return r;
    }
    let gen = {
        let mut st = e.st.lock().unwrap();
        st.gen += 1;
        st.captured = None;
        st.gen
    };
    // the record is its stream's next piece of work
    let (r, seq) = submit(s, Op::Record(e.clone(), gen));
    let mut st = e.st.lock().unwrap();
    if st.gen == gen && r == CUDA_SUCCESS {
        st.rec = Some((s.clone(), seq));
    }
    r
}

/// cuEventRecord: the event completes when the stream's work before it has completed
#[no_mangle]
pub unsafe extern "system" fn cuEventRecord(event: *mut Event, stream: *mut Stream) -> CUresult {
    tri!(context::current());
    let Some(e) = EVENTS.get(event) else { return CUDA_ERROR_INVALID_HANDLE };
    let s = tri!(resolve(stream));
    if s.ctx != e.ctx {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    record(&e, &s)
}

/// cuEventRecordWithFlags: CU_EVENT_RECORD_DEFAULT 0 or CU_EVENT_RECORD_EXTERNAL 1
#[no_mangle]
pub unsafe extern "system" fn cuEventRecordWithFlags(event: *mut Event, stream: *mut Stream, flags: c_uint) -> CUresult {
    if flags & !1 != 0 {
        tri!(global());
        return CUDA_ERROR_INVALID_VALUE;
    }
    cuEventRecord(event, stream)
}

/// cuEventQuery: CUDA_SUCCESS when the work before the last record has completed (or the event was never
/// recorded), CUDA_ERROR_NOT_READY otherwise; CUDA_ERROR_CAPTURED_EVENT for an event recorded in a capture
#[no_mangle]
pub unsafe extern "system" fn cuEventQuery(event: *mut Event) -> CUresult {
    tri!(global());
    let Some(e) = EVENTS.get(event) else { return CUDA_ERROR_INVALID_HANDLE };
    if e.st.lock().unwrap().captured.is_some() {
        return CUDA_ERROR_CAPTURED_EVENT;
    }
    if e.complete() {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_NOT_READY
    }
}

/// cuEventSynchronize: waits for the event's last record (returns at once for a never recorded event)
#[no_mangle]
pub unsafe extern "system" fn cuEventSynchronize(event: *mut Event) -> CUresult {
    tri!(global());
    let Some(e) = EVENTS.get(event) else { return CUDA_ERROR_INVALID_HANDLE };
    let rec = {
        let st = e.st.lock().unwrap();
        if st.captured.is_some() {
            return CUDA_ERROR_CAPTURED_EVENT;
        }
        st.rec.clone()
    };
    if let Some((s, seq)) = rec {
        s.wait(seq);
        crate::device_rt::flush();
        return s.context().sticky.load(Ordering::Acquire);
    }
    crate::device_rt::flush();
    CUDA_SUCCESS
}

/// cuEventElapsedTime: milliseconds between two completed records. CUDA_ERROR_INVALID_HANDLE if either event
/// was never recorded or was created with CU_EVENT_DISABLE_TIMING, CUDA_ERROR_NOT_READY if either has not
/// completed.
#[no_mangle]
pub unsafe extern "system" fn cuEventElapsedTime(ms: *mut f32, start: *mut Event, end: *mut Event) -> CUresult {
    tri!(global());
    if ms.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let (Some(s), Some(e)) = (EVENTS.get(start), EVENTS.get(end)) else { return CUDA_ERROR_INVALID_HANDLE };
    if (s.flags | e.flags) & CU_EVENT_DISABLE_TIMING != 0 {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    for x in [&s, &e] {
        let st = x.st.lock().unwrap();
        if st.rec.is_none() && st.captured.is_none() {
            return CUDA_ERROR_INVALID_HANDLE;
        }
    }
    if !s.complete() || !e.complete() {
        return CUDA_ERROR_NOT_READY;
    }
    let (Some(t0), Some(t1)) = (s.st.lock().unwrap().time, e.st.lock().unwrap().time) else {
        return CUDA_ERROR_NOT_READY;
    };
    *ms = if sim_timing() {
        ((t1.1 - t0.1) / 1e6) as f32
    } else if t1.0 >= t0.0 {
        t1.0.duration_since(t0.0).as_secs_f32() * 1000.0
    } else {
        -(t0.0.duration_since(t1.0).as_secs_f32() * 1000.0)
    };
    CUDA_SUCCESS
}

/// cuEventElapsedTime_v2 (CUDA 12.8): the same
#[no_mangle]
pub unsafe extern "system" fn cuEventElapsedTime_v2(ms: *mut f32, start: *mut Event, end: *mut Event) -> CUresult {
    cuEventElapsedTime(ms, start, end)
}

pub(crate) fn event(h: *mut Event) -> Option<Arc<Event>> {
    EVENTS.get(h)
}
