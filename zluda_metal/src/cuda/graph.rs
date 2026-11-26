//! CUDA graphs (CUDA Driver API reference, "Graph Management", "Stream Management" capture functions): explicit
//! graph construction with kernel, memcpy, memset, host, empty and event nodes, stream capture (with fork / join
//! through events across streams), instantiation and launch. An executable graph runs its nodes in a topological
//! order on the launch stream.
use super::memory::{CUDA_MEMCPY3D, CUDA_MEMSET_NODE_PARAMS};
use super::stream::{self, Event, HostFn, Op};
use super::*;
use std::cell::Cell;

/// CUgraphNodeType
const NODE_KERNEL: c_int = 0;
const NODE_MEMCPY: c_int = 1;
const NODE_MEMSET: c_int = 2;
const NODE_HOST: c_int = 3;
const NODE_EMPTY: c_int = 5;
const NODE_WAIT_EVENT: c_int = 6;
const NODE_EVENT_RECORD: c_int = 7;

/// CUstreamCaptureMode
const MODE_GLOBAL: c_int = 0;
const MODE_THREAD_LOCAL: c_int = 1;
const MODE_RELAXED: c_int = 2;

pub struct GraphNode {
    graph: *const Graph,
    idx: usize,
}

struct NodeData {
    op: Op,
    kind: c_int,
    deps: Vec<usize>,
    handle: Box<GraphNode>,
}

pub struct Graph {
    nodes: Mutex<Vec<NodeData>>,
}
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

pub struct GraphExec {
    ops: Vec<Op>,
}
unsafe impl Send for GraphExec {}
unsafe impl Sync for GraphExec {}

static GRAPHS: Handles<Graph> = Handles::new();
/// live node handles (a node is checked before it is dereferenced)
static NODES: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

fn node_valid(n: *const GraphNode) -> bool {
    NODES.lock().unwrap().as_ref().is_some_and(|s| s.contains(&(n as usize)))
}

impl Drop for Graph {
    fn drop(&mut self) {
        if let Some(s) = NODES.lock().unwrap().as_mut() {
            for d in self.nodes.get_mut().unwrap().iter() {
                s.remove(&(&*d.handle as *const GraphNode as usize));
            }
        }
    }
}
static EXECS: Handles<GraphExec> = Handles::new();

impl Graph {
    fn new() -> Arc<Graph> {
        Arc::new(Graph { nodes: Mutex::new(vec![]) })
    }
    /// add a node depending on nodes `deps` (indices), its handle
    fn add(&self, op: Op, kind: c_int, deps: Vec<usize>) -> *mut GraphNode {
        let mut n = self.nodes.lock().unwrap();
        let idx = n.len();
        let mut handle = Box::new(GraphNode { graph: self, idx });
        let h = &mut *handle as *mut GraphNode;
        NODES.lock().unwrap().get_or_insert_with(HashSet::new).insert(h as usize);
        n.push(NodeData { op, kind, deps, handle });
        h
    }
    fn node_handle(&self, i: usize) -> *mut GraphNode {
        &*self.nodes.lock().unwrap()[i].handle as *const GraphNode as *mut GraphNode
    }
    /// nodes in a topological order (Kahn's algorithm, ties in creation order)
    fn order(&self) -> Vec<usize> {
        let n = self.nodes.lock().unwrap();
        let mut indeg: Vec<usize> = n.iter().map(|d| d.deps.len()).collect();
        let mut out = vec![];
        let mut ready: Vec<usize> = (0..n.len()).filter(|&i| indeg[i] == 0).rev().collect();
        while let Some(i) = ready.pop() {
            out.push(i);
            for (j, d) in n.iter().enumerate() {
                for &x in &d.deps {
                    if x == i {
                        indeg[j] -= 1;
                        if indeg[j] == 0 {
                            ready.insert(0, j);
                            ready.sort_unstable_by(|a, b| b.cmp(a));
                        }
                    }
                }
            }
        }
        out
    }
}

fn kind_of(op: &Op) -> c_int {
    match op {
        Op::Kernel(_) => NODE_KERNEL,
        Op::Copy(_) => NODE_MEMCPY,
        Op::Set(_) => NODE_MEMSET,
        Op::Host(..) | Op::Callback(..) => NODE_HOST,
        Op::RecordNode(_) | Op::Record(..) => NODE_EVENT_RECORD,
        Op::WaitNode(_) | Op::Wait(..) => NODE_WAIT_EVENT,
        _ => NODE_EMPTY,
    }
}

// ---------------------------------------------------------------------------
// Stream capture
// ---------------------------------------------------------------------------
/// One capture sequence: the graph being built, the streams taking part
pub(crate) struct Session {
    graph: Arc<Graph>,
    id: u64,
    mode: c_int,
    thread: std::thread::ThreadId,
    ctx: usize,
    invalid: AtomicBool,
    origin: usize,
    streams: Mutex<Vec<Arc<Stream>>>,
}

/// A capturing stream's state: its session and the nodes its next work depends on
pub(crate) struct StreamCapture {
    session: Arc<Session>,
    frontier: Vec<usize>,
    /// cuStreamGetCaptureInfo's dependency array, kept until the next call
    deps_out: Vec<*mut GraphNode>,
}
unsafe impl Send for StreamCapture {}

static SESSIONS: Mutex<Vec<Arc<Session>>> = Mutex::new(Vec::new());

thread_local! {
    static THREAD_MODE: Cell<c_int> = const { Cell::new(MODE_GLOBAL) };
}

fn invalidate(s: &Session) {
    s.invalid.store(true, Ordering::Release);
}

/// work submitted to a capturing stream becomes a node of its graph
pub(crate) fn capture_op(s: &Arc<Stream>, op: &Op) -> Option<CUresult> {
    let mut c = s.capture.lock().unwrap();
    let c = c.as_mut()?;
    if c.session.invalid.load(Ordering::Acquire) {
        return Some(CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);
    }
    let h = c.session.graph.add(op.clone(), kind_of(op), std::mem::take(&mut c.frontier));
    c.frontier = vec![unsafe { (*h).idx }];
    Some(CUDA_SUCCESS)
}

/// cuEventRecord on a capturing stream: the event marks the stream's current nodes (for cuStreamWaitEvent forks
/// and joins)
pub(crate) fn capture_record(s: &Arc<Stream>, e: &Arc<Event>) -> Option<CUresult> {
    let c = s.capture.lock().unwrap();
    let c = c.as_ref()?;
    let mut st = e.st.lock().unwrap();
    st.gen += 1;
    st.rec = None;
    st.captured = Some((c.session.clone(), c.frontier.clone()));
    Some(CUDA_SUCCESS)
}

/// cuStreamWaitEvent with a captured event joins the stream into that capture (a fork) or merges the event's
/// nodes into its dependencies (a join). A capturing stream may not wait for work outside its capture
/// (CUDA_ERROR_STREAM_CAPTURE_ISOLATION).
pub(crate) fn capture_wait_event(s: &Arc<Stream>, e: &Arc<Event>) -> Option<CUresult> {
    let captured = e.st.lock().unwrap().captured.clone();
    let mut c = s.capture.lock().unwrap();
    match (captured, c.as_mut()) {
        (None, None) => None,
        (None, Some(cs)) => {
            if e.st.lock().unwrap().rec.is_none() {
                return Some(CUDA_SUCCESS);
            }
            invalidate(&cs.session);
            Some(CUDA_ERROR_STREAM_CAPTURE_ISOLATION)
        }
        (Some((sess, nodes)), Some(cs)) => {
            if !Arc::ptr_eq(&sess, &cs.session) {
                invalidate(&cs.session);
                return Some(CUDA_ERROR_STREAM_CAPTURE_ISOLATION);
            }
            for n in nodes {
                if !cs.frontier.contains(&n) {
                    cs.frontier.push(n);
                }
            }
            Some(CUDA_SUCCESS)
        }
        (Some((sess, nodes)), None) => {
            if s.legacy || sess.invalid.load(Ordering::Acquire) {
                invalidate(&sess);
                return Some(CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);
            }
            sess.streams.lock().unwrap().push(s.clone());
            *c = Some(StreamCapture { session: sess, frontier: nodes, deps_out: vec![] });
            Some(CUDA_SUCCESS)
        }
    }
}

fn active_sessions() -> Vec<Arc<Session>> {
    SESSIONS.lock().unwrap().clone()
}

/// Work on the legacy stream while a blocking stream of its context captures (in a mode other than relaxed):
/// that would make the legacy stream depend on captured work (CUDA_ERROR_STREAM_CAPTURE_IMPLICIT, the capture is
/// invalidated)
pub(crate) fn legacy_use(ctx: usize) -> Result<(), CUresult> {
    let mut r = Ok(());
    for sess in active_sessions() {
        if sess.ctx == ctx && sess.mode != MODE_RELAXED {
            let blocking = sess.streams.lock().unwrap().iter().any(|s| s.blocking());
            if blocking {
                invalidate(&sess);
                r = Err(CUDA_ERROR_STREAM_CAPTURE_IMPLICIT);
            }
        }
    }
    r
}

/// Synchronizing or querying a capturing stream (or the context, while any of its streams captures) is not
/// permitted: CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, and the capture is invalidated
pub(crate) fn sync_during_capture(ctx: usize, s: Option<&Arc<Stream>>) -> Result<(), CUresult> {
    if let Some(s) = s {
        if let Some(c) = s.capture.lock().unwrap().as_ref() {
            invalidate(&c.session);
            return Err(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
        }
        if !s.legacy {
            return Ok(());
        }
    }
    let mut r = Ok(());
    for sess in active_sessions() {
        if sess.ctx == ctx && (s.is_none() || sess.mode != MODE_RELAXED) {
            invalidate(&sess);
            r = Err(if s.is_none() { CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED } else { CUDA_ERROR_STREAM_CAPTURE_IMPLICIT });
        }
    }
    r
}

/// A "potentially unsafe" call (cuMemAlloc, cuMemFree) during a global capture, or a thread-local capture of
/// this thread, unless this thread's mode is relaxed (cuThreadExchangeStreamCaptureMode):
/// CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED, and the capture is invalidated
pub(crate) fn unsafe_during_capture() -> Result<(), CUresult> {
    if THREAD_MODE.with(|m| m.get()) == MODE_RELAXED {
        return Ok(());
    }
    let me = std::thread::current().id();
    let mut r = Ok(());
    for sess in active_sessions() {
        if sess.mode == MODE_GLOBAL || (sess.mode == MODE_THREAD_LOCAL && sess.thread == me) {
            invalidate(&sess);
            r = Err(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
        }
    }
    r
}

unsafe fn begin_capture(stream: *mut Stream, mode: c_int) -> CUresult {
    let c = tri!(context::current());
    if !(MODE_GLOBAL..=MODE_RELAXED).contains(&mode) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let s = tri!(stream::resolve(stream));
    // "capture may not be initiated in the legacy stream"
    if s.legacy {
        return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    }
    let mut cap = s.capture.lock().unwrap();
    if cap.is_some() {
        return CUDA_ERROR_ILLEGAL_STATE;
    }
    static IDS: AtomicU64 = AtomicU64::new(1);
    let sess = Arc::new(Session {
        graph: Graph::new(),
        id: IDS.fetch_add(1, Ordering::Relaxed),
        mode,
        thread: std::thread::current().id(),
        ctx: c.key(),
        invalid: AtomicBool::new(false),
        origin: Arc::as_ptr(&s) as usize,
        streams: Mutex::new(vec![s.clone()]),
    });
    SESSIONS.lock().unwrap().push(sess.clone());
    *cap = Some(StreamCapture { session: sess, frontier: vec![], deps_out: vec![] });
    CUDA_SUCCESS
}

/// cuStreamBeginCapture_v2: begins capturing a (non-legacy) stream in a CUstreamCaptureMode; a stream already
/// capturing: CUDA_ERROR_ILLEGAL_STATE
#[no_mangle]
pub unsafe extern "system" fn cuStreamBeginCapture_v2(stream: *mut Stream, mode: c_int) -> CUresult {
    begin_capture(stream, mode)
}

/// cuStreamBeginCapture (CUDA 10.0): global mode
#[no_mangle]
pub unsafe extern "system" fn cuStreamBeginCapture(stream: *mut Stream) -> CUresult {
    begin_capture(stream, MODE_GLOBAL)
}

/// cuStreamEndCapture: ends the capture begun on this stream and returns its graph. CUDA_ERROR_ILLEGAL_STATE if
/// the stream is not capturing, CUDA_ERROR_STREAM_CAPTURE_UNMATCHED if the capture began on another stream,
/// CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD for a non-relaxed capture of another thread,
/// CUDA_ERROR_STREAM_CAPTURE_UNJOINED if a forked stream did not join back, CUDA_ERROR_STREAM_CAPTURE_INVALIDATED
/// (and a NULL graph) if the capture was invalidated.
#[no_mangle]
pub unsafe extern "system" fn cuStreamEndCapture(stream: *mut Stream, graph: *mut *mut Graph) -> CUresult {
    tri!(context::current_unchecked());
    if graph.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let s = tri!(stream::resolve(stream));
    let (sess, frontier) = match s.capture.lock().unwrap().as_ref() {
        None => return CUDA_ERROR_ILLEGAL_STATE,
        Some(c) => (c.session.clone(), c.frontier.clone()),
    };
    if sess.origin != Arc::as_ptr(&s) as usize {
        invalidate(&sess);
        return CUDA_ERROR_STREAM_CAPTURE_UNMATCHED;
    }
    if sess.mode != MODE_RELAXED && sess.thread != std::thread::current().id() {
        return CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD;
    }
    // every other stream's last nodes must be ancestors of the origin's
    let mut ancestors: HashSet<usize> = HashSet::new();
    {
        let nodes = sess.graph.nodes.lock().unwrap();
        let mut todo = frontier.clone();
        while let Some(i) = todo.pop() {
            if ancestors.insert(i) {
                todo.extend(nodes[i].deps.iter().copied());
            }
        }
    }
    let mut unjoined = false;
    for o in sess.streams.lock().unwrap().iter() {
        if let Some(c) = o.capture.lock().unwrap().take() {
            if !Arc::ptr_eq(o, &s) && c.frontier.iter().any(|n| !ancestors.contains(n)) {
                unjoined = true;
            }
        }
    }
    SESSIONS.lock().unwrap().retain(|x| !Arc::ptr_eq(x, &sess));
    *graph = ptr::null_mut();
    if unjoined {
        return CUDA_ERROR_STREAM_CAPTURE_UNJOINED;
    }
    if sess.invalid.load(Ordering::Acquire) {
        return CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    }
    *graph = GRAPHS.insert(sess.graph.clone());
    CUDA_SUCCESS
}

/// CUstreamCaptureStatus: NONE 0, ACTIVE 1, INVALIDATED 2
fn capture_status(s: &Arc<Stream>) -> Result<(c_int, Option<Arc<Session>>), CUresult> {
    if s.legacy {
        legacy_use_query(s.ctx)?;
        return Ok((0, None));
    }
    Ok(match s.capture.lock().unwrap().as_ref() {
        None => (0, None),
        Some(c) if c.session.invalid.load(Ordering::Acquire) => (2, Some(c.session.clone())),
        Some(c) => (1, Some(c.session.clone())),
    })
}

/// Querying the legacy stream while a blocking stream captures: CUDA_ERROR_STREAM_CAPTURE_IMPLICIT
fn legacy_use_query(ctx: usize) -> Result<(), CUresult> {
    for sess in active_sessions() {
        if sess.ctx == ctx && sess.mode != MODE_RELAXED && sess.streams.lock().unwrap().iter().any(|s| s.blocking()) {
            return Err(CUDA_ERROR_STREAM_CAPTURE_IMPLICIT);
        }
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "system" fn cuStreamIsCapturing(stream: *mut Stream, status: *mut c_int) -> CUresult {
    tri!(context::current_unchecked());
    if status.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let s = tri!(stream::resolve(stream));
    *status = tri!(capture_status(&s)).0;
    CUDA_SUCCESS
}

unsafe fn capture_info(
    stream: *mut Stream,
    status: *mut c_int,
    id: *mut u64,
    graph: *mut *mut Graph,
    deps: *mut *const *mut GraphNode,
    edges: *mut *const c_void,
    ndeps: *mut usize,
) -> CUresult {
    tri!(context::current_unchecked());
    if status.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let s = tri!(stream::resolve(stream));
    let (st, sess) = tri!(capture_status(&s));
    *status = st;
    if let Some(sess) = sess.filter(|_| st == 1) {
        if !id.is_null() {
            *id = sess.id;
        }
        if !graph.is_null() {
            *graph = Arc::as_ptr(&sess.graph) as *mut Graph;
        }
        let mut cap = s.capture.lock().unwrap();
        let c = cap.as_mut().unwrap();
        c.deps_out = c.frontier.iter().map(|&i| sess.graph.node_handle(i)).collect();
        if !deps.is_null() {
            *deps = c.deps_out.as_ptr();
        }
        if !ndeps.is_null() {
            *ndeps = c.deps_out.len();
        }
    } else if !ndeps.is_null() {
        *ndeps = 0;
    }
    if !edges.is_null() {
        *edges = ptr::null();
    }
    CUDA_SUCCESS
}

/// cuStreamGetCaptureInfo (CUDA 10.1): status and id
#[no_mangle]
pub unsafe extern "system" fn cuStreamGetCaptureInfo(stream: *mut Stream, status: *mut c_int, id: *mut u64) -> CUresult {
    capture_info(stream, status, id, ptr::null_mut(), ptr::null_mut(), ptr::null_mut(), ptr::null_mut())
}

/// cuStreamGetCaptureInfo_v2: status, id, graph, the stream's current dependency nodes
#[no_mangle]
pub unsafe extern "system" fn cuStreamGetCaptureInfo_v2(
    stream: *mut Stream,
    status: *mut c_int,
    id: *mut u64,
    graph: *mut *mut Graph,
    deps: *mut *const *mut GraphNode,
    ndeps: *mut usize,
) -> CUresult {
    capture_info(stream, status, id, graph, deps, ptr::null_mut(), ndeps)
}

/// cuStreamGetCaptureInfo_v3 (CUDA 12.3): with edge data (none: NULL)
#[no_mangle]
pub unsafe extern "system" fn cuStreamGetCaptureInfo_v3(
    stream: *mut Stream,
    status: *mut c_int,
    id: *mut u64,
    graph: *mut *mut Graph,
    deps: *mut *const *mut GraphNode,
    edges: *mut *const c_void,
    ndeps: *mut usize,
) -> CUresult {
    capture_info(stream, status, id, graph, deps, edges, ndeps)
}

/// cuThreadExchangeStreamCaptureMode: swaps this thread's capture interaction mode
#[no_mangle]
pub unsafe extern "system" fn cuThreadExchangeStreamCaptureMode(mode: *mut c_int) -> CUresult {
    tri!(global());
    if mode.is_null() || !(MODE_GLOBAL..=MODE_RELAXED).contains(&*mode) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *mode = THREAD_MODE.with(|m| m.replace(*mode));
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------
fn graph(g: *mut Graph) -> Result<Arc<Graph>, CUresult> {
    GRAPHS.get(g).ok_or(CUDA_ERROR_INVALID_VALUE)
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphCreate(g: *mut *mut Graph, flags: c_uint) -> CUresult {
    tri!(global());
    if g.is_null() || flags != 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *g = GRAPHS.insert(Graph::new());
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphDestroy(g: *mut Graph) -> CUresult {
    tri!(global());
    match GRAPHS.remove(g) {
        Some(_) => CUDA_SUCCESS,
        None => CUDA_ERROR_INVALID_VALUE,
    }
}

/// The dependency array of a node to add: nodes of the same graph (CUDA_ERROR_INVALID_VALUE otherwise)
unsafe fn deps_of(g: &Arc<Graph>, deps: *const *mut GraphNode, n: usize) -> Result<Vec<usize>, CUresult> {
    if n > 0 && deps.is_null() {
        return Err(CUDA_ERROR_INVALID_VALUE);
    }
    let count = g.nodes.lock().unwrap().len();
    let mut v = vec![];
    for i in 0..n {
        let d = *deps.add(i);
        if !node_valid(d) || (*d).graph != Arc::as_ptr(g) || (*d).idx >= count || v.contains(&(*d).idx) {
            return Err(CUDA_ERROR_INVALID_VALUE);
        }
        v.push((*d).idx);
    }
    Ok(v)
}

unsafe fn add_node(
    node: *mut *mut GraphNode,
    g: *mut Graph,
    deps: *const *mut GraphNode,
    n: usize,
    op: Result<Op, CUresult>,
) -> CUresult {
    tri!(global());
    if node.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let g = tri!(graph(g));
    let d = tri!(deps_of(&g, deps, n));
    let op = tri!(op);
    *node = g.add(op.clone(), kind_of(&op), d);
    CUDA_SUCCESS
}

/// CUDA_KERNEL_NODE_PARAMS (v1) / CUDA_KERNEL_NODE_PARAMS_v2 (adds kern, ctx)
#[repr(C)]
pub struct CUDA_KERNEL_NODE_PARAMS {
    func: *mut Function,
    grid: [c_uint; 3],
    block: [c_uint; 3],
    shared_mem_bytes: c_uint,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
    kern: *mut Function,
    ctx: *mut Context,
}

unsafe fn kernel_op(p: *const CUDA_KERNEL_NODE_PARAMS, v2: bool) -> Result<Op, CUresult> {
    if p.is_null() {
        return Err(CUDA_ERROR_INVALID_VALUE);
    }
    let p = &*p;
    let f = if p.func.is_null() && v2 { p.kern } else { p.func };
    prepare_launch(f, p.grid, p.block, p.shared_mem_bytes, p.kernel_params, p.extra).map(Op::Kernel)
}

/// cuGraphAddKernelNode: the kernel's arguments are copied when the node is added
#[no_mangle]
pub unsafe extern "system" fn cuGraphAddKernelNode(
    node: *mut *mut GraphNode,
    g: *mut Graph,
    deps: *const *mut GraphNode,
    n: usize,
    params: *const CUDA_KERNEL_NODE_PARAMS,
) -> CUresult {
    add_node(node, g, deps, n, kernel_op(params, false))
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphAddKernelNode_v2(
    node: *mut *mut GraphNode,
    g: *mut Graph,
    deps: *const *mut GraphNode,
    n: usize,
    params: *const CUDA_KERNEL_NODE_PARAMS,
) -> CUresult {
    add_node(node, g, deps, n, kernel_op(params, true))
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphAddMemcpyNode(
    node: *mut *mut GraphNode,
    g: *mut Graph,
    deps: *const *mut GraphNode,
    n: usize,
    params: *const CUDA_MEMCPY3D,
    _ctx: *mut Context,
) -> CUresult {
    let op = if params.is_null() {
        Err(CUDA_ERROR_INVALID_VALUE)
    } else {
        memory::copy3d_from(&*params).and_then(|c| memory::validate_copy(&c).map(|_| Op::Copy(c)))
    };
    add_node(node, g, deps, n, op)
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphAddMemsetNode(
    node: *mut *mut GraphNode,
    g: *mut Graph,
    deps: *const *mut GraphNode,
    n: usize,
    params: *const CUDA_MEMSET_NODE_PARAMS,
    _ctx: *mut Context,
) -> CUresult {
    let op = if params.is_null() { Err(CUDA_ERROR_INVALID_VALUE) } else { memory::set_from(&*params).map(Op::Set) };
    add_node(node, g, deps, n, op)
}

/// CUDA_HOST_NODE_PARAMS
#[repr(C)]
pub struct CUDA_HOST_NODE_PARAMS {
    f: Option<HostFn>,
    data: *mut c_void,
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphAddHostNode(
    node: *mut *mut GraphNode,
    g: *mut Graph,
    deps: *const *mut GraphNode,
    n: usize,
    params: *const CUDA_HOST_NODE_PARAMS,
) -> CUresult {
    let op = match params.as_ref() {
        Some(CUDA_HOST_NODE_PARAMS { f: Some(f), data }) => Ok(Op::Host(*f, *data as usize)),
        _ => Err(CUDA_ERROR_INVALID_VALUE),
    };
    add_node(node, g, deps, n, op)
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphAddEmptyNode(
    node: *mut *mut GraphNode,
    g: *mut Graph,
    deps: *const *mut GraphNode,
    n: usize,
) -> CUresult {
    add_node(node, g, deps, n, Ok(Op::Empty))
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphAddEventRecordNode(
    node: *mut *mut GraphNode,
    g: *mut Graph,
    deps: *const *mut GraphNode,
    n: usize,
    event: *mut Event,
) -> CUresult {
    let op = stream::event(event).map(Op::RecordNode).ok_or(CUDA_ERROR_INVALID_HANDLE);
    add_node(node, g, deps, n, op)
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphAddEventWaitNode(
    node: *mut *mut GraphNode,
    g: *mut Graph,
    deps: *const *mut GraphNode,
    n: usize,
    event: *mut Event,
) -> CUresult {
    let op = stream::event(event).map(Op::WaitNode).ok_or(CUDA_ERROR_INVALID_HANDLE);
    add_node(node, g, deps, n, op)
}

/// cuGraphAddDependencies: edges from[i] -> to[i]
#[no_mangle]
pub unsafe extern "system" fn cuGraphAddDependencies(
    g: *mut Graph,
    from: *const *mut GraphNode,
    to: *const *mut GraphNode,
    n: usize,
) -> CUresult {
    tri!(global());
    let g = tri!(graph(g));
    let a = tri!(deps_of(&g, from, n));
    let b = tri!(deps_of(&g, to, n));
    let mut nodes = g.nodes.lock().unwrap();
    for (x, y) in a.into_iter().zip(b) {
        if x == y {
            return CUDA_ERROR_INVALID_VALUE;
        }
        if !nodes[y].deps.contains(&x) {
            nodes[y].deps.push(x);
        }
    }
    CUDA_SUCCESS
}

/// Return a node list: "nodes may be NULL, in which case this function will return the number of nodes in
/// numNodes. Otherwise, numNodes entries will be filled in. If numNodes is higher than the actual number of nodes,
/// the remaining entries in nodes will be set to NULL, and the number of nodes actually obtained will be returned
/// in numNodes."
unsafe fn node_list(list: Vec<*mut GraphNode>, out: *mut *mut GraphNode, n: *mut usize) -> CUresult {
    if n.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if out.is_null() {
        *n = list.len();
        return CUDA_SUCCESS;
    }
    for i in 0..*n {
        *out.add(i) = list.get(i).copied().unwrap_or(ptr::null_mut());
    }
    *n = list.len().min(*n);
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphGetNodes(g: *mut Graph, nodes: *mut *mut GraphNode, n: *mut usize) -> CUresult {
    tri!(global());
    let g = tri!(graph(g));
    let count = g.nodes.lock().unwrap().len();
    node_list((0..count).map(|i| g.node_handle(i)).collect(), nodes, n)
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphGetRootNodes(g: *mut Graph, nodes: *mut *mut GraphNode, n: *mut usize) -> CUresult {
    tri!(global());
    let g = tri!(graph(g));
    let roots: Vec<usize> = g.nodes.lock().unwrap().iter().enumerate().filter(|(_, d)| d.deps.is_empty()).map(|(i, _)| i).collect();
    node_list(roots.into_iter().map(|i| g.node_handle(i)).collect(), nodes, n)
}

/// cuGraphGetEdges: from[i] -> to[i] (NULL arrays: the count)
#[no_mangle]
pub unsafe extern "system" fn cuGraphGetEdges(
    g: *mut Graph,
    from: *mut *mut GraphNode,
    to: *mut *mut GraphNode,
    n: *mut usize,
) -> CUresult {
    tri!(global());
    let g = tri!(graph(g));
    if n.is_null() || from.is_null() != to.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let edges: Vec<(usize, usize)> = g.nodes.lock().unwrap().iter().enumerate()
        .flat_map(|(j, d)| d.deps.iter().map(move |&i| (i, j))).collect();
    if from.is_null() {
        *n = edges.len();
        return CUDA_SUCCESS;
    }
    for k in 0..*n {
        let (a, b) = edges.get(k).map_or((ptr::null_mut(), ptr::null_mut()), |&(i, j)| (g.node_handle(i), g.node_handle(j)));
        *from.add(k) = a;
        *to.add(k) = b;
    }
    *n = edges.len().min(*n);
    CUDA_SUCCESS
}

unsafe fn node(n: *mut GraphNode) -> Result<(Arc<Graph>, usize), CUresult> {
    if !node_valid(n) {
        return Err(CUDA_ERROR_INVALID_VALUE);
    }
    let g = GRAPHS.get((*n).graph).ok_or(CUDA_ERROR_INVALID_VALUE)?;
    Ok((g, (*n).idx))
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphNodeGetType(n: *mut GraphNode, ty: *mut c_int) -> CUresult {
    tri!(global());
    let (g, i) = tri!(node(n));
    if ty.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *ty = g.nodes.lock().unwrap()[i].kind;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphNodeGetDependencies(n: *mut GraphNode, deps: *mut *mut GraphNode, count: *mut usize) -> CUresult {
    tri!(global());
    let (g, i) = tri!(node(n));
    let d = g.nodes.lock().unwrap()[i].deps.clone();
    node_list(d.into_iter().map(|x| g.node_handle(x)).collect(), deps, count)
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphNodeGetDependentNodes(n: *mut GraphNode, deps: *mut *mut GraphNode, count: *mut usize) -> CUresult {
    tri!(global());
    let (g, i) = tri!(node(n));
    let d: Vec<usize> = g.nodes.lock().unwrap().iter().enumerate().filter(|(_, x)| x.deps.contains(&i)).map(|(j, _)| j).collect();
    node_list(d.into_iter().map(|x| g.node_handle(x)).collect(), deps, count)
}

/// cuGraphClone: a copy of the graph (its nodes' parameters copied)
#[no_mangle]
pub unsafe extern "system" fn cuGraphClone(out: *mut *mut Graph, g: *mut Graph) -> CUresult {
    tri!(global());
    if out.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let g = tri!(graph(g));
    let c = Graph::new();
    for d in g.nodes.lock().unwrap().iter() {
        c.add(d.op.clone(), d.kind, d.deps.clone());
    }
    *out = GRAPHS.insert(c);
    CUDA_SUCCESS
}

// ---------------------------------------------------------------------------
// Executable graphs
// ---------------------------------------------------------------------------
/// CUgraphInstantiate_flags: AUTO_FREE_ON_LAUNCH 1, UPLOAD 2, DEVICE_LAUNCH 4, USE_NODE_PRIORITY 8
unsafe fn instantiate(exec: *mut *mut GraphExec, g: *mut Graph, flags: u64) -> CUresult {
    tri!(context::current());
    if exec.is_null() || flags & !0xF != 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let g = tri!(graph(g));
    let order = g.order();
    let nodes = g.nodes.lock().unwrap();
    let ops = order.into_iter().map(|i| nodes[i].op.clone()).collect();
    *exec = EXECS.insert(Arc::new(GraphExec { ops }));
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphInstantiateWithFlags(exec: *mut *mut GraphExec, g: *mut Graph, flags: u64) -> CUresult {
    instantiate(exec, g, flags)
}

/// cuGraphInstantiate / cuGraphInstantiate_v2 (the CUDA 10 / 11 form: error node and log buffer)
#[no_mangle]
pub unsafe extern "system" fn cuGraphInstantiate_v2(
    exec: *mut *mut GraphExec,
    g: *mut Graph,
    err_node: *mut *mut GraphNode,
    log: *mut c_char,
    log_size: usize,
) -> CUresult {
    if !err_node.is_null() {
        *err_node = ptr::null_mut();
    }
    if !log.is_null() && log_size > 0 {
        *log = 0;
    }
    instantiate(exec, g, 0)
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphInstantiate(
    exec: *mut *mut GraphExec,
    g: *mut Graph,
    err_node: *mut *mut GraphNode,
    log: *mut c_char,
    log_size: usize,
) -> CUresult {
    cuGraphInstantiate_v2(exec, g, err_node, log, log_size)
}

/// CUDA_GRAPH_INSTANTIATE_PARAMS
#[repr(C)]
pub struct CUDA_GRAPH_INSTANTIATE_PARAMS {
    flags: u64,
    upload_stream: *mut Stream,
    err_node: *mut GraphNode,
    result: c_int,
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphInstantiateWithParams(
    exec: *mut *mut GraphExec,
    g: *mut Graph,
    params: *mut CUDA_GRAPH_INSTANTIATE_PARAMS,
) -> CUresult {
    if params.is_null() {
        tri!(global());
        return CUDA_ERROR_INVALID_VALUE;
    }
    let p = &mut *params;
    let r = instantiate(exec, g, p.flags);
    p.err_node = ptr::null_mut();
    // CUgraphInstantiateResult: SUCCESS 0, ERROR 1
    p.result = (r != CUDA_SUCCESS) as c_int;
    r
}

/// cuGraphLaunch: the graph's nodes, in a topological order, as the stream's next work
#[no_mangle]
pub unsafe extern "system" fn cuGraphLaunch(exec: *mut GraphExec, s: *mut Stream) -> CUresult {
    let c = tri!(context::current());
    let Some(e) = EXECS.get(exec) else { return CUDA_ERROR_INVALID_VALUE };
    let st = tri!(stream::resolve(s));
    if st.ctx != c.key() {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    for op in &e.ops {
        let (r, _) = stream::submit(&st, op.clone());
        if r != CUDA_SUCCESS {
            return r;
        }
    }
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphUpload(exec: *mut GraphExec, s: *mut Stream) -> CUresult {
    tri!(context::current());
    if EXECS.get(exec).is_none() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    tri!(stream::resolve(s));
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuGraphExecDestroy(exec: *mut GraphExec) -> CUresult {
    tri!(global());
    match EXECS.remove(exec) {
        Some(_) => CUDA_SUCCESS,
        None => CUDA_ERROR_INVALID_VALUE,
    }
}
