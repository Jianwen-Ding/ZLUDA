//! Device execution of independently built graphs, using the MLIR -> AIR library kernels.
use crate::graph::*;
use agx_libs::engine;
use cuda_types::cuda::*;
use std::{collections::HashMap, ffi::{c_void, CStr}, sync::{Arc, Mutex, OnceLock}};

fn gpu<T>(r: std::result::Result<T, engine::Error>) -> Result<T> { r.map_err(|e| format!("AGX inference driver: {e:?}")) }
fn arg<T>(v: &mut T) -> *mut c_void { (v as *mut T).cast() }
struct Driver {
    current: unsafe extern "system" fn(*mut CUcontext) -> CUresult,
    push: unsafe extern "system" fn(CUcontext) -> CUresult,
    pop: unsafe extern "system" fn(*mut CUcontext) -> CUresult,
    _lib: libloading::Library,
}
fn driver() -> Result<&'static Driver> {
    static D: OnceLock<std::result::Result<Driver, String>> = OnceLock::new();
    D.get_or_init(|| unsafe {
        let name = std::env::var_os("ZLUDA_AGX_LIBCUDA").unwrap_or_else(|| "libcuda.so.1".into());
        let lib = libloading::Library::new(name).map_err(|e| e.to_string())?;
        Ok(Driver { current: *lib.get(b"cuCtxGetCurrent\0").map_err(|e| e.to_string())?,
            push: *lib.get(b"cuCtxPushCurrent_v2\0").map_err(|e| e.to_string())?,
            pop: *lib.get(b"cuCtxPopCurrent_v2\0").map_err(|e| e.to_string())?, _lib: lib })
    }).as_ref().map_err(Clone::clone)
}
fn current() -> Result<usize> {
    let mut c = CUcontext(std::ptr::null_mut());
    unsafe { (driver()?.current)(&mut c) }.map_err(|e| format!("cuCtxGetCurrent: {}", e.0))?;
    if c.0.is_null() { return Err("make a ZLUDA CUDA context current before creating an execution context".into()); }
    Ok(c.0 as usize)
}
struct ContextGuard;
impl ContextGuard {
    fn push(ctx: usize) -> Result<Self> {
        unsafe { (driver()?.push)(CUcontext(ctx as *mut _)) }.map_err(|e| format!("cuCtxPushCurrent: {}", e.0))?;
        Ok(Self)
    }
}
impl Drop for ContextGuard {
    fn drop(&mut self) { if let Ok(d) = driver() { let mut c = CUcontext(std::ptr::null_mut()); unsafe { let _ = (d.pop)(&mut c); } } }
}
struct Buffer { ptr: usize, owner: usize }
impl Buffer {
    fn new(bytes: usize, owner: usize) -> Result<Self> { Ok(Self { ptr: unsafe { gpu(engine::alloc(bytes))? } as usize, owner }) }
    fn upload<T: Copy>(data: &[T], owner: usize) -> Result<Self> {
        let b = Self::new(std::mem::size_of_val(data), owner)?;
        unsafe { gpu(engine::write(b.ptr as *mut _, data.as_ptr().cast(), std::mem::size_of_val(data)))?; }
        Ok(b)
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        if let Ok(_scope) = ContextGuard::push(self.owner) { unsafe { let _ = engine::free(self.ptr as *mut _); } }
    }
}

// Immutable GPU state is shared only within the CUDA context that owns it.
pub(crate) struct Resources {
    pointers: Vec<usize>,
    metadata: Vec<Option<usize>>,
    binary_ranks: Vec<i32>,
    _constants: Buffer,
    functions: HashMap<&'static CStr, CUfunction>,
}
// Function handles are immutable; launches validate the current CUDA context.
// Buffer destruction pushes its owning context before freeing device memory.
unsafe impl Send for Resources {}
unsafe impl Sync for Resources {}
pub(crate) type ResourceCache = Mutex<HashMap<usize, Arc<Resources>>>;
impl Resources {
    fn new(plan: &Plan, owner: usize) -> Result<Self> {
        let mut pointers = vec![0usize; plan.graph.tensors.len()];
        let mut constant_data = Vec::with_capacity(plan.constant_bytes);
        let mut constant_offsets = Vec::new();
        fn pack<T: Copy>(bytes: &mut Vec<u8>, data: &[T]) -> usize {
            let offset = bytes.len().div_ceil(256) * 256; bytes.resize(offset, 0);
            let raw = unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) };
            bytes.extend_from_slice(raw); offset
        }
        for (i, t) in plan.graph.tensors.iter().enumerate() {
            if let Some(data) = &t.data {
                let offset = pack(&mut constant_data, data); constant_offsets.push((i, offset));
            }
        }
        let mut metadata = Vec::new();
        let mut binary_ranks = Vec::new();
        let mut functions = HashMap::new();
        let mut get = |image, name| -> Result<()> { functions.insert(name, unsafe { gpu(engine::function(image, name))? }); Ok(()) };
        for step in &plan.steps {
            let out = &plan.graph.tensors[step.node.output].shape;
            let meta = match &step.node.operation {
                Op::ElementWise { a, b, .. } => {
                    get(agx_libs::INFERENCE, c"agx_infer_binary")?;
                    Some(broadcast_metadata(out, &plan.graph.tensors[*a].shape, &plan.graph.tensors[*b].shape))
                }
                Op::Conv2d { .. } => { get(agx_libs::INFERENCE, c"agx_infer_conv2d")?; None }
                Op::Transpose { input, permutation } => {
                    get(agx_libs::INFERENCE, c"agx_infer_transpose")?;
                    let strides = strides(&plan.graph.tensors[*input].shape);
                    let mut m: Vec<i32> = out.iter().map(|&d| d as i32).collect();
                    m.extend(permutation.iter().map(|&i| strides[i] as i32)); Some(m)
                }
                Op::Activation { .. } => { get(agx_libs::INFERENCE, c"agx_infer_unary")?; None }
                Op::Pool2d { .. } => { get(agx_libs::INFERENCE, c"agx_infer_pool")?; None }
                Op::Gather { input, axis, indices, .. } => {
                    get(agx_libs::INFERENCE,c"agx_infer_gather")?;
                    let shape=&plan.graph.tensors[*input].shape;let axis=axis_index(*axis,shape.len())?;
                    Some(indices.iter().map(|&i|if i<0{(i+shape[axis] as i64) as i32}else{i as i32}).collect())
                }
                Op::LayerNorm { input, axis, .. } => {
                    let shape=&plan.graph.tensors[*input].shape;let axis=axis_index(*axis,shape.len())?;
                    let channels:usize=shape[axis..].iter().product();
                    get(agx_libs::INFERENCE,if channels>=32 {c"agx_infer_layer_norm_simd"}else{c"agx_infer_layer_norm"})?;None
                }
                Op::Reduce { input, axis, .. } => {
                    let shape=&plan.graph.tensors[*input].shape;let axis=axis_index(*axis,shape.len())?;
                    get(agx_libs::INFERENCE,if shape[axis]>=32 {c"agx_infer_reduce_simd"}else{c"agx_infer_reduce"})?;None
                }
                Op::Softmax { axis, .. } => {
                    let axis = axis_index(*axis, out.len())?;
                    let inner: usize = out[axis+1..].iter().product();
                    get(agx_libs::INFERENCE, if inner == 1 && out[axis] >= 32 { c"agx_infer_softmax_simd" } else { c"agx_infer_softmax" })?; None
                }
                Op::Concat { .. } => { get(agx_libs::INFERENCE, c"agx_infer_concat")?; None }
                _ => None,
            };
            binary_ranks.push(meta.as_ref().map_or(0, |m| (m.len() / 3) as i32));
            metadata.push(meta.as_deref().map(|v| pack(&mut constant_data, v)));
        }
        let constants = Buffer::upload(&constant_data, owner)?;
        for (i, offset) in constant_offsets { pointers[i] = constants.ptr.checked_add(offset).ok_or("constant address overflow")?; }
        for offset in metadata.iter_mut().flatten() { *offset = constants.ptr.checked_add(*offset).ok_or("metadata address overflow")?; }
        Ok(Self { pointers, metadata, binary_ranks, _constants: constants, functions })
    }
}
pub struct Context {
    pub plan: Arc<Plan>,
    owner: usize,
    pointers: Vec<usize>,
    bound: Vec<bool>,
    bindings_validated: bool,
    _workspace: Buffer,
    resources: Arc<Resources>,
    last_stream: Option<usize>,
}
impl Context {
    pub(crate) fn new(plan: Arc<Plan>, cache: &ResourceCache) -> Result<Self> {
        let owner = current()?;
        let resources = {
            let mut cache = cache.lock().map_err(|_| "GPU resource cache lock poisoned")?;
            if let Some(resources) = cache.get(&owner) { resources.clone() }
            else {
                let resources = Arc::new(Resources::new(&plan, owner)?);
                cache.insert(owner, resources.clone()); resources
            }
        };
        let workspace = Buffer::new(plan.workspace, owner)?;
        let mut pointers = resources.pointers.clone();
        for (i, offset) in plan.offsets.iter().enumerate() {
            if let Some(offset) = offset { pointers[i] = workspace.ptr.checked_add(*offset).ok_or("workspace address overflow")?; }
        }
        let count = pointers.len();
        Ok(Self { plan, owner, pointers, bound: vec![false; count], bindings_validated: false, _workspace: workspace, resources, last_stream: None })
    }
    pub fn bind(&mut self, name: &str, pointer: usize, bytes: usize) -> Result<()> {
        let g = &self.plan.graph;
        let i = g.tensors.iter().position(|t| t.name == name).ok_or_else(|| format!("unknown tensor {name}"))?;
        if !g.inputs.contains(&i) && !g.outputs.contains(&i) { return Err("only graph inputs and outputs can be bound".into()); }
        if pointer == 0 || pointer % 4 != 0 || bytes < g.tensors[i].bytes()? { return Err("binding needs an aligned, non-null device pointer and sufficient bytes".into()); }
        pointer.checked_add(g.tensors[i].bytes()?).ok_or("binding address overflow")?;
        self.pointers[i] = pointer; self.bound[i] = true; self.bindings_validated = false; Ok(())
    }
    unsafe fn launch(&self, name: &'static CStr, threads: usize, params: &mut [*mut c_void], stream: CUstream) -> Result<()> {
        let f = *self.resources.functions.get(name).ok_or("inference kernel was not prepared")?;
        gpu(engine::launch_kernel(f, [threads.div_ceil(128) as u32, 1], [128, 1], params, stream))
    }
    unsafe fn binary(&self, at: usize, a: usize, b: usize, y: usize, kind: Binary, relu: bool, stream: CUstream) -> Result<()> {
        let mut a = self.pointers[a]; let mut b = self.pointers[b]; let mut yptr = self.pointers[y];
        let mut meta = self.resources.metadata[at].ok_or("binary metadata is absent")?;
        let mut n = self.plan.graph.tensors[y].elements()? as i32;
        let mut rank = self.resources.binary_ranks[at];
        let mut op = match kind { Binary::Add => 0i32, Binary::Mul => 1, Binary::Sub => 2, Binary::Div => 3, Binary::Max => 4, Binary::Min => 5 };
        let mut relu = i32::from(relu);
        self.launch(c"agx_infer_binary", n as usize, &mut [arg(&mut a), arg(&mut b), arg(&mut yptr), arg(&mut meta), arg(&mut n), arg(&mut rank), arg(&mut op), arg(&mut relu)], stream)
    }
    unsafe fn unary(&self, input: usize, output: usize, kind: Unary, stream: CUstream) -> Result<()> {
        let mut x = self.pointers[input]; let mut y = self.pointers[output]; let mut n = self.plan.graph.tensors[output].elements()? as i32;
        let mut op = match kind { Unary::Relu => 0i32, Unary::Sigmoid => 1, Unary::Tanh => 2, Unary::Identity => 3, Unary::Sqrt => 4, Unary::Exp => 5, Unary::Log => 6, Unary::Abs => 7, Unary::Neg => 8, Unary::Reciprocal => 9, };
        self.launch(c"agx_infer_unary", n as usize, &mut [arg(&mut x), arg(&mut y), arg(&mut n), arg(&mut op)], stream)
    }
    /// Caller owns device buffers and keeps them alive until the stream finishes. One context is not concurrently enqueued.
    pub unsafe fn enqueue(&mut self, stream: CUstream) -> Result<()> {
        if current()? != self.owner { return Err("execution context belongs to another CUDA context".into()); }
        let g = &self.plan.graph;
        if !self.bindings_validated {
            let io: Vec<usize> = g.inputs.iter().chain(&g.outputs).copied().collect();
            for &i in &io { if !self.bound[i] { return Err(format!("tensor {} has no device binding", g.tensors[i].name)); } }
            for (at, &i) in io.iter().enumerate() {
                for &j in &io[..at] {
                    if self.pointers[i] < self.pointers[j] + g.tensors[j].bytes()? && self.pointers[j] < self.pointers[i] + g.tensors[i].bytes()? {
                        return Err("input/output bindings overlap".into());
                    }
                }
            }
            self.bindings_validated = true;
        }
        // Views track rebinding of external roots and reuse the root allocation's lifetime.
        for (id, &root) in self.plan.aliases.iter().enumerate() {
            if id != root { self.pointers[id] = self.pointers[root]; }
        }
        if let Some(last) = self.last_stream.filter(|&last| last != stream.0 as usize) { gpu(engine::synchronize(CUstream(last as *mut _)))?; }
        self.last_stream = Some(stream.0 as usize);
        for (at, step) in self.plan.steps.iter().enumerate() {
            let yid = step.node.output; let yshape = &g.tensors[yid].shape; let n = g.tensors[yid].elements()?;
            match &step.node.operation {
                Op::Activation { input, kind } => self.unary(*input, yid, *kind, stream)?,
                Op::ElementWise { a, b, kind } => self.binary(at, *a, *b, yid, *kind, step.relu, stream)?,
                Op::Reshape { input, .. } => {
                    if self.plan.aliases[yid] == yid { gpu(engine::copy(self.pointers[yid] as *mut _, self.pointers[*input] as *const _, n * 4, stream))?; }
                }
                Op::MatMul { a, b, trans_a, trans_b } => {
                    let ashape=&g.tensors[*a].shape;let bshape=&g.tensors[*b].shape;
                    let (ar,ac)=(ashape[ashape.len()-2],ashape[ashape.len()-1]);
                    let (br,bc)=(bshape[bshape.len()-2],bshape[bshape.len()-1]);
                    let k=if *trans_a {ar}else{ac};let m=yshape[yshape.len()-2];let cols=yshape[yshape.len()-1];
                    let batches=&yshape[..yshape.len()-2];let count:usize=batches.iter().product();
                    let ap=&ashape[..ashape.len()-2];let bp=&bshape[..bshape.len()-2];
                    let linear=|shape:&[usize]|shape.iter().product::<usize>()==1 || shape==batches;
                    let mut gemm=engine::Gemm { types:agx_libs::Types::S,ta:*trans_b,tb:*trans_a,
                        m:cols as u64,n:m as u64,k:k as u64,alpha:1.,beta:0.,
                        a:self.pointers[*b] as *const _,b:self.pointers[*a] as *const _,c:self.pointers[yid] as *mut _,
                        lda:bc as u64,ldb:ac as u64,ldc:cols as u64,
                        stride_a:if bp.iter().product::<usize>()==1 {0}else{(br*bc) as i64},
                        stride_b:if ap.iter().product::<usize>()==1 {0}else{(ar*ac) as i64},
                        stride_c:(m*cols) as i64,batch:count as u64 };
                    if linear(ap)&&linear(bp){gpu(engine::gemm(&gemm,stream))?;}
                    else {
                        // Mixed broadcast dimensions require different offsets, never replicated input buffers.
                        let offset=|shape:&[usize],mut index:usize|{
                            let mut result=0;let mut stride=1;
                            for axis in (0..batches.len()).rev(){
                                let coordinate=index%batches[axis];index/=batches[axis];
                                if axis+shape.len()>=batches.len(){let d=shape[axis+shape.len()-batches.len()];
                                    if d!=1{result+=coordinate*stride;}stride*=d;}
                            }result
                        };
                        gemm.batch=1;
                        for batch in 0..count{
                            gemm.a=(self.pointers[*b]+offset(bp,batch)*br*bc*4) as *const _;
                            gemm.b=(self.pointers[*a]+offset(ap,batch)*ar*ac*4) as *const _;
                            gemm.c=(self.pointers[yid]+batch*m*cols*4) as *mut _;
                            gpu(engine::gemm(&gemm,stream))?;
                        }
                    }
                }
                Op::Conv2d { input, weights, bias, stride, padding, dilation, groups } => {
                    let x = &g.tensors[*input].shape; let w = &g.tensors[*weights].shape;
                    let mut args: Vec<i32> = [x[0], x[1], x[2], x[3], w[0], w[2], w[3], yshape[2], yshape[3],
                        padding[0], padding[1], stride[0], stride[1], dilation[0], dilation[1], *groups,
                        usize::from(bias.is_some()), usize::from(step.relu)].into_iter().map(|v| v as i32).collect();
                    let mut xp = self.pointers[*input]; let mut wp = self.pointers[*weights];
                    let mut bp = bias.map_or(0, |b| self.pointers[b]); let mut yp = self.pointers[yid];
                    let mut params = vec![arg(&mut xp), arg(&mut wp), arg(&mut bp), arg(&mut yp)];
                    params.extend(args.iter_mut().map(arg));
                    self.launch(c"agx_infer_conv2d", n, &mut params, stream)?;
                }
                Op::Gather { input, axis, indices, .. } => {
                    let shape=&g.tensors[*input].shape;let axis=axis_index(*axis,shape.len())?;
                    let mut x=self.pointers[*input];let mut y=self.pointers[yid];let mut meta=self.resources.metadata[at].ok_or("missing Gather indices")?;
                    let mut count=indices.len() as i32;let mut axis_size=shape[axis] as i32;
                    let mut inner=shape[axis+1..].iter().product::<usize>() as i32;let mut size=n as i32;
                    self.launch(c"agx_infer_gather",n,&mut[arg(&mut x),arg(&mut y),arg(&mut meta),arg(&mut size),arg(&mut axis_size),arg(&mut count),arg(&mut inner)],stream)?;
                }
                Op::LayerNorm { input, axis, epsilon } => {
                    let shape=&g.tensors[*input].shape;let axis=axis_index(*axis,shape.len())?;
                    let mut channels=shape[axis..].iter().product::<usize>() as i32;
                    let mut rows=(n/channels as usize) as i32;let mut epsilon=*epsilon;
                    let mut x=self.pointers[*input];let mut y=self.pointers[yid];let simd=channels>=32;
                    self.launch(if simd{c"agx_infer_layer_norm_simd"}else{c"agx_infer_layer_norm"},rows as usize*if simd{32}else{1},
                        &mut[arg(&mut x),arg(&mut y),arg(&mut rows),arg(&mut channels),arg(&mut epsilon)],stream)?;
                }
                Op::Reduce { input, axis, mean, .. } => {
                    let shape=&g.tensors[*input].shape;let axis=axis_index(*axis,shape.len())?;
                    let mut rows=n as i32;let mut channels=shape[axis] as i32;
                    let mut inner=shape[axis+1..].iter().product::<usize>() as i32;let mut average=i32::from(*mean);
                    let mut x=self.pointers[*input];let mut y=self.pointers[yid];
                    let simd=channels>=32;let name=if simd{c"agx_infer_reduce_simd"}else{c"agx_infer_reduce"};
                    self.launch(name,n*if simd{32}else{1},&mut[arg(&mut x),arg(&mut y),arg(&mut rows),arg(&mut channels),arg(&mut inner),arg(&mut average)],stream)?;
                }
                Op::Softmax { input, axis } => {
                    let axis = axis_index(*axis, yshape.len())?;
                    let inner: usize = yshape[axis+1..].iter().product();
                    let mut rows = (n / yshape[axis]) as i32; let mut channels = yshape[axis] as i32; let mut inner = inner as i32;
                    let mut x = self.pointers[*input]; let mut y = self.pointers[yid];
                    if inner == 1 && channels >= 32 {
                        self.launch(c"agx_infer_softmax_simd", rows as usize * 32, &mut [arg(&mut x), arg(&mut y), arg(&mut rows), arg(&mut channels)], stream)?;
                    } else {
                        self.launch(c"agx_infer_softmax", rows as usize, &mut [arg(&mut x), arg(&mut y), arg(&mut rows), arg(&mut channels), arg(&mut inner)], stream)?;
                    }
                }
                Op::Transpose { input, .. } => {
                    let mut x = self.pointers[*input]; let mut y = self.pointers[yid]; let mut meta = self.resources.metadata[at].unwrap();
                    let mut size = n as i32; let mut rank = yshape.len() as i32;
                    self.launch(c"agx_infer_transpose", n, &mut [arg(&mut x), arg(&mut y), arg(&mut meta), arg(&mut size), arg(&mut rank)], stream)?;
                }
                Op::Pool2d { input, window, stride, padding, kind, include_padding } => {
                    let shape = &g.tensors[*input].shape;
                    let mut args: Vec<i32> = [n, shape[2], shape[3], yshape[2], yshape[3], window[0], window[1], padding[0], padding[1], stride[0], stride[1],
                        usize::from(*kind == Pool::Average), usize::from(*include_padding)].into_iter().map(|v| v as i32).collect();
                    let mut x = self.pointers[*input]; let mut y = self.pointers[yid];
                    let mut params = vec![arg(&mut x), arg(&mut y)]; params.extend(args.iter_mut().map(arg));
                    self.launch(c"agx_infer_pool", n, &mut params, stream)?;
                }
                Op::Concat { inputs, axis } => {
                    let axis = axis_index(*axis, yshape.len())?;
                    let inner: usize = yshape[axis+1..].iter().product(); let mut offset = 0i32;
                    for &input in inputs {
                        let mut size = g.tensors[input].elements()? as i32;
                        let mut chunk = (g.tensors[input].shape[axis] * inner) as i32;
                        let mut span = (yshape[axis] * inner) as i32;
                        let mut x = self.pointers[input]; let mut y = self.pointers[yid];
                        self.launch(c"agx_infer_concat", size as usize, &mut [arg(&mut x), arg(&mut y), arg(&mut size), arg(&mut chunk), arg(&mut span), arg(&mut offset)], stream)?;
                        offset += chunk;
                    }
                }
            }
        }
        Ok(())
    }
    pub fn synchronize(&mut self) -> Result<()> {
        let _scope = ContextGuard::push(self.owner)?;
        if let Some(stream) = self.last_stream.take() { unsafe { gpu(engine::synchronize(CUstream(stream as *mut _)))?; } }
        Ok(())
    }
}
impl Drop for Context { fn drop(&mut self) { let _ = self.synchronize(); } }
