//! Independent TensorRT-style graph builder/runtime on portable MLIR -> AIR -> AGX kernels.
//! The C ABI and engine format are ours. No NVIDIA binary or serialized-engine format is consumed.
pub mod graph;
mod execute;
pub mod onnx;
use graph::*;
use std::{cell::RefCell, ffi::{c_char, c_void, CStr, CString}, panic::{catch_unwind, AssertUnwindSafe}, sync::Arc};
use cuda_types::cuda::CUstream;

thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap()); }
fn set_error(e: String) { ERROR.with(|v| *v.borrow_mut() = CString::new(e.replace('\0', " ")).unwrap()); }
fn boundary<T>(fallback: T, f: impl FnOnce() -> Result<T>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => value,
        Ok(Err(e)) => { set_error(e); fallback },
        Err(_) => { set_error("inference runtime caught an internal panic".into()); fallback },
    }
}
unsafe fn string<'a>(p: *const c_char) -> Result<&'a str> {
    if p.is_null() { return Err("null string".into()); }
    CStr::from_ptr(p).to_str().map_err(|e| e.to_string())
}
unsafe fn slice<'a, T>(p: *const T, n: usize, max: usize) -> Result<&'a [T]> {
    if n > max { return Err("array exceeds size limit".into()); }
    if n == 0 { return Ok(&[]); }
    if p.is_null() { return Err("null array".into()); }
    Ok(std::slice::from_raw_parts(p, n))
}
unsafe fn object<'a, T>(p: *const T) -> Result<&'a T> { p.as_ref().ok_or("null object".into()) }
unsafe fn object_mut<'a, T>(p: *mut T) -> Result<&'a mut T> { p.as_mut().ok_or("null object".into()) }
fn id(i: i32) -> Result<usize> { usize::try_from(i).map_err(|_| "negative tensor ID".into()) }
unsafe fn shape(p: *const i64, rank: usize) -> Result<Vec<usize>> {
    let result: Vec<usize> = slice(p, rank, MAX_RANK)?.iter().map(|&d| usize::try_from(d).map_err(|_| "negative dimension".into())).collect::<Result<_>>()?;
    elements(&result)?; Ok(result)
}
fn pos(v: i32) -> Result<usize> { if v <= 0 { Err("expected positive integer".into()) } else { Ok(v as usize) } }
fn nonneg(v: i32) -> Result<usize> { if v < 0 { Err("expected nonnegative integer".into()) } else { Ok(v as usize) } }
fn unary(v: i32) -> Result<Unary> { match v { 0 => Ok(Unary::Relu), 1 => Ok(Unary::Sigmoid), 2 => Ok(Unary::Tanh), 3 => Ok(Unary::Identity), 4 => Ok(Unary::Sqrt), 5 => Ok(Unary::Exp), 6 => Ok(Unary::Log), 7 => Ok(Unary::Abs), 8 => Ok(Unary::Neg), 9 => Ok(Unary::Reciprocal),  _ => Err("unsupported activation".into()) } }
fn binary(v: i32) -> Result<Binary> { match v { 0 => Ok(Binary::Add), 1 => Ok(Binary::Mul), 2 => Ok(Binary::Sub), 3 => Ok(Binary::Div), 4 => Ok(Binary::Max), 5 => Ok(Binary::Min), _ => Err("unsupported elementwise operation".into()) } }

pub struct Builder { graph: Graph, workspace_limit: usize }
pub struct Runtime;
pub struct Engine { plan: Arc<Plan>, names: Vec<CString>, serialized: Vec<u8>, resources: execute::ResourceCache }
impl Engine {
    fn build(mut graph: Graph, workspace_limit: usize) -> Result<Self> {
        let workspace_limit = graph.workspace_limit.unwrap_or(workspace_limit).min(workspace_limit);
        graph.workspace_limit = Some(workspace_limit);
        graph.refresh_shapes()?; graph.validate()?;
        graph.normalize_constant_outputs()?;
        let graph = graph.eliminate_dead_nodes();
        let serialized = graph.serialize()?;
        let names = graph.tensors.iter().map(|t| CString::new(t.name.as_str()).map_err(|e| e.to_string())).collect::<Result<Vec<_>>>()?;
        let plan = Plan::build(graph)?;
        if plan.workspace > workspace_limit { return Err(format!("workspace {} exceeds limit {workspace_limit}", plan.workspace)); }
        Ok(Self { plan: Arc::new(plan), names, serialized, resources: Default::default() })
    }
    fn tensor(&self, name: &str) -> Result<usize> { self.plan.graph.tensors.iter().position(|t| t.name == name).ok_or_else(|| format!("unknown tensor {name}")) }
}

#[no_mangle] pub extern "C" fn ztrtGetVersion() -> u32 { 100 }
#[no_mangle] pub extern "C" fn ztrtGetLastError() -> *const c_char { ERROR.with(|v| v.borrow().as_ptr()) }
#[no_mangle] pub extern "C" fn ztrtCreateBuilder() -> *mut Builder {
    boundary(std::ptr::null_mut(), || Ok(Box::into_raw(Box::new(Builder { graph: Graph::default(), workspace_limit: 1 << 30 }))))
}
#[no_mangle] pub unsafe extern "C" fn ztrtDestroyBuilder(p: *mut Builder) { boundary((), || { if !p.is_null() { drop(Box::from_raw(p)); } Ok(()) }); }
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderSetWorkspaceLimit(p: *mut Builder, bytes: usize) -> i32 { boundary(-1, || { object_mut(p)?.workspace_limit = bytes; Ok(0) }) }
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderAddInput(p: *mut Builder, name: *const c_char, dims: *const i64, rank: usize) -> i32 {
    boundary(-1, || Ok(object_mut(p)?.graph.add_input(string(name)?.into(), shape(dims, rank)?, None)? as i32))
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderAddConstant(p: *mut Builder, name: *const c_char, dims: *const i64, rank: usize, values: *const f32, count: usize) -> i32 {
    boundary(-1, || {
        let s = shape(dims, rank)?;
        if elements(&s)? != count { return Err("constant count differs from shape".into()); }
        let values = slice(values, count, MAX_ENGINE_BYTES / 4)?.to_vec();
        if values.iter().any(|v| !v.is_finite()) { return Err("constant values must be finite".into()); }
        Ok(object_mut(p)?.graph.add_input(string(name)?.into(), s, Some(values))? as i32)
    })
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderSetTensorName(p: *mut Builder, tensor: i32, name: *const c_char) -> i32 {
    boundary(-1, || {
        let g = &mut object_mut(p)?.graph; let i = id(tensor)?; let name = string(name)?;
        if name.is_empty() || name.len() > 256 || g.tensors.iter().enumerate().any(|(j, t)| j != i && t.name == name) { return Err("invalid or duplicate tensor name".into()); }
        g.tensors.get_mut(i).ok_or("unknown tensor")?.name = name.into(); Ok(0)
    })
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderTensorShape(p: *mut Builder, tensor: i32, dims: *mut i64, capacity: usize) -> i32 {
    boundary(-1, || {
        let g = &mut object_mut(p)?.graph; g.refresh_shapes()?;
        copy_shape(&g.tensors.get(id(tensor)?).ok_or("unknown tensor")?.shape, dims, capacity)
    })
}
unsafe fn copy_shape(s: &[usize], p: *mut i64, capacity: usize) -> Result<i32> {
    if capacity == 0 { return Ok(s.len() as i32); }
    if capacity < s.len() || p.is_null() { return Err("shape output buffer is too small or null".into()); }
    for (i, &d) in s.iter().enumerate() { *p.add(i) = d as i64; }
    Ok(s.len() as i32)
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderMarkOutput(p: *mut Builder, tensor: i32) -> i32 {
    boundary(-1, || { let g = &mut object_mut(p)?.graph; let i = id(tensor)?;
        if i >= g.tensors.len() || g.outputs.contains(&i) { return Err("invalid or duplicate output".into()); }
        g.outputs.push(i); Ok(0)
    })
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderAddMatMul(p: *mut Builder, a: i32, b: i32, ta: i32, tb: i32) -> i32 {
    boundary(-1, || Ok(object_mut(p)?.graph.add_node(Op::MatMul { a: id(a)?, b: id(b)?, trans_a: ta != 0, trans_b: tb != 0 })? as i32))
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderAddElementWise(p: *mut Builder, a: i32, b: i32, kind: i32) -> i32 {
    boundary(-1, || Ok(object_mut(p)?.graph.add_node(Op::ElementWise { a: id(a)?, b: id(b)?, kind: binary(kind)? })? as i32))
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderAddActivation(p: *mut Builder, input: i32, kind: i32) -> i32 {
    boundary(-1, || Ok(object_mut(p)?.graph.add_node(Op::Activation { input: id(input)?, kind: unary(kind)? })? as i32))
}
/// Fused unscaled suffix normalization; scale and bias can be added as elementwise operations.
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderAddLayerNorm(p:*mut Builder,input:i32,axis:i32,epsilon:f32)->i32 {
    boundary(-1,||Ok(object_mut(p)?.graph.add_node(Op::LayerNorm{input:id(input)?,axis:axis as isize,epsilon})? as i32))
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderConfigureLayerNorm(p:*mut Builder,output:i32,epsilon:f32)->i32 {
    boundary(-1,||{
        let g=&mut object_mut(p)?.graph;let output=id(output)?;
        let at=g.nodes.iter().position(|n|n.output==output).ok_or("unknown normalization layer")?;
        let mut candidate=g.nodes[at].operation.clone();
        if let Op::LayerNorm{epsilon:e,..}=&mut candidate{*e=epsilon;}else{return Err("expected normalization layer".into());}
        candidate.shape(&g.tensors)?;g.nodes[at].operation=candidate;Ok(0)
    })
}
/// Independent reduction API: kind 0 = sum, kind 1 = mean; axes are a rank-relative bitmask.
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderAddReduce(p:*mut Builder,input:i32,axes:u32,kind:i32,keepdims:i32)->i32 {
    boundary(-1,||{
        if ![0,1].contains(&kind)||![0,1].contains(&keepdims){return Err("unsupported reduction kind or keepdims flag".into());}
        let g=&mut object_mut(p)?.graph;let mut y=id(input)?;
        let rank=g.tensors.get(y).ok_or("unknown reduction input")?.shape.len();
        if axes>>rank!=0{return Err("reduction axis mask exceeds input rank".into());}
        if axes==0{return Ok(g.add_node(Op::Activation{input:y,kind:Unary::Identity})? as i32);}
        for axis in (0..rank).rev().filter(|&i|axes&(1u32<<i)!=0){
            y=g.add_node(Op::Reduce{input:y,axis:axis as isize,mean:kind==1,keepdims:keepdims!=0})?;
        }
        Ok(y as i32)
    })
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderAddSoftmax(p: *mut Builder, input: i32, axis: i32) -> i32 {
    boundary(-1, || Ok(object_mut(p)?.graph.add_node(Op::Softmax { input: id(input)?, axis: axis as isize })? as i32))
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderAddReshape(p: *mut Builder, input: i32, dims: *const i64, rank: usize) -> i32 {
    boundary(-1, || Ok(object_mut(p)?.graph.add_node(Op::Reshape { input: id(input)?, shape: shape(dims, rank)? })? as i32))
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderAddTranspose(p: *mut Builder, input: i32, permutation: *const i32, rank: usize) -> i32 {
    boundary(-1, || {
        let permutation = slice(permutation, rank, MAX_RANK)?.iter().map(|&p| id(p)).collect::<Result<Vec<_>>>()?;
        Ok(object_mut(p)?.graph.add_node(Op::Transpose { input: id(input)?, permutation })? as i32)
    })
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderAddConcat(p: *mut Builder, inputs: *const i32, count: usize, axis: i32) -> i32 {
    boundary(-1, || {
        let inputs = slice(inputs, count, MAX_TENSORS)?.iter().map(|&p| id(p)).collect::<Result<Vec<_>>>()?;
        Ok(object_mut(p)?.graph.add_node(Op::Concat { inputs, axis: axis as isize })? as i32)
    })
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderAddConv2d(p: *mut Builder, input: i32, weights: i32, bias: i32,
    sh: i32, sw: i32, ph: i32, pw: i32, dh: i32, dw: i32, groups: i32) -> i32 {
    boundary(-1, || Ok(object_mut(p)?.graph.add_node(Op::Conv2d { input: id(input)?, weights: id(weights)?, bias: if bias == -1 { None } else { Some(id(bias)?) },
        stride: [pos(sh)?, pos(sw)?], padding: [nonneg(ph)?, nonneg(pw)?], dilation: [pos(dh)?, pos(dw)?], groups: pos(groups)? })? as i32))
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderAddPool2d(p: *mut Builder, input: i32, kind: i32, rh: i32, rw: i32, sh: i32, sw: i32, ph: i32, pw: i32, include_padding: i32) -> i32 {
    boundary(-1, || Ok(object_mut(p)?.graph.add_node(Op::Pool2d { input: id(input)?, kind: match kind { 0 => Pool::Max, 1 => Pool::Average, _ => return Err("unsupported pooling mode".into()) },
        window: [pos(rh)?, pos(rw)?], stride: [pos(sh)?, pos(sw)?], padding: [nonneg(ph)?, nonneg(pw)?], include_padding: include_padding != 0 })? as i32))
}
/// Replace one operation's attributes transactionally and refresh dependent shapes.
fn modify(g: &mut Graph, output: usize, f: impl FnOnce(&mut Op) -> Result<()>) -> Result<()> {
    let at = g.nodes.iter().position(|n| n.output == output).ok_or("tensor has no operation")?;
    let old = g.nodes[at].operation.clone();
    if let Err(e) = f(&mut g.nodes[at].operation).and_then(|_| g.refresh_shapes()) {
        g.nodes[at].operation = old; let _ = g.refresh_shapes(); return Err(e);
    }
    Ok(())
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderConfigureConv2d(p: *mut Builder, output: i32, sh: i32, sw: i32, ph: i32, pw: i32, dh: i32, dw: i32, groups_: i32) -> i32 {
    boundary(-1, || { modify(&mut object_mut(p)?.graph, id(output)?, |op| {
        let Op::Conv2d { stride, padding, dilation, groups, .. } = op else { return Err("not a convolution".into()) };
        *stride = [pos(sh)?, pos(sw)?]; *padding = [nonneg(ph)?, nonneg(pw)?]; *dilation = [pos(dh)?, pos(dw)?]; *groups = pos(groups_)?; Ok(())
    })?; Ok(0) })
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderConfigurePool2d(p: *mut Builder, output: i32, sh: i32, sw: i32, ph: i32, pw: i32, include_: i32) -> i32 {
    boundary(-1, || { modify(&mut object_mut(p)?.graph, id(output)?, |op| {
        let Op::Pool2d { stride, padding, include_padding, .. } = op else { return Err("not a pooling operation".into()) };
        *stride = [pos(sh)?, pos(sw)?]; *padding = [nonneg(ph)?, nonneg(pw)?]; *include_padding = include_ != 0; Ok(())
    })?; Ok(0) })
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderConfigureSoftmax(p: *mut Builder, output: i32, axis_: i32) -> i32 {
    boundary(-1, || { modify(&mut object_mut(p)?.graph, id(output)?, |op| {
        let Op::Softmax { axis, .. } = op else { return Err("not a softmax".into()) }; *axis = axis_ as isize; Ok(())
    })?; Ok(0) })
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderConfigureReshape(p: *mut Builder, output: i32, dims: *const i64, rank: usize) -> i32 {
    boundary(-1, || { let dims = shape(dims, rank)?; modify(&mut object_mut(p)?.graph, id(output)?, |op| {
        let Op::Reshape { shape, .. } = op else { return Err("not a reshape".into()) }; *shape = dims; Ok(())
    })?; Ok(0) })
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderBuild(p: *const Builder) -> *mut Engine {
    boundary(std::ptr::null_mut(), || { let b = object(p)?; Ok(Box::into_raw(Box::new(Engine::build(b.graph.clone(), b.workspace_limit)?))) })
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuildJson(data: *const u8, size: usize, workspace_limit: usize) -> *mut Engine {
    boundary(std::ptr::null_mut(), || {
        let data = slice(data, size, MAX_ENGINE_BYTES)?;
        let graph: Graph = serde_json::from_slice(data).map_err(|e| e.to_string())?;
        // JSON describes declared shapes: reject inconsistencies instead of silently repairing them.
        graph.validate()?;
        Ok(Box::into_raw(Box::new(Engine::build(graph, workspace_limit)?)))
    })
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuildOnnx(data: *const u8, size: usize, workspace_limit: usize) -> *mut Engine {
    boundary(std::ptr::null_mut(), || {
        let graph = onnx::import(slice(data, size, MAX_ENGINE_BYTES)?)?;
        Ok(Box::into_raw(Box::new(Engine::build(graph, workspace_limit)?)))
    })
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderImportOnnx(p: *mut Builder, data: *const u8, size: usize) -> i32 {
    boundary(-1, || {
        let b = object_mut(p)?;
        if !b.graph.tensors.is_empty() { return Err("ONNX import requires an empty network".into()); }
        let graph = onnx::import(slice(data, size, MAX_ENGINE_BYTES)?)?;
        b.graph = graph; Ok(0)
    })
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderIOCount(p: *const Builder, output: i32) -> i32 {
    boundary(-1, || { let g = &object(p)?.graph; Ok(if output == 0 { g.inputs.len() } else { g.outputs.len() } as i32) })
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderIOTensor(p: *const Builder, output: i32, index: i32) -> i32 {
    boundary(-1, || { let g = &object(p)?.graph; let ids = if output == 0 { &g.inputs } else { &g.outputs };
        Ok(*ids.get(id(index)?).ok_or("I/O index outside network")? as i32)
    })
}
#[no_mangle] pub unsafe extern "C" fn ztrtBuilderTensorName(p: *const Builder, tensor: i32, dest: *mut c_char, capacity: usize) -> usize {
    boundary(0, || {
        let name = &object(p)?.graph.tensors.get(id(tensor)?).ok_or("unknown tensor")?.name;
        let size = name.len() + 1;
        if dest.is_null() && capacity == 0 { return Ok(size); }
        if dest.is_null() || capacity < size { return Err("tensor name buffer is too small or null".into()); }
        std::ptr::copy_nonoverlapping(name.as_ptr(), dest.cast(), name.len()); *dest.add(name.len()) = 0; Ok(size)
    })
}
#[no_mangle] pub extern "C" fn ztrtCreateRuntime() -> *mut Runtime { boundary(std::ptr::null_mut(), || Ok(Box::into_raw(Box::new(Runtime)))) }
#[no_mangle] pub unsafe extern "C" fn ztrtDestroyRuntime(p: *mut Runtime) { boundary((), || { if !p.is_null() { drop(Box::from_raw(p)); } Ok(()) }); }
#[no_mangle] pub unsafe extern "C" fn ztrtDeserializeEngine(p: *const Runtime, data: *const u8, size: usize, workspace_limit: usize) -> *mut Engine {
    boundary(std::ptr::null_mut(), || { object(p)?; let graph = Graph::deserialize(slice(data, size, MAX_ENGINE_BYTES + 16)?)?;
        Ok(Box::into_raw(Box::new(Engine::build(graph, workspace_limit)?)))
    })
}
#[no_mangle] pub unsafe extern "C" fn ztrtDestroyEngine(p: *mut Engine) { boundary((), || { if !p.is_null() { drop(Box::from_raw(p)); } Ok(()) }); }
#[no_mangle] pub unsafe extern "C" fn ztrtSerializeEngine(p: *const Engine, data: *mut u8, capacity: usize) -> usize {
    boundary(0, || { let e = object(p)?; if data.is_null() && capacity == 0 { return Ok(e.serialized.len()); }
        if data.is_null() || capacity < e.serialized.len() { return Err("serialization buffer is too small or null".into()); }
        std::ptr::copy_nonoverlapping(e.serialized.as_ptr(), data, e.serialized.len()); Ok(e.serialized.len())
    })
}
#[no_mangle] pub unsafe extern "C" fn ztrtGetNbIOTensors(p: *const Engine) -> i32 { boundary(-1, || { let e = object(p)?; Ok((e.plan.graph.inputs.len() + e.plan.graph.outputs.len()) as i32) }) }
#[no_mangle] pub unsafe extern "C" fn ztrtGetIOTensorName(p: *const Engine, index: i32) -> *const c_char {
    boundary(std::ptr::null(), || { let e = object(p)?; let i = e.plan.graph.inputs.iter().chain(&e.plan.graph.outputs).nth(id(index)?).ok_or("I/O index outside engine")?; Ok(e.names[*i].as_ptr()) })
}
#[no_mangle] pub unsafe extern "C" fn ztrtGetTensorIOMode(p: *const Engine, name: *const c_char) -> i32 {
    boundary(-1, || { let e = object(p)?; let i = e.tensor(string(name)?)?;
        if e.plan.graph.inputs.contains(&i) { Ok(0) } else if e.plan.graph.outputs.contains(&i) { Ok(1) } else { Err("tensor is not an input or output".into()) }
    })
}
#[no_mangle] pub unsafe extern "C" fn ztrtGetTensorShape(p: *const Engine, name: *const c_char, dims: *mut i64, capacity: usize) -> i32 {
    boundary(-1, || { let e = object(p)?; let i = e.tensor(string(name)?)?; copy_shape(&e.plan.graph.tensors[i].shape, dims, capacity) })
}
#[no_mangle] pub unsafe extern "C" fn ztrtGetTensorBytes(p: *const Engine, name: *const c_char) -> usize {
    boundary(0, || { let e = object(p)?; e.plan.graph.tensors[e.tensor(string(name)?)?].bytes() })
}
#[no_mangle] pub unsafe extern "C" fn ztrtGetWorkspaceSize(p: *const Engine) -> usize { boundary(0, || Ok(object(p)?.plan.workspace)) }
#[no_mangle] pub unsafe extern "C" fn ztrtGetConstantBytes(p: *const Engine) -> usize { boundary(0, || Ok(object(p)?.plan.constant_bytes)) }
#[no_mangle] pub unsafe extern "C" fn ztrtGetFusedActivationCount(p: *const Engine) -> usize { boundary(0, || Ok(object(p)?.plan.fused_activations)) }
#[no_mangle] pub unsafe extern "C" fn ztrtCreateExecutionContext(p: *const Engine) -> *mut execute::Context {
    boundary(std::ptr::null_mut(), || Ok(Box::into_raw(Box::new(execute::Context::new(object(p)?.plan.clone(), &object(p)?.resources)?))))
}
#[no_mangle] pub unsafe extern "C" fn ztrtDestroyExecutionContext(p: *mut execute::Context) { boundary((), || { if !p.is_null() { drop(Box::from_raw(p)); } Ok(()) }); }
#[no_mangle] pub unsafe extern "C" fn ztrtSetTensorAddress(p: *mut execute::Context, name: *const c_char, data: *mut c_void, bytes: usize) -> i32 {
    boundary(-1, || { object_mut(p)?.bind(string(name)?, data as usize, bytes)?; Ok(0) })
}
#[no_mangle] pub unsafe extern "C" fn ztrtEnqueueV3(p: *mut execute::Context, stream: *mut c_void) -> i32 {
    boundary(-1, || { object_mut(p)?.enqueue(CUstream(stream.cast()))?; Ok(0) })
}
#[no_mangle] pub unsafe extern "C" fn ztrtSynchronize(p: *mut execute::Context) -> i32 { boundary(-1, || { object_mut(p)?.synchronize()?; Ok(0) }) }
