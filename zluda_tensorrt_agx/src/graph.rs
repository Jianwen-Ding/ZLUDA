//! Independent graph format. No NVIDIA engine or binary format is parsed here.
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub type Result<T> = std::result::Result<T, String>;
pub const MAX_RANK: usize = 8;
pub const MAX_TENSORS: usize = 65536;
pub const MAGIC: &[u8; 8] = b"ZTRTAGX2";
pub const MAX_ENGINE_BYTES: usize = 256 << 20;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Vec<f32>>,
}
impl Tensor {
    pub fn elements(&self) -> Result<usize> { elements(&self.shape) }
    pub fn bytes(&self) -> Result<usize> { self.elements()?.checked_mul(4).ok_or("tensor byte size overflow".into()) }
}
pub fn elements(shape: &[usize]) -> Result<usize> {
    if shape.len() > MAX_RANK || shape.contains(&0) { return Err("tensor rank must be <= 8 and dimensions positive".into()); }
    let n = shape.iter().try_fold(1usize, |a, &b| a.checked_mul(b)).ok_or("tensor element count overflow")?;
    if n > i32::MAX as usize { return Err("tensor exceeds 32-bit index range".into()); }
    Ok(n)
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Unary { Relu, Sigmoid, Tanh, Identity, Sqrt, Exp, Log, Abs, Neg, Reciprocal }
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Binary { Add, Mul, Sub, Div, Max, Min }
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Pool { Max, Average }
fn pair_one() -> [usize; 2] { [1, 1] }
fn one() -> usize { 1 }
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Op {
    MatMul { a: usize, b: usize, #[serde(default)] trans_a: bool, #[serde(default)] trans_b: bool },
    ElementWise { a: usize, b: usize, kind: Binary },
    Activation { input: usize, kind: Unary },
    Conv2d { input: usize, weights: usize, #[serde(default)] bias: Option<usize>,
        #[serde(default = "pair_one")] stride: [usize; 2], #[serde(default)] padding: [usize; 2],
        #[serde(default = "pair_one")] dilation: [usize; 2], #[serde(default = "one")] groups: usize },
    Pool2d { input: usize, kind: Pool, window: [usize; 2], #[serde(default = "pair_one")] stride: [usize; 2],
        #[serde(default)] padding: [usize; 2], #[serde(default)] include_padding: bool },
    Softmax { input: usize, axis: isize },
    LayerNorm { input: usize, axis: isize, epsilon: f32 },
    Reduce { input: usize, axis: isize, mean: bool, keepdims: bool },
    Reshape { input: usize, shape: Vec<usize> },
    Transpose { input: usize, permutation: Vec<usize> },
    Concat { inputs: Vec<usize>, axis: isize },
}
impl Op {
    fn remap(&mut self, map: &[usize]) {
        match self {
            Self::MatMul { a, b, .. } | Self::ElementWise { a, b, .. } => { *a = map[*a]; *b = map[*b]; }
            Self::Conv2d { input, weights, bias, .. } => { *input = map[*input]; *weights = map[*weights]; if let Some(b) = bias { *b = map[*b]; } }
            Self::Concat { inputs, .. } => { for i in inputs { *i = map[*i]; } }
            Self::Activation { input, .. } | Self::Pool2d { input, .. } | Self::Softmax { input, .. }
                | Self::Reshape { input, .. } | Self::Transpose { input, .. } | Self::Reduce { input, .. } | Self::LayerNorm { input, .. } => *input = map[*input],
        }
    }
    pub fn inputs(&self) -> Vec<usize> {
        match self {
            Self::MatMul { a, b, .. } | Self::ElementWise { a, b, .. } => vec![*a, *b],
            Self::Conv2d { input, weights, bias, .. } => { let mut v = vec![*input, *weights]; v.extend(*bias); v },
            Self::Concat { inputs, .. } => inputs.clone(),
            Self::Activation { input, .. } | Self::Pool2d { input, .. } | Self::Softmax { input, .. }
                | Self::Reshape { input, .. } | Self::Transpose { input, .. } | Self::Reduce { input, .. } | Self::LayerNorm { input, .. } => vec![*input],
        }
    }
    pub fn shape(&self, tensors: &[Tensor]) -> Result<Vec<usize>> {
        let get = |i: usize| tensors.get(i).map(|t| t.shape.as_slice()).ok_or_else(|| format!("unknown tensor {i}"));
        let result = match self {
            Self::MatMul { a, b, trans_a, trans_b } => {
                let (a, b) = (get(*a)?, get(*b)?);
                if a.len()<2 || b.len()<2 { return Err("matmul requires matrix operands of rank >= 2".into()); }
                let (ar,ac)=(a[a.len()-2],a[a.len()-1]);let (br,bc)=(b[b.len()-2],b[b.len()-1]);
                let (m,k)=if *trans_a {(ac,ar)}else{(ar,ac)};
                let (kb,n)=if *trans_b {(bc,br)}else{(br,bc)};
                if k!=kb{return Err("matmul contraction dimensions differ".into());}
                let mut out=broadcast(&a[..a.len()-2],&b[..b.len()-2])?;out.extend([m,n]);out
            }
            Self::ElementWise { a, b, .. } => broadcast(get(*a)?, get(*b)?)?,
            Self::Activation { input, .. } => get(*input)?.to_vec(),
            Self::Conv2d { input, weights, bias, stride, padding, dilation, groups } => {
                let (x, w) = (get(*input)?, get(*weights)?);
                if x.len() != 4 || w.len() != 4 || *groups == 0 || x[1] % groups != 0 || w[0] % groups != 0 || w[1] != x[1] / groups {
                    return Err("conv requires NCHW input, KCRS weights, and compatible groups".into());
                }
                if let Some(b) = bias { if get(*b)? != [w[0]] { return Err("conv bias must have shape [output_channels]".into()); } }
                vec![x[0], w[0], spatial(x[2], w[2], padding[0], stride[0], dilation[0])?, spatial(x[3], w[3], padding[1], stride[1], dilation[1])?]
            }
            Self::Pool2d { input, window, stride, padding, .. } => {
                let x = get(*input)?;
                if x.len() != 4 { return Err("pool requires NCHW input".into()); }
                if window[0].checked_mul(window[1]).is_none_or(|n| n > i32::MAX as usize) { return Err("pool window exceeds index range".into()); }
                vec![x[0], x[1], spatial(x[2], window[0], padding[0], stride[0], 1)?, spatial(x[3], window[1], padding[1], stride[1], 1)?]
            }
            Self::Softmax { input, axis } => { let x = get(*input)?; axis_index(*axis, x.len())?; x.to_vec() },
            Self::LayerNorm { input, axis, epsilon } => {
                let shape=get(*input)?;axis_index(*axis,shape.len())?;
                if !epsilon.is_finite()||*epsilon<0.{return Err("invalid LayerNorm epsilon".into());}shape.to_vec()
            }
            Self::Reduce { input, axis, keepdims, .. } => {
                let mut shape=get(*input)?.to_vec();let axis=axis_index(*axis,shape.len())?;
                if *keepdims{shape[axis]=1;}else{shape.remove(axis);}shape
            }
            Self::Reshape { input, shape } => {
                if elements(get(*input)?)? != elements(shape)? { return Err("reshape changes element count".into()); }
                shape.clone()
            }
            Self::Transpose { input, permutation } => {
                let x = get(*input)?;
                if permutation.len() != x.len() || permutation.iter().copied().collect::<HashSet<_>>().len() != x.len() || permutation.iter().any(|&p| p >= x.len()) {
                    return Err("transpose needs a permutation of every input axis".into());
                }
                permutation.iter().map(|&p| x[p]).collect()
            }
            Self::Concat { inputs, axis } => {
                let mut x = get(*inputs.first().ok_or("concat has no inputs")?)?.to_vec();
                let axis = axis_index(*axis, x.len())?;
                x[axis] = 0;
                for &i in inputs {
                    let t = get(i)?;
                    if t.len() != x.len() || t.iter().enumerate().any(|(a, &d)| a != axis && d != x[a]) { return Err("concat dimensions differ".into()); }
                    x[axis] = x[axis].checked_add(t[axis]).ok_or("concat dimension overflow")?;
                }
                x
            }
        };
        elements(&result)?;
        Ok(result)
    }
}
pub fn spatial(input: usize, kernel: usize, pad: usize, stride: usize, dilation: usize) -> Result<usize> {
    if kernel == 0 || stride == 0 || dilation == 0 { return Err("kernel, stride and dilation must be positive".into()); }
    let padded = pad.checked_mul(2).and_then(|v| input.checked_add(v)).ok_or("padding overflow")?;
    let span = (kernel - 1).checked_mul(dilation).and_then(|v| v.checked_add(1)).ok_or("kernel span overflow")?;
    if stride > i32::MAX as usize || dilation > i32::MAX as usize
        || padded.checked_add(span).is_none_or(|n| n > i32::MAX as usize) {
        return Err("spatial parameters exceed 32-bit index range".into());
    }
    if padded < span { return Err("kernel is larger than padded input".into()); }
    Ok((padded - span) / stride + 1)
}
pub fn axis_index(axis: isize, rank: usize) -> Result<usize> {
    let a = if axis < 0 { axis + rank as isize } else { axis };
    if a < 0 || a as usize >= rank { return Err("axis outside tensor rank".into()); }
    Ok(a as usize)
}
pub fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    (0..rank).map(|i| {
        let ad = if i < rank - a.len() { 1 } else { a[i - (rank - a.len())] };
        let bd = if i < rank - b.len() { 1 } else { b[i - (rank - b.len())] };
        if ad != bd && ad != 1 && bd != 1 { Err("incompatible broadcast dimensions".into()) } else { Ok(ad.max(bd)) }
    }).collect()
}
pub fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() { s[i] = s[i + 1] * shape[i + 1]; }
    s
}
pub fn broadcast_metadata(out: &[usize], a: &[usize], b: &[usize]) -> Vec<i32> {
    let mut m: Vec<i32> = out.iter().map(|&d| d as i32).collect();
    for shape in [a, b] {
        let st = strides(shape);
        for i in 0..out.len() {
            m.push(if i < out.len() - shape.len() { 0 } else {
                let j = i - (out.len() - shape.len()); if shape[j] == 1 { 0 } else { st[j] as i32 }
            });
        }
    }
    // Collapse adjacent axes only when both input address maps remain linear.
    let rank = out.len();
    let mut axes: Vec<(i32, i32, i32)> = Vec::with_capacity(rank);
    for i in 0..rank {
        let (dim, sa, sb) = (m[i], m[rank+i], m[2*rank+i]);
        if dim == 1 { continue; }
        if let Some(last) = axes.last_mut() {
            if i64::from(last.1) == i64::from(sa) * i64::from(dim)
                && i64::from(last.2) == i64::from(sb) * i64::from(dim) {
                last.0 *= dim; last.1 = sa; last.2 = sb; continue;
            }
        }
        axes.push((dim, sa, sb));
    }
    axes.iter().map(|a| a.0).chain(axes.iter().map(|a| a.1)).chain(axes.iter().map(|a| a.2)).collect()
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node { pub output: usize, #[serde(flatten)] pub operation: Op }
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Graph {
    pub tensors: Vec<Tensor>, pub inputs: Vec<usize>, pub outputs: Vec<usize>, pub nodes: Vec<Node>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_limit: Option<usize>,
}
impl Graph {
    /// Constant outputs get an ordinary output-producing operation and a separate caller binding.
    pub fn normalize_constant_outputs(&mut self) -> Result<()> {
        for at in 0..self.outputs.len() {
            let input = self.outputs[at];
            if self.tensors.get(input).ok_or("unknown output tensor")?.data.is_none() { continue; }
            let name = self.tensors[input].name.clone();
            let mut internal = format!("__ztrt_constant_output_{input}");
            while self.tensors.iter().any(|t| t.name == internal) { internal.push('_'); }
            self.tensors[input].name = internal;
            let output = self.add_node(Op::Activation { input, kind: Unary::Identity })?;
            self.tensors[output].name = name; self.outputs[at] = output;
        }
        Ok(())
    }
    /// Remove unreachable operations and constants while retaining the declared input interface.
    pub fn eliminate_dead_nodes(self) -> Self {
        let mut live: HashSet<usize> = self.outputs.iter().chain(&self.inputs).copied().collect();
        for node in self.nodes.iter().rev() { if live.contains(&node.output) { live.extend(node.operation.inputs()); } }
        let mut map = vec![usize::MAX; self.tensors.len()];
        let tensors = self.tensors.into_iter().enumerate().filter_map(|(i, t)| {
            if live.contains(&i) { Some((i, t)) } else { None }
        }).enumerate().map(|(new, (old, t))| { map[old] = new; t }).collect();
        let nodes = self.nodes.into_iter().filter(|n| live.contains(&n.output)).map(|mut n| {
            n.output = map[n.output]; n.operation.remap(&map); n
        }).collect();
        Self { tensors, nodes, inputs: self.inputs.iter().map(|&i| map[i]).collect(), outputs: self.outputs.iter().map(|&i| map[i]).collect(), workspace_limit: self.workspace_limit }
    }
    pub fn add_input(&mut self, name: String, shape: Vec<usize>, data: Option<Vec<f32>>) -> Result<usize> {
        elements(&shape)?;
        if self.tensors.len() >= MAX_TENSORS { return Err("too many tensors".into()); }
        if let Some(values) = &data {
            let count = self.tensors.iter().filter_map(|t| t.data.as_ref()).try_fold(values.len(), |a, v| a.checked_add(v.len())).ok_or("constant size overflow")?;
            if count > MAX_ENGINE_BYTES / 4 { return Err("constant data exceeds engine size budget".into()); }
        }
        let id = self.tensors.len();
        if data.is_none() { self.inputs.push(id); }
        self.tensors.push(Tensor { name, shape, data });
        Ok(id)
    }
    pub fn add_node(&mut self, operation: Op) -> Result<usize> {
        if self.tensors.len() >= MAX_TENSORS || self.nodes.len() >= MAX_TENSORS { return Err("too many tensors or operations".into()); }
        let shape = operation.shape(&self.tensors)?;
        let id = self.tensors.len();
        self.tensors.push(Tensor { name: format!("tensor_{id}"), shape, data: None });
        self.nodes.push(Node { output: id, operation });
        Ok(id)
    }
    pub fn refresh_shapes(&mut self) -> Result<()> {
        for n in &self.nodes {
            let shape = n.operation.shape(&self.tensors)?;
            self.tensors.get_mut(n.output).ok_or("unknown node output tensor")?.shape = shape;
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        if self.tensors.len() > MAX_TENSORS || self.nodes.len() > MAX_TENSORS || self.outputs.is_empty() {
            return Err("graph needs outputs within the tensor/node limit".into());
        }
        let mut names = HashSet::new();
        for t in &self.tensors {
            if t.name.is_empty() || t.name.len() > 256 || t.name.contains('\0') || !names.insert(&t.name) { return Err("tensor names must be unique, nonempty and at most 256 bytes".into()); }
            let n = t.elements()?;
            if let Some(data) = &t.data { if data.len() != n || data.iter().any(|v| !v.is_finite()) { return Err("constant data length or value is invalid".into()); } }
        }
        let mut available = HashSet::new();
        for (i, t) in self.tensors.iter().enumerate() { if t.data.is_some() { available.insert(i); } }
        let mut seen = HashSet::new();
        for &i in &self.inputs {
            if i >= self.tensors.len() || !seen.insert(i) || self.tensors[i].data.is_some() { return Err("invalid or duplicate graph input".into()); }
            available.insert(i);
        }
        for n in &self.nodes {
            if n.output >= self.tensors.len() || available.contains(&n.output) || n.operation.inputs().iter().any(|i| !available.contains(i)) {
                return Err("graph is cyclic, unordered, or has an invalid tensor producer".into());
            }
            if n.operation.shape(&self.tensors)? != self.tensors[n.output].shape { return Err(format!("shape mismatch for {}", self.tensors[n.output].name)); }
            available.insert(n.output);
        }
        seen.clear();
        for &i in &self.outputs { if !available.contains(&i) || !seen.insert(i) || self.inputs.contains(&i) { return Err("output must be a unique operation result or constant, separate from graph inputs".into()); } }
        if available.len() != self.tensors.len() { return Err("graph has tensors without producers".into()); }
        Ok(())
    }
    pub fn serialize(&self) -> Result<Vec<u8>> {
        self.validate()?;
        // Bound writes during serialization rather than allocating an unbounded JSON copy first.
        // The header shares the allocation with the payload, avoiding another full-sized copy.
        struct Writer(Vec<u8>);
        impl std::io::Write for Writer {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                let size = self.0.len().checked_add(bytes.len()).filter(|&n| n <= MAX_ENGINE_BYTES + 16)
                    .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "engine exceeds size limit"))?;
                if size > self.0.capacity() {
                    let capacity = size.max(self.0.capacity().saturating_mul(2)).min(MAX_ENGINE_BYTES + 16);
                    self.0.reserve_exact(capacity - self.0.len());
                }
                self.0.extend_from_slice(bytes); Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        use std::io::Write;
        #[derive(Serialize)]
        struct TensorView<'a> { name: &'a str, shape: &'a [usize] }
        #[derive(Serialize)]
        struct GraphView<'a> {
            tensors: Vec<TensorView<'a>>, inputs: &'a [usize], outputs: &'a [usize],
            nodes: &'a [Node], workspace_limit: Option<usize>,
        }
        let view = GraphView { tensors: self.tensors.iter().map(|t| TensorView { name: &t.name, shape: &t.shape }).collect(),
            inputs: &self.inputs, outputs: &self.outputs, nodes: &self.nodes, workspace_limit: self.workspace_limit };
        let mut writer = Writer(Vec::with_capacity(4096));
        writer.0.extend(MAGIC); writer.0.extend([0u8; 16]);
        serde_json::to_writer(&mut writer, &view).map_err(|e| e.to_string())?;
        let metadata_size = writer.0.len() - 24;
        // Weight data is borrowed, never cloned into an intermediate graph or JSON array.
        for (id, tensor) in self.tensors.iter().enumerate() {
            if let Some(values) = &tensor.data {
                writer.write_all(&(id as u32).to_le_bytes()).map_err(|e| e.to_string())?;
                writer.write_all(&(values.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;
                for chunk in values.chunks(1024) {
                    let mut bytes = [0u8; 4096];
                    for (i, value) in chunk.iter().enumerate() { bytes[i*4..i*4+4].copy_from_slice(&value.to_le_bytes()); }
                    writer.write_all(&bytes[..chunk.len()*4]).map_err(|e| e.to_string())?;
                }
            }
        }
        let size = (writer.0.len() - 16) as u64;
        writer.0[8..16].copy_from_slice(&size.to_le_bytes());
        writer.0[16..24].copy_from_slice(&(metadata_size as u64).to_le_bytes()); Ok(writer.0)
    }
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        if data.len() < 16 || ![MAGIC.as_slice(), b"ZTRTAGX1"].contains(&&data[..8]) {
            return Err("unsupported independently serialized engine; NVIDIA engines are not supported".into());
        }
        let size = u64::from_le_bytes(data[8..16].try_into().unwrap());
        if size > MAX_ENGINE_BYTES as u64 || size as usize != data.len() - 16 { return Err("invalid engine payload size".into()); }
        if &data[..8] == b"ZTRTAGX1" {
            let graph: Self = serde_json::from_slice(&data[16..]).map_err(|e| e.to_string())?;
            graph.validate()?; return Ok(graph);
        }
        let length = data.get(16..24).ok_or("truncated metadata header")?;
        let length = usize::try_from(u64::from_le_bytes(length.try_into().unwrap())).map_err(|_| "metadata length overflow")?;
        let end = 24usize.checked_add(length).filter(|&n| n <= data.len()).ok_or("invalid metadata length")?;
        let mut graph: Self = serde_json::from_slice(&data[24..end]).map_err(|e| e.to_string())?;
        if graph.tensors.len() > MAX_TENSORS || graph.tensors.iter().any(|t| t.data.is_some()) { return Err("invalid binary engine metadata".into()); }
        let mut at = end;
        while at < data.len() {
            let header = data.get(at..at+12).ok_or("truncated weight record")?; at += 12;
            let id = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
            let count = u64::from_le_bytes(header[4..].try_into().unwrap());
            let tensor = graph.tensors.get_mut(id).ok_or("invalid weight tensor ID")?;
            if tensor.data.is_some() || count != tensor.elements()? as u64 { return Err("duplicate or invalid weight record".into()); }
            let bytes = (count as usize).checked_mul(4).and_then(|n| at.checked_add(n)).filter(|&n| n <= data.len()).ok_or("truncated weight data")?;
            tensor.data = Some(data[at..bytes].chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect());
            at = bytes;
        }
        graph.validate()?; Ok(graph)
    }
}

#[derive(Clone, Debug)]
pub struct Step { pub node: Node, pub relu: bool }
#[derive(Clone, Debug)]
pub struct Plan {
    pub graph: Graph, pub steps: Vec<Step>, pub offsets: Vec<Option<usize>>, pub workspace: usize,
    pub constant_bytes: usize, pub fused_activations: usize, pub aliases: Vec<usize>,
}
impl Plan {
    pub fn build(graph: Graph) -> Result<Self> {
        graph.validate()?;
        let mut uses = vec![0usize; graph.tensors.len()];
        for n in &graph.nodes { for i in n.operation.inputs() { uses[i] += 1; } }
        for &o in &graph.outputs { uses[o] += 1; }
        let mut steps: Vec<Step> = Vec::new();
        let mut fused = 0;
        for n in &graph.nodes {
            if let Op::Activation { input, kind: Unary::Relu } = n.operation {
                if uses[input] == 1 {
                    if let Some(prev) = steps.last_mut().filter(|p| p.node.output == input && matches!(p.node.operation, Op::ElementWise { .. } | Op::Conv2d { .. })) {
                        prev.node.output = n.output; prev.relu = true; fused += 1; continue;
                    }
                }
            }
            steps.push(Step { node: n.clone(), relu: false });
        }
        let external: HashSet<usize> = graph.inputs.iter().chain(&graph.outputs).copied().collect();
        let mut aliases: Vec<usize> = (0..graph.tensors.len()).collect();
        for step in &steps {
            if let Op::Reshape { input, .. } = step.node.operation {
                if !external.contains(&step.node.output) { aliases[step.node.output] = aliases[input]; }
            }
        }
        let mut last = vec![0usize; graph.tensors.len()];
        for (at, s) in steps.iter().enumerate() { for i in s.node.operation.inputs() { last[aliases[i]] = last[aliases[i]].max(at); } }
        for &o in &graph.outputs { last[aliases[o]] = steps.len(); }
        let mut offsets = vec![None; graph.tensors.len()];
        let mut free: Vec<(usize, usize)> = Vec::new();
        let mut live: HashMap<usize, (usize, usize)> = HashMap::new();
        let mut workspace = 0usize;

        // Outputs use caller buffers. Intermediates use best-fit reusable aligned slots.
        for (at, step) in steps.iter().enumerate() {
            let expired: Vec<_> = live.keys().copied().filter(|&i| last[i] < at).collect();
            for i in expired { free.push(live.remove(&i).unwrap()); }
            let id = step.node.output;
            if external.contains(&id) || aliases[id] != id { continue; }
            let bytes = graph.tensors[id].bytes()?.checked_add(255).ok_or("workspace overflow")? / 256 * 256;
            let slot = free.iter().enumerate().filter(|(_, s)| s.1 >= bytes).min_by_key(|(_, s)| s.1).map(|(i, _)| i);
            let (offset, size) = if let Some(i) = slot { free.swap_remove(i) } else {
                let o = workspace; workspace = workspace.checked_add(bytes).ok_or("workspace overflow")?; (o, bytes)
            };
            offsets[id] = Some(offset); live.insert(id, (offset, size));
        }
        let constant_bytes = graph.tensors.iter().filter(|t| t.data.is_some()).try_fold(0usize, |a, t| -> Result<usize> { a.checked_add(t.bytes()?).ok_or("constant byte size overflow".into()) })?;
        Ok(Self { graph, steps, offsets, workspace, constant_bytes, fused_activations: fused, aliases })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn graph() -> Graph {
        let mut g = Graph::default();
        let a = g.add_input("x".into(), vec![2, 3], None).unwrap();
        let b = g.add_input("w".into(), vec![3, 2], Some(vec![1.; 6])).unwrap();
        let c = g.add_node(Op::MatMul { a, b, trans_a: false, trans_b: false }).unwrap();
        g.outputs.push(c); g
    }
    #[test] fn reshape_views_extend_root_lifetimes() {
        let mut g=Graph::default();let x=g.add_input("x".into(),vec![2,3],None).unwrap();
        let root=g.add_node(Op::Activation{input:x,kind:Unary::Tanh}).unwrap();
        let view=g.add_node(Op::Reshape{input:root,shape:vec![6]}).unwrap();
        let chain=g.add_node(Op::Reshape{input:view,shape:vec![3,2]}).unwrap();
        let other=g.add_node(Op::Activation{input:x,kind:Unary::Relu}).unwrap();
        let other=g.add_node(Op::Reshape{input:other,shape:vec![3,2]}).unwrap();
        let y=g.add_node(Op::ElementWise{a:chain,b:other,kind:Binary::Add}).unwrap();g.outputs.push(y);
        let p=Plan::build(g).unwrap();assert_eq!(p.aliases[chain],root);
        assert_eq!(p.offsets[view],None);assert_eq!(p.offsets[chain],None);
        assert_eq!(p.workspace,512); // Both source allocations must remain live until the final add.
        let mut g=Graph::default();let x=g.add_input("x".into(),vec![2,3],None).unwrap();
        let v=g.add_node(Op::Reshape{input:x,shape:vec![6]}).unwrap();
        let y=g.add_node(Op::Activation{input:v,kind:Unary::Relu}).unwrap();g.outputs.push(y);
        let p=Plan::build(g).unwrap();assert_eq!(p.workspace,0);assert_eq!(p.aliases[v],x);
    }
    #[test] fn format_roundtrip_and_rejection() {
        let g = graph(); let data = g.serialize().unwrap();
        assert_eq!(Graph::deserialize(&data).unwrap().tensors.len(), 3);
        assert!(Graph::deserialize(b"not an independently generated engine").is_err());
        assert!(Graph::deserialize(&data[..data.len()-1]).is_err());
        let mut extra = data.clone(); extra.push(0); assert!(Graph::deserialize(&extra).is_err());
    }
    #[test] fn rejects_bad_graphs() {
        let mut g = graph(); g.nodes[0].output = 0; assert!(g.validate().is_err());
        let mut g = graph(); g.nodes[0].output = usize::MAX; assert!(g.refresh_shapes().is_err());
        let mut g = graph(); g.tensors[2].shape = vec![3, 2]; assert!(g.validate().is_err());
        assert!(elements(&[usize::MAX, 2]).is_err());
        assert!(broadcast(&[2, 3], &[2]).is_err());
        assert_eq!(broadcast(&[2, 3], &[3]).unwrap(), vec![2, 3]);
        assert_eq!(spatial(5, 3, 1, 2, 1).unwrap(), 3);
    }
    #[test] fn binary_weights_are_exact_and_malformed_records_fail() {
        let mut g = graph(); g.tensors[1].data = Some(vec![-0.0, f32::MIN_POSITIVE, f32::from_bits(1), -2.5, 3.25, 6.]);
        let bytes = g.serialize().unwrap(); let restored = Graph::deserialize(&bytes).unwrap();
        let bits = |g: &Graph| g.tensors[1].data.as_ref().unwrap().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&g), bits(&restored));
        let start = 24 + u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        let mut bad = bytes.clone(); bad[start..start+4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Graph::deserialize(&bad).is_err());
        let mut bad = bytes.clone(); bad.extend_from_slice(&bytes[start..]);
        let size = (bad.len()-16) as u64; bad[8..16].copy_from_slice(&size.to_le_bytes());
        assert!(Graph::deserialize(&bad).is_err());
        let json = serde_json::to_vec(&g).unwrap(); let mut legacy = b"ZTRTAGX1".to_vec();
        legacy.extend((json.len() as u64).to_le_bytes()); legacy.extend(json);
        assert_eq!(bits(&g), bits(&Graph::deserialize(&legacy).unwrap()));
    }
    #[test] fn collapsed_broadcast_preserves_every_address() {
        let shapes: Vec<Vec<usize>> = (0..81usize).map(|mut n| {
            (0..4).map(|_| { let d = n % 3 + 1; n /= 3; d }).collect()
        }).collect();
        for a in &shapes { for b in &shapes {
            let Ok(out) = broadcast(a, b) else { continue; };
            let meta = broadcast_metadata(&out, a, b); let rank = meta.len()/3;
            for id in 0..elements(&out).unwrap() {
                let mut expected = [0usize; 2]; let mut q = id;
                for axis in (0..4).rev() {
                    let coordinate = q % out[axis]; q /= out[axis];
                    for (j, shape) in [a, b].iter().enumerate() {
                        if shape[axis] != 1 { expected[j] += coordinate * strides(shape)[axis]; }
                    }
                }
                let mut actual = [0usize; 2]; let mut q = id;
                for axis in (0..rank).rev() {
                    let coordinate = q % meta[axis] as usize; q /= meta[axis] as usize;
                    for j in 0..2 { actual[j] += coordinate * meta[(j+1)*rank+axis] as usize; }
                }
                assert_eq!(actual, expected, "a={a:?} b={b:?} id={id}");
            }
        }}
    }
    #[test] fn fuses_only_single_use_relu() {
        let mut g = graph(); let a = g.outputs.pop().unwrap();
        let b = g.add_input("bias".into(), vec![2], Some(vec![1.; 2])).unwrap();
        let x = g.add_node(Op::ElementWise { a, b, kind: Binary::Add }).unwrap();
        let y = g.add_node(Op::Activation { input: x, kind: Unary::Relu }).unwrap(); g.outputs.push(y);
        let p = Plan::build(g.clone()).unwrap(); assert_eq!(p.fused_activations, 1); assert_eq!(p.steps.len(), 2);
        g.outputs.push(x); assert_eq!(Plan::build(g).unwrap().fused_activations, 0);
    }
    #[test] fn dead_branches_release_constants_and_preserve_inputs() {
        let mut g = graph();
        let dead = g.add_input("unused_weight".into(), vec![], Some(vec![5.])).unwrap();
        g.add_node(Op::ElementWise { a: 0, b: dead, kind: Binary::Add }).unwrap();
        let g = g.eliminate_dead_nodes();
        g.validate().unwrap();
        assert_eq!((g.tensors.len(), g.nodes.len(), g.inputs.len()), (3, 1, 1));
        assert_eq!(g.tensors[g.inputs[0]].name, "x");
    }
    #[test] fn workspace_is_reused_and_ceiling_roundtrips() {
        let mut g = graph();
        let mut output = g.outputs.pop().unwrap();
        for _ in 0..7 { output = g.add_node(Op::Activation { input: output, kind: Unary::Tanh }).unwrap(); }
        g.outputs.push(output); g.workspace_limit = Some(1024);
        let restored = Graph::deserialize(&g.serialize().unwrap()).unwrap();
        assert_eq!(restored.workspace_limit, Some(1024));
        assert_eq!(Plan::build(restored).unwrap().workspace, 512);
        assert!(spatial(1, 1, 0, usize::MAX, 1).is_err());
        assert!(spatial(1, 1, usize::MAX, 1, 1).is_err());
    }
}
