//! Independent, bounded ONNX protobuf reader. Schema: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
//! Static f32 inference only. External tensor files, custom domains, training graphs and opaque engines are rejected.
use crate::graph::*;
use std::collections::HashMap;

#[derive(Clone, Copy)]
enum Value<'a> { Var(u64), Fixed32(u32), Fixed64, Bytes(&'a [u8]) }
#[derive(Clone, Copy)]
struct Field<'a> { id: u32, value: Value<'a> }
fn varint(data: &[u8], at: &mut usize) -> Result<u64> {
    let mut value=0;
    for shift in (0..70).step_by(7) {
        let b=*data.get(*at).ok_or("truncated protobuf varint")?; *at+=1;
        if shift==63 && b>1 { return Err("protobuf varint overflow".into()); }
        value|=((b&127) as u64)<<shift;
        if b&128==0 { return Ok(value); }
    }
    Err("protobuf varint overflow".into())
}
fn fields(data: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut out=Vec::new(); let mut at=0;
    while at<data.len() {
        if out.len()>1_000_000 { return Err("too many protobuf fields".into()); }
        let tag=varint(data,&mut at)?;
        let id=u32::try_from(tag>>3).map_err(|_| "protobuf field number overflow")?;
        if id==0 { return Err("protobuf field zero".into()); }
        let value=match tag&7 {
            0=>Value::Var(varint(data,&mut at)?),
            1=>{ let end=at.checked_add(8).ok_or("protobuf length overflow")?; data.get(at..end).ok_or("truncated fixed64")?; at=end; Value::Fixed64 },
            2=>{ let n=usize::try_from(varint(data,&mut at)?).map_err(|_| "protobuf length overflow")?; let end=at.checked_add(n).ok_or("protobuf length overflow")?;
                let bytes=data.get(at..end).ok_or("truncated protobuf bytes")?; at=end; Value::Bytes(bytes) },
            5=>{ let end=at.checked_add(4).ok_or("protobuf length overflow")?; let b=data.get(at..end).ok_or("truncated fixed32")?;
                at=end; Value::Fixed32(u32::from_le_bytes(b.try_into().unwrap())) },
            _=>return Err("unsupported protobuf wire type".into()),
        };
        out.push(Field{id,value});
    }
    Ok(out)
}
fn bytes<'a>(f: &[Field<'a>], id: u32) -> Result<&'a [u8]> {
    match f.iter().find(|f|f.id==id).map(|f|f.value) { Some(Value::Bytes(b))=>Ok(b),None=>Ok(&[]),_=>Err("wrong protobuf byte-field type".into()) }
}
fn integer(f: &[Field<'_>], id: u32, default: i64) -> Result<i64> {
    match f.iter().find(|f|f.id==id).map(|f|f.value) { Some(Value::Var(v))=>Ok(v as i64),None=>Ok(default),_=>Err("wrong protobuf integer-field type".into()) }
}
fn text(f: &[Field<'_>], id: u32) -> Result<String> { std::str::from_utf8(bytes(f,id)?).map(str::to_owned).map_err(|e|e.to_string()) }
fn integers(f: &[Field<'_>], id: u32) -> Result<Vec<i64>> {
    let mut out=Vec::new();
    for item in f.iter().filter(|f|f.id==id) { match item.value {
        Value::Var(v)=>out.push(v as i64),
        Value::Bytes(data)=>{let mut at=0;while at<data.len(){out.push(varint(data,&mut at)? as i64);}},
        _=>return Err("wrong protobuf repeated integer type".into()),
    }}
    Ok(out)
}
fn nested<'a>(f: &'a [Field<'a>], id: u32) -> impl Iterator<Item=Result<&'a [u8]>>+'a {
    f.iter().filter(move|f|f.id==id).map(|f|match f.value{Value::Bytes(b)=>Ok(b),_=>Err("wrong protobuf nested-message type".into())})
}
fn dims(v: Vec<i64>) -> Result<Vec<usize>> { v.into_iter().map(|v|usize::try_from(v).map_err(|_|"negative ONNX dimension".into())).collect() }
#[derive(Clone)]
enum Constant { Float(Vec<f32>), Int(Vec<i64>) }
fn tensor(data: &[u8]) -> Result<(String,Vec<usize>,Constant)> {
    let f=fields(data)?; let name=text(&f,8)?; let shape=dims(integers(&f,1)?)?;
    if !bytes(&f,13)?.is_empty() || integer(&f,14,0)?!=0 { return Err("external ONNX tensor storage is not supported".into()); }
    let raw=bytes(&f,9)?; let ty=integer(&f,2,0)?;
    let result=match ty {
        1=>{
            let n=elements(&shape)?;
            let values=if !raw.is_empty(){if raw.len()!=n*4{return Err("ONNX raw f32 length differs from shape".into());}raw.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect()}
            else {let mut v=Vec::new();for p in f.iter().filter(|f|f.id==4){match p.value{Value::Fixed32(bits)=>v.push(f32::from_bits(bits)),Value::Bytes(b)=>{if b.len()%4!=0{return Err("invalid packed float data".into());}v.extend(b.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())));},_=>return Err("invalid ONNX float data".into())}}v};
            if values.len()!=n || values.iter().any(|v|!v.is_finite()){return Err("ONNX constant size or value is invalid".into());}
            Constant::Float(values)
        }
        7=>{
            if shape.len()>MAX_RANK{return Err("ONNX integer tensor rank exceeds limit".into());}
            let n=shape.iter().try_fold(1usize,|a,&b|a.checked_mul(b)).ok_or("ONNX integer tensor size overflow")?;
            if n>MAX_ENGINE_BYTES/8{return Err("ONNX integer tensor exceeds size limit".into());}
            let values=if !raw.is_empty(){if raw.len()!=n*8{return Err("ONNX raw i64 length differs from shape".into());}raw.chunks_exact(8).map(|v|i64::from_le_bytes(v.try_into().unwrap())).collect()}
            else{integers(&f,7)?};
            if values.len()!=n{return Err("ONNX integer constant size differs from shape".into());}Constant::Int(values)
        }
        _=>return Err(format!("unsupported ONNX initializer data type {ty}")),
    };
    Ok((name,shape,result))
}
fn value_info(data:&[u8],static_required:bool)->Result<(String,Option<Vec<usize>>)> {
    let f=fields(data)?;let name=text(&f,1)?;let ty=fields(bytes(&f,2)?)?;let t=fields(bytes(&ty,1)?)?;
    if integer(&t,1,0)?!=1{return Err(format!("ONNX I/O {name} must be f32"));}
    if !t.iter().any(|f|f.id==2){return if static_required{Err("ONNX input needs a static shape".into())}else{Ok((name,None))};}
    let sh=fields(bytes(&t,2)?)?;let mut shape=Vec::new();
    for d in nested(&sh,1){let d=fields(d?)?;let value=integer(&d,1,-1)?;
        if value<=0{if static_required{return Err(format!("ONNX input {name} has a dynamic or empty dimension"));}return Ok((name,None));}
        shape.push(usize::try_from(value).map_err(|_|"ONNX shape overflow")?);
    }
    elements(&shape)?;Ok((name,Some(shape)))
}
#[derive(Clone)]
enum Attribute { Int(i64), Float(f32), Ints(Vec<i64>), String(String), Tensor(Vec<u8>) }
fn attributes(f:&[Field<'_>])->Result<HashMap<String,Attribute>> {
    let mut out=HashMap::new();
    for a in nested(f,5){let a=fields(a?)?;let name=text(&a,1)?;
        let v=match integer(&a,20,0)? {
            1=>match a.iter().find(|f|f.id==2).map(|f|f.value){Some(Value::Fixed32(bits))=>Attribute::Float(f32::from_bits(bits)),_=>return Err("missing ONNX float attribute".into())},
            2=>Attribute::Int(integer(&a,3,0)?),3=>Attribute::String(text(&a,4)?),4=>Attribute::Tensor(bytes(&a,5)?.to_vec()),
            7=>Attribute::Ints(integers(&a,8)?),_=>return Err(format!("unsupported ONNX attribute {name}")),
        };
        if out.insert(name,v).is_some(){return Err("duplicate ONNX attribute".into());}
    }
    Ok(out)
}
fn take_int(a:&mut HashMap<String,Attribute>,n:&str,d:i64)->Result<i64>{match a.remove(n){None=>Ok(d),Some(Attribute::Int(v))=>Ok(v),_=>Err(format!("ONNX {n} needs an integer"))}}
fn take_float(a:&mut HashMap<String,Attribute>,n:&str,d:f32)->Result<f32>{match a.remove(n){None=>Ok(d),Some(Attribute::Float(v))if v.is_finite()=>Ok(v),_=>Err(format!("ONNX {n} needs a finite float"))}}
fn take_ints(a:&mut HashMap<String,Attribute>,n:&str,d:&[i64])->Result<Vec<i64>>{match a.remove(n){None=>Ok(d.to_vec()),Some(Attribute::Ints(v))=>Ok(v),_=>Err(format!("ONNX {n} needs integers"))}}
fn take_string(a:&mut HashMap<String,Attribute>,n:&str,d:&str)->Result<String>{match a.remove(n){None=>Ok(d.into()),Some(Attribute::String(v))=>Ok(v),_=>Err(format!("ONNX {n} needs a string"))}}
fn pair(v:Vec<i64>)->Result<[usize;2]>{let v=dims(v)?;v.try_into().map_err(|_|"expected two spatial dimensions".into())}
fn add_named(g:&mut Graph,name:String,shape:Vec<usize>,data:Option<Vec<f32>>,ids:&mut HashMap<String,usize>)->Result<usize>{
    if ids.contains_key(&name){return Err(format!("duplicate ONNX value {name}"));}
    let i=g.add_input(name.clone(),shape,data)?;ids.insert(name,i);Ok(i)
}
fn spatial_attrs(a:&mut HashMap<String,Attribute>,kernel:[usize;2])->Result<([usize;2],[usize;2])>{
    let stride=pair(take_ints(a,"strides",&[1,1])?)?;
    let pads=take_ints(a,"pads",&[0,0,0,0])?;
    if pads.len()!=4 || pads[0]!=pads[2] || pads[1]!=pads[3]{return Err("only symmetric ONNX padding is supported".into());}
    let mut pad=pair(pads[..2].to_vec())?;
    let auto=take_string(a,"auto_pad","NOTSET")?;
    if auto=="VALID"{pad=[0,0];}else if auto!="NOTSET" && !auto.is_empty(){return Err("ONNX auto_pad requires explicit symmetric pads".into());}
    let declared=take_ints(a,"kernel_shape",&[kernel[0]as i64,kernel[1]as i64])?;
    if declared!=[kernel[0]as i64,kernel[1]as i64]{return Err("ONNX kernel_shape differs from weights".into());}
    Ok((stride,pad))
}

pub fn import(data:&[u8])->Result<Graph>{
    if data.len()>MAX_ENGINE_BYTES{return Err("ONNX model exceeds size limit".into());}
    let model=fields(data)?;let ir=integer(&model,1,0)?;
    if !(4..=13).contains(&ir){return Err(format!("unsupported ONNX IR version {ir}; expected 4 through 13"));}
    if model.iter().any(|f|f.id==20){return Err("ONNX training graphs are not supported".into());}
    let mut opset=None;
    for s in nested(&model,8){let f=fields(s?)?;let domain=text(&f,1)?;if domain.is_empty()||domain=="ai.onnx"{if opset.replace(integer(&f,2,0)?).is_some(){return Err("duplicate ONNX opset".into());}}}
    if !matches!(opset,Some(13..=22)){return Err("supported ONNX standard opsets are 13 through 22".into());}
    let proto=fields(bytes(&model,7)?)?;
    if proto.iter().any(|f|f.id==15){return Err("sparse ONNX initializers are not supported".into());}
    let mut g=Graph::default();let mut ids=HashMap::new();let mut ints=HashMap::new();
    for t in nested(&proto,5){let(name,shape,value)=tensor(t?)?;if name.is_empty()||ids.contains_key(&name)||ints.contains_key(&name){return Err("empty or duplicate ONNX initializer name".into());}match value{
        Constant::Float(v)=>{add_named(&mut g,name,shape,Some(v),&mut ids)?;},
        Constant::Int(v)=>{if ids.contains_key(&name)||ints.insert(name,v).is_some(){return Err("duplicate ONNX constant".into());}},
    }}
    for input in nested(&proto,11){let input=input?;let f=fields(input)?;let name=text(&f,1)?;
        if ids.contains_key(&name)||ints.contains_key(&name){continue;}
        let(name,shape)=value_info(input,true)?;add_named(&mut g,name,shape.unwrap(),None,&mut ids)?;
    }
    for n in nested(&proto,1){
        if g.nodes.len()>=MAX_TENSORS{return Err("too many ONNX nodes".into());}
        let n=fields(n?)?;let kind=text(&n,4)?;let domain=text(&n,7)?;
        if !domain.is_empty()&&domain!="ai.onnx"{return Err(format!("unsupported ONNX domain {domain}"));}
        if !text(&n,8)?.is_empty(){return Err("ONNX local function overloads are not supported".into());}
        let inputs=nested(&n,1).map(|v|std::str::from_utf8(v?).map(str::to_owned).map_err(|e|e.to_string())).collect::<Result<Vec<_>>>()?;
        let outputs=nested(&n,2).map(|v|std::str::from_utf8(v?).map(str::to_owned).map_err(|e|e.to_string())).collect::<Result<Vec<_>>>()?;
        if outputs.len()!=1 || outputs[0].is_empty() || ids.contains_key(&outputs[0]) || ints.contains_key(&outputs[0]){return Err(format!("{kind} requires one unique output"));}
        let expected = match kind.as_str() {
            "MatMul"|"Add"|"Mul"|"Sub"|"Div"|"Max"|"Min"|"Reshape" => 2..=2,
            "Gemm"|"Conv" => 2..=3, "Constant" => 0..=0, "Concat" => 1..=MAX_TENSORS,
            _ => 1..=1,
        };
        if !expected.contains(&inputs.len()) { return Err(format!("{kind}: wrong number of inputs")); }
        let mut attrs=attributes(&n)?;
        let get=|i:usize|->Result<usize>{let name=inputs.get(i).ok_or_else(||format!("{kind}: missing input {i}"))?;ids.get(name).copied().ok_or_else(||format!("{kind}: unknown f32 input {name}"))};
        if kind=="Constant"{
            if !inputs.is_empty(){return Err("Constant must not have inputs".into());}
            let Some(Attribute::Tensor(value))=attrs.remove("value")else{return Err("Constant requires a tensor value attribute".into())};
            if !attrs.is_empty(){return Err("unsupported Constant attributes".into());}
            let(_,shape,value)=tensor(&value)?;match value{
                Constant::Float(v)=>{add_named(&mut g,outputs[0].clone(),shape,Some(v),&mut ids)?;},
                Constant::Int(v)=>{ints.insert(outputs[0].clone(),v);},
            }continue;
        }
        let output=match kind.as_str(){
            "MatMul"=>g.add_node(Op::MatMul{a:get(0)?,b:get(1)?,trans_a:false,trans_b:false})?,
            "Gemm"=>{
                let ta=take_int(&mut attrs,"transA",0)?;let tb=take_int(&mut attrs,"transB",0)?;
                if ![0,1].contains(&ta)||![0,1].contains(&tb){return Err("invalid Gemm transpose flag".into());}
                let alpha=take_float(&mut attrs,"alpha",1.)?;let beta=take_float(&mut attrs,"beta",1.)?;
                let mut y=g.add_node(Op::MatMul{a:get(0)?,b:get(1)?,trans_a:ta!=0,trans_b:tb!=0})?;
                if alpha!=1.{let c=g.add_input(format!("__onnx_alpha_{y}"),vec![],Some(vec![alpha]))?;y=g.add_node(Op::ElementWise{a:y,b:c,kind:Binary::Mul})?;}
                if inputs.get(2).is_some_and(|s|!s.is_empty())&&beta!=0.{let mut b=get(2)?;if beta!=1.{let c=g.add_input(format!("__onnx_beta_{y}"),vec![],Some(vec![beta]))?;b=g.add_node(Op::ElementWise{a:b,b:c,kind:Binary::Mul})?;}y=g.add_node(Op::ElementWise{a:y,b,kind:Binary::Add})?;}y
            }
            "Add"|"Mul"|"Sub"|"Div"|"Max"|"Min"=>{
                if inputs.len()!=2{return Err(format!("{kind} currently requires exactly two inputs"));}
                g.add_node(Op::ElementWise{a:get(0)?,b:get(1)?,kind:match kind.as_str(){"Add"=>Binary::Add,"Mul"=>Binary::Mul,"Sub"=>Binary::Sub,"Div"=>Binary::Div,"Max"=>Binary::Max,_=>Binary::Min}})?
            }
            "Relu"|"Sigmoid"|"Tanh"|"Identity"=>g.add_node(Op::Activation{input:get(0)?,kind:match kind.as_str(){"Relu"=>Unary::Relu,"Sigmoid"=>Unary::Sigmoid,"Tanh"=>Unary::Tanh,_=>Unary::Identity}})?,
            "Softmax"=>g.add_node(Op::Softmax{input:get(0)?,axis:take_int(&mut attrs,"axis",-1)? as isize})?,
            "Conv"=>{
                let x=get(0)?;let w=get(1)?;let shape=&g.tensors[w].shape;
                if shape.len()!=4{return Err("Conv requires 2-D filters".into());}
                let(stride,padding)=spatial_attrs(&mut attrs,[shape[2],shape[3]])?;
                let dilation=pair(take_ints(&mut attrs,"dilations",&[1,1])?)?;
                let groups=usize::try_from(take_int(&mut attrs,"group",1)?).map_err(|_|"negative Conv groups")?;
                let bias=if inputs.get(2).is_some_and(|s|!s.is_empty()){Some(get(2)?)}else{None};
                g.add_node(Op::Conv2d{input:x,weights:w,bias,stride,padding,dilation,groups})?
            }
            "MaxPool"|"AveragePool"=>{
                let window=pair(take_ints(&mut attrs,"kernel_shape",&[])?)?;
                let(stride,padding)=spatial_attrs(&mut attrs,window)?;
                if take_int(&mut attrs,"ceil_mode",0)?!=0||take_int(&mut attrs,"storage_order",0)?!=0||take_ints(&mut attrs,"dilations",&[1,1])?!=[1,1]{return Err("pooling ceil, storage or dilation mode is unsupported".into());}
                let include=take_int(&mut attrs,"count_include_pad",0)?;
                if include!=0&&include!=1{return Err("invalid pooling padding mode".into());}
                g.add_node(Op::Pool2d{input:get(0)?,kind:if kind=="MaxPool"{Pool::Max}else{Pool::Average},window,stride,padding,include_padding:include!=0})?
            }
            "GlobalAveragePool"=>{let x=get(0)?;let s=&g.tensors[x].shape;if s.len()!=4{return Err("GlobalAveragePool requires NCHW input".into());}g.add_node(Op::Pool2d{input:x,kind:Pool::Average,window:[s[2],s[3]],stride:[1,1],padding:[0,0],include_padding:false})?}
            "Flatten"=>{let x=get(0)?;let s=&g.tensors[x].shape;let a=take_int(&mut attrs,"axis",1)?;let a=if a<0{a+s.len() as i64}else{a};if a<0||a as usize>s.len(){return Err("Flatten axis is invalid".into());}g.add_node(Op::Reshape{input:x,shape:vec![s[..a as usize].iter().product(),s[a as usize..].iter().product()]})?}
            "Reshape"=>{
                let x=get(0)?;if take_int(&mut attrs,"allowzero",0)?!=0{return Err("Reshape allowzero is not supported".into());}
                let shape_name=inputs.get(1).ok_or("Reshape needs shape input")?;let target=ints.get(shape_name).ok_or("Reshape shape must be a constant i64 tensor")?;
                let mut shape=Vec::new();let mut infer=None;let mut known=1usize;let total=g.tensors[x].elements()?;
                for(i,&d)in target.iter().enumerate(){let d=if d==0{*g.tensors[x].shape.get(i).ok_or("Reshape zero axis is outside input")? as i64}else{d};
                    if d==-1{if infer.replace(i).is_some(){return Err("multiple inferred Reshape dimensions".into());}shape.push(1);}else{let v=usize::try_from(d).map_err(|_|"negative Reshape dimension")?;if v==0{return Err("empty Reshape dimensions unsupported".into());}known=known.checked_mul(v).ok_or("Reshape size overflow")?;shape.push(v);}}
                if let Some(i)=infer{if total%known!=0{return Err("Reshape size mismatch".into());}shape[i]=total/known;}
                g.add_node(Op::Reshape{input:x,shape})?
            }
            "Transpose"=>{let x=get(0)?;let default:Vec<i64>=(0..g.tensors[x].shape.len() as i64).rev().collect();let permutation=dims(take_ints(&mut attrs,"perm",&default)?)?;g.add_node(Op::Transpose{input:x,permutation})?}
            "Concat"=>{let values=inputs.iter().map(|s|ids.get(s).copied().ok_or("unknown Concat input".into())).collect::<Result<Vec<_>>>()?;g.add_node(Op::Concat{inputs:values,axis:take_int(&mut attrs,"axis",0)? as isize})?}
            _=>return Err(format!("unsupported ONNX operator {kind}")),
        };
        if !attrs.is_empty(){return Err(format!("unsupported {kind} attributes: {:?}",attrs.keys().collect::<Vec<_>>()));}
        g.tensors[output].name=outputs[0].clone();ids.insert(outputs[0].clone(),output);
    }
    for output in nested(&proto,12){let(name,shape)=value_info(output?,false)?;let id=*ids.get(&name).ok_or_else(||format!("unknown ONNX output {name}"))?;
        if shape.is_some_and(|s|s!=g.tensors[id].shape){return Err(format!("ONNX declared output shape differs for {name}"));}g.outputs.push(id);}
    g.validate()?;Ok(g)
}

#[cfg(test)]
mod tests{
    use super::*;
    fn vi(mut v:u64)->Vec<u8>{let mut b=Vec::new();while v>=128{b.push((v as u8&127)|128);v>>=7;}b.push(v as u8);b}
    fn v(id:u64,value:u64)->Vec<u8>{let mut b=vi(id<<3);b.extend(vi(value));b}
    fn b(id:u64,value:&[u8])->Vec<u8>{let mut b=vi((id<<3)|2);b.extend(vi(value.len()as u64));b.extend(value);b}
    fn info(name:&str)->Vec<u8>{let mut shape=Vec::new();for d in [2,3]{shape.extend(b(1,&v(1,d)));}let mut t=v(1,1);t.extend(b(2,&shape));let mut ty=b(1,&t);let mut out=b(1,name.as_bytes());out.extend(b(2,&ty));ty.clear();out}
    #[test]fn imports_public_schema_graph(){
        let mut node=b(1,b"x");node.extend(b(2,b"y"));node.extend(b(4,b"Relu"));
        let mut graph=b(1,&node);graph.extend(b(11,&info("x")));graph.extend(b(12,&info("y")));
        let mut model=v(1,8);model.extend(b(7,&graph));model.extend(b(8,&v(2,17)));
        let g=import(&model).unwrap();assert_eq!(g.nodes.len(),1);assert_eq!(g.tensors[g.outputs[0]].shape,[2,3]);
        assert!(import(&model[..model.len()-1]).is_err());
    }
    #[test]fn malformed_protobuf_fails_without_panics(){
        for data in [vec![0],vec![0x80;11],vec![0x3a,0xff,0xff,0xff,0xff,0x7f],vec![0x0d,1,2],vec![0x0f]]{assert!(import(&data).is_err());}
    }
}
