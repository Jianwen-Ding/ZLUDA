//! Compile public ONNX or our JSON graph into an independently serialized engine. No GPU is needed.
use std::{ffi::CStr, path::PathBuf};
use tensorrt_agx::*;
fn main() {
    if let Err(e)=run(){eprintln!("ztrt-compile: {e}");std::process::exit(1);}
}
fn run()->Result<(),String>{
    let mut format="onnx".to_owned();let mut limit=1usize<<30;let mut paths=Vec::<PathBuf>::new();let mut args=std::env::args().skip(1);
    while let Some(a)=args.next(){match a.as_str(){
        "--format"=>format=args.next().ok_or("--format requires onnx or json")?,
        "--workspace"=>limit=args.next().ok_or("--workspace requires bytes")?.parse().map_err(|_|"invalid workspace byte count")?,
        "--help"|"-h"=>{println!("usage: ztrt-compile [--format onnx|json] [--workspace BYTES] INPUT OUTPUT.ztrt");return Ok(());},
        _ if a.starts_with('-')=>return Err(format!("unknown option {a}")),_=>paths.push(a.into()),
    }}
    if paths.len()!=2{return Err("usage: ztrt-compile [--format onnx|json] [--workspace BYTES] INPUT OUTPUT.ztrt".into());}
    if std::fs::metadata(&paths[0]).map_err(|e|e.to_string())?.len()>graph::MAX_ENGINE_BYTES as u64{return Err("input exceeds size limit".into());}
    let data=std::fs::read(&paths[0]).map_err(|e|e.to_string())?;
    unsafe{
        let engine=match format.as_str(){"onnx"=>ztrtBuildOnnx(data.as_ptr(),data.len(),limit),"json"=>ztrtBuildJson(data.as_ptr(),data.len(),limit),_=>return Err("format must be onnx or json; opaque vendor engines are not supported".into())};
        if engine.is_null(){return Err(CStr::from_ptr(ztrtGetLastError()).to_string_lossy().into_owned());}
        let size=ztrtSerializeEngine(engine,std::ptr::null_mut(),0);let mut out=vec![0u8;size];
        let written=ztrtSerializeEngine(engine,out.as_mut_ptr(),out.len());
        let workspace=ztrtGetWorkspaceSize(engine);let weights=ztrtGetConstantBytes(engine);let fused=ztrtGetFusedActivationCount(engine);
        ztrtDestroyEngine(engine);
        if written!=size||size==0{return Err("engine serialization failed".into());}
        // Atomic replacement avoids leaving a partial engine after a failed write.
        let tmp=paths[1].with_extension(format!("ztrt-tmp-{}",std::process::id()));
        std::fs::write(&tmp,&out).map_err(|e|e.to_string())?;
        if let Err(e)=std::fs::rename(&tmp,&paths[1]){let _=std::fs::remove_file(&tmp);return Err(e.to_string());}
        println!("wrote {size} bytes; workspace {workspace}; weights {weights}; fused activations {fused}");
    }
    Ok(())
}
