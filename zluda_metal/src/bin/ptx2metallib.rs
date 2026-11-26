//! ptx2metallib: compile a PTX file into a .metallib (plus optional .air/.ll dumps)
//!
//! usage: ptx2metallib INPUT.ptx [-o OUT.metallib] [--emit-air DIR] [--emit-ll DIR]
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut input = None;
    let mut output = None;
    let mut air_dir = None;
    let mut ll_dir: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "-o" => output = args.next().map(PathBuf::from),
            "--emit-air" => air_dir = args.next().map(PathBuf::from),
            "--emit-ll" => ll_dir = args.next().map(PathBuf::from),
            "-h" | "--help" => {
                eprintln!("usage: ptx2metallib INPUT.ptx [-o OUT.metallib] [--emit-air DIR] [--emit-ll DIR]");
                return ExitCode::SUCCESS;
            }
            _ => input = Some(PathBuf::from(a)),
        }
    }
    let Some(input) = input else {
        eprintln!(
            "usage: ptx2metallib INPUT.ptx [-o OUT.metallib] [--emit-air DIR] [--emit-ll DIR]"
        );
        return ExitCode::FAILURE;
    };
    let ptx = match std::fs::read_to_string(&input) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {}: {e}", input.display());
            return ExitCode::FAILURE;
        }
    };
    let opts = zluda_metal::compile::Options {
        keep_text: ll_dir.is_some(),
        portable_air: std::env::var_os("ZLUDA_PORTABLE_AIR").is_some(),
    };
    let module = match zluda_metal::compile::compile_ptx(&ptx, opts) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let output = output.unwrap_or_else(|| input.with_extension("metallib"));
    std::fs::write(&output, &module.metallib).unwrap();
    for k in &module.kernels {
        println!(
            "kernel {}: {} params ({} bytes), static shared {} bytes, dynamic shared {}",
            k.name,
            k.params.len(),
            k.params_size,
            k.static_shared,
            k.dynamic_shared
        );
    }
    if let Some(dir) = air_dir {
        std::fs::create_dir_all(&dir).unwrap();
        for k in &module.kernels {
            std::fs::write(dir.join(format!("{}.air", k.name)), &k.air).unwrap();
        }
    }
    if let Some(dir) = ll_dir {
        std::fs::create_dir_all(&dir).unwrap();
        let stem = input.file_stem().unwrap().to_string_lossy().into_owned();
        std::fs::write(
            dir.join(format!("{stem}.zluda.ll")),
            module.zluda_ir_text.unwrap(),
        )
        .unwrap();
        std::fs::write(dir.join(format!("{stem}.air.ll")), module.air_text.unwrap()).unwrap();
    }
    println!("wrote {}", output.display());
    ExitCode::SUCCESS
}
