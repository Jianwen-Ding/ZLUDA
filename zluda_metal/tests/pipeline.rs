//! Tests of the PTX -> AIR -> metallib pipeline and of the CUDA driver API on
//! the Metal backend. Kernels are compiled to PTX with clang (never nvcc).
use std::path::PathBuf;
use std::process::Command;
use zluda_metal::compile;
use zluda_metal::cuda::*;

fn e2e_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/e2e")
}

fn clang_ptx() -> String {
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("kernels.ptx");
    let clang = std::env::var("CLANG").unwrap_or_else(|_| "clang++".to_string());
    let status = Command::new(clang)
        .args([
            "-x",
            "cuda",
            "--cuda-device-only",
            "-nocudainc",
            "-nocudalib",
            "--cuda-gpu-arch=sm_70",
            "-Xclang",
            "-target-feature",
            "-Xclang",
            "+ptx70",
            "-O2",
            "-S",
            "-include",
        ])
        .arg(e2e_dir().join("clang_cuda.h"))
        .arg("-o")
        .arg(&out)
        .arg(e2e_dir().join("kernels.cu"))
        .status()
        .expect("clang is required to build the test kernels");
    assert!(status.success());
    std::fs::read_to_string(out).unwrap()
}

#[test]
fn air_is_well_formed() {
    let module = compile::compile_ptx(&clang_ptx(), compile::Options { keep_text: true }).unwrap();
    let air = module.air_text.unwrap();
    assert!(air.contains("target triple = \"air64-apple-macosx"));
    assert!(air.contains("!air.version"));
    for k in &module.kernels {
        assert!(
            air.contains(&format!("!{{ptr @{}, ", k.name)),
            "{} missing from !air.kernel",
            k.name
        );
    }
    for (line_no, line) in air.lines().enumerate() {
        let ctx = || format!("line {}: {line}", line_no + 1);
        assert!(
            !line.contains("addrspacecast"),
            "address space cast left: {}",
            ctx()
        );
        assert!(
            !line.contains("llvm.amdgcn"),
            "AMD intrinsic left: {}",
            ctx()
        );
        assert!(
            !line.contains("__zluda"),
            "unresolved ZLUDA helper: {}",
            ctx()
        );
        assert!(
            !line.contains("ptrtoint"),
            "pointer to integer cast left: {}",
            ctx()
        );
        if line.contains("inttoptr") {
            assert!(
                line.contains("to ptr addrspace(1)"),
                "only device pointers may come from integers: {}",
                ctx()
            );
        }
        // AIR has thread (0), device (1), constant (2) and threadgroup (3)
        // memory; AMD constant (4) / private (5) numbering must be gone
        for bad in ["addrspace(4)", "addrspace(5)", "addrspace(6)"] {
            assert!(!line.contains(bad), "non-AIR address space: {}", ctx());
        }
    }
}

#[test]
fn metallib_round_trip() {
    let module = compile::compile_ptx(&clang_ptx(), compile::Options::default()).unwrap();
    assert_eq!(&module.metallib[..4], b"MTLB");
    let functions = compile::read_metallib(&module.metallib).unwrap();
    assert_eq!(functions.len(), module.kernels.len());
    for (f, k) in functions.iter().zip(module.kernels.iter()) {
        assert_eq!(f.name, k.name);
        assert_eq!(f.function_type, 2);
        assert_eq!(&f.air[..2], b"BC");
    }
    // corrupting the bitcode must be detected through the function hash
    let mut broken = module.metallib.clone();
    let n = broken.len();
    broken[n - 20] ^= 0xff;
    assert!(compile::read_metallib(&broken).is_err());
}

#[test]
fn fp64_is_rejected() {
    let ptx = r#"
.version 7.0
.target sm_70
.address_size 64
.visible .entry dscale(.param .u64 p, .param .f64 s)
{
    .reg .b64 %rd<3>;
    .reg .f64 %fd<3>;
    ld.param.u64 %rd1, [p];
    ld.param.f64 %fd1, [s];
    cvta.to.global.u64 %rd2, %rd1;
    ld.global.f64 %fd2, [%rd2];
    mul.f64 %fd2, %fd2, %fd1;
    st.global.f64 [%rd2], %fd2;
    ret;
}
"#;
    let err = compile::compile_ptx(ptx, compile::Options::default()).unwrap_err();
    assert!(err.contains("64-bit floating point"), "{err}");
}

#[test]
fn cuda_api_vadd() {
    let mut ptx = clang_ptx().into_bytes();
    ptx.push(0);
    unsafe {
        assert_eq!(cuInit(0), CUDA_SUCCESS);
        let mut module = std::ptr::null_mut();
        assert_eq!(
            cuModuleLoadData(&mut module, ptx.as_ptr().cast()),
            CUDA_SUCCESS
        );
        let mut f = std::ptr::null_mut();
        assert_eq!(
            cuModuleGetFunction(&mut f, module, c"vadd".as_ptr()),
            CUDA_SUCCESS
        );
        let n = 777usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i * i) as f32).collect();
        let (mut da, mut db, mut dc) = (0u64, 0u64, 0u64);
        for p in [&mut da, &mut db, &mut dc] {
            assert_eq!(cuMemAlloc_v2(p, n * 4), CUDA_SUCCESS);
        }
        assert_eq!(cuMemcpyHtoD_v2(da, a.as_ptr().cast(), n * 4), CUDA_SUCCESS);
        assert_eq!(cuMemcpyHtoD_v2(db, b.as_ptr().cast(), n * 4), CUDA_SUCCESS);
        let mut nn = n as i32;
        let mut args = [
            (&mut da as *mut u64).cast(),
            (&mut db as *mut u64).cast(),
            (&mut dc as *mut u64).cast(),
            (&mut nn as *mut i32).cast::<std::ffi::c_void>(),
        ];
        let blocks = n.div_ceil(128) as u32;
        assert_eq!(
            cuLaunchKernel(
                f,
                blocks,
                1,
                1,
                128,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut()
            ),
            CUDA_SUCCESS
        );
        let mut c = vec![0f32; n];
        assert_eq!(
            cuMemcpyDtoH_v2(c.as_mut_ptr().cast(), dc, n * 4),
            CUDA_SUCCESS
        );
        for i in 0..n {
            assert_eq!(c[i], a[i] + b[i], "element {i}");
        }
        for p in [da, db, dc] {
            assert_eq!(cuMemFree_v2(p), CUDA_SUCCESS);
        }
        assert_eq!(cuModuleUnload(module), CUDA_SUCCESS);
    }
}
