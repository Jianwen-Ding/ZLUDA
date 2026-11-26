use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let clang = env::var("ZLUDA_METAL_CLANG").unwrap_or_else(|_| "clang++".to_string());
    let llvm_config =
        env::var("ZLUDA_SYSTEM_LLVM_CONFIG").unwrap_or_else(|_| "llvm-config".to_string());
    println!("cargo:rerun-if-env-changed=ZLUDA_METAL_CLANG");
    println!("cargo:rerun-if-env-changed=ZLUDA_SYSTEM_LLVM_CONFIG");

    // 1. The Metal flavour of the PTX helper library, compiled with upstream clang
    let impl_src = "../ptx/lib/zluda_ptx_metal_impl.cpp";
    println!("cargo:rerun-if-changed={impl_src}");
    let impl_bc = out.join("zluda_ptx_metal_impl.bc");
    let status = Command::new(&clang)
        .args([
            "-std=c++17",
            "--target=amdgcn-amd-amdhsa",
            "-nogpulib",
            "-O2",
            "-ffreestanding",
            "-fno-exceptions",
            "-fno-rtti",
            "-emit-llvm",
            "-c",
            impl_src,
            "-o",
        ])
        .arg(&impl_bc)
        .status()
        .expect("failed to run clang to build zluda_ptx_metal_impl.bc");
    assert!(status.success(), "clang failed on {impl_src}");

    // 2. AIR lowering + metallib container (C++, LLVM)
    let cxxflags = Command::new(&llvm_config)
        .arg("--cxxflags")
        .output()
        .expect("llvm-config --cxxflags failed");
    let cxxflags = String::from_utf8(cxxflags.stdout).unwrap();
    let mut cc = cc::Build::new();
    cc.cpp(true).compiler(&clang);
    for flag in cxxflags.split_whitespace() {
        if !flag.starts_with("-std=") && flag != "-fno-exceptions" {
            cc.flag(flag);
        }
    }
    cc.flag("-std=c++17")
        .flag("-fexceptions")
        .flag("-Wno-unused-parameter");
    for f in ["cpp/air_lower.cpp", "cpp/metallib.cpp"] {
        cc.file(f);
        println!("cargo:rerun-if-changed={f}");
    }
    println!("cargo:rerun-if-changed=cpp/air_lower.h");
    println!("cargo:rerun-if-changed=cpp/metallib.h");
    println!("cargo:rerun-if-changed=cpp/sha256.h");
    cc.compile("zluda_metal_compiler");

    // 3. metal-cpp runtime backend (+ the Linux test shim when not on Apple)
    let metal_cpp = find_metal_cpp();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let apple = target_os == "macos" || target_os == "ios";
    let mut backend = cc::Build::new();
    backend
        .cpp(true)
        .compiler(&clang)
        .flag("-std=c++17")
        .flag("-fblocks");
    backend
        .flag("-Wno-unused-parameter")
        .flag("-Wno-ignored-attributes");
    if !apple {
        backend.include("linux_shim/include");
    }
    backend.include(&metal_cpp).file("cpp/metal_backend.cpp");
    println!("cargo:rerun-if-changed=cpp/metal_backend.cpp");
    println!("cargo:rerun-if-changed=cpp/metal_backend.h");
    if !apple {
        for f in [
            "linux_shim/objc_runtime.cpp",
            "linux_shim/foundation.cpp",
            "linux_shim/swmetal.cpp",
        ] {
            backend.file(f);
            println!("cargo:rerun-if-changed={f}");
        }
        println!("cargo:rerun-if-changed=linux_shim");
    }
    backend.compile("zluda_metal_backend");
    if apple {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=objc");
    } else {
        // software device executor (uses LLVM's ORC JIT)
        let mut exec = cc::Build::new();
        exec.cpp(true).compiler(&clang);
        for flag in cxxflags.split_whitespace() {
            if !flag.starts_with("-std=") && flag != "-fno-exceptions" {
                exec.flag(flag);
            }
        }
        exec.flag("-std=c++17")
            .flag("-fexceptions")
            .flag("-Wno-unused-parameter");
        exec.include("linux_shim/include")
            .file("linux_shim/sw_exec.cpp");
        exec.compile("zluda_metal_swexec");
    }
}

fn find_metal_cpp() -> PathBuf {
    println!("cargo:rerun-if-env-changed=METAL_CPP_DIR");
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let candidates = [
        env::var("METAL_CPP_DIR").ok().map(PathBuf::from),
        Some(manifest.join("../ext/metal-cpp")),
        Some(manifest.join("../../metal-cpp")),
    ];
    for c in candidates.into_iter().flatten() {
        if c.join("Metal/Metal.hpp").exists() {
            return c;
        }
    }
    panic!(
        "metal-cpp not found: set METAL_CPP_DIR to a checkout of metal-cpp \
         (https://developer.apple.com/metal/cpp/)"
    );
}
