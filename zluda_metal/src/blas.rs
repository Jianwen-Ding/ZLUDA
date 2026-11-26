//! cuBLAS on the AGX backend (ZLUDA_METAL_BACKEND=agx): the cublas* entry points of libcublas, exported from this
//! library (a `libcublas.so.12` link to libzluda_metal.so serves applications linked with cuBLAS).
//!
//! The device code is MLIR (blas/gen_kernels.py, the GPU dialect) compiled by the LLVM fork's MLIR: GPU -> AIR
//! (convert-gpu-to-air) -> an AGX object (gpu-module-to-binary with `#air.target<chip = "agx-g14c">`), see build.rs.
//! Its kernels take each pointer and each scalar as one argument, in order ([`agx::Arg`]). Vectors are (pointer,
//! offset, increment) with BLAS's negative increments as an offset; matrices are column-major with explicit row
//! and column strides, so transposes are strides. Reductions are two launches (per-block partials, then one
//! block). Launches are synchronous: the stream of a handle is recorded only.
//!
//! Implemented: handle and modes, Set/GetVector, Set/GetMatrix; S/D axpy, scal, copy, swap, dot, nrm2, asum,
//! amax, amin, gemv, gemm (and strided-batched, batched); Hgemm (and strided-batched), GemmEx /
//! GemmStridedBatchedEx / GemmBatchedEx for f16, bf16, f32 and f64 data with float (double for f64)
//! accumulation. Complex types are not implemented (CUBLAS_STATUS_NOT_SUPPORTED).
#![allow(non_snake_case, dead_code, clippy::missing_safety_doc, clippy::too_many_arguments)]

use crate::agx::{self, Arg};
use std::collections::HashMap;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::{Mutex, OnceLock};

pub type cublasStatus_t = c_int;
pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
pub const CUBLAS_STATUS_NOT_INITIALIZED: cublasStatus_t = 1;
pub const CUBLAS_STATUS_ALLOC_FAILED: cublasStatus_t = 3;
pub const CUBLAS_STATUS_INVALID_VALUE: cublasStatus_t = 7;
pub const CUBLAS_STATUS_ARCH_MISMATCH: cublasStatus_t = 8;
pub const CUBLAS_STATUS_MAPPING_ERROR: cublasStatus_t = 11;
pub const CUBLAS_STATUS_EXECUTION_FAILED: cublasStatus_t = 13;
pub const CUBLAS_STATUS_INTERNAL_ERROR: cublasStatus_t = 14;
pub const CUBLAS_STATUS_NOT_SUPPORTED: cublasStatus_t = 15;

const OP_N: c_int = 0;
const OP_T: c_int = 1;
const OP_C: c_int = 2;
const POINTER_MODE_DEVICE: c_int = 1;

// cudaDataType_t
const CUDA_R_32F: c_int = 0;
const CUDA_R_64F: c_int = 1;
const CUDA_R_16F: c_int = 2;
const CUDA_R_16BF: c_int = 14;

pub struct Handle {
    stream: *mut c_void,
    pointer_mode: c_int,
    math_mode: c_int,
}
pub type cublasHandle_t = *mut Handle;

type R = Result<(), cublasStatus_t>;

fn status(r: R) -> cublasStatus_t {
    match r {
        Ok(()) => CUBLAS_STATUS_SUCCESS,
        Err(e) => e,
    }
}

fn fail(msg: impl std::fmt::Display, st: cublasStatus_t) -> cublasStatus_t {
    if std::env::var_os("ZLUDA_BLAS_TRACE").is_some() || st == CUBLAS_STATUS_EXECUTION_FAILED {
        eprintln!("[zluda blas] {msg}");
    }
    st
}

/// The kernels (blas/zluda_blas.agx.o), loaded once
fn library() -> Result<&'static agx::Library, cublasStatus_t> {
    static LIB: OnceLock<Result<agx::Library, String>> = OnceLock::new();
    LIB.get_or_init(|| agx::Library::from_object(include_bytes!(concat!(env!("OUT_DIR"), "/zluda_blas.agx.o"))))
        .as_ref()
        .map_err(|e| fail(format!("cannot load the BLAS kernels: {e}"), CUBLAS_STATUS_NOT_INITIALIZED))
}

fn kernel(name: &str) -> Result<agx::Kernel, cublasStatus_t> {
    static CACHE: OnceLock<Mutex<HashMap<String, agx::Kernel>>> = OnceLock::new();
    let c = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(k) = c.lock().unwrap().get(name) {
        return Ok(k.clone());
    }
    let k = library()?.kernel(name).map_err(|e| fail(e, CUBLAS_STATUS_INTERNAL_ERROR))?;
    c.lock().unwrap().insert(name.to_string(), k.clone());
    Ok(k)
}

fn launch(name: &str, grid: [u32; 3], block: [u32; 3], args: &[Arg]) -> R {
    if !crate::backend::use_agx() {
        return Err(fail("cuBLAS needs ZLUDA_METAL_BACKEND=agx", CUBLAS_STATUS_NOT_SUPPORTED));
    }
    // a context a kernel faulted in (src/cuda.rs, sticky errors) runs nothing more
    if crate::cuda::current_sticky() != 0 {
        return Err(CUBLAS_STATUS_EXECUTION_FAILED);
    }
    // a fault of the kernel is the context's sticky error, reported by the next CUDA call (as an asynchronous
    // cuBLAS launch); an error of the launch itself fails this call
    match crate::cuda::launch_result(kernel(name)?.launch_args(grid, block, args, 0)) {
        0 => Ok(()),
        e => Err(fail(format!("{name}: launch failed ({e})"), CUBLAS_STATUS_EXECUTION_FAILED)),
    }
}

fn i32v(v: i32) -> Arg<'static> {
    Arg::Value(v.to_le_bytes().to_vec())
}

/// A device buffer for partial results, freed on drop
struct Tmp(u64);
impl Tmp {
    fn new(bytes: u64) -> Result<Tmp, cublasStatus_t> {
        agx::alloc(bytes.max(16)).map(Tmp).map_err(|_| CUBLAS_STATUS_ALLOC_FAILED)
    }
}
impl Drop for Tmp {
    fn drop(&mut self) {
        agx::free(self.0);
    }
}

unsafe fn dev_read<T: Copy>(addr: u64) -> Result<T, cublasStatus_t> {
    let (p, n) = agx::host_pointer(addr).ok_or(CUBLAS_STATUS_INVALID_VALUE)?;
    if (n as usize) < std::mem::size_of::<T>() {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    Ok(std::ptr::read_unaligned(p as *const T))
}

unsafe fn dev_write<T: Copy>(addr: u64, v: T) -> R {
    let (p, n) = agx::host_pointer(addr).ok_or(CUBLAS_STATUS_INVALID_VALUE)?;
    if (n as usize) < std::mem::size_of::<T>() {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    std::ptr::write_unaligned(p as *mut T, v);
    Ok(())
}

/// A scalar argument (alpha, beta) in the handle's pointer mode
unsafe fn scalar<T: Copy>(h: cublasHandle_t, p: *const T) -> Result<T, cublasStatus_t> {
    if p.is_null() {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    if (*h).pointer_mode == POINTER_MODE_DEVICE {
        dev_read(p as u64)
    } else {
        Ok(*p)
    }
}

/// A scalar result in the handle's pointer mode
unsafe fn result<T: Copy>(h: cublasHandle_t, p: *mut T, v: T) -> R {
    if p.is_null() {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    if (*h).pointer_mode == POINTER_MODE_DEVICE {
        dev_write(p as u64, v)
    } else {
        *p = v;
        Ok(())
    }
}

fn check(h: cublasHandle_t) -> R {
    if h.is_null() {
        Err(CUBLAS_STATUS_NOT_INITIALIZED)
    } else {
        Ok(())
    }
}

/// (offset, increment) of a BLAS vector of n elements
fn vec_of(n: i32, inc: i32) -> (i32, i32) {
    if inc < 0 {
        ((n - 1) * -inc, inc)
    } else {
        (0, inc)
    }
}

fn grid1(n: i32, block: u32) -> u32 {
    ((n.max(1) as u32).div_ceil(block)).clamp(1, 1024)
}

/// Element types of the S / D routines
trait Elem: Copy + Default + 'static {
    const P: &'static str;
    fn bytes(self) -> Vec<u8>;
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
}
impl Elem for f32 {
    const P: &'static str = "s";
    fn bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}
impl Elem for f64 {
    const P: &'static str = "d";
    fn bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(v: f64) -> Self {
        v
    }
}

// ------------------------------------------------------------------------------------------------- level 1

unsafe fn axpy<T: Elem>(h: cublasHandle_t, n: c_int, alpha: *const T, x: *const T, incx: c_int, y: *mut T, incy: c_int) -> R {
    check(h)?;
    let a = scalar(h, alpha)?;
    if n <= 0 {
        return Ok(());
    }
    let (xo, xi) = vec_of(n, incx);
    let (yo, yi) = vec_of(n, incy);
    launch(&format!("{}axpy", T::P), [grid1(n, 256), 1, 1], [256, 1, 1], &[
        i32v(n), Arg::Value(a.bytes()), Arg::Addr(x as u64), i32v(xo), i32v(xi), Arg::Addr(y as u64), i32v(yo), i32v(yi),
    ])
}

unsafe fn scal<T: Elem>(h: cublasHandle_t, n: c_int, alpha: *const T, x: *mut T, incx: c_int) -> R {
    check(h)?;
    let a = scalar(h, alpha)?;
    if n <= 0 || incx <= 0 {
        return Ok(());
    }
    launch(&format!("{}scal", T::P), [grid1(n, 256), 1, 1], [256, 1, 1], &[
        i32v(n), Arg::Value(a.bytes()), Arg::Addr(x as u64), i32v(0), i32v(incx),
    ])
}

unsafe fn copy_swap<T: Elem>(h: cublasHandle_t, n: c_int, x: *mut T, incx: c_int, y: *mut T, incy: c_int, swap: bool) -> R {
    check(h)?;
    if n <= 0 {
        return Ok(());
    }
    let (xo, xi) = vec_of(n, incx);
    let (yo, yi) = vec_of(n, incy);
    launch(&format!("{}copy", T::P), [grid1(n, 256), 1, 1], [256, 1, 1], &[
        i32v(n), i32v(swap as i32), Arg::Addr(x as u64), i32v(xo), i32v(xi), Arg::Addr(y as u64), i32v(yo), i32v(yi),
    ])
}

/// op 0 dot, 1 sum of squares, 2 sum of |x|: the sum, by blocks of 32 then one block
unsafe fn reduce<T: Elem>(n: c_int, op: i32, x: u64, incx: c_int, y: u64, incy: c_int) -> Result<T, cublasStatus_t> {
    let sz = std::mem::size_of::<T>() as u64;
    let g = ((n as u32).div_ceil(32)).clamp(1, 256);
    let part = Tmp::new(g as u64 * sz)?;
    let res = Tmp::new(sz)?;
    let (xo, xi) = vec_of(n, incx);
    let (yo, yi) = vec_of(n, incy);
    let k = format!("{}reduce", T::P);
    launch(&k, [g, 1, 1], [32, 1, 1], &[
        i32v(n), i32v(op), Arg::Addr(x), i32v(xo), i32v(xi), Arg::Addr(y), i32v(yo), i32v(yi), Arg::Addr(part.0),
    ])?;
    launch(&k, [1, 1, 1], [32, 1, 1], &[
        i32v(g as i32), i32v(3), Arg::Addr(part.0), i32v(0), i32v(1), Arg::Addr(part.0), i32v(0), i32v(1), Arg::Addr(res.0),
    ])?;
    dev_read::<T>(res.0)
}

unsafe fn dot<T: Elem>(h: cublasHandle_t, n: c_int, x: *const T, incx: c_int, y: *const T, incy: c_int, r: *mut T) -> R {
    check(h)?;
    let v = if n <= 0 { T::default() } else { reduce::<T>(n, 0, x as u64, incx, y as u64, incy)? };
    result(h, r, v)
}

unsafe fn nrm2<T: Elem>(h: cublasHandle_t, n: c_int, x: *const T, incx: c_int, r: *mut T) -> R {
    check(h)?;
    let v = if n <= 0 || incx <= 0 {
        T::default()
    } else {
        T::from_f64(reduce::<T>(n, 1, x as u64, incx, x as u64, incx)?.to_f64().sqrt())
    };
    result(h, r, v)
}

unsafe fn asum<T: Elem>(h: cublasHandle_t, n: c_int, x: *const T, incx: c_int, r: *mut T) -> R {
    check(h)?;
    let v = if n <= 0 || incx <= 0 { T::default() } else { reduce::<T>(n, 2, x as u64, incx, x as u64, incx)? };
    result(h, r, v)
}

/// the 1-based index of the first element of largest (min false) / smallest |x|
unsafe fn amax<T: Elem>(h: cublasHandle_t, n: c_int, x: *const T, incx: c_int, r: *mut c_int, min: bool) -> R {
    check(h)?;
    if n <= 0 || incx <= 0 {
        return result(h, r, 0);
    }
    let sz = std::mem::size_of::<T>() as u64;
    let g = ((n as u32).div_ceil(32)).clamp(1, 256);
    let (pv, pi) = (Tmp::new(g as u64 * sz)?, Tmp::new(g as u64 * sz)?);
    let (rv, ri) = (Tmp::new(sz)?, Tmp::new(sz)?);
    let k = format!("{}amax", T::P);
    launch(&k, [g, 1, 1], [32, 1, 1], &[
        i32v(n), i32v(min as i32), Arg::Addr(x as u64), i32v(0), i32v(incx), Arg::Addr(x as u64), Arg::Addr(pv.0),
        Arg::Addr(pi.0),
    ])?;
    launch(&k, [1, 1, 1], [32, 1, 1], &[
        i32v(g as i32), i32v(2 + min as i32), Arg::Addr(pv.0), i32v(0), i32v(1), Arg::Addr(pi.0), Arg::Addr(rv.0),
        Arg::Addr(ri.0),
    ])?;
    result(h, r, dev_read::<T>(ri.0)?.to_f64() as c_int)
}

// ------------------------------------------------------------------------------------------------- level 2 / 3

fn op_ok(op: c_int) -> bool {
    matches!(op, OP_N | OP_T | OP_C)
}

/// (row stride, column stride) of op(A) for a column-major A with leading dimension ld
fn strides(op: c_int, ld: c_int) -> (i32, i32) {
    if op == OP_N {
        (1, ld)
    } else {
        (ld, 1)
    }
}

unsafe fn gemv<T: Elem>(h: cublasHandle_t, trans: c_int, m: c_int, n: c_int, alpha: *const T, a: *const T, lda: c_int,
                        x: *const T, incx: c_int, beta: *const T, y: *mut T, incy: c_int) -> R {
    check(h)?;
    if !op_ok(trans) || m < 0 || n < 0 || lda < m.max(1) || incx == 0 || incy == 0 {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let (al, be) = (scalar(h, alpha)?, scalar(h, beta)?);
    let (rows, cols) = if trans == OP_N { (m, n) } else { (n, m) };
    if rows == 0 {
        return Ok(());
    }
    let (rs, cs) = strides(trans, lda);
    let (xo, xi) = vec_of(cols, incx);
    let (yo, yi) = vec_of(rows, incy);
    launch(&format!("{}gemv", T::P), [grid1(rows, 64), 1, 1], [64, 1, 1], &[
        i32v(rows), i32v(cols), Arg::Value(al.bytes()), Arg::Addr(a as u64), i32v(0), i32v(rs), i32v(cs),
        Arg::Addr(x as u64), i32v(xo), i32v(xi), Arg::Value(be.bytes()), Arg::Addr(y as u64), i32v(yo), i32v(yi),
    ])
}

/// One GEMM launch (a batch of `batch` matrices `sa` / `sb` / `sc` elements apart): kernel `name`, alpha and beta
/// as the accumulation type's bytes
unsafe fn gemm_launch(name: &str, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: Vec<u8>, a: u64,
                      lda: c_int, sa: i64, b: u64, ldb: c_int, sb: i64, beta: Vec<u8>, c: u64, ldc: c_int, sc: i64,
                      batch: c_int) -> R {
    if !op_ok(ta) || !op_ok(tb) || m < 0 || n < 0 || k < 0 || batch < 0 || ldc < m.max(1)
        || lda < (if ta == OP_N { m } else { k }).max(1) || ldb < (if tb == OP_N { k } else { n }).max(1)
    {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    if m == 0 || n == 0 || batch == 0 {
        return Ok(());
    }
    let big = |v: i64| v > i32::MAX as i64 || v < i32::MIN as i64;
    if big(sa) || big(sb) || big(sc) || big(sc * batch as i64) || big(ldc as i64 * n as i64) {
        return Err(fail("matrices beyond 2^31 elements are not supported", CUBLAS_STATUS_NOT_SUPPORTED));
    }
    let (ars, acs) = strides(ta, lda);
    let (brs, bcs) = strides(tb, ldb);
    // the SIMD-group matrix kernel (float C, m and n multiples of 16, k of 8): alpha I and beta I in a buffer
    let mma = match name {
        "sgemm" => Some("gemm_mma_f32"),
        "gemm_f16_f32" => Some("gemm_mma_f16"),
        "gemm_bf16_f32" => Some("gemm_mma_bf16"),
        _ => None,
    };
    if let Some(mk) = mma.filter(|_| m % 16 == 0 && n % 16 == 0 && k % 8 == 0 && std::env::var_os("ZLUDA_BLAS_NO_MMA").is_none()) {
        let (al, be) = (f32::from_le_bytes(alpha[..4].try_into().unwrap()), f32::from_le_bytes(beta[..4].try_into().unwrap()));
        let mut diag = [0f32; 128];
        for i in 0..8 {
            diag[i * 9] = al;
            diag[64 + i * 9] = be;
        }
        let d = Tmp::new(512)?;
        let (p, _) = agx::host_pointer(d.0).ok_or(CUBLAS_STATUS_INTERNAL_ERROR)?;
        std::ptr::copy_nonoverlapping(diag.as_ptr() as *const u8, p, 512);
        // 32 x 32 tiles (twice the reuse of each loaded block) where they divide C and still give at least 256
        // SIMD groups (simulated M2 Max: 256^3 17.3 us with 16 x 16 tiles, 22.3 us with 32 x 32; 512^3 60.1 us /
        // 46.4 us), else 16 x 16; ZLUDA_BLAS_MMA_TILE=16 / 32 overrides
        let fits32 = m % 32 == 0 && n % 32 == 0;
        let tile = match std::env::var("ZLUDA_BLAS_MMA_TILE").as_deref() {
            Ok("16") => 16,
            Ok("32") if fits32 => 32,
            Ok(_) => 16,
            Err(_) if fits32 && (m as i64 / 32) * (n as i64 / 32) * batch as i64 >= 256 => 32,
            Err(_) => 16,
        };
        let mk = if tile == 32 { mk.replace("gemm_mma_", "gemm_mma32_") } else { mk.to_string() };
        return launch(&mk, [(m / tile) as u32, (n / tile) as u32, batch as u32], [32, 1, 1], &[
            i32v(m), i32v(n), i32v(k), Arg::Addr(d.0), Arg::Addr(a), i32v(0), i32v(ars), i32v(acs), i32v(sa as i32),
            Arg::Addr(b), i32v(0), i32v(brs), i32v(bcs), i32v(sb as i32), i32v((be != 0.0) as i32), Arg::Addr(c),
            i32v(0), i32v(ldc), i32v(sc as i32),
        ]);
    }
    let grid = [(m as u32).div_ceil(64), (n as u32).div_ceil(64), batch as u32];
    launch(name, grid, [16, 16, 1], &[
        i32v(m), i32v(n), i32v(k), Arg::Value(alpha), Arg::Addr(a), i32v(0), i32v(ars), i32v(acs), i32v(sa as i32),
        Arg::Addr(b), i32v(0), i32v(brs), i32v(bcs), i32v(sb as i32), Arg::Value(beta), Arg::Addr(c), i32v(0),
        i32v(ldc), i32v(sc as i32),
    ])
}

unsafe fn gemm<T: Elem>(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const T,
                        a: *const T, lda: c_int, sa: i64, b: *const T, ldb: c_int, sb: i64, beta: *const T, c: *mut T,
                        ldc: c_int, sc: i64, batch: c_int) -> R {
    check(h)?;
    let (al, be) = (scalar(h, alpha)?, scalar(h, beta)?);
    gemm_launch(&format!("{}gemm", T::P), ta, tb, m, n, k, al.bytes(), a as u64, lda, sa, b as u64, ldb, sb,
                be.bytes(), c as u64, ldc, sc, batch)
}

/// The pointer arrays of a batched GEMM (device memory)
unsafe fn ptrs(p: *const *const c_void, i: c_int) -> Result<u64, cublasStatus_t> {
    dev_read::<u64>(p as u64 + 8 * i as u64)
}

// ------------------------------------------------------------------------------------------------- mixed precision

fn half_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 31) as i32;
    let m = (h & 1023) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            // subnormal: m * 2^-24
            return f32::from_bits(s) + (if s != 0 { -1.0 } else { 1.0 }) * m as f32 * 2f32.powi(-24);
        }
    } else if e == 31 {
        s | 0x7F80_0000 | (m << 13)
    } else {
        s | (((e - 15 + 127) as u32) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// kernel of a GemmEx: (name, alpha / beta bytes) from the data types (A and B alike) and the compute type, and
/// alpha / beta (of the compute type: half for CUBLAS_COMPUTE_16F)
unsafe fn gemm_ex_kernel(h: cublasHandle_t, at: c_int, bt: c_int, ct: c_int, compute: c_int, alpha: *const c_void,
                         beta: *const c_void) -> Result<(String, Vec<u8>, Vec<u8>), cublasStatus_t> {
    if at != bt {
        return Err(fail(format!("GemmEx with A type {at} and B type {bt}"), CUBLAS_STATUS_NOT_SUPPORTED));
    }
    // cublasComputeType_t: 16F 64 / 65, 32F 68 / 69 and the fast 74 / 75 / 77, 64F 70 / 71; the legacy API passes
    // a cudaDataType_t
    let half_scalars = matches!(compute, 64 | 65) || (compute == CUDA_R_16F && at == CUDA_R_16F);
    let f32s = |p: *const c_void| -> Result<Vec<u8>, cublasStatus_t> {
        Ok(if half_scalars {
            half_to_f32(scalar(h, p as *const u16)?).to_le_bytes().to_vec()
        } else {
            scalar(h, p as *const f32)?.to_le_bytes().to_vec()
        })
    };
    let name = match (at, ct) {
        (CUDA_R_32F, CUDA_R_32F) => "sgemm".to_string(),
        (CUDA_R_64F, CUDA_R_64F) => {
            let (a, b) = (scalar(h, alpha as *const f64)?, scalar(h, beta as *const f64)?);
            return Ok(("dgemm".into(), a.to_le_bytes().to_vec(), b.to_le_bytes().to_vec()));
        }
        (CUDA_R_16F, CUDA_R_16F) => "gemm_f16_f16".into(),
        (CUDA_R_16F, CUDA_R_32F) => "gemm_f16_f32".into(),
        (CUDA_R_16BF, CUDA_R_16BF) => "gemm_bf16_bf16".into(),
        (CUDA_R_16BF, CUDA_R_32F) => "gemm_bf16_f32".into(),
        _ => return Err(fail(format!("GemmEx types A {at} C {ct}"), CUBLAS_STATUS_NOT_SUPPORTED)),
    };
    Ok((name, f32s(alpha)?, f32s(beta)?))
}

// ================================================================================================= entry points

macro_rules! entry {
    ($(fn $name:ident($($a:ident: $t:ty),*) $body:block)*) => {$(
        #[no_mangle]
        pub unsafe extern "system" fn $name($($a: $t),*) -> cublasStatus_t {
            status((|| -> R { $body })())
        }
    )*};
}

entry! {
    fn cublasCreate_v2(handle: *mut cublasHandle_t) {
        if handle.is_null() {
            return Err(CUBLAS_STATUS_INVALID_VALUE);
        }
        if !crate::backend::use_agx() {
            return Err(fail("cuBLAS needs ZLUDA_METAL_BACKEND=agx", CUBLAS_STATUS_NOT_SUPPORTED));
        }
        library()?;
        *handle = Box::into_raw(Box::new(Handle { stream: std::ptr::null_mut(), pointer_mode: 0, math_mode: 0 }));
        Ok(())
    }
    fn cublasDestroy_v2(handle: cublasHandle_t) {
        check(handle)?;
        drop(Box::from_raw(handle));
        Ok(())
    }
    fn cublasGetVersion_v2(handle: cublasHandle_t, version: *mut c_int) {
        check(handle)?;
        *version = 120400;
        Ok(())
    }
    fn cublasSetStream_v2(handle: cublasHandle_t, stream: *mut c_void) {
        check(handle)?;
        (*handle).stream = stream;
        Ok(())
    }
    fn cublasGetStream_v2(handle: cublasHandle_t, stream: *mut *mut c_void) {
        check(handle)?;
        *stream = (*handle).stream;
        Ok(())
    }
    fn cublasSetPointerMode_v2(handle: cublasHandle_t, mode: c_int) {
        check(handle)?;
        (*handle).pointer_mode = mode;
        Ok(())
    }
    fn cublasGetPointerMode_v2(handle: cublasHandle_t, mode: *mut c_int) {
        check(handle)?;
        *mode = (*handle).pointer_mode;
        Ok(())
    }
    fn cublasSetMathMode(handle: cublasHandle_t, mode: c_int) {
        check(handle)?;
        (*handle).math_mode = mode;
        Ok(())
    }
    fn cublasGetMathMode(handle: cublasHandle_t, mode: *mut c_int) {
        check(handle)?;
        *mode = (*handle).math_mode;
        Ok(())
    }
    fn cublasSetWorkspace_v2(handle: cublasHandle_t, _ws: *mut c_void, _bytes: usize) {
        check(handle)
    }
    fn cublasSetVector(n: c_int, elem: c_int, x: *const c_void, incx: c_int, y: *mut c_void, incy: c_int) {
        strided_copy(n, 1, elem, x as *const u8, incx as i64, 0, false, y as *mut u8, incy as i64, 0, true)
    }
    fn cublasGetVector(n: c_int, elem: c_int, x: *const c_void, incx: c_int, y: *mut c_void, incy: c_int) {
        strided_copy(n, 1, elem, x as *const u8, incx as i64, 0, true, y as *mut u8, incy as i64, 0, false)
    }
    fn cublasSetMatrix(rows: c_int, cols: c_int, elem: c_int, a: *const c_void, lda: c_int, b: *mut c_void, ldb: c_int) {
        strided_copy(rows, cols, elem, a as *const u8, 1, lda as i64, false, b as *mut u8, 1, ldb as i64, true)
    }
    fn cublasGetMatrix(rows: c_int, cols: c_int, elem: c_int, a: *const c_void, lda: c_int, b: *mut c_void, ldb: c_int) {
        strided_copy(rows, cols, elem, a as *const u8, 1, lda as i64, true, b as *mut u8, 1, ldb as i64, false)
    }

    fn cublasSaxpy_v2(h: cublasHandle_t, n: c_int, alpha: *const f32, x: *const f32, incx: c_int, y: *mut f32, incy: c_int) { axpy(h, n, alpha, x, incx, y, incy) }
    fn cublasDaxpy_v2(h: cublasHandle_t, n: c_int, alpha: *const f64, x: *const f64, incx: c_int, y: *mut f64, incy: c_int) { axpy(h, n, alpha, x, incx, y, incy) }
    fn cublasSscal_v2(h: cublasHandle_t, n: c_int, alpha: *const f32, x: *mut f32, incx: c_int) { scal(h, n, alpha, x, incx) }
    fn cublasDscal_v2(h: cublasHandle_t, n: c_int, alpha: *const f64, x: *mut f64, incx: c_int) { scal(h, n, alpha, x, incx) }
    fn cublasScopy_v2(h: cublasHandle_t, n: c_int, x: *const f32, incx: c_int, y: *mut f32, incy: c_int) { copy_swap(h, n, x as *mut f32, incx, y, incy, false) }
    fn cublasDcopy_v2(h: cublasHandle_t, n: c_int, x: *const f64, incx: c_int, y: *mut f64, incy: c_int) { copy_swap(h, n, x as *mut f64, incx, y, incy, false) }
    fn cublasSswap_v2(h: cublasHandle_t, n: c_int, x: *mut f32, incx: c_int, y: *mut f32, incy: c_int) { copy_swap(h, n, x, incx, y, incy, true) }
    fn cublasDswap_v2(h: cublasHandle_t, n: c_int, x: *mut f64, incx: c_int, y: *mut f64, incy: c_int) { copy_swap(h, n, x, incx, y, incy, true) }
    fn cublasSdot_v2(h: cublasHandle_t, n: c_int, x: *const f32, incx: c_int, y: *const f32, incy: c_int, r: *mut f32) { dot(h, n, x, incx, y, incy, r) }
    fn cublasDdot_v2(h: cublasHandle_t, n: c_int, x: *const f64, incx: c_int, y: *const f64, incy: c_int, r: *mut f64) { dot(h, n, x, incx, y, incy, r) }
    fn cublasSnrm2_v2(h: cublasHandle_t, n: c_int, x: *const f32, incx: c_int, r: *mut f32) { nrm2(h, n, x, incx, r) }
    fn cublasDnrm2_v2(h: cublasHandle_t, n: c_int, x: *const f64, incx: c_int, r: *mut f64) { nrm2(h, n, x, incx, r) }
    fn cublasSasum_v2(h: cublasHandle_t, n: c_int, x: *const f32, incx: c_int, r: *mut f32) { asum(h, n, x, incx, r) }
    fn cublasDasum_v2(h: cublasHandle_t, n: c_int, x: *const f64, incx: c_int, r: *mut f64) { asum(h, n, x, incx, r) }
    fn cublasIsamax_v2(h: cublasHandle_t, n: c_int, x: *const f32, incx: c_int, r: *mut c_int) { amax(h, n, x, incx, r, false) }
    fn cublasIdamax_v2(h: cublasHandle_t, n: c_int, x: *const f64, incx: c_int, r: *mut c_int) { amax(h, n, x, incx, r, false) }
    fn cublasIsamin_v2(h: cublasHandle_t, n: c_int, x: *const f32, incx: c_int, r: *mut c_int) { amax(h, n, x, incx, r, true) }
    fn cublasIdamin_v2(h: cublasHandle_t, n: c_int, x: *const f64, incx: c_int, r: *mut c_int) { amax(h, n, x, incx, r, true) }

    fn cublasSgemv_v2(h: cublasHandle_t, t: c_int, m: c_int, n: c_int, alpha: *const f32, a: *const f32, lda: c_int, x: *const f32, incx: c_int, beta: *const f32, y: *mut f32, incy: c_int) {
        gemv(h, t, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }
    fn cublasDgemv_v2(h: cublasHandle_t, t: c_int, m: c_int, n: c_int, alpha: *const f64, a: *const f64, lda: c_int, x: *const f64, incx: c_int, beta: *const f64, y: *mut f64, incy: c_int) {
        gemv(h, t, m, n, alpha, a, lda, x, incx, beta, y, incy)
    }
    fn cublasSgemm_v2(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const f32, a: *const f32, lda: c_int, b: *const f32, ldb: c_int, beta: *const f32, c: *mut f32, ldc: c_int) {
        gemm(h, ta, tb, m, n, k, alpha, a, lda, 0, b, ldb, 0, beta, c, ldc, 0, 1)
    }
    fn cublasDgemm_v2(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const f64, a: *const f64, lda: c_int, b: *const f64, ldb: c_int, beta: *const f64, c: *mut f64, ldc: c_int) {
        gemm(h, ta, tb, m, n, k, alpha, a, lda, 0, b, ldb, 0, beta, c, ldc, 0, 1)
    }
    fn cublasSgemmStridedBatched(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const f32, a: *const f32, lda: c_int, sa: i64, b: *const f32, ldb: c_int, sb: i64, beta: *const f32, c: *mut f32, ldc: c_int, sc: i64, batch: c_int) {
        gemm(h, ta, tb, m, n, k, alpha, a, lda, sa, b, ldb, sb, beta, c, ldc, sc, batch)
    }
    fn cublasDgemmStridedBatched(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const f64, a: *const f64, lda: c_int, sa: i64, b: *const f64, ldb: c_int, sb: i64, beta: *const f64, c: *mut f64, ldc: c_int, sc: i64, batch: c_int) {
        gemm(h, ta, tb, m, n, k, alpha, a, lda, sa, b, ldb, sb, beta, c, ldc, sc, batch)
    }
    fn cublasSgemmBatched(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const f32, a: *const *const f32, lda: c_int, b: *const *const f32, ldb: c_int, beta: *const f32, c: *const *mut f32, ldc: c_int, batch: c_int) {
        for i in 0..batch.max(0) {
            gemm(h, ta, tb, m, n, k, alpha, ptrs(a as _, i)? as *const f32, lda, 0, ptrs(b as _, i)? as *const f32, ldb, 0, beta, ptrs(c as _, i)? as *mut f32, ldc, 0, 1)?;
        }
        Ok(())
    }
    fn cublasDgemmBatched(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const f64, a: *const *const f64, lda: c_int, b: *const *const f64, ldb: c_int, beta: *const f64, c: *const *mut f64, ldc: c_int, batch: c_int) {
        for i in 0..batch.max(0) {
            gemm(h, ta, tb, m, n, k, alpha, ptrs(a as _, i)? as *const f64, lda, 0, ptrs(b as _, i)? as *const f64, ldb, 0, beta, ptrs(c as _, i)? as *mut f64, ldc, 0, 1)?;
        }
        Ok(())
    }
    fn cublasHgemm(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const u16, a: *const u16, lda: c_int, b: *const u16, ldb: c_int, beta: *const u16, c: *mut u16, ldc: c_int) {
        check(h)?;
        let (al, be) = (half_to_f32(scalar(h, alpha)?), half_to_f32(scalar(h, beta)?));
        gemm_launch("gemm_f16_f16", ta, tb, m, n, k, al.to_le_bytes().to_vec(), a as u64, lda, 0, b as u64, ldb, 0, be.to_le_bytes().to_vec(), c as u64, ldc, 0, 1)
    }
    fn cublasHgemmStridedBatched(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const u16, a: *const u16, lda: c_int, sa: i64, b: *const u16, ldb: c_int, sb: i64, beta: *const u16, c: *mut u16, ldc: c_int, sc: i64, batch: c_int) {
        check(h)?;
        let (al, be) = (half_to_f32(scalar(h, alpha)?), half_to_f32(scalar(h, beta)?));
        gemm_launch("gemm_f16_f16", ta, tb, m, n, k, al.to_le_bytes().to_vec(), a as u64, lda, sa, b as u64, ldb, sb, be.to_le_bytes().to_vec(), c as u64, ldc, sc, batch)
    }
    fn cublasGemmEx(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const c_void, a: *const c_void, at: c_int, lda: c_int, b: *const c_void, bt: c_int, ldb: c_int, beta: *const c_void, c: *mut c_void, ct: c_int, ldc: c_int, compute: c_int, _algo: c_int) {
        check(h)?;
        let (name, al, be) = gemm_ex_kernel(h, at, bt, ct, compute, alpha, beta)?;
        gemm_launch(&name, ta, tb, m, n, k, al, a as u64, lda, 0, b as u64, ldb, 0, be, c as u64, ldc, 0, 1)
    }
    fn cublasGemmStridedBatchedEx(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const c_void, a: *const c_void, at: c_int, lda: c_int, sa: i64, b: *const c_void, bt: c_int, ldb: c_int, sb: i64, beta: *const c_void, c: *mut c_void, ct: c_int, ldc: c_int, sc: i64, batch: c_int, compute: c_int, _algo: c_int) {
        check(h)?;
        let (name, al, be) = gemm_ex_kernel(h, at, bt, ct, compute, alpha, beta)?;
        gemm_launch(&name, ta, tb, m, n, k, al, a as u64, lda, sa, b as u64, ldb, sb, be, c as u64, ldc, sc, batch)
    }
    fn cublasGemmBatchedEx(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const c_void, a: *const *const c_void, at: c_int, lda: c_int, b: *const *const c_void, bt: c_int, ldb: c_int, beta: *const c_void, c: *const *mut c_void, ct: c_int, ldc: c_int, batch: c_int, compute: c_int, _algo: c_int) {
        check(h)?;
        let (name, al, be) = gemm_ex_kernel(h, at, bt, ct, compute, alpha, beta)?;
        for i in 0..batch.max(0) {
            gemm_launch(&name, ta, tb, m, n, k, al.clone(), ptrs(a as _, i)?, lda, 0, ptrs(b as _, i)?, ldb, 0, be.clone(), ptrs(c as _, i)?, ldc, 0, 1)?;
        }
        Ok(())
    }
}

/// rows x cols elements of `elem` bytes, element (i, j) at i * inc + j * ld (in elements) on both sides; the
/// device side (src_dev / dst_dev) through its host mapping
unsafe fn strided_copy(rows: c_int, cols: c_int, elem: c_int, src: *const u8, sinc: i64, sld: i64, src_dev: bool,
                       dst: *mut u8, dinc: i64, dld: i64, dst_dev: bool) -> R {
    if rows < 0 || cols < 0 || elem <= 0 || sinc <= 0 || dinc <= 0 || (cols > 1 && (sld < rows as i64 || dld < rows as i64)) {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let e = elem as usize;
    let map = |p: *const u8, dev: bool| -> Result<*mut u8, cublasStatus_t> {
        if !dev {
            return Ok(p as *mut u8);
        }
        agx::host_pointer(p as u64).map(|(h, _)| h).ok_or(CUBLAS_STATUS_MAPPING_ERROR)
    };
    let (s, d) = (map(src, src_dev)?, map(dst, dst_dev)?);
    for j in 0..cols as i64 {
        for i in 0..rows as i64 {
            let so = ((i * sinc + j * sld) as usize) * e;
            let doff = ((i * dinc + j * dld) as usize) * e;
            std::ptr::copy_nonoverlapping(s.add(so), d.add(doff), e);
        }
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "system" fn cublasGetStatusName(s: cublasStatus_t) -> *const c_char {
    (match s {
        0 => "CUBLAS_STATUS_SUCCESS\0",
        1 => "CUBLAS_STATUS_NOT_INITIALIZED\0",
        3 => "CUBLAS_STATUS_ALLOC_FAILED\0",
        7 => "CUBLAS_STATUS_INVALID_VALUE\0",
        8 => "CUBLAS_STATUS_ARCH_MISMATCH\0",
        11 => "CUBLAS_STATUS_MAPPING_ERROR\0",
        13 => "CUBLAS_STATUS_EXECUTION_FAILED\0",
        14 => "CUBLAS_STATUS_INTERNAL_ERROR\0",
        15 => "CUBLAS_STATUS_NOT_SUPPORTED\0",
        _ => "CUBLAS_STATUS_UNKNOWN\0",
    })
    .as_ptr() as *const c_char
}

#[no_mangle]
pub unsafe extern "system" fn cublasGetStatusString(s: cublasStatus_t) -> *const c_char {
    cublasGetStatusName(s)
}
