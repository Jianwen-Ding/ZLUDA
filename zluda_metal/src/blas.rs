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
    atomics_mode: c_int,
    sm_count_target: c_int,
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

/// op 0 dot, 1 sum of squares, 2 sum of |x|: the sum, in one launch (blocks of 256 threads; the last block to
/// finish adds the blocks' partial sums)
unsafe fn reduce<T: Elem>(n: c_int, op: i32, x: u64, incx: c_int, y: u64, incy: c_int) -> Result<T, cublasStatus_t> {
    let sz = std::mem::size_of::<T>() as u64;
    let g = ((n as u32).div_ceil(256)).clamp(1, reduce_blocks());
    let part = Tmp::new(g as u64 * sz)?;
    let res = Tmp::new(sz)?;
    let cnt = Tmp::new(16)?;
    dev_write::<u32>(cnt.0, 0)?;
    let (xo, xi) = vec_of(n, incx);
    let (yo, yi) = vec_of(n, incy);
    launch(&format!("{}sum", T::P), [g, 1, 1], [256, 1, 1], &[
        i32v(n), i32v(op), Arg::Addr(x), i32v(xo), i32v(xi), Arg::Addr(y), i32v(yo), i32v(yi), Arg::Addr(part.0),
        Arg::Addr(cnt.0), Arg::Addr(res.0),
    ])?;
    dev_read::<T>(res.0)
}

fn reduce_blocks() -> u32 {
    std::env::var("ZLUDA_BLAS_REDUCE_BLOCKS").ok().and_then(|s| s.parse().ok()).unwrap_or(128)
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

/// the 1-based index of the first element of largest (min false) / smallest |x|, in one launch
unsafe fn amax<T: Elem>(h: cublasHandle_t, n: c_int, x: *const T, incx: c_int, r: *mut c_int, min: bool) -> R {
    check(h)?;
    if n <= 0 || incx <= 0 {
        return result(h, r, 0);
    }
    let sz = std::mem::size_of::<T>() as u64;
    let g = ((n as u32).div_ceil(256)).clamp(1, reduce_blocks());
    let (pv, pi) = (Tmp::new(g as u64 * sz)?, Tmp::new(g as u64 * 4)?);
    let (cnt, ri) = (Tmp::new(16)?, Tmp::new(16)?);
    dev_write::<u32>(cnt.0, 0)?;
    launch(&format!("{}amax1", T::P), [g, 1, 1], [256, 1, 1], &[
        i32v(n), i32v(min as i32), Arg::Addr(x as u64), i32v(0), i32v(incx), Arg::Addr(pv.0), Arg::Addr(pi.0),
        Arg::Addr(cnt.0), Arg::Addr(ri.0),
    ])?;
    result(h, r, dev_read::<c_int>(ri.0)?)
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
    gemv_sb(h, trans, m, n, alpha, a, lda, 0, x, incx, 0, beta, y, incy, 0, 1)
}

/// A GEMM (a batch of `batch` matrices `sa` / `sb` / `sc` elements apart): kernel `name`, alpha and beta as the
/// accumulation type's bytes. Float outputs run on the SIMD-group matrix kernel for the part of C that is a
/// multiple of its tile (and k of 8), the register-tiled kernel on the rest (the remaining k, the bottom rows and
/// the right columns); f16 / bf16 outputs through an f32 copy of C.
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
    let g = Gemm { ta, tb, a, lda, sa, b, ldb, sb, c, ldc, sc, batch };
    // the SIMD-group matrix kernel: (kernel, operand type, f16 / bf16 C through f32)
    let mma = match name {
        "sgemm" => Some(("gemm_mma_f32", "f32", None)),
        "gemm_f16_f32" => Some(("gemm_mma_f16", "f16", None)),
        "gemm_bf16_f32" => Some(("gemm_mma_bf16", "bf16", None)),
        "gemm_f16_f16" => Some(("gemm_mma_f16", "f16", Some("f16"))),
        "gemm_bf16_bf16" => Some(("gemm_mma_bf16", "bf16", Some("bf16"))),
        _ => None,
    };
    let aligned = m % 16 == 0 && n % 16 == 0 && k % 8 == 0;
    // padded copies (and f32 staging of a half C) pay for themselves beyond small sizes
    let worth = (aligned && mma.is_some_and(|x| x.2.is_none())) || m as i64 * n as i64 * k as i64 >= 1 << 17;
    if let Some((mk, at, half)) = mma.filter(|_| m >= 16 && n >= 16 && k >= 8 && worth && std::env::var_os("ZLUDA_BLAS_NO_MMA").is_none()) {
        return gemm_mma(mk, at, half, &g, m, n, k, &alpha, &beta);
    }
    gemm_generic(name, &g, m, n, k, &alpha, &beta, 0, 0, 0)
}

/// The operands of a GEMM (pointers, leading dimensions, batch strides)
#[derive(Clone, Copy)]
struct Gemm {
    ta: c_int,
    tb: c_int,
    a: u64,
    lda: c_int,
    sa: i64,
    b: u64,
    ldb: c_int,
    sb: i64,
    c: u64,
    ldc: c_int,
    sc: i64,
    batch: c_int,
}

fn round_up(v: i32, t: i32) -> i32 {
    (v + t - 1) / t * t
}

/// The SIMD-group matrix kernel (air.simdgroup_matrix_*: one SIMD group per 16 x 16 or 32 x 32 tile of an f32 C,
/// k in steps of 8; alpha and beta as products with alpha I / beta I). Shapes it does not tile run on copies padded
/// with zeros (op(A) to mp x kp, op(B) to kp x np, C to mp x np in f32, also the f32 copy of an f16 / bf16 C).
unsafe fn gemm_mma(mk: &str, at: &str, half: Option<&str>, g: &Gemm, m: i32, n: i32, k: i32, alpha: &[u8], beta: &[u8]) -> R {
    let batch = g.batch;
    // 32 x 32 tiles (twice the reuse of each loaded block) where they still give at least 256 SIMD groups
    // (simulated M2 Max: 256^3 17.3 us with 16 x 16 tiles, 22.3 us with 32 x 32; 512^3 60.1 us / 46.4 us), else
    // 16 x 16; ZLUDA_BLAS_MMA_TILE=16 / 32 overrides
    let t32 = (round_up(m, 32) as i64 / 32) * (round_up(n, 32) as i64 / 32) * batch as i64 >= 256;
    let tile = match std::env::var("ZLUDA_BLAS_MMA_TILE").as_deref() {
        Ok("32") => 32,
        Ok(_) => 16,
        Err(_) if t32 => 32,
        Err(_) => 16,
    };
    let (mp, np, kp) = (round_up(m, tile), round_up(n, tile), round_up(k, 8));
    let esz = if at == "f32" { 4 } else { 2 };
    let (ars, acs) = strides(g.ta, g.lda);
    let (brs, bcs) = strides(g.tb, g.ldb);
    // op(A), op(B): as they are, or padded copies (column-major, batch after batch)
    let operand = |p: u64, rows: i32, cols: i32, rp: i32, cp: i32, rs: i32, cs: i32, st: i64| -> Result<(u64, i32, i32, i64, Option<Tmp>), cublasStatus_t> {
        if rows == rp && cols == cp {
            return Ok((p, rs, cs, st, None));
        }
        let per = elems(rp, cp)?;
        elems(per, batch)?;
        let t = Tmp::new(per as u64 * batch as u64 * esz)?;
        pad_launch(&format!("pad_{at}"), rp, cp, rows, cols, p, rs, cs, st as i32, t.0, rp, per, batch)?;
        Ok((t.0, 1, rp, per as i64, Some(t)))
    };
    let (pa, pars, pacs, psa, _ka) = operand(g.a, m, k, mp, kp, ars, acs, g.sa)?;
    let (pb, pbrs, pbcs, psb, _kb) = operand(g.b, k, n, kp, np, brs, bcs, g.sb)?;
    let (al, be) = (f32::from_le_bytes(alpha[..4].try_into().unwrap()), f32::from_le_bytes(beta[..4].try_into().unwrap()));
    // C: in place, or an f32 copy (mp x np; its padding is never read back)
    let staged = half.is_some() || mp != m || np != n;
    let ct = if staged {
        let per = elems(mp, np)?;
        elems(per, batch)?;
        let t = Tmp::new(per as u64 * batch as u64 * 4)?;
        if be != 0.0 {
            match half {
                Some(h) => cvt_launch(&format!("cvt_{h}_f32"), m, n, g.c, g.ldc, g.sc as i32, t.0, mp, per, batch)?,
                None => pad_launch("pad_f32", m, n, m, n, g.c, 1, g.ldc, g.sc as i32, t.0, mp, per, batch)?,
            }
        }
        Some((t, per))
    } else {
        None
    };
    let (pc, pldc, psc) = match &ct {
        Some((t, per)) => (t.0, mp, *per as i64),
        None => (g.c, g.ldc, g.sc),
    };
    let mut diag = [0f32; 128];
    for i in 0..8 {
        diag[i * 9] = al;
        diag[64 + i * 9] = be;
    }
    let d = Tmp::new(512)?;
    let (p, _) = agx::host_pointer(d.0).ok_or(CUBLAS_STATUS_INTERNAL_ERROR)?;
    std::ptr::copy_nonoverlapping(diag.as_ptr() as *const u8, p, 512);
    let mk = if tile == 32 { mk.replace("gemm_mma_", "gemm_mma32_") } else { mk.to_string() };
    launch(&mk, [(mp / tile) as u32, (np / tile) as u32, batch as u32], [32, 1, 1], &[
        i32v(mp), i32v(np), i32v(kp), Arg::Addr(d.0), Arg::Addr(pa), i32v(0), i32v(pars), i32v(pacs), i32v(psa as i32),
        Arg::Addr(pb), i32v(0), i32v(pbrs), i32v(pbcs), i32v(psb as i32), i32v((be != 0.0) as i32), Arg::Addr(pc),
        i32v(0), i32v(pldc), i32v(psc as i32),
    ])?;
    if let Some((t, per)) = &ct {
        match half {
            Some(h) => cvt_launch(&format!("cvt_f32_{h}"), m, n, t.0, mp, *per, g.c, g.ldc, g.sc as i32, batch)?,
            None => pad_launch("pad_f32", m, n, m, n, t.0, 1, mp, *per, g.c, g.ldc, g.sc as i32, batch)?,
        }
    }
    Ok(())
}

fn grid2(m: i32, n: i32, batch: i32) -> [u32; 3] {
    [(m as u32).div_ceil(64), ((n as u32).div_ceil(4)).clamp(1, 1024), batch as u32]
}

/// dst (mp x np, leading dimension dld) = src (m x n, strides srs / scs) and zeros beyond it, per batch
unsafe fn pad_launch(name: &str, mp: i32, np: i32, m: i32, n: i32, src: u64, srs: i32, scs: i32, ss: i32, dst: u64, dld: i32,
                     ds: i32, batch: i32) -> R {
    launch(name, grid2(mp, np, batch), [64, 4, 1], &[
        i32v(mp), i32v(np), i32v(m), i32v(n), Arg::Addr(src), i32v(0), i32v(srs), i32v(scs), i32v(ss), Arg::Addr(dst),
        i32v(0), i32v(dld), i32v(ds),
    ])
}

/// f16 / bf16 <-> f32 copy of an m x n matrix, per batch
unsafe fn cvt_launch(name: &str, m: i32, n: i32, src: u64, sld: i32, ss: i32, dst: u64, dld: i32, ds: i32, batch: i32) -> R {
    launch(name, grid2(m, n, batch), [64, 4, 1], &[
        i32v(m), i32v(n), Arg::Addr(src), i32v(0), i32v(sld), i32v(ss), Arg::Addr(dst), i32v(0), i32v(dld), i32v(ds),
    ])
}

/// The register-tiled kernel (4 x 4 elements per thread) on the m x n part of C at element offset co, A and B at
/// ao / bo; narrow parts get narrower blocks (more of them)
unsafe fn gemm_generic(name: &str, g: &Gemm, m: i32, n: i32, k: i32, alpha: &[u8], beta: &[u8], ao: i32, bo: i32, co: i32) -> R {
    let (ars, acs) = strides(g.ta, g.lda);
    let (brs, bcs) = strides(g.tb, g.ldb);
    let t = if name == "dgemm" { 2 } else { 4 };
    // threads along the rows: enough for m (powers of two, 2 to 16), the rest of 256 along the columns
    let bx = ((m as u32).div_ceil(t)).next_power_of_two().clamp(2, 16);
    let by = (if (m as u32) < 64 && (n as u32) < 64 { 64 } else { 256 }) / bx;
    let grid = [(m as u32).div_ceil(t * bx), (n as u32).div_ceil(t * by), g.batch as u32];
    launch(name, grid, [bx, by, 1], &[
        i32v(m), i32v(n), i32v(k), Arg::Value(alpha.to_vec()), Arg::Addr(g.a), i32v(ao), i32v(ars), i32v(acs),
        i32v(g.sa as i32), Arg::Addr(g.b), i32v(bo), i32v(brs), i32v(bcs), i32v(g.sb as i32), Arg::Value(beta.to_vec()),
        Arg::Addr(g.c), i32v(co), i32v(g.ldc), i32v(g.sc as i32),
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

// ------------------------------------------------------------------------------------------------- structured

const FILL_LOWER: c_int = 0;
const FILL_UPPER: c_int = 1;
const DIAG_NON_UNIT: c_int = 0;
const DIAG_UNIT: c_int = 1;
const SIDE_LEFT: c_int = 0;
const SIDE_RIGHT: c_int = 1;

// geam kernel modes (blas/gen_kernels.py): A's storage, its structure, products, the triangle written
const G_FULL: i32 = 1;
const G_BAND: i32 = 2;
const G_PACKED: i32 = 3;
const G_SYM: i32 = 4;
const G_TRI: i32 = 8;
const G_UPPER: i32 = 16;
const G_UNIT: i32 = 32;
const G_PROD: i32 = 64;
const G_WRITE_UPPER: i32 = 128;
const G_WRITE_LOWER: i32 = 256;

fn uplo_ok(u: c_int) -> bool {
    matches!(u, FILL_LOWER | FILL_UPPER)
}
fn diag_ok(d: c_int) -> bool {
    matches!(d, DIAG_NON_UNIT | DIAG_UNIT)
}
fn side_ok(s: c_int) -> bool {
    matches!(s, SIDE_LEFT | SIDE_RIGHT)
}

/// The structure bits of a symmetric / triangular A
fn smode(sym: bool, uplo: c_int, diag: c_int) -> i32 {
    (if sym { G_SYM } else { G_TRI }) | if uplo == FILL_UPPER { G_UPPER } else { 0 } | if !sym && diag == DIAG_UNIT { G_UNIT } else { 0 }
}

/// One launch of the elementwise kernel (C(i, j) = c[co + i crs + j ccs] for m x n)
unsafe fn geam_launch<T: Elem>(m: i32, n: i32, mode: i32, alpha: T, a: u64, ao: i32, ars: i32, acs: i32, kl: i32, ku: i32,
                               beta: T, b: u64, bo: i32, brs: i32, bcs: i32, c: u64, co: i32, crs: i32, ccs: i32) -> R {
    if m <= 0 || n <= 0 {
        return Ok(());
    }
    let gy = ((n as u32).div_ceil(4)).clamp(1, 1024);
    launch(&format!("{}geam", T::P), [(m as u32).div_ceil(64), gy, 1], [64, 4, 1], &[
        i32v(m), i32v(n), i32v(mode), Arg::Value(alpha.bytes()), Arg::Addr(a), i32v(ao), i32v(ars), i32v(acs), i32v(kl),
        i32v(ku), Arg::Value(beta.bytes()), Arg::Addr(b), i32v(bo), i32v(brs), i32v(bcs), Arg::Addr(c), i32v(co),
        i32v(crs), i32v(ccs),
    ])
}

fn elems(rows: i32, cols: i32) -> Result<i32, cublasStatus_t> {
    let v = rows as i64 * cols as i64;
    if v > i32::MAX as i64 {
        Err(fail("matrices beyond 2^31 elements are not supported", CUBLAS_STATUS_NOT_SUPPORTED))
    } else {
        Ok(v as i32)
    }
}

/// A rows x cols dense copy (leading dimension rows) of the structured / banded / packed matrix `a`
unsafe fn expand<T: Elem>(rows: i32, cols: i32, mode: i32, a: u64, lda: i32, kl: i32, ku: i32) -> Result<Tmp, cublasStatus_t> {
    let t = Tmp::new(elems(rows, cols)? as u64 * std::mem::size_of::<T>() as u64)?;
    geam_launch::<T>(rows, cols, mode, T::from_f64(1.0), a, 0, 1, lda, kl, ku, T::default(), a, 0, 0, 0, t.0, 0, 1, rows)?;
    Ok(t)
}

/// A contiguous copy of a vector
unsafe fn vec_copy<T: Elem>(n: i32, x: u64, incx: i32) -> Result<Tmp, cublasStatus_t> {
    let t = Tmp::new(n.max(1) as u64 * std::mem::size_of::<T>() as u64)?;
    let (xo, xi) = vec_of(n, incx);
    launch(&format!("{}copy", T::P), [grid1(n, 256), 1, 1], [256, 1, 1], &[
        i32v(n), i32v(0), Arg::Addr(x), i32v(xo), i32v(xi), Arg::Addr(t.0), i32v(0), i32v(1),
    ])?;
    Ok(t)
}

unsafe fn rot<T: Elem>(h: cublasHandle_t, n: c_int, x: *mut T, incx: c_int, y: *mut T, incy: c_int, hm: [T; 4]) -> R {
    if n <= 0 {
        return Ok(());
    }
    let (xo, xi) = vec_of(n, incx);
    let (yo, yi) = vec_of(n, incy);
    let _ = h;
    launch(&format!("{}rot", T::P), [grid1(n, 256), 1, 1], [256, 1, 1], &[
        i32v(n), Arg::Value(hm[0].bytes()), Arg::Value(hm[1].bytes()), Arg::Value(hm[2].bytes()), Arg::Value(hm[3].bytes()),
        Arg::Addr(x as u64), i32v(xo), i32v(xi), Arg::Addr(y as u64), i32v(yo), i32v(yi),
    ])
}

/// rotg's results (a, b, c, s) for (a, b) (reference BLAS)
fn rotg_f64(a: f64, b: f64) -> (f64, f64, f64, f64) {
    let roe = if a.abs() > b.abs() { a } else { b };
    let scale = a.abs() + b.abs();
    if scale == 0.0 {
        return (0.0, 0.0, 1.0, 0.0);
    }
    let r = scale * ((a / scale).powi(2) + (b / scale).powi(2)).sqrt();
    let r = if roe < 0.0 { -r } else { r };
    let (c, s) = (a / r, b / r);
    let mut z = 1.0;
    if a.abs() > b.abs() {
        z = s;
    }
    if b.abs() >= a.abs() && c != 0.0 {
        z = 1.0 / c;
    }
    (r, z, c, s)
}

/// rotmg (reference BLAS): (d1, d2, x1, param) for (d1, d2, x1, y1)
fn rotmg_f64(mut d1: f64, mut d2: f64, mut x1: f64, y1: f64) -> (f64, f64, f64, [f64; 5]) {
    let (gam, gamsq, rgamsq) = (4096.0f64, 16777216.0f64, 5.9604645e-8f64);
    let (mut h11, mut h12, mut h21, mut h22) = (0.0, 0.0, 0.0, 0.0);
    let flag: f64;
    if d1 < 0.0 {
        return (0.0, 0.0, 0.0, [-1.0, 0.0, 0.0, 0.0, 0.0]);
    }
    let p2 = d2 * y1;
    if p2 == 0.0 {
        return (d1, d2, x1, [-2.0, 0.0, 0.0, 0.0, 0.0]);
    }
    let p1 = d1 * x1;
    let q2 = p2 * y1;
    let q1 = p1 * x1;
    let mut fl;
    if q1.abs() > q2.abs() {
        h21 = -y1 / x1;
        h12 = p2 / p1;
        let u = 1.0 - h12 * h21;
        if u > 0.0 {
            fl = 0.0;
            d1 /= u;
            d2 /= u;
            x1 *= u;
        } else {
            return (0.0, 0.0, 0.0, [-1.0, 0.0, 0.0, 0.0, 0.0]);
        }
    } else if q2 < 0.0 {
        return (0.0, 0.0, 0.0, [-1.0, 0.0, 0.0, 0.0, 0.0]);
    } else {
        fl = 1.0;
        h11 = p1 / p2;
        h22 = x1 / y1;
        let u = 1.0 + h11 * h22;
        let t = d2 / u;
        d2 = d1 / u;
        d1 = t;
        x1 = y1 * u;
    }
    if d1 != 0.0 {
        while d1 <= rgamsq || d1 >= gamsq {
            if fl == 0.0 {
                h11 = 1.0;
                h22 = 1.0;
                fl = -1.0;
            } else {
                h21 = -1.0;
                h12 = 1.0;
                fl = -1.0;
            }
            if d1 <= rgamsq {
                d1 *= gam * gam;
                x1 /= gam;
                h11 /= gam;
                h12 /= gam;
            } else {
                d1 /= gam * gam;
                x1 *= gam;
                h11 *= gam;
                h12 *= gam;
            }
        }
    }
    if d2 != 0.0 {
        while d2.abs() <= rgamsq || d2.abs() >= gamsq {
            if fl == 0.0 {
                h11 = 1.0;
                h22 = 1.0;
                fl = -1.0;
            } else {
                h21 = -1.0;
                h12 = 1.0;
                fl = -1.0;
            }
            if d2.abs() <= rgamsq {
                d2 *= gam * gam;
                h21 /= gam;
                h22 /= gam;
            } else {
                d2 /= gam * gam;
                h21 *= gam;
                h22 *= gam;
            }
        }
    }
    flag = fl;
    let param = if flag < 0.0 {
        [flag, h11, h21, h12, h22]
    } else if flag == 0.0 {
        [flag, 0.0, h21, h12, 0.0]
    } else {
        [flag, h11, 0.0, 0.0, h22]
    };
    (d1, d2, x1, param)
}

/// A scalar argument that is also a result (rotg, rotmg), in the handle's pointer mode
unsafe fn sread<T: Copy>(h: cublasHandle_t, p: *const T) -> Result<T, cublasStatus_t> {
    scalar(h, p)
}

unsafe fn rotm<T: Elem>(h: cublasHandle_t, n: c_int, x: *mut T, incx: c_int, y: *mut T, incy: c_int, param: *const T) -> R {
    check(h)?;
    if param.is_null() {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let p: Vec<f64> = (0..5).map(|i| sread(h, param.add(i)).map(|v| v.to_f64())).collect::<Result<_, _>>()?;
    let f = T::from_f64;
    let hm = match p[0] {
        v if v == -2.0 => return Ok(()),
        v if v == -1.0 => [f(p[1]), f(p[2]), f(p[3]), f(p[4])],
        v if v == 0.0 => [f(1.0), f(p[2]), f(p[3]), f(1.0)],
        v if v == 1.0 => [f(p[1]), f(-1.0), f(1.0), f(p[4])],
        _ => return Err(CUBLAS_STATUS_INVALID_VALUE),
    };
    rot(h, n, x, incx, y, incy, hm)
}

unsafe fn rotg<T: Elem>(h: cublasHandle_t, a: *mut T, b: *mut T, c: *mut T, s: *mut T) -> R {
    check(h)?;
    let (r, z, cv, sv) = rotg_f64(sread(h, a)?.to_f64(), sread(h, b)?.to_f64());
    result(h, a, T::from_f64(r))?;
    result(h, b, T::from_f64(z))?;
    result(h, c, T::from_f64(cv))?;
    result(h, s, T::from_f64(sv))
}

unsafe fn rotmg<T: Elem>(h: cublasHandle_t, d1: *mut T, d2: *mut T, x1: *mut T, y1: *const T, param: *mut T) -> R {
    check(h)?;
    let (a, b, c, p) = rotmg_f64(sread(h, d1)?.to_f64(), sread(h, d2)?.to_f64(), sread(h, x1)?.to_f64(), sread(h, y1)?.to_f64());
    result(h, d1, T::from_f64(a))?;
    result(h, d2, T::from_f64(b))?;
    result(h, x1, T::from_f64(c))?;
    for (i, v) in p.iter().enumerate() {
        result(h, param.add(i), T::from_f64(*v))?;
    }
    Ok(())
}

/// A += alpha x y^T, masked to a triangle (1 upper, 2 lower)
unsafe fn ger_launch<T: Elem>(m: i32, n: i32, mask: i32, alpha: T, x: u64, incx: i32, y: u64, incy: i32, a: u64, lda: i32) -> R {
    if m <= 0 || n <= 0 {
        return Ok(());
    }
    let (xo, xi) = vec_of(m, incx);
    let (yo, yi) = vec_of(n, incy);
    let gy = ((n as u32).div_ceil(4)).clamp(1, 1024);
    launch(&format!("{}ger", T::P), [(m as u32).div_ceil(64), gy, 1], [64, 4, 1], &[
        i32v(m), i32v(n), i32v(mask), Arg::Value(alpha.bytes()), Arg::Addr(x), i32v(xo), i32v(xi), Arg::Addr(y), i32v(yo),
        i32v(yi), Arg::Addr(a), i32v(0), i32v(lda),
    ])
}

unsafe fn ger<T: Elem>(h: cublasHandle_t, m: c_int, n: c_int, alpha: *const T, x: *const T, incx: c_int, y: *const T,
                       incy: c_int, a: *mut T, lda: c_int) -> R {
    check(h)?;
    if m < 0 || n < 0 || incx == 0 || incy == 0 || lda < m.max(1) {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let al = scalar(h, alpha)?;
    elems(lda, n)?;
    ger_launch(m, n, 0, al, x as u64, incx, y as u64, incy, a as u64, lda)
}

unsafe fn syr<T: Elem>(h: cublasHandle_t, uplo: c_int, n: c_int, alpha: *const T, x: *const T, incx: c_int, y: *const T,
                       incy: c_int, a: *mut T, lda: c_int, two: bool) -> R {
    check(h)?;
    if !uplo_ok(uplo) || n < 0 || incx == 0 || (two && incy == 0) || lda < n.max(1) {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let al = scalar(h, alpha)?;
    elems(lda, n)?;
    let mask = if uplo == FILL_UPPER { 1 } else { 2 };
    if two {
        ger_launch(n, n, mask, al, x as u64, incx, y as u64, incy, a as u64, lda)?;
        ger_launch(n, n, mask, al, y as u64, incy, x as u64, incx, a as u64, lda)
    } else {
        ger_launch(n, n, mask, al, x as u64, incx, x as u64, incx, a as u64, lda)
    }
}

/// y = alpha op(A) x + beta y for op(A) rows x cols with A(i, j) = a[i rs + j cs], batched (strides in elements)
unsafe fn gemv_launch<T: Elem>(rows: i32, cols: i32, al: T, a: u64, rs: i32, cs: i32, sa: i64, x: u64, incx: i32, sx: i64,
                               be: T, y: u64, incy: i32, sy: i64, batch: i32) -> R {
    if rows <= 0 || batch <= 0 {
        return Ok(());
    }
    let big = |v: i64| v > i32::MAX as i64 || v < i32::MIN as i64;
    if big(sa) || big(sx) || big(sy) || big(sa * batch as i64) {
        return Err(fail("matrices beyond 2^31 elements are not supported", CUBLAS_STATUS_NOT_SUPPORTED));
    }
    let (xo, xi) = vec_of(cols, incx);
    let (yo, yi) = vec_of(rows, incy);
    // a SIMD group per row where rows are contiguous or few, else 32 rows (lanes) by 8 column groups per block
    let (name, grid, block) = if cs == 1 || rows < 64 || std::env::var_os("ZLUDA_BLAS_GEMV_T").is_some() {
        ("gemv_t", [(rows as u32).div_ceil(8).min(8192), batch as u32, 1], [256, 1, 1])
    } else {
        ("gemv_n", [(rows as u32).div_ceil(32).min(4096), batch as u32, 1], [32, match gemv_groups() { 0 if T::P == "s" => 32, 0 => 16, g => g }, 1])
    };
    launch(&format!("{}{}", T::P, name), grid, block, &[
        i32v(rows), i32v(cols), Arg::Value(al.bytes()), Arg::Addr(a), i32v(0), i32v(rs), i32v(cs), i32v(sa as i32),
        Arg::Addr(x), i32v(xo), i32v(xi), i32v(sx as i32), Arg::Value(be.bytes()), Arg::Addr(y), i32v(yo), i32v(yi),
        i32v(sy as i32),
    ])
}

unsafe fn gemv_sb<T: Elem>(h: cublasHandle_t, trans: c_int, m: c_int, n: c_int, alpha: *const T, a: *const T, lda: c_int,
                           sa: i64, x: *const T, incx: c_int, sx: i64, beta: *const T, y: *mut T, incy: c_int, sy: i64,
                           batch: c_int) -> R {
    check(h)?;
    if !op_ok(trans) || m < 0 || n < 0 || lda < m.max(1) || incx == 0 || incy == 0 || batch < 0 {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let (al, be) = (scalar(h, alpha)?, scalar(h, beta)?);
    let (rows, cols) = if trans == OP_N { (m, n) } else { (n, m) };
    elems(lda, n)?;
    let (rs, cs) = strides(trans, lda);
    gemv_launch(rows, cols, al, a as u64, rs, cs, sa, x as u64, incx, sx, be, y as u64, incy, sy, batch)
}

/// symv / sbmv / spmv: y = alpha A x + beta y for A n x n symmetric, expanded (mode: storage) to a dense copy
unsafe fn symv<T: Elem>(h: cublasHandle_t, kind: i32, uplo: c_int, n: c_int, k: c_int, alpha: *const T, a: *const T,
                        lda: c_int, x: *const T, incx: c_int, beta: *const T, y: *mut T, incy: c_int) -> R {
    check(h)?;
    let ldmin = match kind {
        G_FULL => n.max(1),
        G_BAND => (k + 1).max(1),
        _ => 0,
    };
    if !uplo_ok(uplo) || n < 0 || k < 0 || lda < ldmin || incx == 0 || incy == 0 {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let (al, be) = (scalar(h, alpha)?, scalar(h, beta)?);
    if n == 0 {
        return Ok(());
    }
    let t = expand::<T>(n, n, kind | smode(true, uplo, 0), a as u64, lda, k, k)?;
    gemv_launch(n, n, al, t.0, 1, n, 0, x as u64, incx, 0, be, y as u64, incy, 0, 1)
}

/// gbmv: y = alpha op(A) x + beta y for A m x n banded (kl, ku)
unsafe fn gbmv<T: Elem>(h: cublasHandle_t, trans: c_int, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: *const T,
                        a: *const T, lda: c_int, x: *const T, incx: c_int, beta: *const T, y: *mut T, incy: c_int) -> R {
    check(h)?;
    if !op_ok(trans) || m < 0 || n < 0 || kl < 0 || ku < 0 || lda < kl + ku + 1 || incx == 0 || incy == 0 {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let (al, be) = (scalar(h, alpha)?, scalar(h, beta)?);
    if m == 0 || n == 0 {
        return Ok(());
    }
    let t = expand::<T>(m, n, G_BAND, a as u64, lda, kl, ku)?;
    let (rows, cols) = if trans == OP_N { (m, n) } else { (n, m) };
    let (rs, cs) = strides(trans, m);
    gemv_launch(rows, cols, al, t.0, rs, cs, 0, x as u64, incx, 0, be, y as u64, incy, 0, 1)
}

/// trmv / tbmv / tpmv: x = op(A) x
unsafe fn trmv<T: Elem>(h: cublasHandle_t, kind: i32, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int,
                        a: *const T, lda: c_int, x: *mut T, incx: c_int) -> R {
    check(h)?;
    let ldmin = match kind {
        G_FULL => n.max(1),
        G_BAND => (k + 1).max(1),
        _ => 0,
    };
    if !uplo_ok(uplo) || !op_ok(trans) || !diag_ok(diag) || n < 0 || k < 0 || lda < ldmin || incx == 0 {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    if n == 0 {
        return Ok(());
    }
    let t = expand::<T>(n, n, kind | smode(false, uplo, diag), a as u64, lda, k, k)?;
    let xc = vec_copy::<T>(n, x as u64, incx)?;
    let (rs, cs) = strides(trans, n);
    gemv_launch(n, n, T::from_f64(1.0), t.0, rs, cs, 0, xc.0, 1, 0, T::default(), x as u64, incx, 0, 1)
}

/// The triangular solve kernel: nvec vectors (x + xo + v xb, increment xi) of n elements, M(i, j) = a[i rs + j cs]
unsafe fn trsv_launch<T: Elem>(n: i32, a: u64, rs: i32, cs: i32, lower: bool, unit: bool, alpha: T, x: u64, xo: i32, xi: i32,
                               xb: i32, nvec: i32) -> R {
    if n <= 0 || nvec <= 0 {
        return Ok(());
    }
    launch(&format!("{}trsv", T::P), [nvec as u32, 1, 1], [32, 1, 1], &[
        i32v(n), Arg::Addr(a), i32v(0), i32v(rs), i32v(cs), i32v(lower as i32), i32v(unit as i32), Arg::Value(alpha.bytes()),
        Arg::Addr(x), i32v(xo), i32v(xi), i32v(xb),
    ])
}

/// trsv / tbsv / tpsv: solves op(A) x = b
unsafe fn trsv<T: Elem>(h: cublasHandle_t, kind: i32, uplo: c_int, trans: c_int, diag: c_int, n: c_int, k: c_int,
                        a: *const T, lda: c_int, x: *mut T, incx: c_int) -> R {
    check(h)?;
    let ldmin = match kind {
        G_FULL => n.max(1),
        G_BAND => (k + 1).max(1),
        _ => 0,
    };
    if !uplo_ok(uplo) || !op_ok(trans) || !diag_ok(diag) || n < 0 || k < 0 || lda < ldmin || incx == 0 {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    if n == 0 {
        return Ok(());
    }
    let (xo, xi) = vec_of(n, incx);
    let lower = (uplo == FILL_LOWER) != (trans != OP_N);
    if kind == G_FULL {
        elems(lda, n)?;
        let (rs, cs) = strides(trans, lda);
        return trsv_launch(n, a as u64, rs, cs, lower, diag == DIAG_UNIT, T::from_f64(1.0), x as u64, xo, xi, 0, 1);
    }
    let t = expand::<T>(n, n, kind | smode(false, uplo, diag), a as u64, lda, k, k)?;
    let (rs, cs) = strides(trans, n);
    trsv_launch(n, t.0, rs, cs, lower, diag == DIAG_UNIT, T::from_f64(1.0), x as u64, xo, xi, 0, 1)
}

/// trsm: op(A) X = alpha B (left) or X op(A) = alpha B (right), X over B
unsafe fn trsm<T: Elem>(h: cublasHandle_t, side: c_int, uplo: c_int, trans: c_int, diag: c_int, m: c_int, n: c_int,
                        alpha: *const T, a: *const T, lda: c_int, b: *mut T, ldb: c_int) -> R {
    check(h)?;
    let k = if side == SIDE_LEFT { m } else { n };
    if !side_ok(side) || !uplo_ok(uplo) || !op_ok(trans) || !diag_ok(diag) || m < 0 || n < 0 || lda < k.max(1) || ldb < m.max(1) {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let al = scalar(h, alpha)?;
    if m == 0 || n == 0 {
        return Ok(());
    }
    elems(lda, k)?;
    elems(ldb, n)?;
    let unit = diag == DIAG_UNIT;
    if side == SIDE_LEFT {
        // each column of B: op(A) x = alpha b
        let (rs, cs) = strides(trans, lda);
        trsv_launch(m, a as u64, rs, cs, (uplo == FILL_LOWER) != (trans != OP_N), unit, al, b as u64, 0, 1, ldb, n)
    } else {
        // each row of B: op(A)^T x = alpha b
        let (rs, cs) = strides(if trans == OP_N { OP_T } else { OP_N }, lda);
        trsv_launch(n, a as u64, rs, cs, (uplo == FILL_LOWER) != (trans == OP_N), unit, al, b as u64, 0, ldb, 1, m)
    }
}

/// symm: C = alpha A B + beta C (left) or alpha B A + beta C (right), A symmetric: a dense copy and a GEMM
unsafe fn symm<T: Elem>(h: cublasHandle_t, side: c_int, uplo: c_int, m: c_int, n: c_int, alpha: *const T, a: *const T,
                        lda: c_int, b: *const T, ldb: c_int, beta: *const T, c: *mut T, ldc: c_int) -> R {
    check(h)?;
    let k = if side == SIDE_LEFT { m } else { n };
    if !side_ok(side) || !uplo_ok(uplo) || m < 0 || n < 0 || lda < k.max(1) || ldb < m.max(1) || ldc < m.max(1) {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let (al, be) = (scalar(h, alpha)?, scalar(h, beta)?);
    if m == 0 || n == 0 {
        return Ok(());
    }
    let t = expand::<T>(k, k, G_FULL | smode(true, uplo, 0), a as u64, lda, 0, 0)?;
    let name = format!("{}gemm", T::P);
    if side == SIDE_LEFT {
        gemm_launch(&name, OP_N, OP_N, m, n, m, al.bytes(), t.0, k, 0, b as u64, ldb, 0, be.bytes(), c as u64, ldc, 0, 1)
    } else {
        gemm_launch(&name, OP_N, OP_N, m, n, n, al.bytes(), b as u64, ldb, 0, t.0, k, 0, be.bytes(), c as u64, ldc, 0, 1)
    }
}

/// syrk / syr2k / syrkx: the triangle uplo of C = alpha op(A) op(B)^T [+ alpha op(B) op(A)^T] + beta C, by GEMMs
/// into a copy of C and a copy of its triangle back
unsafe fn syrk<T: Elem>(h: cublasHandle_t, uplo: c_int, trans: c_int, n: c_int, k: c_int, alpha: *const T, a: *const T,
                        lda: c_int, b: *const T, ldb: c_int, beta: *const T, c: *mut T, ldc: c_int, two: bool) -> R {
    check(h)?;
    let rows = if trans == OP_N { n } else { k };
    if !uplo_ok(uplo) || !op_ok(trans) || n < 0 || k < 0 || lda < rows.max(1) || ldb < rows.max(1) || ldc < n.max(1) {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let (al, be) = (scalar(h, alpha)?, scalar(h, beta)?);
    if n == 0 {
        return Ok(());
    }
    elems(ldc, n)?;
    let w = Tmp::new(elems(n, n)? as u64 * std::mem::size_of::<T>() as u64)?;
    let zero = be.to_f64() == 0.0;
    if !zero {
        geam_launch::<T>(n, n, 0, T::from_f64(1.0), c as u64, 0, 1, ldc, 0, 0, T::default(), c as u64, 0, 0, 0, w.0, 0, 1, n)?;
    }
    let (ta, tb) = if trans == OP_N { (OP_N, OP_T) } else { (OP_T, OP_N) };
    let name = format!("{}gemm", T::P);
    gemm_launch(&name, ta, tb, n, n, k, al.bytes(), a as u64, lda, 0, b as u64, ldb, 0, be.bytes(), w.0, n, 0, 1)?;
    if two {
        gemm_launch(&name, ta, tb, n, n, k, al.bytes(), b as u64, ldb, 0, a as u64, lda, 0, T::from_f64(1.0).bytes(), w.0, n, 0, 1)?;
    }
    let mask = if uplo == FILL_UPPER { G_WRITE_UPPER } else { G_WRITE_LOWER };
    geam_launch::<T>(n, n, mask, T::from_f64(1.0), w.0, 0, 1, n, 0, 0, T::default(), w.0, 0, 0, 0, c as u64, 0, 1, ldc)
}

/// trmm: C = alpha op(A) B (left) or alpha B op(A) (right), A triangular
unsafe fn trmm<T: Elem>(h: cublasHandle_t, side: c_int, uplo: c_int, trans: c_int, diag: c_int, m: c_int, n: c_int,
                        alpha: *const T, a: *const T, lda: c_int, b: *const T, ldb: c_int, c: *mut T, ldc: c_int) -> R {
    check(h)?;
    let k = if side == SIDE_LEFT { m } else { n };
    if !side_ok(side) || !uplo_ok(uplo) || !op_ok(trans) || !diag_ok(diag) || m < 0 || n < 0 || lda < k.max(1)
        || ldb < m.max(1) || ldc < m.max(1)
    {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let al = scalar(h, alpha)?;
    if m == 0 || n == 0 {
        return Ok(());
    }
    let t = expand::<T>(k, k, G_FULL | smode(false, uplo, diag), a as u64, lda, 0, 0)?;
    // in place (C = B): B through a copy
    let bc;
    let (bp, ldbp) = if b as u64 == c as u64 {
        bc = Tmp::new(elems(m, n)? as u64 * std::mem::size_of::<T>() as u64)?;
        geam_launch::<T>(m, n, 0, T::from_f64(1.0), b as u64, 0, 1, ldb, 0, 0, T::default(), b as u64, 0, 0, 0, bc.0, 0, 1, m)?;
        (bc.0, m)
    } else {
        (b as u64, ldb)
    };
    let name = format!("{}gemm", T::P);
    let zero = T::default().bytes();
    if side == SIDE_LEFT {
        gemm_launch(&name, trans, OP_N, m, n, m, al.bytes(), t.0, k, 0, bp, ldbp, 0, zero, c as u64, ldc, 0, 1)
    } else {
        gemm_launch(&name, OP_N, trans, m, n, n, al.bytes(), bp, ldbp, 0, t.0, k, 0, zero, c as u64, ldc, 0, 1)
    }
}

/// geam: C = alpha op(A) + beta op(B)
unsafe fn geam<T: Elem>(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, alpha: *const T, a: *const T, lda: c_int,
                        beta: *const T, b: *const T, ldb: c_int, c: *mut T, ldc: c_int) -> R {
    check(h)?;
    let need = |t: c_int| if t == OP_N { m } else { n };
    if !op_ok(ta) || !op_ok(tb) || m < 0 || n < 0 || lda < need(ta).max(1) || ldb < need(tb).max(1) || ldc < m.max(1) {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    // in place only as the same matrix
    if (c as u64 == a as u64 && (ta != OP_N || lda != ldc)) || (c as u64 == b as u64 && (tb != OP_N || ldb != ldc)) {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let (al, be) = (scalar(h, alpha)?, scalar(h, beta)?);
    elems(ldc, n)?;
    let (ars, acs) = strides(ta, lda);
    let (brs, bcs) = strides(tb, ldb);
    geam_launch(m, n, 0, al, a as u64, 0, ars, acs, 0, 0, be, b as u64, 0, brs, bcs, c as u64, 0, 1, ldc)
}

/// dgmm: C = A diag(x) (right) or diag(x) A (left)
unsafe fn dgmm<T: Elem>(h: cublasHandle_t, side: c_int, m: c_int, n: c_int, a: *const T, lda: c_int, x: *const T, incx: c_int,
                        c: *mut T, ldc: c_int) -> R {
    check(h)?;
    if !side_ok(side) || m < 0 || n < 0 || lda < m.max(1) || ldc < m.max(1) {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    elems(ldc.max(lda), n)?;
    let (xo, xi) = vec_of(if side == SIDE_LEFT { m } else { n }, incx);
    let (brs, bcs) = if side == SIDE_LEFT { (xi, 0) } else { (0, xi) };
    geam_launch(m, n, G_PROD, T::from_f64(1.0), a as u64, 0, 1, lda, 0, 0, T::default(), x as u64, xo, brs, bcs, c as u64, 0, 1, ldc)
}

// ------------------------------------------------------------------------------------------------- complex

/// cuComplex / cuDoubleComplex: (re, im) of the real element type
type Cx<T> = [T; 2];

fn cp<T: Elem>() -> &'static str {
    if T::P == "s" { "c" } else { "z" }
}

fn cbytes<T: Elem>(v: Cx<T>) -> [Arg<'static>; 2] {
    [Arg::Value(v[0].bytes()), Arg::Value(v[1].bytes())]
}

unsafe fn caxpy<T: Elem>(h: cublasHandle_t, n: c_int, alpha: Cx<T>, x: u64, incx: c_int, y: u64, incy: c_int, scal: bool) -> R {
    check(h)?;
    if n <= 0 || (scal && incx <= 0) {
        return Ok(());
    }
    let (xo, xi) = if scal { (0, incx) } else { vec_of(n, incx) };
    let (yo, yi) = vec_of(n, incy);
    let [ar, ai] = cbytes(alpha);
    launch(&format!("{}axpy", cp::<T>()), [grid1(n, 256), 1, 1], [256, 1, 1], &[
        i32v(n), i32v(scal as i32), ar, ai, Arg::Addr(x), i32v(xo), i32v(xi), Arg::Addr(y), i32v(yo), i32v(yi),
    ])
}

/// copy / swap of complex vectors: the real kernel on the real and the imaginary parts
unsafe fn ccopy<T: Elem>(h: cublasHandle_t, n: c_int, x: u64, incx: c_int, y: u64, incy: c_int, swap: bool) -> R {
    check(h)?;
    if n <= 0 {
        return Ok(());
    }
    let (xo, xi) = vec_of(n, incx);
    let (yo, yi) = vec_of(n, incy);
    for part in 0..2 {
        launch(&format!("{}copy", T::P), [grid1(n, 256), 1, 1], [256, 1, 1], &[
            i32v(n), i32v(swap as i32), Arg::Addr(x), i32v(2 * xo + part), i32v(2 * xi), Arg::Addr(y), i32v(2 * yo + part),
            i32v(2 * yi),
        ])?;
    }
    Ok(())
}

unsafe fn cdot<T: Elem>(h: cublasHandle_t, n: c_int, x: u64, incx: c_int, y: u64, incy: c_int, r: *mut Cx<T>, conj: bool) -> R {
    check(h)?;
    let v = if n <= 0 {
        [T::default(); 2]
    } else {
        let sz = 2 * std::mem::size_of::<T>() as u64;
        let g = ((n as u32).div_ceil(32)).clamp(1, 256);
        let part = Tmp::new(g as u64 * sz)?;
        let res = Tmp::new(sz)?;
        let (xo, xi) = vec_of(n, incx);
        let (yo, yi) = vec_of(n, incy);
        let k = format!("{}reduce", cp::<T>());
        launch(&k, [g, 1, 1], [32, 1, 1], &[
            i32v(n), i32v(conj as i32), Arg::Addr(x), i32v(xo), i32v(xi), Arg::Addr(y), i32v(yo), i32v(yi), Arg::Addr(part.0),
        ])?;
        launch(&k, [1, 1, 1], [32, 1, 1], &[
            i32v(g as i32), i32v(3), Arg::Addr(part.0), i32v(0), i32v(1), Arg::Addr(part.0), i32v(0), i32v(1), Arg::Addr(res.0),
        ])?;
        dev_read::<Cx<T>>(res.0)?
    };
    result(h, r, v)
}

/// nrm2 (op 1) / asum (op 2) of a complex vector: the real reductions of both parts
unsafe fn cnorm<T: Elem>(h: cublasHandle_t, n: c_int, x: u64, incx: c_int, r: *mut T, op: i32) -> R {
    check(h)?;
    let v = if n <= 0 || incx <= 0 {
        T::default()
    } else if incx == 1 {
        reduce::<T>(2 * n, op, x, 1, x, 1)?
    } else {
        let sz = std::mem::size_of::<T>() as u64;
        let s0 = reduce::<T>(n, op, x, 2 * incx, x, 2 * incx)?;
        let s1 = reduce::<T>(n, op, x + sz, 2 * incx, x + sz, 2 * incx)?;
        T::from_f64(s0.to_f64() + s1.to_f64())
    };
    result(h, r, if op == 1 { T::from_f64(v.to_f64().sqrt()) } else { v })
}

unsafe fn cgemv<T: Elem>(h: cublasHandle_t, trans: c_int, m: c_int, n: c_int, alpha: *const Cx<T>, a: u64, lda: c_int,
                         x: u64, incx: c_int, beta: *const Cx<T>, y: u64, incy: c_int) -> R {
    check(h)?;
    if !op_ok(trans) || m < 0 || n < 0 || lda < m.max(1) || incx == 0 || incy == 0 {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let (al, be) = (scalar(h, alpha)?, scalar(h, beta)?);
    let (rows, cols) = if trans == OP_N { (m, n) } else { (n, m) };
    if rows == 0 {
        return Ok(());
    }
    elems(lda, n)?;
    let (rs, cs) = strides(trans, lda);
    let (xo, xi) = vec_of(cols, incx);
    let (yo, yi) = vec_of(rows, incy);
    let ([ar, ai], [br, bi]) = (cbytes(al), cbytes(be));
    launch(&format!("{}gemv", cp::<T>()), [grid1(rows, 64), 1, 1], [64, 1, 1], &[
        i32v(rows), i32v(cols), i32v((trans == OP_C) as i32), ar, ai, Arg::Addr(a), i32v(0), i32v(rs), i32v(cs),
        Arg::Addr(x), i32v(xo), i32v(xi), br, bi, Arg::Addr(y), i32v(yo), i32v(yi),
    ])
}

unsafe fn cgemm<T: Elem>(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const Cx<T>, a: u64,
                         lda: c_int, sa: i64, b: u64, ldb: c_int, sb: i64, beta: *const Cx<T>, c: u64, ldc: c_int, sc: i64,
                         batch: c_int) -> R {
    check(h)?;
    if !op_ok(ta) || !op_ok(tb) || m < 0 || n < 0 || k < 0 || batch < 0 || ldc < m.max(1)
        || lda < (if ta == OP_N { m } else { k }).max(1) || ldb < (if tb == OP_N { k } else { n }).max(1)
    {
        return Err(CUBLAS_STATUS_INVALID_VALUE);
    }
    let (al, be) = (scalar(h, alpha)?, scalar(h, beta)?);
    if m == 0 || n == 0 || batch == 0 {
        return Ok(());
    }
    let big = |v: i64| v > i32::MAX as i64 / 2 || v < i32::MIN as i64 / 2;
    if big(sa) || big(sb) || big(sc) || big(sc * batch as i64) || big(ldc as i64 * n as i64) {
        return Err(fail("matrices beyond 2^30 elements are not supported", CUBLAS_STATUS_NOT_SUPPORTED));
    }
    let (ars, acs) = strides(ta, lda);
    let (brs, bcs) = strides(tb, ldb);
    let tile = if T::P == "s" { 32 } else { 16 };
    let ([ar, ai], [br, bi]) = (cbytes(al), cbytes(be));
    launch(&format!("{}gemm", cp::<T>()), [(m as u32).div_ceil(tile), (n as u32).div_ceil(tile), batch as u32], [16, 16, 1], &[
        i32v(m), i32v(n), i32v(k), ar, ai, Arg::Addr(a), i32v(0), i32v(ars), i32v(acs), i32v(sa as i32), i32v((ta == OP_C) as i32),
        Arg::Addr(b), i32v(0), i32v(brs), i32v(bcs), i32v(sb as i32), i32v((tb == OP_C) as i32), br, bi, Arg::Addr(c), i32v(0),
        i32v(ldc), i32v(sc as i32),
    ])
}

/// Csrot / Zdrot: the real rotation of one part (0 real, 1 imaginary) of complex vectors
unsafe fn crot<T: Elem>(n: c_int, x: u64, incx: c_int, y: u64, incy: c_int, part: i32, hm: [T; 4]) -> R {
    if n <= 0 {
        return Ok(());
    }
    let (xo, xi) = vec_of(n, incx);
    let (yo, yi) = vec_of(n, incy);
    launch(&format!("{}rot", T::P), [grid1(n, 256), 1, 1], [256, 1, 1], &[
        i32v(n), Arg::Value(hm[0].bytes()), Arg::Value(hm[1].bytes()), Arg::Value(hm[2].bytes()), Arg::Value(hm[3].bytes()),
        Arg::Addr(x), i32v(2 * xo + part), i32v(2 * xi), Arg::Addr(y), i32v(2 * yo + part), i32v(2 * yi),
    ])
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
        *handle = Box::into_raw(Box::new(Handle { stream: std::ptr::null_mut(), pointer_mode: 0, math_mode: 0, atomics_mode: 0, sm_count_target: 0 }));
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
    fn cublasSetAtomicsMode(handle: cublasHandle_t, mode: c_int) {
        check(handle)?;
        if !matches!(mode, 0 | 1) {
            return Err(CUBLAS_STATUS_INVALID_VALUE);
        }
        (*handle).atomics_mode = mode;
        Ok(())
    }
    fn cublasGetAtomicsMode(handle: cublasHandle_t, mode: *mut c_int) {
        check(handle)?;
        if mode.is_null() {
            return Err(CUBLAS_STATUS_INVALID_VALUE);
        }
        *mode = (*handle).atomics_mode;
        Ok(())
    }
    fn cublasSetSmCountTarget(handle: cublasHandle_t, count: c_int) {
        check(handle)?;
        if count < 0 {
            return Err(CUBLAS_STATUS_INVALID_VALUE);
        }
        (*handle).sm_count_target = count;
        Ok(())
    }
    fn cublasGetSmCountTarget(handle: cublasHandle_t, count: *mut c_int) {
        check(handle)?;
        if count.is_null() {
            return Err(CUBLAS_STATUS_INVALID_VALUE);
        }
        *count = (*handle).sm_count_target;
        Ok(())
    }
    // libraryPropertyType: major, minor, patch of the version cublasGetVersion reports (12.4.0)
    fn cublasGetProperty(kind: c_int, value: *mut c_int) {
        if value.is_null() {
            return Err(CUBLAS_STATUS_INVALID_VALUE);
        }
        *value = match kind {
            0 => 12,
            1 => 4,
            2 => 0,
            _ => return Err(CUBLAS_STATUS_INVALID_VALUE),
        };
        Ok(())
    }
    fn cublasLoggerConfigure(_on: c_int, _stdout: c_int, _stderr: c_int, _file: *const c_char) {
        Ok(())
    }
    // the asynchronous copies are synchronous (as the launches)
    fn cublasSetVectorAsync(n: c_int, elem: c_int, x: *const c_void, incx: c_int, y: *mut c_void, incy: c_int, _s: *mut c_void) {
        strided_copy(n, 1, elem, x as *const u8, incx as i64, 0, false, y as *mut u8, incy as i64, 0, true)
    }
    fn cublasGetVectorAsync(n: c_int, elem: c_int, x: *const c_void, incx: c_int, y: *mut c_void, incy: c_int, _s: *mut c_void) {
        strided_copy(n, 1, elem, x as *const u8, incx as i64, 0, true, y as *mut u8, incy as i64, 0, false)
    }
    fn cublasSetMatrixAsync(rows: c_int, cols: c_int, elem: c_int, a: *const c_void, lda: c_int, b: *mut c_void, ldb: c_int, _s: *mut c_void) {
        strided_copy(rows, cols, elem, a as *const u8, 1, lda as i64, false, b as *mut u8, 1, ldb as i64, true)
    }
    fn cublasGetMatrixAsync(rows: c_int, cols: c_int, elem: c_int, a: *const c_void, lda: c_int, b: *mut c_void, ldb: c_int, _s: *mut c_void) {
        strided_copy(rows, cols, elem, a as *const u8, 1, lda as i64, true, b as *mut u8, 1, ldb as i64, false)
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

entry! {
    fn cublasSrot_v2(h: cublasHandle_t, n: c_int, x: *mut f32, incx: c_int, y: *mut f32, incy: c_int, c: *const f32, s: *const f32) { check(h)?; let (c, s) = (scalar(h, c)?, scalar(h, s)?); rot(h, n, x, incx, y, incy, [c, -s, s, c]) }
    fn cublasSrotg_v2(h: cublasHandle_t, a: *mut f32, b: *mut f32, c: *mut f32, s: *mut f32) { rotg(h, a, b, c, s) }
    fn cublasSrotm_v2(h: cublasHandle_t, n: c_int, x: *mut f32, incx: c_int, y: *mut f32, incy: c_int, param: *const f32) { rotm(h, n, x, incx, y, incy, param) }
    fn cublasSrotmg_v2(h: cublasHandle_t, d1: *mut f32, d2: *mut f32, x1: *mut f32, y1: *const f32, param: *mut f32) { rotmg(h, d1, d2, x1, y1, param) }
    fn cublasSger_v2(h: cublasHandle_t, m: c_int, n: c_int, alpha: *const f32, x: *const f32, incx: c_int, y: *const f32, incy: c_int, a: *mut f32, lda: c_int) { ger(h, m, n, alpha, x, incx, y, incy, a, lda) }
    fn cublasSsyr_v2(h: cublasHandle_t, uplo: c_int, n: c_int, alpha: *const f32, x: *const f32, incx: c_int, a: *mut f32, lda: c_int) { syr(h, uplo, n, alpha, x, incx, x, incx, a, lda, false) }
    fn cublasSsyr2_v2(h: cublasHandle_t, uplo: c_int, n: c_int, alpha: *const f32, x: *const f32, incx: c_int, y: *const f32, incy: c_int, a: *mut f32, lda: c_int) { syr(h, uplo, n, alpha, x, incx, y, incy, a, lda, true) }
    fn cublasSgemvStridedBatched(h: cublasHandle_t, t: c_int, m: c_int, n: c_int, alpha: *const f32, a: *const f32, lda: c_int, sa: i64, x: *const f32, incx: c_int, sx: i64, beta: *const f32, y: *mut f32, incy: c_int, sy: i64, batch: c_int) { gemv_sb(h, t, m, n, alpha, a, lda, sa, x, incx, sx, beta, y, incy, sy, batch) }
    fn cublasSgemvBatched(h: cublasHandle_t, t: c_int, m: c_int, n: c_int, alpha: *const f32, a: *const *const f32, lda: c_int, x: *const *const f32, incx: c_int, beta: *const f32, y: *const *mut f32, incy: c_int, batch: c_int) { for i in 0..batch.max(0) { gemv_sb(h, t, m, n, alpha, ptrs(a as _, i)? as *const f32, lda, 0, ptrs(x as _, i)? as *const f32, incx, 0, beta, ptrs(y as _, i)? as *mut f32, incy, 0, 1)?; } if batch < 0 { return Err(CUBLAS_STATUS_INVALID_VALUE); } Ok(()) }
    fn cublasSsymv_v2(h: cublasHandle_t, uplo: c_int, n: c_int, alpha: *const f32, a: *const f32, lda: c_int, x: *const f32, incx: c_int, beta: *const f32, y: *mut f32, incy: c_int) { symv(h, G_FULL, uplo, n, 0, alpha, a, lda, x, incx, beta, y, incy) }
    fn cublasSsbmv_v2(h: cublasHandle_t, uplo: c_int, n: c_int, k: c_int, alpha: *const f32, a: *const f32, lda: c_int, x: *const f32, incx: c_int, beta: *const f32, y: *mut f32, incy: c_int) { symv(h, G_BAND, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy) }
    fn cublasSspmv_v2(h: cublasHandle_t, uplo: c_int, n: c_int, alpha: *const f32, ap: *const f32, x: *const f32, incx: c_int, beta: *const f32, y: *mut f32, incy: c_int) { symv(h, G_PACKED, uplo, n, 0, alpha, ap, 1, x, incx, beta, y, incy) }
    fn cublasSgbmv_v2(h: cublasHandle_t, t: c_int, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: *const f32, a: *const f32, lda: c_int, x: *const f32, incx: c_int, beta: *const f32, y: *mut f32, incy: c_int) { gbmv(h, t, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy) }
    fn cublasStrmv_v2(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: c_int, a: *const f32, lda: c_int, x: *mut f32, incx: c_int) { trmv(h, G_FULL, uplo, t, diag, n, 0, a, lda, x, incx) }
    fn cublasStbmv_v2(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: c_int, k: c_int, a: *const f32, lda: c_int, x: *mut f32, incx: c_int) { trmv(h, G_BAND, uplo, t, diag, n, k, a, lda, x, incx) }
    fn cublasStpmv_v2(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: c_int, ap: *const f32, x: *mut f32, incx: c_int) { trmv(h, G_PACKED, uplo, t, diag, n, 0, ap, 1, x, incx) }
    fn cublasStrsv_v2(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: c_int, a: *const f32, lda: c_int, x: *mut f32, incx: c_int) { trsv(h, G_FULL, uplo, t, diag, n, 0, a, lda, x, incx) }
    fn cublasStbsv_v2(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: c_int, k: c_int, a: *const f32, lda: c_int, x: *mut f32, incx: c_int) { trsv(h, G_BAND, uplo, t, diag, n, k, a, lda, x, incx) }
    fn cublasStpsv_v2(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: c_int, ap: *const f32, x: *mut f32, incx: c_int) { trsv(h, G_PACKED, uplo, t, diag, n, 0, ap, 1, x, incx) }
    fn cublasSsymm_v2(h: cublasHandle_t, side: c_int, uplo: c_int, m: c_int, n: c_int, alpha: *const f32, a: *const f32, lda: c_int, b: *const f32, ldb: c_int, beta: *const f32, c: *mut f32, ldc: c_int) { symm(h, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc) }
    fn cublasSsyrk_v2(h: cublasHandle_t, uplo: c_int, t: c_int, n: c_int, k: c_int, alpha: *const f32, a: *const f32, lda: c_int, beta: *const f32, c: *mut f32, ldc: c_int) { syrk(h, uplo, t, n, k, alpha, a, lda, a, lda, beta, c, ldc, false) }
    fn cublasSsyr2k_v2(h: cublasHandle_t, uplo: c_int, t: c_int, n: c_int, k: c_int, alpha: *const f32, a: *const f32, lda: c_int, b: *const f32, ldb: c_int, beta: *const f32, c: *mut f32, ldc: c_int) { syrk(h, uplo, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc, true) }
    fn cublasSsyrkx(h: cublasHandle_t, uplo: c_int, t: c_int, n: c_int, k: c_int, alpha: *const f32, a: *const f32, lda: c_int, b: *const f32, ldb: c_int, beta: *const f32, c: *mut f32, ldc: c_int) { syrk(h, uplo, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc, false) }
    fn cublasStrmm_v2(h: cublasHandle_t, side: c_int, uplo: c_int, t: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const f32, a: *const f32, lda: c_int, b: *const f32, ldb: c_int, c: *mut f32, ldc: c_int) { trmm(h, side, uplo, t, diag, m, n, alpha, a, lda, b, ldb, c, ldc) }
    fn cublasStrsm_v2(h: cublasHandle_t, side: c_int, uplo: c_int, t: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const f32, a: *const f32, lda: c_int, b: *mut f32, ldb: c_int) { trsm(h, side, uplo, t, diag, m, n, alpha, a, lda, b, ldb) }
    fn cublasStrsmBatched(h: cublasHandle_t, side: c_int, uplo: c_int, t: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const f32, a: *const *const f32, lda: c_int, b: *const *mut f32, ldb: c_int, batch: c_int) { if batch < 0 { return Err(CUBLAS_STATUS_INVALID_VALUE); } for i in 0..batch { trsm(h, side, uplo, t, diag, m, n, alpha, ptrs(a as _, i)? as *const f32, lda, ptrs(b as _, i)? as *mut f32, ldb)?; } Ok(()) }
    fn cublasSgeam(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, alpha: *const f32, a: *const f32, lda: c_int, beta: *const f32, b: *const f32, ldb: c_int, c: *mut f32, ldc: c_int) { geam(h, ta, tb, m, n, alpha, a, lda, beta, b, ldb, c, ldc) }
    fn cublasSdgmm(h: cublasHandle_t, side: c_int, m: c_int, n: c_int, a: *const f32, lda: c_int, x: *const f32, incx: c_int, c: *mut f32, ldc: c_int) { dgmm(h, side, m, n, a, lda, x, incx, c, ldc) }
    fn cublasDrot_v2(h: cublasHandle_t, n: c_int, x: *mut f64, incx: c_int, y: *mut f64, incy: c_int, c: *const f64, s: *const f64) { check(h)?; let (c, s) = (scalar(h, c)?, scalar(h, s)?); rot(h, n, x, incx, y, incy, [c, -s, s, c]) }
    fn cublasDrotg_v2(h: cublasHandle_t, a: *mut f64, b: *mut f64, c: *mut f64, s: *mut f64) { rotg(h, a, b, c, s) }
    fn cublasDrotm_v2(h: cublasHandle_t, n: c_int, x: *mut f64, incx: c_int, y: *mut f64, incy: c_int, param: *const f64) { rotm(h, n, x, incx, y, incy, param) }
    fn cublasDrotmg_v2(h: cublasHandle_t, d1: *mut f64, d2: *mut f64, x1: *mut f64, y1: *const f64, param: *mut f64) { rotmg(h, d1, d2, x1, y1, param) }
    fn cublasDger_v2(h: cublasHandle_t, m: c_int, n: c_int, alpha: *const f64, x: *const f64, incx: c_int, y: *const f64, incy: c_int, a: *mut f64, lda: c_int) { ger(h, m, n, alpha, x, incx, y, incy, a, lda) }
    fn cublasDsyr_v2(h: cublasHandle_t, uplo: c_int, n: c_int, alpha: *const f64, x: *const f64, incx: c_int, a: *mut f64, lda: c_int) { syr(h, uplo, n, alpha, x, incx, x, incx, a, lda, false) }
    fn cublasDsyr2_v2(h: cublasHandle_t, uplo: c_int, n: c_int, alpha: *const f64, x: *const f64, incx: c_int, y: *const f64, incy: c_int, a: *mut f64, lda: c_int) { syr(h, uplo, n, alpha, x, incx, y, incy, a, lda, true) }
    fn cublasDgemvStridedBatched(h: cublasHandle_t, t: c_int, m: c_int, n: c_int, alpha: *const f64, a: *const f64, lda: c_int, sa: i64, x: *const f64, incx: c_int, sx: i64, beta: *const f64, y: *mut f64, incy: c_int, sy: i64, batch: c_int) { gemv_sb(h, t, m, n, alpha, a, lda, sa, x, incx, sx, beta, y, incy, sy, batch) }
    fn cublasDgemvBatched(h: cublasHandle_t, t: c_int, m: c_int, n: c_int, alpha: *const f64, a: *const *const f64, lda: c_int, x: *const *const f64, incx: c_int, beta: *const f64, y: *const *mut f64, incy: c_int, batch: c_int) { for i in 0..batch.max(0) { gemv_sb(h, t, m, n, alpha, ptrs(a as _, i)? as *const f64, lda, 0, ptrs(x as _, i)? as *const f64, incx, 0, beta, ptrs(y as _, i)? as *mut f64, incy, 0, 1)?; } if batch < 0 { return Err(CUBLAS_STATUS_INVALID_VALUE); } Ok(()) }
    fn cublasDsymv_v2(h: cublasHandle_t, uplo: c_int, n: c_int, alpha: *const f64, a: *const f64, lda: c_int, x: *const f64, incx: c_int, beta: *const f64, y: *mut f64, incy: c_int) { symv(h, G_FULL, uplo, n, 0, alpha, a, lda, x, incx, beta, y, incy) }
    fn cublasDsbmv_v2(h: cublasHandle_t, uplo: c_int, n: c_int, k: c_int, alpha: *const f64, a: *const f64, lda: c_int, x: *const f64, incx: c_int, beta: *const f64, y: *mut f64, incy: c_int) { symv(h, G_BAND, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy) }
    fn cublasDspmv_v2(h: cublasHandle_t, uplo: c_int, n: c_int, alpha: *const f64, ap: *const f64, x: *const f64, incx: c_int, beta: *const f64, y: *mut f64, incy: c_int) { symv(h, G_PACKED, uplo, n, 0, alpha, ap, 1, x, incx, beta, y, incy) }
    fn cublasDgbmv_v2(h: cublasHandle_t, t: c_int, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: *const f64, a: *const f64, lda: c_int, x: *const f64, incx: c_int, beta: *const f64, y: *mut f64, incy: c_int) { gbmv(h, t, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy) }
    fn cublasDtrmv_v2(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: c_int, a: *const f64, lda: c_int, x: *mut f64, incx: c_int) { trmv(h, G_FULL, uplo, t, diag, n, 0, a, lda, x, incx) }
    fn cublasDtbmv_v2(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: c_int, k: c_int, a: *const f64, lda: c_int, x: *mut f64, incx: c_int) { trmv(h, G_BAND, uplo, t, diag, n, k, a, lda, x, incx) }
    fn cublasDtpmv_v2(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: c_int, ap: *const f64, x: *mut f64, incx: c_int) { trmv(h, G_PACKED, uplo, t, diag, n, 0, ap, 1, x, incx) }
    fn cublasDtrsv_v2(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: c_int, a: *const f64, lda: c_int, x: *mut f64, incx: c_int) { trsv(h, G_FULL, uplo, t, diag, n, 0, a, lda, x, incx) }
    fn cublasDtbsv_v2(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: c_int, k: c_int, a: *const f64, lda: c_int, x: *mut f64, incx: c_int) { trsv(h, G_BAND, uplo, t, diag, n, k, a, lda, x, incx) }
    fn cublasDtpsv_v2(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: c_int, ap: *const f64, x: *mut f64, incx: c_int) { trsv(h, G_PACKED, uplo, t, diag, n, 0, ap, 1, x, incx) }
    fn cublasDsymm_v2(h: cublasHandle_t, side: c_int, uplo: c_int, m: c_int, n: c_int, alpha: *const f64, a: *const f64, lda: c_int, b: *const f64, ldb: c_int, beta: *const f64, c: *mut f64, ldc: c_int) { symm(h, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc) }
    fn cublasDsyrk_v2(h: cublasHandle_t, uplo: c_int, t: c_int, n: c_int, k: c_int, alpha: *const f64, a: *const f64, lda: c_int, beta: *const f64, c: *mut f64, ldc: c_int) { syrk(h, uplo, t, n, k, alpha, a, lda, a, lda, beta, c, ldc, false) }
    fn cublasDsyr2k_v2(h: cublasHandle_t, uplo: c_int, t: c_int, n: c_int, k: c_int, alpha: *const f64, a: *const f64, lda: c_int, b: *const f64, ldb: c_int, beta: *const f64, c: *mut f64, ldc: c_int) { syrk(h, uplo, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc, true) }
    fn cublasDsyrkx(h: cublasHandle_t, uplo: c_int, t: c_int, n: c_int, k: c_int, alpha: *const f64, a: *const f64, lda: c_int, b: *const f64, ldb: c_int, beta: *const f64, c: *mut f64, ldc: c_int) { syrk(h, uplo, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc, false) }
    fn cublasDtrmm_v2(h: cublasHandle_t, side: c_int, uplo: c_int, t: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const f64, a: *const f64, lda: c_int, b: *const f64, ldb: c_int, c: *mut f64, ldc: c_int) { trmm(h, side, uplo, t, diag, m, n, alpha, a, lda, b, ldb, c, ldc) }
    fn cublasDtrsm_v2(h: cublasHandle_t, side: c_int, uplo: c_int, t: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const f64, a: *const f64, lda: c_int, b: *mut f64, ldb: c_int) { trsm(h, side, uplo, t, diag, m, n, alpha, a, lda, b, ldb) }
    fn cublasDtrsmBatched(h: cublasHandle_t, side: c_int, uplo: c_int, t: c_int, diag: c_int, m: c_int, n: c_int, alpha: *const f64, a: *const *const f64, lda: c_int, b: *const *mut f64, ldb: c_int, batch: c_int) { if batch < 0 { return Err(CUBLAS_STATUS_INVALID_VALUE); } for i in 0..batch { trsm(h, side, uplo, t, diag, m, n, alpha, ptrs(a as _, i)? as *const f64, lda, ptrs(b as _, i)? as *mut f64, ldb)?; } Ok(()) }
    fn cublasDgeam(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, alpha: *const f64, a: *const f64, lda: c_int, beta: *const f64, b: *const f64, ldb: c_int, c: *mut f64, ldc: c_int) { geam(h, ta, tb, m, n, alpha, a, lda, beta, b, ldb, c, ldc) }
    fn cublasDdgmm(h: cublasHandle_t, side: c_int, m: c_int, n: c_int, a: *const f64, lda: c_int, x: *const f64, incx: c_int, c: *mut f64, ldc: c_int) { dgmm(h, side, m, n, a, lda, x, incx, c, ldc) }
}

entry! {
    fn cublasCaxpy_v2(h: cublasHandle_t, n: c_int, alpha: *const Cx<f32>, x: *const Cx<f32>, incx: c_int, y: *mut Cx<f32>, incy: c_int) { check(h)?; caxpy::<f32>(h, n, scalar(h, alpha)?, x as u64, incx, y as u64, incy, false) }
    fn cublasCscal_v2(h: cublasHandle_t, n: c_int, alpha: *const Cx<f32>, x: *mut Cx<f32>, incx: c_int) { check(h)?; caxpy::<f32>(h, n, scalar(h, alpha)?, x as u64, incx, x as u64, incx, true) }
    fn cublasCsscal_v2(h: cublasHandle_t, n: c_int, alpha: *const f32, x: *mut Cx<f32>, incx: c_int) { check(h)?; caxpy::<f32>(h, n, [scalar(h, alpha)?, 0.0], x as u64, incx, x as u64, incx, true) }
    fn cublasCcopy_v2(h: cublasHandle_t, n: c_int, x: *const Cx<f32>, incx: c_int, y: *mut Cx<f32>, incy: c_int) { ccopy::<f32>(h, n, x as u64, incx, y as u64, incy, false) }
    fn cublasCswap_v2(h: cublasHandle_t, n: c_int, x: *mut Cx<f32>, incx: c_int, y: *mut Cx<f32>, incy: c_int) { ccopy::<f32>(h, n, x as u64, incx, y as u64, incy, true) }
    fn cublasCdotu_v2(h: cublasHandle_t, n: c_int, x: *const Cx<f32>, incx: c_int, y: *const Cx<f32>, incy: c_int, r: *mut Cx<f32>) { cdot::<f32>(h, n, x as u64, incx, y as u64, incy, r, false) }
    fn cublasCdotc_v2(h: cublasHandle_t, n: c_int, x: *const Cx<f32>, incx: c_int, y: *const Cx<f32>, incy: c_int, r: *mut Cx<f32>) { cdot::<f32>(h, n, x as u64, incx, y as u64, incy, r, true) }
    fn cublasScnrm2_v2(h: cublasHandle_t, n: c_int, x: *const Cx<f32>, incx: c_int, r: *mut f32) { cnorm::<f32>(h, n, x as u64, incx, r, 1) }
    fn cublasScasum_v2(h: cublasHandle_t, n: c_int, x: *const Cx<f32>, incx: c_int, r: *mut f32) { cnorm::<f32>(h, n, x as u64, incx, r, 2) }
    fn cublasCsrot_v2(h: cublasHandle_t, n: c_int, x: *mut Cx<f32>, incx: c_int, y: *mut Cx<f32>, incy: c_int, c: *const f32, s: *const f32) { check(h)?; let (c, s) = (scalar(h, c)?, scalar(h, s)?); for part in 0..2 { crot::<f32>(n, x as u64, incx, y as u64, incy, part, [c, -s, s, c])?; } Ok(()) }
    fn cublasCgemv_v2(h: cublasHandle_t, t: c_int, m: c_int, n: c_int, alpha: *const Cx<f32>, a: *const Cx<f32>, lda: c_int, x: *const Cx<f32>, incx: c_int, beta: *const Cx<f32>, y: *mut Cx<f32>, incy: c_int) { cgemv(h, t, m, n, alpha, a as u64, lda, x as u64, incx, beta, y as u64, incy) }
    fn cublasCgemm_v2(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const Cx<f32>, a: *const Cx<f32>, lda: c_int, b: *const Cx<f32>, ldb: c_int, beta: *const Cx<f32>, c: *mut Cx<f32>, ldc: c_int) { cgemm(h, ta, tb, m, n, k, alpha, a as u64, lda, 0, b as u64, ldb, 0, beta, c as u64, ldc, 0, 1) }
    fn cublasCgemm3m(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const Cx<f32>, a: *const Cx<f32>, lda: c_int, b: *const Cx<f32>, ldb: c_int, beta: *const Cx<f32>, c: *mut Cx<f32>, ldc: c_int) { cgemm(h, ta, tb, m, n, k, alpha, a as u64, lda, 0, b as u64, ldb, 0, beta, c as u64, ldc, 0, 1) }
    fn cublasCgemmStridedBatched(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const Cx<f32>, a: *const Cx<f32>, lda: c_int, sa: i64, b: *const Cx<f32>, ldb: c_int, sb: i64, beta: *const Cx<f32>, c: *mut Cx<f32>, ldc: c_int, sc: i64, batch: c_int) { cgemm(h, ta, tb, m, n, k, alpha, a as u64, lda, sa, b as u64, ldb, sb, beta, c as u64, ldc, sc, batch) }
    fn cublasCgemmBatched(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const Cx<f32>, a: *const *const Cx<f32>, lda: c_int, b: *const *const Cx<f32>, ldb: c_int, beta: *const Cx<f32>, c: *const *mut Cx<f32>, ldc: c_int, batch: c_int) { if batch < 0 { return Err(CUBLAS_STATUS_INVALID_VALUE); } for i in 0..batch { cgemm(h, ta, tb, m, n, k, alpha, ptrs(a as _, i)?, lda, 0, ptrs(b as _, i)?, ldb, 0, beta, ptrs(c as _, i)?, ldc, 0, 1)?; } Ok(()) }
    fn cublasZaxpy_v2(h: cublasHandle_t, n: c_int, alpha: *const Cx<f64>, x: *const Cx<f64>, incx: c_int, y: *mut Cx<f64>, incy: c_int) { check(h)?; caxpy::<f64>(h, n, scalar(h, alpha)?, x as u64, incx, y as u64, incy, false) }
    fn cublasZscal_v2(h: cublasHandle_t, n: c_int, alpha: *const Cx<f64>, x: *mut Cx<f64>, incx: c_int) { check(h)?; caxpy::<f64>(h, n, scalar(h, alpha)?, x as u64, incx, x as u64, incx, true) }
    fn cublasZdscal_v2(h: cublasHandle_t, n: c_int, alpha: *const f64, x: *mut Cx<f64>, incx: c_int) { check(h)?; caxpy::<f64>(h, n, [scalar(h, alpha)?, 0.0], x as u64, incx, x as u64, incx, true) }
    fn cublasZcopy_v2(h: cublasHandle_t, n: c_int, x: *const Cx<f64>, incx: c_int, y: *mut Cx<f64>, incy: c_int) { ccopy::<f64>(h, n, x as u64, incx, y as u64, incy, false) }
    fn cublasZswap_v2(h: cublasHandle_t, n: c_int, x: *mut Cx<f64>, incx: c_int, y: *mut Cx<f64>, incy: c_int) { ccopy::<f64>(h, n, x as u64, incx, y as u64, incy, true) }
    fn cublasZdotu_v2(h: cublasHandle_t, n: c_int, x: *const Cx<f64>, incx: c_int, y: *const Cx<f64>, incy: c_int, r: *mut Cx<f64>) { cdot::<f64>(h, n, x as u64, incx, y as u64, incy, r, false) }
    fn cublasZdotc_v2(h: cublasHandle_t, n: c_int, x: *const Cx<f64>, incx: c_int, y: *const Cx<f64>, incy: c_int, r: *mut Cx<f64>) { cdot::<f64>(h, n, x as u64, incx, y as u64, incy, r, true) }
    fn cublasDznrm2_v2(h: cublasHandle_t, n: c_int, x: *const Cx<f64>, incx: c_int, r: *mut f64) { cnorm::<f64>(h, n, x as u64, incx, r, 1) }
    fn cublasDzasum_v2(h: cublasHandle_t, n: c_int, x: *const Cx<f64>, incx: c_int, r: *mut f64) { cnorm::<f64>(h, n, x as u64, incx, r, 2) }
    fn cublasZdrot_v2(h: cublasHandle_t, n: c_int, x: *mut Cx<f64>, incx: c_int, y: *mut Cx<f64>, incy: c_int, c: *const f64, s: *const f64) { check(h)?; let (c, s) = (scalar(h, c)?, scalar(h, s)?); for part in 0..2 { crot::<f64>(n, x as u64, incx, y as u64, incy, part, [c, -s, s, c])?; } Ok(()) }
    fn cublasZgemv_v2(h: cublasHandle_t, t: c_int, m: c_int, n: c_int, alpha: *const Cx<f64>, a: *const Cx<f64>, lda: c_int, x: *const Cx<f64>, incx: c_int, beta: *const Cx<f64>, y: *mut Cx<f64>, incy: c_int) { cgemv(h, t, m, n, alpha, a as u64, lda, x as u64, incx, beta, y as u64, incy) }
    fn cublasZgemm_v2(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const Cx<f64>, a: *const Cx<f64>, lda: c_int, b: *const Cx<f64>, ldb: c_int, beta: *const Cx<f64>, c: *mut Cx<f64>, ldc: c_int) { cgemm(h, ta, tb, m, n, k, alpha, a as u64, lda, 0, b as u64, ldb, 0, beta, c as u64, ldc, 0, 1) }
    fn cublasZgemm3m(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const Cx<f64>, a: *const Cx<f64>, lda: c_int, b: *const Cx<f64>, ldb: c_int, beta: *const Cx<f64>, c: *mut Cx<f64>, ldc: c_int) { cgemm(h, ta, tb, m, n, k, alpha, a as u64, lda, 0, b as u64, ldb, 0, beta, c as u64, ldc, 0, 1) }
    fn cublasZgemmStridedBatched(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const Cx<f64>, a: *const Cx<f64>, lda: c_int, sa: i64, b: *const Cx<f64>, ldb: c_int, sb: i64, beta: *const Cx<f64>, c: *mut Cx<f64>, ldc: c_int, sc: i64, batch: c_int) { cgemm(h, ta, tb, m, n, k, alpha, a as u64, lda, sa, b as u64, ldb, sb, beta, c as u64, ldc, sc, batch) }
    fn cublasZgemmBatched(h: cublasHandle_t, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: *const Cx<f64>, a: *const *const Cx<f64>, lda: c_int, b: *const *const Cx<f64>, ldb: c_int, beta: *const Cx<f64>, c: *const *mut Cx<f64>, ldc: c_int, batch: c_int) { if batch < 0 { return Err(CUBLAS_STATUS_INVALID_VALUE); } for i in 0..batch { cgemm(h, ta, tb, m, n, k, alpha, ptrs(a as _, i)?, lda, 0, ptrs(b as _, i)?, ldb, 0, beta, ptrs(c as _, i)?, ldc, 0, 1)?; } Ok(()) }
}

/// A dimension of a _64 entry point (beyond int32: not supported)
fn n32(v: i64) -> Result<c_int, cublasStatus_t> {
    c_int::try_from(v).map_err(|_| fail(format!("dimension {v} beyond 2^31"), CUBLAS_STATUS_NOT_SUPPORTED))
}

fn ok(st: cublasStatus_t) -> R {
    if st == CUBLAS_STATUS_SUCCESS { Ok(()) } else { Err(st) }
}

// the _64 entry points: the int32 routines for dimensions that fit (larger ones are not supported)
entry! {
    fn cublasSetVectorAsync_64(n: i64, elem: c_int, x: *const c_void, incx: i64, y: *mut c_void, incy: i64, _s: *mut c_void) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasSetVectorAsync(n, elem, x, incx, y, incy, _s)) }
    fn cublasGetVectorAsync_64(n: i64, elem: c_int, x: *const c_void, incx: i64, y: *mut c_void, incy: i64, _s: *mut c_void) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasGetVectorAsync(n, elem, x, incx, y, incy, _s)) }
    fn cublasSetMatrixAsync_64(rows: i64, cols: i64, elem: c_int, a: *const c_void, lda: i64, b: *mut c_void, ldb: i64, _s: *mut c_void) { let rows = n32(rows)?; let cols = n32(cols)?; let lda = n32(lda)?; let ldb = n32(ldb)?; ok(cublasSetMatrixAsync(rows, cols, elem, a, lda, b, ldb, _s)) }
    fn cublasGetMatrixAsync_64(rows: i64, cols: i64, elem: c_int, a: *const c_void, lda: i64, b: *mut c_void, ldb: i64, _s: *mut c_void) { let rows = n32(rows)?; let cols = n32(cols)?; let lda = n32(lda)?; let ldb = n32(ldb)?; ok(cublasGetMatrixAsync(rows, cols, elem, a, lda, b, ldb, _s)) }
    fn cublasSetVector_64(n: i64, elem: c_int, x: *const c_void, incx: i64, y: *mut c_void, incy: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasSetVector(n, elem, x, incx, y, incy)) }
    fn cublasGetVector_64(n: i64, elem: c_int, x: *const c_void, incx: i64, y: *mut c_void, incy: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasGetVector(n, elem, x, incx, y, incy)) }
    fn cublasSetMatrix_64(rows: i64, cols: i64, elem: c_int, a: *const c_void, lda: i64, b: *mut c_void, ldb: i64) { let rows = n32(rows)?; let cols = n32(cols)?; let lda = n32(lda)?; let ldb = n32(ldb)?; ok(cublasSetMatrix(rows, cols, elem, a, lda, b, ldb)) }
    fn cublasGetMatrix_64(rows: i64, cols: i64, elem: c_int, a: *const c_void, lda: i64, b: *mut c_void, ldb: i64) { let rows = n32(rows)?; let cols = n32(cols)?; let lda = n32(lda)?; let ldb = n32(ldb)?; ok(cublasGetMatrix(rows, cols, elem, a, lda, b, ldb)) }
    fn cublasSaxpy_v2_64(h: cublasHandle_t, n: i64, alpha: *const f32, x: *const f32, incx: i64, y: *mut f32, incy: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasSaxpy_v2(h, n, alpha, x, incx, y, incy)) }
    fn cublasDaxpy_v2_64(h: cublasHandle_t, n: i64, alpha: *const f64, x: *const f64, incx: i64, y: *mut f64, incy: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasDaxpy_v2(h, n, alpha, x, incx, y, incy)) }
    fn cublasSscal_v2_64(h: cublasHandle_t, n: i64, alpha: *const f32, x: *mut f32, incx: i64) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasSscal_v2(h, n, alpha, x, incx)) }
    fn cublasDscal_v2_64(h: cublasHandle_t, n: i64, alpha: *const f64, x: *mut f64, incx: i64) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasDscal_v2(h, n, alpha, x, incx)) }
    fn cublasScopy_v2_64(h: cublasHandle_t, n: i64, x: *const f32, incx: i64, y: *mut f32, incy: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasScopy_v2(h, n, x, incx, y, incy)) }
    fn cublasDcopy_v2_64(h: cublasHandle_t, n: i64, x: *const f64, incx: i64, y: *mut f64, incy: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasDcopy_v2(h, n, x, incx, y, incy)) }
    fn cublasSswap_v2_64(h: cublasHandle_t, n: i64, x: *mut f32, incx: i64, y: *mut f32, incy: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasSswap_v2(h, n, x, incx, y, incy)) }
    fn cublasDswap_v2_64(h: cublasHandle_t, n: i64, x: *mut f64, incx: i64, y: *mut f64, incy: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasDswap_v2(h, n, x, incx, y, incy)) }
    fn cublasSdot_v2_64(h: cublasHandle_t, n: i64, x: *const f32, incx: i64, y: *const f32, incy: i64, r: *mut f32) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasSdot_v2(h, n, x, incx, y, incy, r)) }
    fn cublasDdot_v2_64(h: cublasHandle_t, n: i64, x: *const f64, incx: i64, y: *const f64, incy: i64, r: *mut f64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasDdot_v2(h, n, x, incx, y, incy, r)) }
    fn cublasSnrm2_v2_64(h: cublasHandle_t, n: i64, x: *const f32, incx: i64, r: *mut f32) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasSnrm2_v2(h, n, x, incx, r)) }
    fn cublasDnrm2_v2_64(h: cublasHandle_t, n: i64, x: *const f64, incx: i64, r: *mut f64) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasDnrm2_v2(h, n, x, incx, r)) }
    fn cublasSasum_v2_64(h: cublasHandle_t, n: i64, x: *const f32, incx: i64, r: *mut f32) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasSasum_v2(h, n, x, incx, r)) }
    fn cublasDasum_v2_64(h: cublasHandle_t, n: i64, x: *const f64, incx: i64, r: *mut f64) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasDasum_v2(h, n, x, incx, r)) }
    fn cublasIsamax_v2_64(h: cublasHandle_t, n: i64, x: *const f32, incx: i64, r: *mut i64) { let n = n32(n)?; let incx = n32(incx)?; check(h)?; let mode = (*h).pointer_mode; let mut r32: c_int = 0; (*h).pointer_mode = 0; let st = cublasIsamax_v2(h, n, x, incx, &mut r32 as *mut c_int); (*h).pointer_mode = mode; ok(st)?; result(h, r, r32 as i64) }
    fn cublasIdamax_v2_64(h: cublasHandle_t, n: i64, x: *const f64, incx: i64, r: *mut i64) { let n = n32(n)?; let incx = n32(incx)?; check(h)?; let mode = (*h).pointer_mode; let mut r32: c_int = 0; (*h).pointer_mode = 0; let st = cublasIdamax_v2(h, n, x, incx, &mut r32 as *mut c_int); (*h).pointer_mode = mode; ok(st)?; result(h, r, r32 as i64) }
    fn cublasIsamin_v2_64(h: cublasHandle_t, n: i64, x: *const f32, incx: i64, r: *mut i64) { let n = n32(n)?; let incx = n32(incx)?; check(h)?; let mode = (*h).pointer_mode; let mut r32: c_int = 0; (*h).pointer_mode = 0; let st = cublasIsamin_v2(h, n, x, incx, &mut r32 as *mut c_int); (*h).pointer_mode = mode; ok(st)?; result(h, r, r32 as i64) }
    fn cublasIdamin_v2_64(h: cublasHandle_t, n: i64, x: *const f64, incx: i64, r: *mut i64) { let n = n32(n)?; let incx = n32(incx)?; check(h)?; let mode = (*h).pointer_mode; let mut r32: c_int = 0; (*h).pointer_mode = 0; let st = cublasIdamin_v2(h, n, x, incx, &mut r32 as *mut c_int); (*h).pointer_mode = mode; ok(st)?; result(h, r, r32 as i64) }
    fn cublasSgemv_v2_64(h: cublasHandle_t, t: c_int, m: i64, n: i64, alpha: *const f32, a: *const f32, lda: i64, x: *const f32, incx: i64, beta: *const f32, y: *mut f32, incy: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasSgemv_v2(h, t, m, n, alpha, a, lda, x, incx, beta, y, incy)) }
    fn cublasDgemv_v2_64(h: cublasHandle_t, t: c_int, m: i64, n: i64, alpha: *const f64, a: *const f64, lda: i64, x: *const f64, incx: i64, beta: *const f64, y: *mut f64, incy: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasDgemv_v2(h, t, m, n, alpha, a, lda, x, incx, beta, y, incy)) }
    fn cublasSgemm_v2_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const f32, a: *const f32, lda: i64, b: *const f32, ldb: i64, beta: *const f32, c: *mut f32, ldc: i64) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasSgemm_v2(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) }
    fn cublasDgemm_v2_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const f64, a: *const f64, lda: i64, b: *const f64, ldb: i64, beta: *const f64, c: *mut f64, ldc: i64) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasDgemm_v2(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) }
    fn cublasSgemmStridedBatched_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const f32, a: *const f32, lda: i64, sa: i64, b: *const f32, ldb: i64, sb: i64, beta: *const f32, c: *mut f32, ldc: i64, sc: i64, batch: i64) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; let batch = n32(batch)?; ok(cublasSgemmStridedBatched(h, ta, tb, m, n, k, alpha, a, lda, sa, b, ldb, sb, beta, c, ldc, sc, batch)) }
    fn cublasDgemmStridedBatched_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const f64, a: *const f64, lda: i64, sa: i64, b: *const f64, ldb: i64, sb: i64, beta: *const f64, c: *mut f64, ldc: i64, sc: i64, batch: i64) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; let batch = n32(batch)?; ok(cublasDgemmStridedBatched(h, ta, tb, m, n, k, alpha, a, lda, sa, b, ldb, sb, beta, c, ldc, sc, batch)) }
    fn cublasSgemmBatched_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const f32, a: *const *const f32, lda: i64, b: *const *const f32, ldb: i64, beta: *const f32, c: *const *mut f32, ldc: i64, batch: i64) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; let batch = n32(batch)?; ok(cublasSgemmBatched(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch)) }
    fn cublasDgemmBatched_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const f64, a: *const *const f64, lda: i64, b: *const *const f64, ldb: i64, beta: *const f64, c: *const *mut f64, ldc: i64, batch: i64) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; let batch = n32(batch)?; ok(cublasDgemmBatched(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch)) }
    fn cublasHgemm_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const u16, a: *const u16, lda: i64, b: *const u16, ldb: i64, beta: *const u16, c: *mut u16, ldc: i64) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasHgemm(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) }
    fn cublasHgemmStridedBatched_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const u16, a: *const u16, lda: i64, sa: i64, b: *const u16, ldb: i64, sb: i64, beta: *const u16, c: *mut u16, ldc: i64, sc: i64, batch: i64) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; let batch = n32(batch)?; ok(cublasHgemmStridedBatched(h, ta, tb, m, n, k, alpha, a, lda, sa, b, ldb, sb, beta, c, ldc, sc, batch)) }
    fn cublasGemmEx_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const c_void, a: *const c_void, at: c_int, lda: i64, b: *const c_void, bt: c_int, ldb: i64, beta: *const c_void, c: *mut c_void, ct: c_int, ldc: i64, compute: c_int, _algo: c_int) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasGemmEx(h, ta, tb, m, n, k, alpha, a, at, lda, b, bt, ldb, beta, c, ct, ldc, compute, _algo)) }
    fn cublasGemmStridedBatchedEx_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const c_void, a: *const c_void, at: c_int, lda: i64, sa: i64, b: *const c_void, bt: c_int, ldb: i64, sb: i64, beta: *const c_void, c: *mut c_void, ct: c_int, ldc: i64, sc: i64, batch: i64, compute: c_int, _algo: c_int) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; let batch = n32(batch)?; ok(cublasGemmStridedBatchedEx(h, ta, tb, m, n, k, alpha, a, at, lda, sa, b, bt, ldb, sb, beta, c, ct, ldc, sc, batch, compute, _algo)) }
    fn cublasGemmBatchedEx_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const c_void, a: *const *const c_void, at: c_int, lda: i64, b: *const *const c_void, bt: c_int, ldb: i64, beta: *const c_void, c: *const *mut c_void, ct: c_int, ldc: i64, batch: i64, compute: c_int, _algo: c_int) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; let batch = n32(batch)?; ok(cublasGemmBatchedEx(h, ta, tb, m, n, k, alpha, a, at, lda, b, bt, ldb, beta, c, ct, ldc, batch, compute, _algo)) }
    fn cublasSrot_v2_64(h: cublasHandle_t, n: i64, x: *mut f32, incx: i64, y: *mut f32, incy: i64, c: *const f32, s: *const f32) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasSrot_v2(h, n, x, incx, y, incy, c, s)) }
    fn cublasSrotm_v2_64(h: cublasHandle_t, n: i64, x: *mut f32, incx: i64, y: *mut f32, incy: i64, param: *const f32) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasSrotm_v2(h, n, x, incx, y, incy, param)) }
    fn cublasSger_v2_64(h: cublasHandle_t, m: i64, n: i64, alpha: *const f32, x: *const f32, incx: i64, y: *const f32, incy: i64, a: *mut f32, lda: i64) { let m = n32(m)?; let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; let lda = n32(lda)?; ok(cublasSger_v2(h, m, n, alpha, x, incx, y, incy, a, lda)) }
    fn cublasSsyr_v2_64(h: cublasHandle_t, uplo: c_int, n: i64, alpha: *const f32, x: *const f32, incx: i64, a: *mut f32, lda: i64) { let n = n32(n)?; let incx = n32(incx)?; let lda = n32(lda)?; ok(cublasSsyr_v2(h, uplo, n, alpha, x, incx, a, lda)) }
    fn cublasSsyr2_v2_64(h: cublasHandle_t, uplo: c_int, n: i64, alpha: *const f32, x: *const f32, incx: i64, y: *const f32, incy: i64, a: *mut f32, lda: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; let lda = n32(lda)?; ok(cublasSsyr2_v2(h, uplo, n, alpha, x, incx, y, incy, a, lda)) }
    fn cublasSgemvStridedBatched_64(h: cublasHandle_t, t: c_int, m: i64, n: i64, alpha: *const f32, a: *const f32, lda: i64, sa: i64, x: *const f32, incx: i64, sx: i64, beta: *const f32, y: *mut f32, incy: i64, sy: i64, batch: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let incx = n32(incx)?; let incy = n32(incy)?; let batch = n32(batch)?; ok(cublasSgemvStridedBatched(h, t, m, n, alpha, a, lda, sa, x, incx, sx, beta, y, incy, sy, batch)) }
    fn cublasSgemvBatched_64(h: cublasHandle_t, t: c_int, m: i64, n: i64, alpha: *const f32, a: *const *const f32, lda: i64, x: *const *const f32, incx: i64, beta: *const f32, y: *const *mut f32, incy: i64, batch: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let incx = n32(incx)?; let incy = n32(incy)?; let batch = n32(batch)?; ok(cublasSgemvBatched(h, t, m, n, alpha, a, lda, x, incx, beta, y, incy, batch)) }
    fn cublasSsymv_v2_64(h: cublasHandle_t, uplo: c_int, n: i64, alpha: *const f32, a: *const f32, lda: i64, x: *const f32, incx: i64, beta: *const f32, y: *mut f32, incy: i64) { let n = n32(n)?; let lda = n32(lda)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasSsymv_v2(h, uplo, n, alpha, a, lda, x, incx, beta, y, incy)) }
    fn cublasSsbmv_v2_64(h: cublasHandle_t, uplo: c_int, n: i64, k: i64, alpha: *const f32, a: *const f32, lda: i64, x: *const f32, incx: i64, beta: *const f32, y: *mut f32, incy: i64) { let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasSsbmv_v2(h, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)) }
    fn cublasSspmv_v2_64(h: cublasHandle_t, uplo: c_int, n: i64, alpha: *const f32, ap: *const f32, x: *const f32, incx: i64, beta: *const f32, y: *mut f32, incy: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasSspmv_v2(h, uplo, n, alpha, ap, x, incx, beta, y, incy)) }
    fn cublasSgbmv_v2_64(h: cublasHandle_t, t: c_int, m: i64, n: i64, kl: i64, ku: i64, alpha: *const f32, a: *const f32, lda: i64, x: *const f32, incx: i64, beta: *const f32, y: *mut f32, incy: i64) { let m = n32(m)?; let n = n32(n)?; let kl = n32(kl)?; let ku = n32(ku)?; let lda = n32(lda)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasSgbmv_v2(h, t, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)) }
    fn cublasStrmv_v2_64(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: i64, a: *const f32, lda: i64, x: *mut f32, incx: i64) { let n = n32(n)?; let lda = n32(lda)?; let incx = n32(incx)?; ok(cublasStrmv_v2(h, uplo, t, diag, n, a, lda, x, incx)) }
    fn cublasStbmv_v2_64(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: i64, k: i64, a: *const f32, lda: i64, x: *mut f32, incx: i64) { let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let incx = n32(incx)?; ok(cublasStbmv_v2(h, uplo, t, diag, n, k, a, lda, x, incx)) }
    fn cublasStpmv_v2_64(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: i64, ap: *const f32, x: *mut f32, incx: i64) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasStpmv_v2(h, uplo, t, diag, n, ap, x, incx)) }
    fn cublasStrsv_v2_64(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: i64, a: *const f32, lda: i64, x: *mut f32, incx: i64) { let n = n32(n)?; let lda = n32(lda)?; let incx = n32(incx)?; ok(cublasStrsv_v2(h, uplo, t, diag, n, a, lda, x, incx)) }
    fn cublasStbsv_v2_64(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: i64, k: i64, a: *const f32, lda: i64, x: *mut f32, incx: i64) { let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let incx = n32(incx)?; ok(cublasStbsv_v2(h, uplo, t, diag, n, k, a, lda, x, incx)) }
    fn cublasStpsv_v2_64(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: i64, ap: *const f32, x: *mut f32, incx: i64) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasStpsv_v2(h, uplo, t, diag, n, ap, x, incx)) }
    fn cublasSsymm_v2_64(h: cublasHandle_t, side: c_int, uplo: c_int, m: i64, n: i64, alpha: *const f32, a: *const f32, lda: i64, b: *const f32, ldb: i64, beta: *const f32, c: *mut f32, ldc: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasSsymm_v2(h, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)) }
    fn cublasSsyrk_v2_64(h: cublasHandle_t, uplo: c_int, t: c_int, n: i64, k: i64, alpha: *const f32, a: *const f32, lda: i64, beta: *const f32, c: *mut f32, ldc: i64) { let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldc = n32(ldc)?; ok(cublasSsyrk_v2(h, uplo, t, n, k, alpha, a, lda, beta, c, ldc)) }
    fn cublasSsyr2k_v2_64(h: cublasHandle_t, uplo: c_int, t: c_int, n: i64, k: i64, alpha: *const f32, a: *const f32, lda: i64, b: *const f32, ldb: i64, beta: *const f32, c: *mut f32, ldc: i64) { let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasSsyr2k_v2(h, uplo, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) }
    fn cublasSsyrkx_64(h: cublasHandle_t, uplo: c_int, t: c_int, n: i64, k: i64, alpha: *const f32, a: *const f32, lda: i64, b: *const f32, ldb: i64, beta: *const f32, c: *mut f32, ldc: i64) { let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasSsyrkx(h, uplo, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) }
    fn cublasStrmm_v2_64(h: cublasHandle_t, side: c_int, uplo: c_int, t: c_int, diag: c_int, m: i64, n: i64, alpha: *const f32, a: *const f32, lda: i64, b: *const f32, ldb: i64, c: *mut f32, ldc: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasStrmm_v2(h, side, uplo, t, diag, m, n, alpha, a, lda, b, ldb, c, ldc)) }
    fn cublasStrsm_v2_64(h: cublasHandle_t, side: c_int, uplo: c_int, t: c_int, diag: c_int, m: i64, n: i64, alpha: *const f32, a: *const f32, lda: i64, b: *mut f32, ldb: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let ldb = n32(ldb)?; ok(cublasStrsm_v2(h, side, uplo, t, diag, m, n, alpha, a, lda, b, ldb)) }
    fn cublasStrsmBatched_64(h: cublasHandle_t, side: c_int, uplo: c_int, t: c_int, diag: c_int, m: i64, n: i64, alpha: *const f32, a: *const *const f32, lda: i64, b: *const *mut f32, ldb: i64, batch: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let batch = n32(batch)?; ok(cublasStrsmBatched(h, side, uplo, t, diag, m, n, alpha, a, lda, b, ldb, batch)) }
    fn cublasSgeam_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, alpha: *const f32, a: *const f32, lda: i64, beta: *const f32, b: *const f32, ldb: i64, c: *mut f32, ldc: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasSgeam(h, ta, tb, m, n, alpha, a, lda, beta, b, ldb, c, ldc)) }
    fn cublasSdgmm_64(h: cublasHandle_t, side: c_int, m: i64, n: i64, a: *const f32, lda: i64, x: *const f32, incx: i64, c: *mut f32, ldc: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let incx = n32(incx)?; let ldc = n32(ldc)?; ok(cublasSdgmm(h, side, m, n, a, lda, x, incx, c, ldc)) }
    fn cublasDrot_v2_64(h: cublasHandle_t, n: i64, x: *mut f64, incx: i64, y: *mut f64, incy: i64, c: *const f64, s: *const f64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasDrot_v2(h, n, x, incx, y, incy, c, s)) }
    fn cublasDrotm_v2_64(h: cublasHandle_t, n: i64, x: *mut f64, incx: i64, y: *mut f64, incy: i64, param: *const f64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasDrotm_v2(h, n, x, incx, y, incy, param)) }
    fn cublasDger_v2_64(h: cublasHandle_t, m: i64, n: i64, alpha: *const f64, x: *const f64, incx: i64, y: *const f64, incy: i64, a: *mut f64, lda: i64) { let m = n32(m)?; let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; let lda = n32(lda)?; ok(cublasDger_v2(h, m, n, alpha, x, incx, y, incy, a, lda)) }
    fn cublasDsyr_v2_64(h: cublasHandle_t, uplo: c_int, n: i64, alpha: *const f64, x: *const f64, incx: i64, a: *mut f64, lda: i64) { let n = n32(n)?; let incx = n32(incx)?; let lda = n32(lda)?; ok(cublasDsyr_v2(h, uplo, n, alpha, x, incx, a, lda)) }
    fn cublasDsyr2_v2_64(h: cublasHandle_t, uplo: c_int, n: i64, alpha: *const f64, x: *const f64, incx: i64, y: *const f64, incy: i64, a: *mut f64, lda: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; let lda = n32(lda)?; ok(cublasDsyr2_v2(h, uplo, n, alpha, x, incx, y, incy, a, lda)) }
    fn cublasDgemvStridedBatched_64(h: cublasHandle_t, t: c_int, m: i64, n: i64, alpha: *const f64, a: *const f64, lda: i64, sa: i64, x: *const f64, incx: i64, sx: i64, beta: *const f64, y: *mut f64, incy: i64, sy: i64, batch: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let incx = n32(incx)?; let incy = n32(incy)?; let batch = n32(batch)?; ok(cublasDgemvStridedBatched(h, t, m, n, alpha, a, lda, sa, x, incx, sx, beta, y, incy, sy, batch)) }
    fn cublasDgemvBatched_64(h: cublasHandle_t, t: c_int, m: i64, n: i64, alpha: *const f64, a: *const *const f64, lda: i64, x: *const *const f64, incx: i64, beta: *const f64, y: *const *mut f64, incy: i64, batch: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let incx = n32(incx)?; let incy = n32(incy)?; let batch = n32(batch)?; ok(cublasDgemvBatched(h, t, m, n, alpha, a, lda, x, incx, beta, y, incy, batch)) }
    fn cublasDsymv_v2_64(h: cublasHandle_t, uplo: c_int, n: i64, alpha: *const f64, a: *const f64, lda: i64, x: *const f64, incx: i64, beta: *const f64, y: *mut f64, incy: i64) { let n = n32(n)?; let lda = n32(lda)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasDsymv_v2(h, uplo, n, alpha, a, lda, x, incx, beta, y, incy)) }
    fn cublasDsbmv_v2_64(h: cublasHandle_t, uplo: c_int, n: i64, k: i64, alpha: *const f64, a: *const f64, lda: i64, x: *const f64, incx: i64, beta: *const f64, y: *mut f64, incy: i64) { let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasDsbmv_v2(h, uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)) }
    fn cublasDspmv_v2_64(h: cublasHandle_t, uplo: c_int, n: i64, alpha: *const f64, ap: *const f64, x: *const f64, incx: i64, beta: *const f64, y: *mut f64, incy: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasDspmv_v2(h, uplo, n, alpha, ap, x, incx, beta, y, incy)) }
    fn cublasDgbmv_v2_64(h: cublasHandle_t, t: c_int, m: i64, n: i64, kl: i64, ku: i64, alpha: *const f64, a: *const f64, lda: i64, x: *const f64, incx: i64, beta: *const f64, y: *mut f64, incy: i64) { let m = n32(m)?; let n = n32(n)?; let kl = n32(kl)?; let ku = n32(ku)?; let lda = n32(lda)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasDgbmv_v2(h, t, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy)) }
    fn cublasDtrmv_v2_64(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: i64, a: *const f64, lda: i64, x: *mut f64, incx: i64) { let n = n32(n)?; let lda = n32(lda)?; let incx = n32(incx)?; ok(cublasDtrmv_v2(h, uplo, t, diag, n, a, lda, x, incx)) }
    fn cublasDtbmv_v2_64(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: i64, k: i64, a: *const f64, lda: i64, x: *mut f64, incx: i64) { let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let incx = n32(incx)?; ok(cublasDtbmv_v2(h, uplo, t, diag, n, k, a, lda, x, incx)) }
    fn cublasDtpmv_v2_64(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: i64, ap: *const f64, x: *mut f64, incx: i64) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasDtpmv_v2(h, uplo, t, diag, n, ap, x, incx)) }
    fn cublasDtrsv_v2_64(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: i64, a: *const f64, lda: i64, x: *mut f64, incx: i64) { let n = n32(n)?; let lda = n32(lda)?; let incx = n32(incx)?; ok(cublasDtrsv_v2(h, uplo, t, diag, n, a, lda, x, incx)) }
    fn cublasDtbsv_v2_64(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: i64, k: i64, a: *const f64, lda: i64, x: *mut f64, incx: i64) { let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let incx = n32(incx)?; ok(cublasDtbsv_v2(h, uplo, t, diag, n, k, a, lda, x, incx)) }
    fn cublasDtpsv_v2_64(h: cublasHandle_t, uplo: c_int, t: c_int, diag: c_int, n: i64, ap: *const f64, x: *mut f64, incx: i64) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasDtpsv_v2(h, uplo, t, diag, n, ap, x, incx)) }
    fn cublasDsymm_v2_64(h: cublasHandle_t, side: c_int, uplo: c_int, m: i64, n: i64, alpha: *const f64, a: *const f64, lda: i64, b: *const f64, ldb: i64, beta: *const f64, c: *mut f64, ldc: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasDsymm_v2(h, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)) }
    fn cublasDsyrk_v2_64(h: cublasHandle_t, uplo: c_int, t: c_int, n: i64, k: i64, alpha: *const f64, a: *const f64, lda: i64, beta: *const f64, c: *mut f64, ldc: i64) { let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldc = n32(ldc)?; ok(cublasDsyrk_v2(h, uplo, t, n, k, alpha, a, lda, beta, c, ldc)) }
    fn cublasDsyr2k_v2_64(h: cublasHandle_t, uplo: c_int, t: c_int, n: i64, k: i64, alpha: *const f64, a: *const f64, lda: i64, b: *const f64, ldb: i64, beta: *const f64, c: *mut f64, ldc: i64) { let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasDsyr2k_v2(h, uplo, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) }
    fn cublasDsyrkx_64(h: cublasHandle_t, uplo: c_int, t: c_int, n: i64, k: i64, alpha: *const f64, a: *const f64, lda: i64, b: *const f64, ldb: i64, beta: *const f64, c: *mut f64, ldc: i64) { let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasDsyrkx(h, uplo, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) }
    fn cublasDtrmm_v2_64(h: cublasHandle_t, side: c_int, uplo: c_int, t: c_int, diag: c_int, m: i64, n: i64, alpha: *const f64, a: *const f64, lda: i64, b: *const f64, ldb: i64, c: *mut f64, ldc: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasDtrmm_v2(h, side, uplo, t, diag, m, n, alpha, a, lda, b, ldb, c, ldc)) }
    fn cublasDtrsm_v2_64(h: cublasHandle_t, side: c_int, uplo: c_int, t: c_int, diag: c_int, m: i64, n: i64, alpha: *const f64, a: *const f64, lda: i64, b: *mut f64, ldb: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let ldb = n32(ldb)?; ok(cublasDtrsm_v2(h, side, uplo, t, diag, m, n, alpha, a, lda, b, ldb)) }
    fn cublasDtrsmBatched_64(h: cublasHandle_t, side: c_int, uplo: c_int, t: c_int, diag: c_int, m: i64, n: i64, alpha: *const f64, a: *const *const f64, lda: i64, b: *const *mut f64, ldb: i64, batch: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let batch = n32(batch)?; ok(cublasDtrsmBatched(h, side, uplo, t, diag, m, n, alpha, a, lda, b, ldb, batch)) }
    fn cublasDgeam_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, alpha: *const f64, a: *const f64, lda: i64, beta: *const f64, b: *const f64, ldb: i64, c: *mut f64, ldc: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasDgeam(h, ta, tb, m, n, alpha, a, lda, beta, b, ldb, c, ldc)) }
    fn cublasDdgmm_64(h: cublasHandle_t, side: c_int, m: i64, n: i64, a: *const f64, lda: i64, x: *const f64, incx: i64, c: *mut f64, ldc: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let incx = n32(incx)?; let ldc = n32(ldc)?; ok(cublasDdgmm(h, side, m, n, a, lda, x, incx, c, ldc)) }
    fn cublasCaxpy_v2_64(h: cublasHandle_t, n: i64, alpha: *const Cx<f32>, x: *const Cx<f32>, incx: i64, y: *mut Cx<f32>, incy: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasCaxpy_v2(h, n, alpha, x, incx, y, incy)) }
    fn cublasCscal_v2_64(h: cublasHandle_t, n: i64, alpha: *const Cx<f32>, x: *mut Cx<f32>, incx: i64) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasCscal_v2(h, n, alpha, x, incx)) }
    fn cublasCsscal_v2_64(h: cublasHandle_t, n: i64, alpha: *const f32, x: *mut Cx<f32>, incx: i64) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasCsscal_v2(h, n, alpha, x, incx)) }
    fn cublasCcopy_v2_64(h: cublasHandle_t, n: i64, x: *const Cx<f32>, incx: i64, y: *mut Cx<f32>, incy: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasCcopy_v2(h, n, x, incx, y, incy)) }
    fn cublasCswap_v2_64(h: cublasHandle_t, n: i64, x: *mut Cx<f32>, incx: i64, y: *mut Cx<f32>, incy: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasCswap_v2(h, n, x, incx, y, incy)) }
    fn cublasCdotu_v2_64(h: cublasHandle_t, n: i64, x: *const Cx<f32>, incx: i64, y: *const Cx<f32>, incy: i64, r: *mut Cx<f32>) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasCdotu_v2(h, n, x, incx, y, incy, r)) }
    fn cublasCdotc_v2_64(h: cublasHandle_t, n: i64, x: *const Cx<f32>, incx: i64, y: *const Cx<f32>, incy: i64, r: *mut Cx<f32>) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasCdotc_v2(h, n, x, incx, y, incy, r)) }
    fn cublasScnrm2_v2_64(h: cublasHandle_t, n: i64, x: *const Cx<f32>, incx: i64, r: *mut f32) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasScnrm2_v2(h, n, x, incx, r)) }
    fn cublasScasum_v2_64(h: cublasHandle_t, n: i64, x: *const Cx<f32>, incx: i64, r: *mut f32) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasScasum_v2(h, n, x, incx, r)) }
    fn cublasCsrot_v2_64(h: cublasHandle_t, n: i64, x: *mut Cx<f32>, incx: i64, y: *mut Cx<f32>, incy: i64, c: *const f32, s: *const f32) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasCsrot_v2(h, n, x, incx, y, incy, c, s)) }
    fn cublasCgemv_v2_64(h: cublasHandle_t, t: c_int, m: i64, n: i64, alpha: *const Cx<f32>, a: *const Cx<f32>, lda: i64, x: *const Cx<f32>, incx: i64, beta: *const Cx<f32>, y: *mut Cx<f32>, incy: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasCgemv_v2(h, t, m, n, alpha, a, lda, x, incx, beta, y, incy)) }
    fn cublasCgemm_v2_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const Cx<f32>, a: *const Cx<f32>, lda: i64, b: *const Cx<f32>, ldb: i64, beta: *const Cx<f32>, c: *mut Cx<f32>, ldc: i64) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasCgemm_v2(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) }
    fn cublasCgemm3m_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const Cx<f32>, a: *const Cx<f32>, lda: i64, b: *const Cx<f32>, ldb: i64, beta: *const Cx<f32>, c: *mut Cx<f32>, ldc: i64) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasCgemm3m(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) }
    fn cublasCgemmStridedBatched_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const Cx<f32>, a: *const Cx<f32>, lda: i64, sa: i64, b: *const Cx<f32>, ldb: i64, sb: i64, beta: *const Cx<f32>, c: *mut Cx<f32>, ldc: i64, sc: i64, batch: i64) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; let batch = n32(batch)?; ok(cublasCgemmStridedBatched(h, ta, tb, m, n, k, alpha, a, lda, sa, b, ldb, sb, beta, c, ldc, sc, batch)) }
    fn cublasCgemmBatched_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const Cx<f32>, a: *const *const Cx<f32>, lda: i64, b: *const *const Cx<f32>, ldb: i64, beta: *const Cx<f32>, c: *const *mut Cx<f32>, ldc: i64, batch: i64) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; let batch = n32(batch)?; ok(cublasCgemmBatched(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch)) }
    fn cublasZaxpy_v2_64(h: cublasHandle_t, n: i64, alpha: *const Cx<f64>, x: *const Cx<f64>, incx: i64, y: *mut Cx<f64>, incy: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasZaxpy_v2(h, n, alpha, x, incx, y, incy)) }
    fn cublasZscal_v2_64(h: cublasHandle_t, n: i64, alpha: *const Cx<f64>, x: *mut Cx<f64>, incx: i64) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasZscal_v2(h, n, alpha, x, incx)) }
    fn cublasZdscal_v2_64(h: cublasHandle_t, n: i64, alpha: *const f64, x: *mut Cx<f64>, incx: i64) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasZdscal_v2(h, n, alpha, x, incx)) }
    fn cublasZcopy_v2_64(h: cublasHandle_t, n: i64, x: *const Cx<f64>, incx: i64, y: *mut Cx<f64>, incy: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasZcopy_v2(h, n, x, incx, y, incy)) }
    fn cublasZswap_v2_64(h: cublasHandle_t, n: i64, x: *mut Cx<f64>, incx: i64, y: *mut Cx<f64>, incy: i64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasZswap_v2(h, n, x, incx, y, incy)) }
    fn cublasZdotu_v2_64(h: cublasHandle_t, n: i64, x: *const Cx<f64>, incx: i64, y: *const Cx<f64>, incy: i64, r: *mut Cx<f64>) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasZdotu_v2(h, n, x, incx, y, incy, r)) }
    fn cublasZdotc_v2_64(h: cublasHandle_t, n: i64, x: *const Cx<f64>, incx: i64, y: *const Cx<f64>, incy: i64, r: *mut Cx<f64>) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasZdotc_v2(h, n, x, incx, y, incy, r)) }
    fn cublasDznrm2_v2_64(h: cublasHandle_t, n: i64, x: *const Cx<f64>, incx: i64, r: *mut f64) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasDznrm2_v2(h, n, x, incx, r)) }
    fn cublasDzasum_v2_64(h: cublasHandle_t, n: i64, x: *const Cx<f64>, incx: i64, r: *mut f64) { let n = n32(n)?; let incx = n32(incx)?; ok(cublasDzasum_v2(h, n, x, incx, r)) }
    fn cublasZdrot_v2_64(h: cublasHandle_t, n: i64, x: *mut Cx<f64>, incx: i64, y: *mut Cx<f64>, incy: i64, c: *const f64, s: *const f64) { let n = n32(n)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasZdrot_v2(h, n, x, incx, y, incy, c, s)) }
    fn cublasZgemv_v2_64(h: cublasHandle_t, t: c_int, m: i64, n: i64, alpha: *const Cx<f64>, a: *const Cx<f64>, lda: i64, x: *const Cx<f64>, incx: i64, beta: *const Cx<f64>, y: *mut Cx<f64>, incy: i64) { let m = n32(m)?; let n = n32(n)?; let lda = n32(lda)?; let incx = n32(incx)?; let incy = n32(incy)?; ok(cublasZgemv_v2(h, t, m, n, alpha, a, lda, x, incx, beta, y, incy)) }
    fn cublasZgemm_v2_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const Cx<f64>, a: *const Cx<f64>, lda: i64, b: *const Cx<f64>, ldb: i64, beta: *const Cx<f64>, c: *mut Cx<f64>, ldc: i64) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasZgemm_v2(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) }
    fn cublasZgemm3m_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const Cx<f64>, a: *const Cx<f64>, lda: i64, b: *const Cx<f64>, ldb: i64, beta: *const Cx<f64>, c: *mut Cx<f64>, ldc: i64) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; ok(cublasZgemm3m(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) }
    fn cublasZgemmStridedBatched_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const Cx<f64>, a: *const Cx<f64>, lda: i64, sa: i64, b: *const Cx<f64>, ldb: i64, sb: i64, beta: *const Cx<f64>, c: *mut Cx<f64>, ldc: i64, sc: i64, batch: i64) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; let batch = n32(batch)?; ok(cublasZgemmStridedBatched(h, ta, tb, m, n, k, alpha, a, lda, sa, b, ldb, sb, beta, c, ldc, sc, batch)) }
    fn cublasZgemmBatched_64(h: cublasHandle_t, ta: c_int, tb: c_int, m: i64, n: i64, k: i64, alpha: *const Cx<f64>, a: *const *const Cx<f64>, lda: i64, b: *const *const Cx<f64>, ldb: i64, beta: *const Cx<f64>, c: *const *mut Cx<f64>, ldc: i64, batch: i64) { let m = n32(m)?; let n = n32(n)?; let k = n32(k)?; let lda = n32(lda)?; let ldb = n32(ldb)?; let ldc = n32(ldc)?; let batch = n32(batch)?; ok(cublasZgemmBatched(h, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch)) }
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
pub unsafe extern "system" fn cublasGetCudartVersion() -> usize {
    12040
}

#[no_mangle]
pub unsafe extern "system" fn cublasGetStatusString(s: cublasStatus_t) -> *const c_char {
    cublasGetStatusName(s)
}

fn gemv_groups() -> u32 {
    std::env::var("ZLUDA_BLAS_GEMV_GROUPS").ok().and_then(|s| s.parse().ok()).unwrap_or(0)
}
