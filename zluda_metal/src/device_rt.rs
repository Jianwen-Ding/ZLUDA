//! CUDA's device runtime services, host side: the printf FIFO (vprintf), the device heap (malloc / free), the clocks
//! (%clock64, %globaltimer) and the context limits (cuCtxSetLimit / cuCtxGetLimit). The device side is
//! ptx/lib/metal_impl/devrt.inc; its module variables (`__zluda_devrt_*`) are filled before each launch
//! ([`before_launch`]) and read after it ([`after_launch`]).
//!
//! printf output is printed on the host's stdout when CUDA prints it: at the start of the next launch, at
//! cuCtxSynchronize / cuStreamSynchronize / cuEventSynchronize, at blocking memory copies, at module load and unload
//! and at context destruction ([`flush`]). The FIFO is circular (CU_LIMIT_PRINTF_FIFO_SIZE, 1 MiB by default): a
//! launch that prints more than it holds loses its oldest output, as on CUDA.
//!
//! State is per context (keyed by the context's address, `crate::cuda::current_context_key`).
#![allow(non_snake_case, clippy::missing_safety_doc)]

use crate::backend;
use crate::cuda::{CUresult, CUDA_ERROR_INVALID_VALUE, CUDA_SUCCESS};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void};
use std::sync::{Mutex, OnceLock};

pub const CU_LIMIT_STACK_SIZE: c_int = 0;
pub const CU_LIMIT_PRINTF_FIFO_SIZE: c_int = 1;
pub const CU_LIMIT_MALLOC_HEAP_SIZE: c_int = 2;
pub const CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH: c_int = 3;
pub const CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT: c_int = 4;
pub const CU_LIMIT_MAX_L2_FETCH_GRANULARITY: c_int = 5;
pub const CU_LIMIT_PERSISTING_L2_CACHE_SIZE: c_int = 6;
const LIMITS: usize = 7;
/// CUDA's defaults (documented): 1 KiB of stack per thread, a 1 MiB printf FIFO, an 8 MiB heap, sync depth 2,
/// 2048 pending launches, 64-byte L2 fetches, no persisting L2
const DEFAULT_LIMITS: [u64; LIMITS] = [1024, 1 << 20, 8 << 20, 2, 2048, 64, 0];
pub const CUDA_ERROR_UNSUPPORTED_LIMIT: CUresult = 215;

/// devrt.inc: the heap's header (bump offset, size, the free lists) before the arena
const HEAP_ARENA: u64 = 1280;
/// devrt.inc: the FIFO's header
const FIFO_HEADER: u64 = 16;
/// the clock rate the device reports (cuDeviceGetAttribute CLOCK_RATE, kHz)
const CLOCK_KHZ: u64 = 1_398_000;
/// the multiprocessors the device reports: a launch's instructions are spread over them for %clock64
const SMS: u64 = crate::agx::CORES as u64;

struct State {
    limits: [u64; LIMITS],
    /// printf FIFO (device address, 0 until a kernel that prints runs)
    fifo: u64,
    /// device heap (device address, 0 until a kernel that allocates runs); its size is fixed from then on
    heap: u64,
    /// %clock64's running count and the last %globaltimer value handed out
    cycles: u64,
    timer_ns: u64,
}

impl State {
    fn new() -> State {
        State { limits: DEFAULT_LIMITS, fifo: 0, heap: 0, cycles: 0, timer_ns: 0 }
    }
}

fn states() -> &'static Mutex<HashMap<usize, State>> {
    static S: OnceLock<Mutex<HashMap<usize, State>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let key = crate::cuda::current_context_key();
    let mut s = states().lock().unwrap();
    f(s.entry(key).or_insert_with(State::new))
}

/// The device-runtime variables of a kernel's module (device addresses in its globals buffer)
#[derive(Clone, Copy, Default)]
pub struct Bindings {
    printf: Option<u64>,
    heap: Option<u64>,
    clock: Option<u64>,
}

impl Bindings {
    /// `lookup(name)`: the device address of the module variable `name`, if the module has it
    pub fn new(lookup: impl Fn(&str) -> Option<u64>) -> Bindings {
        Bindings {
            printf: lookup("__zluda_devrt_printf"),
            heap: lookup("__zluda_devrt_heap"),
            clock: lookup("__zluda_devrt_clock"),
        }
    }
}

unsafe fn write_u64(addr: u64, v: u64) {
    if let Some((p, n)) = backend::host_pointer(addr) {
        if n >= 8 {
            (p as *mut u64).write_unaligned(v);
        }
    }
}
unsafe fn read_u64(addr: u64) -> u64 {
    match backend::host_pointer(addr) {
        Some((p, n)) if n >= 8 => (p as *const u64).read_unaligned(),
        _ => 0,
    }
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Before a launch: the previous launches' printf output, then the FIFO, heap and clock of the kernel's module.
/// Returns the token [`after_launch`] takes; Err: the FIFO or heap could not be allocated.
pub fn before_launch(b: &Bindings) -> Result<usize, String> {
    flush();
    let token = crate::agx::last_stats().lock().unwrap().len();
    with_state(|s| unsafe {
        if let Some(v) = b.printf {
            if s.fifo == 0 {
                let cap = s.limits[CU_LIMIT_PRINTF_FIFO_SIZE as usize].clamp(64, u32::MAX as u64 - 64) & !7;
                let a = backend::alloc(FIFO_HEADER + cap)?;
                write_u64(a, cap << 32);
                s.fifo = a;
            }
            write_u64(v, s.fifo);
        }
        if let Some(v) = b.heap {
            if s.heap == 0 {
                let size = HEAP_ARENA + s.limits[CU_LIMIT_MALLOC_HEAP_SIZE as usize];
                let a = backend::alloc(size)?;
                write_u64(a, HEAP_ARENA);
                write_u64(a + 8, size);
                s.heap = a;
            }
            write_u64(v, s.heap);
        }
        if let Some(c) = b.clock {
            s.timer_ns = now_ns().max(s.timer_ns + 1);
            write_u64(c, s.cycles);
            write_u64(c + 8, s.cycles);
            write_u64(c + 16, s.timer_ns);
        }
        Ok(token)
    })
}

/// After a launch: the clocks move on by the launch's instructions (the functional model's count, none on the Metal
/// backend)
pub fn after_launch(b: &Bindings, token: usize) {
    let instructions: u64 = crate::agx::last_stats().lock().unwrap().get(token..).map_or(0, |v| v.iter().map(|x| x.1).sum());
    with_state(|s| unsafe {
        if let Some(c) = b.clock {
            let end = read_u64(c);
            if end > s.cycles {
                s.timer_ns = s.timer_ns.max(read_u64(c + 16) + (end - s.cycles) * 1_000_000 / CLOCK_KHZ);
                s.cycles = end;
            }
        }
        s.cycles += instructions / SMS;
        s.timer_ns += instructions / SMS * 1_000_000 / CLOCK_KHZ;
    })
}

/// Prints the current context's pending printf output (CUDA's flush points call this)
pub fn flush() {
    let fifo = with_state(|s| s.fifo);
    if fifo != 0 {
        unsafe { print_fifo(fifo) };
    }
}

/// The current context is destroyed (or the primary one reset): its output is printed, its FIFO and heap freed and
/// its limits back to the defaults
pub fn context_destroyed(key: usize) {
    if key == crate::cuda::current_context_key() {
        flush();
    }
    if let Some(s) = states().lock().unwrap().remove(&key) {
        if s.fifo != 0 {
            backend::free(s.fifo);
        }
        if s.heap != 0 {
            backend::free(s.heap);
        }
    }
}

// ---------------------------------------------------------------------------
// printf records -> text
// ---------------------------------------------------------------------------
extern "C" {
    fn snprintf(buf: *mut c_char, n: usize, fmt: *const c_char, ...) -> c_int;
    fn fwrite(p: *const c_void, size: usize, n: usize, f: *mut c_void) -> usize;
    fn fflush(f: *mut c_void) -> c_int;
    #[cfg_attr(target_os = "macos", link_name = "__stdoutp")]
    static mut stdout: *mut c_void;
}

/// One conversion of a format (devrt.inc parse_spec, which this mirrors)
struct Spec {
    stars: u32,
    /// 0: consumes nothing, 4 / 8: an integer of that size or (8) a double / pointer, 1: a string
    size: u32,
    /// the characters after '%'
    len: usize,
    flags_width_prec: String,
    conv: u8,
}

fn parse_spec(p: &[u8]) -> Spec {
    let at = |i: usize| p.get(i).copied().unwrap_or(0);
    let mut i = 0;
    let mut stars = 0;
    while matches!(at(i), b'-' | b'+' | b' ' | b'#' | b'0' | b'\'') {
        i += 1;
    }
    if at(i) == b'*' {
        stars += 1;
        i += 1;
    } else {
        while at(i).is_ascii_digit() {
            i += 1;
        }
    }
    if at(i) == b'.' {
        i += 1;
        if at(i) == b'*' {
            stars += 1;
            i += 1;
        } else {
            while at(i).is_ascii_digit() {
                i += 1;
            }
        }
    }
    let fwp = String::from_utf8_lossy(&p[..i.min(p.len())]).into_owned();
    let mut wide = false;
    let mut h = 0;
    loop {
        match at(i) {
            b'h' => {
                h += 1;
                i += 1
            }
            b'l' | b'L' | b'q' | b'j' | b'z' | b't' => {
                wide = true;
                i += 1
            }
            _ => break,
        }
    }
    let c = at(i);
    if c == 0 {
        return Spec { stars: 0, size: 0, len: i, flags_width_prec: fwp, conv: 0 };
    }
    i += 1;
    let size = match c {
        b'd' | b'i' | b'u' | b'x' | b'X' | b'o' | b'c' => if wide { 8 } else { 4 },
        b'f' | b'F' | b'e' | b'E' | b'g' | b'G' | b'a' | b'A' | b'p' | b'n' => 8,
        b's' => 1,
        _ => 0,
    };
    // the 'h' modifiers apply to the promoted int, as in C
    let fwp = if size == 4 && c != b'c' { fwp + &"h".repeat(h.min(2)) } else { fwp };
    Spec { stars: if size == 0 { 0 } else { stars }, size, len: i, flags_width_prec: fwp, conv: c }
}

const PRINTF_MAX_ARGS: u32 = 32;

/// Formats one record: `fmt` the format, `slot(i)` its i-th 8-byte slot after the header
fn format_record(fmt: &[u8], slot: &dyn Fn(usize) -> u64, string: &dyn Fn(usize, usize) -> Vec<u8>) -> Vec<u8> {
    let mut out = vec![];
    let mut w = 0usize;
    let mut nargs = 0;
    let mut i = 0;
    let mut buf = vec![0u8; 512];
    while i < fmt.len() {
        if fmt[i] != b'%' {
            out.push(fmt[i]);
            i += 1;
            continue;
        }
        let s = parse_spec(&fmt[i + 1..]);
        let text = &fmt[i..(i + 1 + s.len).min(fmt.len())];
        i += 1 + s.len;
        if s.size == 0 {
            if s.conv == b'%' && s.len == 1 {
                out.push(b'%');
            } else {
                out.extend_from_slice(text);
            }
            continue;
        }
        if nargs >= PRINTF_MAX_ARGS {
            out.extend_from_slice(text);
            out.extend_from_slice(&fmt[i..]);
            break;
        }
        let mut star = [0 as c_int; 2];
        let mut nstar = 0;
        for k in 0..s.stars as usize {
            if nargs < PRINTF_MAX_ARGS {
                star[k] = slot(w) as u32 as i32;
                w += 1;
                nargs += 1;
                nstar += 1;
            }
        }
        if nargs >= PRINTF_MAX_ARGS {
            out.extend_from_slice(&fmt[i..]);
            break;
        }
        nargs += 1;
        let v = slot(w);
        w += 1;
        let conv = s.conv as char;
        let (spec, kind) = match (s.size, s.conv) {
            (_, b'n') => continue,
            (1, _) => (format!("%{}s", s.flags_width_prec), 's'),
            (_, b'p') => (format!("%{}p", s.flags_width_prec), 'p'),
            (_, b'f' | b'F' | b'e' | b'E' | b'g' | b'G' | b'a' | b'A') => (format!("%{}{conv}", s.flags_width_prec), 'f'),
            (_, b'c') => (format!("%{}c", s.flags_width_prec), 'i'),
            (8, _) => (format!("%{}ll{conv}", s.flags_width_prec), 'l'),
            _ => (format!("%{}{conv}", s.flags_width_prec), 'i'),
        };
        let mut str_bytes = vec![];
        if kind == 's' && v != u64::MAX {
            str_bytes = string(w, v as usize);
            w += (v as usize).div_ceil(8);
            str_bytes.push(0);
        }
        let cspec = std::ffi::CString::new(spec).unwrap_or_default();
        for _ in 0..2 {
            let n = buf.len();
            let bp = buf.as_mut_ptr() as *mut c_char;
            let f = cspec.as_ptr();
            let r = unsafe {
                match (kind, nstar) {
                    ('s', 0) => snprintf(bp, n, f, if v == u64::MAX { std::ptr::null() } else { str_bytes.as_ptr() as *const c_char }),
                    ('s', 1) => snprintf(bp, n, f, star[0], if v == u64::MAX { std::ptr::null() } else { str_bytes.as_ptr() as *const c_char }),
                    ('s', _) => snprintf(bp, n, f, star[0], star[1], if v == u64::MAX { std::ptr::null() } else { str_bytes.as_ptr() as *const c_char }),
                    ('p', 0) => snprintf(bp, n, f, v as usize as *const c_void),
                    ('p', 1) => snprintf(bp, n, f, star[0], v as usize as *const c_void),
                    ('p', _) => snprintf(bp, n, f, star[0], star[1], v as usize as *const c_void),
                    ('f', 0) => snprintf(bp, n, f, f64::from_bits(v)),
                    ('f', 1) => snprintf(bp, n, f, star[0], f64::from_bits(v)),
                    ('f', _) => snprintf(bp, n, f, star[0], star[1], f64::from_bits(v)),
                    ('l', 0) => snprintf(bp, n, f, v as i64),
                    ('l', 1) => snprintf(bp, n, f, star[0], v as i64),
                    ('l', _) => snprintf(bp, n, f, star[0], star[1], v as i64),
                    (_, 0) => snprintf(bp, n, f, v as u32 as c_int),
                    (_, 1) => snprintf(bp, n, f, star[0], v as u32 as c_int),
                    (_, _) => snprintf(bp, n, f, star[0], star[1], v as u32 as c_int),
                }
            };
            if r < 0 {
                break;
            }
            if (r as usize) < n {
                out.extend_from_slice(&buf[..r as usize]);
                break;
            }
            buf.resize(r as usize + 1, 0);
        }
    }
    out
}

/// A NUL-terminated string in device memory (at most 64 KiB)
fn device_cstr(addr: u64) -> Option<Vec<u8>> {
    let (p, n) = backend::host_pointer(addr)?;
    let b = unsafe { std::slice::from_raw_parts(p, (n as usize).min(65536)) };
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    Some(b[..end].to_vec())
}

unsafe fn print_fifo(fifo: u64) {
    let Some((p, n)) = backend::host_pointer(fifo) else { return };
    let hdr = p as *mut u32;
    let head = hdr.read() as u64;
    let cap = hdr.add(1).read() as u64;
    if head == 0 || cap == 0 || n < FIFO_HEADER + cap {
        return;
    }
    let data = p.add(FIFO_HEADER as usize);
    let word = |abs: u64| (data.add((abs % cap) as usize) as *const u64).read_unaligned();
    let mut out = vec![];
    // the oldest record still whole: the first 8-aligned position in the last `cap` bytes whose header names it
    let mut pos = (head.saturating_sub(cap) + 7) & !7;
    while pos + 16 <= head {
        let h = word(pos);
        let (start, len) = (h & 0xFFFF_FFFF, h >> 32);
        if start != pos & 0xFFFF_FFFF || len < 16 || len % 8 != 0 || pos + len > head {
            pos += 8;
            continue;
        }
        let fmt = device_cstr(word(pos + 8)).unwrap_or_else(|| b"<unreadable printf format>\n".to_vec());
        let base = pos + 16;
        let slot = |i: usize| word(base + 8 * i as u64);
        let string = |i: usize, len: usize| {
            (0..len).map(|k| *data.add(((base + 8 * i as u64 + k as u64) % cap) as usize)).collect::<Vec<u8>>()
        };
        out.extend(format_record(&fmt, &slot, &string));
        pos += len;
    }
    hdr.write(0);
    if !out.is_empty() {
        fwrite(out.as_ptr().cast(), 1, out.len(), stdout);
        fflush(stdout);
    }
}

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------
/// cuCtxSetLimit for the current context. CU_LIMIT_PRINTF_FIFO_SIZE prints the pending output and replaces the
/// FIFO; CU_LIMIT_MALLOC_HEAP_SIZE cannot change once a kernel that allocates has run in the context (the heap's
/// blocks stay valid across launches); the stack size is recorded (ZLUDA's per-thread stack is the kernel's own:
/// a kernel that needs more faults with CUDA_ERROR_HARDWARE_STACK_ERROR).
pub fn set_limit(limit: c_int, value: u64) -> CUresult {
    if !(0..LIMITS as c_int).contains(&limit) {
        return CUDA_ERROR_UNSUPPORTED_LIMIT;
    }
    if limit == CU_LIMIT_PRINTF_FIFO_SIZE {
        flush();
    }
    with_state(|s| {
        match limit {
            CU_LIMIT_PRINTF_FIFO_SIZE => {
                if s.fifo != 0 {
                    backend::free(s.fifo);
                    s.fifo = 0;
                }
            }
            CU_LIMIT_MALLOC_HEAP_SIZE if s.heap != 0 && value != s.limits[limit as usize] => {
                return CUDA_ERROR_INVALID_VALUE;
            }
            CU_LIMIT_MAX_L2_FETCH_GRANULARITY if value > 128 => return CUDA_ERROR_INVALID_VALUE,
            _ => {}
        }
        s.limits[limit as usize] = value;
        CUDA_SUCCESS
    })
}

pub fn get_limit(limit: c_int) -> Result<u64, CUresult> {
    if !(0..LIMITS as c_int).contains(&limit) {
        return Err(CUDA_ERROR_UNSUPPORTED_LIMIT);
    }
    Ok(with_state(|s| s.limits[limit as usize]))
}

#[no_mangle]
pub unsafe extern "system" fn cuCtxSetLimit(limit: c_int, value: usize) -> CUresult {
    if let Err(e) = crate::cuda::check_current() {
        return e;
    }
    set_limit(limit, value as u64)
}

#[no_mangle]
pub unsafe extern "system" fn cuCtxGetLimit(value: *mut usize, limit: c_int) -> CUresult {
    if let Err(e) = crate::cuda::check_current() {
        return e;
    }
    if value.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    match get_limit(limit) {
        Ok(v) => {
            *value = v as usize;
            CUDA_SUCCESS
        }
        Err(e) => e,
    }
}

/// The blocks a cooperative launch may have: a grid-wide barrier needs every block resident at once, and the AGX
/// functional model runs the workgroups of a dispatch one after the other (each to its end), so only a one-block
/// grid can synchronize. Larger grids fail as CUDA fails a grid that does not fit the device.
pub const MAX_COOPERATIVE_BLOCKS: u64 = 1;

// cuLaunchCooperativeKernel (src/cuda.rs) bounds the grid by MAX_COOPERATIVE_BLOCKS

#[cfg(test)]
mod tests {
    use super::*;
    fn fmt(f: &str, slots: &[u64]) -> String {
        let s = slots.to_vec();
        String::from_utf8(format_record(f.as_bytes(), &|i| s[i], &|_, _| vec![])).unwrap()
    }
    #[test]
    fn formats() {
        assert_eq!(fmt("a %d b %5.2f %x %%\n", &[(-3i32) as u32 as u64, 2.5f64.to_bits(), 255]), "a -3 b  2.50 ff %\n");
        assert_eq!(fmt("%lld %llu %c %*d", &[u64::MAX, u64::MAX, b'z' as u64, 4, 7]),
                   "-1 18446744073709551615 z    7");
        assert_eq!(fmt("%e %g %hhd", &[1e10f64.to_bits(), 0.0001f64.to_bits(), 0x1ff]), "1.000000e+10 0.0001 -1");
    }
}
