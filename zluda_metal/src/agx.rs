//! AGX simulator backend: `.metallib` -> `llvm-air2agx` -> AGX machine code,
//! run functionally on the AGX Performance Simulator (`agx_sim`).
//!
//! Selected with `ZLUDA_METAL_BACKEND=agx`. Every tool in the path is open
//! source: the metallib is ZLUDA's own (cpp/metallib.cpp), `llvm-air2agx`
//! is the AGX backend of the llvm-project fork (`ZLUDA_AGX_AIR2AGX`, default
//! `llvm-air2agx` on PATH), and the simulator is linked in as a library.
//!
//! The launch follows llvm/docs/AGXUsage.rst ("Kernel descriptors",
//! ".agx.kernels"): each argument of the record is a Metal buffer location;
//! kind 0 arguments are bytes in uniform registers, kind 1 a 64-bit pointer
//! to them. `threadgroups_per_grid`, the global data buffer and extensions
//! 4 (threadgroup bytes), 6 (constants), 7 (launch divisors), 8 (per-CTA
//! memory) and 9 (32-bit address base) are filled as a driver would.
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

use agx_sim::func::{self, Dispatch, Memory};

/// First device address handed out: below 4 GiB, so that the 32-bit pointers of
/// `.address_size 32` modules (zero-extended) reach the buffers too.
const HEAP_BASE: u64 = 0x1000_0000;
const ALIGN: u64 = 256;

struct Heap {
    next: u64,
    /// base -> bytes
    allocs: BTreeMap<u64, Block>,
}

/// The bytes of a device allocation: the heap's own, or host memory that the device sees at its host address
/// (pinned, registered and managed memory of the CUDA driver API, src/cuda/memory.rs)
enum Block {
    Owned(Vec<u8>),
    Host(*mut u8, usize),
}
unsafe impl Send for Block {}

impl Block {
    fn bytes(&self) -> &[u8] {
        match self {
            Block::Owned(v) => v,
            Block::Host(p, n) => unsafe { std::slice::from_raw_parts(*p, *n) },
        }
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Block::Owned(v) => v,
            Block::Host(p, n) => unsafe { std::slice::from_raw_parts_mut(*p, *n) },
        }
    }
    fn len(&self) -> usize {
        self.bytes().len()
    }
}

fn heap() -> &'static Mutex<Heap> {
    static H: OnceLock<Mutex<Heap>> = OnceLock::new();
    H.get_or_init(|| {
        Mutex::new(Heap {
            next: HEAP_BASE,
            allocs: BTreeMap::new(),
        })
    })
}

impl Heap {
    /// the first block overlapping [base, base + len)
    fn overlap(&self, base: u64, len: u64) -> Option<(u64, u64)> {
        if let Some((b, v)) = self.allocs.range(..=base).next_back() {
            if base < b + v.len() as u64 {
                return Some((*b, v.len() as u64));
            }
        }
        self.allocs.range(base..base.saturating_add(len)).next().map(|(b, v)| (*b, v.len() as u64))
    }
}

pub fn alloc(size: u64) -> Result<u64, String> {
    let mut h = heap().lock().unwrap();
    let mut base = h.next;
    // skip host memory registered in the heap's range
    while let Some((b, n)) = h.overlap(base, size.max(1) + ALIGN) {
        base = (b + n + ALIGN + 0xFFF) & !0xFFF;
    }
    // keep a gap between allocations so overruns fault instead of corrupting a neighbour
    h.next = (base + size.max(1) + ALIGN + 0xFFF) & !0xFFF;
    h.allocs.insert(base, Block::Owned(vec![0u8; size.max(1) as usize]));
    Ok(base)
}

pub fn free(addr: u64) -> bool {
    let mut h = heap().lock().unwrap();
    match h.allocs.get(&addr) {
        Some(Block::Owned(_)) => h.allocs.remove(&addr).is_some(),
        _ => false,
    }
}

/// Host memory the device accesses at its own address (unified virtual addressing): `len` bytes at `p`
pub fn register_host(p: *mut u8, len: u64) -> Result<u64, String> {
    let mut h = heap().lock().unwrap();
    let base = p as u64;
    if len == 0 || base == 0 {
        return Err("empty host range".into());
    }
    if let Some((b, n)) = h.overlap(base, len) {
        return Err(format!("host range 0x{base:x} + {len} overlaps device memory 0x{b:x} + {n}"));
    }
    h.allocs.insert(base, Block::Host(p, len as usize));
    Ok(base)
}

pub fn unregister_host(addr: u64) -> bool {
    let mut h = heap().lock().unwrap();
    match h.allocs.get(&addr) {
        Some(Block::Host(..)) => h.allocs.remove(&addr).is_some(),
        _ => false,
    }
}

pub fn range(addr: u64) -> Option<(u64, u64)> {
    let h = heap().lock().unwrap();
    let (b, v) = h.allocs.range(..=addr).next_back()?;
    (addr < b + v.len() as u64).then_some((*b, v.len() as u64))
}

/// Host view of device memory. The vectors are never reallocated while they
/// live, so the pointer stays valid until `free`.
pub fn host_pointer(addr: u64) -> Option<(*mut u8, u64)> {
    let mut h = heap().lock().unwrap();
    let (b, v) = h.allocs.range_mut(..=addr).next_back()?;
    let off = addr - *b;
    if off >= v.len() as u64 {
        return None;
    }
    let n = v.len() as u64;
    Some((unsafe { v.bytes_mut().as_mut_ptr().add(off as usize) }, n - off))
}

/// The simulated GPU's clock: the predicted time of every kernel timed so far (ns). Kernels are timed when
/// `ZLUDA_AGX_STATS` is set or something asked for the time (`want_timing`: a CUDA event that records time).
static SIM_CLOCK_NS: Mutex<f64> = Mutex::new(0.0);
static TIMING_WANTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn sim_clock_ns() -> f64 {
    *SIM_CLOCK_NS.lock().unwrap()
}
pub fn want_timing() {
    TIMING_WANTED.store(true, std::sync::atomic::Ordering::Relaxed);
}
fn timing_enabled() -> bool {
    std::env::var_os("ZLUDA_AGX_STATS").is_some() || TIMING_WANTED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Occupancy of a launch of `k` (the simulator's dispatch model, M2 Max): (resident workgroups per core, workgroups
/// per core guaranteed to run concurrently), for `threads` per workgroup and `shared` threadgroup bytes in all
pub fn occupancy(k: &Kernel, threads: u32, shared: u32) -> (u32, u32) {
    let mut l = agx_sim::dispatch::Launch::default();
    l.wg = [threads, 1, 1];
    l.regs = (k.record.gprs * 2).max(8);
    l.tg_bytes = shared;
    (agx_sim::dispatch::occupancy(&l).1, agx_sim::dispatch::co_resident_workgroups(&l))
}

/// Serializes the simulated device: a launch copies every allocation in and out, so kernels and host copies of
/// different threads (CUDA stream workers, src/cuda/stream.rs) must not interleave
pub fn device_lock() -> std::sync::MutexGuard<'static, ()> {
    static L: Mutex<()> = Mutex::new(());
    L.lock().unwrap_or_else(|e| e.into_inner())
}

/// GPU cores of the simulated device (M2 Max)
pub const CORES: u32 = 38;

// ---------------------------------------------------------------------------
// ELF reading (64-bit little-endian, what llvm-air2agx writes)
// ---------------------------------------------------------------------------
struct Elf<'a> {
    d: &'a [u8],
    sections: Vec<(String, u32, u64, u64, u32, u64)>, // name, type, offset, size, link, entsize
}

fn rd<const N: usize>(d: &[u8], at: usize) -> Result<[u8; N], String> {
    d.get(at..at + N)
        .map(|s| s.try_into().unwrap())
        .ok_or_else(|| "truncated AGX object".to_string())
}
fn u16le(d: &[u8], at: usize) -> Result<u16, String> {
    Ok(u16::from_le_bytes(rd(d, at)?))
}
fn u32le(d: &[u8], at: usize) -> Result<u32, String> {
    Ok(u32::from_le_bytes(rd(d, at)?))
}
fn u64le(d: &[u8], at: usize) -> Result<u64, String> {
    Ok(u64::from_le_bytes(rd(d, at)?))
}

impl<'a> Elf<'a> {
    fn parse(d: &'a [u8]) -> Result<Self, String> {
        if d.len() < 64 || &d[0..4] != b"\x7fELF" || d[4] != 2 || d[5] != 1 {
            return Err("llvm-air2agx did not write a 64-bit little-endian ELF object".into());
        }
        let shoff = u64le(d, 0x28)? as usize;
        let shentsize = u16le(d, 0x3A)? as usize;
        let shnum = u16le(d, 0x3C)? as usize;
        let shstrndx = u16le(d, 0x3E)? as usize;
        let mut raw = vec![];
        for i in 0..shnum {
            let s = shoff + i * shentsize;
            raw.push((
                u32le(d, s)?,
                u32le(d, s + 4)?,
                u64le(d, s + 0x18)?,
                u64le(d, s + 0x20)?,
                u32le(d, s + 0x28)?,
                u64le(d, s + 0x38)?,
            ));
        }
        let stro = raw.get(shstrndx).ok_or("bad ELF string table")?.2 as usize;
        let mut sections = vec![];
        for (name, ty, off, size, link, ent) in raw {
            sections.push((cstr(d, stro + name as usize), ty, off, size, link, ent));
        }
        Ok(Elf { d, sections })
    }
    fn section(&self, name: &str) -> Option<&'a [u8]> {
        self.sections
            .iter()
            .find(|s| s.0 == name)
            .and_then(|s| self.d.get(s.2 as usize..(s.2 + s.3) as usize))
    }
    fn index(&self, name: &str) -> Option<usize> {
        self.sections.iter().position(|s| s.0 == name)
    }
    /// (name, value, size, section index)
    fn symbols(&self) -> Result<Vec<(String, u64, u64, u16)>, String> {
        let Some(st) = self.sections.iter().find(|s| s.1 == 2) else {
            return Ok(vec![]);
        };
        let stro = self.sections[st.4 as usize].2 as usize;
        let mut out = vec![];
        for i in 0..(st.3 / 24) as usize {
            let e = st.2 as usize + i * 24;
            out.push((
                cstr(self.d, stro + u32le(self.d, e)? as usize),
                u64le(self.d, e + 8)?,
                u64le(self.d, e + 16)?,
                u16le(self.d, e + 6)?,
            ));
        }
        Ok(out)
    }
}

fn cstr(d: &[u8], at: usize) -> String {
    let end = d[at.min(d.len())..].iter().position(|&b| b == 0).unwrap_or(0);
    String::from_utf8_lossy(&d[at.min(d.len())..at.min(d.len()) + end]).into_owned()
}

// ---------------------------------------------------------------------------
// .agx.kernels
// ---------------------------------------------------------------------------
#[derive(Clone, Debug, Default)]
pub struct Record {
    pub name: String,
    pub uniform_halves: u32,
    pub tg_half: u32,
    pub glob_half: u32,
    pub gprs: u32,
    pub tg_bytes: u32,
    pub stack: u32,
    /// (kind, bytes, first half)
    pub args: Vec<(u32, u32, u32)>,
    pub cluster: Option<[u32; 3]>,
    pub tg_bytes_half: Option<u32>,
    pub texture_heap_half: Option<u32>,
    pub constants: Vec<(u32, u32)>,
    pub launch_values: Vec<(u32, u32, u32)>,
    pub percta: Option<(u32, u32)>,
    pub address32_half: Option<u32>,
}

fn parse_records(d: &[u8]) -> Result<Vec<Record>, String> {
    let mut out = vec![];
    let mut pos = 0usize;
    while pos + 8 <= d.len() {
        let size = u32le(d, pos)? as usize;
        if size == 0 {
            break;
        }
        let end = pos + size;
        let nlen = u32le(d, pos + 8)? as usize;
        let mut k = Record {
            name: String::from_utf8_lossy(d.get(pos + 12..pos + 12 + nlen).ok_or("bad record")?).into_owned(),
            ..Default::default()
        };
        let mut q = pos + 12 + (nlen + 3) / 4 * 4;
        let w = |i: usize| u32le(d, q + 4 * i);
        k.uniform_halves = w(0)?;
        k.tg_half = w(1)?;
        k.glob_half = w(2)?;
        k.gprs = w(3)?;
        k.tg_bytes = w(4)?;
        k.stack = w(5)?;
        let nargs = w(6)? as usize;
        q += 28;
        for i in 0..nargs {
            k.args.push((u32le(d, q + 12 * i)?, u32le(d, q + 12 * i + 4)?, u32le(d, q + 12 * i + 8)?));
        }
        q += 12 * nargs;
        while q < end {
            let tag = u32le(d, q)?;
            match tag {
                1 => {
                    k.cluster = Some([u32le(d, q + 4)?, u32le(d, q + 8)?, u32le(d, q + 12)?]);
                    q += 16
                }
                2 => {
                    let n = u32le(d, q + 8)? as usize;
                    q += 20 + 16 * n
                }
                3 => q += 12,
                4 => {
                    k.tg_bytes_half = Some(u32le(d, q + 4)?);
                    q += 8
                }
                5 => {
                    k.texture_heap_half = Some(u32le(d, q + 4)?);
                    q += 8
                }
                6 => {
                    let n = u32le(d, q + 4)? as usize;
                    for i in 0..n {
                        k.constants.push((u32le(d, q + 8 + 8 * i)?, u32le(d, q + 12 + 8 * i)?));
                    }
                    q += 8 + 8 * n
                }
                7 => {
                    let n = u32le(d, q + 4)? as usize;
                    for i in 0..n {
                        let e = q + 8 + 12 * i;
                        k.launch_values.push((u32le(d, e)?, u32le(d, e + 4)?, u32le(d, e + 8)?));
                    }
                    q += 8 + 12 * n
                }
                8 => {
                    k.percta = Some((u32le(d, q + 4)?, u32le(d, q + 8)?));
                    q += 12
                }
                9 => {
                    k.address32_half = Some(u32le(d, q + 4)?);
                    q += 8
                }
                _ => break,
            }
        }
        out.push(k);
        pos = end;
    }
    Ok(out)
}

/// Extension 7: the multiplier and shifts of a division by `d` (AGXUsage.rst).
pub fn launch_value(d: u32, kind: u32) -> u32 {
    let (mut d, mut kind) = (d as u64, kind);
    if kind >= 6 {
        d = d.max(1);
        kind -= 6;
    }
    if kind >= 3 && d >> 31 != 0 {
        d = (1u64 << 32) - d;
    }
    if d == 0 {
        return 0;
    }
    let l = 64 - (d - 1).leading_zeros() as u64;
    let l = if d == 1 { 0 } else { l };
    (match kind % 3 {
        0 => (((1u128 << 32) * ((1u128 << l) - d as u128)) / d as u128 + 1) as u64,
        1 => l.min(1),
        _ => l.max(1) - 1,
    }) as u32
}

// ---------------------------------------------------------------------------
// Libraries and kernels
// ---------------------------------------------------------------------------
/// The prefix of a launch error that is a fault of the running kernel (an unmapped access, a step limit, ...),
/// not of the launch itself
pub const DEVICE_FAULT: &str = "device fault: ";

/// A kernel argument of [`Kernel::launch_args`]: bytes copied into a temporary buffer (whose address is the
/// argument), a device address, or a value (the argument's own bytes: a scalar of an MLIR kernel)
pub enum Arg<'a> {
    Bytes(&'a [u8]),
    Addr(u64),
    Value(Vec<u8>),
}

pub struct Library {
    /// kernel -> (record, its object's code, the kernel's offset in it, global data buffer of its object); a
    /// kernel's code is copied out when it is first asked for (a library of many kernels loads in one pass)
    records: HashMap<String, (Record, Arc<Vec<u8>>, usize, Option<u64>)>,
    /// global data buffers (one per object), freed with the library
    globals: Vec<u64>,
}

#[derive(Clone)]
pub struct Kernel {
    record: Record,
    code: Arc<Vec<u8>>,
    globals: Option<u64>,
}

fn air2agx() -> String {
    std::env::var("ZLUDA_AGX_AIR2AGX").unwrap_or_else(|_| "llvm-air2agx".into())
}

pub fn mcpu() -> String {
    std::env::var("ZLUDA_AGX_MCPU").unwrap_or_else(|_| "agx-g14c".into())
}

/// Compiles a metallib with llvm-air2agx; returns one ELF object per AIR module.
pub fn compile_metallib(metallib: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let dir = std::env::temp_dir().join(format!(
        "zluda-agx-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let out_dir = dir.join("out");
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let lib = dir.join("m.metallib");
    std::fs::write(&lib, metallib).map_err(|e| e.to_string())?;
    let mut cmd = std::process::Command::new(air2agx());
    cmd.arg(format!("-mcpu={}", mcpu()))
        .arg(&lib)
        .arg("--emit=obj")
        .arg("--output-dir")
        .arg(&out_dir);
    // PTX's NaN results are the canonical 0x7fffffff (AGXUsage.rst, canonical-nan)
    // (strict floating point mode only; fast keeps the hardware's NaN bits)
    let flags = std::env::var("ZLUDA_AGX_FLAGS").unwrap_or_else(|_| match crate::compile::FpMode::from_env() {
        crate::compile::FpMode::Strict => "-mattr=+canonical-nan".into(),
        _ => String::new(),
    });
    cmd.args(flags.split_whitespace());
    let out = cmd
        .output()
        .map_err(|e| format!("cannot run {} (set ZLUDA_AGX_AIR2AGX): {e}", air2agx()))?;
    let res = if out.status.success() {
        let mut objs = vec![];
        let mut names: Vec<_> = std::fs::read_dir(&out_dir)
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        names.sort();
        for n in names {
            objs.push(std::fs::read(&n).map_err(|e| e.to_string())?);
        }
        if objs.is_empty() {
            Err("llvm-air2agx wrote no object".to_string())
        } else {
            Ok(objs)
        }
    } else {
        Err(format!(
            "llvm-air2agx failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ))
    };
    if std::env::var_os("ZLUDA_AGX_KEEP").is_none() {
        let _ = std::fs::remove_dir_all(&dir);
    }
    res
}

static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl Library {
    pub fn new(metallib: &[u8]) -> Result<Self, String> {
        let objs = compile_metallib(metallib)?;
        let mut lib = Library { records: HashMap::new(), globals: vec![] };
        for o in &objs {
            let mut l = Self::from_object(o)?;
            lib.records.extend(std::mem::take(&mut l.records));
            lib.globals.extend(std::mem::take(&mut l.globals));
        }
        Ok(lib)
    }

    /// One AGX object: its kernels see its own global data buffer.
    pub fn from_object(obj: &[u8]) -> Result<Self, String> {
        let elf = Elf::parse(obj)?;
        let text = elf.section(".text").unwrap_or(&[]);
        let text_idx = elf.index(".text").unwrap_or(usize::MAX) as u16;
        let recs = parse_records(elf.section(".agx.kernels").ok_or("no .agx.kernels section")?)?;
        let syms = elf.symbols()?;
        // global data buffer, relocated in place
        let mut globals = None;
        if let Some(g) = elf.section(".agx.globals") {
            let mut data = g.to_vec();
            let va = alloc(data.len().max(16) as u64)?;
            let gidx = elf.index(".agx.globals").unwrap();
            for (i, s) in elf.sections.iter().enumerate() {
                if s.1 != 4 || s.0 != ".rela.agx.globals" {
                    continue;
                }
                let _ = i;
                for r in 0..(s.3 / 24) as usize {
                    let e = s.2 as usize + r * 24;
                    let off = u64le(obj, e)? as usize;
                    let info = u64le(obj, e + 8)?;
                    let addend = u64le(obj, e + 16)? as i64;
                    let sym = &syms[(info >> 32) as usize];
                    if sym.3 as usize != gidx {
                        return Err(format!("relocation against {} outside .agx.globals", sym.0));
                    }
                    let v = (va + sym.1).wrapping_add(addend as u64);
                    match info & 0xFFFF_FFFF {
                        1 => data[off..off + 8].copy_from_slice(&v.to_le_bytes()),
                        2 => data[off..off + 4].copy_from_slice(&(v as u32).to_le_bytes()),
                        t => return Err(format!("unknown AGX relocation type {t}")),
                    }
                }
            }
            let (p, _) = host_pointer(va).unwrap();
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len()) };
            globals = Some((va, data.len()));
        }
        let mut records = HashMap::new();
        let text = Arc::new(text.to_vec());
        for r in recs {
            let start = syms
                .iter()
                .find(|s| s.0 == r.name && s.3 == text_idx)
                .ok_or_else(|| format!("kernel {} has no symbol", r.name))?;
            if start.1 as usize > text.len() {
                return Err("kernel outside .text".into());
            }
            records.insert(r.name.clone(), (r, text.clone(), start.1 as usize, globals.map(|g: (u64, usize)| g.0)));
        }
        Ok(Library { records, globals: globals.map(|g| vec![g.0]).unwrap_or_default() })
    }

    pub fn kernel(&self, name: &str) -> Result<Kernel, String> {
        let (r, text, start, g) = self
            .records
            .get(name)
            .ok_or_else(|| format!("kernel {name} not found in the AGX object"))?;
        Ok(Kernel {
            record: r.clone(),
            code: Arc::new(text[*start..].to_vec()),
            globals: *g,
        })
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        for va in &self.globals {
            free(*va);
        }
    }
}

/// Uniform register writes of one launch
struct Uniforms(BTreeMap<usize, u32>);
impl Uniforms {
    fn put(&mut self, half: u32, value: u64, words: usize) {
        if half == u32::MAX {
            return;
        }
        for w in 0..words {
            self.0.insert(half as usize / 2 + w, (value >> (32 * w)) as u32);
        }
    }
    fn put_bytes(&mut self, half: u32, bytes: &[u8]) {
        for (i, c) in bytes.chunks(4).enumerate() {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            self.0.insert(half as usize / 2 + i, u32::from_le_bytes(w));
        }
    }
    fn get(&self, half: u32) -> u32 {
        *self.0.get(&(half as usize / 2)).unwrap_or(&0)
    }
}

pub struct Stats {
    pub instructions: u64,
}

pub fn last_stats() -> &'static Mutex<Vec<(String, u64)>> {
    static S: OnceLock<Mutex<Vec<(String, u64)>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(vec![]))
}

impl Kernel {
    pub fn max_threads(&self) -> u64 {
        1024
    }
    pub fn static_shared(&self) -> u64 {
        self.record.tg_bytes as u64
    }
    pub fn record(&self) -> &Record {
        &self.record
    }

    /// A launch of a kernel compiled from PTX: buffer 0 is ZLUDA's parameter buffer
    pub fn launch(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        params: &[u8],
        dynamic_shared: u32,
    ) -> Result<(), String> {
        self.launch_args(grid, block, &[Arg::Bytes(params)], dynamic_shared)
    }

    /// A launch with the kernel's buffer arguments in order (`args[i]` for buffer i; the others get 4 KiB of
    /// zeros). An AIR kernel from MLIR (src/blas.rs) has its pointers as buffers in order and its scalars packed
    /// into one last buffer (AIRScalarBufferPacking: in order, naturally aligned).
    pub fn launch_args(
        &self,
        grid: [u32; 3],
        block: [u32; 3],
        args: &[Arg],
        dynamic_shared: u32,
    ) -> Result<(), String> {
        let _device = device_lock();
        let params: &[u8] = match args.first() {
            Some(Arg::Bytes(b)) => b,
            _ => &[],
        };
        let r = &self.record;
        // thread-block clusters (extension 1, agx-cluster-dims): a cluster of C CTAs is one threadgroup, the grid
        // divided by the cluster dims and the threadgroup's z multiplied by C
        let (grid, block, dynamic_shared) = match r.cluster {
            Some(cd) if cd.iter().product::<u32>() > 1 => {
                if (0..3).any(|d| cd[d] == 0 || grid[d] % cd[d] != 0) {
                    return Err(format!("grid {:?} is not a multiple of the cluster dims {:?}", grid, cd));
                }
                let n: u32 = cd.iter().product();
                ([0, 1, 2].map(|d| grid[d] / cd[d]), [block[0], block[1], block[2] * n], dynamic_shared * n)
            }
            _ => (grid, block, dynamic_shared),
        };
        let mut u = Uniforms(BTreeMap::new());
        let mut temps = vec![];
        // Metal buffer table: location 0 is ZLUDA's parameter buffer (`[[buffer(0)]]`), the
        // others unused. A buffer argument's 8 bytes in the uniforms are its address (kind 0);
        // kind 1 would be a pointer to them. A by-reference argument (kind 0 with the size of
        // its bytes, not 8) has its bytes in the uniforms.
        for (i, &(kind, bytes, half)) in r.args.iter().enumerate() {
            if half == u32::MAX {
                continue;
            }
            if let Some(Arg::Value(v)) = args.get(i) {
                if kind != 0 || v.len() != bytes as usize {
                    return Err(format!("{}: argument {i} is {bytes} bytes (kind {kind}), given {}", r.name, v.len()));
                }
                u.put_bytes(half, v);
                continue;
            }
            if let Some(Arg::Addr(va)) = args.get(i) {
                if kind == 0 && bytes == 8 {
                    u.put(half, *va, 2);
                } else {
                    let pa = alloc(16)?;
                    let (pp, _) = host_pointer(pa).unwrap();
                    unsafe { std::ptr::copy_nonoverlapping(va.to_le_bytes().as_ptr(), pp, 8) };
                    temps.push(pa);
                    u.put(half, pa, 2);
                }
                continue;
            }
            let data: Vec<u8> = match args.get(i) {
                Some(Arg::Bytes(b)) => b.to_vec(),
                _ => vec![0u8; 4096],
            };
            if args.len() > 1 && i >= args.len() {
                return Err(format!("{}: {} arguments given, the kernel has {}", r.name, args.len(), r.args.len()));
            }
            if kind == 0 && bytes != 8 {
                let mut b = data.clone();
                b.resize(bytes as usize, 0);
                u.put_bytes(half, &b);
                continue;
            }
            let va = alloc(data.len().max(16) as u64)?;
            let (p, _) = host_pointer(va).unwrap();
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len()) };
            temps.push(va);
            if kind == 0 {
                u.put(half, va, 2);
            } else {
                // a pointer to the 8 bytes of the address
                let pa = alloc(16)?;
                let (pp, _) = host_pointer(pa).unwrap();
                unsafe { std::ptr::copy_nonoverlapping(va.to_le_bytes().as_ptr(), pp, 8) };
                temps.push(pa);
                u.put(half, pa, 2);
            }
        }
        let groups = [0, 1, 2].map(|d| grid[d]);
        for d in 0..3 {
            u.put(r.tg_half + 2 * d as u32, groups[d] as u64, 1);
        }
        if let Some(g) = self.globals {
            u.put(r.glob_half, g, 2);
        }
        let shared = r.tg_bytes + dynamic_shared;
        if let Some(h) = r.tg_bytes_half {
            u.put(h, shared as u64, 1);
        }
        for &(h, v) in &r.constants {
            u.put(h, v as u64, 1);
        }
        for &(h, src, kind) in &r.launch_values {
            let d = u.get(src);
            u.put(h, launch_value(d, kind) as u64, 1);
        }
        if let Some((h, size)) = r.percta {
            let n = groups.iter().map(|&x| x as u64).product::<u64>();
            let va = alloc(n * size as u64 + 1024)?;
            temps.push(va);
            u.put(h, (va + 1023) & !1023, 2);
        }
        if let Some(h) = r.address32_half {
            u.put(h, 0, 2);
        }
        if r.texture_heap_half.is_some() {
            return Err("textures are not supported by the AGX simulator backend yet".into());
        }
        let mut code = (*self.code).clone();
        while code.len() > 2 && code[code.len() - 2..] == [0, 0] {
            code.truncate(code.len() - 2);
        }
        let d = Dispatch {
            code,
            grid: [0, 1, 2].map(|i| grid[i] * block[i]),
            wg: block,
            shared_bytes: shared,
            uniforms: u.0.into_iter().collect(),
            num_registers: (r.gprs as usize).clamp(128, 256),
            max_steps: std::env::var("ZLUDA_AGX_MAX_STEPS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2_000_000_000),
            placement: 0,
            launch: agx_sim::cost::SrLaunch {
                tg_bytes: shared,
                ..Default::default()
            },
        };
        if std::env::var_os("ZLUDA_AGX_TRACE").is_some() {
            eprintln!("[agx] {} grid {:?} wg {:?} shared {} record {:?}", r.name, d.grid, d.wg, shared, r.args);
            eprintln!("[agx] uniforms {:x?}", d.uniforms);
            eprintln!("[agx] params {:x?}", params);
        }
        // the simulated GPU sees every allocation at its address
        let mut mem = Memory::default();
        {
            let h = heap().lock().unwrap();
            for (va, v) in &h.allocs {
                mem.map(*va, v.bytes().to_vec());
            }
        }
        let timing = timing_enabled().then(|| time_dispatch(&d, &mem, r));
        let res = func::run(&d, &mut mem);
        {
            let mut h = heap().lock().unwrap();
            for (va, v) in h.allocs.iter_mut() {
                if let Ok(b) = mem.read_bytes(*va, v.len()) {
                    v.bytes_mut().copy_from_slice(&b);
                }
            }
        }
        for t in temps {
            free(t);
        }
        // an error of the functional model is the kernel's own fault (src/cuda.rs: a sticky context error)
        let st = res.map_err(|e| format!("{DEVICE_FAULT}{e}"))?;
        last_stats().lock().unwrap().push((r.name.clone(), st.instructions));
        if let Some(Ok((us, _))) = &timing {
            if us.is_finite() {
                *SIM_CLOCK_NS.lock().unwrap() += us * 1000.0;
            }
        }
        if let Some(path) = std::env::var_os("ZLUDA_AGX_STATS") {
            use std::io::Write;
            let (time_us, cycles, err) = match timing {
                Some(Ok(t)) => (t.0, t.1, String::new()),
                Some(Err(e)) => (f64::NAN, f64::NAN, e),
                None => (f64::NAN, f64::NAN, String::new()),
            };
            let line = serde_json::json!({
                "kernel": r.name, "grid": grid, "block": block, "instructions": st.instructions,
                "time_us": if time_us.is_finite() { serde_json::json!(time_us) } else { serde_json::Value::Null },
                "core_cycles": if cycles.is_finite() { serde_json::json!(cycles) } else { serde_json::Value::Null },
                "gprs": r.gprs, "code_bytes": self.code.len(), "timing_error": err,
            });
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(f, "{line}");
            }
        }
        Ok(())
    }
}

/// The AGX Performance Simulator's prediction of the dispatch (CUDA-runtime launch profile, each
/// SIMD group's path from the functional model over the launch's memory): (microseconds, core cycles).
fn time_dispatch(d: &Dispatch, mem: &Memory, r: &Record) -> Result<(f64, f64), String> {
    use agx_sim::dispatch::{self, Launch};
    static PARAMS: OnceLock<Result<agx_sim::cost::Params, String>> = OnceLock::new();
    let p = PARAMS.get_or_init(agx_sim::cost::Params::new).as_ref().map_err(|e| e.clone())?;
    let mut l = Launch::default();
    l.grid = d.grid;
    l.wg = d.wg;
    l.regs = (r.gprs * 2).max(8);
    l.tg_bytes = d.shared_bytes;
    l.uniforms = d.uniforms.clone();
    l.regions = mem.regions.iter().map(|(va, b)| (*va, b.len() as u64)).collect();
    // the runtime profile (agx_dispatch --launch-profile runtime): back-to-back commands
    (l.launch_ns, l.wg_launch_ns, l.group_launch_ns, l.fetch_line_ns, l.stack_launch_ns, l.store_gbps,
     l.store_launch_ns, l.later_waves_warm, l.partial_warm, l.chain_waves) =
        (2300.0, 100.0, 3.2, 70.0, 5000.0, 365.0, 700.0, true, true, true);
    l.fetch_jump_line_ns = 300.0;
    l.stack_slot_ns = 1150.0;
    l.icache_lines = 192;
    l.icache_victims = 192;
    l.validate()?;
    let prepared = func::PreparedDispatch::new(d)?;
    let tracer = |wg: [u32; 3]| prepared.trace_workgroup(mem, wg);
    let rep = dispatch::simulate_with(&d.code, &l, p, Some(&tracer))?;
    Ok((rep.time_ns / 1000.0, rep.core_cycles))
}
