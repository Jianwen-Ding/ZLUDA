//! Constant-memory specialization (`ZLUDA_AGX_CONST_SPEC`, see the AGX docs): the kernels of a PTX module that has
//! host-written `.const` variables (declared without an initializer, written through `cuModuleGetGlobal`'s address,
//! as CUDA's `__constant__` parameters are) are compiled once more with those variables' current contents folded in
//! (`specialize::fold_const_loads`): a load at a constant offset becomes the value it reads, and the branches and
//! code that depend on it fold. Each launch compares the variables' bytes with those of the code it would run, so
//! the folded code only runs while it computes what the generic code computes; until a kernel's folded code is
//! compiled (in the background) and whenever the bytes differ, the generic code runs.
//!
//! A large module (Blender Cycles) gets one folded copy of its PTX per set of bytes, whose front end runs once
//! (portable AIR, `compile_ptx_kernel`), and each kernel's code from it on first use.

use crate::driver::{compile_ptx, compile_ptx_kernel, CompileSlot};
use agx_runtime::LoadedModule;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A set of bytes is folded once it has been seen at this many launches in a row (a variable rewritten before
/// every launch is never folded).
const STABLE_AFTER: u32 = 2;
/// At most this many sets of bytes per module; their code is never freed (a launch may still be running it).
const MAX_VARIANTS: usize = 4;

/// `ZLUDA_AGX_CONST_SPEC`: `0` off; `scalars` folds values up to 4 bytes only (8-byte values are mostly device
/// addresses); `wait`, for checking the folded code, folds all from the first launch and makes each launch wait
/// for its kernel's folded code; anything else, the default, folds all.
pub(crate) fn mode() -> Option<bool> {
    match std::env::var("ZLUDA_AGX_CONST_SPEC").as_deref() {
        Ok("0") => None,
        Ok("scalars") => Some(false),
        _ => Some(true),
    }
}

fn wait() -> bool {
    static WAIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *WAIT.get_or_init(|| std::env::var("ZLUDA_AGX_CONST_SPEC").is_ok_and(|v| v == "wait"))
}

type Slot = Arc<Mutex<Option<Result<(Vec<u8>, String), String>>>>;

enum KernelState {
    Pending(Slot),
    Ready(LoadedModule),
    Failed,
}

/// The folded PTX of one set of bytes, made by the first compilation that needs it, and for a large module the
/// directory of its portable AIR, whose first compilation (the front end) runs before the others start.
struct Folded {
    ptx: std::sync::OnceLock<Arc<str>>,
    portable: Option<tempfile::TempDir>,
    front_end: Mutex<()>,
}

struct Variant {
    bytes: Vec<u8>,
    folded: Arc<Folded>,
    kernels: HashMap<String, KernelState>,
}

pub(crate) struct ConstSpec {
    ptx: Arc<str>,
    /// name, offset in the module's global data, size
    vars: Vec<(String, usize, usize)>,
    large: bool,
    all: bool,
    cpu: String,
    variants: Vec<Variant>,
    current: Option<usize>,
    candidate: Option<(Vec<u8>, u32)>,
}

impl ConstSpec {
    /// The state of a module of PTX `ptx`, whose global data has `symbols` (name -> offset, size); `None` when it
    /// has no host-written `.const` variable.
    pub(crate) fn new(
        ptx: Arc<str>,
        symbols: &HashMap<String, (u64, u64)>,
        large: bool,
        cpu: String,
    ) -> Option<ConstSpec> {
        let all = mode()?;
        let vars: Vec<_> = crate::specialize::host_written_consts(&ptx)
            .into_iter()
            .filter_map(|(name, size)| {
                let &(off, sym_size) = symbols.get(&name)?;
                (sym_size as usize == size).then_some((name, off as usize, size))
            })
            .collect();
        if vars.is_empty() {
            return None;
        }
        if crate::log_enabled() {
            eprintln!("[zluda_agx] const specialization: {:?}", vars.iter().map(|(n, _, s)| (n, s)).collect::<Vec<_>>());
        }
        Some(ConstSpec { ptx, vars, large, all, cpu, variants: Vec::new(), current: None, candidate: None })
    }

    fn matches(&self, v: usize, globals: &[u8]) -> bool {
        let mut at = 0;
        let bytes = &self.variants[v].bytes;
        self.vars.iter().all(|(_, off, size)| {
            let same = globals.get(*off..off + size) == bytes.get(at..at + size);
            at += size;
            same
        })
    }

    /// The folded code of `name` for the current contents of the variables (`globals`, the module's global data)
    /// if it is compiled and can run `threads` threads per threadgroup; starts its compilation otherwise.
    pub(crate) fn code(&mut self, name: &str, globals: &[u8], threads: u64, load: impl FnOnce(&[u8], &str) -> Option<LoadedModule>) -> Option<&LoadedModule> {
        let v = match self.current.filter(|&v| self.matches(v, globals)) {
            Some(v) => v,
            None => self.switch(globals)?,
        };
        let variant = &mut self.variants[v];
        let state = variant.kernels.entry(name.to_string()).or_insert_with(|| {
            KernelState::Pending(start(variant.folded.clone(), &self.ptx, &self.vars, &variant.bytes, self.all, name, self.large, &self.cpu))
        });
        if let KernelState::Pending(slot) = state {
            if wait() {
                while slot.lock().unwrap_or_else(|e| e.into_inner()).is_none() {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
            }
            let done = slot.lock().unwrap_or_else(|e| e.into_inner()).take();
            match done {
                None => return None,
                Some(Ok((object, asm))) => {
                    *state = match load(&object, &asm) {
                        Some(l) if l.module.kernels.contains_key(name) => {
                            if crate::log_enabled() {
                                let (g, f) = (l.module.kernels[name].gprs, l.module.kernels[name].stack_bytes);
                                eprintln!("[zluda_agx] const-specialized {name}: ready ({g} registers, {f} B stack)");
                            }
                            KernelState::Ready(l)
                        }
                        _ => {
                            eprintln!("[zluda_agx] const-specialized {name}: failed to load");
                            KernelState::Failed
                        }
                    }
                }
                Some(Err(e)) => {
                    eprintln!("[zluda_agx] const specialization of {name} failed: {e}");
                    *state = KernelState::Failed;
                }
            }
        }
        match &self.variants[v].kernels[name] {
            KernelState::Ready(l) if l.module.kernels[name].max_threads_per_threadgroup() as u64 >= threads => Some(l),
            _ => None,
        }
    }

    /// The folded code of `name` that `code` last found ready.
    pub(crate) fn ready(&self, name: &str) -> Option<&LoadedModule> {
        match self.variants.get(self.current?)?.kernels.get(name)? {
            KernelState::Ready(l) => Some(l),
            _ => None,
        }
    }

    /// The variant of the variables' current bytes, if they have been stable for long enough (made if new).
    fn switch(&mut self, globals: &[u8]) -> Option<usize> {
        self.current = None;
        if let Some(v) = (0..self.variants.len()).find(|&v| self.matches(v, globals)) {
            self.current = Some(v);
            return self.current;
        }
        let mut bytes = Vec::new();
        for (_, off, size) in &self.vars {
            bytes.extend_from_slice(globals.get(*off..off + size)?);
        }
        let seen = match self.candidate.take() {
            Some((b, n)) if b == bytes => n + 1,
            _ => 1,
        };
        if (seen < STABLE_AFTER && !wait()) || self.variants.len() >= MAX_VARIANTS {
            self.candidate = Some((bytes, seen));
            return None;
        }
        let portable = if self.large { tempfile::tempdir().ok() } else { None };
        let folded = Arc::new(Folded { ptx: Default::default(), portable, front_end: Mutex::new(()) });
        self.variants.push(Variant { bytes, folded, kernels: HashMap::new() });
        if crate::log_enabled() {
            eprintln!("[zluda_agx] const specialization: new set of bytes ({} so far)", self.variants.len());
        }
        self.current = Some(self.variants.len() - 1);
        self.current
    }
}

/// Starts compiling `name`'s folded code in the background.
#[allow(clippy::too_many_arguments)]
fn start(folded: Arc<Folded>, ptx: &Arc<str>, vars: &[(String, usize, usize)], bytes: &[u8], all: bool, name: &str, large: bool, cpu: &str) -> Slot {
    let slot: Slot = Arc::new(Mutex::new(None));
    let (out, ptx, name, cpu, bytes) = (slot.clone(), ptx.clone(), name.to_string(), cpu.to_string(), bytes.to_vec());
    let vars = vars.to_vec();
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(|| {
            let text = folded
                .ptx
                .get_or_init(|| {
                    let mut at = 0;
                    let values: Vec<(&str, Vec<u8>)> = vars
                        .iter()
                        .map(|(n, _, size)| {
                            let b = bytes[at..at + size].to_vec();
                            at += size;
                            (n.as_str(), b)
                        })
                        .collect();
                    let refs: Vec<(&str, &[u8])> = values.iter().map(|(n, b)| (*n, &b[..])).collect();
                    let (text, n) = crate::specialize::fold_const_loads(&ptx, &refs, if all { 8 } else { 4 });
                    if crate::log_enabled() {
                        eprintln!("[zluda_agx] const specialization: {n} loads folded");
                    }
                    Arc::from(text)
                })
                .clone();
            match &folded.portable {
                Some(dir) => {
                    let path = dir.path().join("module.portable.air");
                    // the first compilation runs the front end and keeps the portable AIR; the others wait for it
                    let first = folded.front_end.lock().unwrap_or_else(|e| e.into_inner());
                    if path.exists() {
                        drop(first);
                    }
                    let _slot = CompileSlot::take();
                    compile_ptx_kernel(text.as_bytes(), &name, &path, &cpu)
                }
                None if large => Err("no directory for the portable AIR".into()),
                None => {
                    let _slot = CompileSlot::take();
                    compile_ptx(crate::specialize::without_other_entries(&text, &name).as_bytes(), &cpu)
                }
            }
        })
        .unwrap_or_else(|_| Err(format!("{name}: const specialization panicked")));
        *out.lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
    });
    slot
}
