//! A CUDA-style context on one AGX device: its GPU address space, a default stream, device allocations
//! (`cuMemAlloc` / `cuMemFree`), copies and fills, and the device attributes `cuDeviceGetAttribute` reports.
//!
//! Apple GPUs share memory with the CPU and the buffer objects are mapped write-back, so copies are CPU copies of the
//! mapped bytes after the work that might touch them has finished. CPU writes made while a kernel runs are only
//! guaranteed to be visible to later launches (ptx-re-blockers B7).

use crate::*;
use std::collections::BTreeMap;

/// Device attributes derived from the kernel's parameters and the measured G14 limits.
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceProps {
    pub name: String,
    pub llvm_cpu: String,
    pub cores: u32,
    pub clock_khz: u32,
    pub warp_size: u32,
    pub max_threads_per_block: u32,
    pub max_block_dim: [u32; 3],
    pub max_grid_dim: [u32; 3],
    pub shared_memory_per_block: u32,
    pub registers_per_thread: u32,
    pub total_memory: u64,
}

impl DeviceProps {
    pub fn from_params(p: &drm_asahi_params_global, total_memory: u64) -> DeviceProps {
        let variant = (p.gpu_variant as u8 as char).to_ascii_uppercase();
        let cores = p.core_masks.iter().map(|m| m.count_ones()).sum();
        let marketing = match (p.gpu_generation, variant) {
            (13, 'G') => "Apple M1",
            (13, 'S') => "Apple M1 Pro",
            (13, 'C') => "Apple M1 Max",
            (13, 'D') => "Apple M1 Ultra",
            (14, 'G') => "Apple M2",
            (14, 'S') => "Apple M2 Pro",
            (14, 'C') => "Apple M2 Max",
            (14, 'D') => "Apple M2 Ultra",
            _ => "Apple GPU",
        };
        DeviceProps {
            name: format!("{marketing} (G{}{variant}, {cores} cores)", p.gpu_generation),
            llvm_cpu: format!("agx-g{}{}", p.gpu_generation, variant.to_ascii_lowercase()),
            cores,
            clock_khz: p.max_frequency_khz,
            warp_size: 32,
            // Measured on the M2 Max (AGX Performance Simulator tasks/ptx-re-blockers.md, B22): threadgroups of 1024
            // threads along x, y or z run; a 2048-thread threadgroup runs nothing.
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 1024],
            // The CDM global size words are 32-bit thread counts per dimension (2^32 - 32 threads along x run), so a
            // launch also needs grid * block < 2^32 in each dimension. y and z take more than CUDA's 65535 CTAs.
            max_grid_dim: [i32::MAX as u32, 65535, 65535],
            shared_memory_per_block: 65536,
            registers_per_thread: 128,
            total_memory,
        }
    }
}

fn system_memory() -> u64 {
    let mut s: libc::sysinfo = unsafe { std::mem::zeroed() };
    if unsafe { libc::sysinfo(&mut s) } != 0 {
        return 0;
    }
    s.totalram as u64 * s.mem_unit as u64
}

/// The memory limit of the process's cgroup (v2 memory.max) and its current use, if it has a limit.
fn cgroup_memory() -> Option<(u64, u64)> {
    let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = cgroup.lines().find_map(|l| l.strip_prefix("0::"))?;
    let dir = std::path::Path::new("/sys/fs/cgroup").join(path.trim_start_matches('/'));
    let max = std::fs::read_to_string(dir.join("memory.max")).ok()?.trim().parse::<u64>().ok()?;
    let current = std::fs::read_to_string(dir.join("memory.current")).ok()?.trim().parse::<u64>().unwrap_or(0);
    Some((max, current))
}

/// The allocation containing `[va, va + len)`, by its start address.
fn find<T>(allocs: &BTreeMap<u64, T>, va: u64, len: u64, size: impl Fn(&T) -> u64) -> Option<u64> {
    let (&start, a) = allocs.range(..=va).next_back()?;
    (va.checked_add(len)? <= start + size(a)).then_some(start)
}

/// The memory of an allocation: an object of its own, or part of a physical allocation mapped at the address
/// (the virtual memory management API: cuMemCreate, cuMemMap).
enum Memory {
    Bo(Bo),
    // the mapping first: fields drop in order, and the binding goes before the object can
    Mapped { mapping: crate::Mapping, phys: std::rc::Rc<Bo>, offset: u64 },
}

impl Memory {
    fn unified(&self) -> bool {
        match self {
            Memory::Bo(bo) => bo.unified(),
            Memory::Mapped { .. } => false,
        }
    }

    fn as_slice(&self) -> &[u8] {
        match self {
            Memory::Bo(bo) => bo.as_slice(),
            Memory::Mapped { phys, offset, mapping } => unsafe {
                std::slice::from_raw_parts(phys.map.add(*offset as usize), mapping.size as usize)
            },
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            Memory::Bo(bo) => bo.as_mut_slice(),
            // the physical allocation's pages, which other mappings may also show
            Memory::Mapped { phys, offset, mapping } => unsafe {
                std::slice::from_raw_parts_mut(phys.map.add(*offset as usize), mapping.size as usize)
            },
        }
    }
}

/// A copy between CPU-mapped memory and host memory (non-overlapping), on several threads from 4 MiB: one thread
/// moves ~30-35 GB/s, under the memory system's rate. ZLUDA_AGX_COPY_THREADS sets the most threads (1: one).
fn par_copy(dst: &mut [u8], src: &[u8]) {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let most = *THREADS.get_or_init(|| {
        std::env::var("ZLUDA_AGX_COPY_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4).max(1)
    });
    let n = dst.len().min(src.len());
    let threads = most.min(n >> 22).max(1);
    if threads == 1 {
        dst[..n].copy_from_slice(&src[..n]);
        return;
    }
    // chunks on 64-byte boundaries
    let part = (n / threads).div_ceil(64) * 64;
    std::thread::scope(|sc| {
        let mut rest = (&mut dst[..n], &src[..n]);
        while !rest.0.is_empty() {
            let take = part.min(rest.0.len());
            let (d, dr) = std::mem::take(&mut rest.0).split_at_mut(take);
            let (s, sr) = rest.1.split_at(take);
            rest = (dr, sr);
            if rest.0.is_empty() {
                d.copy_from_slice(s);
            } else {
                sc.spawn(move || d.copy_from_slice(s));
            }
        }
    });
}

/// The reuse cache's limit in bytes: ZLUDA_AGX_ALLOC_CACHE (MiB; 0 turns it off), 512 MiB by default.
fn cache_limit() -> u64 {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| std::env::var("ZLUDA_AGX_ALLOC_CACHE").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(512) << 20)
}

/// Fields drop in declaration order: the allocations and the queue are unbound before the VM is destroyed (a VM
/// destroyed with objects still bound strands their memory in the kernel, also past process exit; B42).
pub struct Context {
    allocs: BTreeMap<u64, (Memory, u64)>,
    /// Freed allocations kept for reuse by `mem_alloc_cached` (oldest first): creating, binding and mapping an object
    /// cost 17 us at 4 KiB, 160 us at 1 MiB and 6.6 ms at 64 MiB (allocation and free). At most `cache_limit()`
    /// bytes; emptied when an allocation fails; counted as free by `mem_info`.
    free_cache: Vec<Bo>,
    /// The addresses of live allocations from `mem_alloc_cached`, which return to the cache when freed.
    cacheable: std::collections::HashSet<u64>,
    pub stream: Queue,
    pub vm: Vm,
    pub props: DeviceProps,
    pub device: Device,
}

fn invalid(m: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, m.into())
}

impl Context {
    pub fn new(node: &str) -> io::Result<Context> {
        let device = Device::open(node)?;
        let props = DeviceProps::from_params(&device.params, system_memory());
        let mut vm = Vm::new(&device)?;
        let stream = Queue::new(&mut vm)?;
        Ok(Context { device, props, vm, stream, allocs: BTreeMap::new(), free_cache: Vec::new(), cacheable: Default::default() })
    }

    /// cuMemAlloc: `size` bytes, zeroed, 16 KiB aligned.
    pub fn mem_alloc(&mut self, size: u64) -> io::Result<u64> {
        let bo = self.vm.alloc_data(size)?;
        let va = bo.va;
        self.allocs.insert(va, (Memory::Bo(bo), size.max(1)));
        Ok(va)
    }

    /// cuMemMap: `size` bytes of the physical allocation `phys` from `offset`, at `va` (in a range from
    /// `vm.reserve`).
    pub fn mem_map(&mut self, va: u64, phys: &std::rc::Rc<Bo>, offset: u64, size: u64) -> io::Result<()> {
        let end = va.checked_add(size).filter(|_| size != 0).ok_or_else(|| invalid("empty or overflowing mapping"))?;
        // Test actual bytes: a zero-length query at the previous allocation's
        // end would incorrectly reject an adjacent mapping.
        if find(&self.allocs, va, 1, |a| a.1).is_some() || find(&self.allocs, end - 1, 1, |a| a.1).is_some()
            || self.allocs.range(va..end).next().is_some()
        {
            return Err(invalid(format!("{va:#x}+{size} overlaps a mapping")));
        }
        let mapping = self.vm.map_at(phys, offset, va, size)?;
        self.allocs.insert(va, (Memory::Mapped { mapping, phys: phys.clone(), offset }, size));
        Ok(())
    }

    /// cuMemUnmap: the mappings in `[va, va + size)` (each wholly inside it), after the default stream.
    pub fn mem_unmap(&mut self, va: u64, size: u64) -> io::Result<()> {
        let starts: Vec<u64> = self.allocs.range(va..va + size).map(|(&s, _)| s).collect();
        if starts.is_empty()
            || starts.iter().any(|s| !matches!(self.allocs[s].0, Memory::Mapped { .. }) || s + self.allocs[s].1 > va + size)
        {
            return Err(invalid(format!("{va:#x}+{size} is not a range of mappings")));
        }
        self.stream.synchronize()?;
        for s in starts {
            self.allocs.remove(&s);
        }
        Ok(())
    }

    /// The physical allocation mapped at `va`, if any (cuMemRetainAllocationHandle).
    pub fn mapped_allocation(&self, va: u64) -> Option<&std::rc::Rc<Bo>> {
        let start = find(&self.allocs, va, 0, |a| a.1)?;
        match &self.allocs[&start].0 {
            Memory::Mapped { phys, .. } => Some(phys),
            Memory::Bo(_) => None,
        }
    }

    /// The physical allocation mapped at `va` and the offset of `va` in it (cuMulticastBindAddr).
    pub fn mapped_location(&self, va: u64) -> Option<(std::rc::Rc<Bo>, u64)> {
        let start = find(&self.allocs, va, 0, |a| a.1)?;
        match &self.allocs[&start].0 {
            Memory::Mapped { phys, offset, .. } => Some((phys.clone(), offset + (va - start))),
            Memory::Bo(_) => None,
        }
    }

    /// Takes `bo` (bound) as an allocation of `size` bytes at its address, which mem_free releases: an imported
    /// dma-buf (cuIpcOpenMemHandle), registered host memory (cuMemHostRegister).
    pub fn insert_bo(&mut self, bo: Bo, size: u64) -> u64 {
        let va = bo.va;
        self.allocs.insert(va, (Memory::Bo(bo), size.max(1)));
        va
    }

    /// A dma-buf of the object behind `va`, with the offset of `va` in it and its size (cuIpcGetMemHandle).
    pub fn export_dmabuf(&self, va: u64) -> io::Result<(std::os::fd::OwnedFd, u64, u64)> {
        let start = find(&self.allocs, va, 0, |a| a.1).ok_or_else(|| invalid(format!("{va:#x} is not allocated")))?;
        match &self.allocs[&start].0 {
            Memory::Bo(bo) => Ok((bo.export_dmabuf()?, va - start, bo.size)),
            Memory::Mapped { phys, offset, .. } => Ok((phys.export_dmabuf()?, offset + (va - start), phys.size)),
        }
    }

    /// cuMemFree: waits for the default stream first, as CUDA does. A data object that `mem_alloc_cached` made goes
    /// to the reuse cache (idle: nothing queued uses it after the wait).
    pub fn mem_free(&mut self, va: u64) -> io::Result<()> {
        if !self.allocs.get(&va).is_some_and(|a| matches!(a.0, Memory::Bo(_))) {
            return Err(invalid(format!("{va:#x} is not an allocation")));
        }
        self.stream.synchronize()?;
        if let Some((Memory::Bo(bo), _)) = self.allocs.remove(&va) {
            let limit = cache_limit();
            if self.cacheable.remove(&va) && bo.size <= limit / 2 {
                self.free_cache.push(bo);
                // the oldest go first
                while self.free_cache.iter().map(|b| b.size).sum::<u64>() > limit {
                    self.free_cache.remove(0);
                }
            }
        }
        Ok(())
    }

    /// cuMemAlloc through the reuse cache: a freed data object of between `size` and 1.25 `size` bytes (the
    /// smallest), else a new one (after emptying the cache if the first attempt fails). A reused object keeps its old
    /// contents, as CUDA's allocations may (AGX_ZERO_ALLOC=1 zeroes it). ZLUDA_AGX_ALLOC_CACHE=0: no cache.
    pub fn mem_alloc_cached(&mut self, size: u64) -> io::Result<u64> {
        let need = size.max(1).div_ceil(PAGE) * PAGE;
        let best = (0..self.free_cache.len())
            .filter(|&i| (need..=need + need / 4).contains(&self.free_cache[i].size))
            .min_by_key(|&i| self.free_cache[i].size);
        if let Some(i) = best {
            let mut bo = self.free_cache.remove(i);
            if crate::zero_alloc() {
                bo.as_mut_slice().fill(0);
            }
            let va = bo.va;
            self.allocs.insert(va, (Memory::Bo(bo), size.max(1)));
            self.cacheable.insert(va);
            return Ok(va);
        }
        let bo = match self.vm.alloc_data(size) {
            Err(_) if !self.free_cache.is_empty() => {
                self.free_cache.clear();
                self.vm.alloc_data(size)?
            }
            r => r?,
        };
        let va = bo.va;
        self.allocs.insert(va, (Memory::Bo(bo), size.max(1)));
        if cache_limit() > 0 {
            self.cacheable.insert(va);
        }
        Ok(va)
    }

    /// cuMemAllocManaged: an allocation whose GPU address is also its CPU address. Fails if that CPU range is taken.
    pub fn mem_alloc_unified(&mut self, size: u64) -> io::Result<u64> {
        let bo = self.vm.alloc_data(size)?;
        if !bo.unified() {
            return Err(io::Error::new(io::ErrorKind::Unsupported, "the CPU range of the GPU address is in use"));
        }
        let va = bo.va;
        self.allocs.insert(va, (Memory::Bo(bo), size.max(1)));
        Ok(va)
    }

    /// Whether `va` is in an allocation that the CPU can use at the same address.
    pub fn is_unified(&self, va: u64) -> bool {
        find(&self.allocs, va, 0, |a| a.1).is_some_and(|s| self.allocs[&s].0.unified())
    }

    /// The CPU address of `len` bytes of device memory at `va`, for copies the caller does itself (2D copies).
    /// Valid until the allocation is freed.
    pub fn cpu_ptr(&mut self, va: u64, len: u64) -> io::Result<*mut u8> {
        Ok(self.bytes_mut(va, len)?.as_mut_ptr())
    }

    /// Waits for the default stream (before the CPU touches memory the GPU may use).
    pub fn synchronize(&mut self) -> io::Result<()> {
        self.stream.synchronize()
    }

    /// The current contents of the allocation `va` points into, and its base, without waiting for the GPU (a dump
    /// after a fault).
    pub fn allocation(&self, va: u64) -> Option<(u64, &[u8])> {
        let start = find(&self.allocs, va, 0, |a| a.1)?;
        let (bo, size) = &self.allocs[&start];
        Some((start, &bo.as_slice()[..*size as usize]))
    }

    /// `[lo, hi)` of every allocation, in address order.
    pub fn allocation_ranges(&self) -> Vec<(u64, u64)> {
        self.allocs.iter().map(|(&va, (_, size))| (va, va + size)).collect()
    }

    /// The allocation's base and size for a pointer into it (cuMemGetAddressRange).
    pub fn address_range(&self, va: u64) -> Option<(u64, u64)> {
        let start = find(&self.allocs, va, 0, |a| a.1)?;
        Some((start, self.allocs[&start].1))
    }

    fn bytes(&self, va: u64, len: u64) -> io::Result<&[u8]> {
        let start = find(&self.allocs, va, len, |a| a.1).ok_or_else(|| invalid(format!("{va:#x}+{len} not allocated")))?;
        let off = (va - start) as usize;
        Ok(&self.allocs[&start].0.as_slice()[off..off + len as usize])
    }

    fn bytes_mut(&mut self, va: u64, len: u64) -> io::Result<&mut [u8]> {
        let start = find(&self.allocs, va, len, |a| a.1).ok_or_else(|| invalid(format!("{va:#x}+{len} not allocated")))?;
        let off = (va - start) as usize;
        Ok(&mut self.allocs.get_mut(&start).unwrap().0.as_mut_slice()[off..off + len as usize])
    }

    /// cuMemcpyHtoD (synchronous with respect to the default stream).
    pub fn memcpy_htod(&mut self, dst: u64, src: &[u8]) -> io::Result<()> {
        self.stream.synchronize()?;
        par_copy(self.bytes_mut(dst, src.len() as u64)?, src);
        Ok(())
    }

    /// cuMemcpyDtoH.
    pub fn memcpy_dtoh(&mut self, dst: &mut [u8], src: u64) -> io::Result<()> {
        self.stream.synchronize()?;
        par_copy(dst, self.bytes(src, dst.len() as u64)?);
        Ok(())
    }

    /// cuMemcpyDtoD (the regions may be in the same or different allocations, and may overlap).
    pub fn memcpy_dtod(&mut self, dst: u64, src: u64, len: u64) -> io::Result<()> {
        self.stream.synchronize()?;
        // one memmove, overlapping or not (a temporary copy held a second buffer the size of the copy)
        let from = self.bytes(src, len)?.as_ptr();
        let to = self.bytes_mut(dst, len)?.as_mut_ptr();
        if from as usize + len as usize <= to as usize || to as usize + len as usize <= from as usize {
            par_copy(unsafe { std::slice::from_raw_parts_mut(to, len as usize) }, unsafe { std::slice::from_raw_parts(from, len as usize) });
        } else {
            unsafe { std::ptr::copy(from, to, len as usize) };
        }
        Ok(())
    }

    /// cuMemsetD8 / D16 / D32: `count` copies of `pattern` (1, 2 or 4 bytes).
    pub fn memset(&mut self, dst: u64, pattern: &[u8], count: u64) -> io::Result<()> {
        self.stream.synchronize()?;
        let n = pattern.len() as u64;
        fill_pattern(self.bytes_mut(dst, count * n)?, pattern);
        Ok(())
    }

    /// Reads a NUL-terminated string from device memory (for `%s` in device `printf`).
    pub fn read_c_string(&self, va: u64) -> Option<Vec<u8>> {
        let start = find(&self.allocs, va, 0, |a| a.1)?;
        let (bo, size) = &self.allocs[&start];
        let s = &bo.as_slice()[(va - start) as usize..*size as usize];
        Some(s[..s.iter().position(|&c| c == 0)?].to_vec())
    }

    /// cuMemGetInfo: (free, total). The GPU shares system memory and the UAPI reports no figures of its own
    /// (ptx-re-blockers B39): free is the system's available memory (MemAvailable, which counts reclaimable
    /// caches), both within the process's cgroup limit if it has one.
    pub fn mem_info(&self) -> (u64, u64) {
        let cached: u64 = self.free_cache.iter().map(|b| b.size).sum();
        let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let field = |name: &str| {
            meminfo
                .lines()
                .find_map(|l| l.strip_prefix(name)?.trim().strip_suffix("kB")?.trim().parse::<u64>().ok())
                .map(|kb| kb * 1024)
        };
        let mut s: libc::sysinfo = unsafe { std::mem::zeroed() };
        let free = field("MemAvailable:").unwrap_or_else(|| {
            if unsafe { libc::sysinfo(&mut s) } == 0 { s.freeram as u64 * s.mem_unit as u64 } else { 0 }
        });
        let mut total = self.props.total_memory;
        let mut free = free.min(total);
        if let Some((limit, used)) = cgroup_memory() {
            total = total.min(limit);
            free = free.min(limit.saturating_sub(used));
        }
        // the reuse cache's objects are free to the application
        ((free + cached).min(total), total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_allocation_of_a_pointer() {
        let a: BTreeMap<u64, u64> = [(0x1000, 0x100), (0x4000, 0x10)].into();
        let size = |s: &u64| *s;
        assert_eq!(find(&a, 0x1000, 0x100, size), Some(0x1000));
        assert_eq!(find(&a, 0x10ff, 1, size), Some(0x1000));
        assert_eq!(find(&a, 0x10ff, 2, size), None);
        assert_eq!(find(&a, 0x0fff, 1, size), None);
        assert_eq!(find(&a, 0x4008, 8, size), Some(0x4000));
        assert_eq!(find(&a, 0x2000, 0, size), None);
        assert_eq!(find(&a, u64::MAX, 2, size), None);
    }

    #[test]
    fn m2_max_properties() {
        let mut p: drm_asahi_params_global = unsafe { std::mem::zeroed() };
        p.gpu_generation = 14;
        p.gpu_variant = b'C' as u32;
        p.max_frequency_khz = 1_398_000;
        // 38 cores in clusters of 10, 9, 9 and 10
        p.core_masks[..4].copy_from_slice(&[0x3ff, 0x1ff, 0x1ff, 0x3ff]);
        let d = DeviceProps::from_params(&p, 64 << 30);
        assert_eq!(d.name, "Apple M2 Max (G14C, 38 cores)");
        assert_eq!(d.llvm_cpu, "agx-g14c");
        assert_eq!((d.cores, d.clock_khz, d.warp_size), (38, 1_398_000, 32));
    }
}

/// Fills `bytes` with copies of `pattern` (1, 2 or 4 bytes; `bytes` a multiple of it) as `fill` on bytes, halves or
/// words: a copy of the pattern per element, its length known only at run time, ran at ~0.45 GB/s (a 384 MiB
/// cudaMemsetAsync of llama.cpp's KV cache took 870 ms).
pub fn fill_pattern(bytes: &mut [u8], pattern: &[u8]) {
    fn typed<T: Copy>(bytes: &mut [u8], v: T, pattern: &[u8]) {
        // SAFETY: any bit pattern is a valid u16 / u32; head and tail are the unaligned bytes
        let (head, mid, tail) = unsafe { bytes.align_to_mut::<T>() };
        let n = pattern.len();
        // the elements are at multiples of n from the start: the head's bytes continue the pattern from its start
        for (i, b) in head.iter_mut().enumerate() {
            *b = pattern[i % n];
        }
        if head.len() % n != 0 {
            // the aligned part starts inside an element: rotate the value by the head's phase
            let phase = head.len() % n;
            let rotated: Vec<u8> = (0..n).map(|i| pattern[(i + phase) % n]).collect();
            let mut w = v;
            unsafe { std::ptr::copy_nonoverlapping(rotated.as_ptr(), &mut w as *mut T as *mut u8, n) };
            mid.fill(w);
        } else {
            mid.fill(v);
        }
        let start = head.len() + mid.len() * n;
        for (i, b) in tail.iter_mut().enumerate() {
            *b = pattern[(start + i) % n];
        }
    }
    match *pattern {
        [b] => bytes.fill(b),
        [a, b] => typed(bytes, u16::from_ne_bytes([a, b]), pattern),
        [a, b, c, d] => typed(bytes, u32::from_ne_bytes([a, b, c, d]), pattern),
        _ => {
            for chunk in bytes.chunks_exact_mut(pattern.len()) {
                chunk.copy_from_slice(pattern);
            }
        }
    }
}

#[cfg(test)]
mod fill_tests {
    #[test]
    fn fill_pattern_matches_elementwise() {
        for pattern in [&[7u8][..], &[1, 2], &[1, 2, 3, 4], &[9, 9, 9, 9]] {
            for offset in 0..4 {
                for count in [0usize, 1, 2, 3, 7, 64, 1000, 4097] {
                    let mut buf = vec![0u8; offset + count * pattern.len() + 3];
                    let fast = &mut buf[offset..offset + count * pattern.len()];
                    super::fill_pattern(fast, pattern);
                    let slow: Vec<u8> = pattern.iter().copied().cycle().take(count * pattern.len()).collect();
                    assert_eq!(fast, &slow[..], "pattern {pattern:?} offset {offset} count {count}");
                }
            }
        }
    }
}
