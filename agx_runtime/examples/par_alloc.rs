//! Times creating `TOTAL` bytes of GEM objects as one object, or as chunks created on several threads (each thread
//! with its own render-node fd; GEM_CREATE allocates and zeroes the pages), then a first touch of every page.
//!     cargo run --release -p agx_runtime --example par_alloc [TOTAL_MIB] [THREADS] [CHUNK_MIB]
use asahi_sys::*;
use std::time::Instant;
const DRM_IOCTL_GEM_CLOSE: u64 = 0x4008_6409;

fn create(fd: i32, size: u64) -> u32 {
    let mut gc = drm_asahi_gem_create { size, flags: DRM_ASAHI_GEM_WRITEBACK as u32, vm_id: 0, handle: 0, pad: 0 };
    let r = unsafe { libc::ioctl(fd, DRM_IOCTL_ASAHI_GEM_CREATE as _, &mut gc) };
    assert_eq!(r, 0, "gem create {}", std::io::Error::last_os_error());
    if std::env::var_os("MO").is_some() {
        let mut mo = drm_asahi_gem_mmap_offset { handle: gc.handle, flags: 0, offset: 0 };
        let r = unsafe { libc::ioctl(fd, DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET as _, &mut mo) };
        assert_eq!(r, 0);
        if std::env::var_os("MM").is_some() {
            let m = unsafe { libc::mmap(std::ptr::null_mut(), size as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, mo.offset as _) };
            assert!(m != libc::MAP_FAILED);
            if std::env::var_os("TOUCH").is_some() { for o in (0..size as usize).step_by(16384) { unsafe { *(m as *mut u8).add(o) = 1 } } }
            unsafe { libc::munmap(m, size as usize) };
        }
    }
    gc.handle
}

fn open() -> i32 {
    unsafe { libc::open(c"/dev/dri/renderD128".as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) }
}

fn main() {
    let a: Vec<u64> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let total = a.first().copied().unwrap_or(4096) << 20;
    let threads = a.get(1).copied().unwrap_or(8);
    let chunk = a.get(2).copied().unwrap_or(256) << 20;
    for _ in 0..3 {
        let fd = open();
        let t = Instant::now();
        let h = create(fd, total);
        let one = t.elapsed();
        let mut gem_close = drm_gem_close { handle: h, pad: 0 };
        let t = Instant::now();
        unsafe { libc::ioctl(fd, DRM_IOCTL_GEM_CLOSE as _, &mut gem_close) };
        let close = t.elapsed();
        unsafe { libc::close(fd) };
        let n = total / chunk;
        let t = Instant::now();
        let hs: Vec<(i32, Vec<u32>)> = std::thread::scope(|s| {
            let js: Vec<_> = (0..threads)
                .map(|i| s.spawn(move || { let fd = open(); let v: Vec<u32> = (0..n).filter(|c| c % threads == i).map(|_| create(fd, chunk)).collect(); (fd, v) }))
                .collect();
            js.into_iter().map(|j| j.join().unwrap()).collect()
        });
        let par = t.elapsed();
        let t = Instant::now();
        std::thread::scope(|s| {
            for (fd, v) in &hs {
                let (fd, v) = (*fd, v.clone());
                s.spawn(move || { for h in v { let mut c = drm_gem_close { handle: h, pad: 0 }; unsafe { libc::ioctl(fd, DRM_IOCTL_GEM_CLOSE as _, &mut c) }; } unsafe { libc::close(fd) }; });
            }
        });
        let pclose = t.elapsed();
        println!("{} MiB: one object {:.1} ms (close {:.1}); {n} x {} MiB on {threads} threads {:.1} ms (close {:.1})",
            total >> 20, one.as_secs_f64() * 1e3, close.as_secs_f64() * 1e3, chunk >> 20, par.as_secs_f64() * 1e3, pclose.as_secs_f64() * 1e3);
    }
}
