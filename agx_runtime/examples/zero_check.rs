//! Checks that new GEM objects read as zeros without the CPU fill (AGX_ZERO_ALLOC), recycled pages included: each
//! round allocates objects, checks them, writes them with a pattern and frees them, so the next round's objects
//! come from the pages just freed. Prints the allocation time of each size.
//!
//!     cargo run --release -p agx_runtime --example zero_check [ROUNDS]
use std::time::Instant;

fn main() -> std::io::Result<()> {
    let rounds: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(8);
    let dev = agx_runtime::Device::open("/dev/dri/renderD128")?;
    let mut vm = agx_runtime::Vm::new(&dev)?;
    let sizes = [4096u64, 1 << 16, 1 << 20, 64 << 20, 512 << 20];
    for round in 0..rounds {
        for &size in &sizes {
            let started = Instant::now();
            let mut bo = vm.alloc_data(size)?;
            let alloc = started.elapsed();
            let data = bo.as_mut_slice();
            if let Some(i) = data.iter().position(|&b| b != 0) {
                eprintln!("round {round}: a new {size}-byte object has byte {i} = {:#x}", data[i]);
                std::process::exit(1);
            }
            data.fill(0xa5);
            if round == 0 {
                println!("{size:>10} bytes: allocated in {:>9.1} us", alloc.as_secs_f64() * 1e6);
            }
        }
    }
    println!("all objects of {rounds} rounds read as zeros");
    // where the time of a large allocation goes: an unbound object (create + mmap, no VM_BIND), then its first
    // touch of every page, against a bound one
    for _ in 0..3 {
        let size = 512u64 << 20;
        let started = Instant::now();
        let mut bo = vm.alloc_unbound(size)?;
        let create = started.elapsed();
        let started = Instant::now();
        for page in bo.as_mut_slice().chunks_mut(16384) {
            page[0] = 1;
        }
        let touch = started.elapsed();
        println!("{size} bytes unbound: created in {:.1} us, every page touched in {:.1} us", create.as_secs_f64() * 1e6, touch.as_secs_f64() * 1e6);
    }
    Ok(())
}
