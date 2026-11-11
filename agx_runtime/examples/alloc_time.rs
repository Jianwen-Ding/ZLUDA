//! Times vm.alloc_data + drop of large objects:  cargo run --release -p agx_runtime --example alloc_time [MIB]
use std::time::Instant;
fn main() -> std::io::Result<()> {
    let mib: u64 = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(4096);
    let dev = agx_runtime::Device::open("/dev/dri/renderD128")?;
    let mut vm = agx_runtime::Vm::new(&dev)?;
    for _ in 0..3 {
        let t = Instant::now();
        let bo = vm.alloc_data(mib << 20)?;
        let a = t.elapsed();
        let t = Instant::now();
        drop(bo);
        println!("{mib} MiB: alloc {:.1} ms, free {:.1} ms", a.as_secs_f64() * 1e3, t.elapsed().as_secs_f64() * 1e3);
    }
    Ok(())
}
