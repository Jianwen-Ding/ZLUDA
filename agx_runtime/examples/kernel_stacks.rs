//! Lists the kernels of AGX objects (zoc's .o) that use a stack, with their stack and register sizes: a stack costs
//! ~7.5 us of setup per launch (playground novel-results/2026-11-01-stack-launch).
//!
//!     cargo run --release -p agx_runtime --example kernel_stacks OBJECT...
fn main() {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let module = agx_runtime::module::Module::load_object(&bytes).unwrap_or_else(|e| panic!("{path}: {e:?}"));
        let mut stacked: Vec<_> = module.kernels.iter().filter(|(_, k)| k.stack_bytes > 0).collect();
        stacked.sort_by_key(|(n, _)| n.to_string());
        println!("{path}: {} kernels, {} with a stack", module.kernels.len(), stacked.len());
        for (name, k) in stacked {
            println!("  {:5} bytes  {}", k.stack_bytes, name);
        }
    }
}
