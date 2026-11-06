//! `agx_specialize PTX KERNEL --block x,y,z ARG...`: prints the PTX that launch-time specialization compiles for a
//! launch of KERNEL with these arguments (see `specialize.rs`), to check it against the generic code on captured
//! launches. Each ARG is one parameter's bytes in hex, little-endian (`2a000000` is the u32 42).
//!
//! `agx_specialize --const PTX NAME=FILE...`: prints the PTX that constant-memory specialization compiles when the
//! host-written `.const` variable NAME holds FILE's bytes (`fold_const_loads`).
use std::process::ExitCode;

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.first().map(String::as_str) == Some("--const") && a.len() >= 3 {
        let src = std::fs::read_to_string(&a[1]).expect("PTX");
        let owned: Vec<(String, Vec<u8>)> = a[2..]
            .iter()
            .map(|v| {
                let (n, f) = v.split_once('=').expect("NAME=FILE");
                (n.to_string(), std::fs::read(f).expect("FILE"))
            })
            .collect();
        let values: Vec<(&str, &[u8])> = owned.iter().map(|(n, b)| (n.as_str(), &b[..])).collect();
        eprintln!("host-written constants: {:?}", nvcuda_agx::specialize::host_written_consts(&src));
        let (text, n) = nvcuda_agx::specialize::fold_const_loads(&src, &values, std::env::var("MAX_BYTES").ok().and_then(|v| v.parse().ok()).unwrap_or(8));
        eprintln!("{n} loads folded");
        print!("{text}");
        return ExitCode::SUCCESS;
    }
    if a.len() < 4 || a[2] != "--block" {
        eprintln!("usage: agx_specialize PTX KERNEL --block x,y,z ARG...\n       agx_specialize --const PTX NAME=FILE...");
        return ExitCode::from(2);
    }
    let src = std::fs::read_to_string(&a[0]).expect("PTX");
    let block: Vec<u32> = a[3].split(',').map(|v| v.parse().expect("block")).collect();
    let args: Vec<Vec<u8>> = a[4..]
        .iter()
        .map(|h| (0..h.len()).step_by(2).map(|i| u8::from_str_radix(&h[i..i + 2], 16).expect("hex")).collect())
        .collect();
    let refs: Vec<&[u8]> = args.iter().map(|v| &v[..]).collect();
    match nvcuda_agx::specialize::specialize_ptx(&src, &a[1], &refs, [block[0], block[1], block[2]]) {
        Some((text, n)) => {
            eprintln!("{n} instructions rewritten");
            print!("{text}");
            ExitCode::SUCCESS
        }
        None => {
            eprintln!("nothing to specialize");
            ExitCode::FAILURE
        }
    }
}
