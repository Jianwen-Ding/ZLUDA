//! `agx_texture_check OUT [options]`: samples a random texture at 2^20 coordinates, some far outside the texture,
//! through the CUDA driver API of zluda_agx, and writes the results to OUT (and the coordinates and texels to
//! OUT.coords / OUT.texels). Run it once with ZLUDA_AGX_HARDWARE_TEXTURES=1 (the texture unit, for the forms that get
//! hardware descriptors) and once without (the software sampler) and compare the files (agx_texture_compare.py).
//!
//! Options (the default: a 1024 x 512 RGBA8 array, normalized coordinates, wrap, linear filtering asked for, read as
//! integers):
//! - `--float`: read as floats (`tex.*.v4.f32.f32`; normalized for 8- and 16-bit integer formats), else as integers
//!   (`CU_TRSF_READ_AS_INTEGER`, `tex.*.v4.u32.f32` / `.s32` for signed formats)
//! - `--format u8|u16|s8|s16|f16|f32`, `--channels 1|2|4`
//! - `--address wrap|clamp|mirror|border`, `--point` (point filtering), `--unnormalized`
//! - `--1d` (a 1D array, `tex.1d`), `--pitch` (2D pitch-linear memory instead of an array)
//! - `--size WxH`

use cuda_types::cuda::*;
use nvcuda_agx::*;
use std::ffi::c_void;
use std::ptr::null_mut;

fn check(what: &str, r: CUresult) {
    if let Err(e) = r {
        eprintln!("{what} failed: {e:?}");
        std::process::exit(1);
    }
}

fn ptx(ty: &str, rank: u32) -> String {
    let (dims, coords, load) = if rank == 1 {
        ("1d", "{%x}", "ld.global.f32 %x, [%a];")
    } else {
        ("2d", "{%x, %y}", "ld.global.v2.f32 {%x, %y}, [%a];")
    };
    format!(
        ".version 7.0
.target sm_75
.address_size 64
.visible .entry s(.param .u64 tex, .param .u64 pin, .param .u64 pout, .param .u32 n)
{{
    .reg .u32 %i, %n;
    .reg .u64 %t, %a, %b, %o;
    .reg .f32 %x, %y;
    .reg .{ty} %r<4>;
    .reg .pred %p;
    mov.u32 %i, %ctaid.x;
    mad.lo.u32 %i, %i, %ntid.x, %tid.x;
    ld.param.u32 %n, [n];
    setp.ge.u32 %p, %i, %n;
    @%p bra DONE;
    ld.param.u64 %t, [tex];
    ld.param.u64 %a, [pin];
    ld.param.u64 %b, [pout];
    cvta.to.global.u64 %a, %a;
    cvta.to.global.u64 %b, %b;
    mul.wide.u32 %o, %i, 8;
    add.u64 %a, %a, %o;
    {load}
    tex.{dims}.v4.{ty}.f32 {{%r0, %r1, %r2, %r3}}, [%t, {coords}];
    mul.wide.u32 %o, %i, 16;
    add.u64 %b, %b, %o;
    st.global.v4.{ty} [%b], {{%r0, %r1, %r2, %r3}};
DONE:
    ret;
}}
\0"
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: agx_texture_check OUT [--float] [--format F] [--channels N] [--address A] [--point] \
                 [--unnormalized] [--1d] [--pitch] [--size WxH]";
    let out = args.get(1).expect(usage).clone();
    let has = |o: &str| args.iter().any(|a| a == o);
    let value = |o: &str| args.iter().position(|a| a == o).and_then(|i| args.get(i + 1)).cloned();
    let float = has("--float");
    let format = value("--format").unwrap_or_else(|| "u8".into());
    let channels: usize = value("--channels").map_or(4, |c| c.parse().expect(usage));
    let rank = if has("--1d") { 1 } else { 2 };
    let pitch_linear = has("--pitch");
    let (w, h) = value("--size")
        .map(|s| {
            let (a, b) = s.split_once('x').expect(usage);
            (a.parse::<usize>().expect(usage), b.parse::<usize>().expect(usage))
        })
        .unwrap_or((1024, 512));
    let h = if rank == 1 { 1 } else { h };
    let normalized = !has("--unnormalized");
    let address = match value("--address").as_deref().unwrap_or("wrap") {
        "wrap" => CUaddress_mode::CU_TR_ADDRESS_MODE_WRAP,
        "clamp" => CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP,
        "mirror" => CUaddress_mode::CU_TR_ADDRESS_MODE_MIRROR,
        "border" => CUaddress_mode::CU_TR_ADDRESS_MODE_BORDER,
        _ => panic!("{usage}"),
    };
    use CUarray_format_enum as F;
    let (cu_format, bytes, signed, half) = match format.as_str() {
        "u8" => (F::CU_AD_FORMAT_UNSIGNED_INT8, 1, false, false),
        "u16" => (F::CU_AD_FORMAT_UNSIGNED_INT16, 2, false, false),
        "s8" => (F::CU_AD_FORMAT_SIGNED_INT8, 1, true, false),
        "s16" => (F::CU_AD_FORMAT_SIGNED_INT16, 2, true, false),
        "f16" => (F::CU_AD_FORMAT_HALF, 2, false, true),
        "f32" => (F::CU_AD_FORMAT_FLOAT, 4, false, false),
        _ => panic!("{usage}"),
    };
    let float_texels = half || format == "f32";
    let ty = if float || float_texels { "f32" } else if signed { "s32" } else { "u32" };
    let n = 1usize << 20;
    let mut x = 0x2545f4914f6cdd1du64;
    let mut rnd = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    unsafe {
        check("cuInit", cuInit(0));
        let mut dev = 0;
        check("cuDeviceGet", cuDeviceGet(&mut dev, 0));
        let mut ctx = CUcontext(null_mut());
        check("cuCtxCreate", cuCtxCreate_v2(&mut ctx, 0, dev));
        let src = ptx(ty, rank);
        let mut module = CUmodule(null_mut());
        check("cuModuleLoadData", cuModuleLoadData(&mut module, src.as_ptr() as *const c_void));
        let mut f = CUfunction(null_mut());
        check("cuModuleGetFunction", cuModuleGetFunction(&mut f, module, c"s".as_ptr()));

        // the texels: random bits for integers; for floats, finite values of mixed magnitudes
        let row = w * channels * bytes;
        let mut texels: Vec<u8> = Vec::with_capacity(row * h);
        for _ in 0..w * h * channels {
            let r = rnd();
            if half {
                // finite halves: exponent below 31
                let v = (r as u16 & 0x83ff) | (((r >> 16) % 31) as u16) << 10;
                texels.extend_from_slice(&v.to_le_bytes());
            } else if format == "f32" {
                let v = ((r >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * (2f32).powi((r % 13) as i32 - 6);
                texels.extend_from_slice(&v.to_le_bytes());
            } else {
                texels.extend_from_slice(&r.to_le_bytes()[..bytes]);
            }
        }
        let mut rd: CUDA_RESOURCE_DESC = std::mem::zeroed();
        let mut device_texels = CUdeviceptr_v2(null_mut());
        if pitch_linear {
            // rows 256-byte aligned
            let pitch = row.div_ceil(256) * 256;
            check("cuMemAlloc", cuMemAlloc_v2(&mut device_texels, pitch * h));
            let mut cp: CUDA_MEMCPY2D = std::mem::zeroed();
            cp.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST;
            cp.srcHost = texels.as_ptr() as *const c_void;
            cp.srcPitch = row;
            cp.dstMemoryType = CUmemorytype::CU_MEMORYTYPE_DEVICE;
            cp.dstDevice = device_texels;
            cp.dstPitch = pitch;
            cp.WidthInBytes = row;
            cp.Height = h;
            check("cuMemcpy2D", cuMemcpy2D_v2(&cp));
            rd.resType = CUresourcetype::CU_RESOURCE_TYPE_PITCH2D;
            rd.res.pitch2D.devPtr = device_texels;
            rd.res.pitch2D.format = cu_format;
            rd.res.pitch2D.numChannels = channels as u32;
            rd.res.pitch2D.width = w;
            rd.res.pitch2D.height = h;
            rd.res.pitch2D.pitchInBytes = pitch;
        } else {
            let mut ad: CUDA_ARRAY_DESCRIPTOR = std::mem::zeroed();
            ad.Width = w;
            ad.Height = if rank == 1 { 0 } else { h };
            ad.Format = cu_format;
            ad.NumChannels = channels as u32;
            let mut array: CUarray = null_mut();
            check("cuArrayCreate", cuArrayCreate_v2(&mut array, &ad));
            let mut cp: CUDA_MEMCPY2D = std::mem::zeroed();
            cp.srcMemoryType = CUmemorytype::CU_MEMORYTYPE_HOST;
            cp.srcHost = texels.as_ptr() as *const c_void;
            cp.srcPitch = row;
            cp.dstMemoryType = CUmemorytype::CU_MEMORYTYPE_ARRAY;
            cp.dstArray = array;
            cp.WidthInBytes = row;
            cp.Height = h;
            check("cuMemcpy2D", cuMemcpy2D_v2(&cp));
            rd.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
            rd.res.array.hArray = array;
        }
        let mut td: CUDA_TEXTURE_DESC = std::mem::zeroed();
        td.addressMode = [address; 3];
        td.filterMode = if has("--point") { CUfilter_mode::CU_TR_FILTER_MODE_POINT } else { CUfilter_mode::CU_TR_FILTER_MODE_LINEAR };
        td.flags = if normalized { CU_TRSF_NORMALIZED_COORDINATES } else { 0 }
            | if float || float_texels { 0 } else { CU_TRSF_READ_AS_INTEGER };
        let mut tex = 0u64;
        check("cuTexObjectCreate", cuTexObjectCreate(&mut tex as *mut u64 as *mut CUtexObject, &rd, &td, null_mut()));

        // coordinates: texel centres, edges, and random values in [-3, 3) of the extent
        let coords: Vec<f32> = (0..2 * n)
            .map(|i| {
                let extent = if i % 2 == 0 { w } else { h } as f32;
                let v = match i % 8 {
                    0 => ((rnd() % 4096) as f32 + 0.5) / extent,
                    1 => (rnd() % 4097) as f32 / 1024.0,
                    _ => (rnd() >> 40) as f32 / (1u64 << 24) as f32 * 6.0 - 3.0,
                };
                if normalized { v } else { v * extent }
            })
            .collect();
        let (mut din, mut dout) = (CUdeviceptr_v2(null_mut()), CUdeviceptr_v2(null_mut()));
        check("cuMemAlloc", cuMemAlloc_v2(&mut din, n * 8));
        check("cuMemAlloc", cuMemAlloc_v2(&mut dout, n * 16));
        check("cuMemcpyHtoD", cuMemcpyHtoD_v2(din, coords.as_ptr() as *const c_void, n * 8));
        let nn = n as u32;
        let mut params: [*mut c_void; 4] = [
            &mut tex as *mut _ as *mut c_void,
            &mut din as *mut _ as *mut c_void,
            &mut dout as *mut _ as *mut c_void,
            &nn as *const u32 as *mut c_void,
        ];
        check("cuLaunchKernel", cuLaunchKernel(f, (n / 256) as u32, 1, 1, 256, 1, 1, 0, CUstream(null_mut()), params.as_mut_ptr(), null_mut()));
        let mut result = vec![0u8; n * 16];
        check("cuMemcpyDtoH", cuMemcpyDtoH_v2(result.as_mut_ptr() as *mut c_void, dout, n * 16));
        std::fs::write(&out, &result).expect("write");
        let mut c = Vec::with_capacity(n * 8);
        for v in &coords {
            c.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(format!("{out}.coords"), &c).expect("write");
        std::fs::write(format!("{out}.texels"), &texels).expect("write");
        println!("{{\"samples\":{n},\"type\":\"{ty}\"}}");
    }
}
