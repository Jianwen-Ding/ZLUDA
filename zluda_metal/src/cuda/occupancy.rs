//! Occupancy (CUDA Driver API reference, "Occupancy"): from the AGX simulator's dispatch model
//! (`agx_sim::dispatch::occupancy`: resident SIMD groups per core by registers, workgroups by threadgroup memory).
use super::*;

/// Resident blocks of `threads` threads and `dynamic` bytes of dynamic shared memory per multiprocessor
pub(crate) fn blocks_per_sm(f: &Function, threads: u32, dynamic: u32) -> u32 {
    if threads == 0 || threads > 1024 || threads as u64 > f.kernel.max_threads {
        return 0;
    }
    let shared = f.static_shared() + dynamic;
    if shared > device::SHARED_PER_SM {
        return 0;
    }
    match f.kernel.agx() {
        Some(k) => crate::agx::occupancy(k, threads, shared).0,
        None => {
            // the Metal backend: threads and threadgroup memory only
            let by_threads = device::THREADS_PER_SM / threads.div_ceil(32).max(1) / 32;
            let by_shared = if shared > 0 { device::SHARED_PER_SM / shared } else { u32::MAX };
            by_threads.min(by_shared).min(device::BLOCKS_PER_SM)
        }
    }
}

/// A cooperative launch's grid must be resident at once (blocks guaranteed to run concurrently on every core)
pub(crate) unsafe fn check_cooperative(l: &KernelLaunch) -> Result<(), CUresult> {
    let f = &*(l.f as *const Function);
    let threads = l.block.iter().product::<u32>();
    let per_sm = match f.kernel.agx() {
        Some(k) => crate::agx::occupancy(k, threads, f.static_shared() + l.shared).1,
        None => blocks_per_sm(f, threads, l.shared),
    };
    let blocks = l.grid.iter().map(|&g| g as u64).product::<u64>();
    // a grid-wide barrier needs every block resident; the functional model runs the blocks one after another, so
    // only a one-block grid can synchronize (src/device_rt.rs MAX_COOPERATIVE_BLOCKS)
    let limit = (per_sm as u64 * device::multiprocessors() as u64).min(crate::device_rt::MAX_COOPERATIVE_BLOCKS);
    if blocks > limit {
        return Err(CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE);
    }
    Ok(())
}

/// cuOccupancyMaxActiveBlocksPerMultiprocessor(WithFlags): flags CU_OCCUPANCY_DEFAULT (0) or
/// CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE (1); a block size of 0 is CUDA_ERROR_INVALID_VALUE
#[no_mangle]
pub unsafe extern "system" fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    blocks: *mut c_int,
    func: *mut Function,
    block_size: c_int,
    dynamic: usize,
    flags: c_uint,
) -> CUresult {
    tri!(global());
    if blocks.is_null() || block_size <= 0 || flags & !1 != 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let f = tri!(function(func));
    *blocks = if dynamic > f.attrs.lock().unwrap().max_dynamic_shared as usize {
        0
    } else {
        blocks_per_sm(f, block_size as u32, dynamic as u32) as c_int
    };
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
    blocks: *mut c_int,
    func: *mut Function,
    block_size: c_int,
    dynamic: usize,
) -> CUresult {
    cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(blocks, func, block_size, dynamic, 0)
}

/// size_t (*CUoccupancyB2DSize)(int blockSize): dynamic shared memory for a block size
pub type B2DSize = unsafe extern "system" fn(c_int) -> usize;

/// cuOccupancyMaxPotentialBlockSize(WithFlags): the block size (a multiple of the warp size, at most
/// blockSizeLimit if non-zero) with the highest occupancy (the larger one on a tie), and the minimum grid size
/// that reaches it (blocks per multiprocessor x multiprocessors). blockSizeToDynamicSMemSize, if not NULL, gives
/// the dynamic shared memory of each block size instead of dynamicSMemSize.
#[no_mangle]
pub unsafe extern "system" fn cuOccupancyMaxPotentialBlockSizeWithFlags(
    min_grid: *mut c_int,
    block_size: *mut c_int,
    func: *mut Function,
    to_smem: Option<B2DSize>,
    dynamic: usize,
    limit: c_int,
    flags: c_uint,
) -> CUresult {
    tri!(global());
    if min_grid.is_null() || block_size.is_null() || limit < 0 || flags & !1 != 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let f = tri!(function(func));
    let max_dyn = f.attrs.lock().unwrap().max_dynamic_shared as usize;
    let mut top = f.max_threads() as c_int;
    if limit > 0 {
        top = top.min(limit);
    }
    let (mut best, mut best_blocks, mut best_threads) = (0, 0, 0u64);
    let sizes: Vec<c_int> = if top >= 32 { (1..=top / 32).rev().map(|k| k * 32).collect() } else { vec![top] };
    for t in sizes {
        let smem = to_smem.map_or(dynamic, |cb| cb(t));
        let b = if smem > max_dyn { 0 } else { blocks_per_sm(f, t as u32, smem as u32) };
        let active = b as u64 * t as u64;
        if active > best_threads {
            (best, best_blocks, best_threads) = (t, b, active);
        }
    }
    *block_size = best;
    *min_grid = (best_blocks * device::multiprocessors()) as c_int;
    CUDA_SUCCESS
}

#[no_mangle]
pub unsafe extern "system" fn cuOccupancyMaxPotentialBlockSize(
    min_grid: *mut c_int,
    block_size: *mut c_int,
    func: *mut Function,
    to_smem: Option<B2DSize>,
    dynamic: usize,
    limit: c_int,
) -> CUresult {
    cuOccupancyMaxPotentialBlockSizeWithFlags(min_grid, block_size, func, to_smem, dynamic, limit, 0)
}

/// cuOccupancyAvailableDynamicSMemPerBlock: the most dynamic shared memory per block with which numBlocks blocks
/// of blockSize threads still fit on one multiprocessor
#[no_mangle]
pub unsafe extern "system" fn cuOccupancyAvailableDynamicSMemPerBlock(
    dynamic: *mut usize,
    func: *mut Function,
    num_blocks: c_int,
    block_size: c_int,
) -> CUresult {
    tri!(global());
    if dynamic.is_null() || num_blocks <= 0 || block_size <= 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let f = tri!(function(func));
    if blocks_per_sm(f, block_size as u32, 0) < num_blocks as u32 {
        *dynamic = 0;
        return CUDA_SUCCESS;
    }
    let per_block = (device::SHARED_PER_SM / num_blocks as u32).saturating_sub(f.static_shared());
    *dynamic = per_block.min(f.attrs.lock().unwrap().max_dynamic_shared) as usize;
    CUDA_SUCCESS
}
