Programs for `air_ptx_formalization_check.py`, for a thread that polls for what a thread of its SIMD-group writes.
Every poll is bounded, so none can hang the GPU.

- `a-poll-first`, `b-write-first`: the two sides of one branch, in either order. Starve with
  `ZLUDA_AIR_POLLERS_AS_THEY_COME=1`.
- `i-seq`: the same by hand, as two branches on a volatile, with a writer that takes 1000 steps.
- `k-after`: a write every thread does after the poller's branch. Starves if that code has no copy of its own.
- `i-mutual`: threads that wait for each other. **Expected to fail** on `O[3]` until that is emulated.
