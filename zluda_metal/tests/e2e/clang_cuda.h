// Minimal device-side prelude for compiling CUDA kernels to PTX with clang
// alone (-nocudainc -nocudalib): no NVIDIA headers or libraries involved.
#pragma once
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ __attribute__((always_inline)) inline
#include <__clang_cuda_builtin_vars.h>

static __device__ inline int atomicAdd(int *p, int v) { return __nvvm_atom_add_gen_i(p, v); }
static __device__ inline unsigned atomicAdd(unsigned *p, unsigned v) { return (unsigned)__nvvm_atom_add_gen_i((int *)p, (int)v); }
static __device__ inline float atomicAdd(float *p, float v) { return __nvvm_atom_add_gen_f(p, v); }
static __device__ inline int atomicMax(int *p, int v) { return __nvvm_atom_max_gen_i(p, v); }
static __device__ inline int atomicMin(int *p, int v) { return __nvvm_atom_min_gen_i(p, v); }
static __device__ inline int atomicExch(int *p, int v) { return __nvvm_atom_xchg_gen_i(p, v); }
static __device__ inline int atomicCAS(int *p, int cmp, int v) { return __nvvm_atom_cas_gen_i(p, cmp, v); }
static __device__ inline int atomicAnd(int *p, int v) { return __nvvm_atom_and_gen_i(p, v); }
static __device__ inline int atomicOr(int *p, int v) { return __nvvm_atom_or_gen_i(p, v); }
static __device__ inline int __shfl_down_sync(unsigned mask, int v, unsigned d) { return __nvvm_shfl_sync_down_i32(mask, v, d, 0x1f); }
static __device__ inline int __shfl_xor_sync(unsigned mask, int v, int m) { return __nvvm_shfl_sync_bfly_i32(mask, v, m, 0x1f); }
static __device__ inline int __shfl_sync(unsigned mask, int v, int lane) { return __nvvm_shfl_sync_idx_i32(mask, v, lane, 0x1f); }
static __device__ inline int __shfl_up_sync(unsigned mask, int v, unsigned d) { return __nvvm_shfl_sync_up_i32(mask, v, d, 0); }
static __device__ inline unsigned __ballot_sync(unsigned mask, int p) { return __nvvm_vote_ballot_sync(mask, p); }
static __device__ inline int __popc(unsigned x) { return __builtin_popcount(x); }

static __device__ inline float sqrtf_(float x) { return __builtin_sqrtf(x); }
static __device__ inline float fmaf_(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
