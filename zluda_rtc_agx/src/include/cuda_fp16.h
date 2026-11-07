// cuda_fp16.h of zluda_rtc_agx: the half precision type and functions NVRTC programs include, on clang's _Float16
// (PTX .f16 arithmetic). Written for this project from the CUDA math API's documented interface: __half / half,
// __half2 / half2, the conversions and the h-prefixed functions. Device code (NVRTC compiles device code only).
#pragma once
#ifndef __CUDA_FP16_H__
#define __CUDA_FP16_H__

typedef _Float16 __half;
typedef __half half;
typedef _Float16 __half2 __attribute__((ext_vector_type(2)));
typedef __half2 half2;
typedef struct __align__(2) { unsigned short x; } __half_raw;
typedef struct __align__(4) { unsigned short x, y; } __half2_raw;

#define AGX_H static __device__ __forceinline__
#define AGX_HH static __host__ __device__ __forceinline__

// conversions
AGX_HH __half __float2half(float f) { return (__half)f; }
AGX_HH __half __float2half_rn(float f) { return (__half)f; }
AGX_HH float __half2float(__half h) { return (float)h; }
AGX_HH __half __double2half(double d) { return (__half)d; }
AGX_HH __half __int2half_rn(int i) { return (__half)i; }
AGX_HH __half __uint2half_rn(unsigned i) { return (__half)i; }
AGX_HH int __half2int_rz(__half h) { return (int)h; }
AGX_HH unsigned __half2uint_rz(__half h) { return (unsigned)h; }
AGX_HH __half __short2half_rn(short i) { return (__half)i; }
AGX_HH short __half2short_rz(__half h) { return (short)h; }
AGX_HH unsigned short __half_as_ushort(__half h) { return __builtin_bit_cast(unsigned short, h); }
AGX_HH short __half_as_short(__half h) { return __builtin_bit_cast(short, h); }
AGX_HH __half __ushort_as_half(unsigned short u) { return __builtin_bit_cast(__half, u); }
AGX_HH __half __short_as_half(short u) { return __builtin_bit_cast(__half, u); }
AGX_HH __half2 __float2half2_rn(float f) { return (__half2){(__half)f, (__half)f}; }
AGX_HH __half2 __floats2half2_rn(float a, float b) { return (__half2){(__half)a, (__half)b}; }
AGX_HH __half2 __halves2half2(__half a, __half b) { return (__half2){a, b}; }
AGX_HH __half2 __half2half2(__half a) { return (__half2){a, a}; }
AGX_HH __half2 make_half2(__half a, __half b) { return (__half2){a, b}; }
AGX_HH float __low2float(__half2 h) { return (float)h.x; }
AGX_HH float __high2float(__half2 h) { return (float)h.y; }
AGX_HH __half __low2half(__half2 h) { return h.x; }
AGX_HH __half __high2half(__half2 h) { return h.y; }
AGX_HH float2 __half22float2(__half2 h) { return make_float2((float)h.x, (float)h.y); }
AGX_HH __half2 __float22half2_rn(float2 f) { return (__half2){(__half)f.x, (__half)f.y}; }

// arithmetic and comparisons (operators are the type's own)
AGX_H __half __hadd(__half a, __half b) { return a + b; }
AGX_H __half __hsub(__half a, __half b) { return a - b; }
AGX_H __half __hmul(__half a, __half b) { return a * b; }
AGX_H __half __hdiv(__half a, __half b) { return a / b; }
AGX_H __half __hneg(__half a) { return -a; }
AGX_H __half __habs(__half a) { return a < (__half)0 ? -a : a; }
AGX_H __half __hfma(__half a, __half b, __half c) { return __builtin_fmaf16(a, b, c); }
AGX_H __half __hmax(__half a, __half b) { return __builtin_fmaxf16(a, b); }
AGX_H __half __hmin(__half a, __half b) { return __builtin_fminf16(a, b); }
AGX_H bool __heq(__half a, __half b) { return a == b; }
AGX_H bool __hne(__half a, __half b) { return a != b; }
AGX_H bool __hlt(__half a, __half b) { return a < b; }
AGX_H bool __hle(__half a, __half b) { return a <= b; }
AGX_H bool __hgt(__half a, __half b) { return a > b; }
AGX_H bool __hge(__half a, __half b) { return a >= b; }
AGX_H bool __hisnan(__half a) { return a != a; }
AGX_H int __hisinf(__half a) { return a == (__half)__builtin_inff() ? 1 : a == (__half)-__builtin_inff() ? -1 : 0; }
AGX_H __half2 __hadd2(__half2 a, __half2 b) { return a + b; }
AGX_H __half2 __hsub2(__half2 a, __half2 b) { return a - b; }
AGX_H __half2 __hmul2(__half2 a, __half2 b) { return a * b; }
AGX_H __half2 __h2div(__half2 a, __half2 b) { return a / b; }
AGX_H __half2 __hfma2(__half2 a, __half2 b, __half2 c) { return a * b + c; }
AGX_H __half2 __hneg2(__half2 a) { return -a; }

// math, computed in f32 and rounded once (as accurate as CUDA's, which documents these against the f32 functions)
#define AGX_HMATH(name, f) AGX_H __half name(__half x) { return (__half)f((float)x); }
AGX_HMATH(hsqrt, sqrtf)
AGX_HMATH(hrsqrt, rsqrtf)
AGX_HMATH(hexp, expf)
AGX_HMATH(hexp2, exp2f)
AGX_HMATH(hexp10, exp10f)
AGX_HMATH(hlog, logf)
AGX_HMATH(hlog2, log2f)
AGX_HMATH(hlog10, log10f)
AGX_HMATH(hsin, sinf)
AGX_HMATH(hcos, cosf)
AGX_HMATH(htrunc, truncf)
AGX_HMATH(hceil, ceilf)
AGX_HMATH(hfloor, floorf)
AGX_HMATH(hrint, rintf)
AGX_H __half hrcp(__half x) { return (__half)(1.0f / (float)x); }
#undef AGX_HMATH

// warp shuffles of halves (32-bit lanes)
AGX_H __half __shfl_sync(unsigned mask, __half v, int src, int width = 32) {
  return __ushort_as_half((unsigned short)__shfl_sync(mask, (int)__half_as_ushort(v), src, width));
}
AGX_H __half __shfl_xor_sync(unsigned mask, __half v, int lane, int width = 32) {
  return __ushort_as_half((unsigned short)__shfl_xor_sync(mask, (int)__half_as_ushort(v), lane, width));
}
AGX_H __half __shfl_down_sync(unsigned mask, __half v, unsigned d, int width = 32) {
  return __ushort_as_half((unsigned short)__shfl_down_sync(mask, (int)__half_as_ushort(v), d, width));
}
AGX_H __half __shfl_up_sync(unsigned mask, __half v, unsigned d, int width = 32) {
  return __ushort_as_half((unsigned short)__shfl_up_sync(mask, (int)__half_as_ushort(v), d, width));
}

#undef AGX_H
#undef AGX_HH
#endif
