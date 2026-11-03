// Device prelude for building Blender Cycles' CUDA kernel (Apache-2.0 source) with upstream clang's NVPTX target and
// no CUDA toolkit: no nvcc, no libdevice, no NVIDIA header. It gives the kernel what its compat.h expects of the CUDA
// language: keywords, vector types, warp intrinsics, the texture read and the math library. Everything is written
// here against the public PTX ISA; the math functions are this project's own approximations.
#pragma once

#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __noinline__ __attribute__((noinline))
#define __forceinline__ __attribute__((always_inline))
#define __align__(n) __attribute__((aligned(n)))
#define __restrict__ __restrict
#define __ZINL __device__ inline __attribute__((always_inline))

extern "C" __device__ int printf(const char *, ...);
extern "C" __device__ void *memcpy(void *, const void *, __SIZE_TYPE__);
extern "C" __device__ void *memset(void *, int, __SIZE_TYPE__);

#include <__clang_cuda_builtin_vars.h>

typedef __SIZE_TYPE__ size_t;

// The vector types of the CUDA language: 2 and 4 component types naturally aligned, 3 component ones packed.
#define __ZVEC(T, N)                 \
  struct __attribute__((aligned(sizeof(T) * 2))) N##2 { T x, y; };                \
  struct N##3 { T x, y, z; };                                                     \
  struct __attribute__((aligned(sizeof(T) * 4))) N##4 { T x, y, z, w; };          \
  __ZINL N##2 make_##N##2(T x, T y) { N##2 r; r.x = x; r.y = y; return r; }       \
  __ZINL N##3 make_##N##3(T x, T y, T z) { N##3 r; r.x = x; r.y = y; r.z = z; return r; } \
  __ZINL N##4 make_##N##4(T x, T y, T z, T w) { N##4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }
__ZVEC(float, float)
__ZVEC(int, int)
__ZVEC(unsigned int, uint)
__ZVEC(short, short)
__ZVEC(unsigned short, ushort)
__ZVEC(char, char)
__ZVEC(unsigned char, uchar)
#undef __ZVEC

// Warp, bit and clamp intrinsics
__ZINL unsigned __ballot_sync(unsigned mask, int pred) { return __nvvm_vote_ballot_sync(mask, pred); }
__ZINL unsigned __popc(unsigned x) { return __builtin_popcount(x); }
__ZINL int __clz(int x) { return x ? __builtin_clz(x) : 32; }
__ZINL int __ffs(int x) { return __builtin_ffs(x); }
__ZINL unsigned __brev(unsigned x) { return __builtin_bitreverse32(x); }
__ZINL float __saturatef(float x) { return __builtin_fminf(__builtin_fmaxf(x, 0.0f), 1.0f); }

// Texture reads: PTX tex through a 64-bit texture object
template<typename T> __ZINL T tex2D(unsigned long long t, float x, float y);
template<> __ZINL float4 tex2D<float4>(unsigned long long t, float x, float y) {
  float4 r;
  __asm__ volatile("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
                   : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(t), "f"(x), "f"(y));
  return r;
}
template<> __ZINL float tex2D<float>(unsigned long long t, float x, float y) { return tex2D<float4>(t, x, y).x; }

// Math. These are what --use_fast_math style kernels ask for: the hardware approximations where PTX has them.
__ZINL float __fabsf_(float x) { return __builtin_fabsf(x); }
__ZINL float fabsf(float x) { return __builtin_fabsf(x); }
__ZINL float fabs(float x) { return __builtin_fabsf(x); }
__ZINL float sqrtf(float x) { return __builtin_sqrtf(x); }
__ZINL float sqrt(float x) { return __builtin_sqrtf(x); }
__ZINL float floorf(float x) { return __builtin_floorf(x); }
__ZINL float ceilf(float x) { return __builtin_ceilf(x); }
__ZINL float truncf(float x) { return __builtin_truncf(x); }
__ZINL float rintf(float x) { return __builtin_rintf(x); }
__ZINL float roundf(float x) { float t = __builtin_truncf(x); return __builtin_fabsf(x - t) >= 0.5f ? t + __builtin_copysignf(1.0f, x) : t; }
__ZINL float fminf(float a, float b) { return __builtin_fminf(a, b); }
__ZINL float fmaxf(float a, float b) { return __builtin_fmaxf(a, b); }
__ZINL float copysignf(float a, float b) { return __builtin_copysignf(a, b); }
__ZINL float fmaf(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
__ZINL int min(int a, int b) { return a < b ? a : b; }
__ZINL int max(int a, int b) { return a > b ? a : b; }
__ZINL unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
__ZINL unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
__ZINL long long min(long long a, long long b) { return a < b ? a : b; }
__ZINL long long max(long long a, long long b) { return a > b ? a : b; }
__ZINL unsigned long long min(unsigned long long a, unsigned long long b) { return a < b ? a : b; }
__ZINL unsigned long long max(unsigned long long a, unsigned long long b) { return a > b ? a : b; }
__ZINL float min(float a, float b) { return __builtin_fminf(a, b); }
__ZINL float max(float a, float b) { return __builtin_fmaxf(a, b); }
__ZINL double min(double a, double b) { return a < b ? a : b; }
__ZINL double max(double a, double b) { return a > b ? a : b; }
__ZINL int abs(int a) { return a < 0 ? -a : a; }
__ZINL float __zsin(float x) { float r; __asm__("sin.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__ZINL float __zcos(float x) { float r; __asm__("cos.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__ZINL float __zlg2(float x) { float r; __asm__("lg2.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__ZINL float __zex2(float x) { float r; __asm__("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
// sin.approx is accurate on any argument PTX allows it, which is what the nvcc fast forms rely on too
__ZINL float __sinf(float x) { return __zsin(x); }
__ZINL float __cosf(float x) { return __zcos(x); }
__ZINL float __tanf(float x) { return __zsin(x) / __zcos(x); }
__ZINL float __logf(float x) { return __zlg2(x) * 0.693147180559945f; }
__ZINL float __expf(float x) { return __zex2(x * 1.44269504088896f); }
__ZINL float __powf(float x, float y) { return __zex2(y * __zlg2(x)); }
__ZINL float log2f(float x) { return __zlg2(x); }
__ZINL float log10f(float x) { return __zlg2(x) * 0.301029995663981f; }
__ZINL float exp2f(float x) { return __zex2(x); }
__ZINL float fmodf(float a, float b) {
  // exact while a / b fits in a float's mantissa, the range shading uses it in
  float q = __builtin_truncf(a / b);
  float r = __builtin_fmaf(-q, b, a);
  return (__builtin_fabsf(r) >= __builtin_fabsf(b)) ? 0.0f : __builtin_copysignf(__builtin_fabsf(r), a);
}
// atan on [0, 1]: minimax polynomial in x^2, error about 1e-5 rad before the range reduction
__ZINL float __zatan01(float x) {
  float s = x * x;
  float p = -0.0117212f;
  p = p * s + 0.05265332f;
  p = p * s - 0.11643287f;
  p = p * s + 0.19354346f;
  p = p * s - 0.33262347f;
  p = p * s + 0.99997726f;
  return p * x;
}
__ZINL float atanf(float x) {
  float a = __builtin_fabsf(x);
  float r = a > 1.0f ? 1.57079632679f - __zatan01(1.0f / a) : __zatan01(a);
  return __builtin_copysignf(r, x);
}
__ZINL float atan2f(float y, float x) {
  float ax = __builtin_fabsf(x), ay = __builtin_fabsf(y);
  float mx = __builtin_fmaxf(ax, ay), mn = __builtin_fminf(ax, ay);
  float r = mx == 0.0f ? 0.0f : __zatan01(mn / mx);
  if (ay > ax) r = 1.57079632679f - r;
  if (x < 0.0f) r = 3.14159265359f - r;
  return __builtin_copysignf(r, y);
}
__ZINL float asinf(float x) {
  float a = __builtin_fabsf(x);
  float r = a > 0.5f ? 1.57079632679f - 2.0f * atanf(__builtin_sqrtf((1.0f - a) * 0.5f) / __builtin_sqrtf(__builtin_fmaxf(1.0f - (1.0f - a) * 0.5f, 0.0f)))
                     : atanf(a / __builtin_sqrtf(1.0f - a * a));
  return __builtin_copysignf(r, x);
}
__ZINL float acosf(float x) { return 1.57079632679f - asinf(x); }
__ZINL float sinhf(float x) { float e = __expf(x); return 0.5f * (e - 1.0f / e); }
__ZINL float coshf(float x) { float e = __expf(x); return 0.5f * (e + 1.0f / e); }
__ZINL float tanhf(float x) { float a = __builtin_fabsf(x); if (a > 9.0f) return __builtin_copysignf(1.0f, x); float e = __expf(2.0f * x); return (e - 1.0f) / (e + 1.0f); }
__ZINL int isfinite(float x) { return __builtin_isfinite(x); }
__ZINL int isnan(float x) { return __builtin_isnan(x); }
__ZINL int isinf(float x) { return __builtin_isinf(x); }
__ZINL int signbit(float x) { return __builtin_signbit(x); }
__ZINL float __float_max() { return 3.4028234663852886e+38f; }
// Stirling's series with the shift to x >= 7, enough for the Beta function the kernel builds from it
__ZINL float lgammaf(float x) {
  float s = 0.0f;
  while (x < 7.0f) { s -= __logf(x); x += 1.0f; }
  float i = 1.0f / x, i2 = i * i;
  return s + (x - 0.5f) * __logf(x) - x + 0.918938533204673f + i * (0.0833333333f - i2 * (0.00277777778f - i2 * 0.000793650794f));
}

__ZINL float ldexpf(float x, int e) { return __builtin_ldexpf(x, e); }
__ZINL float frexpf(float x, int *e) { return __builtin_frexpf(x, e); }

// Atomics of global memory
__ZINL int atomicAdd(int *p, int v) { return __nvvm_atom_add_gen_i(p, v); }
__ZINL unsigned atomicAdd(unsigned *p, unsigned v) { return (unsigned)__nvvm_atom_add_gen_i((int *)p, (int)v); }
__ZINL float atomicAdd(float *p, float v) { return __nvvm_atom_add_gen_f(p, v); }
__ZINL int atomicSub(int *p, int v) { return __nvvm_atom_sub_gen_i(p, v); }
__ZINL unsigned atomicSub(unsigned *p, unsigned v) { return (unsigned)__nvvm_atom_sub_gen_i((int *)p, (int)v); }
__ZINL int atomicMin(int *p, int v) { return __nvvm_atom_min_gen_i(p, v); }
__ZINL int atomicMax(int *p, int v) { return __nvvm_atom_max_gen_i(p, v); }
__ZINL unsigned atomicMin(unsigned *p, unsigned v) { return __nvvm_atom_min_gen_ui(p, v); }
__ZINL unsigned atomicMax(unsigned *p, unsigned v) { return __nvvm_atom_max_gen_ui(p, v); }
__ZINL int atomicAnd(int *p, int v) { return __nvvm_atom_and_gen_i(p, v); }
__ZINL int atomicOr(int *p, int v) { return __nvvm_atom_or_gen_i(p, v); }
__ZINL int atomicXor(int *p, int v) { return __nvvm_atom_xor_gen_i(p, v); }
__ZINL int atomicExch(int *p, int v) { return __nvvm_atom_xchg_gen_i(p, v); }
__ZINL int atomicCAS(int *p, int c, int v) { return __nvvm_atom_cas_gen_i(p, c, v); }
__ZINL unsigned atomicCAS(unsigned *p, unsigned c, unsigned v) { return (unsigned)__nvvm_atom_cas_gen_i((int *)p, (int)c, (int)v); }

#define FLT_MIN 1.175494350822287507969e-38f
#define FLT_MAX 340282346638528859811704183484516925440.0f
#define FLT_EPSILON 1.192092896e-07F
#define M_PI 3.14159265358979323846
#define NULL 0
#define INT_MAX 2147483647
#define INT_MIN (-INT_MAX - 1)
#define UINT_MAX 0xffffffffu
// Limits the tuning of nvcc's scheduler: nothing here to tell (zoc picks registers and threads for AGX itself)
#define __launch_bounds__(...)
namespace std {
template<class A, class B> struct is_same { static constexpr bool value = false; };
template<class A> struct is_same<A, A> { static constexpr bool value = true; };
template<class A, class B> inline constexpr bool is_same_v = is_same<A, B>::value;
template<bool C, class A = void> struct enable_if {};
template<class A> struct enable_if<true, A> { using type = A; };
template<bool C, class A = void> using enable_if_t = typename enable_if<C, A>::type;
}  // namespace std
