// cuda_bf16.h of zluda_rtc_agx: the bfloat16 type and functions NVRTC programs include, on clang's __bf16. Written for
// this project from the CUDA math API's documented interface: __nv_bfloat16 / nv_bfloat16, __nv_bfloat162, the
// conversions and the h-prefixed functions. Device code (NVRTC compiles device code only).
#pragma once
#ifndef __CUDA_BF16_H__
#define __CUDA_BF16_H__

typedef __bf16 __nv_bfloat16;
typedef __nv_bfloat16 nv_bfloat16;
typedef __bf16 __nv_bfloat162 __attribute__((ext_vector_type(2)));
typedef __nv_bfloat162 nv_bfloat162;

#define AGX_B static __device__ __forceinline__
#define AGX_BB static __host__ __device__ __forceinline__

AGX_BB __nv_bfloat16 __float2bfloat16(float f) { return (__bf16)f; }
AGX_BB __nv_bfloat16 __float2bfloat16_rn(float f) { return (__bf16)f; }
AGX_BB float __bfloat162float(__nv_bfloat16 h) { return (float)h; }
AGX_BB __nv_bfloat16 __double2bfloat16(double d) { return (__bf16)(float)d; }
AGX_BB __nv_bfloat16 __int2bfloat16_rn(int i) { return (__bf16)(float)i; }
AGX_BB int __bfloat162int_rz(__nv_bfloat16 h) { return (int)(float)h; }
AGX_BB unsigned short __bfloat16_as_ushort(__nv_bfloat16 h) { return __builtin_bit_cast(unsigned short, h); }
AGX_BB __nv_bfloat16 __ushort_as_bfloat16(unsigned short u) { return __builtin_bit_cast(__bf16, u); }
AGX_BB __nv_bfloat162 __floats2bfloat162_rn(float a, float b) { return (__nv_bfloat162){(__bf16)a, (__bf16)b}; }
AGX_BB __nv_bfloat162 make_bfloat162(__nv_bfloat16 a, __nv_bfloat16 b) { return (__nv_bfloat162){a, b}; }
AGX_BB float __low2float(__nv_bfloat162 h) { return (float)h.x; }
AGX_BB float __high2float(__nv_bfloat162 h) { return (float)h.y; }

AGX_B __nv_bfloat16 __hadd(__nv_bfloat16 a, __nv_bfloat16 b) { return (__bf16)((float)a + (float)b); }
AGX_B __nv_bfloat16 __hsub(__nv_bfloat16 a, __nv_bfloat16 b) { return (__bf16)((float)a - (float)b); }
AGX_B __nv_bfloat16 __hmul(__nv_bfloat16 a, __nv_bfloat16 b) { return (__bf16)((float)a * (float)b); }
AGX_B __nv_bfloat16 __hdiv(__nv_bfloat16 a, __nv_bfloat16 b) { return (__bf16)((float)a / (float)b); }
AGX_B __nv_bfloat16 __hfma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
  return (__bf16)__builtin_fmaf((float)a, (float)b, (float)c);
}
AGX_B __nv_bfloat16 __hmax(__nv_bfloat16 a, __nv_bfloat16 b) { return (__bf16)__builtin_fmaxf((float)a, (float)b); }
AGX_B __nv_bfloat16 __hmin(__nv_bfloat16 a, __nv_bfloat16 b) { return (__bf16)__builtin_fminf((float)a, (float)b); }

#define AGX_BMATH(name, f) AGX_B __nv_bfloat16 name(__nv_bfloat16 x) { return (__bf16)f((float)x); }
AGX_BMATH(hsqrt, sqrtf)
AGX_BMATH(hrsqrt, rsqrtf)
AGX_BMATH(hexp, expf)
AGX_BMATH(hexp2, exp2f)
AGX_BMATH(hlog, logf)
AGX_BMATH(hlog2, log2f)
AGX_BMATH(hsin, sinf)
AGX_BMATH(hcos, cosf)
AGX_BMATH(htrunc, truncf)
AGX_BMATH(hceil, ceilf)
AGX_BMATH(hfloor, floorf)
AGX_BMATH(hrint, rintf)
AGX_B __nv_bfloat16 hrcp(__nv_bfloat16 x) { return (__bf16)(1.0f / (float)x); }
#undef AGX_BMATH

#undef AGX_B
#undef AGX_BB
#endif
