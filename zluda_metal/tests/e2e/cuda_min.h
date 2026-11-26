/* The CUDA driver API subset used by the tests, declared from the public
 * documentation (no NVIDIA headers needed). */
#pragma once
#include <stddef.h>
#include <stdint.h>
typedef int CUresult;
typedef int CUdevice;
typedef uint64_t CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
CUresult cuInit(unsigned flags);
CUresult cuDriverGetVersion(int *v);
CUresult cuDeviceGetCount(int *count);
CUresult cuDeviceGet(CUdevice *dev, int ordinal);
CUresult cuDeviceGetName(char *name, int len, CUdevice dev);
CUresult cuDeviceGetAttribute(int *value, int attrib, CUdevice dev);
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev);
CUresult cuCtxCreate_v2(CUcontext *ctx, unsigned flags, CUdevice dev);
CUresult cuCtxDestroy_v2(CUcontext ctx);
CUresult cuCtxSynchronize(void);
CUresult cuModuleLoadData(CUmodule *mod, const void *image);
CUresult cuModuleUnload(CUmodule mod);
CUresult cuModuleGetFunction(CUfunction *f, CUmodule mod, const char *name);
CUresult cuModuleGetGlobal_v2(CUdeviceptr *p, size_t *bytes, CUmodule mod, const char *name);
CUresult cuFuncGetAttribute(int *v, int attrib, CUfunction f);
CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t size);
CUresult cuMemFree_v2(CUdeviceptr p);
CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t n);
CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t n);
CUresult cuMemcpyDtoD_v2(CUdeviceptr dst, CUdeviceptr src, size_t n);
CUresult cuMemsetD32_v2(CUdeviceptr dst, unsigned v, size_t n);
CUresult cuMemGetAddressRange_v2(CUdeviceptr *base, size_t *size, CUdeviceptr p);
CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                        unsigned shared, CUstream s, void **params, void **extra);
CUresult cuStreamCreate(CUstream *s, unsigned flags);
CUresult cuStreamSynchronize(CUstream s);
CUresult cuStreamDestroy_v2(CUstream s);
CUresult cuEventCreate(CUevent *e, unsigned flags);
CUresult cuEventRecord(CUevent e, CUstream s);
CUresult cuEventElapsedTime(float *ms, CUevent a, CUevent b);
CUresult cuEventDestroy_v2(CUevent e);
CUresult cuGetErrorName(CUresult e, const char **s);
#define cuMemAlloc cuMemAlloc_v2
#define cuMemFree cuMemFree_v2
#define cuMemcpyHtoD cuMemcpyHtoD_v2
#define cuMemcpyDtoH cuMemcpyDtoH_v2
#define cuMemcpyDtoD cuMemcpyDtoD_v2
#define cuMemsetD32 cuMemsetD32_v2
#define cuCtxCreate cuCtxCreate_v2
#define cuCtxDestroy cuCtxDestroy_v2
