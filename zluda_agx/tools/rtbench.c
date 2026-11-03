// rtbench: host cost of the driver API calls behind libcudart's common runtime calls, on zluda_agx (own prototypes)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef int CUresult; typedef void *CUcontext; typedef uint64_t CUdeviceptr; typedef void *CUstream; typedef void *CUevent;
typedef void *CUmodule; typedef void *CUfunction;
CUresult cuInit(unsigned); CUresult cuDevicePrimaryCtxRetain(CUcontext *, int); CUresult cuCtxSetCurrent(CUcontext);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t); CUresult cuMemFree_v2(CUdeviceptr);
CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t); CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t);
CUresult cuMemcpyDtoD_v2(CUdeviceptr, CUdeviceptr, size_t);
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr, const void *, size_t, CUstream); CUresult cuMemcpyDtoHAsync_v2(void *, CUdeviceptr, size_t, CUstream);
CUresult cuMemsetD32_v2(CUdeviceptr, unsigned, size_t); CUresult cuMemsetD8Async(CUdeviceptr, unsigned char, size_t, CUstream);
CUresult cuMemAllocHost_v2(void **, size_t); CUresult cuMemFreeHost(void *);
CUresult cuStreamCreate(CUstream *, unsigned); CUresult cuStreamSynchronize(CUstream); CUresult cuStreamQuery(CUstream);
CUresult cuEventCreate(CUevent *, unsigned); CUresult cuEventRecord(CUevent, CUstream); CUresult cuEventSynchronize(CUevent);
CUresult cuEventElapsedTime(float *, CUevent, CUevent); CUresult cuEventQuery(CUevent);
CUresult cuCtxSynchronize(void); CUresult cuMemGetInfo_v2(size_t *, size_t *);
CUresult cuModuleLoadData(CUmodule *, const void *); CUresult cuModuleGetFunction(CUfunction *, CUmodule, const char *);
CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void **, void **);
#define CK(x) do { int e = (x); if (e) { fprintf(stderr, "%s: %d\n", #x, e); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#define T(name, iters, body) do { for (int i_ = 0; i_ < 5; i_++) { body; } double t_ = now(); for (int i_ = 0; i_ < (iters); i_++) { body; } \
    printf("%-34s %9.2f us\n", name, (now() - t_) / (iters) * 1e6); } while (0)
static const char *ptx =
    ".version 7.0\n.target sm_75\n.address_size 64\n.visible .entry k(.param .u64 p)\n{\n.reg .u64 %rd<2>;\nld.param.u64 %rd1, [p];\nst.global.u32 [%rd1], 1;\nret;\n}\n";
int main(void) {
    CUcontext c; CK(cuInit(0)); CK(cuDevicePrimaryCtxRetain(&c, 0)); CK(cuCtxSetCurrent(c));
    CUstream s; CK(cuStreamCreate(&s, 1));
    CUdeviceptr d, d2; CK(cuMemAlloc_v2(&d, 64 << 20)); CK(cuMemAlloc_v2(&d2, 64 << 20));
    char *h = malloc(64 << 20); memset(h, 1, 64 << 20); void *ph; CK(cuMemAllocHost_v2(&ph, 64 << 20)); memset(ph, 1, 64 << 20);
    CUmodule m; CK(cuModuleLoadData(&m, ptx)); CUfunction f; CK(cuModuleGetFunction(&f, m, "k"));
    void *args[] = {&d};
    CUevent e0, e1; CK(cuEventCreate(&e0, 0)); CK(cuEventCreate(&e1, 0));
    size_t fr, tot;
    T("cuMemAlloc+Free 4 KiB", 200, { CUdeviceptr x; CK(cuMemAlloc_v2(&x, 4096)); CK(cuMemFree_v2(x)); });
    T("cuMemAlloc+Free 1 MiB", 200, { CUdeviceptr x; CK(cuMemAlloc_v2(&x, 1 << 20)); CK(cuMemFree_v2(x)); });
    T("cuMemAlloc+Free 64 MiB", 50, { CUdeviceptr x; CK(cuMemAlloc_v2(&x, 64 << 20)); CK(cuMemFree_v2(x)); });
    T("cuMemAllocHost+FreeHost 1 MiB", 50, { void *x; CK(cuMemAllocHost_v2(&x, 1 << 20)); CK(cuMemFreeHost(x)); });
    T("cuMemGetInfo", 1000, CK(cuMemGetInfo_v2(&fr, &tot)));
    T("HtoD 4 B (sync)", 1000, CK(cuMemcpyHtoD_v2(d, h, 4)));
    T("HtoD 4 KiB (sync)", 1000, CK(cuMemcpyHtoD_v2(d, h, 4096)));
    T("DtoH 4 B (sync)", 1000, CK(cuMemcpyDtoH_v2(h, d, 4)));
    T("DtoH 4 KiB (sync)", 1000, CK(cuMemcpyDtoH_v2(h, d, 4096)));
    T("HtoD 16 MiB pageable", 20, CK(cuMemcpyHtoD_v2(d, h, 16 << 20)));
    T("HtoD 16 MiB pinned", 20, CK(cuMemcpyHtoD_v2(d, ph, 16 << 20)));
    T("DtoH 16 MiB pageable", 20, CK(cuMemcpyDtoH_v2(h, d, 16 << 20)));
    T("DtoH 16 MiB pinned", 20, CK(cuMemcpyDtoH_v2(ph, d, 16 << 20)));
    T("DtoD 16 MiB (+ctx sync)", 20, { CK(cuMemcpyDtoD_v2(d2, d, 16 << 20)); CK(cuCtxSynchronize()); });
    T("HtoDAsync 4 KiB + stream sync", 500, { CK(cuMemcpyHtoDAsync_v2(d, h, 4096, s)); CK(cuStreamSynchronize(s)); });
    T("DtoHAsync 4 B + stream sync", 500, { CK(cuMemcpyDtoHAsync_v2(h, d, 4, s)); CK(cuStreamSynchronize(s)); });
    T("memset D32 1 MiB (sync)", 200, CK(cuMemsetD32_v2(d, 7, 1 << 18)));
    T("memsetAsync 1 MiB + sync", 200, { CK(cuMemsetD8Async(d, 7, 1 << 20, s)); CK(cuStreamSynchronize(s)); });
    T("launch (async, no sync)", 2000, CK(cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, s, args, NULL)));
    CK(cuStreamSynchronize(s));
    T("launch + stream sync", 500, { CK(cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, s, args, NULL)); CK(cuStreamSynchronize(s)); });
    T("stream sync (idle)", 1000, CK(cuStreamSynchronize(s)));
    T("stream query (idle)", 1000, cuStreamQuery(s));
    T("ctx sync (idle)", 1000, CK(cuCtxSynchronize()));
    T("event record (idle stream)", 1000, CK(cuEventRecord(e0, s)));
    T("event record x2 + sync + elapsed", 500, { float ms; CK(cuEventRecord(e0, s)); CK(cuEventRecord(e1, s)); CK(cuEventSynchronize(e1)); CK(cuEventElapsedTime(&ms, e0, e1)); });
    T("event query", 1000, cuEventQuery(e1));
    return 0;
}
