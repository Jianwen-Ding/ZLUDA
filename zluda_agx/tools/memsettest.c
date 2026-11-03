// Build: gcc -O2 -o memsettest memsettest.c -L$LIB -lcuda -lcublas (LIB: libcuda.so and libcublas.so of zluda_agx and
// zluda_blas_agx); run with LD_LIBRARY_PATH=$LIB ZLUDA_AGX_ZOC=zoc; exit status 1 on a wrong byte.
// memsettest: async memsets (8/16/32-bit patterns, odd offsets and sizes) issued while the stream is busy with a GEMM,
// checked byte by byte (and the bytes around them untouched), and timed against ZLUDA_AGX_GPU_COPIES=0.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef int CUresult; typedef void *CUcontext; typedef uint64_t CUdeviceptr; typedef void *H;
CUresult cuInit(unsigned); CUresult cuDevicePrimaryCtxRetain(CUcontext *, int); CUresult cuCtxSetCurrent(CUcontext);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t); CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t); CUresult cuCtxSynchronize(void);
CUresult cuMemsetD8Async(CUdeviceptr, unsigned char, size_t, void *); CUresult cuMemsetD16Async(CUdeviceptr, unsigned short, size_t, void *);
CUresult cuMemsetD32Async(CUdeviceptr, unsigned, size_t, void *);
int cublasCreate_v2(H *); int cublasSgemm_v2(H, int, int, int, int, int, const float *, const float *, int, const float *, int, const float *, float *, int);
#define CK(x) do { int e = (x); if (e) { fprintf(stderr, "%s: %d\n", #x, e); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(void) {
    CUcontext c; CK(cuInit(0)); CK(cuDevicePrimaryCtxRetain(&c, 0)); CK(cuCtxSetCurrent(c));
    int n = 2048; CUdeviceptr a, b, cc; CK(cuMemAlloc_v2(&a, (size_t)n * n * 4)); CK(cuMemAlloc_v2(&b, (size_t)n * n * 4)); CK(cuMemAlloc_v2(&cc, (size_t)n * n * 4));
    H h; CK(cublasCreate_v2(&h)); float one = 1, zero = 0;
    size_t size = 1 << 22; CUdeviceptr m; CK(cuMemAlloc_v2(&m, size));
    uint8_t *ref = malloc(size), *got = malloc(size); memset(ref, 0x5a, size); CK(cuMemcpyHtoD_v2(m, ref, size));
    struct { int width; size_t off, count; uint32_t v; } cases[] = {
        {1, 0, 1 << 20, 0x11}, {1, 3, 1001, 0x22}, {1, 17, 3, 0x33}, {2, 2, 777, 0xbeef}, {2, 4096, 1 << 18, 0x1234},
        {4, 4, 999, 0xdeadbeef}, {4, 1 << 21, 1 << 18, 0x01020304}, {4, 8, 1, 0xcafef00d}, {1, 1 << 21 | 5, 16, 0x44}};
    double t = now();
    for (unsigned i = 0; i < sizeof cases / sizeof *cases; i++) {
        CK(cublasSgemm_v2(h, 0, 0, n, n, n, &one, (float *)a, n, (float *)b, n, &zero, (float *)cc, n));  // the stream busy
        size_t o = cases[i].off; CUdeviceptr p = m + o;
        if (cases[i].width == 1) { CK(cuMemsetD8Async(p, cases[i].v, cases[i].count, 0)); memset(ref + o, cases[i].v, cases[i].count); }
        if (cases[i].width == 2) { CK(cuMemsetD16Async(p, cases[i].v, cases[i].count, 0)); for (size_t k = 0; k < cases[i].count; k++) memcpy(ref + o + 2 * k, &cases[i].v, 2); }
        if (cases[i].width == 4) { CK(cuMemsetD32Async(p, cases[i].v, cases[i].count, 0)); for (size_t k = 0; k < cases[i].count; k++) memcpy(ref + o + 4 * k, &cases[i].v, 4); }
    }
    double host = now() - t;
    CK(cuCtxSynchronize()); CK(cuMemcpyDtoH_v2(got, m, size));
    size_t bad = 0; for (size_t i = 0; i < size; i++) bad += got[i] != ref[i];
    printf("{\"cases\":%zu,\"wrong_bytes\":%zu,\"host_ms\":%.2f}\n", sizeof cases / sizeof *cases, bad, host * 1e3);
    return bad != 0;
}
