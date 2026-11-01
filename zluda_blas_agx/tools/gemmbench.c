// Build: gcc -O2 -o gemmbench gemmbench.c -L$LIB -lcuda -lcublas (LIB: libcuda.so -> libnvcuda_agx.so, libcublas.so ->
// libcublas_agx.so); run with LD_LIBRARY_PATH=$LIB (agx_libs loads libcuda.so.1 by name): gemmbench M N K [ITERS] [f16].
// gemmbench: times cublasGemmEx of the replacement cuBLAS (libcublas_agx) through the driver API of zluda_agx, as an
// application using the CUDA runtime would reach them (the runtime itself is not used: it is not available here).
// Own prototypes; enum values from the cuBLAS / CUDA documentation.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
typedef int CUresult; typedef void *CUcontext; typedef uint64_t CUdeviceptr; typedef void *cublasHandle_t; typedef void *CUevent;
CUresult cuInit(unsigned); CUresult cuDevicePrimaryCtxRetain(CUcontext *, int); CUresult cuCtxSetCurrent(CUcontext);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t); CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuCtxSynchronize(void); CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t); CUresult cuEventCreate(CUevent *, unsigned); CUresult cuEventRecord(CUevent, void *);
CUresult cuEventSynchronize(CUevent); CUresult cuEventElapsedTime(float *, CUevent, CUevent);
int cublasCreate_v2(cublasHandle_t *);
int cublasGemmEx(cublasHandle_t, int, int, int, int, int, const void *, const void *, int, int, const void *, int, int,
                 const void *, void *, int, int, int, int);
enum { OP_N = 0, OP_T = 1, R_16F = 2, R_32F = 0, COMPUTE_32F = 68, COMPUTE_16F = 64, ALGO_DEFAULT = -1 };
#define CK(x) do { int e = (x); if (e) { fprintf(stderr, "%s: %d\n", #x, e); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]), iters = argc > 4 ? atoi(argv[4]) : 50;
    int c16 = argc > 5 && !strcmp(argv[5], "f16");  // f16 C with f16 compute, else f32 C, f32 compute
    CUcontext ctx; CK(cuInit(0)); CK(cuDevicePrimaryCtxRetain(&ctx, 0)); CK(cuCtxSetCurrent(ctx));
    CUdeviceptr a, b, c; CK(cuMemAlloc_v2(&a, (size_t)m * k * 2)); CK(cuMemAlloc_v2(&b, (size_t)k * n * 2)); CK(cuMemAlloc_v2(&c, (size_t)m * n * 4));
    uint16_t *h = malloc((size_t)(m > n ? m : n) * k * 2);
    for (size_t i = 0; i < (size_t)(m > n ? m : n) * k; i++) h[i] = 0x3c00 ^ (uint16_t)((i * 2654435761u) >> 22 & 0x3ff); // ~[1, 2)
    CK(cuMemcpyHtoD_v2(a, h, (size_t)m * k * 2)); CK(cuMemcpyHtoD_v2(b, h, (size_t)k * n * 2));
    cublasHandle_t bl; CK(cublasCreate_v2(&bl));
    float one = 1, zero = 0; uint16_t one16 = 0x3c00, zero16 = 0;
    const void *al = c16 ? (void *)&one16 : (void *)&one, *be = c16 ? (void *)&zero16 : (void *)&zero;
    // ggml: C (m x n) = A^T B, A = weights (k x m), B = activations (k x n)
#define GEMM() CK(cublasGemmEx(bl, OP_T, OP_N, m, n, k, al, (void *)a, R_16F, k, (void *)b, R_16F, k, be, (void *)c, c16 ? R_16F : R_32F, m, c16 ? COMPUTE_16F : COMPUTE_32F, ALGO_DEFAULT))
    for (int i = 0; i < 5; i++) GEMM();
    CK(cuCtxSynchronize());
    CUevent e0, e1; CK(cuEventCreate(&e0, 0)); CK(cuEventCreate(&e1, 0));
    double t0 = now(); CK(cuEventRecord(e0, 0));
    for (int i = 0; i < iters; i++) GEMM();
    CK(cuEventRecord(e1, 0)); double host = now() - t0; CK(cuEventSynchronize(e1));
    float ms; CK(cuEventElapsedTime(&ms, e0, e1));
    double flops = 2.0 * m * n * k;
    // a checksum of C (to compare plans): the sum of the f32 values in double, and of their bits
    size_t cn = (size_t)m * n; float *hc = malloc(cn * 4); double sum = 0; uint64_t bits = 0;
    if (!c16) { CK(cuMemcpyDtoH_v2(hc, c, cn * 4)); for (size_t i = 0; i < cn; i++) { sum += hc[i]; uint32_t u; memcpy(&u, &hc[i], 4); bits = bits * 31 + u; } }
    printf("{\"m\":%d,\"n\":%d,\"k\":%d,\"c\":\"%s\",\"us_per_gemm\":%.1f,\"host_us_per_call\":%.1f,\"tflops\":%.3f,\"sum\":%.6e,\"bits\":\"%016llx\"}\n", m, n, k,
           c16 ? "f16" : "f32", ms * 1e3 / iters, host * 1e6 / iters, flops / (ms * 1e-3 / iters) / 1e12, sum, (unsigned long long)bits);
    return 0;
}
