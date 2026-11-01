// fftbench N BATCH ITERS [d]: cufftExecC2C (d: cufftExecZ2Z) of the replacement cuFFT (libcufft_agx) through zluda_agx; own prototypes
// (values from the cuFFT documentation). Checks the forward transform of a known signal against a CPU DFT (one batch).
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
typedef int CUresult; typedef void *CUcontext; typedef uint64_t CUdeviceptr; typedef int cufftHandle;
CUresult cuInit(unsigned); CUresult cuDevicePrimaryCtxRetain(CUcontext *, int); CUresult cuCtxSetCurrent(CUcontext);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t); CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t); CUresult cuCtxSynchronize(void);
int cufftPlan1d(cufftHandle *, int, int, int); int cufftExecC2C(cufftHandle, void *, void *, int);
int cufftExecZ2Z(cufftHandle, void *, void *, int);
#define CK(x) do { int e = (x); if (e) { fprintf(stderr, "%s: %d\n", #x, e); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    int n = atoi(argv[1]), batch = atoi(argv[2]), it = atoi(argv[3]), dbl = argc > 4 && argv[4][0] == 'd';
    CUcontext c; CK(cuInit(0)); CK(cuDevicePrimaryCtxRetain(&c, 0)); CK(cuCtxSetCurrent(c));
    size_t bytes = (size_t)n * batch * (dbl ? 16 : 8); CUdeviceptr x, y; CK(cuMemAlloc_v2(&x, bytes)); CK(cuMemAlloc_v2(&y, bytes));
    double *h = malloc((size_t)n * batch * 16);
    for (size_t i = 0; i < (size_t)n * batch; i++) { h[2 * i] = sinf(0.37f * i) + 0.5f * cosf(1.3f * (i % n)); h[2 * i + 1] = 0.25f * sinf(0.11f * i); }
    float *hf = malloc(bytes);
    for (size_t i = 0; i < 2 * (size_t)n * batch; i++) hf[i] = (float)h[i];
    CK(cuMemcpyHtoD_v2(x, dbl ? (void *)h : (void *)hf, bytes));
    int (*exec)(cufftHandle, void *, void *, int) = dbl ? cufftExecZ2Z : cufftExecC2C;
    cufftHandle p; CK(cufftPlan1d(&p, n, dbl ? 0x69 /* CUFFT_Z2Z */ : 0x29 /* CUFFT_C2C */, batch));
    CK(exec(p, (void *)x, (void *)y, -1)); CK(cuCtxSynchronize());
    // warm the GPU clock first: ~0.3 s of the same work queued in bursts (fewer iterations, or a sync after each,
    // measure a partly idle clock)
    for (double w = now(); now() - w < 0.3;) { for (int i = 0; i < 50; i++) CK(exec(p, (void *)x, (void *)y, -1)); CK(cuCtxSynchronize()); }
    double t = now();
    for (int i = 0; i < it; i++) CK(exec(p, (void *)x, (void *)y, -1));
    CK(cuCtxSynchronize()); double el = (now() - t) / it;
    void *ob = malloc(bytes); CK(cuMemcpyDtoH_v2(ob, y, bytes));
    double *o = malloc((size_t)n * batch * 16);
    for (size_t i = 0; i < 2 * (size_t)n * batch; i++) o[i] = dbl ? ((double *)ob)[i] : ((float *)ob)[i];
    double maxrel = 0, norm = 0;
    for (int k = 0; k < n; k++) { double re = 0, im = 0;
        for (int j = 0; j < n; j++) { double a = -2 * M_PI * (double)j * k / n; re += h[2*j] * cos(a) - h[2*j+1] * sin(a); im += h[2*j] * sin(a) + h[2*j+1] * cos(a); }
        double e = hypot(o[2*k] - re, o[2*k+1] - im); norm = fmax(norm, hypot(re, im)); maxrel = fmax(maxrel, e); }
    printf("{\"type\":\"%s\",\"n\":%d,\"batch\":%d,\"us\":%.1f,\"maxerr_rel\":%.2e}\n", dbl ? "Z2Z" : "C2C", n, batch, el * 1e6, maxrel / norm);
    return 0;
}
