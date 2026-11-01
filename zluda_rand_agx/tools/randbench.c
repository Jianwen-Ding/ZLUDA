// randbench N ITERS: curandGenerateUniform / Normal of the replacement cuRAND (libcurand_agx) through zluda_agx; own
// prototypes (values from the cuRAND documentation). Prints us per call, GB/s and the sample mean.
// M2 Max, 64 Mi floats, warm (2026-11-01): uniform 1056 us (254 GB/s, ~70% of the GPU's ~365 GB/s writes), normal
// 1790 us (150 GB/s, Box-Muller with both outputs used: compute-bound), the same for XORWOW / MRG32k3a / Philox.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
typedef int CUresult; typedef void *CUcontext; typedef uint64_t CUdeviceptr; typedef void *G;
CUresult cuInit(unsigned); CUresult cuDevicePrimaryCtxRetain(CUcontext *, int); CUresult cuCtxSetCurrent(CUcontext);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t); CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t); CUresult cuCtxSynchronize(void);
int curandCreateGenerator(G *, int); int curandSetPseudoRandomGeneratorSeed(G, unsigned long long);
int curandGenerateUniform(G, float *, size_t); int curandGenerateNormal(G, float *, size_t, float, float);
#define CK(x) do { int e = (x); if (e) { fprintf(stderr, "%s: %d\n", #x, e); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    size_t n = strtoull(argv[1], 0, 0); int it = atoi(argv[2]); int rng = argc > 3 ? atoi(argv[3]) : 100;
    CUcontext c; CK(cuInit(0)); CK(cuDevicePrimaryCtxRetain(&c, 0)); CK(cuCtxSetCurrent(c));
    CUdeviceptr d; CK(cuMemAlloc_v2(&d, n * 4));
    G g; CK(curandCreateGenerator(&g, rng)); CK(curandSetPseudoRandomGeneratorSeed(g, 1234));
    for (int k = 0; k < 2; k++) {
        for (double w = now(); now() - w < 0.3;) { for (int i = 0; i < 5; i++) CK(k ? curandGenerateNormal(g, (float *)d, n, 0, 1) : curandGenerateUniform(g, (float *)d, n)); CK(cuCtxSynchronize()); }
        double t = now();
        for (int i = 0; i < it; i++) CK(k ? curandGenerateNormal(g, (float *)d, n, 0, 1) : curandGenerateUniform(g, (float *)d, n));
        CK(cuCtxSynchronize()); double el = (now() - t) / it;
        float *h = malloc(n * 4); CK(cuMemcpyDtoH_v2(h, d, n * 4)); double m = 0; for (size_t i = 0; i < n; i++) m += h[i]; free(h);
        printf("{\"rng\":%d,\"%s\":%zu,\"us\":%.1f,\"GBps\":%.0f,\"mean\":%.4f}\n", rng, k ? "normal" : "uniform", n, el * 1e6, n * 4 / el / 1e9, m / n);
    }
    return 0;
}
