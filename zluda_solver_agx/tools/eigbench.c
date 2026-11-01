// eigbench N ITERS: cusolverDnSgesvd (values only) and cusolverDnSsyevd (values only, lower) of the replacement
// cuSOLVER through zluda_agx; own prototypes (values from the cuSOLVER documentation). Checks: sum of sigma^2 =
// |A|_F^2; for the symmetric matrix sum of eigenvalues = trace and sum of their squares = |A|_F^2.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
typedef int CUresult; typedef void *CUcontext; typedef uint64_t CUdeviceptr; typedef void *H;
CUresult cuInit(unsigned); CUresult cuDevicePrimaryCtxRetain(CUcontext *, int); CUresult cuCtxSetCurrent(CUcontext);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t); CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t); CUresult cuCtxSynchronize(void);
int cusolverDnCreate(H *);
int cusolverDnSgesvd_bufferSize(H, int, int, int *);
int cusolverDnSgesvd(H, signed char, signed char, int, int, float *, int, float *, float *, int, float *, int, float *, int, float *, int *);
int cusolverDnSsyevd_bufferSize(H, int, int, int, const float *, int, const float *, int *);
int cusolverDnSsyevd(H, int, int, int, float *, int, float *, float *, int, int *);
#define CK(x) do { int e = (x); if (e) { fprintf(stderr, "%s: %d\n", #x, e); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    int n = atoi(argv[1]), it = atoi(argv[2]);
    CUcontext c; CK(cuInit(0)); CK(cuDevicePrimaryCtxRetain(&c, 0)); CK(cuCtxSetCurrent(c));
    size_t sz = (size_t)n * n * 4; float *a = malloc(sz), *s = malloc(sz), *v = malloc(n * 4);
    srand(1); for (size_t i = 0; i < (size_t)n * n; i++) a[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) s[(size_t)j * n + i] = 0.5f * (a[(size_t)j * n + i] + a[(size_t)i * n + j]);
    double fa = 0, fs = 0, tr = 0; for (size_t i = 0; i < (size_t)n * n; i++) { fa += (double)a[i] * a[i]; fs += (double)s[i] * s[i]; }
    for (int i = 0; i < n; i++) tr += s[(size_t)i * n + i];
    CUdeviceptr da, dv, dw, di; CK(cuMemAlloc_v2(&da, sz)); CK(cuMemAlloc_v2(&dv, n * 4)); CK(cuMemAlloc_v2(&di, 4));
    H h; CK(cusolverDnCreate(&h));
    int lw = 0; CK(cusolverDnSgesvd_bufferSize(h, n, n, &lw)); CK(cuMemAlloc_v2(&dw, (size_t)lw * 4 + 4));
    double tot = 0;
    for (int i = 0; i < it; i++) { CK(cuMemcpyHtoD_v2(da, a, sz)); CK(cuCtxSynchronize()); double t = now();
        CK(cusolverDnSgesvd(h, 'N', 'N', n, n, (float *)da, n, (float *)dv, 0, 1, 0, 1, (float *)dw, lw, 0, (int *)di)); CK(cuCtxSynchronize()); tot += now() - t; }
    CK(cuMemcpyDtoH_v2(v, dv, n * 4)); double ss = 0; for (int i = 0; i < n; i++) ss += (double)v[i] * v[i];
    printf("{\"n\":%d,\"gesvd_ms\":%.1f,\"sumsq_err\":%.2e}\n", n, tot / it * 1e3, fabs(ss - fa) / fa);
    int lw2 = 0; CK(cusolverDnSsyevd_bufferSize(h, 0, 0, n, (float *)da, n, (float *)dv, &lw2)); CUdeviceptr dw2; CK(cuMemAlloc_v2(&dw2, (size_t)lw2 * 4 + 4));
    tot = 0;
    for (int i = 0; i < it; i++) { CK(cuMemcpyHtoD_v2(da, s, sz)); CK(cuCtxSynchronize()); double t = now();
        CK(cusolverDnSsyevd(h, 0, 0, n, (float *)da, n, (float *)dv, (float *)dw2, lw2, (int *)di)); CK(cuCtxSynchronize()); tot += now() - t; }
    CK(cuMemcpyDtoH_v2(v, dv, n * 4)); double sw = 0, sq = 0; for (int i = 0; i < n; i++) { sw += v[i]; sq += (double)v[i] * v[i]; }
    printf("{\"n\":%d,\"syevd_ms\":%.1f,\"trace_err\":%.2e,\"sumsq_err\":%.2e}\n", n, tot / it * 1e3, fabs(sw - tr) / sqrt(fs), fabs(sq - fs) / fs);
    return 0;
}
