// lubench N ITERS: cusolverDnSgetrf + Sgetrs of the replacement cuSOLVER (libcusolver_agx) through zluda_agx; own
// prototypes (values from the cuSOLVER documentation). Times getrf; checks |A x - b| / |b| for one right-hand side.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
typedef int CUresult; typedef void *CUcontext; typedef uint64_t CUdeviceptr; typedef void *H;
CUresult cuInit(unsigned); CUresult cuDevicePrimaryCtxRetain(CUcontext *, int); CUresult cuCtxSetCurrent(CUcontext);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t); CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t); CUresult cuCtxSynchronize(void);
int cusolverDnCreate(H *); int cusolverDnSgetrf_bufferSize(H, int, int, float *, int, int *);
int cusolverDnSgetrf(H, int, int, float *, int, float *, int *, int *);
int cusolverDnSgetrs(H, int, int, int, const float *, int, const int *, float *, int, int *);
#define CK(x) do { int e = (x); if (e) { fprintf(stderr, "%s: %d\n", #x, e); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    int n = atoi(argv[1]), it = atoi(argv[2]);
    CUcontext c; CK(cuInit(0)); CK(cuDevicePrimaryCtxRetain(&c, 0)); CK(cuCtxSetCurrent(c));
    size_t sz = (size_t)n * n * 4; float *a = malloc(sz), *b = malloc(n * 4), *x = malloc(n * 4);
    srand(1); for (size_t i = 0; i < (size_t)n * n; i++) a[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < n; i++) { a[(size_t)i * n + i] += 0.0f; b[i] = (float)rand() / RAND_MAX; }
    CUdeviceptr da, dw, dp, di, db; CK(cuMemAlloc_v2(&da, sz)); CK(cuMemAlloc_v2(&dp, n * 4)); CK(cuMemAlloc_v2(&di, 4)); CK(cuMemAlloc_v2(&db, n * 4));
    H h; CK(cusolverDnCreate(&h)); int lw = 0; CK(cusolverDnSgetrf_bufferSize(h, n, n, (float *)da, n, &lw)); CK(cuMemAlloc_v2(&dw, (size_t)lw * 4 + 4));
    // warm
    for (int i = 0; i < 2; i++) { CK(cuMemcpyHtoD_v2(da, a, sz)); CK(cusolverDnSgetrf(h, n, n, (float *)da, n, (float *)dw, (int *)dp, (int *)di)); CK(cuCtxSynchronize()); }
    double tot = 0;
    for (int i = 0; i < it; i++) { CK(cuMemcpyHtoD_v2(da, a, sz)); CK(cuCtxSynchronize()); double t = now(); CK(cusolverDnSgetrf(h, n, n, (float *)da, n, (float *)dw, (int *)dp, (int *)di)); CK(cuCtxSynchronize()); tot += now() - t; }
    double el = tot / it; int info; CK(cuMemcpyDtoH_v2(&info, di, 4));
    CK(cuMemcpyHtoD_v2(db, b, n * 4)); CK(cusolverDnSgetrs(h, 0, n, 1, (float *)da, n, (int *)dp, (float *)db, n, (int *)di)); CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH_v2(x, db, n * 4)); double rn = 0, bn = 0;
    for (int i = 0; i < n; i++) { double r = -b[i]; for (int j = 0; j < n; j++) r += (double)a[(size_t)j * n + i] * x[j]; rn += r * r; bn += (double)b[i] * b[i]; }
    printf("{\"n\":%d,\"getrf_ms\":%.2f,\"GFLOPs\":%.0f,\"info\":%d,\"rel_residual\":%.2e}\n", n, el * 1e3, 2.0 / 3 * n * n * (double)n / el / 1e9, info, sqrt(rn / bn));
    return 0;
}
