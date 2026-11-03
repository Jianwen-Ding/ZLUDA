// l2bench M N ITERS: cublasSgemv (N and T) and cublasStrsm (left, lower, unit, N x N against N x 64) of the
// replacement cuBLAS through zluda_agx; own prototypes (values from the cuBLAS documentation). CPU-checked.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
typedef int CUresult; typedef void *CUcontext; typedef uint64_t CUdeviceptr; typedef void *H;
CUresult cuInit(unsigned); CUresult cuDevicePrimaryCtxRetain(CUcontext *, int); CUresult cuCtxSetCurrent(CUcontext);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t); CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t); CUresult cuCtxSynchronize(void);
int cublasCreate_v2(H *); int cublasSgemv_v2(H, int, int, int, const float *, const float *, int, const float *, int, const float *, float *, int);
int cublasStrsm_v2(H, int, int, int, int, int, int, const float *, const float *, int, float *, int);
#define CK(x) do { int e = (x); if (e) { fprintf(stderr, "%s: %d\n", #x, e); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    int m = atoi(argv[1]), n = atoi(argv[2]), it = atoi(argv[3]);
    CUcontext c; CK(cuInit(0)); CK(cuDevicePrimaryCtxRetain(&c, 0)); CK(cuCtxSetCurrent(c));
    size_t sz = (size_t)m * n * 4; float *a = malloc(sz), *x = malloc((m > n ? m : n) * 4), *y = malloc((m > n ? m : n) * 4);
    srand(1); for (size_t i = 0; i < (size_t)m * n; i++) a[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < (m > n ? m : n); i++) x[i] = (float)rand() / RAND_MAX;
    CUdeviceptr da, dx, dy; CK(cuMemAlloc_v2(&da, sz)); CK(cuMemAlloc_v2(&dx, (m > n ? m : n) * 4)); CK(cuMemAlloc_v2(&dy, (m > n ? m : n) * 4));
    CK(cuMemcpyHtoD_v2(da, a, sz)); CK(cuMemcpyHtoD_v2(dx, x, (m > n ? m : n) * 4));
    H h; CK(cublasCreate_v2(&h)); float one = 1, zero = 0;
    for (int tr = 0; tr < 2; tr++) {
        int ylen = tr ? n : m;
        for (double w = now(); now() - w < 0.3;) { for (int i = 0; i < 10; i++) CK(cublasSgemv_v2(h, tr, m, n, &one, (float *)da, m, (float *)dx, 1, &zero, (float *)dy, 1)); CK(cuCtxSynchronize()); }
        double t = now(); for (int i = 0; i < it; i++) CK(cublasSgemv_v2(h, tr, m, n, &one, (float *)da, m, (float *)dx, 1, &zero, (float *)dy, 1)); CK(cuCtxSynchronize());
        double el = (now() - t) / it; CK(cuMemcpyDtoH_v2(y, dy, ylen * 4)); double err = 0, mx = 0;
        for (int i = 0; i < ylen; i += ylen / 97 + 1) { double r = 0; for (int j = 0; j < (tr ? m : n); j++) r += (double)(tr ? a[(size_t)i * m + j] : a[(size_t)j * m + i]) * x[j]; err = fmax(err, fabs(r - y[i])); mx = fmax(mx, fabs(r)); }
        printf("{\"op\":\"sgemv_%c\",\"m\":%d,\"n\":%d,\"us\":%.1f,\"GBps\":%.0f,\"relerr\":%.1e}\n", tr ? 'T' : 'N', m, n, el * 1e6, sz / el / 1e9, err / mx);
    }
    // trsm: L (n x n unit lower from a's leading part, m >= n) X = B, B n x 64
    int nr = 64; CUdeviceptr db; CK(cuMemAlloc_v2(&db, (size_t)n * nr * 4)); float *b = malloc((size_t)n * nr * 4), *xb = malloc((size_t)n * nr * 4);
    for (size_t i = 0; i < (size_t)n * nr; i++) b[i] = (float)rand() / RAND_MAX;
    // a small enough lower part for a well-conditioned unit triangle
    float *l = malloc((size_t)n * n * 4); for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) l[(size_t)j * n + i] = i > j ? a[(size_t)j * m + i] * 0.5f / sqrtf(n) : (i == j);
    CUdeviceptr dl; CK(cuMemAlloc_v2(&dl, (size_t)n * n * 4)); CK(cuMemcpyHtoD_v2(dl, l, (size_t)n * n * 4));
    double tt = 0;
    for (int i = 0; i < 3; i++) { CK(cuMemcpyHtoD_v2(db, b, (size_t)n * nr * 4)); CK(cuCtxSynchronize()); double t = now();
        CK(cublasStrsm_v2(h, 0, 0, 0, 1, n, nr, &one, (float *)dl, n, (float *)db, n)); CK(cuCtxSynchronize()); if (i) tt += now() - t; }
    CK(cuMemcpyDtoH_v2(xb, db, (size_t)n * nr * 4)); double err = 0;
    for (int col = 0; col < nr; col += 7) for (int i = 0; i < n; i += n / 61 + 1) { double r = 0; for (int j = 0; j <= i; j++) r += (double)l[(size_t)j * n + i] * xb[(size_t)col * n + j]; err = fmax(err, fabs(r - b[(size_t)col * n + i])); }
    printf("{\"op\":\"strsm_LLNU\",\"n\":%d,\"nrhs\":%d,\"ms\":%.2f,\"maxerr\":%.1e}\n", n, nr, tt / 2 * 1e3, err);
    return 0;
}
