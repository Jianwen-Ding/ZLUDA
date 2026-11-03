// qrbench M N ITERS: cusolverDnSgeqrf of the replacement cuSOLVER through zluda_agx; own prototypes (values from the
// cuSOLVER documentation). Check: |R(:, j)| = |A(:, j)| for every column (Q orthogonal: column norms of R e_j's
// leading part equal A's), the largest relative difference.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
typedef int CUresult; typedef void *CUcontext; typedef uint64_t CUdeviceptr; typedef void *H;
CUresult cuInit(unsigned); CUresult cuDevicePrimaryCtxRetain(CUcontext *, int); CUresult cuCtxSetCurrent(CUcontext);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t); CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t); CUresult cuCtxSynchronize(void);
int cusolverDnCreate(H *); int cusolverDnSgeqrf_bufferSize(H, int, int, float *, int, int *);
int cusolverDnSgeqrf(H, int, int, float *, int, float *, float *, int, int *);
int cusolverDnSormqr_bufferSize(H, int, int, int, int, int, const float *, int, const float *, const float *, int, int *);
int cusolverDnSormqr(H, int, int, int, int, int, const float *, int, const float *, float *, int, float *, int, int *);
#define CK(x) do { int e = (x); if (e) { fprintf(stderr, "%s: %d\n", #x, e); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    int m = atoi(argv[1]), n = atoi(argv[2]), it = atoi(argv[3]);
    CUcontext c; CK(cuInit(0)); CK(cuDevicePrimaryCtxRetain(&c, 0)); CK(cuCtxSetCurrent(c));
    size_t sz = (size_t)m * n * 4; float *a = malloc(sz), *r = malloc(sz);
    srand(1); for (size_t i = 0; i < (size_t)m * n; i++) a[i] = (float)rand() / RAND_MAX - 0.5f;
    CUdeviceptr da, dw, dt, di; CK(cuMemAlloc_v2(&da, sz)); CK(cuMemAlloc_v2(&dt, n * 4)); CK(cuMemAlloc_v2(&di, 4));
    H h; CK(cusolverDnCreate(&h)); int lw = 0; CK(cusolverDnSgeqrf_bufferSize(h, m, n, (float *)da, m, &lw)); CK(cuMemAlloc_v2(&dw, (size_t)lw * 4 + 4));
    for (int i = 0; i < 2; i++) { CK(cuMemcpyHtoD_v2(da, a, sz)); CK(cusolverDnSgeqrf(h, m, n, (float *)da, m, (float *)dt, (float *)dw, lw, (int *)di)); CK(cuCtxSynchronize()); }
    double tot = 0;
    for (int i = 0; i < it; i++) { CK(cuMemcpyHtoD_v2(da, a, sz)); CK(cuCtxSynchronize()); double t = now(); CK(cusolverDnSgeqrf(h, m, n, (float *)da, m, (float *)dt, (float *)dw, lw, (int *)di)); CK(cuCtxSynchronize()); tot += now() - t; }
    double el = tot / it; CK(cuMemcpyDtoH_v2(r, da, sz)); double worst = 0;
    for (int j = 0; j < n; j++) { double na = 0, nr = 0; for (int i = 0; i < m; i++) na += (double)a[(size_t)j * m + i] * a[(size_t)j * m + i];
        for (int i = 0; i <= j && i < m; i++) nr += (double)r[(size_t)j * m + i] * r[(size_t)j * m + i]; worst = fmax(worst, fabs(sqrt(nr) - sqrt(na)) / sqrt(na)); }
    // Q^T A (ormqr from the reflectors) must be R above the diagonal and 0 below: the largest difference over |A|
    CUdeviceptr dc; CK(cuMemAlloc_v2(&dc, sz)); CK(cuMemcpyHtoD_v2(dc, a, sz)); int lo = 0;
    CK(cusolverDnSormqr_bufferSize(h, 0, 1, m, n, n < m ? n : m, (float *)da, m, (float *)dt, (float *)dc, m, &lo));
    CUdeviceptr dwo; CK(cuMemAlloc_v2(&dwo, (size_t)lo * 4 + 4));
    CK(cusolverDnSormqr(h, 0, 1, m, n, n < m ? n : m, (float *)da, m, (float *)dt, (float *)dc, m, (float *)dwo, lo, (int *)di)); CK(cuCtxSynchronize());
    float *qa = malloc(sz); CK(cuMemcpyDtoH_v2(qa, dc, sz)); double qerr = 0, amax = 0;
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { size_t o = (size_t)j * m + i; amax = fmax(amax, fabs(a[o]));
        double want = i <= j ? r[o] : 0.0; qerr = fmax(qerr, fabs(qa[o] - want)); }
    double fl = 2.0 * n * (double)n * (m - n / 3.0);
    printf("{\"m\":%d,\"n\":%d,\"geqrf_ms\":%.2f,\"GFLOPs\":%.0f,\"norm_err\":%.2e,\"qtA_err\":%.2e}\n", m, n, el * 1e3, fl / el / 1e9, worst, qerr / amax);
    return 0;
}
