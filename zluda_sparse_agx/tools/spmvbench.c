// spmvbench ROWS NNZ_PER_ROW ITERS: cusparseSpMV (CSR, f32, 32-bit indices) of the replacement cuSPARSE
// (libcusparse_agx) through zluda_agx; own prototypes, enum values from the cuSPARSE documentation. Banded columns
// (row i: i - k/2 .. i + k/2, wrapped), checked against the CPU at sampled rows.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
typedef int CUresult; typedef void *CUcontext; typedef uint64_t CUdeviceptr; typedef void *H;
CUresult cuInit(unsigned); CUresult cuDevicePrimaryCtxRetain(CUcontext *, int); CUresult cuCtxSetCurrent(CUcontext);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t); CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t); CUresult cuCtxSynchronize(void);
int cusparseCreate(H *); int cusparseCreateCsr(H *, int64_t, int64_t, int64_t, void *, void *, void *, int, int, int, int);
int cusparseCreateDnVec(H *, int64_t, void *, int);
int cusparseSpMV_bufferSize(H, int, const void *, H, H, const void *, H, int, int, size_t *);
int cusparseSpMV(H, int, const void *, H, H, const void *, H, int, int, void *);
#define CK(x) do { int e = (x); if (e) { fprintf(stderr, "%s: %d\n", #x, e); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    int64_t n = atoll(argv[1]); int k = atoi(argv[2]), it = atoi(argv[3]); int64_t nnz = n * k;
    CUcontext c; CK(cuInit(0)); CK(cuDevicePrimaryCtxRetain(&c, 0)); CK(cuCtxSetCurrent(c));
    int *rp = malloc((n + 1) * 4), *ci = malloc(nnz * 4); float *v = malloc(nnz * 4), *x = malloc(n * 4), *y = malloc(n * 4);
    for (int64_t i = 0; i <= n; i++) rp[i] = (int)(i * k);
    for (int64_t i = 0; i < n; i++) for (int j = 0; j < k; j++) { int64_t col = (i - k / 2 + j + n) % n; ci[i * k + j] = (int)col; v[i * k + j] = (float)((i * 7 + j) % 13) / 16.0f; }
    for (int64_t i = 0; i < n; i++) x[i] = (float)(i % 11) / 8.0f;
    CUdeviceptr drp, dci, dv, dx, dy; CK(cuMemAlloc_v2(&drp, (n + 1) * 4)); CK(cuMemAlloc_v2(&dci, nnz * 4)); CK(cuMemAlloc_v2(&dv, nnz * 4));
    CK(cuMemAlloc_v2(&dx, n * 4)); CK(cuMemAlloc_v2(&dy, n * 4));
    CK(cuMemcpyHtoD_v2(drp, rp, (n + 1) * 4)); CK(cuMemcpyHtoD_v2(dci, ci, nnz * 4)); CK(cuMemcpyHtoD_v2(dv, v, nnz * 4)); CK(cuMemcpyHtoD_v2(dx, x, n * 4));
    H h, A, X, Y; CK(cusparseCreate(&h));
    CK(cusparseCreateCsr(&A, n, n, nnz, (void *)drp, (void *)dci, (void *)dv, 2, 2, 0, 0)); // 32I, 32I, base 0, R_32F
    CK(cusparseCreateDnVec(&X, n, (void *)dx, 0)); CK(cusparseCreateDnVec(&Y, n, (void *)dy, 0));
    float one = 1, zero = 0; size_t ws = 0;
    CK(cusparseSpMV_bufferSize(h, 0, &one, A, X, &zero, Y, 0, 0, &ws)); CUdeviceptr dw = 0; if (ws) CK(cuMemAlloc_v2(&dw, ws));
#define RUN() CK(cusparseSpMV(h, 0, &one, A, X, &zero, Y, 0, 0, (void *)dw))
    for (double w = now(); now() - w < 0.3;) { for (int i = 0; i < 10; i++) RUN(); CK(cuCtxSynchronize()); }
    double t = now(); for (int i = 0; i < it; i++) RUN(); CK(cuCtxSynchronize()); double el = (now() - t) / it;
    CK(cuMemcpyDtoH_v2(y, dy, n * 4)); double err = 0;
    for (int64_t i = 0; i < n; i += n / 997 + 1) { double r = 0; for (int j = 0; j < k; j++) r += (double)v[i * k + j] * x[ci[i * k + j]]; err = fmax(err, fabs(r - y[i])); }
    double bytes = nnz * 8.0 + n * 12.0;
    printf("{\"rows\":%lld,\"nnz_per_row\":%d,\"us\":%.1f,\"GBps\":%.0f,\"maxerr\":%.2g}\n", (long long)n, k, el * 1e6, bytes / el / 1e9, err);
    return 0;
}
