// spmmbench ROWS NNZ_PER_ROW NCOLS ORDER ITERS: cusparseSpMM (CSR A, f32, 32-bit indices; dense B and C of NCOLS
// columns, ORDER 1 column-major / 2 row-major) of the replacement cuSPARSE through zluda_agx; own prototypes, enum
// values from the cuSPARSE documentation. Checked against the CPU at sampled entries.
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
int cusparseCreateDnMat(H *, int64_t, int64_t, int64_t, void *, int, int);
int cusparseSpMM_bufferSize(H, int, int, const void *, H, H, const void *, H, int, int, size_t *);
int cusparseSpMM(H, int, int, const void *, H, H, const void *, H, int, int, void *);
#define CK(x) do { int e = (x); if (e) { fprintf(stderr, "%s: %d\n", #x, e); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    int64_t n = atoll(argv[1]); int k = atoi(argv[2]), nc = atoi(argv[3]), order = atoi(argv[4]), it = atoi(argv[5]); int64_t nnz = n * k;
    CUcontext c; CK(cuInit(0)); CK(cuDevicePrimaryCtxRetain(&c, 0)); CK(cuCtxSetCurrent(c));
    int *rp = malloc((n + 1) * 4), *ci = malloc(nnz * 4); float *v = malloc(nnz * 4), *b = malloc(n * nc * 4), *y = malloc(n * nc * 4);
    for (int64_t i = 0; i <= n; i++) rp[i] = (int)(i * k);
    for (int64_t i = 0; i < n; i++) for (int j = 0; j < k; j++) { ci[i * k + j] = (int)((i - k / 2 + j + n) % n); v[i * k + j] = (float)((i * 7 + j) % 13) / 16.0f; }
    for (int64_t i = 0; i < n * nc; i++) b[i] = (float)(i % 11) / 8.0f;
    CUdeviceptr drp, dci, dv, db, dy; CK(cuMemAlloc_v2(&drp, (n + 1) * 4)); CK(cuMemAlloc_v2(&dci, nnz * 4)); CK(cuMemAlloc_v2(&dv, nnz * 4));
    CK(cuMemAlloc_v2(&db, n * nc * 4)); CK(cuMemAlloc_v2(&dy, n * nc * 4));
    CK(cuMemcpyHtoD_v2(drp, rp, (n + 1) * 4)); CK(cuMemcpyHtoD_v2(dci, ci, nnz * 4)); CK(cuMemcpyHtoD_v2(dv, v, nnz * 4)); CK(cuMemcpyHtoD_v2(db, b, n * nc * 4));
    H h, A, B, C; CK(cusparseCreate(&h));
    CK(cusparseCreateCsr(&A, n, n, nnz, (void *)drp, (void *)dci, (void *)dv, 2, 2, 0, 0));
    int64_t ld = order == 1 ? n : nc;
    CK(cusparseCreateDnMat(&B, n, nc, ld, (void *)db, 0, order)); CK(cusparseCreateDnMat(&C, n, nc, ld, (void *)dy, 0, order));
    float one = 1, zero = 0; size_t ws = 0;
    CK(cusparseSpMM_bufferSize(h, 0, 0, &one, A, B, &zero, C, 0, 0, &ws)); CUdeviceptr dw = 0; if (ws) CK(cuMemAlloc_v2(&dw, ws));
#define RUN() CK(cusparseSpMM(h, 0, 0, &one, A, B, &zero, C, 0, 0, (void *)dw))
    for (double w = now(); now() - w < 0.3;) { for (int i = 0; i < 5; i++) RUN(); CK(cuCtxSynchronize()); }
    double t = now(); for (int i = 0; i < it; i++) RUN(); CK(cuCtxSynchronize()); double el = (now() - t) / it;
    CK(cuMemcpyDtoH_v2(y, dy, n * nc * 4)); double err = 0;
#define AT(m, r, col) (order == 1 ? m[(int64_t)(col) * n + (r)] : m[(int64_t)(r) * nc + (col)])
    for (int64_t i = 0; i < n; i += n / 499 + 1) for (int j = 0; j < nc; j += nc / 7 + 1) {
        double r = 0; for (int q = 0; q < k; q++) r += (double)v[i * k + q] * AT(b, ci[i * k + q], j); err = fmax(err, fabs(r - AT(y, i, j))); }
    double bytes = nnz * 8.0 + 2.0 * n * nc * 4;
    printf("{\"rows\":%lld,\"nnz_per_row\":%d,\"ncols\":%d,\"order\":%d,\"us\":%.1f,\"GBps_min_traffic\":%.0f,\"GFLOPs\":%.0f,\"maxerr\":%.2g}\\n",
           (long long)n, k, nc, order, el * 1e6, bytes / el / 1e9, 2.0 * nnz * nc / el / 1e9, err);
    return 0;
}
