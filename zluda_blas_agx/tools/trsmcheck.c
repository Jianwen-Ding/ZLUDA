// trsmcheck: cublasStrsm for all side / uplo / trans / diag at a few sizes against a CPU check (op(A) X = alpha B
// or X op(A) = alpha B); own prototypes (values from the cuBLAS documentation).
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
typedef int CUresult; typedef void *CUcontext; typedef uint64_t CUdeviceptr; typedef void *H;
CUresult cuInit(unsigned); CUresult cuDevicePrimaryCtxRetain(CUcontext *, int); CUresult cuCtxSetCurrent(CUcontext);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t); CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t); CUresult cuCtxSynchronize(void); CUresult cuMemFree_v2(CUdeviceptr);
int cublasCreate_v2(H *); int cublasStrsm_v2(H, int, int, int, int, int, int, const float *, const float *, int, float *, int);
#define CK(x) do { int e = (x); if (e) { fprintf(stderr, "%s: %d\n", #x, e); exit(1); } } while (0)
int main(void) {
    CUcontext c; CK(cuInit(0)); CK(cuDevicePrimaryCtxRetain(&c, 0)); CK(cuCtxSetCurrent(c)); H h; CK(cublasCreate_v2(&h));
    int sizes[][2] = {{50, 30}, {30, 50}, {100, 70}, {200, 130}, {64, 64}, {65, 3}, {3, 65}}; double worst = 0; int bad = 0;
    for (int si = 0; si < 7; si++) for (int side = 0; side < 2; side++) for (int up = 0; up < 2; up++) for (int tr = 0; tr < 2; tr++) for (int unit = 0; unit < 2; unit++) {
        int m = sizes[si][0], n = sizes[si][1], na = side ? n : m, lda = na + 3, ldb = m + 5; float alpha = 0.75f;
        float *a = malloc((size_t)lda * na * 4), *b = malloc((size_t)ldb * n * 4), *x = malloc((size_t)ldb * n * 4);
        for (int j = 0; j < na; j++) for (int i = 0; i < lda; i++) a[(size_t)j * lda + i] = i == j ? 1.5f + (float)rand() / RAND_MAX : ((float)rand() / RAND_MAX - 0.5f) * 1.0f / na;
        for (size_t i = 0; i < (size_t)ldb * n; i++) b[i] = (float)rand() / RAND_MAX - 0.5f;
        CUdeviceptr da, db; CK(cuMemAlloc_v2(&da, (size_t)lda * na * 4)); CK(cuMemAlloc_v2(&db, (size_t)ldb * n * 4));
        CK(cuMemcpyHtoD_v2(da, a, (size_t)lda * na * 4)); CK(cuMemcpyHtoD_v2(db, b, (size_t)ldb * n * 4));
        CK(cublasStrsm_v2(h, side, up, tr, unit, m, n, &alpha, (float *)da, lda, (float *)db, ldb)); CK(cuCtxSynchronize());
        CK(cuMemcpyDtoH_v2(x, db, (size_t)ldb * n * 4));
        // op(A)(r, c) with the triangle and diagonal applied
        #define A_(r, c) ({ int rr = tr ? (c) : (r), cc = tr ? (r) : (c); double v = rr == cc ? (unit ? 1.0 : a[(size_t)cc * lda + rr]) : ((up ? rr < cc : rr > cc) ? a[(size_t)cc * lda + rr] : 0.0); v; })
        double err = 0;
        for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { double s = 0;
            if (!side) for (int k = 0; k < m; k++) s += A_(i, k) * x[(size_t)j * ldb + k];
            else for (int k = 0; k < n; k++) s += x[(size_t)k * ldb + i] * A_(k, j);
            err = fmax(err, fabs(s - alpha * b[(size_t)j * ldb + i])); }
        for (int j = 0; j < n; j++) for (int i = m; i < ldb; i++) if (x[(size_t)j * ldb + i] != b[(size_t)j * ldb + i]) { err = 1e9; }
        if (err > 1e-4) { bad++; printf("BAD m %d n %d side %d up %d tr %d unit %d err %.2e\n", m, n, side, up, tr, unit, err); }
        worst = fmax(worst, err); cuMemFree_v2(da); cuMemFree_v2(db); free(a); free(b); free(x);
    }
    printf("trsm cases bad %d worst residual %.2e\n", bad, worst); return bad != 0;
}
