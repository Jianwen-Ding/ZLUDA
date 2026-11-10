// cublasGemmEx with alpha 2, beta 0.5 and a non-zero C (f32 or f16 C): prints a checksum of C and the max relative
// error against a CPU reference, to compare ZLUDA_AGX_GEMM_SPLITK=0 and 1. usage: gemmbeta M N K f32|f16
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
typedef int CUresult; typedef void *CUcontext; typedef uint64_t CUdeviceptr; typedef void *cublasHandle_t;
CUresult cuInit(unsigned); CUresult cuDevicePrimaryCtxRetain(CUcontext *, int); CUresult cuCtxSetCurrent(CUcontext);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t); CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuCtxSynchronize(void); CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t);
int cublasCreate_v2(cublasHandle_t *);
int cublasGemmEx(cublasHandle_t, int, int, int, int, int, const void *, const void *, int, int, const void *, int, int,
                 const void *, void *, int, int, int, int);
static float h2f(uint16_t h) {
    uint32_t e = (h >> 10) & 31, f = h & 1023, s = (uint32_t)(h >> 15) << 31, u = e ? s | (e + 112) << 23 | f << 13 : s;
    float x; memcpy(&x, &u, 4); return x;
}
static uint16_t f2h(float x) {  // round to nearest even, normal range only (the test's values)
    uint32_t u; memcpy(&u, &x, 4); uint32_t s = (u >> 16) & 0x8000; int e = ((u >> 23) & 255) - 112;
    if (e <= 0) return s; uint32_t m = u & 0x7fffff, r = m & 0x1fff; m >>= 13;
    uint32_t h = s | e << 10 | m; if (r > 0x1000 || (r == 0x1000 && (m & 1))) h++; return h;
}
#define CK(x) do { int e_ = (x); if (e_) { fprintf(stderr, "%s: %d\n", #x, e_); exit(1); } } while (0)
int main(int argc, char **argv) {
    int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]), c16 = !strcmp(argv[4], "f16");
    CUcontext ctx; CK(cuInit(0)); CK(cuDevicePrimaryCtxRetain(&ctx, 0)); CK(cuCtxSetCurrent(ctx));
    size_t na = (size_t)m * k, nb = (size_t)k * n, nc = (size_t)m * n;
    uint16_t *ha = malloc(na * 2), *hb = malloc(nb * 2);
    for (size_t i = 0; i < na; i++) ha[i] = f2h((float)((int)((i * 2654435761u) >> 24 & 63) - 31) / 32);
    for (size_t i = 0; i < nb; i++) hb[i] = f2h((float)((int)((i * 40503u + 7) >> 8 & 63) - 31) / 32);
    float *hc = malloc(nc * 4); uint16_t *hc16 = malloc(nc * 2);
    for (size_t i = 0; i < nc; i++) { hc[i] = (float)((int)(i % 13) - 6) / 4; hc16[i] = f2h(hc[i]); }
    CUdeviceptr a, b, c; CK(cuMemAlloc_v2(&a, na * 2)); CK(cuMemAlloc_v2(&b, nb * 2)); CK(cuMemAlloc_v2(&c, nc * 4));
    CK(cuMemcpyHtoD_v2(a, ha, na * 2)); CK(cuMemcpyHtoD_v2(b, hb, nb * 2));
    CK(cuMemcpyHtoD_v2(c, c16 ? (void *)hc16 : (void *)hc, nc * (c16 ? 2 : 4)));
    cublasHandle_t bl; CK(cublasCreate_v2(&bl));
    float al = 2, be = 0.5f; uint16_t al16 = 0x4000, be16 = 0x3800;
    // C (m x n, col-major) = alpha A^T B + beta C, A k x m, B k x n (ggml's layout)
    CK(cublasGemmEx(bl, 1, 0, m, n, k, c16 ? (void *)&al16 : (void *)&al, (void *)a, 2, k, (void *)b, 2, k,
                    c16 ? (void *)&be16 : (void *)&be, (void *)c, c16 ? 2 : 0, m, c16 ? 64 : 68, -1));
    CK(cuCtxSynchronize());
    float *out = malloc(nc * 4);
    if (c16) { CK(cuMemcpyDtoH_v2(hc16, c, nc * 2)); for (size_t i = 0; i < nc; i++) out[i] = h2f(hc16[i]); }
    else CK(cuMemcpyDtoH_v2(out, c, nc * 4));
    double maxrel = 0, sum = 0;
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) {
        double r = 0;
        for (int l = 0; l < k; l++) r += (double)h2f(ha[(size_t)i * k + l]) * h2f(hb[(size_t)j * k + l]);
        r = 2 * r + 0.5 * hc[(size_t)j * m + i];
        double g = out[(size_t)j * m + i], e = fabs(g - r) / (fabs(r) + 1);
        if (e > maxrel) maxrel = e;
        sum += g;
    }
    printf("%dx%dx%d %s maxrel %.3g sum %.6f\n", m, n, k, argv[4], maxrel, sum);
    return 0;
}
