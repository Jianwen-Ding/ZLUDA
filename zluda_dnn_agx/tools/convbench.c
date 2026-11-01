// Build: gcc -O2 -o convbench convbench.c -L$LIB -lcuda -lcudnn (LIB: libcuda.so -> libnvcuda_agx.so, libcudnn.so ->
// libcudnn_agx.so); run with LD_LIBRARY_PATH=$LIB. maxerr: against a CPU convolution at ~1000 sampled outputs.
// convbench: cudnnConvolutionForward of the replacement cuDNN (libcudnn_agx) through zluda_agx's driver API; own
// prototypes, enum values from the cuDNN documentation. convbench N C H W K R ITERS [bd|bf]; bd:
// cudnnConvolutionBackwardData (dx from dy, a pattern in y), bf: cudnnConvolutionBackwardFilter (dw from x and dy),
// checked the same way.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef int CUresult; typedef void *CUcontext; typedef uint64_t CUdeviceptr; typedef void *H;
CUresult cuInit(unsigned); CUresult cuDevicePrimaryCtxRetain(CUcontext *, int); CUresult cuCtxSetCurrent(CUcontext);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t); CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t); CUresult cuCtxSynchronize(void);
int cudnnCreate(H *); int cudnnCreateTensorDescriptor(H *); int cudnnSetTensor4dDescriptor(H, int, int, int, int, int, int);
int cudnnCreateFilterDescriptor(H *); int cudnnSetFilter4dDescriptor(H, int, int, int, int, int, int);
int cudnnCreateConvolutionDescriptor(H *); int cudnnSetConvolution2dDescriptor(H, int, int, int, int, int, int, int, int);
int cudnnConvolutionForward(H, const void *, H, const void *, H, const void *, H, int, void *, size_t, const void *, H, void *);
int cudnnConvolutionBackwardData(H, const void *, H, const void *, H, const void *, H, int, void *, size_t, const void *, H, void *);
int cudnnConvolutionBackwardFilter(H, const void *, H, const void *, H, const void *, H, int, void *, size_t, const void *, H, void *);
#define CK(x) do { int e = (x); if (e) { fprintf(stderr, "%s: %d\n", #x, e); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    int n = atoi(argv[1]), c = atoi(argv[2]), h = atoi(argv[3]), w = atoi(argv[4]), k = atoi(argv[5]), r = atoi(argv[6]), it = atoi(argv[7]);
    int pad = r / 2, bd = argc > 8 && !strcmp(argv[8], "bd"), bf = argc > 8 && !strcmp(argv[8], "bf");
    CUcontext ctx; CK(cuInit(0)); CK(cuDevicePrimaryCtxRetain(&ctx, 0)); CK(cuCtxSetCurrent(ctx));
    size_t xs = (size_t)n * c * h * w, ws = (size_t)k * c * r * r, ys = (size_t)n * k * h * w;
    CUdeviceptr x, wt, y; CK(cuMemAlloc_v2(&x, xs * 4)); CK(cuMemAlloc_v2(&wt, ws * 4)); CK(cuMemAlloc_v2(&y, ys * 4));
    float *hb = malloc((xs > ws ? xs : ws) * 4);
    for (size_t i = 0; i < xs; i++) hb[i] = (float)((i * 2654435761u >> 20) & 15) / 16.0f;
    CK(cuMemcpyHtoD_v2(x, hb, xs * 4));
    for (size_t i = 0; i < ws; i++) hb[i] = (float)((i * 40503u >> 7) & 7) / 8.0f - 0.4375f;
    CK(cuMemcpyHtoD_v2(wt, hb, ws * 4));
    if (bd || bf) {
        // dy: the same kind of pattern, in y
        float *hd = malloc(ys * 4);
        for (size_t i = 0; i < ys; i++) hd[i] = (float)((i * 2246822519u >> 21) & 15) / 16.0f - 0.5f;
        CK(cuMemcpyHtoD_v2(y, hd, ys * 4));
        free(hd);
    }
    H d, xd, yd, wd, cd; CK(cudnnCreate(&d));
    CK(cudnnCreateTensorDescriptor(&xd)); CK(cudnnSetTensor4dDescriptor(xd, 0, 0, n, c, h, w));
    CK(cudnnCreateTensorDescriptor(&yd)); CK(cudnnSetTensor4dDescriptor(yd, 0, 0, n, k, h, w));
    CK(cudnnCreateFilterDescriptor(&wd)); CK(cudnnSetFilter4dDescriptor(wd, 0, 0, k, c, r, r));
    CK(cudnnCreateConvolutionDescriptor(&cd)); CK(cudnnSetConvolution2dDescriptor(cd, pad, pad, 1, 1, 1, 1, 1, 0));
    float one = 1, zero = 0;
#define CONV() (bf ? cudnnConvolutionBackwardFilter(d, &one, xd, (void *)x, yd, (void *)y, cd, 1, NULL, 0, &zero, wd, (void *)wt) \
              : bd ? cudnnConvolutionBackwardData(d, &one, wd, (void *)wt, yd, (void *)y, cd, 1, NULL, 0, &zero, xd, (void *)x) \
                   : cudnnConvolutionForward(d, &one, xd, (void *)x, wd, (void *)wt, cd, 0, NULL, 0, &zero, yd, (void *)y))
    float *hx0 = malloc(xs * 4); CK(cuMemcpyDtoH_v2(hx0, x, xs * 4));
    float *hw0 = malloc(ws * 4); CK(cuMemcpyDtoH_v2(hw0, wt, ws * 4));
    // warm the GPU clock first: ~0.3 s of the same work queued in bursts (one call measures a partly idle clock)
    for (double w = now(); now() - w < 0.3;) {
        for (int i = 0; i < 10; i++) CK(CONV());
        CK(cuCtxSynchronize());
    }
    double t = now();
    for (int i = 0; i < it; i++) CK(CONV());
    double host = now() - t; CK(cuCtxSynchronize()); double all = now() - t;
    // the result: y (forward) or dx (bd)
    size_t os = bf ? ws : bd ? xs : ys;
    float *hy = malloc(os * 4); CK(cuMemcpyDtoH_v2(hy, bf ? wt : bd ? x : y, os * 4)); double sum = 0; uint64_t bits = 0;
    for (size_t i = 0; i < os; i++) { sum += hy[i]; uint32_t u; memcpy(&u, &hy[i], 4); bits = bits * 31 + u; }
    // CPU reference at sampled outputs (cross-correlation, zero padding)
    float *hx = hx0, *hw = hw0;
    float *hdy = malloc(ys * 4); CK(cuMemcpyDtoH_v2(hdy, y, ys * 4));
    double maxerr = 0;
    // data gradient: dx[n, c, iy, ix] = sum over k, a, b of dy[n, k, iy - a + pad, ix - b + pad] w[k, c, a, b]
    for (size_t s = 0; bd && s < xs; s += xs / 997 + 1) {
        int ix = s % w, iy = s / w % h, ci = s / ((size_t)w * h) % c, in = s / ((size_t)w * h * c);
        double ref = 0;
        for (int ki = 0; ki < k; ki++) for (int a = 0; a < r; a++) for (int b = 0; b < r; b++) {
            int oy = iy - a + pad, ox = ix - b + pad;
            if (oy < 0 || oy >= h || ox < 0 || ox >= w) continue;
            ref += (double)hdy[(((size_t)in * k + ki) * h + oy) * w + ox] * hw[(((size_t)ki * c + ci) * r + a) * r + b];
        }
        double e = ref - hy[s]; if (e < 0) e = -e; if (e > maxerr) maxerr = e;
    }
    // filter gradient: dw[k, c, a, b] = sum over n, oy, ox of dy[n, k, oy, ox] x[n, c, oy + a - pad, ox + b - pad]
    for (size_t s = 0; bf && s < ws; s += ws / 997 + 1) {
        int b = s % r, a = s / r % r, ci = s / ((size_t)r * r) % c, ki = s / ((size_t)r * r * c);
        double ref = 0;
        for (int ni = 0; ni < n; ni++) for (int oy = 0; oy < h; oy++) for (int ox = 0; ox < w; ox++) {
            int iy = oy + a - pad, ix = ox + b - pad;
            if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
            ref += (double)hdy[(((size_t)ni * k + ki) * h + oy) * w + ox] * hx[(((size_t)ni * c + ci) * h + iy) * w + ix];
        }
        double e = ref - hy[s]; if (e < 0) e = -e; if (e > maxerr) maxerr = e;
    }
    for (size_t s = 0; !bd && !bf && s < ys; s += ys / 997 + 1) {
        int ox = s % w, oy = s / w % h, ok = s / ((size_t)w * h) % k, on = s / ((size_t)w * h * k);
        double ref = 0;
        for (int ci = 0; ci < c; ci++) for (int a = 0; a < r; a++) for (int b = 0; b < r; b++) {
            int iy = oy + a - pad, ix = ox + b - pad;
            if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
            ref += (double)hx[(((size_t)on * c + ci) * h + iy) * w + ix] * hw[(((size_t)ok * c + ci) * r + a) * r + b];
        }
        double e = ref - hy[s]; if (e < 0) e = -e; if (e > maxerr) maxerr = e;
    }
    printf("{\"maxerr\":%.3g,", maxerr);
    printf("\"n\":%d,\"c\":%d,\"hw\":%d,\"k\":%d,\"r\":%d,\"ms_per_conv\":%.3f,\"host_ms_per_call\":%.3f,\"sum\":%.6e,\"bits\":\"%016llx\"}\n",
           n, c, h, k, r, all * 1e3 / it, host * 1e3 / it, sum, (unsigned long long)bits);
    return 0;
}
