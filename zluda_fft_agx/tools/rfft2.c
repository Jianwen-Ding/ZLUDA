// rfft2 NX NY ITERS [d]: 2-D real transforms of the replacement cuFFT (libcufft_agx) through zluda_agx; own prototypes
// (values from the cuFFT documentation). Checks R2C (D2Z) against a CPU DFT and C2R (Z2D) of it against NX NY times
// the input, then times each.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
typedef int CUresult; typedef void *CUcontext; typedef uint64_t CUdeviceptr; typedef int cufftHandle;
CUresult cuInit(unsigned); CUresult cuDevicePrimaryCtxRetain(CUcontext *, int); CUresult cuCtxSetCurrent(CUcontext);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t); CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t); CUresult cuCtxSynchronize(void);
int cufftPlan2d(cufftHandle *, int, int, int);
int cufftExecR2C(cufftHandle, void *, void *); int cufftExecC2R(cufftHandle, void *, void *);
int cufftExecD2Z(cufftHandle, void *, void *); int cufftExecZ2D(cufftHandle, void *, void *);
#define CK(x) do { int e = (x); if (e) { fprintf(stderr, "%s: %d\n", #x, e); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    int nx = atoi(argv[1]), ny = atoi(argv[2]), it = atoi(argv[3]), dbl = argc > 4 && argv[4][0] == 'd';
    int hy = ny / 2 + 1, es = dbl ? 8 : 4;
    size_t n = (size_t)nx * ny, nh = (size_t)nx * hy;
    CUcontext c; CK(cuInit(0)); CK(cuDevicePrimaryCtxRetain(&c, 0)); CK(cuCtxSetCurrent(c));
    CUdeviceptr x, y, z; CK(cuMemAlloc_v2(&x, n * es)); CK(cuMemAlloc_v2(&y, nh * 2 * es)); CK(cuMemAlloc_v2(&z, n * es));
    double *h = malloc(n * 8); void *hb = malloc(n * es);
    for (size_t i = 0; i < n; i++) { h[i] = sin(0.37 * i) + 0.5 * cos(1.3 * (i % ny)); if (dbl) ((double *)hb)[i] = h[i]; else ((float *)hb)[i] = (float)h[i]; }
    CK(cuMemcpyHtoD_v2(x, hb, n * es));
    cufftHandle pf, pb; CK(cufftPlan2d(&pf, nx, ny, dbl ? 0x6a : 0x2a)); CK(cufftPlan2d(&pb, nx, ny, dbl ? 0x6c : 0x2c));
    int (*fwd)(cufftHandle, void *, void *) = dbl ? cufftExecD2Z : cufftExecR2C;
    int (*bwd)(cufftHandle, void *, void *) = dbl ? cufftExecZ2D : cufftExecC2R;
    CK(fwd(pf, (void *)x, (void *)y)); CK(cuCtxSynchronize());
    void *sb = malloc(nh * 2 * es); CK(cuMemcpyDtoH_v2(sb, y, nh * 2 * es));
    double err = 0, norm = 0;
    if (n <= 16384) for (int a = 0; a < nx; a++) for (int b = 0; b < hy; b++) {
        double re = 0, im = 0;
        for (int i = 0; i < nx; i++) for (int j = 0; j < ny; j++) { double t = -2 * M_PI * ((double)a * i / nx + (double)b * j / ny); re += h[(size_t)i * ny + j] * cos(t); im += h[(size_t)i * ny + j] * sin(t); }
        size_t o = (size_t)a * hy + b; double gr = dbl ? ((double *)sb)[2 * o] : ((float *)sb)[2 * o], gi = dbl ? ((double *)sb)[2 * o + 1] : ((float *)sb)[2 * o + 1];
        err = fmax(err, hypot(gr - re, gi - im)); norm = fmax(norm, hypot(re, im)); }
    CK(bwd(pb, (void *)y, (void *)z)); CK(cuCtxSynchronize());
    void *rb = malloc(n * es); CK(cuMemcpyDtoH_v2(rb, z, n * es));
    double rerr = 0;
    for (size_t i = 0; i < n; i++) rerr = fmax(rerr, fabs((dbl ? ((double *)rb)[i] : ((float *)rb)[i]) / n - h[i]));
    double t = now(); for (int i = 0; i < it; i++) CK(fwd(pf, (void *)x, (void *)y)); CK(cuCtxSynchronize()); double tf = (now() - t) / it;
    t = now(); for (int i = 0; i < it; i++) { CK(cuMemcpyHtoD_v2(y, sb, 0)); CK(bwd(pb, (void *)y, (void *)z)); } CK(cuCtxSynchronize()); double tb = (now() - t) / it;
    printf("{\"type\":\"%s\",\"nx\":%d,\"ny\":%d,\"fwd_us\":%.1f,\"inv_us\":%.1f,\"r2c_err_rel\":%.2e,\"roundtrip_err\":%.2e}\n",
           dbl ? "D2Z/Z2D" : "R2C/C2R", nx, ny, tf * 1e6, tb * 1e6, norm ? err / norm : -1.0, rerr);
    return 0;
}
