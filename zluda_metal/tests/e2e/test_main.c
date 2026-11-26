/* End-to-end tests of ZLUDA's Metal backend through the CUDA driver API.
 * usage: zluda_metal_e2e kernels.ptx */
#include "cuda_min.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0, checks = 0;

#define CU(call)                                                                                 \
    do                                                                                           \
    {                                                                                            \
        CUresult r_ = (call);                                                                    \
        if (r_ != 0)                                                                             \
        {                                                                                        \
            const char *n_ = "?";                                                                \
            cuGetErrorName(r_, &n_);                                                             \
            fprintf(stderr, "%s:%d: %s failed: %s (%d)\n", __FILE__, __LINE__, #call, n_, r_);   \
            exit(2);                                                                             \
        }                                                                                        \
    } while (0)

static void check(const char *test, int ok, const char *fmt, ...)
{
    checks++;
    if (!ok)
    {
        failures++;
        printf("  FAIL %s: %s\n", test, fmt);
    }
}

static void report(const char *test, int before)
{
    printf("[%s] %s\n", failures == before ? " OK " : "FAIL", test);
}

static char *read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
    {
        perror(path);
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n)
        exit(2);
    buf[n] = 0;
    fclose(f);
    return buf;
}

static CUdeviceptr dalloc(size_t n)
{
    CUdeviceptr p;
    CU(cuMemAlloc(&p, n));
    return p;
}

static CUfunction get(CUmodule m, const char *name)
{
    CUfunction f;
    CU(cuModuleGetFunction(&f, m, name));
    return f;
}

static void test_vadd(CUmodule m)
{
    int before = failures;
    const int n = 10000;
    float *a = malloc(n * 4), *b = malloc(n * 4), *c = malloc(n * 4);
    for (int i = 0; i < n; i++)
    {
        a[i] = (float)i * 0.5f;
        b[i] = (float)(n - i);
    }
    CUdeviceptr da = dalloc(n * 4), db = dalloc(n * 4), dc = dalloc(n * 4);
    CU(cuMemcpyHtoD(da, a, n * 4));
    CU(cuMemcpyHtoD(db, b, n * 4));
    int nn = n;
    void *args[] = {&da, &db, &dc, &nn};
    CU(cuLaunchKernel(get(m, "vadd"), (n + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
    CU(cuMemcpyDtoH(c, dc, n * 4));
    int bad = 0;
    for (int i = 0; i < n; i++)
        bad += c[i] != a[i] + b[i];
    check("vadd", bad == 0, "mismatching elements");
    cuMemFree(da);
    cuMemFree(db);
    cuMemFree(dc);
    free(a);
    free(b);
    free(c);
    report("vadd (10000 floats, 40 blocks)", before);
}

static void test_saxpy(CUmodule m)
{
    int before = failures;
    const int n = 5000;
    float *x = malloc(n * 4), *y = malloc(n * 4), *r = malloc(n * 4);
    for (int i = 0; i < n; i++)
    {
        x[i] = (float)(i % 17);
        y[i] = (float)(i % 5);
    }
    CUdeviceptr dx = dalloc(n * 4), dy = dalloc(n * 4);
    CU(cuMemcpyHtoD(dx, x, n * 4));
    CU(cuMemcpyHtoD(dy, y, n * 4));
    int nn = n;
    float alpha = 2.5f;
    void *args[] = {&nn, &alpha, &dx, &dy};
    /* grid-stride loop: fewer threads than elements */
    CU(cuLaunchKernel(get(m, "saxpy"), 4, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
    CU(cuMemcpyDtoH(r, dy, n * 4));
    int bad = 0;
    for (int i = 0; i < n; i++)
        bad += fabsf(r[i] - (alpha * x[i] + y[i])) > 1e-5f;
    check("saxpy", bad == 0, "mismatch");
    cuMemFree(dx);
    cuMemFree(dy);
    free(x);
    free(y);
    free(r);
    report("saxpy (scalar float param, grid-stride loop)", before);
}

static void test_reduce(CUmodule m)
{
    int before = failures;
    const int n = 100000;
    int *in = malloc(n * 4);
    long long expect = 0;
    for (int i = 0; i < n; i++)
    {
        in[i] = (i * 7919) % 1000 - 300;
        expect += in[i];
    }
    CUdeviceptr din = dalloc(n * 4), dout = dalloc(4);
    CU(cuMemcpyHtoD(din, in, n * 4));
    CU(cuMemsetD32(dout, 0, 1));
    int nn = n;
    void *args[] = {&din, &dout, &nn};
    CU(cuLaunchKernel(get(m, "reduce_sum"), (n + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
    int got = 0;
    CU(cuMemcpyDtoH(&got, dout, 4));
    check("reduce_sum", got == (int)expect, "wrong sum");
    if (got != (int)expect)
        printf("    got %d expected %lld\n", got, expect);
    int shared = 0;
    CU(cuFuncGetAttribute(&shared, 1, get(m, "reduce_sum")));
    check("reduce_sum shared size", shared == 1024, "static shared size");
    cuMemFree(din);
    cuMemFree(dout);
    free(in);
    report("reduce_sum (static shared memory, barriers, atomicAdd)", before);
}

static void test_block_reverse(CUmodule m)
{
    int before = failures;
    const int bs = 192, blocks = 7, n = bs * blocks;
    float *in = malloc(n * 4), *out = malloc(n * 4);
    for (int i = 0; i < n; i++)
        in[i] = (float)i;
    CUdeviceptr din = dalloc(n * 4), dout = dalloc(n * 4);
    CU(cuMemcpyHtoD(din, in, n * 4));
    void *args[] = {&din, &dout};
    CU(cuLaunchKernel(get(m, "block_reverse"), blocks, 1, 1, bs, 1, 1, bs * 4, NULL, args, NULL));
    CU(cuMemcpyDtoH(out, dout, n * 4));
    int bad = 0;
    for (int b = 0; b < blocks; b++)
        for (int t = 0; t < bs; t++)
            bad += out[b * bs + t] != in[b * bs + bs - 1 - t];
    check("block_reverse", bad == 0, "mismatch");
    cuMemFree(din);
    cuMemFree(dout);
    free(in);
    free(out);
    report("block_reverse (dynamic shared memory)", before);
}

static void test_warp_ops(CUmodule m)
{
    int before = failures;
    const int n = 256;
    int in[256], sums[8], xors[256], bcast[256];
    for (int i = 0; i < n; i++)
        in[i] = i * 3 + 1;
    CUdeviceptr din = dalloc(n * 4), ds = dalloc(8 * 4), dx = dalloc(n * 4), db = dalloc(n * 4);
    CU(cuMemcpyHtoD(din, in, sizeof in));
    void *args[] = {&din, &ds, &dx, &db};
    CU(cuLaunchKernel(get(m, "warp_ops"), 2, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
    CU(cuMemcpyDtoH(sums, ds, sizeof sums));
    CU(cuMemcpyDtoH(xors, dx, sizeof xors));
    CU(cuMemcpyDtoH(bcast, db, sizeof bcast));
    int bad = 0;
    for (int w = 0; w < 8; w++)
    {
        int s = 0;
        for (int l = 0; l < 32; l++)
            s += in[w * 32 + l];
        bad += sums[w] != s;
        for (int l = 0; l < 32; l++)
        {
            bad += xors[w * 32 + l] != s;
            bad += bcast[w * 32 + l] != in[w * 32 + 7];
        }
    }
    check("warp_ops", bad == 0, "shuffle results");
    cuMemFree(din);
    cuMemFree(ds);
    cuMemFree(dx);
    cuMemFree(db);
    report("warp_ops (shfl.sync down/bfly/idx)", before);
}

static void test_ballot(CUmodule m)
{
    int before = failures;
    int in[128];
    unsigned ballots[4];
    int counts[4];
    for (int i = 0; i < 128; i++)
        in[i] = ((i * 37) % 11) > 5 ? 1 : -1;
    CUdeviceptr din = dalloc(sizeof in), db = dalloc(sizeof ballots), dc = dalloc(sizeof counts);
    CU(cuMemcpyHtoD(din, in, sizeof in));
    void *args[] = {&din, &db, &dc};
    CU(cuLaunchKernel(get(m, "ballot_popc"), 1, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
    CU(cuMemcpyDtoH(ballots, db, sizeof ballots));
    CU(cuMemcpyDtoH(counts, dc, sizeof counts));
    int bad = 0;
    for (int w = 0; w < 4; w++)
    {
        unsigned e = 0;
        for (int l = 0; l < 32; l++)
            if (in[w * 32 + l] > 0)
                e |= 1u << l;
        bad += ballots[w] != e;
        bad += counts[w] != __builtin_popcount(e);
    }
    check("ballot", bad == 0, "ballot/popc");
    cuMemFree(din);
    cuMemFree(db);
    cuMemFree(dc);
    report("ballot_popc (vote.ballot.sync, popc)", before);
}

static void test_transpose(CUmodule m)
{
    int before = failures;
    const int w = 70, h = 45;
    float *in = malloc(w * h * 4), *out = malloc(w * h * 4);
    for (int i = 0; i < w * h; i++)
        in[i] = (float)i;
    CUdeviceptr din = dalloc(w * h * 4), dout = dalloc(w * h * 4);
    CU(cuMemcpyHtoD(din, in, w * h * 4));
    int ww = w, hh = h;
    void *args[] = {&din, &dout, &ww, &hh};
    CU(cuLaunchKernel(get(m, "transpose"), (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, NULL, args, NULL));
    CU(cuMemcpyDtoH(out, dout, w * h * 4));
    int bad = 0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            bad += out[x * h + y] != in[y * w + x];
    check("transpose", bad == 0, "mismatch");
    cuMemFree(din);
    cuMemFree(dout);
    free(in);
    free(out);
    report("transpose (2D grid and blocks, padded shared tile)", before);
}

static void test_atomics(CUmodule m)
{
    int before = failures;
    const int n = 3000;
    int *in = malloc(n * 4);
    int mx = -1000000, mn = 1000000, orv = 0;
    for (int i = 0; i < n; i++)
    {
        in[i] = (i * 2654435761u) % 100003 - 50000;
        if (in[i] > mx)
            mx = in[i];
        if (in[i] < mn)
            mn = in[i];
        orv |= 1 << (in[i] & 15);
    }
    int stats[4] = {-2147483647, 2147483647, 0, 0};
    float fsum = 0;
    CUdeviceptr din = dalloc(n * 4), ds = dalloc(16), df = dalloc(4);
    CU(cuMemcpyHtoD(din, in, n * 4));
    CU(cuMemcpyHtoD(ds, stats, 16));
    CU(cuMemcpyHtoD(df, &fsum, 4));
    int nn = n;
    void *args[] = {&din, &nn, &ds, &df};
    CU(cuLaunchKernel(get(m, "atomics"), (n + 127) / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
    CU(cuMemcpyDtoH(stats, ds, 16));
    CU(cuMemcpyDtoH(&fsum, df, 4));
    check("atomicMax", stats[0] == mx, "max");
    check("atomicMin", stats[1] == mn, "min");
    check("atomicCAS loop", stats[2] == n, "cas count");
    check("atomicOr", stats[3] == orv, "or");
    check("atomicAdd f32", fsum == n * 0.5f, "float sum");
    cuMemFree(din);
    cuMemFree(ds);
    cuMemFree(df);
    free(in);
    report("atomics (max, min, CAS loop, or, float add)", before);
}

static int poly_ref(const int *c, int x)
{
    int local[8];
    for (int k = 0; k < 8; k++)
        local[k] = c[k] * (k + 1);
    int r = 0;
    for (int k = 7; k >= 0; k--)
        r = r * x + local[(k * 3) & 7];
    return r;
}

static void test_call(CUmodule m)
{
    int before = failures;
    int coeffs[8] = {3, -1, 4, 1, -5, 9, 2, -6};
    const int n = 100;
    int out[100];
    CUdeviceptr dc = dalloc(sizeof coeffs), dout = dalloc(sizeof out);
    CU(cuMemcpyHtoD(dc, coeffs, sizeof coeffs));
    int nn = n;
    void *args[] = {&dc, &dout, &nn};
    CU(cuLaunchKernel(get(m, "call_device_fn"), 1, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
    CU(cuMemcpyDtoH(out, dout, sizeof out));
    int bad = 0;
    for (int i = 0; i < n; i++)
        bad += out[i] != poly_ref(coeffs, i % 5 - 2);
    check("call", bad == 0, "mismatch");
    cuMemFree(dc);
    cuMemFree(dout);
    report("call_device_fn (.func call, private array)", before);
}

struct Params
{
    char tag;
    double unused_but_aligned;
    int scale;
    short offset;
};

static void test_struct_param(CUmodule m)
{
    int before = failures;
    struct Params p = {5, 1.0, 3, -7};
    unsigned char bias = 200;
    const int per_block = 4 * 3 * 2, blocks = 3, n = per_block * blocks;
    int out[72];
    CUdeviceptr dout = dalloc(sizeof out);
    void *args[] = {&p, &bias, &dout};
    CU(cuLaunchKernel(get(m, "struct_param"), blocks, 1, 1, 4, 3, 2, 0, NULL, args, NULL));
    CU(cuMemcpyDtoH(out, dout, sizeof out));
    int bad = 0;
    for (int g = 0; g < n; g++)
        bad += out[g] != g * p.scale + p.offset + p.tag + bias;
    check("struct_param", bad == 0, "mismatch");
    cuMemFree(dout);
    report("struct_param (by-value struct, u8 param, 3D blocks)", before);
}

static unsigned rev32(unsigned x)
{
    unsigned r = 0;
    for (int i = 0; i < 32; i++)
        if (x & (1u << i))
            r |= 1u << (31 - i);
    return r;
}

static void test_math(CUmodule m)
{
    int before = failures;
    const int n = 64;
    unsigned u[64], uo[256];
    float f[64], fo[256];
    for (int i = 0; i < n; i++)
    {
        u[i] = i == 0 ? 0 : (0x9E3779B9u * i) >> (i % 9);
        f[i] = 0.25f + 0.73f * i;
    }
    CUdeviceptr du = dalloc(sizeof u), df = dalloc(sizeof f), duo = dalloc(sizeof uo), dfo = dalloc(sizeof fo);
    CU(cuMemcpyHtoD(du, u, sizeof u));
    CU(cuMemcpyHtoD(df, f, sizeof f));
    int nn = n;
    void *args[] = {&du, &df, &duo, &dfo, &nn};
    CU(cuLaunchKernel(get(m, "math_ops"), 1, 1, 1, 64, 1, 1, 0, NULL, args, NULL));
    CU(cuMemcpyDtoH(uo, duo, sizeof uo));
    CU(cuMemcpyDtoH(fo, dfo, sizeof fo));
    int bad = 0;
    for (int i = 0; i < n; i++)
    {
        unsigned x = u[i];
        bad += uo[4 * i] != (unsigned)__builtin_popcount(x);
        bad += uo[4 * i + 1] != (x ? (unsigned)__builtin_clz(x) : 32u);
        bad += uo[4 * i + 2] != rev32(x);
        bad += uo[4 * i + 3] != ((x >> 7) | (x << 25));
        float y = f[i];
        bad += fabsf(fo[4 * i] - sqrtf(y)) > 1e-6f * sqrtf(y);
        bad += fabsf(fo[4 * i + 1] - y / 3.0f) > 1e-6f * y;
        bad += fo[4 * i + 2] != fminf(y, 2.0f) + fmaxf(y, 1.0f);
        bad += fo[4 * i + 3] != floorf(y * 1.5f) - (float)(int)(y * 2.25f);
    }
    check("math", bad == 0, "mismatch");
    if (bad)
        printf("    %d mismatches\n", bad);
    cuMemFree(du);
    cuMemFree(df);
    cuMemFree(duo);
    cuMemFree(dfo);
    report("math_ops (popc, clz, brev, funnel shift, sqrt, div, min/max, floor, cvt)", before);
}

static void test_shared_scan(CUmodule m)
{
    int before = failures;
    const int blocks = 3, n = 128 * blocks;
    int in[384], out[384];
    for (int i = 0; i < n; i++)
        in[i] = i % 13 - 4;
    CUdeviceptr din = dalloc(sizeof in), dout = dalloc(sizeof out);
    CU(cuMemcpyHtoD(din, in, sizeof in));
    void *args[] = {&din, &dout};
    CU(cuLaunchKernel(get(m, "shared_scan"), blocks, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
    CU(cuMemcpyDtoH(out, dout, sizeof out));
    int bad = 0;
    for (int b = 0; b < blocks; b++)
    {
        int acc = 0;
        for (int t = 0; t < 128; t++)
        {
            acc += in[b * 128 + t];
            bad += out[b * 128 + t] != acc;
        }
    }
    check("shared_scan", bad == 0, "mismatch");
    cuMemFree(din);
    cuMemFree(dout);
    report("shared_scan (pointer arithmetic over shared memory)", before);
}

static void test_device_globals(CUmodule m)
{
    int before = failures;
    const int n = 300;
    int idx[300];
    float out[300];
    for (int i = 0; i < n; i++)
        idx[i] = i * 5;
    CUdeviceptr didx = dalloc(sizeof idx), dout = dalloc(sizeof out), counter, bias;
    size_t csize = 0, bsize = 0;
    CU(cuModuleGetGlobal_v2(&counter, &csize, m, "g_counter"));
    CU(cuModuleGetGlobal_v2(&bias, &bsize, m, "g_bias"));
    check("global sizes", csize == 4 && bsize == 16, "sizes");
    float init[4];
    CU(cuMemcpyDtoH(init, bias, 16));
    check("global initializer", init[0] == 1.0f && init[3] == 4.0f, "initializer");
    int zero = 0;
    CU(cuMemcpyHtoD(counter, &zero, 4));
    CU(cuMemcpyHtoD(didx, idx, sizeof idx));
    int nn = n;
    void *args[] = {&didx, &dout, &nn};
    CU(cuLaunchKernel(get(m, "device_globals"), 3, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
    CU(cuMemcpyDtoH(out, dout, sizeof out));
    static const int table[8] = {1, 1, 2, 3, 5, 8, 13, 21};
    int bad = 0;
    for (int i = 0; i < n; i++)
        bad += out[i] != table[(i * 5) & 7] * (float)((i & 3) + 1);
    check("device_globals values", bad == 0, "values");
    int count = 0;
    CU(cuMemcpyDtoH(&count, counter, 4));
    check("device_globals counter", count == n, "counter");
    CUdeviceptr last;
    CU(cuModuleGetGlobal_v2(&last, NULL, m, "g_last"));
    int lastv = 0;
    CU(cuMemcpyDtoH(&lastv, last, 4));
    check("device_globals kernel write", lastv == 1234, "kernel write");
    cuMemFree(didx);
    cuMemFree(dout);
    report("device_globals (__device__ variables via cuModuleGetGlobal, __constant__ table)", before);
}

static void test_api(void)
{
    int before = failures;
    int v = 0;
    CU(cuDriverGetVersion(&v));
    check("version", v >= 12000, "driver version");
    int count = 0;
    CU(cuDeviceGetCount(&count));
    check("device count", count == 1, "count");
    int warp = 0;
    CU(cuDeviceGetAttribute(&warp, 10, 0));
    check("warp size", warp == 32, "warp");
    CUdeviceptr p = dalloc(1000), base = 0;
    size_t size = 0;
    CU(cuMemGetAddressRange_v2(&base, &size, p + 10));
    check("address range", base == p && size == 1000, "range");
    unsigned pattern[250];
    CU(cuMemsetD32(p, 0xdeadbeef, 250));
    CUdeviceptr q = dalloc(1000);
    CU(cuMemcpyDtoD(q, p, 1000));
    CU(cuMemcpyDtoH(pattern, q, 1000));
    int bad = 0;
    for (int i = 0; i < 250; i++)
        bad += pattern[i] != 0xdeadbeefu;
    check("memset/dtod", bad == 0, "pattern");
    CUstream s;
    CUevent e0, e1;
    CU(cuStreamCreate(&s, 0));
    CU(cuEventCreate(&e0, 0));
    CU(cuEventCreate(&e1, 0));
    CU(cuEventRecord(e0, s));
    CU(cuEventRecord(e1, s));
    float ms = -1;
    CU(cuEventElapsedTime(&ms, e0, e1));
    check("events", ms >= 0, "elapsed");
    CU(cuStreamSynchronize(s));
    CU(cuStreamDestroy_v2(s));
    cuEventDestroy_v2(e0);
    cuEventDestroy_v2(e1);
    cuMemFree(p);
    cuMemFree(q);
    report("driver API basics (attributes, memset, DtoD, address range, streams, events)", before);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "usage: %s kernels.ptx\n", argv[0]);
        return 2;
    }
    char *ptx = read_file(argv[1]);
    CU(cuInit(0));
    CUdevice dev;
    CU(cuDeviceGet(&dev, 0));
    char name[256];
    CU(cuDeviceGetName(name, sizeof name, dev));
    printf("device: %s\n", name);
    CUcontext ctx;
    CU(cuCtxCreate(&ctx, 0, dev));
    CUmodule m;
    CU(cuModuleLoadData(&m, ptx));
    test_api();
    test_vadd(m);
    test_saxpy(m);
    test_reduce(m);
    test_block_reverse(m);
    test_warp_ops(m);
    test_ballot(m);
    test_transpose(m);
    test_atomics(m);
    test_call(m);
    test_struct_param(m);
    test_math(m);
    test_shared_scan(m);
    test_device_globals(m);
    CU(cuModuleUnload(m));
    CU(cuCtxDestroy(ctx));
    printf("%d checks, %d failures\n", checks, failures);
    free(ptx);
    return failures ? 1 : 0;
}
