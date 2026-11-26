// Device runtime tests (tests/device/test_device.py): printf, malloc / free, clocks, __nanosleep.
// Compiled with clang alone: clang++ -x cuda --cuda-device-only -nocudainc -nocudalib (no NVIDIA header); the
// declarations below are written here, as CUDA documents them.
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#include <__clang_cuda_builtin_vars.h>
typedef unsigned long long u64;
typedef __SIZE_TYPE__ size_t;

// clang lowers printf calls in device code to vprintf(format, argument buffer)
extern "C" __device__ int printf(const char *, ...);
extern "C" __device__ void *malloc(size_t);
extern "C" __device__ void free(void *);

static __device__ inline u64 clock64() { return __nvvm_read_ptx_sreg_clock64(); }
static __device__ inline u64 globaltimer()
{
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
static __device__ inline void nanosleep(unsigned ns) { asm volatile("nanosleep.u32 %0;" ::"r"(ns)); }
static __device__ inline unsigned smid()
{
    unsigned r;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
    return r;
}
static __device__ inline unsigned nsmid()
{
    unsigned r;
    asm volatile("mov.u32 %0, %%nsmid;" : "=r"(r));
    return r;
}

extern "C" __global__ void print_basic(int *ret)
{
    int r = printf("tid %d of %u: %x %5.2f %e %g %c %s %lld %llu %% %08.3f|%-4d|\n", (int)threadIdx.x - 2, blockDim.x,
                   255u + threadIdx.x, 3.14159, 12345.678, 0.0001, 'A' + (int)threadIdx.x, "str",
                   -1234567890123ll, 18446744073709551615ull, 2.5f, 7);
    ret[threadIdx.x] = r;
}

extern "C" __global__ void print_misc(int *ret)
{
    ret[0] = printf("no args\n");
    ret[1] = printf("%s/%s %p %*d|%.*f|%hhd %hd %ld\n", (threadIdx.x & 1) ? "odd" : "even", (const char *)0, (void *)0x1234,
                    6, 42, 2, 1.23456, 0x1ff, 0x18000, -5l);
}

extern "C" __global__ void print_many(int n)
{
    for (int i = 0; i < n; i++)
        printf("line %d block %d\n", i, (int)blockIdx.x);
}

// every thread allocates, writes, reads back; half the blocks are freed and reallocated
extern "C" __global__ void heap_basic(u64 *out, int bytes)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    int *p = (int *)malloc(bytes);
    out[3 * t] = (u64)p;
    if (!p)
        return;
    for (int i = 0; i < bytes / 4; i++)
        p[i] = t * 1000 + i;
    int ok = 1;
    for (int i = 0; i < bytes / 4; i++)
        ok &= p[i] == (int)(t * 1000 + i);
    free(p);
    int *q = (int *)malloc(bytes);
    out[3 * t + 1] = (u64)q;
    out[3 * t + 2] = ok;
}

// allocated in one launch, read and freed in the next
extern "C" __global__ void heap_keep(u64 *slot, int v)
{
    int *p = (int *)malloc(64);
    *p = v;
    *slot = (u64)p;
}
extern "C" __global__ void heap_read(u64 *slot, int *out)
{
    int *p = (int *)*slot;
    *out = *p;
    free(p);
}
extern "C" __global__ void heap_big(u64 *out, u64 bytes)
{
    out[0] = (u64)malloc(bytes);
}

extern "C" __global__ void clocks(u64 *out)
{
    u64 c0 = clock64();
    u64 g0 = globaltimer();
    nanosleep(100000);
    u64 c1 = clock64();
    u64 g1 = globaltimer();
    unsigned c32 = (unsigned)__nvvm_read_ptx_sreg_clock();
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    out[8 * t + 0] = c0;
    out[8 * t + 1] = c1;
    out[8 * t + 2] = g0;
    out[8 * t + 3] = g1;
    out[8 * t + 4] = c32;
    out[8 * t + 5] = smid();
    out[8 * t + 6] = nsmid();
    out[8 * t + 7] = __nvvm_read_ptx_sreg_warpid();
}
