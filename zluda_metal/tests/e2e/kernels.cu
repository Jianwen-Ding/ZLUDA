// Test kernels for the ZLUDA Metal backend. Compiled to PTX with:
//   clang++ -x cuda --cuda-device-only -nocudainc -nocudalib --cuda-gpu-arch=sm_70 -O2 -S -include clang_cuda.h
#define EXPORT extern "C" __global__

EXPORT void vadd(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

EXPORT void saxpy(int n, float alpha, const float *x, float *y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        y[i] = fmaf_(alpha, x[i], y[i]);
}

// static shared memory, barriers, global atomics
EXPORT void reduce_sum(const int *in, int *out, int n)
{
    __shared__ int s[256];
    int t = threadIdx.x;
    int i = blockIdx.x * 256 + t;
    s[t] = i < n ? in[i] : 0;
    __syncthreads();
    for (int k = 128; k > 0; k >>= 1)
    {
        if (t < k)
            s[t] += s[t + k];
        __syncthreads();
    }
    if (t == 0)
        atomicAdd(out, s[0]);
}

// dynamic shared memory (extern __shared__) with a reversal inside a block
extern __shared__ float dyn_smem[];
EXPORT void block_reverse(const float *in, float *out)
{
    int t = threadIdx.x;
    int base = blockIdx.x * blockDim.x;
    dyn_smem[t] = in[base + t];
    __syncthreads();
    out[base + t] = dyn_smem[blockDim.x - 1 - t];
}

// warp shuffles: warp-level sum reduction + butterfly + broadcast
EXPORT void warp_ops(const int *in, int *sum_out, int *xor_out, int *bcast_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int v = in[i];
    int s = v;
    for (int d = 16; d > 0; d >>= 1)
        s += __shfl_down_sync(0xffffffffu, s, d);
    int x = v;
    for (int m = 1; m < 32; m <<= 1)
        x += __shfl_xor_sync(0xffffffffu, x, m);
    int b = __shfl_sync(0xffffffffu, v, 7);
    if ((threadIdx.x & 31) == 0)
        sum_out[i / 32] = s;
    xor_out[i] = x;
    bcast_out[i] = b;
}

EXPORT void ballot_popc(const int *in, unsigned *ballots, int *counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned m = __ballot_sync(0xffffffffu, in[i] > 0);
    if ((threadIdx.x & 31) == 0)
    {
        ballots[i / 32] = m;
        counts[i / 32] = __popc(m);
    }
}

// 2D grid, tiled transpose through shared memory with padding
#define TILE 16
EXPORT void transpose(const float *in, float *out, int w, int h)
{
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < w && y < h)
        tile[threadIdx.y][threadIdx.x] = in[y * w + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < h && y < w)
        out[y * h + x] = tile[threadIdx.x][threadIdx.y];
}

// atomics: max/min/CAS-based increment/float add
EXPORT void atomics(const int *in, int n, int *stats, float *fsum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    atomicMax(&stats[0], in[i]);
    atomicMin(&stats[1], in[i]);
    int old = stats[2], assumed;
    do
    {
        assumed = old;
        old = atomicCAS(&stats[2], assumed, assumed + 1);
    } while (old != assumed);
    atomicOr(&stats[3], 1 << (in[i] & 15));
    atomicAdd(fsum, 0.5f);
}

// non-inlined device function with a private array (thread memory)
__device__ __attribute__((noinline)) int poly(const int *coeffs, int x)
{
    int local[8];
    for (int k = 0; k < 8; k++)
        local[k] = coeffs[k] * (k + 1);
    int r = 0;
    for (int k = 7; k >= 0; k--)
        r = r * x + local[(k * 3) & 7];
    return r;
}

EXPORT void call_device_fn(const int *coeffs, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = poly(coeffs, i % 5 - 2);
}

// by-value struct parameter with mixed alignment, 3D blocks
struct Params
{
    char tag;
    double unused_but_aligned;
    int scale;
    short offset;
};
EXPORT void struct_param(Params p, unsigned char bias, int *out)
{
    int t = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    int g = t + blockIdx.x * blockDim.x * blockDim.y * blockDim.z;
    out[g] = g * p.scale + p.offset + p.tag + bias;
}

// integer and bit manipulation, float math
EXPORT void math_ops(const unsigned *u, const float *f, unsigned *uo, float *fo, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    unsigned x = u[i];
    uo[4 * i + 0] = __builtin_popcount(x);
    uo[4 * i + 1] = x ? __builtin_clz(x) : 32;
    uo[4 * i + 2] = __builtin_bitreverse32(x);
    uo[4 * i + 3] = (x >> 7) | (x << 25);
    float y = f[i];
    fo[4 * i + 0] = sqrtf_(y);
    fo[4 * i + 1] = y / 3.0f;
    fo[4 * i + 2] = __builtin_fminf(y, 2.0f) + __builtin_fmaxf(y, 1.0f);
    fo[4 * i + 3] = __builtin_floorf(y * 1.5f) - (float)(int)(y * 2.25f);
}

// pointer walk over shared memory (integer pointer arithmetic in PTX)
EXPORT void shared_scan(const int *in, int *out)
{
    __shared__ int buf[128];
    int t = threadIdx.x;
    buf[t] = in[blockIdx.x * 128 + t];
    __syncthreads();
    if (t == 0)
    {
        int acc = 0;
        for (int *p = buf; p != buf + 128; ++p)
        {
            acc += *p;
            *p = acc;
        }
    }
    __syncthreads();
    out[blockIdx.x * 128 + t] = buf[t];
}

// writable module-scope device variables (emulated with a module buffer)
// and constant-memory tables
__device__ int g_counter;
__device__ int g_last;
__device__ float g_bias[4] = {1.0f, 2.0f, 3.0f, 4.0f};
__constant__ int c_table[8] = {1, 1, 2, 3, 5, 8, 13, 21};
EXPORT void device_globals(const int *idx, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    atomicAdd(&g_counter, 1);
    out[i] = c_table[idx[i] & 7] * g_bias[i & 3];
    if (i == n - 1)
        g_last = 1234; // visible to the host and to later launches
}
