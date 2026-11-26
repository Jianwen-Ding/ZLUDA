// C interface of the metal-cpp backend used by the CUDA driver API layer
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ZmtlDeviceInfo
{
    char name[256];
    uint64_t total_memory;
    uint64_t max_threads_per_threadgroup;
    uint64_t max_threadgroup_memory;
    uint64_t simd_width;
    int unified_memory;
} ZmtlDeviceInfo;

// All functions return 0 on success. On failure `*error` (when non-null) gets
// a malloc'ed message to release with zmtl_free_string.
int zmtl_init(ZmtlDeviceInfo *info, char **error);
void zmtl_free_string(char *s);

// Device memory: Metal buffers with shared storage, addressed by their GPU address
int zmtl_mem_alloc(uint64_t size, uint64_t *gpu_address, char **error);
int zmtl_mem_free(uint64_t gpu_address);
// Host pointer for a device address (inside any live allocation), or null
void *zmtl_host_pointer(uint64_t gpu_address, uint64_t *bytes_left);
// Allocation containing `gpu_address`
int zmtl_mem_range(uint64_t gpu_address, uint64_t *base, uint64_t *size);
// Waits for all submitted work
int zmtl_synchronize(char **error);

typedef struct ZmtlLibrary ZmtlLibrary;
typedef struct ZmtlKernel ZmtlKernel;

int zmtl_library_create(const uint8_t *metallib, size_t len, ZmtlLibrary **lib, char **error);
void zmtl_library_destroy(ZmtlLibrary *lib);
int zmtl_kernel_create(ZmtlLibrary *lib, const char *name, ZmtlKernel **kernel, uint64_t *max_threads,
                       uint64_t *static_shared, char **error);
void zmtl_kernel_destroy(ZmtlKernel *kernel);

// Enqueues one dispatch. `params` is the packed CUDA parameter buffer.
int zmtl_launch(ZmtlKernel *kernel, const uint32_t grid[3], const uint32_t block[3], const void *params,
                size_t params_size, uint32_t dynamic_shared, char **error);

#ifdef __cplusplus
}
#endif
