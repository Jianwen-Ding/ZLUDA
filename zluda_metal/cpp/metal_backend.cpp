// metal-cpp backend for ZLUDA's CUDA driver API implementation.
//
// This file only talks to Metal through Apple's open-source metal-cpp headers.
// On macOS it links against Metal.framework; on Linux it is built against the
// test shim in linux_shim/ (an objc-compatible runtime and a software device),
// so the very same code path is exercised by the Linux tests.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal_backend.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace
{

struct State
{
    std::mutex mutex;
    MTL::Device *device = nullptr;
    MTL::CommandQueue *queue = nullptr;
    // live allocations by GPU address
    std::map<uint64_t, MTL::Buffer *> buffers;
    std::vector<MTL::CommandBuffer *> inFlight;
};

State &state()
{
    static State s;
    return s;
}

int setError(char **error, const std::string &msg)
{
    if (error)
        *error = strdup(msg.c_str());
    return 1;
}

std::string describe(NS::Error *err)
{
    if (!err)
        return "unknown error";
    NS::String *d = err->localizedDescription();
    return d ? d->utf8String() : "unknown error";
}

// Caller holds the mutex
int waitAll(State &s, char **error)
{
    std::string failure;
    for (MTL::CommandBuffer *cb : s.inFlight)
    {
        cb->waitUntilCompleted();
        if (cb->status() == MTL::CommandBufferStatusError && failure.empty())
            failure = "kernel execution failed: " + describe(cb->error());
        cb->release();
    }
    s.inFlight.clear();
    if (!failure.empty())
        return setError(error, failure);
    return 0;
}

} // namespace

struct ZmtlLibrary
{
    MTL::Library *library;
};

struct ZmtlKernel
{
    MTL::ComputePipelineState *pipeline;
};

extern "C" void zmtl_free_string(char *s) { free(s); }

extern "C" int zmtl_init(ZmtlDeviceInfo *info, char **error)
{
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
    if (!s.device)
    {
        s.device = MTL::CreateSystemDefaultDevice();
        if (!s.device)
        {
            pool->release();
            return setError(error, "no Metal device available");
        }
        s.queue = s.device->newCommandQueue();
    }
    if (info)
    {
        memset(info, 0, sizeof(*info));
        NS::String *name = s.device->name();
        strncpy(info->name, name ? name->utf8String() : "Metal device", sizeof(info->name) - 1);
        info->total_memory = s.device->recommendedMaxWorkingSetSize();
        MTL::Size maxThreads = s.device->maxThreadsPerThreadgroup();
        info->max_threads_per_threadgroup = maxThreads.width;
        info->max_threadgroup_memory = s.device->maxThreadgroupMemoryLength();
        info->simd_width = 32;
        info->unified_memory = s.device->hasUnifiedMemory();
    }
    pool->release();
    return 0;
}

extern "C" int zmtl_mem_alloc(uint64_t size, uint64_t *gpu_address, char **error)
{
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    // Zero sized allocations still get a distinct address
    MTL::Buffer *b = s.device->newBuffer(size ? size : 1, MTL::ResourceStorageModeShared);
    if (!b)
        return setError(error, "out of memory");
    uint64_t addr = b->gpuAddress();
    s.buffers[addr] = b;
    *gpu_address = addr;
    return 0;
}

extern "C" int zmtl_mem_free(uint64_t gpu_address)
{
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.buffers.find(gpu_address);
    if (it == s.buffers.end())
        return 1;
    // the buffer may still be used by submitted work
    waitAll(s, nullptr);
    it->second->release();
    s.buffers.erase(it);
    return 0;
}

extern "C" void *zmtl_host_pointer(uint64_t gpu_address, uint64_t *bytes_left)
{
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.buffers.upper_bound(gpu_address);
    if (it == s.buffers.begin())
        return nullptr;
    --it;
    uint64_t base = it->first;
    uint64_t len = it->second->length();
    if (gpu_address >= base + len)
        return nullptr;
    if (bytes_left)
        *bytes_left = base + len - gpu_address;
    return (char *)it->second->contents() + (gpu_address - base);
}

extern "C" int zmtl_mem_range(uint64_t gpu_address, uint64_t *base, uint64_t *size)
{
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.buffers.upper_bound(gpu_address);
    if (it == s.buffers.begin())
        return 1;
    --it;
    uint64_t len = it->second->length();
    if (gpu_address >= it->first + len)
        return 1;
    *base = it->first;
    *size = len;
    return 0;
}

extern "C" int zmtl_synchronize(char **error)
{
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    return waitAll(s, error);
}

extern "C" int zmtl_library_create(const uint8_t *metallib, size_t len, ZmtlLibrary **lib, char **error)
{
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
    dispatch_data_t data = dispatch_data_create(metallib, len, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NS::Error *err = nullptr;
    MTL::Library *library = s.device->newLibrary(data, &err);
    dispatch_release(data);
    if (!library)
    {
        int r = setError(error, "newLibraryWithData failed: " + describe(err));
        pool->release();
        return r;
    }
    *lib = new ZmtlLibrary{library};
    pool->release();
    return 0;
}

extern "C" void zmtl_library_destroy(ZmtlLibrary *lib)
{
    if (!lib)
        return;
    lib->library->release();
    delete lib;
}

extern "C" int zmtl_kernel_create(ZmtlLibrary *lib, const char *name, ZmtlKernel **kernel, uint64_t *max_threads,
                                  uint64_t *static_shared, char **error)
{
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
    MTL::Function *fn = lib->library->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
    if (!fn)
    {
        pool->release();
        return setError(error, std::string("function not found in metallib: ") + name);
    }
    NS::Error *err = nullptr;
    MTL::ComputePipelineState *pso = s.device->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso)
    {
        int r = setError(error, "newComputePipelineStateWithFunction failed: " + describe(err));
        pool->release();
        return r;
    }
    if (max_threads)
        *max_threads = pso->maxTotalThreadsPerThreadgroup();
    if (static_shared)
        *static_shared = pso->staticThreadgroupMemoryLength();
    *kernel = new ZmtlKernel{pso};
    pool->release();
    return 0;
}

extern "C" void zmtl_kernel_destroy(ZmtlKernel *kernel)
{
    if (!kernel)
        return;
    kernel->pipeline->release();
    delete kernel;
}

extern "C" int zmtl_launch(ZmtlKernel *kernel, const uint32_t grid[3], const uint32_t block[3], const void *params,
                           size_t params_size, uint32_t dynamic_shared, char **error)
{
    State &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
    MTL::CommandBuffer *cb = s.queue->commandBuffer();
    MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(kernel->pipeline);
    // CUDA kernel parameters: one constant buffer at index 0
    static const uint64_t zero[4] = {};
    MTL::Buffer *paramBuffer = nullptr;
    if (params_size <= 4096)
        enc->setBytes(params_size ? params : zero, params_size ? params_size : sizeof(zero), 0);
    else
    {
        paramBuffer = s.device->newBuffer(params, params_size, MTL::ResourceStorageModeShared);
        enc->setBuffer(paramBuffer, 0, 0);
    }
    if (dynamic_shared)
        enc->setThreadgroupMemoryLength((dynamic_shared + 15) & ~15u, 0);
    // Kernels reach memory through raw GPU addresses: make every allocation resident
    std::vector<const MTL::Resource *> resident;
    resident.reserve(s.buffers.size());
    for (auto &kv : s.buffers)
        resident.push_back(kv.second);
    if (!resident.empty())
        enc->useResources(resident.data(), resident.size(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    enc->dispatchThreadgroups(MTL::Size(grid[0], grid[1], grid[2]), MTL::Size(block[0], block[1], block[2]));
    enc->endEncoding();
    cb->commit();
    cb->retain();
    s.inFlight.push_back(cb);
    if (paramBuffer)
    {
        // keep it alive until the work is done
        int r = waitAll(s, error);
        paramBuffer->release();
        pool->release();
        return r;
    }
    pool->release();
    return 0;
}
