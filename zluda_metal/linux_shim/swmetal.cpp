// Linux shim: a software MTLDevice, reachable through the unmodified
// metal-cpp headers (MTL::CreateSystemDefaultDevice() & co.).
//
// It implements the compute subset of the Metal API that ZLUDA's backend
// uses: shared-storage buffers with GPU addresses, libraries created from
// .metallib data, compute pipelines, command queues/buffers and compute
// encoders. Kernels run on the CPU through sw_exec.cpp.
#include "shim.h"
#include "sw_exec.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace zshim
{
namespace
{

struct Size3
{
    uint64_t width, height, depth;
};
struct Range
{
    uint64_t location, length;
};

// MTLResourceOptions / MTLStorageMode
constexpr uint64_t STORAGE_MODE_SHIFT = 4;
constexpr uint64_t STORAGE_MODE_PRIVATE = 2;
// MTLCommandBufferStatus
constexpr uint64_t CB_STATUS_NOT_ENQUEUED = 0;
constexpr uint64_t CB_STATUS_COMMITTED = 2;
constexpr uint64_t CB_STATUS_COMPLETED = 4;
constexpr uint64_t CB_STATUS_ERROR = 5;
// MTLFunctionType
constexpr uint64_t FUNCTION_TYPE_KERNEL = 3;

struct DeviceObj : Object
{
};

struct BufferObj : Object
{
    void *memory = nullptr;
    uint64_t length = 0;
    uint64_t options = 0;
    Object *label = nullptr;
};

struct LibraryObj : Object
{
    std::vector<zsw::LibraryFunction> functions;
};

struct FunctionObj : Object
{
    std::shared_ptr<zsw::LibraryFunction> fn;
};

struct PipelineObj : Object
{
    std::shared_ptr<zsw::CompiledKernel> kernel;
};

struct QueueObj : Object
{
    std::mutex executionMutex;
};

struct Dispatch
{
    std::shared_ptr<zsw::CompiledKernel> kernel;
    zsw::DispatchArgs args;
    std::vector<std::vector<uint8_t>> inlineBytes; // keeps setBytes data alive
};

struct CommandBufferObj : Object
{
    QueueObj *queue = nullptr;
    std::vector<std::shared_ptr<Dispatch>> dispatches;
    uint64_t status = CB_STATUS_NOT_ENQUEUED;
    Object *error = nullptr;
};

struct EncoderObj : Object
{
    CommandBufferObj *commandBuffer = nullptr;
    PipelineObj *pipeline = nullptr;
    zsw::DispatchArgs args;
    std::vector<std::shared_ptr<std::vector<uint8_t>>> bytes;
    void *bytesFor[zsw::MAX_BUFFERS] = {};
};

ShimClass *g_device, *g_buffer, *g_library, *g_function, *g_pipeline, *g_queue, *g_cmdbuf, *g_encoder;
DeviceObj *g_theDevice;

const char *DEVICE_NAME = "ZLUDA Software Metal Device (AIR JIT, Linux test shim)";

Object *setError(Object **slot, const std::string &msg, long code = 1)
{
    if (slot)
        *slot = autorelease(newError("MTLLibraryErrorDomain", code, msg));
    return nullptr;
}

// --- MTLDevice ---
id devName(id, SEL) { return (id)autorelease(newString(DEVICE_NAME)); }
id devNewCommandQueue(id, SEL) { return (id)make<QueueObj>(g_queue); }
id devNewCommandQueueMax(id self, SEL s, uint64_t) { return devNewCommandQueue(self, s); }
id devNewBuffer(id, SEL, uint64_t length, uint64_t options)
{
    if (((options >> STORAGE_MODE_SHIFT) & 0xF) == STORAGE_MODE_PRIVATE)
    {
        // private storage is fine on the software device: it is host memory too
    }
    BufferObj *b = make<BufferObj>(g_buffer);
    b->length = length;
    b->options = options;
    size_t alloc = length ? (length + 255) & ~size_t(255) : 256;
    b->memory = aligned_alloc(256, alloc);
    if (!b->memory)
    {
        delete b;
        return nullptr;
    }
    memset(b->memory, 0, alloc);
    b->destroy = [](Object *p) {
        BufferObj *buf = (BufferObj *)p;
        free(buf->memory);
        release(buf->label);
        delete buf;
    };
    return (id)b;
}
id devNewBufferWithBytes(id self, SEL s, const void *bytes, uint64_t length, uint64_t options)
{
    id b = devNewBuffer(self, s, length, options);
    if (b && length)
        memcpy(((BufferObj *)b)->memory, bytes, length);
    return b;
}
id devNewLibraryWithData(id, SEL, dispatch_data_t data, Object **error)
{
    LibraryObj *lib = make<LibraryObj>(g_library);
    std::string err;
    if (!zsw::parseLibrary(zluda_dispatch_data_bytes(data), dispatch_data_get_size(data), lib->functions, err))
    {
        delete lib;
        return (id)setError(error, "invalid metallib: " + err);
    }
    return (id)lib;
}
id devNewComputePipeline(id, SEL, id function, Object **error)
{
    FunctionObj *f = (FunctionObj *)function;
    if (!f)
        return (id)setError(error, "no function");
    std::string err;
    auto kernel = zsw::compileKernel(*f->fn, err);
    if (!kernel)
        return (id)setError(error, "failed to compile " + f->fn->name + ": " + err, 3);
    PipelineObj *p = make<PipelineObj>(g_pipeline);
    p->kernel = kernel;
    return (id)p;
}
Size3 devMaxThreadsPerThreadgroup(id, SEL) { return {1024, 1024, 1024}; }
uint64_t devMaxThreadgroupMemoryLength(id, SEL) { return 32768; }
uint64_t devRecommendedMaxWorkingSetSize(id, SEL) { return uint64_t(8) << 30; }
uint64_t devMaxBufferLength(id, SEL) { return uint64_t(4) << 30; }
bool devHasUnifiedMemory(id, SEL) { return true; }
uint64_t devRegistryID(id, SEL) { return 0x5A4C554441ull; }
bool devSupportsFamily(id, SEL, intptr_t family)
{
    // MTLGPUFamilyApple1..9 (1001..1009), Common1-3 (3001-3003), Metal3 (5001)
    return (family >= 1001 && family <= 1009) || (family >= 3001 && family <= 3003) || family == 5001;
}
uint64_t devCurrentAllocatedSize(id, SEL) { return 0; }
bool devLowPower(id, SEL) { return false; }

// --- MTLBuffer ---
void *bufContents(id self, SEL) { return ((BufferObj *)self)->memory; }
uint64_t bufLength(id self, SEL) { return ((BufferObj *)self)->length; }
uint64_t bufGpuAddress(id self, SEL) { return (uint64_t)(uintptr_t)((BufferObj *)self)->memory; }
void bufDidModifyRange(id, SEL, Range) {}
void bufSetLabel(id self, SEL, id label)
{
    retain((Object *)label);
    release(((BufferObj *)self)->label);
    ((BufferObj *)self)->label = (Object *)label;
}
id bufLabel(id self, SEL) { return (id)((BufferObj *)self)->label; }
uint64_t bufStorageMode(id self, SEL) { return (((BufferObj *)self)->options >> STORAGE_MODE_SHIFT) & 0xF; }
uint64_t bufResourceOptions(id self, SEL) { return ((BufferObj *)self)->options; }
id bufDevice(id, SEL) { return (id)g_theDevice; }

// --- MTLLibrary / MTLFunction ---
id libNewFunction(id self, SEL, id name)
{
    LibraryObj *lib = (LibraryObj *)self;
    const char *n = stringUTF8((Object *)name);
    for (auto &f : lib->functions)
        if (f.name == n)
        {
            FunctionObj *fo = make<FunctionObj>(g_function);
            fo->fn = std::make_shared<zsw::LibraryFunction>(f);
            return (id)fo;
        }
    return nullptr;
}
id libFunctionNames(id self, SEL)
{
    LibraryObj *lib = (LibraryObj *)self;
    std::vector<Object *> names;
    for (auto &f : lib->functions)
        names.push_back(newString(f.name));
    Object *arr = newArray(names.data(), names.size());
    for (Object *o : names)
        release(o);
    return (id)autorelease(arr);
}
void anySetLabel(id, SEL, id) {}
id anyLabel(id, SEL) { return nullptr; }
id anyDevice(id, SEL) { return (id)g_theDevice; }
id fnName(id self, SEL) { return (id)autorelease(newString(((FunctionObj *)self)->fn->name)); }
uint64_t fnType(id, SEL) { return FUNCTION_TYPE_KERNEL; }

// --- MTLComputePipelineState ---
uint64_t psoMaxThreads(id, SEL) { return 1024; }
uint64_t psoExecutionWidth(id, SEL) { return 32; }
uint64_t psoStaticTgMemory(id self, SEL) { return zsw::staticThreadgroupMemory(*((PipelineObj *)self)->kernel); }

// --- MTLCommandQueue / MTLCommandBuffer ---
id queueCommandBuffer(id self, SEL)
{
    CommandBufferObj *cb = make<CommandBufferObj>(g_cmdbuf);
    cb->queue = (QueueObj *)self;
    retain(cb->queue);
    cb->destroy = [](Object *p) {
        CommandBufferObj *c = (CommandBufferObj *)p;
        release(c->queue);
        release(c->error);
        delete c;
    };
    return (id)autorelease(cb);
}
id cbComputeEncoder(id self, SEL)
{
    EncoderObj *e = make<EncoderObj>(g_encoder);
    e->commandBuffer = (CommandBufferObj *)self;
    return (id)autorelease(e);
}
id cbComputeEncoderWithType(id self, SEL s, uint64_t) { return cbComputeEncoder(self, s); }
void cbCommit(id self, SEL)
{
    // The software device executes the work at commit time, in order, which
    // matches the in-order semantics of a single MTLCommandQueue.
    CommandBufferObj *cb = (CommandBufferObj *)self;
    std::lock_guard<std::mutex> lock(cb->queue->executionMutex);
    cb->status = CB_STATUS_COMMITTED;
    for (auto &d : cb->dispatches)
    {
        std::string err;
        if (!zsw::dispatch(*d->kernel, d->args, err))
        {
            cb->error = newError("MTLCommandBufferErrorDomain", 1, err);
            cb->status = CB_STATUS_ERROR;
            return;
        }
    }
    cb->dispatches.clear();
    cb->status = CB_STATUS_COMPLETED;
}
void cbWait(id, SEL) {}
uint64_t cbStatus(id self, SEL) { return ((CommandBufferObj *)self)->status; }
id cbError(id self, SEL) { return (id)((CommandBufferObj *)self)->error; }

// --- MTLComputeCommandEncoder ---
void encSetPipeline(id self, SEL, id pso) { ((EncoderObj *)self)->pipeline = (PipelineObj *)pso; }
void encSetBuffer(id self, SEL, id buffer, uint64_t offset, uint64_t index)
{
    if (index >= zsw::MAX_BUFFERS)
        return;
    EncoderObj *e = (EncoderObj *)self;
    e->args.buffers[index] = buffer ? (uint8_t *)((BufferObj *)buffer)->memory + offset : nullptr;
}
void encSetBytes(id self, SEL, const void *bytes, uint64_t length, uint64_t index)
{
    if (index >= zsw::MAX_BUFFERS)
        return;
    EncoderObj *e = (EncoderObj *)self;
    auto copy = std::make_shared<std::vector<uint8_t>>((const uint8_t *)bytes, (const uint8_t *)bytes + length);
    copy->resize(std::max<size_t>(length, 16));
    e->bytes.push_back(copy);
    e->args.buffers[index] = copy->data();
}
void encSetTgMemory(id self, SEL, uint64_t length, uint64_t index)
{
    if (index < zsw::MAX_BUFFERS)
        ((EncoderObj *)self)->args.threadgroupLength[index] = length;
}
void encUseResource(id, SEL, id, uint64_t) {}
void encUseResources(id, SEL, const id *, uint64_t, uint64_t) {}
void encDispatchThreadgroups(id self, SEL, Size3 groups, Size3 threads)
{
    EncoderObj *e = (EncoderObj *)self;
    if (!e->pipeline)
    {
        fprintf(stderr, "[zluda metal shim] dispatch without a compute pipeline state\n");
        abort();
    }
    auto d = std::make_shared<Dispatch>();
    d->kernel = e->pipeline->kernel;
    d->args = e->args;
    d->args.threadgroups[0] = uint32_t(groups.width);
    d->args.threadgroups[1] = uint32_t(groups.height);
    d->args.threadgroups[2] = uint32_t(groups.depth);
    d->args.threadsPerThreadgroup[0] = uint32_t(threads.width);
    d->args.threadsPerThreadgroup[1] = uint32_t(threads.height);
    d->args.threadsPerThreadgroup[2] = uint32_t(threads.depth);
    for (auto &b : e->bytes)
        d->inlineBytes.push_back(*b);
    // re-point inline byte bindings to the dispatch's own copies
    size_t k = 0;
    for (unsigned i = 0; i < zsw::MAX_BUFFERS; i++)
        for (size_t j = 0; j < e->bytes.size(); j++)
            if (e->args.buffers[i] == e->bytes[j]->data())
            {
                d->args.buffers[i] = d->inlineBytes[j].data();
                k++;
            }
    (void)k;
    e->commandBuffer->dispatches.push_back(d);
}
void encEndEncoding(id, SEL) {}
void encMemoryBarrier(id, SEL, uint64_t) {}

void registerAll()
{
    ensureFoundation();
    ShimClass *root = rootClass();
    g_device = defineClass("ZludaSWDevice", root);
    addMethod(g_device, "name", (void *)devName);
    addMethod(g_device, "newCommandQueue", (void *)devNewCommandQueue);
    addMethod(g_device, "newCommandQueueWithMaxCommandBufferCount:", (void *)devNewCommandQueueMax);
    addMethod(g_device, "newBufferWithLength:options:", (void *)devNewBuffer);
    addMethod(g_device, "newBufferWithBytes:length:options:", (void *)devNewBufferWithBytes);
    addMethod(g_device, "newLibraryWithData:error:", (void *)devNewLibraryWithData);
    addMethod(g_device, "newComputePipelineStateWithFunction:error:", (void *)devNewComputePipeline);
    addMethod(g_device, "maxThreadsPerThreadgroup", (void *)devMaxThreadsPerThreadgroup);
    addMethod(g_device, "maxThreadgroupMemoryLength", (void *)devMaxThreadgroupMemoryLength);
    addMethod(g_device, "recommendedMaxWorkingSetSize", (void *)devRecommendedMaxWorkingSetSize);
    addMethod(g_device, "maxBufferLength", (void *)devMaxBufferLength);
    addMethod(g_device, "hasUnifiedMemory", (void *)devHasUnifiedMemory);
    addMethod(g_device, "registryID", (void *)devRegistryID);
    addMethod(g_device, "supportsFamily:", (void *)devSupportsFamily);
    addMethod(g_device, "currentAllocatedSize", (void *)devCurrentAllocatedSize);
    addMethod(g_device, "isLowPower", (void *)devLowPower);

    g_buffer = defineClass("ZludaSWBuffer", root);
    addMethod(g_buffer, "contents", (void *)bufContents);
    addMethod(g_buffer, "length", (void *)bufLength);
    addMethod(g_buffer, "allocatedSize", (void *)bufLength);
    addMethod(g_buffer, "gpuAddress", (void *)bufGpuAddress);
    addMethod(g_buffer, "didModifyRange:", (void *)bufDidModifyRange);
    addMethod(g_buffer, "setLabel:", (void *)bufSetLabel);
    addMethod(g_buffer, "label", (void *)bufLabel);
    addMethod(g_buffer, "storageMode", (void *)bufStorageMode);
    addMethod(g_buffer, "resourceOptions", (void *)bufResourceOptions);
    addMethod(g_buffer, "device", (void *)bufDevice);

    g_library = defineClass("ZludaSWLibrary", root);
    addMethod(g_library, "newFunctionWithName:", (void *)libNewFunction);
    addMethod(g_library, "functionNames", (void *)libFunctionNames);
    addMethod(g_library, "setLabel:", (void *)anySetLabel);
    addMethod(g_library, "label", (void *)anyLabel);
    addMethod(g_library, "device", (void *)anyDevice);

    g_function = defineClass("ZludaSWFunction", root);
    addMethod(g_function, "name", (void *)fnName);
    addMethod(g_function, "functionType", (void *)fnType);
    addMethod(g_function, "device", (void *)anyDevice);
    addMethod(g_function, "setLabel:", (void *)anySetLabel);
    addMethod(g_function, "label", (void *)anyLabel);

    g_pipeline = defineClass("ZludaSWComputePipelineState", root);
    addMethod(g_pipeline, "maxTotalThreadsPerThreadgroup", (void *)psoMaxThreads);
    addMethod(g_pipeline, "threadExecutionWidth", (void *)psoExecutionWidth);
    addMethod(g_pipeline, "staticThreadgroupMemoryLength", (void *)psoStaticTgMemory);
    addMethod(g_pipeline, "device", (void *)anyDevice);
    addMethod(g_pipeline, "label", (void *)anyLabel);

    g_queue = defineClass("ZludaSWCommandQueue", root);
    addMethod(g_queue, "commandBuffer", (void *)queueCommandBuffer);
    addMethod(g_queue, "commandBufferWithUnretainedReferences", (void *)queueCommandBuffer);
    addMethod(g_queue, "setLabel:", (void *)anySetLabel);
    addMethod(g_queue, "label", (void *)anyLabel);
    addMethod(g_queue, "device", (void *)anyDevice);

    g_cmdbuf = defineClass("ZludaSWCommandBuffer", root);
    addMethod(g_cmdbuf, "computeCommandEncoder", (void *)cbComputeEncoder);
    addMethod(g_cmdbuf, "computeCommandEncoderWithDispatchType:", (void *)cbComputeEncoderWithType);
    addMethod(g_cmdbuf, "commit", (void *)cbCommit);
    addMethod(g_cmdbuf, "enqueue", (void *)cbWait);
    addMethod(g_cmdbuf, "waitUntilCompleted", (void *)cbWait);
    addMethod(g_cmdbuf, "waitUntilScheduled", (void *)cbWait);
    addMethod(g_cmdbuf, "status", (void *)cbStatus);
    addMethod(g_cmdbuf, "error", (void *)cbError);
    addMethod(g_cmdbuf, "setLabel:", (void *)anySetLabel);
    addMethod(g_cmdbuf, "label", (void *)anyLabel);
    addMethod(g_cmdbuf, "device", (void *)anyDevice);

    g_encoder = defineClass("ZludaSWComputeCommandEncoder", root);
    addMethod(g_encoder, "setComputePipelineState:", (void *)encSetPipeline);
    addMethod(g_encoder, "setBuffer:offset:atIndex:", (void *)encSetBuffer);
    addMethod(g_encoder, "setBytes:length:atIndex:", (void *)encSetBytes);
    addMethod(g_encoder, "setThreadgroupMemoryLength:atIndex:", (void *)encSetTgMemory);
    addMethod(g_encoder, "useResource:usage:", (void *)encUseResource);
    addMethod(g_encoder, "useResources:count:usage:", (void *)encUseResources);
    addMethod(g_encoder, "dispatchThreadgroups:threadsPerThreadgroup:", (void *)encDispatchThreadgroups);
    addMethod(g_encoder, "endEncoding", (void *)encEndEncoding);
    addMethod(g_encoder, "memoryBarrierWithScope:", (void *)encMemoryBarrier);
    addMethod(g_encoder, "setLabel:", (void *)anySetLabel);
    addMethod(g_encoder, "label", (void *)anyLabel);
    addMethod(g_encoder, "device", (void *)anyDevice);

    g_theDevice = make<DeviceObj>(g_device);
}

} // namespace

void ensureMetal()
{
    static std::once_flag once;
    std::call_once(once, registerAll);
}

} // namespace zshim

using namespace zshim;

extern "C"
{
    // MTLCreateSystemDefaultDevice follows the "Create" rule: the caller owns a reference
    void *MTLCreateSystemDefaultDevice()
    {
        ensureMetal();
        retain(g_theDevice);
        return g_theDevice;
    }
    void *MTLCopyAllDevices()
    {
        ensureMetal();
        Object *d = g_theDevice;
        return newArray(&d, 1);
    }
    void *MTLCopyAllDevicesWithObserver(void **observer, void *)
    {
        if (observer)
            *observer = nullptr;
        return MTLCopyAllDevices();
    }
    void MTLRemoveDeviceObserver(void *) {}
}
