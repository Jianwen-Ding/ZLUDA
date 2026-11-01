#pragma once
#include "NvInferRuntimeBase.h"
#include <cstddef>
#include <cstdint>
namespace probe {
class Runtime;
class Registry {
protected:virtual ~Registry() noexcept=default;
public:
    virtual void setErrorRecorder(void*) noexcept=0;
    virtual void* getErrorRecorder() const noexcept=0;
    virtual bool isParentSearchEnabled() const=0;
    virtual void setParentSearchEnabled(bool)=0;
    virtual void* loadLibrary(char const*) noexcept=0;
    virtual void deregisterLibrary(void*) noexcept=0;
    virtual bool registerCreator(void*,char const*) noexcept=0;
    virtual void* const* getAllCreators(int32_t*) const noexcept=0;
};
class Dispatch {
public:
    virtual ~Dispatch() noexcept=default;
    virtual Runtime* getPImpl() noexcept=0;
    virtual void* deserializeCudaEngine(void const*,size_t) noexcept=0;
    virtual void setDLACore(int32_t) noexcept=0;
    virtual int32_t getDLACore() const noexcept=0;
    virtual int32_t getNbDLACores() const noexcept=0;
    virtual void setGpuAllocator(void*) noexcept=0;
    virtual void setErrorRecorder(void*) noexcept=0;
    virtual void* getErrorRecorder() const noexcept=0;
    virtual nvinfer1::ILogger* getLogger() const noexcept=0;
    virtual bool setMaxThreads(int32_t) noexcept=0;
    virtual int32_t getMaxThreads() const noexcept=0;
    virtual void setTemporaryDirectory(char const*) noexcept=0;
    virtual char const* getTemporaryDirectory() const noexcept=0;
    virtual void setTempfileControlFlags(uint32_t) noexcept=0;
    virtual uint32_t getTempfileControlFlags() const noexcept=0;
    virtual Registry& getPluginRegistry() noexcept=0;
    virtual void setPluginRegistryParent(void*) noexcept=0;
    virtual Runtime* loadRuntime(char const*) noexcept=0;
    virtual void setEngineHostCodeAllowed(bool) noexcept=0;
    virtual bool getEngineHostCodeAllowed() const noexcept=0;
    virtual void* deserializeCudaEngineV2(void*) noexcept=0;
    virtual int64_t getEngineHeaderSize() const noexcept=0;
    virtual int32_t getEngineValidity(void const*,int64_t,uint64_t*) const noexcept=0;
    virtual void setDeferredWeightsLoading(bool) noexcept=0;
    virtual bool getDeferredWeightsLoading() const noexcept=0;
    virtual bool setDLAWorkspaceAllocationStrategy(int32_t) noexcept=0;
    virtual int32_t getDLAWorkspaceAllocationStrategy() const noexcept=0;
};
struct Runtime {virtual ~Runtime() noexcept=default;Dispatch* impl;};
}
