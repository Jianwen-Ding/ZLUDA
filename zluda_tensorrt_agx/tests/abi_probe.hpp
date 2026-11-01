// Independently declared test-only dispatch prefixes; never include the implementation contract.
#pragma once
#include "NvInferRuntimeBase.h"
#include <cstddef>
namespace probe {
using nvinfer1::Dims;using nvinfer1::DataType;using nvinfer1::TensorIOMode;
struct Engine;struct HostMemory;
class HostDispatch {public:virtual ~HostDispatch() noexcept=default;virtual void* data() const noexcept=0;virtual size_t size() const noexcept=0;virtual DataType type() const noexcept=0;};
struct HostMemory {virtual ~HostMemory() noexcept=default;HostDispatch* impl;};
class EngineDispatch {
public:
    virtual ~EngineDispatch() noexcept=default;
    virtual Engine* getPImpl() noexcept=0;
    virtual int32_t getNbLayers() const noexcept=0;
    virtual HostMemory* serialize() const noexcept=0;
    virtual void* createExecutionContext(int32_t) noexcept=0;
    virtual bool isRefittable() const noexcept=0;
    virtual char const* getName() const noexcept=0;
    virtual int32_t getNbOptimizationProfiles() const noexcept=0;
    virtual int32_t getEngineCapability() const noexcept=0;
    virtual void setErrorRecorder(void*) noexcept=0;
    virtual void* getErrorRecorder() const noexcept=0;
    virtual uint32_t getTacticSources() const noexcept=0;
    virtual int32_t getProfilingVerbosity() const noexcept=0;
    virtual void* createEngineInspector() const noexcept=0;
    virtual Dims getTensorShape(char const*) const noexcept=0;
    virtual DataType getTensorDataType(char const*) const noexcept=0;
    virtual int32_t getTensorLocation(char const*) const noexcept=0;
    virtual bool isShapeInferenceIO(char const*) const noexcept=0;
    virtual TensorIOMode getTensorIOMode(char const*) const noexcept=0;
    virtual int32_t getTensorBytesPerComponent(char const*) const noexcept=0;
    virtual int32_t getTensorComponentsPerElement(char const*) const noexcept=0;
    virtual int32_t getTensorFormat(char const*) const noexcept=0;
    virtual char const* getTensorFormatDesc(char const*) const noexcept=0;
    virtual int32_t getTensorVectorizedDim(char const*) const noexcept=0;
    virtual Dims getProfileShape(char const*,int32_t,int32_t) const noexcept=0;
    virtual int32_t getNbIOTensors() const noexcept=0;
    virtual char const* getIOTensorName(int32_t) const noexcept=0;
    virtual int32_t getHardwareCompatibilityLevel() const noexcept=0;
    virtual int32_t getNbAuxStreams() const noexcept=0;
    virtual int32_t getTensorBytesPerComponentV2(char const*,int32_t) const noexcept=0;
    virtual int32_t getTensorComponentsPerElementV2(char const*,int32_t) const noexcept=0;
    virtual int32_t getTensorFormatV2(char const*,int32_t) const noexcept=0;
    virtual char const* getTensorFormatDescV2(char const*,int32_t) const noexcept=0;
    virtual int32_t getTensorVectorizedDimV2(char const*,int32_t) const noexcept=0;
    virtual void* createSerializationConfig() noexcept=0;
    virtual HostMemory* serializeWithConfig(void*) const noexcept=0;
    virtual void* createRefitter(void*) noexcept=0;
    virtual int64_t getStreamableWeightsSize() const noexcept=0;
    virtual bool isDebugTensor(char const*) const noexcept=0;
    virtual bool setWeightStreamingBudgetV2(int64_t) noexcept=0;
    virtual int64_t getWeightStreamingBudgetV2() const noexcept=0;
    virtual int64_t getWeightStreamingAutomaticBudget() const noexcept=0;
    virtual int64_t getWeightStreamingScratchMemorySize() const noexcept=0;
    virtual int64_t getDeviceMemorySizeV2() const noexcept=0;
    virtual int64_t getDeviceMemorySizeForProfileV2(int32_t) const noexcept=0;
    virtual int64_t const* getProfileTensorValuesV2(char const*,int32_t,int32_t) const noexcept=0;
    virtual void* createExecutionContextWithRuntimeConfig(void*) noexcept=0;
    virtual void* createRuntimeConfig() noexcept=0;
};
struct Engine {virtual ~Engine() noexcept=default;EngineDispatch* impl;};
}

namespace probe {
struct Context;
class ContextDispatch {
public:
    virtual ~ContextDispatch() noexcept=default;
    virtual Context* getPImpl() noexcept=0;
    virtual void setDebugSync(bool) noexcept=0;
    virtual bool getDebugSync() const noexcept=0;
    virtual void setProfiler(void*) noexcept=0;
    virtual void* getProfiler() const noexcept=0;
    virtual Engine const& getEngine() const noexcept=0;
    virtual void setName(char const*) noexcept=0;
    virtual char const* getName() const noexcept=0;
    virtual void setDeviceMemory(void*) noexcept=0;
    virtual int32_t getOptimizationProfile() const noexcept=0;
    virtual bool allInputDimensionsSpecified() const noexcept=0;
    virtual void setErrorRecorder(void*) noexcept=0;
    virtual void* getErrorRecorder() const noexcept=0;
    virtual bool executeV2(void* const*) noexcept=0;
    virtual bool setOptimizationProfileAsync(int32_t,void*) noexcept=0;
    virtual void setEnqueueEmitsProfile(bool) noexcept=0;
    virtual bool getEnqueueEmitsProfile() const noexcept=0;
    virtual bool reportToProfiler() const noexcept=0;
    virtual bool setInputShape(char const*,Dims const&) noexcept=0;
    virtual Dims getTensorShape(char const*) const noexcept=0;
    virtual Dims getTensorStrides(char const*) const noexcept=0;
    virtual bool setTensorAddress(char const*,void*) noexcept=0;
    virtual void const* getTensorAddress(char const*) const noexcept=0;
    virtual bool setInputTensorAddress(char const*,void const*) noexcept=0;
    virtual bool setOutputTensorAddress(char const*,void*) noexcept=0;
    virtual int32_t inferShapes(int32_t,char const**) noexcept=0;
    virtual bool setInputConsumedEvent(void*) noexcept=0;
    virtual void* getInputConsumedEvent() const noexcept=0;
    virtual void* getOutputTensorAddress(char const*) const noexcept=0;
    virtual bool setOutputAllocator(char const*,void*) noexcept=0;
    virtual void* getOutputAllocator(char const*) noexcept=0;
    virtual int64_t getMaxOutputSize(char const*) const noexcept=0;
    virtual bool setTemporaryStorageAllocator(void*) noexcept=0;
    virtual void* getTemporaryStorageAllocator() const noexcept=0;
    virtual bool enqueueV3(void*) noexcept=0;
    virtual void setPersistentCacheLimit(size_t) noexcept=0;
    virtual size_t getPersistentCacheLimit() const noexcept=0;
    virtual bool setNvtxVerbosity(int32_t) noexcept=0;
    virtual int32_t getNvtxVerbosity() const noexcept=0;
    virtual void setAuxStreams(void**,int32_t) noexcept=0;
    virtual bool setDebugListener(void*) noexcept=0;
    virtual void* getDebugListener() noexcept=0;
    virtual bool setTensorDebugState(char const*,bool) noexcept=0;
    virtual bool getDebugState(char const*) const noexcept=0;
    virtual bool setAllTensorsDebugState(bool) noexcept=0;
    virtual size_t updateDeviceMemorySizeForShapes() noexcept=0;
    virtual void setDeviceMemoryV2(void*,int64_t) noexcept=0;
    virtual void* getRuntimeConfig() const noexcept=0;
    virtual bool setUnfusedTensorsDebugState(bool) noexcept=0;
    virtual bool getUnfusedTensorsDebugState() const noexcept=0;
    virtual bool isStreamCapturable(void*) const noexcept=0;
    virtual bool setCommunicator(void*) noexcept=0;
};
struct Context {virtual ~Context() noexcept=default;ContextDispatch* impl;};
}

namespace probe {
struct Config;
class ConfigDispatch {
public:
    virtual ~ConfigDispatch() noexcept=default;
    virtual Config* getPImpl() noexcept=0;
    virtual void setExecutionContextAllocationStrategy(int32_t) noexcept=0;
    virtual int32_t getExecutionContextAllocationStrategy() const noexcept=0;
    virtual void* createRuntimeCache() const noexcept=0;
    virtual bool setRuntimeCache(void const*) noexcept=0;
    virtual void* getRuntimeCache() const noexcept=0;
    virtual void setDynamicShapesKernelSpecializationStrategy(int32_t) noexcept=0;
    virtual int32_t getDynamicShapesKernelSpecializationStrategy() const noexcept=0;
    virtual bool setCudaGraphStrategy(int32_t) noexcept=0;
    virtual int32_t getCudaGraphStrategy() const noexcept=0;
};
struct Config {virtual ~Config() noexcept=default;ConfigDispatch* impl;};
}
