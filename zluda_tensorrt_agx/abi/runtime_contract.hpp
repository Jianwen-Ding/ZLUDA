// SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// Minimal public runtime dispatch declarations adapted from Apache-2.0 RTX API docs.
// Prototype only; see README.md. No proprietary runtime code is included.
#pragma once
#include "NvInferRuntimeBase.h"
#include <cstddef>
struct CUstream_st;
namespace nvinfer1 {
class IRuntime; class ICudaEngine;
namespace v_1_0 { class IGpuAllocator; class IErrorRecorder; class IStreamReaderV2; class IPluginCreatorInterface; class IPluginResource; }
using IGpuAllocator=v_1_0::IGpuAllocator;
using IErrorRecorder=v_1_0::IErrorRecorder;
using IStreamReaderV2=v_1_0::IStreamReaderV2;
using IPluginCreatorInterface=v_1_0::IPluginCreatorInterface;
using IPluginResource=v_1_0::IPluginResource;
enum class SeekPosition:int32_t { kSET=0,kCUR=1,kEND=2 };
namespace v_1_0 {
class IStreamReaderV2:public IVersionedInterface {
public:
    ~IStreamReaderV2() override=default;
    IStreamReaderV2()=default;
    InterfaceInfo getInterfaceInfo() const noexcept override{return {"IStreamReaderV2",1,0};}
    virtual int64_t read(void*,int64_t,CUstream_st*) noexcept=0;
    virtual bool seek(int64_t,SeekPosition) noexcept=0;
protected:
    IStreamReaderV2(IStreamReaderV2 const&)=default;
    IStreamReaderV2(IStreamReaderV2&&)=default;
    IStreamReaderV2& operator=(IStreamReaderV2 const&) & =default;
    IStreamReaderV2& operator=(IStreamReaderV2&&) & =default;
};
}
enum class EngineValidity:int32_t { kVALID=0,kSUBOPTIMAL=1,kINVALID=2 };
enum class EngineInvalidityDiagnostics:uint64_t { kMALFORMED_ENGINE=1ULL<<5 };
enum class DLAWorkspaceAllocationStrategy:int32_t;
using TempfileControlFlags=uint32_t;
class IPluginRegistry {
protected: virtual ~IPluginRegistry() noexcept=default;
public:
    using PluginLibraryHandle=void*;
    virtual void setErrorRecorder(IErrorRecorder* const) noexcept=0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept=0;
    virtual bool isParentSearchEnabled() const=0;
    virtual void setParentSearchEnabled(bool const)=0;
    virtual PluginLibraryHandle loadLibrary(AsciiChar const*) noexcept=0;
    virtual void deregisterLibrary(PluginLibraryHandle) noexcept=0;
    virtual bool registerCreator(IPluginCreatorInterface&,AsciiChar const* const) noexcept=0;
    virtual IPluginCreatorInterface* const* getAllCreators(int32_t* const) const noexcept=0;
    virtual IPluginCreatorInterface* getCreator(AsciiChar const* const,AsciiChar const* const,AsciiChar const* const) noexcept=0;
    virtual bool deregisterCreator(IPluginCreatorInterface const&) noexcept=0;
    virtual IPluginResource* acquirePluginResource(AsciiChar const*,IPluginResource*) noexcept=0;
    virtual int32_t releasePluginResource(AsciiChar const*) noexcept=0;
    virtual IPluginCreatorInterface* const* getAllCreatorsRecursive(int32_t* const) noexcept=0;
};
namespace apiv {
class VRoot { public: virtual ~VRoot() noexcept=default; };
class VRuntime:public VRoot {
public:
    virtual IRuntime* getPImpl() noexcept=0;
    virtual ICudaEngine* deserializeCudaEngine(void const*,std::size_t) noexcept=0;
    virtual void setDLACore(int32_t) noexcept=0;
    virtual int32_t getDLACore() const noexcept=0;
    virtual int32_t getNbDLACores() const noexcept=0;
    virtual void setGpuAllocator(IGpuAllocator*) noexcept=0;
    virtual void setErrorRecorder(IErrorRecorder*) noexcept=0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept=0;
    virtual ILogger* getLogger() const noexcept=0;
    virtual bool setMaxThreads(int32_t) noexcept=0;
    virtual int32_t getMaxThreads() const noexcept=0;
    virtual void setTemporaryDirectory(char const*) noexcept=0;
    virtual char const* getTemporaryDirectory() const noexcept=0;
    virtual void setTempfileControlFlags(TempfileControlFlags) noexcept=0;
    virtual TempfileControlFlags getTempfileControlFlags() const noexcept=0;
    virtual IPluginRegistry& getPluginRegistry() noexcept=0;
    virtual void setPluginRegistryParent(IPluginRegistry*) noexcept=0;
    virtual IRuntime* loadRuntime(char const*) noexcept=0;
    virtual void setEngineHostCodeAllowed(bool) noexcept=0;
    virtual bool getEngineHostCodeAllowed() const noexcept=0;
    virtual ICudaEngine* deserializeCudaEngineV2(IStreamReaderV2&) noexcept=0;
    virtual int64_t getEngineHeaderSize() const noexcept=0;
    virtual EngineValidity getEngineValidity(void const*,int64_t,uint64_t*) const noexcept=0;
    virtual void setDeferredWeightsLoading(bool) noexcept=0;
    virtual bool getDeferredWeightsLoading() const noexcept=0;
    virtual bool setDLAWorkspaceAllocationStrategy(DLAWorkspaceAllocationStrategy) noexcept=0;
    virtual DLAWorkspaceAllocationStrategy getDLAWorkspaceAllocationStrategy() const noexcept=0;
};
}
// Prefix shape of the public wrapper: one virtual destructor and one implementation pointer.
class IRuntime {
public: virtual ~IRuntime() noexcept=default;
protected: apiv::VRuntime* mImpl=nullptr;
};
}
// Public engine/host-memory dispatch declarations; see the same Apache-2.0 provenance above.
namespace nvinfer1 {
class IHostMemory;class IExecutionContext;class IEngineInspector;class ISerializationConfig;class IRefitter;class IRuntimeConfig;class IWeightsManager;
enum class ExecutionContextAllocationStrategy:int32_t;
enum class EngineCapability:int32_t;
enum class ProfilingVerbosity:int32_t;
enum class TensorLocation:int32_t;
enum class TensorFormat:int32_t;
enum class OptProfileSelector:int32_t;
enum class HardwareCompatibilityLevel:int32_t;
enum class EngineStat:int32_t;
namespace apiv {
class VHostMemory:public VRoot {
public:
    virtual void* data() const noexcept=0;
    virtual std::size_t size() const noexcept=0;
    virtual DataType type() const noexcept=0;
};
class VCudaEngine:public VRoot {
public:
    virtual ICudaEngine* getPImpl() noexcept=0;
    virtual int32_t getNbLayers() const noexcept=0;
    virtual IHostMemory* serialize() const noexcept=0;
    virtual IExecutionContext* createExecutionContext(ExecutionContextAllocationStrategy) noexcept=0;
    virtual bool isRefittable() const noexcept=0;
    virtual char const* getName() const noexcept=0;
    virtual int32_t getNbOptimizationProfiles() const noexcept=0;
    virtual EngineCapability getEngineCapability() const noexcept=0;
    virtual void setErrorRecorder(IErrorRecorder*) noexcept=0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept=0;
    virtual uint32_t getTacticSources() const noexcept=0;
    virtual ProfilingVerbosity getProfilingVerbosity() const noexcept=0;
    virtual IEngineInspector* createEngineInspector() const noexcept=0;
    virtual Dims getTensorShape(char const*) const noexcept=0;
    virtual DataType getTensorDataType(char const*) const noexcept=0;
    virtual TensorLocation getTensorLocation(char const*) const noexcept=0;
    virtual bool isShapeInferenceIO(char const*) const noexcept=0;
    virtual TensorIOMode getTensorIOMode(char const*) const noexcept=0;
    virtual int32_t getTensorBytesPerComponent(char const*) const noexcept=0;
    virtual int32_t getTensorComponentsPerElement(char const*) const noexcept=0;
    virtual TensorFormat getTensorFormat(char const*) const noexcept=0;
    virtual char const* getTensorFormatDesc(char const*) const noexcept=0;
    virtual int32_t getTensorVectorizedDim(char const*) const noexcept=0;
    virtual Dims getProfileShape(char const*,int32_t,OptProfileSelector) const noexcept=0;
    virtual int32_t getNbIOTensors() const noexcept=0;
    virtual char const* getIOTensorName(int32_t) const noexcept=0;
    virtual HardwareCompatibilityLevel getHardwareCompatibilityLevel() const noexcept=0;
    virtual int32_t getNbAuxStreams() const noexcept=0;
    virtual int32_t getTensorBytesPerComponentV2(char const*,int32_t) const noexcept=0;
    virtual int32_t getTensorComponentsPerElementV2(char const*,int32_t) const noexcept=0;
    virtual TensorFormat getTensorFormatV2(char const*,int32_t) const noexcept=0;
    virtual char const* getTensorFormatDescV2(char const*,int32_t) const noexcept=0;
    virtual int32_t getTensorVectorizedDimV2(char const*,int32_t) const noexcept=0;
    virtual ISerializationConfig* createSerializationConfig() noexcept=0;
    virtual IHostMemory* serializeWithConfig(ISerializationConfig&) const noexcept=0;
    virtual IRefitter* createRefitter(ILogger&) noexcept=0;
    virtual int64_t getStreamableWeightsSize() const noexcept=0;
    virtual bool isDebugTensor(char const*) const noexcept=0;
    virtual bool setWeightStreamingBudgetV2(int64_t) noexcept=0;
    virtual int64_t getWeightStreamingBudgetV2() const noexcept=0;
    virtual int64_t getWeightStreamingAutomaticBudget() const noexcept=0;
    virtual int64_t getWeightStreamingScratchMemorySize() const noexcept=0;
    virtual int64_t getDeviceMemorySizeV2() const noexcept=0;
    virtual int64_t getDeviceMemorySizeForProfileV2(int32_t) const noexcept=0;
    virtual int64_t const* getProfileTensorValuesV2(char const*,int32_t,OptProfileSelector) const noexcept=0;
    virtual IExecutionContext* createExecutionContextWithRuntimeConfig(IRuntimeConfig*) noexcept=0;
    virtual IRuntimeConfig* createRuntimeConfig() noexcept=0;
    virtual int64_t getEngineStat(EngineStat) const noexcept=0;
    virtual char const* getAliasedInputTensor(char const*) const noexcept=0;
    virtual bool areWeightsLoaded() const noexcept=0;
    virtual bool loadWeights(void const*,int64_t) noexcept=0;
    virtual bool loadWeightsAsync(IStreamReaderV2&,CUstream_st*) noexcept=0;
    virtual bool unloadWeights() noexcept=0;
    virtual IWeightsManager* createWeightsManager() noexcept=0;
};
}
class IHostMemory {public:virtual ~IHostMemory() noexcept=default;protected:apiv::VHostMemory* mImpl=nullptr;};
class ICudaEngine {public:virtual ~ICudaEngine() noexcept=default;protected:apiv::VCudaEngine* mImpl=nullptr;};
}
// Public execution-context dispatch declarations, adapted under the same Apache-2.0 provenance.
struct CUevent_st;
namespace nvinfer1 {
namespace v_1_0 {class IProfiler;class IOutputAllocator;class IDebugListener;}
using IProfiler=v_1_0::IProfiler;
using IOutputAllocator=v_1_0::IOutputAllocator;
using IDebugListener=v_1_0::IDebugListener;
namespace apiv {
class VExecutionContext:public VRoot {
public:
    virtual IExecutionContext* getPImpl() noexcept=0;
    virtual void setDebugSync(bool) noexcept=0;
    virtual bool getDebugSync() const noexcept=0;
    virtual void setProfiler(IProfiler*) noexcept=0;
    virtual IProfiler* getProfiler() const noexcept=0;
    virtual ICudaEngine const& getEngine() const noexcept=0;
    virtual void setName(char const*) noexcept=0;
    virtual char const* getName() const noexcept=0;
    virtual void setDeviceMemory(void*) noexcept=0;
    virtual int32_t getOptimizationProfile() const noexcept=0;
    virtual bool allInputDimensionsSpecified() const noexcept=0;
    virtual void setErrorRecorder(IErrorRecorder*) noexcept=0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept=0;
    virtual bool executeV2(void* const*) noexcept=0;
    virtual bool setOptimizationProfileAsync(int32_t,CUstream_st*) noexcept=0;
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
    virtual bool setInputConsumedEvent(CUevent_st*) noexcept=0;
    virtual CUevent_st* getInputConsumedEvent() const noexcept=0;
    virtual void* getOutputTensorAddress(char const*) const noexcept=0;
    virtual bool setOutputAllocator(char const*,IOutputAllocator*) noexcept=0;
    virtual IOutputAllocator* getOutputAllocator(char const*) noexcept=0;
    virtual int64_t getMaxOutputSize(char const*) const noexcept=0;
    virtual bool setTemporaryStorageAllocator(IGpuAllocator*) noexcept=0;
    virtual IGpuAllocator* getTemporaryStorageAllocator() const noexcept=0;
    virtual bool enqueueV3(CUstream_st*) noexcept=0;
    virtual void setPersistentCacheLimit(size_t) noexcept=0;
    virtual size_t getPersistentCacheLimit() const noexcept=0;
    virtual bool setNvtxVerbosity(ProfilingVerbosity) noexcept=0;
    virtual ProfilingVerbosity getNvtxVerbosity() const noexcept=0;
    virtual void setAuxStreams(CUstream_st**,int32_t) noexcept=0;
    virtual bool setDebugListener(IDebugListener*) noexcept=0;
    virtual IDebugListener* getDebugListener() noexcept=0;
    virtual bool setTensorDebugState(char const*,bool) noexcept=0;
    virtual bool getDebugState(char const*) const noexcept=0;
    virtual bool setAllTensorsDebugState(bool) noexcept=0;
    virtual size_t updateDeviceMemorySizeForShapes() noexcept=0;
    virtual void setDeviceMemoryV2(void*,int64_t) noexcept=0;
    virtual IRuntimeConfig* getRuntimeConfig() const noexcept=0;
    virtual bool setUnfusedTensorsDebugState(bool) noexcept=0;
    virtual bool getUnfusedTensorsDebugState() const noexcept=0;
    virtual bool isStreamCapturable(CUstream_st*) const noexcept=0;
    virtual bool setCommunicator(void*) noexcept=0;
};
}
class IExecutionContext {public:virtual ~IExecutionContext() noexcept=default;protected:apiv::VExecutionContext* mImpl=nullptr;};
}
