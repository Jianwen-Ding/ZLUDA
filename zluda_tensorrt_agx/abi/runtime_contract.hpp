// SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// Minimal public runtime dispatch declarations adapted from Apache-2.0 RTX API docs.
// Prototype only; see README.md. No proprietary runtime code is included.
#pragma once
#include "NvInferRuntimeBase.h"
#include <cstddef>
namespace nvinfer1 {
class IRuntime; class ICudaEngine;
namespace v_1_0 { class IGpuAllocator; class IErrorRecorder; class IStreamReaderV2; class IPluginCreatorInterface; class IPluginResource; }
using IGpuAllocator=v_1_0::IGpuAllocator;
using IErrorRecorder=v_1_0::IErrorRecorder;
using IStreamReaderV2=v_1_0::IStreamReaderV2;
using IPluginCreatorInterface=v_1_0::IPluginCreatorInterface;
using IPluginResource=v_1_0::IPluginResource;
enum class EngineValidity:int32_t { kVALID=0,kSUBOPTIMAL=1,kINVALID=2 };
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
