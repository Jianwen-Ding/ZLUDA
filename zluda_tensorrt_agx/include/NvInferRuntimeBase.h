// SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// Public interface declarations adapted from the Apache-2.0 TensorRT RTX API documentation.
// Original subset implementation; this file does not establish complete TensorRT binary compatibility.
// See ../abi/README.md for provenance, limits and remaining work. See the distributed LICENSE-APACHE for the license text.
#ifndef ZLUDA_AGX_NVINFER_RUNTIME_BASE_H
#define ZLUDA_AGX_NVINFER_RUNTIME_BASE_H
#include <cstdint>
#include <cstddef>
namespace nvinfer1 {
using char_t=char;
using AsciiChar=char_t;
enum class DataType : int32_t { kFLOAT=0, kHALF=1, kINT8=2, kINT32=3, kBOOL=4, kUINT8=5, kFP8=6, kBF16=7, kINT64=8, kINT4=9, kFP4=10, kE8M0=11 };
enum class TensorIOMode : int32_t { kNONE=0, kINPUT=1, kOUTPUT=2 };
enum class APILanguage : int32_t { kCPP=0, kPYTHON=1 };
using InterfaceKind=char const*;
struct InterfaceInfo { InterfaceKind kind; int32_t major; int32_t minor; };
class IVersionedInterface {
public:
    virtual APILanguage getAPILanguage() const noexcept {return APILanguage::kCPP;}
    virtual InterfaceInfo getInterfaceInfo() const noexcept=0;
    virtual ~IVersionedInterface() noexcept=default;
protected:
    IVersionedInterface()=default;
    IVersionedInterface(IVersionedInterface const&)=default;
    IVersionedInterface(IVersionedInterface&&)=default;
    IVersionedInterface& operator=(IVersionedInterface const&) & = default;
    IVersionedInterface& operator=(IVersionedInterface&&) & = default;
};
enum class ErrorCode:int32_t { kSUCCESS=0,kUNSPECIFIED_ERROR=1,kINTERNAL_ERROR=2,kINVALID_ARGUMENT=3,kINVALID_CONFIG=4,kFAILED_ALLOCATION=5,kFAILED_INITIALIZATION=6,kFAILED_EXECUTION=7,kFAILED_COMPUTATION=8,kINVALID_STATE=9,kUNSUPPORTED_STATE=10 };
namespace v_1_0 {
class IErrorRecorder:public IVersionedInterface {
public:
    InterfaceInfo getInterfaceInfo() const noexcept override{return {"IErrorRecorder",1,0};}
    using ErrorDesc=char const*;using RefCount=int32_t;
    static constexpr size_t kMAX_DESC_LENGTH=127;
    IErrorRecorder()=default;~IErrorRecorder() noexcept override=default;
    virtual int32_t getNbErrors() const noexcept=0;
    virtual ErrorCode getErrorCode(int32_t) const noexcept=0;
    virtual ErrorDesc getErrorDesc(int32_t) const noexcept=0;
    virtual bool hasOverflowed() const noexcept=0;
    virtual void clear() noexcept=0;
    virtual bool reportError(ErrorCode,ErrorDesc) noexcept=0;
    virtual RefCount incRefCount() noexcept=0;
    virtual RefCount decRefCount() noexcept=0;
protected:
    IErrorRecorder(IErrorRecorder const&)=default;IErrorRecorder(IErrorRecorder&&)=default;
    IErrorRecorder& operator=(IErrorRecorder const&) & =default;
    IErrorRecorder& operator=(IErrorRecorder&&) & =default;
};
class ILogger : public IVersionedInterface {
public:
    InterfaceInfo getInterfaceInfo() const noexcept override {return {"ILogger",1,0};}
    enum class Severity : int32_t { kINTERNAL_ERROR=0, kERROR=1, kWARNING=2, kINFO=3, kVERBOSE=4 };
    virtual void log(Severity severity,AsciiChar const* message) noexcept=0;
    ILogger()=default;
    ~ILogger() override=default;
};
}
using ILogger=v_1_0::ILogger;
using IErrorRecorder=v_1_0::IErrorRecorder;
struct Dims64 { static constexpr int32_t MAX_DIMS=8; int32_t nbDims=0; int64_t d[MAX_DIMS]{}; };
using Dims=Dims64;
}
#endif
