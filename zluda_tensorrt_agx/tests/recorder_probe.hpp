#pragma once
#include "NvInferRuntimeBase.h"
#include <atomic>
#include <cassert>
#include <cstring>
#include <mutex>
#include <thread>
#include <chrono>
struct Recorder final:nvinfer1::IErrorRecorder {
    std::atomic<int> references{0},reports{0},increments{0},decrements{0};
    mutable std::atomic<bool> pauseValidation{false},validationEntered{false},resumeValidation{false};
    bool supported=true;bool fatal=false;void (*callback)(void*)=nullptr;void* user=nullptr;
    mutable std::mutex mutex;char description[128]{};nvinfer1::ErrorCode code=nvinfer1::ErrorCode::kSUCCESS;
    nvinfer1::InterfaceInfo getInterfaceInfo() const noexcept override{
        if(pauseValidation.exchange(false)){
            validationEntered=true;auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
            while(!resumeValidation.load()){assert(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();}
            assert(references>0); // A parent-to-child inheritance lease must still hold this recorder.
        }
        return {"IErrorRecorder",supported?1:2,0};}
    int32_t getNbErrors() const noexcept override{return reports.load();}
    nvinfer1::ErrorCode getErrorCode(int32_t) const noexcept override{std::lock_guard<std::mutex> lock(mutex);return code;}
    char const* getErrorDesc(int32_t) const noexcept override{return description;}
    bool hasOverflowed() const noexcept override{return false;}
    void clear() noexcept override{reports=0;}
    bool reportError(nvinfer1::ErrorCode value,char const* text) noexcept override{
        assert(references>0);assert(text&&std::strlen(text)<=127);
        {std::lock_guard<std::mutex> lock(mutex);code=value;std::strncpy(description,text,127);description[127]=0;}
        ++reports;if(callback)callback(user);return fatal;
    }
    RefCount incRefCount() noexcept override{++increments;return ++references;}
    RefCount decRefCount() noexcept override{++decrements;auto before=references.fetch_sub(1);assert(before>0);return before-1;}
    ~Recorder() noexcept override{assert(references==0&&increments==decrements);}
};
