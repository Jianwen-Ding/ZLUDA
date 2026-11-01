// Original callback lease helper. Callbacks execute outside the slot lock, including reference callbacks.
#pragma once
#include "runtime_contract.hpp"
#include <memory>
#include <mutex>
#include <cstring>
#include <utility>
class ZtrtRecorderSlot {
    struct Lease {
        nvinfer1::IErrorRecorder* value;
        explicit Lease(nvinfer1::IErrorRecorder* p):value(p){value->incRefCount();}
        ~Lease(){value->decRefCount();}
    };
    mutable std::mutex mutex;std::shared_ptr<Lease> recorder;
    std::shared_ptr<Lease> snapshot() const {std::lock_guard<std::mutex> lock(mutex);return recorder;}
    static void log(nvinfer1::ILogger* logger,char const* text) noexcept{if(logger)logger->log(nvinfer1::ILogger::Severity::kERROR,text);}
public:
    class Pin {
        std::shared_ptr<Lease> lease;
        friend class ZtrtRecorderSlot;
        explicit Pin(std::shared_ptr<Lease> value):lease(std::move(value)){}
    public:
        nvinfer1::IErrorRecorder* get() const noexcept{return lease?lease->value:nullptr;}
    };
    Pin pin() const noexcept{try{return Pin(snapshot());}catch(...){return Pin(nullptr);}}
    void set(nvinfer1::IErrorRecorder* p,nvinfer1::ILogger* logger) noexcept {
        if(p){auto info=p->getInterfaceInfo();if(p->getAPILanguage()!=nvinfer1::APILanguage::kCPP||!info.kind||std::strcmp(info.kind,"IErrorRecorder")||info.major!=1||info.minor!=0){log(logger,"unsupported error recorder interface");return;}}
        try{
            auto replacement=p?std::make_shared<Lease>(p):nullptr;
            {std::lock_guard<std::mutex> lock(mutex);recorder.swap(replacement);}
        }catch(...){log(logger,"error recorder registration failed");}
    }
    nvinfer1::IErrorRecorder* get() const noexcept{try{auto lease=snapshot();return lease?lease->value:nullptr;}catch(...){return nullptr;}}
    void report(nvinfer1::ILogger* logger,char const* text,nvinfer1::ErrorCode code=nvinfer1::ErrorCode::kUNSPECIFIED_ERROR) const noexcept{
        try{auto lease=snapshot();if(lease){char description[128];size_t n=0;if(text){while(n<127&&text[n]){description[n]=text[n];++n;}}description[n]=0;lease->value->reportError(code,description);return;}}
        catch(...){ }
        log(logger,text);
    }
};
