// Independent public reader call shapes; no implementation contract header.
#pragma once
#include "NvInferRuntimeBase.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
namespace probe {
class Reader:public nvinfer1::IVersionedInterface {
public:
    nvinfer1::InterfaceInfo getInterfaceInfo() const noexcept override{return {"IStreamReaderV2",1,0};}
    virtual int64_t read(void*,int64_t,void*) noexcept=0;
    virtual bool seek(int64_t,int32_t) noexcept=0;
};
class BytesReader final:public Reader {
public:
    unsigned char const* data;size_t length,position=0,chunk;int reads=0,seeks=0;int failure=0;
    bool supported=true;
    BytesReader(void const* p,size_t n,size_t limit):data(static_cast<unsigned char const*>(p)),length(n),chunk(limit){}
    nvinfer1::InterfaceInfo getInterfaceInfo() const noexcept override{return {"IStreamReaderV2",supported?1:2,0};}
    int64_t read(void* destination,int64_t count,void* stream) noexcept override{
        ++reads;assert(stream==nullptr&&count>0&&count<=(1<<20));
        if(failure==1){return -1;}
        if(failure==2){return count+1;}
        size_t n=std::min({static_cast<size_t>(count),length-position,chunk});
        if(n){std::memcpy(destination,data+position,n);position+=n;}return static_cast<int64_t>(n);
    }
    bool seek(int64_t,int32_t) noexcept override{++seeks;return false;}
};
}
