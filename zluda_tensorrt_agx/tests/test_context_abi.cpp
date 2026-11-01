// GPU execution through independently declared engine/context dispatch prefixes.
#include "abi_probe.hpp"
#include "zluda_tensorrt.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
struct Logger final:nvinfer1::ILogger{int errors=0;void log(Severity,char const*)noexcept override{++errors;}};
template<class T>T symbol(void* library,char const* name){auto p=dlsym(library,name);if(!p)throw std::runtime_error(name);return reinterpret_cast<T>(p);}
void check(int code){if(code)throw std::runtime_error("driver error "+std::to_string(code));}
int main(int argc,char** argv){assert(argc==2);auto path=std::getenv("ZLUDA_AGX_LIBCUDA");assert(path);void* driver=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!driver)throw std::runtime_error(dlerror());
    check(symbol<int(*)(unsigned)>(driver,"cuInit")(0));void* primary=nullptr;
    check(symbol<int(*)(void**,int)>(driver,"cuDevicePrimaryCtxRetain")(&primary,0));check(symbol<int(*)(void*)>(driver,"cuCtxSetCurrent")(primary));
    void* library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!library)throw std::runtime_error(dlerror());auto wrap=symbol<void*(*)(void*,void*)>(library,"ztrtRtxWrapNativeEngine");Logger logger;
    char const source[]=R"({"tensors":[{"name":"input","shape":[2,3]},{"name":"weights","shape":[2,3],"data":[1,2,3,4,5,6]},{"name":"middle","shape":[2,3]},{"name":"output","shape":[6]}],"inputs":[0],"outputs":[3],"nodes":[{"output":2,"op":"element_wise","a":0,"b":1,"kind":"add"},{"output":3,"op":"reshape","input":2,"shape":[6]}]})";
    auto native=ztrtBuildJson(reinterpret_cast<uint8_t const*>(source),sizeof(source)-1,4096);assert(native);
    auto engine=static_cast<probe::Engine*>(wrap(native,&logger));assert(engine);auto object=static_cast<probe::Context*>(engine->impl->createExecutionContext(0));assert(object);auto c=object->impl;assert(c&&c->getPImpl()==object&& &c->getEngine()==engine);
    assert(c->getOptimizationProfile()==0&&c->allInputDimensionsSpecified()&&c->inferShapes(0,nullptr)==0);
    auto shape=c->getTensorShape("input");assert(shape.nbDims==2&&shape.d[0]==2&&shape.d[1]==3);assert(c->setInputShape("input",shape));shape.d[0]=3;assert(!c->setInputShape("input",shape));
    auto strides=c->getTensorStrides("input");assert(strides.nbDims==2&&strides.d[0]==3&&strides.d[1]==1);assert(c->getTensorShape("missing").nbDims==-1);
    c->setName("ABI GPU context");assert(std::strcmp(c->getName(),"ABI GPU context")==0);c->setDebugSync(true);assert(c->getDebugSync());
    assert(!c->enqueueV3(nullptr)&&!c->getTensorAddress("input"));
    float input[6]={-5,-2,-1,1,3,7},output[6];void* dx=nullptr;void* dy=nullptr;auto alloc=symbol<int(*)(void**,size_t)>(driver,"cuMemAlloc_v2");check(alloc(&dx,sizeof(input)));check(alloc(&dy,sizeof(output)));
    auto upload=symbol<int(*)(void*,void const*,size_t)>(driver,"cuMemcpyHtoD_v2");auto download=symbol<int(*)(void*,void const*,size_t)>(driver,"cuMemcpyDtoH_v2");check(upload(dx,input,sizeof(input)));
    assert(c->setInputTensorAddress("input",dx)&&c->setOutputTensorAddress("output",dy));assert(!c->setInputTensorAddress("output",dx)&&!c->setOutputTensorAddress("input",dy));
    assert(c->getTensorAddress("input")==dx&&c->getOutputTensorAddress("output")==dy&&c->getMaxOutputSize("output")==24&&c->getMaxOutputSize("input")==-1);
    assert(c->enqueueV3(nullptr));check(symbol<int(*)()>(driver,"cuCtxSynchronize")());check(download(output,dy,sizeof(output)));for(int i=0;i<6;++i)assert(std::abs(output[i]-(input[i]+float(i+1)))<1e-5f);
    assert(c->setTensorAddress("input",nullptr)&&!c->enqueueV3(nullptr)&&!c->getTensorAddress("input"));assert(c->setInputTensorAddress("input",dx));
    c->setDeviceMemory(dy);assert(!c->enqueueV3(nullptr));c->setDeviceMemory(nullptr);
    for(float& v:input){v*=2;}
    check(upload(dx,input,sizeof(input)));void* bindings[2]={dx,dy};assert(c->executeV2(bindings));check(download(output,dy,sizeof(output)));for(int i=0;i<6;++i)assert(std::abs(output[i]-(input[i]+float(i+1)))<1e-5f);
    void* stream=nullptr;check(symbol<int(*)(void**,unsigned)>(driver,"cuStreamCreate")(&stream,1));assert(c->setOptimizationProfileAsync(0,stream)&&!c->setOptimizationProfileAsync(1,stream));assert(c->enqueueV3(stream));check(symbol<int(*)(void*)>(driver,"cuStreamSynchronize")(stream));assert(c->isStreamCapturable(stream)&&!c->isStreamCapturable(nullptr));
    void* graph=nullptr;void* graphExec=nullptr;
    check(symbol<int(*)(void*,int)>(driver,"cuStreamBeginCapture_v2")(stream,0));assert(c->enqueueV3(stream));
    check(symbol<int(*)(void*,void**)>(driver,"cuStreamEndCapture")(stream,&graph));
    check(symbol<int(*)(void**,void*,uint64_t)>(driver,"cuGraphInstantiateWithFlags")(&graphExec,graph,0));
    for(float& v:input){v+=1;}
    check(upload(dx,input,sizeof(input)));check(symbol<int(*)(void*,void*)>(driver,"cuGraphLaunch")(graphExec,stream));
    check(symbol<int(*)(void*)>(driver,"cuStreamSynchronize")(stream));check(download(output,dy,sizeof(output)));
    for(int i=0;i<6;++i){assert(std::abs(output[i]-(input[i]+float(i+1)))<1e-5f);}
    check(symbol<int(*)(void*)>(driver,"cuGraphExecDestroy")(graphExec));check(symbol<int(*)(void*)>(driver,"cuGraphDestroy")(graph));
    check(symbol<int(*)(void*)>(driver,"cuStreamDestroy_v2")(stream));
    assert(c->updateDeviceMemorySizeForShapes()==256&&c->getPersistentCacheLimit()==0);
    assert(c->setNvtxVerbosity(1)&&!c->setNvtxVerbosity(2)&&c->getNvtxVerbosity()==1);
    assert(c->setAllTensorsDebugState(false)&&!c->setAllTensorsDebugState(true)&&!c->getRuntimeConfig());
    assert(c->setCommunicator(nullptr)&&!c->setCommunicator(reinterpret_cast<void*>(uintptr_t(1))));
    c->setDeviceMemoryV2(dy,24);assert(!c->enqueueV3(nullptr));c->setDeviceMemoryV2(nullptr,0);
    assert(c->enqueueV3(nullptr));check(symbol<int(*)()>(driver,"cuCtxSynchronize")());
    assert(!c->setInputConsumedEvent(reinterpret_cast<void*>(uintptr_t(1)))&&!c->getInputConsumedEvent());assert(c->setOutputAllocator("output",nullptr)&&!c->setOutputAllocator("output",reinterpret_cast<void*>(uintptr_t(1))));
    void* other=nullptr;check(symbol<int(*)(void**,unsigned,int)>(driver,"cuCtxCreate_v2")(&other,0,0));assert(!c->enqueueV3(nullptr));
    delete object;delete engine;void* current=nullptr;check(symbol<int(*)(void**)>(driver,"cuCtxGetCurrent")(&current));assert(current==other);
    check(symbol<int(*)(void*)>(driver,"cuCtxSetCurrent")(primary));check(symbol<int(*)(void*)>(driver,"cuCtxDestroy_v2")(other));auto free=symbol<int(*)(void*)>(driver,"cuMemFree_v2");check(free(dx));check(free(dy));
    assert(logger.errors>=5);dlclose(library);dlclose(driver);std::cout<<"PASS: public-style context dispatch GPU execution, bindings, shapes, streams and context restoration\n";
}
