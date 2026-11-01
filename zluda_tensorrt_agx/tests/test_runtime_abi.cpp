// Separately compiled Linux C++ ABI-prefix probe. Deliberately does not include runtime_contract.hpp.
// Public method order is recorded from the Apache-2.0 RTX interface docs in abi/README.md.
#include "NvInferRuntimeBase.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include "runtime_probe.hpp"

struct Logger final:nvinfer1::ILogger {int count=0;void log(Severity,char const*) noexcept override{++count;}};
template<class T>T symbol(void* lib,char const* name){auto p=dlsym(lib,name);if(!p)throw std::runtime_error(name);return reinterpret_cast<T>(p);}
int main(int argc,char** argv){assert(argc==2);void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib)throw std::runtime_error(dlerror());
    auto version=symbol<int32_t(*)()>(lib,"getInferLibVersion");auto create=symbol<void*(*)(void*,int32_t)>(lib,"createInferRuntime_INTERNAL");
    assert(version()==10701);Logger logger;assert(!create(nullptr,10701));assert(!create(&logger,10600));
    for(int i=0;i<1000;++i){auto object=static_cast<probe::Runtime*>(create(&logger,10701));assert(object);auto p=object->impl;assert(p&&p->getPImpl()==object);
        assert(p->getLogger()==&logger&&p->getNbDLACores()==0&&p->getDLACore()==0);
        assert(p->getMaxThreads()==1&&p->setMaxThreads(7)&&p->getMaxThreads()==7);assert(!p->setMaxThreads(0)&&p->getMaxThreads()==7);
        assert(p->getTemporaryDirectory()==nullptr);char directory[]="temporary";p->setTemporaryDirectory(directory);directory[0]='X';assert(std::strcmp(p->getTemporaryDirectory(),"temporary")==0);p->setTemporaryDirectory(nullptr);assert(!p->getTemporaryDirectory());
        assert(p->getTempfileControlFlags()==3);p->setTempfileControlFlags(1);assert(p->getTempfileControlFlags()==1);p->setTempfileControlFlags(8);assert(p->getTempfileControlFlags()==1);
        auto& registry=p->getPluginRegistry();assert(registry.isParentSearchEnabled());registry.setParentSearchEnabled(false);assert(!registry.isParentSearchEnabled());int32_t creators=-1;assert(!registry.getAllCreators(&creators)&&creators==0);assert(!registry.loadLibrary("not-loaded"));
        assert(!p->getErrorRecorder());assert(!p->loadRuntime("not-loaded"));p->setEngineHostCodeAllowed(true);assert(!p->getEngineHostCodeAllowed());
        p->setDeferredWeightsLoading(true);assert(!p->getDeferredWeightsLoading());
        // Unknown valid buffers are rejected before custom payload decoding.
        char unknown[24]={};assert(!p->deserializeCudaEngine(unknown,sizeof(unknown)));
        assert(!p->deserializeCudaEngine(nullptr,4096));assert(!p->deserializeCudaEngine(unknown,7));uint64_t diagnostics=123;
        assert(p->getEngineValidity(unknown,24,&diagnostics)==2&&diagnostics==0&&p->getEngineHeaderSize()==0);
        assert(!p->setDLAWorkspaceAllocationStrategy(0));assert(p->getDLAWorkspaceAllocationStrategy()==0);
        delete object;
    }
    assert(logger.count>1000);dlclose(lib);std::cout<<"PASS: runtime factory/configuration prefix, callbacks, rejection paths and 1000 destruction cycles\n";
}
