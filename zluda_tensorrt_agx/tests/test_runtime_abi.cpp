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
#include <sys/mman.h>
#include <unistd.h>
#include <thread>
#include <vector>
#include <string>
#include "runtime_probe.hpp"
#include "recorder_probe.hpp"

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
        assert(p->getEngineValidity(unknown,24,&diagnostics)==2&&diagnostics==(1ULL<<5)&&p->getEngineHeaderSize()==24);
        assert(!p->setDLAWorkspaceAllocationStrategy(0));assert(p->getDLAWorkspaceAllocationStrategy()==0);
        delete object;
    }
    // A protected next page proves the validity call never reads beyond the 24-byte header.
    auto object=static_cast<probe::Runtime*>(create(&logger,10701));assert(object);auto p=object->impl;
    auto page=static_cast<size_t>(sysconf(_SC_PAGESIZE));assert(page>=24);
    auto mapping=static_cast<unsigned char*>(mmap(nullptr,page*2,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));assert(mapping!=MAP_FAILED);
    assert(mprotect(mapping+page,page,PROT_NONE)==0);auto header=mapping+page-24;
    std::memset(header,0,24);std::memcpy(header,"ZTRTAGX2",8);header[8]=64;header[16]=32;
    uint64_t diagnostics=123;assert(p->getEngineValidity(header,24,&diagnostics)==0&&diagnostics==0);
    assert(p->getEngineValidity(header,4096,nullptr)==0); // Body is deliberately inaccessible.
    for(int64_t size=-1;size<24;++size){assert(p->getEngineValidity(header,size,&diagnostics)==2&&diagnostics==(1ULL<<5));}
    assert(p->getEngineValidity(nullptr,24,&diagnostics)==2&&diagnostics==(1ULL<<5));
    header[16]=0;assert(p->getEngineValidity(header,24,&diagnostics)==2&&diagnostics==(1ULL<<5));
    header[16]=57;assert(p->getEngineValidity(header,24,nullptr)==2);
    header[16]=32;header[8]=8;assert(p->getEngineValidity(header,24,nullptr)==2);
    header[8]=64;header[15]=255;assert(p->getEngineValidity(header,24,nullptr)==2);header[15]=0;
    header[7]='1';assert(p->getEngineValidity(header,24,&diagnostics)==0&&diagnostics==0);
    header[7]='9';assert(p->getEngineValidity(header,24,&diagnostics)==2&&diagnostics==(1ULL<<5));
    assert(munmap(mapping,page*2)==0);delete object;
    Recorder recorder,invalid;invalid.supported=false;
    object=static_cast<probe::Runtime*>(create(&logger,10701));p=object->impl;
    p->setErrorRecorder(&recorder);assert(p->getErrorRecorder()==&recorder&&recorder.references==1);
    p->setErrorRecorder(&invalid);assert(p->getErrorRecorder()==&recorder&&invalid.references==0);
    auto logged=logger.count;assert(!p->setMaxThreads(0));assert(recorder.reports==1&&logger.count==logged);
    recorder.fatal=true;recorder.user=p;
    recorder.callback=[](void* value){static_cast<probe::Dispatch*>(value)->setErrorRecorder(nullptr);};
    assert(!p->setMaxThreads(0));assert(!p->getErrorRecorder()&&recorder.references==0&&recorder.reports==2);
    recorder.callback=nullptr;p->setErrorRecorder(&recorder);p->setErrorRecorder(&recorder);assert(recorder.references==1);
    auto& registry=p->getPluginRegistry();registry.setErrorRecorder(&recorder);
    assert(registry.getErrorRecorder()==&recorder&&recorder.references==2);assert(!registry.loadLibrary("not-loaded"));
    registry.setErrorRecorder(nullptr);assert(recorder.references==1);
    std::string payload="{\""+std::string(256,'x')+"\":0}";
    std::string malformed="ZTRTAGX1";for(unsigned i=0;i<8;++i){malformed+=char(uint64_t(payload.size())>>(8*i));}malformed+=payload;
    assert(!p->deserializeCudaEngine(malformed.data(),malformed.size()));
    assert(std::strlen(recorder.getErrorDesc(0))==127&&recorder.getErrorCode(0)==nvinfer1::ErrorCode::kUNSPECIFIED_ERROR);
    auto before=recorder.reports.load();std::vector<std::thread> workers;
    for(int i=0;i<4;++i){workers.emplace_back([&]{for(int j=0;j<1000;++j){assert(!p->setMaxThreads(0));}});}
    for(int i=0;i<1000;++i){p->setErrorRecorder(&recorder);}
    for(auto& worker:workers){worker.join();}
    assert(recorder.reports==before+4000&&recorder.references==1);
    delete object;assert(recorder.references==0);
    assert(logger.count>1000);dlclose(lib);std::cout<<"PASS: runtime factory/configuration prefix, callbacks, rejection paths and 1000 destruction cycles\n";
}
