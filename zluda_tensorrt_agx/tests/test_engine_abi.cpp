// Independently declared engine dispatch-prefix probe; no adapter contract header included.
#include "NvInferRuntimeBase.h"
#include "zluda_tensorrt.h"
#include <cassert>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "abi_probe.hpp"
#include "runtime_probe.hpp"

struct Logger final:nvinfer1::ILogger{int errors=0;void log(Severity,char const*)noexcept override{++errors;}};
int main(int argc,char** argv){assert(argc==2);void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib)throw std::runtime_error(dlerror());
    auto wrap=reinterpret_cast<void*(*)(void*,void*)>(dlsym(lib,"ztrtRtxWrapNativeEngine"));assert(wrap);Logger logger;assert(!wrap(nullptr,&logger));
    auto createRuntime=reinterpret_cast<void*(*)(void*,int32_t)>(dlsym(lib,"createInferRuntime_INTERNAL"));assert(createRuntime);
    char const json[]=R"({"tensors":[{"name":"input","shape":[2,3]},{"name":"intermediate","shape":[2,3]},{"name":"output","shape":[6]},{"name":"weights","shape":[2,3],"data":[1,2,3,4,5,6]}],"inputs":[0],"outputs":[2],"nodes":[{"output":1,"op":"element_wise","a":0,"b":3,"kind":"add"},{"output":2,"op":"reshape","input":1,"shape":[6]}]})";
    for(int iteration=0;iteration<100;++iteration){
        auto native=ztrtBuildJson(reinterpret_cast<uint8_t const*>(json),sizeof(json)-1,4096);assert(native);
        if(iteration==0){assert(!wrap(native,nullptr));assert(ztrtGetLayerCount(native)==2);}
        auto engine=static_cast<probe::Engine*>(wrap(native,&logger));assert(engine);auto e=engine->impl;assert(e&&e->getPImpl()==engine);
        assert(e->getNbLayers()==2&&e->getNbIOTensors()==2&&e->getNbOptimizationProfiles()==1&&!e->isRefittable());
        assert(std::strcmp(e->getIOTensorName(0),"input")==0&&std::strcmp(e->getIOTensorName(1),"output")==0&&!e->getIOTensorName(2));
        auto shape=e->getTensorShape("input");assert(shape.nbDims==2&&shape.d[0]==2&&shape.d[1]==3);shape=e->getTensorShape("output");assert(shape.nbDims==1&&shape.d[0]==6);
        assert(e->getTensorShape("missing").nbDims==-1&&e->getTensorShape("intermediate").nbDims==-1);
        assert(e->getTensorDataType("input")==nvinfer1::DataType::kFLOAT&&e->getTensorLocation("input")==0);
        assert(e->getTensorDataType("missing")==nvinfer1::DataType::kFLOAT&&e->getTensorLocation("missing")==0&&e->getTensorFormat("missing")==0&&std::strcmp(e->getTensorFormatDesc("missing"),"")==0);
        assert(e->getTensorIOMode("input")==nvinfer1::TensorIOMode::kINPUT&&e->getTensorIOMode("output")==nvinfer1::TensorIOMode::kOUTPUT&&e->getTensorIOMode("missing")==nvinfer1::TensorIOMode::kNONE);
        assert(e->getTensorBytesPerComponent("input")==-1&&e->getTensorComponentsPerElement("input")==-1&&e->getTensorFormat("input")==0&&e->getTensorVectorizedDim("input")==-1);
        assert(e->getProfileShape("input",0,2).nbDims==2&&e->getProfileShape("input",1,0).nbDims==-1);
        assert(e->getTensorBytesPerComponentV2("input",0)==-1&&e->getTensorBytesPerComponentV2("input",1)==-1);
        assert(e->getTensorFormatDesc("input")&&e->getDeviceMemorySizeV2()==256&&e->getDeviceMemorySizeForProfileV2(1)==-1);
        assert(!e->createExecutionContext(1)&&!e->createSerializationConfig()&&!e->setWeightStreamingBudgetV2(1));
        auto memory=e->serialize();assert(memory);delete engine; // Returned custom bytes outlive the engine wrapper.
        auto h=memory->impl;assert(h->type()==nvinfer1::DataType::kINT8&&h->size()>24&&std::memcmp(h->data(),"ZTRTAGX2",8)==0);
        auto runtime=ztrtCreateRuntime();assert(runtime);auto restored=ztrtDeserializeEngine(runtime,static_cast<uint8_t const*>(h->data()),h->size(),4096);assert(restored);
        assert(ztrtGetConstantBytes(restored)==24);std::vector<uint8_t> copy(h->size());
        assert(ztrtSerializeEngine(restored,copy.data(),copy.size())==copy.size()&&std::memcmp(copy.data(),h->data(),copy.size())==0);
        auto publicRuntime=static_cast<probe::Runtime*>(createRuntime(&logger,10701));assert(publicRuntime);
        if(iteration==0){
            for(size_t length=0;length<copy.size();++length){assert(!publicRuntime->impl->deserializeCudaEngine(copy.data(),length));}
            auto broken=copy;broken[8]^=1;assert(!publicRuntime->impl->deserializeCudaEngine(broken.data(),broken.size()));
            broken=copy;for(size_t i=16;i<24;++i){broken[i]=255;}assert(!publicRuntime->impl->deserializeCudaEngine(broken.data(),broken.size()));
            broken=copy;broken[0]='?';assert(!publicRuntime->impl->deserializeCudaEngine(broken.data(),broken.size()));
            assert(!publicRuntime->impl->deserializeCudaEngine(copy.data(),(size_t(256)<<20)+17));
        }
        auto loaded=static_cast<probe::Engine*>(publicRuntime->impl->deserializeCudaEngine(copy.data(),copy.size()));assert(loaded);
        delete publicRuntime;auto encoded=loaded->impl->serialize();assert(encoded&&encoded->impl->size()==copy.size());
        assert(std::memcmp(encoded->impl->data(),copy.data(),copy.size())==0);delete loaded;delete encoded;
        ztrtDestroyRuntime(runtime);auto again=static_cast<probe::Engine*>(wrap(restored,&logger));assert(again&&again->impl->getNbIOTensors()==2);delete again;delete memory;
    }
    assert(logger.errors>=300);dlclose(lib);std::cout<<"PASS: engine metadata dispatch, native serialization ownership, rejected paths and 100 roundtrips\n";
}
