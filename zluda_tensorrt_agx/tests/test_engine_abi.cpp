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
#include "stream_probe.hpp"

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
        auto config=static_cast<probe::Config*>(e->createRuntimeConfig());assert(config&&config->impl->getPImpl()==config);
        auto settings=config->impl;assert(settings->getExecutionContextAllocationStrategy()==0);
        settings->setExecutionContextAllocationStrategy(2);assert(settings->getExecutionContextAllocationStrategy()==2);
        settings->setExecutionContextAllocationStrategy(99);assert(settings->getExecutionContextAllocationStrategy()==2);
        assert(!e->createExecutionContextWithRuntimeConfig(config));settings->setExecutionContextAllocationStrategy(0);
        for(int value:{0,1,2}){settings->setDynamicShapesKernelSpecializationStrategy(value);assert(settings->getDynamicShapesKernelSpecializationStrategy()==value);}
        settings->setDynamicShapesKernelSpecializationStrategy(99);assert(!e->createExecutionContextWithRuntimeConfig(config));
        assert(!settings->createRuntimeCache()&&!settings->getRuntimeCache());
        assert(settings->getCudaGraphStrategy()==0&&settings->setCudaGraphStrategy(0)&&!settings->setCudaGraphStrategy(1)&&!settings->setCudaGraphStrategy(99));
        if(iteration==0){
            settings->setDynamicShapesKernelSpecializationStrategy(0);
            auto otherNative=ztrtBuildJson(reinterpret_cast<uint8_t const*>(json),sizeof(json)-1,4096);assert(otherNative);
            auto other=static_cast<probe::Engine*>(wrap(otherNative,&logger));assert(other);
            assert(!other->impl->createExecutionContextWithRuntimeConfig(config));delete other;
        }
        delete config;
        auto memory=e->serialize();assert(memory);delete engine; // Returned custom bytes outlive the engine wrapper.
        auto h=memory->impl;assert(h->type()==nvinfer1::DataType::kINT8&&h->size()>24&&std::memcmp(h->data(),"ZTRTAGX2",8)==0);
        auto runtime=ztrtCreateRuntime();assert(runtime);auto restored=ztrtDeserializeEngine(runtime,static_cast<uint8_t const*>(h->data()),h->size(),4096);assert(restored);
        assert(ztrtGetConstantBytes(restored)==24);std::vector<uint8_t> copy(h->size());
        assert(ztrtSerializeEngine(restored,copy.data(),copy.size())==copy.size()&&std::memcmp(copy.data(),h->data(),copy.size())==0);
        auto publicRuntime=static_cast<probe::Runtime*>(createRuntime(&logger,10701));assert(publicRuntime);
        uint64_t diagnostics=123;
        assert(publicRuntime->impl->getEngineHeaderSize()==24);
        assert(publicRuntime->impl->getEngineValidity(copy.data(),24,&diagnostics)==0&&diagnostics==0);
        if(iteration==0){
            auto corruptBody=copy;corruptBody[24]=0;
            assert(publicRuntime->impl->getEngineValidity(corruptBody.data(),24,&diagnostics)==0&&diagnostics==0);
            assert(!publicRuntime->impl->deserializeCudaEngine(corruptBody.data(),corruptBody.size()));
            for(size_t length=0;length<copy.size();++length){assert(!publicRuntime->impl->deserializeCudaEngine(copy.data(),length));}
            auto broken=copy;broken[8]^=1;assert(!publicRuntime->impl->deserializeCudaEngine(broken.data(),broken.size()));
            broken=copy;for(size_t i=16;i<24;++i){broken[i]=255;}assert(!publicRuntime->impl->deserializeCudaEngine(broken.data(),broken.size()));
            broken=copy;broken[0]='?';assert(!publicRuntime->impl->deserializeCudaEngine(broken.data(),broken.size()));
            assert(!publicRuntime->impl->deserializeCudaEngine(copy.data(),(size_t(256)<<20)+17));
        }
        auto loaded=static_cast<probe::Engine*>(publicRuntime->impl->deserializeCudaEngine(copy.data(),copy.size()));assert(loaded);
        if(iteration==0){
            for(size_t chunk:{size_t(1),size_t(7),size_t(1024*1024)}){
                probe::BytesReader reader(copy.data(),copy.size(),chunk);
                auto streamed=static_cast<probe::Engine*>(publicRuntime->impl->deserializeCudaEngineV2(&reader));assert(streamed);
                assert(reader.position==copy.size()&&reader.seeks==0);auto encodedStream=streamed->impl->serialize();assert(encodedStream);
                assert(encodedStream->impl->size()==copy.size()&&std::memcmp(encodedStream->impl->data(),copy.data(),copy.size())==0);
                delete encodedStream;delete streamed;
            }
            for(size_t end=0;end<copy.size();++end){probe::BytesReader reader(copy.data(),end,7);assert(!publicRuntime->impl->deserializeCudaEngineV2(&reader));}
            for(int failure:{1,2}){probe::BytesReader reader(copy.data(),copy.size(),7);reader.failure=failure;assert(!publicRuntime->impl->deserializeCudaEngineV2(&reader));}
            probe::BytesReader incompatible(copy.data(),copy.size(),7);incompatible.supported=false;
            assert(!publicRuntime->impl->deserializeCudaEngineV2(&incompatible)&&incompatible.reads==0);
            auto trailing=copy;trailing.resize(copy.size()+17,255);probe::BytesReader reader(trailing.data(),trailing.size(),7);
            auto streamed=static_cast<probe::Engine*>(publicRuntime->impl->deserializeCudaEngineV2(&reader));assert(streamed&&reader.position==copy.size());delete streamed;
            auto bad=copy;bad[0]='?';probe::BytesReader unknown(bad.data(),bad.size(),7);assert(!publicRuntime->impl->deserializeCudaEngineV2(&unknown)&&unknown.position==16);
        }
        delete publicRuntime;auto encoded=loaded->impl->serialize();assert(encoded&&encoded->impl->size()==copy.size());
        assert(std::memcmp(encoded->impl->data(),copy.data(),copy.size())==0);delete loaded;delete encoded;
        ztrtDestroyRuntime(runtime);auto again=static_cast<probe::Engine*>(wrap(restored,&logger));assert(again&&again->impl->getNbIOTensors()==2);delete again;delete memory;
    }
    // Exercise the 1 MiB read limit with real independently generated weight records.
    auto builder=ztrtCreateBuilder();assert(builder);int64_t dims[1]={262145};std::vector<float> weights(size_t(dims[0]),1.25f);
    int x=ztrtBuilderAddInput(builder,"large_input",dims,1);int w=ztrtBuilderAddConstant(builder,"large_weights",dims,1,weights.data(),weights.size());
    assert(x>=0&&w>=0);int y=ztrtBuilderAddElementWise(builder,x,w,0);assert(y>=0&&ztrtBuilderMarkOutput(builder,y)==0);
    auto native=ztrtBuilderBuild(builder);assert(native);ztrtDestroyBuilder(builder);
    size_t size=ztrtSerializeEngine(native,nullptr,0);assert(size>(1<<20));std::vector<uint8_t> bytes(size);assert(ztrtSerializeEngine(native,bytes.data(),size)==size);ztrtDestroyEngine(native);
    auto runtime=static_cast<probe::Runtime*>(createRuntime(&logger,10701));assert(runtime);
    probe::BytesReader reader(bytes.data(),bytes.size(),size_t(1)<<20);
    auto loaded=static_cast<probe::Engine*>(runtime->impl->deserializeCudaEngineV2(&reader));assert(loaded&&reader.reads>=3&&reader.position==size);
    auto memory=loaded->impl->serialize();assert(memory&&memory->impl->size()==size&&std::memcmp(memory->impl->data(),bytes.data(),size)==0);
    delete memory;delete loaded;delete runtime;
    assert(logger.errors>=300);dlclose(lib);std::cout<<"PASS: engine metadata dispatch, native serialization ownership, rejected paths and 100 roundtrips\n";
}
