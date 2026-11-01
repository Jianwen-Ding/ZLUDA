// Compile against the independent public RTX headers, never our adapter contract/source wrapper.
#include <NvInferRuntime.h>
#include "zluda_tensorrt.h"
#include <cassert>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <stdexcept>
struct Logger final:nvinfer1::ILogger {
    int errors=0;
    void log(Severity,char const*) noexcept override{++errors;}
};
int main(int argc,char** argv){
    assert(argc==2);
    static_assert(NV_TENSORRT_MAJOR==1&&NV_TENSORRT_MINOR==7&&NV_TENSORRT_PATCH==1,"reference target is RTX 1.7.1");
    static_assert(sizeof(nvinfer1::Dims)==72,"Linux 64-bit dimensions");
    void* library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!library)throw std::runtime_error(dlerror());
    auto create=reinterpret_cast<void*(*)(void*,int32_t)>(dlsym(library,"createInferRuntime_INTERNAL"));assert(create);
    Logger logger;std::unique_ptr<nvinfer1::IRuntime> runtime(static_cast<nvinfer1::IRuntime*>(create(&logger,10701)));assert(runtime);
    assert(runtime->getMaxThreads()==1&&runtime->setMaxThreads(3)&&runtime->getMaxThreads()==3);
    char const json[]=R"({"tensors":[{"name":"input","shape":[2,3]},{"name":"output","shape":[2,3]}],"inputs":[0],"outputs":[1],"nodes":[{"output":1,"op":"activation","input":0,"kind":"relu"}]})";
    auto native=ztrtBuildJson(reinterpret_cast<uint8_t const*>(json),sizeof(json)-1,4096);assert(native);
    auto size=ztrtSerializeEngine(native,nullptr,0);assert(size);std::unique_ptr<uint8_t[]> bytes(new uint8_t[size]);assert(ztrtSerializeEngine(native,bytes.get(),size)==size);ztrtDestroyEngine(native);
    uint64_t diagnostics=99;assert(runtime->getEngineHeaderSize()==24);
    assert(runtime->getEngineValidity(bytes.get(),24,&diagnostics)==nvinfer1::EngineValidity::kVALID&&diagnostics==0);
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(bytes.get(),size));assert(engine);runtime.reset();
    assert(engine->getNbIOTensors()==2&&engine->getNbLayers()==1);
    assert(engine->getTensorIOMode("input")==nvinfer1::TensorIOMode::kINPUT);
    auto shape=engine->getTensorShape("output");assert(shape.nbDims==2&&shape.d[0]==2&&shape.d[1]==3);
    std::unique_ptr<nvinfer1::IRuntimeConfig> config(engine->createRuntimeConfig());assert(config);
    config->setExecutionContextAllocationStrategy(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED);
    assert(config->getExecutionContextAllocationStrategy()==nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED);
    std::unique_ptr<nvinfer1::ISerializationConfig> options(engine->createSerializationConfig());assert(options&&options->getFlags()==0);
    std::unique_ptr<nvinfer1::IHostMemory> encoded(engine->serializeWithConfig(*options));assert(encoded&&encoded->size()==size&&std::memcmp(encoded->data(),bytes.get(),size)==0);
    std::unique_ptr<nvinfer1::IEngineInspector> inspector(engine->createEngineInspector());assert(inspector);
    assert(std::strcmp(inspector->getLayerInformation(0,nvinfer1::LayerInformationFormat::kONELINE),"output")==0);
    inspector.reset();options.reset();config.reset();engine.reset();encoded.reset();dlclose(library);
    std::cout<<"PASS: independent RTX 1.7.1 reference-header runtime/engine/configuration/serialization/inspector client\n";
}
