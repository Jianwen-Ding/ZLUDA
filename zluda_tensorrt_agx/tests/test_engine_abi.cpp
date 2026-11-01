// Independently declared engine dispatch-prefix probe; no adapter contract header included.
#include "NvInferRuntimeBase.h"
#include "zluda_tensorrt.h"
#include <cassert>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace probe {
using nvinfer1::Dims;using nvinfer1::DataType;using nvinfer1::TensorIOMode;
struct Engine;struct HostMemory;
class HostDispatch {public:virtual ~HostDispatch() noexcept=default;virtual void* data() const noexcept=0;virtual size_t size() const noexcept=0;virtual DataType type() const noexcept=0;};
struct HostMemory {virtual ~HostMemory() noexcept=default;HostDispatch* impl;};
class EngineDispatch {
public:
    virtual ~EngineDispatch() noexcept=default;
    virtual Engine* getPImpl() noexcept=0;
    virtual int32_t getNbLayers() const noexcept=0;
    virtual HostMemory* serialize() const noexcept=0;
    virtual void* createExecutionContext(int32_t) noexcept=0;
    virtual bool isRefittable() const noexcept=0;
    virtual char const* getName() const noexcept=0;
    virtual int32_t getNbOptimizationProfiles() const noexcept=0;
    virtual int32_t getEngineCapability() const noexcept=0;
    virtual void setErrorRecorder(void*) noexcept=0;
    virtual void* getErrorRecorder() const noexcept=0;
    virtual uint32_t getTacticSources() const noexcept=0;
    virtual int32_t getProfilingVerbosity() const noexcept=0;
    virtual void* createEngineInspector() const noexcept=0;
    virtual Dims getTensorShape(char const*) const noexcept=0;
    virtual DataType getTensorDataType(char const*) const noexcept=0;
    virtual int32_t getTensorLocation(char const*) const noexcept=0;
    virtual bool isShapeInferenceIO(char const*) const noexcept=0;
    virtual TensorIOMode getTensorIOMode(char const*) const noexcept=0;
    virtual int32_t getTensorBytesPerComponent(char const*) const noexcept=0;
    virtual int32_t getTensorComponentsPerElement(char const*) const noexcept=0;
    virtual int32_t getTensorFormat(char const*) const noexcept=0;
    virtual char const* getTensorFormatDesc(char const*) const noexcept=0;
    virtual int32_t getTensorVectorizedDim(char const*) const noexcept=0;
    virtual Dims getProfileShape(char const*,int32_t,int32_t) const noexcept=0;
    virtual int32_t getNbIOTensors() const noexcept=0;
    virtual char const* getIOTensorName(int32_t) const noexcept=0;
    virtual int32_t getHardwareCompatibilityLevel() const noexcept=0;
    virtual int32_t getNbAuxStreams() const noexcept=0;
    virtual int32_t getTensorBytesPerComponentV2(char const*,int32_t) const noexcept=0;
    virtual int32_t getTensorComponentsPerElementV2(char const*,int32_t) const noexcept=0;
    virtual int32_t getTensorFormatV2(char const*,int32_t) const noexcept=0;
    virtual char const* getTensorFormatDescV2(char const*,int32_t) const noexcept=0;
    virtual int32_t getTensorVectorizedDimV2(char const*,int32_t) const noexcept=0;
    virtual void* createSerializationConfig() noexcept=0;
    virtual HostMemory* serializeWithConfig(void*) const noexcept=0;
    virtual void* createRefitter(void*) noexcept=0;
    virtual int64_t getStreamableWeightsSize() const noexcept=0;
    virtual bool isDebugTensor(char const*) const noexcept=0;
    virtual bool setWeightStreamingBudgetV2(int64_t) noexcept=0;
    virtual int64_t getWeightStreamingBudgetV2() const noexcept=0;
    virtual int64_t getWeightStreamingAutomaticBudget() const noexcept=0;
    virtual int64_t getWeightStreamingScratchMemorySize() const noexcept=0;
    virtual int64_t getDeviceMemorySizeV2() const noexcept=0;
    virtual int64_t getDeviceMemorySizeForProfileV2(int32_t) const noexcept=0;
};
struct Engine {virtual ~Engine() noexcept=default;EngineDispatch* impl;};
}
struct Logger final:nvinfer1::ILogger{int errors=0;void log(Severity,char const*)noexcept override{++errors;}};
int main(int argc,char** argv){assert(argc==2);void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib)throw std::runtime_error(dlerror());
    auto wrap=reinterpret_cast<void*(*)(void*,void*)>(dlsym(lib,"ztrtRtxWrapNativeEngine"));assert(wrap);Logger logger;assert(!wrap(nullptr,&logger));
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
        assert(!e->createExecutionContext(0)&&!e->createSerializationConfig()&&!e->setWeightStreamingBudgetV2(1));
        auto memory=e->serialize();assert(memory);delete engine; // Returned custom bytes outlive the engine wrapper.
        auto h=memory->impl;assert(h->type()==nvinfer1::DataType::kINT8&&h->size()>24&&std::memcmp(h->data(),"ZTRTAGX2",8)==0);
        auto runtime=ztrtCreateRuntime();assert(runtime);auto restored=ztrtDeserializeEngine(runtime,static_cast<uint8_t const*>(h->data()),h->size(),4096);assert(restored);
        assert(ztrtGetConstantBytes(restored)==24);std::vector<uint8_t> copy(h->size());
        assert(ztrtSerializeEngine(restored,copy.data(),copy.size())==copy.size()&&std::memcmp(copy.data(),h->data(),copy.size())==0);
        ztrtDestroyRuntime(runtime);auto again=static_cast<probe::Engine*>(wrap(restored,&logger));assert(again&&again->impl->getNbIOTensors()==2);delete again;delete memory;
    }
    assert(logger.errors>=300);dlclose(lib);std::cout<<"PASS: engine metadata dispatch, native serialization ownership, rejected paths and 100 roundtrips\n";
}
