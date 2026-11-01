// Original native-engine to public dispatch adapter. No NVIDIA engine decoder or runtime code.
#include "runtime_contract.hpp"
#include "extensions.h"
#include "context.hpp"
#include "config.hpp"
#include "inspector.hpp"
#include <atomic>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <memory>
namespace {
using namespace nvinfer1;
void error(ILogger* l,char const* text) noexcept{if(l)l->log(ILogger::Severity::kERROR,text);}
class MemoryImpl final:public apiv::VHostMemory {
    std::vector<uint8_t> bytes;
public:
    explicit MemoryImpl(ZtrtEngine* engine){auto n=ztrtSerializeEngine(engine,nullptr,0);if(!n)throw std::runtime_error("native serialization failed");bytes.resize(n);if(ztrtSerializeEngine(engine,bytes.data(),n)!=n)throw std::runtime_error("native serialization changed size");}
    void* data() const noexcept override{return const_cast<uint8_t*>(bytes.data());}
    size_t size() const noexcept override{return bytes.size();}
    DataType type() const noexcept override{return DataType::kINT8;}
};
class HostMemory final:public IHostMemory {
public:explicit HostMemory(ZtrtEngine* e){mImpl=new MemoryImpl(e);}~HostMemory() noexcept override{delete mImpl;}
};
class EngineImpl final:public apiv::VCudaEngine {
    ICudaEngine* owner;ZtrtEngine* native;ILogger* logger;bool owns=false;std::atomic<bool> prepared{false};
    bool valid(char const* name) const noexcept{return name&&ztrtGetTensorIOMode(native,name)>=0;}
    Dims invalid() const noexcept{Dims d;d.nbDims=-1;return d;}
public:
    EngineImpl(ICudaEngine* o,ZtrtEngine* e,ILogger* l):owner(o),native(e),logger(l){}
    ~EngineImpl() noexcept override{if(owns)ztrtDestroyEngine(native);}
    void claim() noexcept{owns=true;}
    ICudaEngine* getPImpl() noexcept override{return owner;}
    int32_t getNbLayers() const noexcept override{return ztrtGetLayerCount(native);}
    IHostMemory* serialize() const noexcept override{try{return new HostMemory(native);}catch(...){error(logger,"native engine serialization failed");return nullptr;}}
    IExecutionContext* createExecutionContext(ExecutionContextAllocationStrategy strategy) noexcept override{if(int32_t(strategy)!=0&&int32_t(strategy)!=2){error(logger,"only static and user-managed context allocation are supported");return nullptr;}auto context=ztrtAbiCreateContext(owner,native,logger,int32_t(strategy)==2);if(context)prepared=true;return context;}
    bool isRefittable() const noexcept override{return false;}
    char const* getName() const noexcept override{return "ZLUDA AGX native engine";}
    int32_t getNbOptimizationProfiles() const noexcept override{return 1;}
    EngineCapability getEngineCapability() const noexcept override{return static_cast<EngineCapability>(0);}
    void setErrorRecorder(IErrorRecorder* p) noexcept override{if(p)error(logger,"ABI prototype: error recorders are unsupported");}
    IErrorRecorder* getErrorRecorder() const noexcept override{return nullptr;}
    uint32_t getTacticSources() const noexcept override{return 0;}
    ProfilingVerbosity getProfilingVerbosity() const noexcept override{return static_cast<ProfilingVerbosity>(0);}
    IEngineInspector* createEngineInspector() const noexcept override{return ztrtAbiCreateInspector(owner,native,logger);}
    Dims getTensorShape(char const* name) const noexcept override{if(!valid(name))return invalid();Dims d;d.nbDims=ztrtGetTensorShape(native,name,d.d,8);return d;}
    DataType getTensorDataType(char const*) const noexcept override{return DataType::kFLOAT;}
    TensorLocation getTensorLocation(char const*) const noexcept override{return static_cast<TensorLocation>(0);}
    bool isShapeInferenceIO(char const*) const noexcept override{return false;}
    TensorIOMode getTensorIOMode(char const* name) const noexcept override{int mode=name?ztrtGetTensorIOMode(native,name):-1;return mode==0?TensorIOMode::kINPUT:mode==1?TensorIOMode::kOUTPUT:TensorIOMode::kNONE;}
    int32_t getTensorBytesPerComponent(char const*) const noexcept override{return -1;}
    int32_t getTensorComponentsPerElement(char const*) const noexcept override{return -1;}
    TensorFormat getTensorFormat(char const*) const noexcept override{return static_cast<TensorFormat>(0);}
    char const* getTensorFormatDesc(char const* name) const noexcept override{return valid(name)?"Row-major contiguous float32":"";}
    int32_t getTensorVectorizedDim(char const*) const noexcept override{return -1;}
    Dims getProfileShape(char const* name,int32_t profile,OptProfileSelector select) const noexcept override{return profile==0&&int32_t(select)>=0&&int32_t(select)<=2?getTensorShape(name):invalid();}
    int32_t getNbIOTensors() const noexcept override{return ztrtGetNbIOTensors(native);}
    char const* getIOTensorName(int32_t index) const noexcept override{return ztrtGetIOTensorName(native,index);}
    HardwareCompatibilityLevel getHardwareCompatibilityLevel() const noexcept override{return static_cast<HardwareCompatibilityLevel>(0);}
    int32_t getNbAuxStreams() const noexcept override{return 0;}
    int32_t getTensorBytesPerComponentV2(char const* n,int32_t p) const noexcept override{return p==0?getTensorBytesPerComponent(n):-1;}
    int32_t getTensorComponentsPerElementV2(char const* n,int32_t p) const noexcept override{return p==0?getTensorComponentsPerElement(n):-1;}
    TensorFormat getTensorFormatV2(char const* n,int32_t p) const noexcept override{return p==0?getTensorFormat(n):static_cast<TensorFormat>(0);}
    char const* getTensorFormatDescV2(char const* n,int32_t p) const noexcept override{return p==0?getTensorFormatDesc(n):"";}
    int32_t getTensorVectorizedDimV2(char const*,int32_t) const noexcept override{return -1;}
    ISerializationConfig* createSerializationConfig() noexcept override{return ztrtAbiCreateSerializationConfig(logger);}
    IHostMemory* serializeWithConfig(ISerializationConfig& config) const noexcept override{
        uint32_t flags=0;if(!ztrtAbiGetSerializationFlags(config,flags)){error(logger,"foreign serialization configuration");return nullptr;}
        if(flags&5U){error(logger,"weight exclusion and refit serialization are unsupported");return nullptr;}
        // Our format never embeds a lean runtime, so excluding it needs no payload change.
        return serialize();
    }
    IRefitter* createRefitter(ILogger&) noexcept override{error(logger,"ABI prototype: refitting is unsupported");return nullptr;}
    int64_t getStreamableWeightsSize() const noexcept override{return 0;}
    bool isDebugTensor(char const*) const noexcept override{return false;}
    bool setWeightStreamingBudgetV2(int64_t) noexcept override{error(logger,"ABI prototype: weight streaming is unsupported");return false;}
    int64_t getWeightStreamingBudgetV2() const noexcept override{return 0;}
    int64_t getWeightStreamingAutomaticBudget() const noexcept override{return 0;}
    int64_t getWeightStreamingScratchMemorySize() const noexcept override{return 0;}
    int64_t getDeviceMemorySizeV2() const noexcept override{return int64_t(ztrtGetWorkspaceSize(native));}
    int64_t getDeviceMemorySizeForProfileV2(int32_t p) const noexcept override{return p==0?getDeviceMemorySizeV2():-1;}
    int64_t const* getProfileTensorValuesV2(char const*,int32_t,OptProfileSelector) const noexcept override{return nullptr;}
    IExecutionContext* createExecutionContextWithRuntimeConfig(IRuntimeConfig* config) noexcept override{
        std::unique_ptr<IRuntimeConfig> copy(ztrtAbiCloneConfig(config,owner,logger));if(!copy)return nullptr;
        bool managed=int32_t(copy->mImpl->getExecutionContextAllocationStrategy())==2;
        auto context=ztrtAbiCreateContext(owner,native,logger,managed,copy.get());if(context)prepared=true;return context;
    }
    IRuntimeConfig* createRuntimeConfig() noexcept override{return ztrtAbiCreateConfig(owner,logger);}
    int64_t getEngineStat(EngineStat) const noexcept override{error(logger,"ABI prototype: engine statistics are unsupported");return -1;}
    char const* getAliasedInputTensor(char const*) const noexcept override{return nullptr;}
    bool areWeightsLoaded() const noexcept override{return prepared.load();}
    bool loadWeights(void const*,int64_t) noexcept override{error(logger,"ABI prototype: weight loading is unsupported");return false;}
    bool loadWeightsAsync(IStreamReaderV2&,CUstream_st*) noexcept override{error(logger,"ABI prototype: weight loading is unsupported");return false;}
    bool unloadWeights() noexcept override{error(logger,"ABI prototype: weight unloading is unsupported");return false;}
    IWeightsManager* createWeightsManager() noexcept override{error(logger,"ABI prototype: weight management is unsupported");return nullptr;}
};
class Engine final:public ICudaEngine {
public:
    Engine(ZtrtEngine* e,ILogger* l){mImpl=new EngineImpl(this,e,l);}
    void claim() noexcept{static_cast<EngineImpl*>(mImpl)->claim();}
    ~Engine() noexcept override{delete mImpl;}
};
}
// Ownership of a valid native engine transfers only on a successful return. Logger must outlive the wrapper.
extern "C" __attribute__((visibility("default"))) void* ztrtRtxWrapNativeEngine(ZtrtEngine* native,void* opaqueLogger) noexcept{
    if(!native||!opaqueLogger)return nullptr;
    auto logger=static_cast<nvinfer1::ILogger*>(opaqueLogger);auto info=logger->getInterfaceInfo();
    if(logger->getAPILanguage()!=nvinfer1::APILanguage::kCPP||!info.kind||std::strcmp(info.kind,"ILogger")||info.major!=1||info.minor!=0)return nullptr;
    try{auto e=new Engine(static_cast<ZtrtEngine*>(native),static_cast<nvinfer1::ILogger*>(opaqueLogger));e->claim();return e;}
    catch(...){error(static_cast<nvinfer1::ILogger*>(opaqueLogger),"engine adapter allocation failed");return nullptr;}
}
