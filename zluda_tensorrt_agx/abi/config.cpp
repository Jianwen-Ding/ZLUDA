// Original static-engine configuration implementation; public dispatch declarations are in runtime_contract.hpp.
#include "config.hpp"
namespace {
using namespace nvinfer1;
class ConfigImpl final:public apiv::VRuntimeConfig {
public:
    IRuntimeConfig* owner;ICudaEngine* engine;ILogger* logger;int32_t allocation=0,specialization=0;
    bool allocationValid=true,specializationValid=true;
    ConfigImpl(IRuntimeConfig* o,ICudaEngine* e,ILogger* l):owner(o),engine(e),logger(l){}
    void error(char const* text) const noexcept{if(logger)logger->log(ILogger::Severity::kERROR,text);}
    IRuntimeConfig* getPImpl() noexcept override{return owner;}
    void setExecutionContextAllocationStrategy(ExecutionContextAllocationStrategy value) noexcept override{
        auto v=int32_t(value);allocationValid=v==0||v==2;
        if(allocationValid){allocation=v;}else{error("only static and user-managed allocation are supported");}
    }
    ExecutionContextAllocationStrategy getExecutionContextAllocationStrategy() const noexcept override{return static_cast<ExecutionContextAllocationStrategy>(allocation);}
    IRuntimeCache* createRuntimeCache() const noexcept override{error("runtime cache objects are unsupported");return nullptr;}
    bool setRuntimeCache(IRuntimeCache const&) noexcept override{error("runtime cache objects are unsupported");return false;}
    IRuntimeCache* getRuntimeCache() const noexcept override{return nullptr;}
    void setDynamicShapesKernelSpecializationStrategy(DynamicShapesKernelSpecializationStrategy value) noexcept override{
        auto v=int32_t(value);specializationValid=v>=0&&v<=2;
        if(specializationValid){specialization=v;}else{error("invalid dynamic-shape specialization strategy");}
    }
    DynamicShapesKernelSpecializationStrategy getDynamicShapesKernelSpecializationStrategy() const noexcept override{return static_cast<DynamicShapesKernelSpecializationStrategy>(specialization);}
    bool setCudaGraphStrategy(CudaGraphStrategy value) noexcept override{
        if(int32_t(value)!=0){error("automatic whole-graph capture is unsupported");return false;}return true;
    }
    CudaGraphStrategy getCudaGraphStrategy() const noexcept override{return CudaGraphStrategy::kDISABLED;}
};
class Config final:public IRuntimeConfig {
public:
    Config(ICudaEngine* e,ILogger* l,int32_t strategy){mImpl=new ConfigImpl(this,e,l);mImpl->setExecutionContextAllocationStrategy(static_cast<ExecutionContextAllocationStrategy>(strategy));}
    ~Config() noexcept override{delete mImpl;}
};
}
nvinfer1::IRuntimeConfig* ztrtAbiCreateConfig(nvinfer1::ICudaEngine* engine,nvinfer1::ILogger* logger,int32_t strategy) noexcept{
    try{return new Config(engine,logger,strategy);}catch(...){if(logger)logger->log(nvinfer1::ILogger::Severity::kERROR,"runtime config allocation failed");return nullptr;}
}
nvinfer1::IRuntimeConfig* ztrtAbiCloneConfig(nvinfer1::IRuntimeConfig* source,nvinfer1::ICudaEngine* engine,nvinfer1::ILogger* logger) noexcept{
    if(!source)return ztrtAbiCreateConfig(engine,logger);
    auto impl=dynamic_cast<ConfigImpl*>(source->mImpl);
    if(!impl||impl->engine!=engine||!impl->allocationValid||!impl->specializationValid){if(logger)logger->log(nvinfer1::ILogger::Severity::kERROR,"invalid or foreign runtime configuration");return nullptr;}
    auto result=ztrtAbiCreateConfig(engine,logger,impl->allocation);
    if(result)result->mImpl->setDynamicShapesKernelSpecializationStrategy(static_cast<nvinfer1::DynamicShapesKernelSpecializationStrategy>(impl->specialization));
    return result;
}
