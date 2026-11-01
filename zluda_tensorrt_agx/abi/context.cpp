// Original static-f32 execution adapter. Unsupported features fail without borrowing unchecked native objects.
#include "context.hpp"
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
namespace {
using namespace nvinfer1;
void error(ILogger* l,char const* text) noexcept{if(l)l->log(ILogger::Severity::kERROR,text);}
struct Tensor {std::string name;Dims shape;size_t bytes;int mode;void* address=nullptr;};
class ContextImpl final:public apiv::VExecutionContext {
    IExecutionContext* owner;ICudaEngine* engine;ZtrtContext* native;ILogger* logger;
    std::vector<Tensor> tensors;std::string name;size_t workspace;bool debug=false,emitsProfile=true,externalMemory=false,warmed=false;CUstream_st* lastStream=nullptr;
    Tensor* find(char const* key) noexcept{if(key)for(auto& t:tensors)if(t.name==key)return &t;return nullptr;}
    Tensor const* find(char const* key) const noexcept{if(key)for(auto const& t:tensors)if(t.name==key)return &t;return nullptr;}
    Dims invalid() const noexcept{Dims d;d.nbDims=-1;return d;}
    bool ready() noexcept{if(externalMemory){error(logger,"context workspace configuration is invalid");return false;}for(auto const& t:tensors)if(!t.address){error(logger,"I/O tensor is unbound");return false;}return true;}
    bool status(int32_t rc) noexcept{if(rc!=0){error(logger,ztrtGetLastError());return false;}return true;}
public:
    ContextImpl(IExecutionContext* o,ICudaEngine* e,ZtrtContext* c,ZtrtEngine* model,ILogger* l):owner(o),engine(e),native(c),logger(l),workspace(ztrtGetWorkspaceSize(model)){
        int n=ztrtGetNbIOTensors(model);if(n<0)throw std::runtime_error("invalid native engine I/O");tensors.reserve(size_t(n));
        for(int i=0;i<n;++i){auto key=ztrtGetIOTensorName(model,i);if(!key)throw std::runtime_error("missing I/O name");Dims shape;shape.nbDims=ztrtGetTensorShape(model,key,shape.d,8);if(shape.nbDims<0)throw std::runtime_error("invalid I/O shape");tensors.push_back(Tensor{key,shape,ztrtGetTensorBytes(model,key),ztrtGetTensorIOMode(model,key),nullptr});}
    }
    ~ContextImpl() noexcept override{ztrtDestroyExecutionContext(native);}
    IExecutionContext* getPImpl() noexcept override{return owner;}
    void setDebugSync(bool enabled) noexcept override{debug=enabled;}
    bool getDebugSync() const noexcept override{return debug;}
    void setProfiler(IProfiler* p) noexcept override{if(p)error(logger,"ABI prototype: profiling callbacks are unsupported");}
    IProfiler* getProfiler() const noexcept override{return nullptr;}
    ICudaEngine const& getEngine() const noexcept override{return *engine;}
    void setName(char const* value) noexcept override{try{name=value?value:"";}catch(...){error(logger,"context name allocation failed");}}
    char const* getName() const noexcept override{return name.c_str();}
    void setDeviceMemory(void* p) noexcept override{setDeviceMemoryV2(p,p?int64_t(workspace):0);}
    int32_t getOptimizationProfile() const noexcept override{return 0;}
    bool allInputDimensionsSpecified() const noexcept override{return true;}
    void setErrorRecorder(IErrorRecorder* p) noexcept override{if(p)error(logger,"ABI prototype: error recorders are unsupported");}
    IErrorRecorder* getErrorRecorder() const noexcept override{return nullptr;}
    bool executeV2(void* const* bindings) noexcept override{if(!bindings)return false;for(size_t i=0;i<tensors.size();++i)if(!setTensorAddress(tensors[i].name.c_str(),bindings[i]))return false;return enqueueV3(nullptr)&&status(ztrtSynchronize(native));}
    bool setOptimizationProfileAsync(int32_t p,CUstream_st*) noexcept override{if(p!=0){error(logger,"only static optimization profile 0 is supported");return false;}return true;}
    void setEnqueueEmitsProfile(bool enabled) noexcept override{emitsProfile=enabled;}
    bool getEnqueueEmitsProfile() const noexcept override{return emitsProfile;}
    bool reportToProfiler() const noexcept override{return false;}
    bool setInputShape(char const* key,Dims const& d) noexcept override{auto t=find(key);if(!t||t->mode!=0||d.nbDims!=t->shape.nbDims)return false;for(int i=0;i<d.nbDims;++i)if(d.d[i]!=t->shape.d[i]){error(logger,"dynamic shape changes are unsupported");return false;}return true;}
    Dims getTensorShape(char const* key) const noexcept override{auto t=find(key);return t?t->shape:invalid();}
    Dims getTensorStrides(char const* key) const noexcept override{auto t=find(key);if(!t)return invalid();Dims result;result.nbDims=t->shape.nbDims;int64_t stride=1;for(int i=result.nbDims-1;i>=0;--i){result.d[i]=stride;stride*=t->shape.d[i];}return result;}
    bool setTensorAddress(char const* key,void* p) noexcept override{auto t=find(key);if(!t)return false;if(p&&!status(ztrtSetTensorAddress(native,key,p,t->bytes)))return false;t->address=p;return true;}
    void const* getTensorAddress(char const* key) const noexcept override{auto t=find(key);return t?t->address:nullptr;}
    bool setInputTensorAddress(char const* key,void const* p) noexcept override{auto t=find(key);return t&&t->mode==0&&setTensorAddress(key,const_cast<void*>(p));}
    bool setOutputTensorAddress(char const* key,void* p) noexcept override{auto t=find(key);return t&&t->mode==1&&setTensorAddress(key,p);}
    int32_t inferShapes(int32_t maxNames,char const**) noexcept override{return maxNames<0?-1:0;}
    bool setInputConsumedEvent(CUevent_st* p) noexcept override{if(p){error(logger,"ABI prototype: input-consumed events are unsupported");return false;}return true;}
    CUevent_st* getInputConsumedEvent() const noexcept override{return nullptr;}
    void* getOutputTensorAddress(char const* key) const noexcept override{auto t=find(key);return t&&t->mode==1?t->address:nullptr;}
    bool setOutputAllocator(char const* key,IOutputAllocator* p) noexcept override{auto t=find(key);if(!t||t->mode!=1)return false;if(p){error(logger,"ABI prototype: output allocators are unsupported");return false;}return true;}
    IOutputAllocator* getOutputAllocator(char const*) noexcept override{return nullptr;}
    int64_t getMaxOutputSize(char const* key) const noexcept override{auto t=find(key);return t&&t->mode==1?int64_t(t->bytes):-1;}
    bool setTemporaryStorageAllocator(IGpuAllocator* p) noexcept override{if(p){error(logger,"ABI prototype: temporary allocators are unsupported");return false;}return true;}
    IGpuAllocator* getTemporaryStorageAllocator() const noexcept override{return nullptr;}
    bool enqueueV3(CUstream_st* stream) noexcept override{if(!ready()||!status(ztrtEnqueueV3(native,stream)))return false;warmed=true;lastStream=stream;return true;}
    void setPersistentCacheLimit(size_t size) noexcept override{if(size)error(logger,"ABI prototype: persistent cache controls are unsupported");}
    size_t getPersistentCacheLimit() const noexcept override{return 0;}
    bool setNvtxVerbosity(ProfilingVerbosity v) noexcept override{return int32_t(v)==1;}
    ProfilingVerbosity getNvtxVerbosity() const noexcept override{return static_cast<ProfilingVerbosity>(1);}
    void setAuxStreams(CUstream_st**,int32_t count) noexcept override{if(count)error(logger,"ABI prototype: auxiliary streams are unsupported");}
    bool setDebugListener(IDebugListener* p) noexcept override{return p==nullptr;}
    IDebugListener* getDebugListener() noexcept override{return nullptr;}
    bool setTensorDebugState(char const* key,bool enabled) noexcept override{return find(key)&&!enabled;}
    bool getDebugState(char const*) const noexcept override{return false;}
    bool setAllTensorsDebugState(bool enabled) noexcept override{return !enabled;}
    size_t updateDeviceMemorySizeForShapes() noexcept override{return workspace;}
    void setDeviceMemoryV2(void* p,int64_t size) noexcept override{
        externalMemory=size<0;
        if(externalMemory){error(logger,"negative workspace size");return;}
        externalMemory=!status(ztrtSetDeviceMemory(native,p,size_t(size)));
        warmed=false;lastStream=nullptr;
    }
    IRuntimeConfig* getRuntimeConfig() const noexcept override{return nullptr;}
    bool setUnfusedTensorsDebugState(bool enabled) noexcept override{return !enabled;}
    bool getUnfusedTensorsDebugState() const noexcept override{return false;}
    bool isStreamCapturable(CUstream_st* stream) const noexcept override{if(!stream||!warmed||lastStream!=stream||externalMemory)return false;for(auto const& t:tensors)if(!t.address)return false;return true;}
    bool setCommunicator(void* p) noexcept override{return p==nullptr;}
};
class Context final:public IExecutionContext {
public:
    Context(ICudaEngine* engine,ZtrtEngine* model,ILogger* logger,bool managed){
        std::unique_ptr<ZtrtContext,decltype(&ztrtDestroyExecutionContext)> native(managed?ztrtCreateExecutionContextWithoutDeviceMemory(model):ztrtCreateExecutionContext(model),ztrtDestroyExecutionContext);
        if(!native)throw std::runtime_error(ztrtGetLastError());
        mImpl=new ContextImpl(this,engine,native.get(),model,logger);native.release();
    }
    ~Context() noexcept override{delete mImpl;}
};
}
nvinfer1::IExecutionContext* ztrtAbiCreateContext(nvinfer1::ICudaEngine* engine,ZtrtEngine* model,nvinfer1::ILogger* logger,bool managed) noexcept{
    try{return new Context(engine,model,logger,managed);}catch(std::exception const& e){error(logger,e.what());return nullptr;}catch(...){error(logger,"context adapter allocation failed");return nullptr;}
}
