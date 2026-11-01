// Original runtime-factory/configuration prototype. Engine ABI is intentionally not yet exposed.
#include "runtime_contract.hpp"
#include <atomic>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
namespace {
using namespace nvinfer1;
constexpr int32_t kContractVersion=10701; // RTX 1.7.1 public documentation snapshot, not Geekbench 1.6.
void error(ILogger* logger,char const* message) noexcept {if(logger)logger->log(ILogger::Severity::kERROR,message);}
class EmptyRegistry final:public IPluginRegistry {
    ILogger* logger;std::atomic<bool> search{true};
public:
    explicit EmptyRegistry(ILogger* p):logger(p){}
    void setErrorRecorder(IErrorRecorder* const p) noexcept override{if(p)error(logger,"ABI prototype: error recorders are unsupported");}
    IErrorRecorder* getErrorRecorder() const noexcept override{return nullptr;}
    bool isParentSearchEnabled() const override{return search.load();}
    void setParentSearchEnabled(bool const enabled) override{search=enabled;}
    PluginLibraryHandle loadLibrary(AsciiChar const*) noexcept override{error(logger,"ABI prototype: plugin loading is unsupported");return nullptr;}
    void deregisterLibrary(PluginLibraryHandle p) noexcept override{if(p)error(logger,"ABI prototype: no plugin libraries are loaded");}
    bool registerCreator(IPluginCreatorInterface&,AsciiChar const* const) noexcept override{error(logger,"ABI prototype: plugin registration is unsupported");return false;}
    IPluginCreatorInterface* const* getAllCreators(int32_t* const count) const noexcept override{if(count)*count=0;return nullptr;}
    IPluginCreatorInterface* getCreator(AsciiChar const* const,AsciiChar const* const,AsciiChar const* const) noexcept override{return nullptr;}
    bool deregisterCreator(IPluginCreatorInterface const&) noexcept override{return false;}
    IPluginResource* acquirePluginResource(AsciiChar const*,IPluginResource*) noexcept override{error(logger,"ABI prototype: plugin resources are unsupported");return nullptr;}
    int32_t releasePluginResource(AsciiChar const*) noexcept override{return -1;}
    IPluginCreatorInterface* const* getAllCreatorsRecursive(int32_t* const count) noexcept override{if(count)*count=0;return nullptr;}
};
class RuntimeImpl final:public apiv::VRuntime {
    IRuntime* owner;ILogger* logger;EmptyRegistry registry;
    std::atomic<int32_t> threads{1};std::atomic<uint32_t> flags{3};
    mutable std::mutex directoryMutex;std::string directory;
public:
    RuntimeImpl(IRuntime* o,ILogger* l):owner(o),logger(l),registry(l){}
    IRuntime* getPImpl() noexcept override{return owner;}
    ICudaEngine* deserializeCudaEngine(void const*,std::size_t) noexcept override{error(logger,"ABI prototype: engine dispatch is not implemented; use the native ZTRT API");return nullptr;}
    void setDLACore(int32_t core) noexcept override{if(core!=0)error(logger,"ABI prototype: DLA is unavailable");}
    int32_t getDLACore() const noexcept override{return 0;}
    int32_t getNbDLACores() const noexcept override{return 0;}
    void setGpuAllocator(IGpuAllocator* p) noexcept override{if(p)error(logger,"ABI prototype: custom allocators are unsupported");}
    void setErrorRecorder(IErrorRecorder* p) noexcept override{if(p)error(logger,"ABI prototype: error recorders are unsupported");}
    IErrorRecorder* getErrorRecorder() const noexcept override{return nullptr;}
    ILogger* getLogger() const noexcept override{return logger;}
    bool setMaxThreads(int32_t n) noexcept override{if(n<=0){error(logger,"maxThreads must be positive");return false;}threads=n;return true;}
    int32_t getMaxThreads() const noexcept override{return threads.load();}
    void setTemporaryDirectory(char const* path) noexcept override{try{std::lock_guard<std::mutex> lock(directoryMutex);directory=path?path:"";}catch(...){error(logger,"temporary directory allocation failed");}}
    char const* getTemporaryDirectory() const noexcept override{try{std::lock_guard<std::mutex> lock(directoryMutex);return directory.empty()?nullptr:directory.c_str();}catch(...){error(logger,"temporary directory lock failed");return nullptr;}}
    void setTempfileControlFlags(TempfileControlFlags f) noexcept override{if(f&~3U){error(logger,"invalid tempfile control flags");return;}flags=f;}
    TempfileControlFlags getTempfileControlFlags() const noexcept override{return flags.load();}
    IPluginRegistry& getPluginRegistry() noexcept override{return registry;}
    void setPluginRegistryParent(IPluginRegistry* p) noexcept override{if(p)error(logger,"ABI prototype: parent plugin registries are unsupported");}
    IRuntime* loadRuntime(char const*) noexcept override{error(logger,"ABI prototype: external runtime loading is unsupported");return nullptr;}
    void setEngineHostCodeAllowed(bool allowed) noexcept override{if(allowed)error(logger,"ABI prototype: engine host code is unsupported");}
    bool getEngineHostCodeAllowed() const noexcept override{return false;}
    ICudaEngine* deserializeCudaEngineV2(IStreamReaderV2&) noexcept override{error(logger,"ABI prototype: stream engine loading is unsupported");return nullptr;}
    int64_t getEngineHeaderSize() const noexcept override{return 0;}
    EngineValidity getEngineValidity(void const*,int64_t,uint64_t* diagnostics) const noexcept override{if(diagnostics)*diagnostics=0;return EngineValidity::kINVALID;}
    void setDeferredWeightsLoading(bool enabled) noexcept override{if(enabled)error(logger,"ABI prototype: deferred weights are unsupported");}
    bool getDeferredWeightsLoading() const noexcept override{return false;}
    bool setDLAWorkspaceAllocationStrategy(DLAWorkspaceAllocationStrategy) noexcept override{error(logger,"ABI prototype: DLA is unavailable");return false;}
    DLAWorkspaceAllocationStrategy getDLAWorkspaceAllocationStrategy() const noexcept override{return static_cast<DLAWorkspaceAllocationStrategy>(0);}
};
class Runtime final:public IRuntime {
public:
    explicit Runtime(ILogger* l){mImpl=new RuntimeImpl(this,l);}
    ~Runtime() noexcept override{delete mImpl;}
};
}
extern "C" __attribute__((visibility("default"))) int32_t getInferLibVersion() noexcept{return kContractVersion;}
extern "C" __attribute__((visibility("default"))) void* createInferRuntime_INTERNAL(void* opaque,int32_t version) noexcept{
    auto logger=static_cast<nvinfer1::ILogger*>(opaque);
    if(!logger)return nullptr;
    auto info=logger->getInterfaceInfo();
    if(logger->getAPILanguage()!=nvinfer1::APILanguage::kCPP||!info.kind||std::strcmp(info.kind,"ILogger")||info.major!=1||info.minor!=0)return nullptr;
    if(version!=kContractVersion){error(logger,"ABI prototype: only the reviewed RTX 1.7.1 contract is accepted");return nullptr;}
    try{return new Runtime(logger);}catch(...){error(logger,"runtime allocation failed");return nullptr;}
}
