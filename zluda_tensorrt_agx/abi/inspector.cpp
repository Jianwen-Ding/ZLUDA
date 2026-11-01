// Inspector for our own graph metadata only. No weight or serialized vendor payload is inspected.
#include "inspector.hpp"
#include "context.hpp"
#include "errors.hpp"
#include <vector>
#include <string>
#include <stdexcept>
namespace {
using namespace nvinfer1;
std::string quote(std::string const& value){
    std::string result="\"";constexpr char hex[]="0123456789abcdef";
    for(unsigned char c:value){
        if(c=='"'||c=='\\'){result+='\\';result+=char(c);}
        else if(c<32){result+="\\u00";result+=hex[c>>4];result+=hex[c&15];}
        else{result+=char(c);}
    }
    result+='"';return result;
}
class InspectorImpl final:public apiv::VEngineInspector {
    ZtrtRecorderSlot errors;
    IEngineInspector* owner;ICudaEngine* engine;ILogger* logger;
    IExecutionContext const* context=nullptr;std::vector<std::string> names;mutable std::string result;
    void error(char const* message) const noexcept{errors.report(logger,message);}
    bool format(LayerInformationFormat f) const noexcept{return int32_t(f)==0||int32_t(f)==1;}
public:
    InspectorImpl(IEngineInspector* o,ICudaEngine* e,ZtrtEngine* native,ILogger* l,IErrorRecorder* recorder):owner(o),engine(e),logger(l){
        errors.set(recorder,l);
        auto count=ztrtGetLayerCount(native);if(count<0)throw std::runtime_error("invalid layer count");names.reserve(size_t(count));
        for(int i=0;i<count;++i){auto name=ztrtGetLayerName(native,i);if(!name)throw std::runtime_error("missing layer name");names.emplace_back(name);}
    }
    IEngineInspector* getPImpl() noexcept override{return owner;}
    bool setExecutionContext(IExecutionContext const* value) noexcept override{
        if(value&&!ztrtAbiContextMatches(value,engine)){error("inspector context belongs to another engine or implementation");return false;}context=value;return true;
    }
    IExecutionContext const* getExecutionContext() const noexcept override{return context;}
    char const* getLayerInformation(int32_t index,LayerInformationFormat f) const noexcept override{
        if(index<0||size_t(index)>=names.size()||!format(f)){error("invalid inspector layer or format");return nullptr;}
        try{result=int32_t(f)==0?names[size_t(index)]:"{\"Name\":"+quote(names[size_t(index)])+"}";return result.c_str();}
        catch(...){error("inspector string allocation failed");return nullptr;}
    }
    char const* getEngineInformation(LayerInformationFormat f) const noexcept override{
        if(!format(f)){error("invalid inspector format");return nullptr;}
        try{
            result=int32_t(f)==1?"{\"Layers\":[":"";
            for(size_t i=0;i<names.size();++i){
                if(i){result+=int32_t(f)==1?",":"\n";}
                result+=int32_t(f)==1?"{\"Name\":"+quote(names[i])+"}":names[i];
            }
            if(int32_t(f)==1){result+="]}";}return result.c_str();
        }catch(...){error("inspector string allocation failed");return nullptr;}
    }
    void setErrorRecorder(IErrorRecorder* p) noexcept override{errors.set(p,logger);}
    IErrorRecorder* getErrorRecorder() const noexcept override{return errors.get();}
};
class Inspector final:public IEngineInspector {
public:
    Inspector(ICudaEngine* engine,ZtrtEngine* native,ILogger* logger,IErrorRecorder* recorder){mImpl=new InspectorImpl(this,engine,native,logger,recorder);}
    ~Inspector() noexcept override{delete mImpl;}
};
}
nvinfer1::IEngineInspector* ztrtAbiCreateInspector(nvinfer1::ICudaEngine* engine,ZtrtEngine* native,nvinfer1::ILogger* logger,nvinfer1::IErrorRecorder* recorder) noexcept{
    try{return new Inspector(engine,native,logger,recorder);}catch(...){if(logger)logger->log(nvinfer1::ILogger::Severity::kERROR,"inspector allocation failed");return nullptr;}
}
