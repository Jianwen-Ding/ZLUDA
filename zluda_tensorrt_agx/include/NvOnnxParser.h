#ifndef ZLUDA_AGX_NVONNXPARSER_H
#define ZLUDA_AGX_NVONNXPARSER_H
// Independent source-level ONNX parser interface. No NVIDIA parser/runtime is loaded.
#include "NvInfer.h"
namespace nvonnxparser {
class IParserError {
    std::string mMessage;
public:
    explicit IParserError(std::string message):mMessage(std::move(message)){}
    char const* desc() const{return mMessage.c_str();}
};
class IParser {
    nvinfer1::INetworkDefinition& mNetwork;
    nvinfer1::ILogger& mLogger;
    std::unique_ptr<IParserError> mError;
public:
    IParser(nvinfer1::INetworkDefinition& network,nvinfer1::ILogger& logger):mNetwork(network),mLogger(logger){}
    bool parse(void const* data,size_t size,char const* modelPath=nullptr){
        (void)modelPath;
        if(ztrtBuilderImportOnnx(mNetwork.native(),static_cast<uint8_t const*>(data),size)){
            mError=std::make_unique<IParserError>(ztrtGetLastError());
            mLogger.log(nvinfer1::ILogger::Severity::kERROR,mError->desc());return false;
        }
        mError.reset();return true;
    }
    int32_t getNbErrors() const{return mError?1:0;}
    IParserError const* getError(int32_t index) const{return index==0?mError.get():nullptr;}
    void clearErrors(){mError.reset();}
};
inline IParser* createParser(nvinfer1::INetworkDefinition& network,nvinfer1::ILogger& logger){return new IParser(network,logger);}
}
#endif
