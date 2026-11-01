#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iterator>
#include <memory>
struct Logger final:nvinfer1::ILogger{int errors=0;void log(Severity,char const*)noexcept override{++errors;}};
static_assert(int(nvinfer1::TensorIOMode::kNONE)==0&&int(nvinfer1::TensorIOMode::kINPUT)==1&&int(nvinfer1::TensorIOMode::kOUTPUT)==2);
static_assert(int(nvinfer1::ElementWiseOperation::kMAX)==2&&int(nvinfer1::ElementWiseOperation::kMIN)==3&&int(nvinfer1::ElementWiseOperation::kSUB)==4&&int(nvinfer1::ElementWiseOperation::kDIV)==5);
static_assert(int(nvinfer1::BuilderFlag::kTF32)==6);
static_assert(sizeof(nvinfer1::Dims)==72&&offsetof(nvinfer1::Dims,d)==8);
static_assert(sizeof(nvinfer1::InterfaceInfo)==16&&offsetof(nvinfer1::InterfaceInfo,major)==8);
static_assert(sizeof(nvinfer1::Weights)==24&&offsetof(nvinfer1::Weights,values)==8&&offsetof(nvinfer1::Weights,count)==16);
int main(int argc,char** argv){
    using namespace nvinfer1;Logger logger;
    IVersionedInterface* versioned=&logger;
    assert(versioned->getAPILanguage()==APILanguage::kCPP);
    auto info=versioned->getInterfaceInfo();assert(std::strcmp(info.kind,"ILogger")==0&&info.major==1&&info.minor==0);
    assert(tensorIOMode(-1)==TensorIOMode::kNONE&&tensorIOMode(0)==TensorIOMode::kINPUT&&tensorIOMode(1)==TensorIOMode::kOUTPUT);

    std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0));
    auto input=network->addInput("image",DataType::kFLOAT,Dims4(1,2,5,6));assert(input);
    assert(std::strcmp(input->getName(),"image")==0);
    float weights[72];std::fill(std::begin(weights),std::end(weights),.25f);float bias[4]={1,2,3,4};
    auto conv=network->addConvolutionNd(*input,4,Dims2(3,3),Weights{DataType::kFLOAT,weights,72},Weights{DataType::kFLOAT,bias,4});assert(conv);
    conv->setStrideNd(Dims2(2,2));conv->setPaddingNd(Dims2(1,1));
    auto shape=conv->getOutput(0)->getDimensions();assert(shape.nbDims==4&&shape.d[2]==3&&shape.d[3]==3);
    auto relu=network->addActivation(*conv->getOutput(0),ActivationType::kRELU);assert(relu);
    auto pool=network->addPoolingNd(*relu->getOutput(0),PoolingType::kAVERAGE,Dims2(2,2));assert(pool);
    pool->setStrideNd(Dims2(2,2));pool->getOutput(0)->setName("result");network->markOutput(*pool->getOutput(0));
    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,4096);
    std::unique_ptr<IHostMemory> blob(builder->buildSerializedNetwork(*network,*config));assert(blob&&blob->size()>16);
    assert(std::memcmp(blob->data(),"ZTRTAGX2",8)==0);
    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(blob->data(),blob->size()));assert(engine);
    assert(engine->getNbIOTensors()==2&&engine->getTensorIOMode("image")==TensorIOMode::kINPUT&&engine->getTensorIOMode("result")==TensorIOMode::kOUTPUT);
    shape=engine->getTensorShape("result");assert(shape.nbDims==4&&shape.d[0]==1&&shape.d[1]==4&&shape.d[2]==1&&shape.d[3]==1);
    assert(engine->getTensorIOMode("missing")==TensorIOMode::kNONE);
    assert(logger.errors==0);
    // Invalid setters cannot silently build a model using the previous attribute value.
    conv->setStrideNd(Dims2(-1,2));assert(!std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network,*config)));
    conv->setStrideNd(Dims2(2,2));assert(std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network,*config)));
    config->setFlag(BuilderFlag::kFP16);assert(!std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network,*config)));
    char bad[]="not an independently generated engine";assert(!std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(bad,sizeof(bad))));
    std::unique_ptr<INetworkDefinition> other(builder->createNetworkV2(0));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*other,logger));assert(!parser->parse(bad,sizeof(bad)));assert(parser->getNbErrors()==1&&parser->getError(0));
    std::unique_ptr<INetworkDefinition> reduced(builder->createNetworkV2(0));
    auto rx=reduced->addInput("reduction_input",DataType::kFLOAT,Dims3(2,3,4));assert(rx);
    auto squareRoot=reduced->addUnary(*rx,UnaryOperation::kSQRT);assert(squareRoot);
    assert(!reduced->addUnary(*rx,static_cast<UnaryOperation>(999)));
    auto sum=reduced->addReduce(*rx,ReduceOperation::kSUM,5,false);assert(sum);
    auto rd=sum->getOutput(0)->getDimensions();assert(rd.nbDims==1&&rd.d[0]==3);
    auto avg=reduced->addReduce(*rx,ReduceOperation::kAVG,2,true);assert(avg);
    rd=avg->getOutput(0)->getDimensions();assert(rd.nbDims==3&&rd.d[1]==1);
    assert(!reduced->addReduce(*rx,ReduceOperation::kSUM,8,false));
    assert(!reduced->addReduce(*rx,ReduceOperation::kPROD,1,false));
    std::unique_ptr<INetworkDefinition> normNetwork(builder->createNetworkV2(0));
    auto nx=normNetwork->addInput("norm_input",DataType::kFLOAT,Dims2(2,4));assert(nx);
    float ns[4]={1,1,1,1},nb[4]={0,0,0,0};
    auto sn=normNetwork->addConstant(Dims{1,{4}},Weights{DataType::kFLOAT,ns,4});assert(sn);
    auto bn=normNetwork->addConstant(Dims{1,{4}},Weights{DataType::kFLOAT,nb,4});assert(bn);
    auto norm=normNetwork->addNormalization(*nx,*sn->getOutput(0),*bn->getOutput(0),2);assert(norm);
    norm->setEpsilon(.003f);assert(norm->getEpsilon()==.003f);
    normNetwork->markOutput(*norm->getOutput(0));
    std::unique_ptr<IBuilderConfig> normConfig(builder->createBuilderConfig());
    assert(std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*normNetwork,*normConfig)));
    norm->setEpsilon(-1.f);assert(!std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*normNetwork,*normConfig)));
    norm->setEpsilon(.003f);assert(std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*normNetwork,*normConfig)));
    auto gatherBuilder=ztrtCreateBuilder();assert(gatherBuilder);
    int64_t gxShape[2]={2,3},giShape[1]={2},gi[2]={-1,0},gyShape[2]={0,0};
    auto gx=ztrtBuilderAddInput(gatherBuilder,"gather_input",gxShape,2);assert(gx>=0);
    auto gy=ztrtBuilderAddGather(gatherBuilder,gx,1,gi,2,giShape,1);assert(gy>=0);
    assert(ztrtBuilderTensorShape(gatherBuilder,gy,gyShape,2)==2&&gyShape[0]==2&&gyShape[1]==2);
    assert(ztrtBuilderMarkOutput(gatherBuilder,gy)==0);auto ge=ztrtBuilderBuild(gatherBuilder);assert(ge);
    ztrtDestroyEngine(ge);ztrtDestroyBuilder(gatherBuilder);
    // Public enum integers differ from our stable C dispatch codes; verify every supported translation.
    for(auto operation:{ElementWiseOperation::kSUM,ElementWiseOperation::kPROD,ElementWiseOperation::kSUB,ElementWiseOperation::kDIV,ElementWiseOperation::kMAX,ElementWiseOperation::kMIN}){
        std::unique_ptr<INetworkDefinition> ew(builder->createNetworkV2(0));
        auto a=ew->addInput("a",DataType::kFLOAT,Dims2(2,3));auto b=ew->addInput("b",DataType::kFLOAT,Dims2(2,3));
        auto result=ew->addElementWise(*a,*b,operation);assert(result);ew->markOutput(*result->getOutput(0));
        std::unique_ptr<IBuilderConfig> ewConfig(builder->createBuilderConfig());
        std::unique_ptr<IHostMemory> saved(builder->buildSerializedNetwork(*ew,*ewConfig));assert(saved);
        auto bytes=static_cast<unsigned char const*>(saved->data());uint64_t length=0;
        for(int i=0;i<8;++i)length|=uint64_t(bytes[16+i])<<(8*i);
        assert(length<=saved->size()-24);std::string metadata(reinterpret_cast<char const*>(bytes+24),size_t(length));
        char const* kind=operation==ElementWiseOperation::kSUM?"add":operation==ElementWiseOperation::kPROD?"mul":operation==ElementWiseOperation::kSUB?"sub":operation==ElementWiseOperation::kDIV?"div":operation==ElementWiseOperation::kMAX?"max":"min";
        assert(metadata.find(std::string("\"kind\":\"")+kind+"\"")!=std::string::npos);
        assert(!ew->addElementWise(*a,*b,ElementWiseOperation::kPOW));
    }
    if(argc>1){
        std::ifstream file(argv[1],std::ios::binary);assert(file);
        std::vector<char> model((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
        parser->clearErrors();assert(parser->parse(model.data(),model.size())&&parser->getNbErrors()==0);
        std::unique_ptr<IBuilderConfig> importedConfig(builder->createBuilderConfig());
        std::unique_ptr<IHostMemory> imported(builder->buildSerializedNetwork(*other,*importedConfig));assert(imported);
        std::unique_ptr<ICudaEngine> importedEngine(runtime->deserializeCudaEngine(imported->data(),imported->size()));assert(importedEngine);
        auto dims=importedEngine->getTensorShape("y");assert(dims.nbDims==2&&dims.d[0]==2&&dims.d[1]==4);
    }
    std::cout<<"PASS: C++ builder, configuration, shape refresh, serialization, runtime and error paths\n";
}
