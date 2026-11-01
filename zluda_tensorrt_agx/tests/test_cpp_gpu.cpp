// End-to-end C++ public-enum translation tests using only our runtime and CUDA driver.
#include "NvInfer.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <stdexcept>
struct Logger final:nvinfer1::ILogger{void log(Severity,char const* s)noexcept override{std::cerr<<s<<'\n';}};
template<class F> F symbol(void* lib,char const* name){auto p=dlsym(lib,name);if(!p)throw std::runtime_error(name);return reinterpret_cast<F>(p);}
void check(int rc){if(rc)throw std::runtime_error("driver error "+std::to_string(rc));}
int main(){try{
    using namespace nvinfer1;Logger logger;
    auto path=std::getenv("ZLUDA_AGX_LIBCUDA");void* driver=dlopen(path?path:"libcuda.so.1",RTLD_NOW);if(!driver)throw std::runtime_error(dlerror());
    check(symbol<int(*)(unsigned)>(driver,"cuInit")(0));void* primary=nullptr;
    check(symbol<int(*)(void**,int)>(driver,"cuDevicePrimaryCtxRetain")(&primary,0));check(symbol<int(*)(void*)>(driver,"cuCtxSetCurrent")(primary));
    float x[6]={-5,-2,-1,1,3,7},rhs[6]={2,2,2,2,2,2},actual[6];void* dx=nullptr;void* dy=nullptr;
    auto alloc=symbol<int(*)(void**,size_t)>(driver,"cuMemAlloc_v2");check(alloc(&dx,sizeof(x)));check(alloc(&dy,sizeof(actual)));
    check(symbol<int(*)(void*,void const*,size_t)>(driver,"cuMemcpyHtoD_v2")(dx,x,sizeof(x)));
    for(auto op:{ElementWiseOperation::kSUM,ElementWiseOperation::kPROD,ElementWiseOperation::kSUB,ElementWiseOperation::kDIV,ElementWiseOperation::kMAX,ElementWiseOperation::kMIN}){
        std::unique_ptr<IBuilder> builder(createInferBuilder(logger));std::unique_ptr<INetworkDefinition> net(builder->createNetworkV2(0));
        auto a=net->addInput("input",DataType::kFLOAT,Dims2(2,3));auto b=net->addConstant(Dims2(2,3),Weights{DataType::kFLOAT,rhs,6});assert(a&&b);
        auto result=net->addElementWise(*a,*b->getOutput(0),op);assert(result);result->getOutput(0)->setName("output");net->markOutput(*result->getOutput(0));
        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());std::unique_ptr<IHostMemory> blob(builder->buildSerializedNetwork(*net,*config));assert(blob);
        std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(blob->data(),blob->size()));assert(engine);
        assert(engine->getTensorIOMode("input")==TensorIOMode::kINPUT&&engine->getTensorIOMode("output")==TensorIOMode::kOUTPUT);
        std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());assert(context);
        assert(context->setInputTensorAddress("input",dx)&&context->setOutputTensorAddress("output",dy));
        assert(context->enqueueV3(nullptr)&&context->synchronize());check(symbol<int(*)(void*,void const*,size_t)>(driver,"cuMemcpyDtoH_v2")(actual,dy,sizeof(actual)));
        for(int i=0;i<6;++i){float expected=op==ElementWiseOperation::kSUM?x[i]+2:op==ElementWiseOperation::kPROD?x[i]*2:op==ElementWiseOperation::kSUB?x[i]-2:op==ElementWiseOperation::kDIV?x[i]/2:op==ElementWiseOperation::kMAX?std::max(x[i],2.f):std::min(x[i],2.f);assert(std::abs(actual[i]-expected)<1e-5f);}
    }
    for(auto op:{ReduceOperation::kSUM,ReduceOperation::kAVG,ReduceOperation::kPROD,ReduceOperation::kMAX,ReduceOperation::kMIN}){
        std::unique_ptr<IBuilder> builder(createInferBuilder(logger));std::unique_ptr<INetworkDefinition> net(builder->createNetworkV2(0));
        auto input=net->addInput("input",DataType::kFLOAT,Dims2(2,3));assert(input);auto result=net->addReduce(*input,op,2,false);assert(result);
        result->getOutput(0)->setName("output");net->markOutput(*result->getOutput(0));
        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());std::unique_ptr<IHostMemory> blob(builder->buildSerializedNetwork(*net,*config));assert(blob);
        std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(blob->data(),blob->size()));assert(engine);
        std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());assert(context&&context->setInputTensorAddress("input",dx)&&context->setOutputTensorAddress("output",dy));
        assert(context->enqueueV3(nullptr)&&context->synchronize());check(symbol<int(*)(void*,void const*,size_t)>(driver,"cuMemcpyDtoH_v2")(actual,dy,2*sizeof(float)));
        for(int row=0;row<2;++row){
            float expected=op==ReduceOperation::kPROD?1:op==ReduceOperation::kMAX?-std::numeric_limits<float>::infinity():op==ReduceOperation::kMIN?std::numeric_limits<float>::infinity():0;
            for(int column=0;column<3;++column){auto value=x[row*3+column];expected=op==ReduceOperation::kPROD?expected*value:op==ReduceOperation::kMAX?std::max(expected,value):op==ReduceOperation::kMIN?std::min(expected,value):expected+value;}
            if(op==ReduceOperation::kAVG){expected/=3;}assert(std::abs(actual[row]-expected)<1e-5f);
        }
    }
    auto free=symbol<int(*)(void*)>(driver,"cuMemFree_v2");check(free(dx));check(free(dy));dlclose(driver);
    std::cout<<"PASS: six elementwise and five reduction C++ mappings, I/O roles and GPU engine roundtrips\n";return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
