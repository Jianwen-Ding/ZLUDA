// A complete independently built/serialized MLP, using the source-level C++ API and CUDA Driver API only.
#include "NvInfer.h"
#include <dlfcn.h>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>
struct Logger final : nvinfer1::ILogger {
    void log(Severity, char const* s) noexcept override { std::cerr<<s<<'\n'; }
};
template<class F> F load(void* lib,char const* name){auto p=dlsym(lib,name);if(!p)throw std::runtime_error(name);return reinterpret_cast<F>(p);}
void check(int error){if(error)throw std::runtime_error("CUDA Driver API error "+std::to_string(error));}
int main(){try{
    Logger logger;
    using namespace nvinfer1;
    std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0));
    auto input=network->addInput("input",DataType::kFLOAT,Dims2(2,3));
    float weights[]={1,2,3,4,5,6},bias[]={1,-1};
    auto w=network->addConstant(Dims2(3,2),Weights{DataType::kFLOAT,weights,6});
    Dims biasShape;biasShape.nbDims=1;biasShape.d[0]=2;
    auto b=network->addConstant(biasShape,Weights{DataType::kFLOAT,bias,2});
    if(!input||!w||!b)throw std::runtime_error("network inputs");
    auto mm=network->addMatrixMultiply(*input,MatrixOperation::kNONE,*w->getOutput(0),MatrixOperation::kNONE);
    if(!mm)throw std::runtime_error("matmul");
    auto add=network->addElementWise(*mm->getOutput(0),*b->getOutput(0),ElementWiseOperation::kSUM);
    if(!add)throw std::runtime_error("bias");
    auto relu=network->addActivation(*add->getOutput(0),ActivationType::kRELU);
    if(!relu)throw std::runtime_error("activation");
    relu->getOutput(0)->setName("output");network->markOutput(*relu->getOutput(0));
    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,1<<20);
    std::unique_ptr<IHostMemory> blob(builder->buildSerializedNetwork(*network,*config));
    if(!blob)throw std::runtime_error("engine build");
    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(blob->data(),blob->size()));
    if(!engine)throw std::runtime_error("native deserialization");
    char const* path=std::getenv("ZLUDA_AGX_LIBCUDA");void* driver=dlopen(path?path:"libcuda.so.1",RTLD_NOW);
    if(!driver)throw std::runtime_error(dlerror());
    check(load<int(*)(unsigned)>(driver,"cuInit")(0));
    void* cudaContext=nullptr;check(load<int(*)(void**,int)>(driver,"cuDevicePrimaryCtxRetain")(&cudaContext,0));
    check(load<int(*)(void*)>(driver,"cuCtxSetCurrent")(cudaContext));
    float x[]={1,-2,3,4,5,-6},y[4]={};void* dx=nullptr;void* dy=nullptr;
    auto alloc=load<int(*)(void**,size_t)>(driver,"cuMemAlloc_v2");check(alloc(&dx,sizeof(x)));check(alloc(&dy,sizeof(y)));
    check(load<int(*)(void*,void const*,size_t)>(driver,"cuMemcpyHtoD_v2")(dx,x,sizeof(x)));
    {
        std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
        if(!context||!context->setTensorAddress("input",dx)||!context->setTensorAddress("output",dy)||!context->enqueueV3(nullptr)||!context->synchronize())throw std::runtime_error("inference");
        check(load<int(*)(void*,void const*,size_t)>(driver,"cuMemcpyDtoH_v2")(y,dy,sizeof(y)));
    }
    auto free=load<int(*)(void*)>(driver,"cuMemFree_v2");check(free(dx));check(free(dy));
    float expected[]={11,11,0,0};for(int i=0;i<4;++i)if(std::abs(y[i]-expected[i])>1e-5f)throw std::runtime_error("numerical mismatch");
    std::cout<<"MLIR/AIR inference: ["<<y[0]<<", "<<y[1]<<", "<<y[2]<<", "<<y[3]<<"], native serialized bytes="<<blob->size()<<'\n';
    dlclose(driver);return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
