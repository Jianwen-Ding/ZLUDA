#ifndef ZLUDA_AGX_NVINFER_H
#define ZLUDA_AGX_NVINFER_H
// Independently implemented source-level subset of the documented TensorRT builder/runtime workflow.
// This header and libtensorrt_agx have their own ABI and engine format. They are not NVIDIA binary ABI shims.
#include "zluda_tensorrt.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nvinfer1 {
enum class DataType : int32_t { kFLOAT=0, kHALF=1, kINT8=2, kINT32=3, kBOOL=4 };
enum class TensorIOMode : int32_t { kNONE=-1, kINPUT=0, kOUTPUT=1 };
enum class ActivationType : int32_t { kRELU=0, kSIGMOID=1, kTANH=2 };
enum class ElementWiseOperation : int32_t { kSUM=0, kPROD=1, kSUB=2, kDIV=3, kMAX=4, kMIN=5 };
enum class ReduceOperation : int32_t { kSUM=0, kPROD=1, kMAX=2, kMIN=3, kAVG=4 };
enum class MatrixOperation : int32_t { kNONE=0, kTRANSPOSE=1, kVECTOR=2 };
enum class PoolingType : int32_t { kMAX=0, kAVERAGE=1 };
enum class MemoryPoolType : int32_t { kWORKSPACE=0 };
enum class BuilderFlag : int32_t { kFP16=0, kINT8=1, kTF32=2 };
enum class NetworkDefinitionCreationFlag : int32_t { kEXPLICIT_BATCH=0, kSTRONGLY_TYPED=1 };
using NetworkDefinitionCreationFlags = uint32_t;
struct Dims { int32_t nbDims=0; int64_t d[8]{}; };
struct Dims2 : Dims { Dims2(int64_t a, int64_t b) { nbDims=2; d[0]=a; d[1]=b; } };
struct Dims3 : Dims { Dims3(int64_t a, int64_t b, int64_t c) { nbDims=3; d[0]=a; d[1]=b; d[2]=c; } };
struct Dims4 : Dims { Dims4(int64_t a, int64_t b, int64_t c, int64_t e) { nbDims=4; d[0]=a; d[1]=b; d[2]=c; d[3]=e; } };
struct Weights { DataType type; void const* values; int64_t count; };
class ILogger {
public:
    enum class Severity : int32_t { kINTERNAL_ERROR=0, kERROR=1, kWARNING=2, kINFO=3, kVERBOSE=4 };
    virtual void log(Severity severity, char const* message) noexcept = 0;
    virtual ~ILogger() = default;
};
inline void report(ILogger* l, char const* message=nullptr) { if (l) l->log(ILogger::Severity::kERROR, message ? message : ztrtGetLastError()); }
inline bool validDims(Dims const& d) { if (d.nbDims<0 || d.nbDims>8) return false; for(int i=0;i<d.nbDims;++i) if(d.d[i]<=0) return false; return true; }
inline int32_t narrow(int64_t v) { return v>=0 && v<=std::numeric_limits<int32_t>::max() ? int32_t(v) : -1; }

class ITensor {
    ZtrtBuilder* mBuilder;
    int32_t mId;
    std::string mName;
    ILogger* mLogger;
public:
    ITensor(ZtrtBuilder* b, int32_t id, ILogger* logger):mBuilder(b),mId(id),mLogger(logger){size_t n=ztrtBuilderTensorName(b,id,nullptr,0);if(n){std::vector<char> name(n);if(ztrtBuilderTensorName(b,id,name.data(),n))mName=name.data();}}
    int32_t id() const { return mId; }
    void setName(char const* name) { if (ztrtBuilderSetTensorName(mBuilder,mId,name)==0) mName=name; else report(mLogger); }
    char const* getName() const { return mName.c_str(); }
    Dims getDimensions() const { Dims d; d.nbDims=ztrtBuilderTensorShape(mBuilder,mId,d.d,8); return d; }
    DataType getType() const { return DataType::kFLOAT; }
};
class ILayer {
protected:
    ITensor* mOutput;
    ZtrtBuilder* mBuilder;
    ILogger* mLogger;
    bool mValid=true;
    void accept(int32_t status){mValid=status==0;if(!mValid)report(mLogger);}
public:
    ILayer(ITensor* out,ZtrtBuilder* b,ILogger* l):mOutput(out),mBuilder(b),mLogger(l){}
    virtual ~ILayer()=default;
    bool valid() const{return mValid;}
    ITensor* getOutput(int32_t index) const { return index==0 ? mOutput : nullptr; }
    int32_t getNbOutputs() const { return 1; }
};
class IConvolutionLayer : public ILayer {
    Dims mStride=Dims2(1,1),mPadding=Dims2(0,0),mDilation=Dims2(1,1);
    int32_t mGroups;
    void apply() { accept(ztrtBuilderConfigureConv2d(mBuilder,mOutput->id(),narrow(mStride.d[0]),narrow(mStride.d[1]),narrow(mPadding.d[0]),narrow(mPadding.d[1]),narrow(mDilation.d[0]),narrow(mDilation.d[1]),mGroups)); }
public:
    IConvolutionLayer(ITensor* t,ZtrtBuilder* b,ILogger* l,int32_t groups):ILayer(t,b,l),mGroups(groups){}
    void setStrideNd(Dims d) { if(d.nbDims!=2) { mValid=false; report(mLogger,"only 2-D convolution is supported"); return; } mStride=d; apply(); }
    void setPaddingNd(Dims d) { if(d.nbDims!=2) { mValid=false; report(mLogger,"only symmetric 2-D padding is supported"); return; } mPadding=d; apply(); }
    void setDilationNd(Dims d) { if(d.nbDims!=2) { mValid=false; report(mLogger,"only 2-D dilation is supported"); return; } mDilation=d; apply(); }
    void setNbGroups(int32_t n) { mGroups=n; apply(); }
};
class IPoolingLayer : public ILayer {
    Dims mStride=Dims2(1,1),mPadding=Dims2(0,0);
    bool mExclude=true;
    void apply() { accept(ztrtBuilderConfigurePool2d(mBuilder,mOutput->id(),narrow(mStride.d[0]),narrow(mStride.d[1]),narrow(mPadding.d[0]),narrow(mPadding.d[1]),!mExclude)); }
public:
    using ILayer::ILayer;
    void setStrideNd(Dims d) { if(d.nbDims!=2) { mValid=false; report(mLogger,"only 2-D pooling is supported"); return; } mStride=d; apply(); }
    void setPaddingNd(Dims d) { if(d.nbDims!=2) { mValid=false; report(mLogger,"only symmetric 2-D padding is supported"); return; } mPadding=d; apply(); }
    void setAverageCountExcludesPadding(bool b) { mExclude=b; apply(); }
};
class ISoftMaxLayer : public ILayer {
public:
    using ILayer::ILayer;
    void setAxes(uint32_t mask) {
        if(!mask || (mask&(mask-1))) { mValid=false; report(mLogger,"softmax needs exactly one axis"); return; }
        int axis=0; while((mask>>axis)!=1) ++axis;
        accept(ztrtBuilderConfigureSoftmax(mBuilder,mOutput->id(),axis));
    }
};
class IShuffleLayer : public ILayer {
    Dims mOriginal;
public:
    IShuffleLayer(ITensor* t,ZtrtBuilder* b,ILogger* l,Dims original):ILayer(t,b,l),mOriginal(original){}
    void setReshapeDimensions(Dims d) {
        if(d.nbDims<0 || d.nbDims>8) { mValid=false; report(mLogger,"invalid reshape rank"); return; }
        uint64_t total=1,known=1; int infer=-1;
        for(int i=0;i<mOriginal.nbDims;++i) total*=uint64_t(mOriginal.d[i]);
        for(int i=0;i<d.nbDims;++i) {
            if(d.d[i]==0 && i<mOriginal.nbDims) d.d[i]=mOriginal.d[i];
            if(d.d[i]==-1) { if(infer>=0) { mValid=false; report(mLogger,"reshape has multiple inferred dimensions"); return; } infer=i; }
            else if(d.d[i]<=0 || uint64_t(d.d[i])>total || known>total/uint64_t(d.d[i])) { mValid=false; report(mLogger,"invalid reshape dimensions"); return; }
            else known*=uint64_t(d.d[i]);
        }
        if(infer>=0) { if(!known || total%known) { mValid=false; report(mLogger,"reshape element count differs"); return; } d.d[infer]=int64_t(total/known); }
        accept(ztrtBuilderConfigureReshape(mBuilder,mOutput->id(),d.d,size_t(d.nbDims)));
    }
};
class IActivationLayer : public ILayer { public: using ILayer::ILayer; };
class IElementWiseLayer : public ILayer { public: using ILayer::ILayer; };
class IMatrixMultiplyLayer : public ILayer { public: using ILayer::ILayer; };
class IReduceLayer : public ILayer { public: using ILayer::ILayer; };
class IConstantLayer : public ILayer { public: using ILayer::ILayer; };

class INetworkDefinition {
    ZtrtBuilder* mBuilder=ztrtCreateBuilder();
    ILogger* mLogger;
    uint32_t mUnique=0;
    std::vector<std::unique_ptr<ITensor>> mTensors;
    std::vector<std::unique_ptr<ILayer>> mLayers;
    ITensor* tensor(int32_t id) {
        if(id<0) { report(mLogger); return nullptr; }
        for(auto const& t:mTensors)if(t->id()==id)return t.get();
        auto t=std::make_unique<ITensor>(mBuilder,id,mLogger); auto p=t.get(); mTensors.push_back(std::move(t)); return p;
    }
    template<class T,class... Args> T* layer(int32_t id,Args&&... args) {
        auto t=tensor(id); if(!t) return nullptr;
        auto l=std::make_unique<T>(t,mBuilder,mLogger,std::forward<Args>(args)...); auto p=l.get(); mLayers.push_back(std::move(l)); return p;
    }
    int32_t constant(Dims dims,Weights weights) {
        if(!validDims(dims) || weights.type!=DataType::kFLOAT || weights.count<0) { report(mLogger,"only static f32 weights are supported"); return -1; }
        auto name=std::string("__ztrt_weight_")+std::to_string(mUnique++);
        return ztrtBuilderAddConstant(mBuilder,name.c_str(),dims.d,size_t(dims.nbDims),static_cast<float const*>(weights.values),size_t(weights.count));
    }
public:
    explicit INetworkDefinition(ILogger* logger):mLogger(logger){}
    ~INetworkDefinition(){ztrtDestroyBuilder(mBuilder);}
    INetworkDefinition(INetworkDefinition const&)=delete;
    INetworkDefinition& operator=(INetworkDefinition const&)=delete;
    ZtrtBuilder* native() const { return mBuilder; }
    bool valid() const{for(auto const& layer:mLayers)if(!layer->valid())return false;return true;}
    int32_t getNbInputs() const{return ztrtBuilderIOCount(mBuilder,0);}
    int32_t getNbOutputs() const{return ztrtBuilderIOCount(mBuilder,1);}
    ITensor* getInput(int32_t i){return tensor(ztrtBuilderIOTensor(mBuilder,0,i));}
    ITensor* getOutput(int32_t i){return tensor(ztrtBuilderIOTensor(mBuilder,1,i));}
    ITensor* addInput(char const* name,DataType type,Dims dims) {
        if(type!=DataType::kFLOAT || !validDims(dims)) { report(mLogger,"only static f32 inputs are supported"); return nullptr; }
        auto p=tensor(ztrtBuilderAddInput(mBuilder,name,dims.d,size_t(dims.nbDims))); if(p) p->setName(name); return p;
    }
    void markOutput(ITensor& t) { if(ztrtBuilderMarkOutput(mBuilder,t.id())) report(mLogger); }
    IConstantLayer* addConstant(Dims dims,Weights weights) { return layer<IConstantLayer>(constant(dims,weights)); }
    IActivationLayer* addActivation(ITensor& x,ActivationType type) { return layer<IActivationLayer>(ztrtBuilderAddActivation(mBuilder,x.id(),int32_t(type))); }
    ILayer* addIdentity(ITensor& x) { return layer<ILayer>(ztrtBuilderAddActivation(mBuilder,x.id(),ZTRT_IDENTITY)); }
    IReduceLayer* addReduce(ITensor& x,ReduceOperation op,uint32_t axes,bool keepDimensions) {
        if(op!=ReduceOperation::kSUM && op!=ReduceOperation::kAVG){report(mLogger,"only sum and average reductions are supported");return nullptr;}
        return layer<IReduceLayer>(ztrtBuilderAddReduce(mBuilder,x.id(),axes,op==ReduceOperation::kAVG?1:0,keepDimensions?1:0));
    }
    IElementWiseLayer* addElementWise(ITensor& a,ITensor& b,ElementWiseOperation op) { return layer<IElementWiseLayer>(ztrtBuilderAddElementWise(mBuilder,a.id(),b.id(),int32_t(op))); }
    IMatrixMultiplyLayer* addMatrixMultiply(ITensor& a,MatrixOperation ta,ITensor& b,MatrixOperation tb) {
        if(ta==MatrixOperation::kVECTOR || tb==MatrixOperation::kVECTOR) { report(mLogger,"matmul requires two matrices"); return nullptr; }
        return layer<IMatrixMultiplyLayer>(ztrtBuilderAddMatMul(mBuilder,a.id(),b.id(),int32_t(ta),int32_t(tb)));
    }
    IConvolutionLayer* addConvolutionNd(ITensor& x,int32_t channels,Dims kernel,Weights weights,Weights bias) {
        auto input=x.getDimensions();
        if(input.nbDims!=4 || kernel.nbDims!=2 || channels<=0 || kernel.d[0]<=0 || kernel.d[1]<=0 || weights.count<=0) { report(mLogger,"convolution requires NCHW input and 2-D weights"); return nullptr; }
        uint64_t plane=uint64_t(channels)*uint64_t(kernel.d[0]);
        if(plane>uint64_t(std::numeric_limits<int32_t>::max()) || uint64_t(kernel.d[1])>uint64_t(std::numeric_limits<int32_t>::max())/plane) { report(mLogger,"convolution weight shape overflow"); return nullptr; }
        plane*=uint64_t(kernel.d[1]);
        if(uint64_t(weights.count)%plane) { report(mLogger,"convolution weights have an invalid count"); return nullptr; }
        int64_t per=int64_t(uint64_t(weights.count)/plane);
        if(per<=0 || input.d[1]%per) { report(mLogger,"convolution channels and weights differ"); return nullptr; }
        int32_t groups=narrow(input.d[1]/per);
        int32_t w=constant(Dims4(channels,per,kernel.d[0],kernel.d[1]),weights); if(w<0) { report(mLogger); return nullptr; }
        int32_t b=-1;
        if(bias.count) { Dims bd; bd.nbDims=1; bd.d[0]=channels; b=constant(bd,bias); if(b<0){report(mLogger);return nullptr;} }
        return layer<IConvolutionLayer>(ztrtBuilderAddConv2d(mBuilder,x.id(),w,b,1,1,0,0,1,1,groups),groups);
    }
    IPoolingLayer* addPoolingNd(ITensor& x,PoolingType type,Dims window) {
        if(window.nbDims!=2) { report(mLogger,"only 2-D pooling is supported"); return nullptr; }
        return layer<IPoolingLayer>(ztrtBuilderAddPool2d(mBuilder,x.id(),int32_t(type),narrow(window.d[0]),narrow(window.d[1]),1,1,0,0,0));
    }
    ISoftMaxLayer* addSoftMax(ITensor& x) { return layer<ISoftMaxLayer>(ztrtBuilderAddSoftmax(mBuilder,x.id(),-1)); }
    IShuffleLayer* addShuffle(ITensor& x) { auto d=x.getDimensions(); return layer<IShuffleLayer>(ztrtBuilderAddReshape(mBuilder,x.id(),d.d,size_t(d.nbDims)),d); }
};
class IBuilderConfig {
    size_t mWorkspace=size_t(1)<<30;
    bool mSupported=true;
public:
    void setMemoryPoolLimit(MemoryPoolType pool,size_t bytes) { if(pool==MemoryPoolType::kWORKSPACE) mWorkspace=bytes; else mSupported=false; }
    size_t getMemoryPoolLimit(MemoryPoolType) const { return mWorkspace; }
    void setFlag(BuilderFlag) { mSupported=false; }
    bool supported() const { return mSupported; }
};
class IHostMemory {
    std::vector<uint8_t> mData;
public:
    explicit IHostMemory(ZtrtEngine const* engine) {
        size_t n=ztrtSerializeEngine(engine,nullptr,0); mData.resize(n);
        if(n && ztrtSerializeEngine(engine,mData.data(),n)!=n) mData.clear();
    }
    void* data(){return mData.data();}
    void const* data() const{return mData.data();}
    size_t size() const{return mData.size();}
};
class IExecutionContext {
    ZtrtContext* mContext;
    ILogger* mLogger;
    struct Info { size_t bytes; Dims shape; TensorIOMode mode; };
    std::unordered_map<std::string,Info> mInfo;
    std::vector<std::string> mOrder;
public:
    IExecutionContext(ZtrtContext* context,ZtrtEngine const* engine,ILogger* logger):mContext(context),mLogger(logger) {
        int32_t n=ztrtGetNbIOTensors(engine);
        for(int32_t i=0;i<n;++i){auto name=ztrtGetIOTensorName(engine,i);Dims d;d.nbDims=ztrtGetTensorShape(engine,name,d.d,8);mInfo.emplace(name,Info{ztrtGetTensorBytes(engine,name),d,TensorIOMode(ztrtGetTensorIOMode(engine,name))});mOrder.emplace_back(name);}
    }
    ~IExecutionContext(){ztrtDestroyExecutionContext(mContext);}
    IExecutionContext(IExecutionContext const&)=delete;
    IExecutionContext& operator=(IExecutionContext const&)=delete;
    bool setTensorAddress(char const* name,void* data){auto i=mInfo.find(name ? name : "");if(i==mInfo.end()){report(mLogger,"unknown I/O tensor");return false;}if(ztrtSetTensorAddress(mContext,name,data,i->second.bytes)){report(mLogger);return false;}return true;}
    bool setInputTensorAddress(char const* name,void const* data){auto i=mInfo.find(name?name:"");return i!=mInfo.end()&&i->second.mode==TensorIOMode::kINPUT&&setTensorAddress(name,const_cast<void*>(data));}
    bool setOutputTensorAddress(char const* name,void* data){auto i=mInfo.find(name?name:"");return i!=mInfo.end()&&i->second.mode==TensorIOMode::kOUTPUT&&setTensorAddress(name,data);}
    bool setInputShape(char const* name,Dims d){auto i=mInfo.find(name ? name : "");if(i==mInfo.end() || i->second.mode!=TensorIOMode::kINPUT || d.nbDims!=i->second.shape.nbDims)return false;for(int a=0;a<d.nbDims;++a)if(d.d[a]!=i->second.shape.d[a]){report(mLogger,"dynamic shapes are not supported");return false;}return true;}
    bool enqueueV3(void* stream){if(ztrtEnqueueV3(mContext,stream)){report(mLogger);return false;}return true;}
    bool synchronize(){if(ztrtSynchronize(mContext)){report(mLogger);return false;}return true;}
    bool executeV2(void* const* bindings){if(!bindings)return false;for(size_t i=0;i<mOrder.size();++i)if(!setTensorAddress(mOrder[i].c_str(),bindings[i]))return false;return enqueueV3(nullptr)&&synchronize();}
};
class ICudaEngine {
    ZtrtEngine* mEngine;
    ILogger* mLogger;
public:
    ICudaEngine(ZtrtEngine* engine,ILogger* logger):mEngine(engine),mLogger(logger){}
    ~ICudaEngine(){ztrtDestroyEngine(mEngine);}
    ICudaEngine(ICudaEngine const&)=delete;
    ICudaEngine& operator=(ICudaEngine const&)=delete;
    IExecutionContext* createExecutionContext(){auto p=ztrtCreateExecutionContext(mEngine);if(!p){report(mLogger);return nullptr;}return new IExecutionContext(p,mEngine,mLogger);}
    IHostMemory* serialize() const{return new IHostMemory(mEngine);}
    int32_t getNbIOTensors() const{return ztrtGetNbIOTensors(mEngine);}
    char const* getIOTensorName(int32_t i) const{return ztrtGetIOTensorName(mEngine,i);}
    TensorIOMode getTensorIOMode(char const* name) const{return TensorIOMode(ztrtGetTensorIOMode(mEngine,name));}
    Dims getTensorShape(char const* name) const{Dims d;d.nbDims=ztrtGetTensorShape(mEngine,name,d.d,8);return d;}
    DataType getTensorDataType(char const*) const{return DataType::kFLOAT;}
    int64_t getDeviceMemorySizeV2() const{return int64_t(ztrtGetWorkspaceSize(mEngine));}
};
class IBuilder {
    ILogger* mLogger;
public:
    explicit IBuilder(ILogger& logger):mLogger(&logger){}
    INetworkDefinition* createNetworkV2(uint32_t flags){if(flags&~3U){report(mLogger,"unsupported network creation flags");return nullptr;}return new INetworkDefinition(mLogger);}
    IBuilderConfig* createBuilderConfig(){return new IBuilderConfig;}
    ICudaEngine* buildEngineWithConfig(INetworkDefinition& network,IBuilderConfig& config){if(!config.supported()||!network.valid()){report(mLogger,"unsupported configuration or invalid layer attributes");return nullptr;}ztrtBuilderSetWorkspaceLimit(network.native(),config.getMemoryPoolLimit(MemoryPoolType::kWORKSPACE));auto p=ztrtBuilderBuild(network.native());if(!p){report(mLogger);return nullptr;}return new ICudaEngine(p,mLogger);}
    IHostMemory* buildSerializedNetwork(INetworkDefinition& network,IBuilderConfig& config){std::unique_ptr<ICudaEngine> e(buildEngineWithConfig(network,config));return e ? e->serialize() : nullptr;}
};
class IRuntime {
    ZtrtRuntime* mRuntime=ztrtCreateRuntime();
    ILogger* mLogger;
public:
    explicit IRuntime(ILogger& logger):mLogger(&logger){}
    ~IRuntime(){ztrtDestroyRuntime(mRuntime);}
    IRuntime(IRuntime const&)=delete;
    IRuntime& operator=(IRuntime const&)=delete;
    ICudaEngine* deserializeCudaEngine(void const* data,size_t size){auto p=ztrtDeserializeEngine(mRuntime,static_cast<uint8_t const*>(data),size,size_t(1)<<30);if(!p){report(mLogger);return nullptr;}return new ICudaEngine(p,mLogger);}
};
inline IBuilder* createInferBuilder(ILogger& logger){return new IBuilder(logger);}
inline IRuntime* createInferRuntime(ILogger& logger){return new IRuntime(logger);}
inline int32_t getInferLibVersion(){return int32_t(ztrtGetVersion());}
} // namespace nvinfer1
#endif
