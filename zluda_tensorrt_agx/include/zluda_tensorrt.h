#ifndef ZLUDA_TENSORRT_H
#define ZLUDA_TENSORRT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Independent ABI. All tensors are contiguous f32, with static dimensions and rank <= 8.
 * Status: 0 success, -1 failure. Tensor IDs: >= 0 success, -1 failure.
 * Factory failures return NULL; ztrtGetLastError is thread-local.
 * Device buffers and the CUDA context must outlive their queued uses. */
typedef struct ZtrtBuilder ZtrtBuilder;
typedef struct ZtrtRuntime ZtrtRuntime;
typedef struct ZtrtEngine ZtrtEngine;
typedef struct ZtrtContext ZtrtContext;
enum ZtrtUnary { ZTRT_RELU=0, ZTRT_SIGMOID=1, ZTRT_TANH=2, ZTRT_IDENTITY=3 };
enum ZtrtBinary { ZTRT_ADD=0, ZTRT_MUL=1, ZTRT_SUB=2, ZTRT_DIV=3, ZTRT_MAX=4, ZTRT_MIN=5 };
enum ZtrtPool { ZTRT_MAX_POOL=0, ZTRT_AVG_POOL=1 };
uint32_t ztrtGetVersion(void);
const char* ztrtGetLastError(void);
ZtrtBuilder* ztrtCreateBuilder(void);
void ztrtDestroyBuilder(ZtrtBuilder*);
int32_t ztrtBuilderSetWorkspaceLimit(ZtrtBuilder*, size_t);
int32_t ztrtBuilderAddInput(ZtrtBuilder*, const char*, const int64_t*, size_t);
int32_t ztrtBuilderAddConstant(ZtrtBuilder*, const char*, const int64_t*, size_t, const float*, size_t);
int32_t ztrtBuilderSetTensorName(ZtrtBuilder*, int32_t, const char*);
int32_t ztrtBuilderTensorShape(ZtrtBuilder*, int32_t, int64_t*, size_t);
int32_t ztrtBuilderMarkOutput(ZtrtBuilder*, int32_t);
int32_t ztrtBuilderAddMatMul(ZtrtBuilder*, int32_t, int32_t, int32_t, int32_t);
int32_t ztrtBuilderAddElementWise(ZtrtBuilder*, int32_t, int32_t, int32_t);
int32_t ztrtBuilderAddActivation(ZtrtBuilder*, int32_t, int32_t);
int32_t ztrtBuilderAddSoftmax(ZtrtBuilder*, int32_t, int32_t);
int32_t ztrtBuilderAddReshape(ZtrtBuilder*, int32_t, const int64_t*, size_t);
int32_t ztrtBuilderAddTranspose(ZtrtBuilder*, int32_t, const int32_t*, size_t);
int32_t ztrtBuilderAddConcat(ZtrtBuilder*, const int32_t*, size_t, int32_t);
int32_t ztrtBuilderAddConv2d(ZtrtBuilder*, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
int32_t ztrtBuilderAddPool2d(ZtrtBuilder*, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
int32_t ztrtBuilderConfigureConv2d(ZtrtBuilder*, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
int32_t ztrtBuilderConfigurePool2d(ZtrtBuilder*, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
int32_t ztrtBuilderConfigureSoftmax(ZtrtBuilder*, int32_t, int32_t);
int32_t ztrtBuilderConfigureReshape(ZtrtBuilder*, int32_t, const int64_t*, size_t);
ZtrtEngine* ztrtBuilderBuild(const ZtrtBuilder*);
ZtrtEngine* ztrtBuildJson(const uint8_t*, size_t, size_t);
ZtrtEngine* ztrtBuildOnnx(const uint8_t*, size_t, size_t);
int32_t ztrtBuilderImportOnnx(ZtrtBuilder*, const uint8_t*, size_t);
int32_t ztrtBuilderIOCount(const ZtrtBuilder*, int32_t output);
int32_t ztrtBuilderIOTensor(const ZtrtBuilder*, int32_t output, int32_t index);
size_t ztrtBuilderTensorName(const ZtrtBuilder*, int32_t, char*, size_t);
ZtrtRuntime* ztrtCreateRuntime(void);
void ztrtDestroyRuntime(ZtrtRuntime*);
ZtrtEngine* ztrtDeserializeEngine(const ZtrtRuntime*, const uint8_t*, size_t, size_t);
void ztrtDestroyEngine(ZtrtEngine*);
/* With NULL data and zero capacity, returns required bytes. Never partially writes. */
size_t ztrtSerializeEngine(const ZtrtEngine*, uint8_t*, size_t);
int32_t ztrtGetNbIOTensors(const ZtrtEngine*);
const char* ztrtGetIOTensorName(const ZtrtEngine*, int32_t);
int32_t ztrtGetTensorIOMode(const ZtrtEngine*, const char*);
int32_t ztrtGetTensorShape(const ZtrtEngine*, const char*, int64_t*, size_t);
size_t ztrtGetTensorBytes(const ZtrtEngine*, const char*);
size_t ztrtGetWorkspaceSize(const ZtrtEngine*);
size_t ztrtGetConstantBytes(const ZtrtEngine*);
size_t ztrtGetFusedActivationCount(const ZtrtEngine*);
ZtrtContext* ztrtCreateExecutionContext(const ZtrtEngine*);
void ztrtDestroyExecutionContext(ZtrtContext*);
/* Contexts are not concurrently callable. Bindings must not overlap each other. */
int32_t ztrtSetTensorAddress(ZtrtContext*, const char*, void*, size_t);
int32_t ztrtEnqueueV3(ZtrtContext*, void* cuda_stream);
int32_t ztrtSynchronize(ZtrtContext*);
#ifdef __cplusplus
}
#endif
#endif
