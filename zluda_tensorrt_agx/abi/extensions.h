#ifndef ZLUDA_TENSORRT_ABI_EXTENSIONS_H
#define ZLUDA_TENSORRT_ABI_EXTENSIONS_H
#include "zluda_tensorrt.h"
#ifdef __cplusplus
extern "C" {
#define ZTRT_ABI_NOEXCEPT noexcept
#else
#define ZTRT_ABI_NOEXCEPT
#endif
/* Prototype extension: convert a valid native engine handle into an opaque public-engine wrapper.
 * Ownership transfers only on success. The logger is a borrowed versioned C++ ILogger pointer.
 * This is not a decoder for NVIDIA engines. See abi/README.md for supported dispatch operations. */
void* ztrtRtxWrapNativeEngine(ZtrtEngine*,void* logger) ZTRT_ABI_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#undef ZTRT_ABI_NOEXCEPT
#endif
