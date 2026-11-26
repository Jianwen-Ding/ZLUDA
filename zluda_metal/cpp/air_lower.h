// C interface of the ZLUDA IR -> Metal AIR lowering (air_lower.cpp)
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ZludaAirInput
{
    const uint8_t *main_bc; // module from ptx::to_llvm_module
    size_t main_len;
    const uint8_t *attributes_bc; // optional attributes module
    size_t attributes_len;
    const uint8_t *impl_bc; // zluda_ptx_metal_impl.bc
    size_t impl_len;
    int want_text; // also return the whole AIR module as text
} ZludaAirInput;

typedef struct ZludaAirKernel
{
    char *name;
    size_t param_count;
    uint64_t *param_offsets; // offsets in the parameter buffer (buffer(0))
    uint64_t *param_sizes;
    uint64_t params_size;
    int dynamic_shared;     // kernel takes a [[threadgroup(0)]] buffer for extern .shared
    uint64_t static_shared; // bytes of static .shared variables
    uint8_t *air;           // AIR bitcode module holding just this kernel
    size_t air_len;
    // Offset in the parameter buffer of the 64-bit address of the module's
    // device-globals buffer, or UINT64_MAX if the kernel uses no device globals
    uint64_t globals_param_offset;
} ZludaAirKernel;

// A writable module-scope `.global` variable, emulated as a slice of a
// per-module device buffer
typedef struct ZludaAirGlobal
{
    char *name;
    uint64_t offset;
    uint64_t size;
} ZludaAirGlobal;

typedef struct ZludaAirOutput
{
    ZludaAirKernel *kernels;
    size_t kernel_count;
    ZludaAirGlobal *globals;
    size_t global_count;
    uint8_t *globals_init; // initial contents of the device-globals buffer
    uint64_t globals_size;
    char *module_text;
} ZludaAirOutput;

int zluda_air_lower(const ZludaAirInput *input, ZludaAirOutput *output, char **error);
void zluda_air_free(ZludaAirOutput *output);
void zluda_air_free_string(char *s);

#ifdef __cplusplus
}
#endif
