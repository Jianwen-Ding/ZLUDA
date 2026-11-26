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
    int portable_air; // keep doubles and directed roundings (for llvm-air2agx's air-emulate-f64)
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
    // Offset in the parameter buffer of a u32 the launch sets to its dynamic shared memory size, or UINT64_MAX
    uint64_t dyn_smem_param_offset;
} ZludaAirKernel;

// A writable module-scope `.global` variable, emulated as a slice of a
// per-module device buffer
typedef struct ZludaAirGlobal
{
    char *name;
    uint64_t offset;
    uint64_t size;
} ZludaAirGlobal;

// An address in a device variable's initializer: the runtime writes the low `size` bytes of
// (address of the device-globals buffer + addend), or with a mask, of (v & mask) >> ctz(mask), at offset `at`
typedef struct ZludaAirReloc
{
    uint64_t at;
    uint64_t size;
    uint64_t addend;
    uint64_t mask;
} ZludaAirReloc;

typedef struct ZludaAirOutput
{
    ZludaAirKernel *kernels;
    size_t kernel_count;
    ZludaAirGlobal *globals;
    size_t global_count;
    uint8_t *globals_init; // initial contents of the device-globals buffer
    uint64_t globals_size;
    char *module_text;
    ZludaAirReloc *relocs;
    size_t reloc_count;
} ZludaAirOutput;

int zluda_air_lower(const ZludaAirInput *input, ZludaAirOutput *output, char **error);
void zluda_air_free(ZludaAirOutput *output);
void zluda_air_free_string(char *s);

#ifdef __cplusplus
}
#endif
