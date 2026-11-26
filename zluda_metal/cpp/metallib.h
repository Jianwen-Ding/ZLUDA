// Minimal open implementation of the Metal library (`.metallib`) container.
//
// The layout follows the publicly documented, reverse-engineered description
// of the format (a "MTLB" header, a function list made of tagged records,
// public/private metadata sections and a section holding one wrapped AIR
// bitcode module per function). No Apple tool or code is involved.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ZludaMetallibFunction
{
    const char *name;
    const uint8_t *air; // raw AIR bitcode (starts with 'BC'), wrapped on write
    size_t air_len;
} ZludaMetallibFunction;

// Writes a metallib holding `count` kernel functions. Returns 0 on success.
// The result must be released with zluda_metallib_free.
int zluda_metallib_write(const ZludaMetallibFunction *functions, size_t count, uint8_t **out, size_t *out_len);
void zluda_metallib_free(uint8_t *data);

// Parsing. The returned functions point into `data` (bitcode without the
// wrapper header). Release the array with zluda_metallib_free_functions.
typedef struct ZludaMetallibEntry
{
    char *name;
    uint8_t type; // 2 = kernel
    const uint8_t *air;
    size_t air_len;
    uint8_t hash[32];
} ZludaMetallibEntry;

int zluda_metallib_read(const uint8_t *data, size_t len, ZludaMetallibEntry **entries, size_t *count, char **error);
void zluda_metallib_free_entries(ZludaMetallibEntry *entries, size_t count);

#ifdef __cplusplus
}
#endif
