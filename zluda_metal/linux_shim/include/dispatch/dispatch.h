// Linux shim: dispatch_data_t / dispatch_queue_t as used by metal-cpp
#pragma once
#include <stddef.h>
typedef struct zluda_dispatch_data_s *dispatch_data_t;
typedef struct zluda_dispatch_queue_s *dispatch_queue_t;
typedef void (^dispatch_block_t)(void);
#ifdef __cplusplus
extern "C" {
#endif
// Copies `buffer` (the DISPATCH_DATA_DESTRUCTOR_DEFAULT behaviour). `queue`
// and `destructor` must be null.
dispatch_data_t dispatch_data_create(const void *buffer, size_t size, dispatch_queue_t queue, void *destructor);
size_t dispatch_data_get_size(dispatch_data_t data);
const void *zluda_dispatch_data_bytes(dispatch_data_t data);
void dispatch_release(void *object);
#ifdef __cplusplus
}
#endif
#define DISPATCH_DATA_DESTRUCTOR_DEFAULT nullptr
