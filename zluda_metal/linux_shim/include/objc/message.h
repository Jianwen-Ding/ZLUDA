#pragma once
#include <objc/runtime.h>
#ifdef __cplusplus
extern "C" {
#endif
// Declared with an empty parameter list, like Apple's OBJC_OLD_DISPATCH_PROTOTYPES=0
// variant: callers always cast to the precise function type.
void objc_msgSend(void);
void objc_msgSend_stret(void);
void objc_msgSend_fpret(void);
#ifdef __cplusplus
}
#endif
