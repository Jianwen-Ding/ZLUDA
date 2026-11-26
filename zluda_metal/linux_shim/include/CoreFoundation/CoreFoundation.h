// Linux shim: the few CoreFoundation types metal-cpp mentions
#pragma once
#include <stdint.h>
#include <stdbool.h>
typedef double CFTimeInterval;
typedef const void *CFTypeRef;
typedef long CFIndex;
typedef struct
{
    CFIndex location;
    CFIndex length;
} CFRange;
typedef uint32_t CFStringEncoding;
typedef const struct __CFString *CFStringRef;
#ifdef __cplusplus
extern "C" {
#endif
CFTypeRef CFRetain(CFTypeRef cf);
void CFRelease(CFTypeRef cf);
#ifdef __cplusplus
}
#endif
#include <math.h> // Apple's CoreFoundation pulls in <math.h> (INFINITY, ...)
