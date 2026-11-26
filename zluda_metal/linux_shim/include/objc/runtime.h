// Linux shim: the subset of the Objective-C runtime interface used by metal-cpp.
// Implemented by linux_shim/objc_runtime.cpp. This is NOT Apple's runtime: it
// is a small message dispatcher that is ABI compatible with how metal-cpp
// calls objc_msgSend (a cast to the concrete C function type).
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__aarch64__) && !defined(__arm64__)
#define __arm64__ 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct objc_class *Class;
struct objc_object
{
    Class isa;
};
typedef struct objc_object *id;
typedef const struct objc_selector *SEL;
typedef id (*IMP)(id, SEL, ...);
typedef bool BOOL;
#define YES true
#define NO false
#ifndef nil
#define nil nullptr
#endif
typedef struct objc_object Protocol;

SEL sel_registerName(const char *str);
const char *sel_getName(SEL sel);
Class objc_lookUpClass(const char *name);
Class objc_getClass(const char *name);
Protocol *objc_getProtocol(const char *name);
BOOL class_respondsToSelector(Class cls, SEL sel);
const char *class_getName(Class cls);
Class object_getClass(id obj);

#ifdef __cplusplus
}
#endif
