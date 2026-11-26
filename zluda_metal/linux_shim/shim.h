// Internal interface of the Linux metal-cpp shim (objc runtime, Foundation
// subset, software Metal device). Not used on Apple platforms.
#pragma once
#include <objc/runtime.h>
#include <dispatch/dispatch.h>
#include <CoreFoundation/CoreFoundation.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace zshim
{

struct Object;

struct ShimClass
{
    ShimClass *isa = nullptr; // metaclass (class methods)
    const char *name = nullptr;
    ShimClass *super = nullptr;
    bool isMeta = false;
    Object *(*allocator)(ShimClass *) = nullptr; // used by +alloc
    std::unordered_map<SEL, void *> methods;
};

// Every shim object starts with an isa pointer, like an Objective-C object,
// so no virtual functions here (a vtable pointer would come first).
struct Object
{
    ShimClass *isa = nullptr;
    std::atomic<intptr_t> refcount{1};
    void (*destroy)(Object *) = nullptr;
};

template <class T> T *make(ShimClass *cls)
{
    T *o = new T();
    o->isa = cls;
    o->destroy = [](Object *p) { delete static_cast<T *>(p); };
    return o;
}

template <class T> void setAllocator(ShimClass *cls)
{
    cls->allocator = [](ShimClass *c) -> Object * { return make<T>(c); };
}

SEL sel(const char *name);
ShimClass *defineClass(const char *name, ShimClass *super);
void addMethod(ShimClass *cls, const char *selector, void *imp);
void addClassMethod(ShimClass *cls, const char *selector, void *imp);
void *findMethod(ShimClass *cls, SEL s);

// Foundation helpers
ShimClass *rootClass();
void retain(Object *o);
void release(Object *o);
Object *autorelease(Object *o);
// NSString
Object *newString(const std::string &s); // +1
const char *stringUTF8(Object *s);
// NSError
Object *newError(const char *domain, long code, const std::string &description); // +1
// NSArray of retained objects
Object *newArray(Object **objects, size_t count); // +1

// make sure all shim classes are registered (idempotent)
void ensureFoundation();
void ensureMetal();

} // namespace zshim
