// Linux shim: the Foundation subset needed by metal-cpp users
// (NSObject memory management, NSString, NSError, NSArray, NSAutoreleasePool).
#include "shim.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace zshim
{

namespace
{
struct StringObj : Object
{
    std::string value;
};
struct ErrorObj : Object
{
    long code = 0;
    Object *domain = nullptr;
    Object *description = nullptr;
};
struct ArrayObj : Object
{
    std::vector<Object *> items;
};
struct PoolObj : Object
{
    std::vector<Object *> objects;
};

thread_local std::vector<PoolObj *> poolStack;

ShimClass *g_root, *g_string, *g_error, *g_array, *g_pool;

// --- NSObject ---
id objAlloc(ShimClass *cls, SEL)
{
    for (ShimClass *c = cls; c; c = c->super)
        if (c->allocator)
            return (id)c->allocator(cls);
    Object *o = make<Object>(cls);
    return (id)o;
}
id objInit(id self, SEL) { return self; }
id objNew(ShimClass *cls, SEL s) { return objAlloc(cls, s); }
id objRetain(id self, SEL)
{
    retain((Object *)self);
    return self;
}
void objRelease(id self, SEL) { release((Object *)self); }
id objAutorelease(id self, SEL) { return (id)autorelease((Object *)self); }
uintptr_t objRetainCount(id self, SEL) { return (uintptr_t)((Object *)self)->refcount.load(); }
bool objResponds(id self, SEL, SEL s) { return findMethod(((Object *)self)->isa, s) != nullptr; }
id objMethodSignature(id, SEL, SEL) { return nullptr; }
uintptr_t objHash(id self, SEL) { return (uintptr_t)self; }
bool objIsEqual(id self, SEL, id other) { return self == other; }
id objDescription(id self, SEL)
{
    char buf[128];
    snprintf(buf, sizeof buf, "<%s: %p>", ((Object *)self)->isa->name, (void *)self);
    return (id)autorelease(newString(buf));
}
id objClass(id self, SEL) { return (id)((Object *)self)->isa; }
bool clsResponds(ShimClass *cls, SEL, SEL s) { return findMethod(cls->isa, s) != nullptr; }

// --- NSString ---
id strWithUTF8(ShimClass *, SEL, const char *s) { return (id)autorelease(newString(s ? s : "")); }
id strWithCString(ShimClass *, SEL, const char *s, uintptr_t) { return (id)autorelease(newString(s ? s : "")); }
id strInitUTF8(id self, SEL, const char *s)
{
    ((StringObj *)self)->value = s ? s : "";
    return self;
}
id strInitCString(id self, SEL, const char *s, uintptr_t) { return strInitUTF8(self, nullptr, s); }
id strInitBytes(id self, SEL, const void *b, uintptr_t len, uintptr_t)
{
    ((StringObj *)self)->value.assign((const char *)b, len);
    return self;
}
const char *strUTF8(id self, SEL) { return ((StringObj *)self)->value.c_str(); }
const char *strCString(id self, SEL, uintptr_t) { return ((StringObj *)self)->value.c_str(); }
uintptr_t strLength(id self, SEL) { return ((StringObj *)self)->value.size(); }
uintptr_t strLengthOfBytes(id self, SEL, uintptr_t) { return ((StringObj *)self)->value.size(); }
bool strIsEqualToString(id self, SEL, id other)
{
    return other && ((StringObj *)self)->value == ((StringObj *)other)->value;
}
uintptr_t strHash(id self, SEL) { return std::hash<std::string>()(((StringObj *)self)->value); }
id strDescription(id self, SEL) { return self; }

// --- NSError ---
intptr_t errCode(id self, SEL) { return ((ErrorObj *)self)->code; }
id errDomain(id self, SEL) { return (id)((ErrorObj *)self)->domain; }
id errLocalizedDescription(id self, SEL) { return (id)((ErrorObj *)self)->description; }
id errUserInfo(id, SEL) { return nullptr; }

// --- NSArray ---
uintptr_t arrCount(id self, SEL) { return ((ArrayObj *)self)->items.size(); }
id arrObjectAtIndex(id self, SEL, uintptr_t i)
{
    auto &v = ((ArrayObj *)self)->items;
    return i < v.size() ? (id)v[i] : nullptr;
}
id arrWithObjects(ShimClass *, SEL, Object **objs, uintptr_t n) { return (id)autorelease(newArray(objs, n)); }

// --- NSAutoreleasePool ---
id poolInit(id self, SEL)
{
    poolStack.push_back((PoolObj *)self);
    return self;
}
void poolDrain(id self, SEL)
{
    PoolObj *p = (PoolObj *)self;
    while (!poolStack.empty())
    {
        PoolObj *top = poolStack.back();
        poolStack.pop_back();
        std::vector<Object *> objs;
        objs.swap(top->objects);
        for (Object *o : objs)
            release(o);
        if (top == p)
            break;
    }
    if (p->refcount.fetch_sub(1) == 1)
        p->destroy(p);
}

void registerAll()
{
    g_root = defineClass("NSObject", nullptr);
    addClassMethod(g_root, "alloc", (void *)objAlloc);
    addClassMethod(g_root, "new", (void *)objNew);
    addClassMethod(g_root, "respondsToSelector:", (void *)clsResponds);
    addClassMethod(g_root, "instancesRespondToSelector:", (void *)clsResponds);
    addMethod(g_root, "init", (void *)objInit);
    addMethod(g_root, "retain", (void *)objRetain);
    addMethod(g_root, "release", (void *)objRelease);
    addMethod(g_root, "autorelease", (void *)objAutorelease);
    addMethod(g_root, "retainCount", (void *)objRetainCount);
    addMethod(g_root, "respondsToSelector:", (void *)objResponds);
    addMethod(g_root, "methodSignatureForSelector:", (void *)objMethodSignature);
    addMethod(g_root, "hash", (void *)objHash);
    addMethod(g_root, "isEqual:", (void *)objIsEqual);
    addMethod(g_root, "description", (void *)objDescription);
    addMethod(g_root, "debugDescription", (void *)objDescription);
    addMethod(g_root, "class", (void *)objClass);

    g_string = defineClass("NSString", g_root);
    setAllocator<StringObj>(g_string);
    addClassMethod(g_string, "stringWithUTF8String:", (void *)strWithUTF8);
    addClassMethod(g_string, "stringWithCString:encoding:", (void *)strWithCString);
    addMethod(g_string, "initWithUTF8String:", (void *)strInitUTF8);
    addMethod(g_string, "initWithCString:encoding:", (void *)strInitCString);
    addMethod(g_string, "initWithBytes:length:encoding:", (void *)strInitBytes);
    addMethod(g_string, "initWithBytesNoCopy:length:encoding:freeWhenDone:", (void *)strInitBytes);
    addMethod(g_string, "UTF8String", (void *)strUTF8);
    addMethod(g_string, "cStringUsingEncoding:", (void *)strCString);
    addMethod(g_string, "length", (void *)strLength);
    addMethod(g_string, "lengthOfBytesUsingEncoding:", (void *)strLengthOfBytes);
    addMethod(g_string, "maximumLengthOfBytesUsingEncoding:", (void *)strLengthOfBytes);
    addMethod(g_string, "isEqualToString:", (void *)strIsEqualToString);
    addMethod(g_string, "isEqual:", (void *)strIsEqualToString);
    addMethod(g_string, "hash", (void *)strHash);
    addMethod(g_string, "description", (void *)strDescription);

    g_error = defineClass("NSError", g_root);
    setAllocator<ErrorObj>(g_error);
    addMethod(g_error, "code", (void *)errCode);
    addMethod(g_error, "domain", (void *)errDomain);
    addMethod(g_error, "localizedDescription", (void *)errLocalizedDescription);
    addMethod(g_error, "localizedFailureReason", (void *)errLocalizedDescription);
    addMethod(g_error, "description", (void *)errLocalizedDescription);
    addMethod(g_error, "userInfo", (void *)errUserInfo);

    g_array = defineClass("NSArray", g_root);
    setAllocator<ArrayObj>(g_array);
    addClassMethod(g_array, "arrayWithObjects:count:", (void *)arrWithObjects);
    addMethod(g_array, "count", (void *)arrCount);
    addMethod(g_array, "objectAtIndex:", (void *)arrObjectAtIndex);

    g_pool = defineClass("NSAutoreleasePool", g_root);
    setAllocator<PoolObj>(g_pool);
    addMethod(g_pool, "init", (void *)poolInit);
    addMethod(g_pool, "drain", (void *)poolDrain);
    addMethod(g_pool, "release", (void *)poolDrain);
}

struct Registrar
{
    Registrar() { ensureFoundation(); }
} registrar;
} // namespace

void ensureFoundation()
{
    static std::once_flag once;
    std::call_once(once, registerAll);
}

ShimClass *rootClass()
{
    ensureFoundation();
    return g_root;
}

void retain(Object *o)
{
    if (o)
        o->refcount.fetch_add(1);
}

void release(Object *o)
{
    if (o && o->refcount.fetch_sub(1) == 1)
        o->destroy(o);
}

Object *autorelease(Object *o)
{
    if (!o)
        return o;
    if (poolStack.empty())
    {
        // No pool: leak, like the real runtime does (it also warns)
        return o;
    }
    poolStack.back()->objects.push_back(o);
    return o;
}

Object *newString(const std::string &s)
{
    ensureFoundation();
    StringObj *o = make<StringObj>(g_string);
    o->value = s;
    return o;
}

const char *stringUTF8(Object *s) { return s ? ((StringObj *)s)->value.c_str() : ""; }

Object *newError(const char *domain, long code, const std::string &description)
{
    ensureFoundation();
    ErrorObj *e = make<ErrorObj>(g_error);
    e->code = code;
    e->domain = newString(domain);
    e->description = newString(description);
    auto destroyBase = e->destroy;
    (void)destroyBase;
    e->destroy = [](Object *p) {
        ErrorObj *err = (ErrorObj *)p;
        release(err->domain);
        release(err->description);
        delete err;
    };
    return e;
}

Object *newArray(Object **objects, size_t count)
{
    ensureFoundation();
    ArrayObj *a = make<ArrayObj>(g_array);
    for (size_t i = 0; i < count; i++)
    {
        retain(objects[i]);
        a->items.push_back(objects[i]);
    }
    a->destroy = [](Object *p) {
        ArrayObj *arr = (ArrayObj *)p;
        for (Object *o : arr->items)
            release(o);
        delete arr;
    };
    return a;
}

} // namespace zshim

extern "C"
{
    struct zluda_dispatch_data_s
    {
        std::vector<unsigned char> bytes;
    };

    dispatch_data_t dispatch_data_create(const void *buffer, size_t size, dispatch_queue_t, void *)
    {
        auto *d = new zluda_dispatch_data_s();
        d->bytes.assign((const unsigned char *)buffer, (const unsigned char *)buffer + size);
        return d;
    }
    size_t dispatch_data_get_size(dispatch_data_t data) { return data ? data->bytes.size() : 0; }
    const void *zluda_dispatch_data_bytes(dispatch_data_t data) { return data ? data->bytes.data() : nullptr; }
    void dispatch_release(void *object) { delete (zluda_dispatch_data_s *)object; }
}
