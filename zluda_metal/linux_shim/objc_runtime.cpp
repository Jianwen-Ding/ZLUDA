// Linux shim: a tiny Objective-C compatible message dispatcher.
//
// metal-cpp calls methods by casting `objc_msgSend` to the exact C function
// type of the method and calling it with (receiver, selector, args...). We
// provide `objc_msgSend` as an assembly trampoline that looks up the IMP for
// (receiver->isa, selector) and tail-jumps to it with all argument registers
// and the stack untouched, which is exactly how the real runtime behaves.
// Classes and methods are registered from C++ (see shim.h).

#include "shim.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

namespace zshim
{

namespace
{
std::mutex &registryMutex()
{
    static std::mutex m;
    return m;
}
std::unordered_map<std::string, SEL> &selectorTable()
{
    static std::unordered_map<std::string, SEL> t;
    return t;
}
std::map<std::string, ShimClass *> &classTable()
{
    static std::map<std::string, ShimClass *> t;
    return t;
}
} // namespace

SEL sel(const char *name) { return sel_registerName(name); }

ShimClass *defineClass(const char *name, ShimClass *super)
{
    std::lock_guard<std::mutex> lock(registryMutex());
    auto &t = classTable();
    auto it = t.find(name);
    if (it != t.end())
        return it->second;
    auto *meta = new ShimClass();
    auto *cls = new ShimClass();
    meta->name = strdup(name);
    meta->super = super ? super->isa : nullptr;
    meta->isMeta = true;
    cls->isa = meta;
    cls->name = meta->name;
    cls->super = super;
    t[name] = cls;
    return cls;
}

void addMethod(ShimClass *cls, const char *selector, void *imp)
{
    std::lock_guard<std::mutex> lock(registryMutex());
    cls->methods[sel_registerName(selector)] = imp;
}

void addClassMethod(ShimClass *cls, const char *selector, void *imp) { addMethod(cls->isa, selector, imp); }

void *findMethod(ShimClass *cls, SEL s)
{
    std::lock_guard<std::mutex> lock(registryMutex());
    for (ShimClass *c = cls; c; c = c->super)
    {
        auto it = c->methods.find(s);
        if (it != c->methods.end())
            return it->second;
    }
    return nullptr;
}

} // namespace zshim

using namespace zshim;

extern "C"
{
    SEL sel_registerName(const char *str)
    {
        static std::mutex m;
        std::lock_guard<std::mutex> lock(m);
        auto &t = selectorTable();
        auto it = t.find(str);
        if (it != t.end())
            return it->second;
        SEL s = (SEL)strdup(str);
        t.emplace(str, s);
        return s;
    }

    const char *sel_getName(SEL sel) { return (const char *)sel; }

    Class objc_lookUpClass(const char *name)
    {
        // metal-cpp resolves classes during static initialization, possibly
        // before the shim's own initializers ran: register on first use.
        ensureFoundation();
        ensureMetal();
        std::lock_guard<std::mutex> lock(registryMutex());
        auto &t = classTable();
        auto it = t.find(name);
        return it == t.end() ? nullptr : (Class)it->second;
    }

    Class objc_getClass(const char *name) { return objc_lookUpClass(name); }

    Protocol *objc_getProtocol(const char *) { return nullptr; }

    BOOL class_respondsToSelector(Class cls, SEL sel) { return findMethod((ShimClass *)cls, sel) != nullptr; }

    const char *class_getName(Class cls) { return cls ? ((ShimClass *)cls)->name : "nil"; }

    Class object_getClass(id obj) { return obj ? obj->isa : nullptr; }

    // Called by the trampolines
    void *zluda_shim_lookup(id self, SEL s)
    {
        ShimClass *cls = (ShimClass *)self->isa;
        void *imp = findMethod(cls, s);
        if (!imp)
        {
            fprintf(stderr, "[zluda metal shim] %s%s does not recognize selector '%s'\n", cls->isMeta ? "+" : "-",
                    cls->name, (const char *)s);
            abort();
        }
        return imp;
    }

    // Blocks runtime symbols referenced by metal-cpp headers
    void *_NSConcreteStackBlock[32];
    void *_NSConcreteGlobalBlock[32];
    void _Block_object_assign(void *, const void *, const int) {}
    void _Block_object_dispose(const void *, const int) {}

    CFTypeRef CFRetain(CFTypeRef cf) { return cf; }
    void CFRelease(CFTypeRef) {}
}

// ---------------------------------------------------------------------------
// objc_msgSend / objc_msgSend_stret / objc_msgSend_fpret trampolines
// ---------------------------------------------------------------------------
#if defined(__x86_64__)
__asm__(R"(
    .text
    .globl objc_msgSend
    .globl objc_msgSend_fpret
    .type objc_msgSend, @function
    .type objc_msgSend_fpret, @function
objc_msgSend:
objc_msgSend_fpret:
    testq %rdi, %rdi
    jz 1f
    subq $0xc8, %rsp
    movq %rdi, 0x00(%rsp)
    movq %rsi, 0x08(%rsp)
    movq %rdx, 0x10(%rsp)
    movq %rcx, 0x18(%rsp)
    movq %r8,  0x20(%rsp)
    movq %r9,  0x28(%rsp)
    movq %rax, 0x30(%rsp)
    movdqu %xmm0, 0x38(%rsp)
    movdqu %xmm1, 0x48(%rsp)
    movdqu %xmm2, 0x58(%rsp)
    movdqu %xmm3, 0x68(%rsp)
    movdqu %xmm4, 0x78(%rsp)
    movdqu %xmm5, 0x88(%rsp)
    movdqu %xmm6, 0x98(%rsp)
    movdqu %xmm7, 0xa8(%rsp)
    call zluda_shim_lookup
    movq %rax, %r11
    movq 0x00(%rsp), %rdi
    movq 0x08(%rsp), %rsi
    movq 0x10(%rsp), %rdx
    movq 0x18(%rsp), %rcx
    movq 0x20(%rsp), %r8
    movq 0x28(%rsp), %r9
    movq 0x30(%rsp), %rax
    movdqu 0x38(%rsp), %xmm0
    movdqu 0x48(%rsp), %xmm1
    movdqu 0x58(%rsp), %xmm2
    movdqu 0x68(%rsp), %xmm3
    movdqu 0x78(%rsp), %xmm4
    movdqu 0x88(%rsp), %xmm5
    movdqu 0x98(%rsp), %xmm6
    movdqu 0xa8(%rsp), %xmm7
    addq $0xc8, %rsp
    jmp *%r11
1:
    xorl %eax, %eax
    xorl %edx, %edx
    pxor %xmm0, %xmm0
    pxor %xmm1, %xmm1
    ret
    .size objc_msgSend, .-objc_msgSend

    .globl objc_msgSend_stret
    .type objc_msgSend_stret, @function
objc_msgSend_stret:
    testq %rsi, %rsi
    jz 2f
    subq $0xc8, %rsp
    movq %rdi, 0x00(%rsp)
    movq %rsi, 0x08(%rsp)
    movq %rdx, 0x10(%rsp)
    movq %rcx, 0x18(%rsp)
    movq %r8,  0x20(%rsp)
    movq %r9,  0x28(%rsp)
    movq %rax, 0x30(%rsp)
    movdqu %xmm0, 0x38(%rsp)
    movdqu %xmm1, 0x48(%rsp)
    movdqu %xmm2, 0x58(%rsp)
    movdqu %xmm3, 0x68(%rsp)
    movdqu %xmm4, 0x78(%rsp)
    movdqu %xmm5, 0x88(%rsp)
    movdqu %xmm6, 0x98(%rsp)
    movdqu %xmm7, 0xa8(%rsp)
    movq %rsi, %rdi
    movq %rdx, %rsi
    call zluda_shim_lookup
    movq %rax, %r11
    movq 0x00(%rsp), %rdi
    movq 0x08(%rsp), %rsi
    movq 0x10(%rsp), %rdx
    movq 0x18(%rsp), %rcx
    movq 0x20(%rsp), %r8
    movq 0x28(%rsp), %r9
    movq 0x30(%rsp), %rax
    movdqu 0x38(%rsp), %xmm0
    movdqu 0x48(%rsp), %xmm1
    movdqu 0x58(%rsp), %xmm2
    movdqu 0x68(%rsp), %xmm3
    movdqu 0x78(%rsp), %xmm4
    movdqu 0x88(%rsp), %xmm5
    movdqu 0x98(%rsp), %xmm6
    movdqu 0xa8(%rsp), %xmm7
    addq $0xc8, %rsp
    jmp *%r11
2:
    movq %rdi, %rax
    ret
    .size objc_msgSend_stret, .-objc_msgSend_stret
    .section .note.GNU-stack,"",@progbits
)");
#elif defined(__aarch64__)
// On arm64 struct returns go through x8, so objc_msgSend covers everything.
__asm__(R"(
    .text
    .globl objc_msgSend
    .globl objc_msgSend_fpret
    .globl objc_msgSend_stret
    .type objc_msgSend, %function
objc_msgSend:
objc_msgSend_fpret:
objc_msgSend_stret:
    cbz x0, 1f
    stp x29, x30, [sp, #-224]!
    mov x29, sp
    stp x0, x1, [sp, #16]
    stp x2, x3, [sp, #32]
    stp x4, x5, [sp, #48]
    stp x6, x7, [sp, #64]
    str x8, [sp, #80]
    stp q0, q1, [sp, #96]
    stp q2, q3, [sp, #128]
    stp q4, q5, [sp, #160]
    stp q6, q7, [sp, #192]
    bl zluda_shim_lookup
    mov x16, x0
    ldp x0, x1, [sp, #16]
    ldp x2, x3, [sp, #32]
    ldp x4, x5, [sp, #48]
    ldp x6, x7, [sp, #64]
    ldr x8, [sp, #80]
    ldp q0, q1, [sp, #96]
    ldp q2, q3, [sp, #128]
    ldp q4, q5, [sp, #160]
    ldp q6, q7, [sp, #192]
    ldp x29, x30, [sp], #224
    br x16
1:
    mov x0, #0
    mov x1, #0
    movi d0, #0
    ret
    .size objc_msgSend, .-objc_msgSend
    .section .note.GNU-stack,"",@progbits
)");
#else
#error "zluda metal shim: unsupported architecture"
#endif
