// ZLUDA IR -> Metal AIR lowering.
//
// Input: the LLVM module produced by ZLUDA's PTX translator (ptx crate). That
// module uses the AMDGPU address space numbering (flat 0, global 1, shared 3,
// constant 4, private 5) and AMDGPU kernel calling convention, and calls the
// `__zluda_ptx_impl_*` helpers, which we link from zluda_ptx_metal_impl.bc.
//
// Output: one AIR module per kernel, i.e. LLVM bitcode following the Metal AIR
// conventions (air64 triple and data layout, AIR address spaces, kernel
// argument metadata in !air.kernel, air.* intrinsics). These modules are the
// `.air` files that metallib.cpp packs into a `.metallib` container.
//
// Everything here is implemented from the publicly observable AIR conventions
// (the shape of bitcode emitted for Metal Shading Language) and LLVM's own
// APIs; it does not use or link any Apple tool, header or library.

#include "air_lower.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Support/KnownBits.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/ReplaceConstant.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/DCE.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/InferAddressSpaces.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace llvm;

namespace
{

// ---------------------------------------------------------------------------
// Address spaces
// ---------------------------------------------------------------------------
namespace amd
{
constexpr unsigned Flat = 0, Global = 1, Shared = 3, Constant = 4, Private = 5, Constant32 = 6;
}
namespace air
{
constexpr unsigned Thread = 0, Device = 1, Constant = 2, Threadgroup = 3;
}

const char *AIR_TRIPLE = "air64-apple-macosx14.0.0";
const char *AIR_DATALAYOUT =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-"
    "v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-"
    "v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

struct LowerError : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const Twine &msg) { throw LowerError(msg.str()); }

std::string toString(const Value *v)
{
    std::string s;
    raw_string_ostream os(s);
    v->print(os);
    return s;
}

unsigned mapAddressSpace(unsigned as)
{
    switch (as)
    {
    case amd::Private:
        return air::Thread;
    case amd::Flat: // generic pointers that survived inference: assume device memory
    case amd::Global:
        return air::Device;
    case amd::Constant:
    case amd::Constant32:
        return air::Constant;
    case amd::Shared:
        return air::Threadgroup;
    default:
        return air::Device;
    }
}

Type *mapType(Type *t)
{
    if (auto *pt = dyn_cast<PointerType>(t))
        return PointerType::get(t->getContext(), mapAddressSpace(pt->getAddressSpace()));
    if (auto *vt = dyn_cast<VectorType>(t))
    {
        Type *e = mapType(vt->getElementType());
        if (e != vt->getElementType())
            return VectorType::get(e, vt->getElementCount());
        return t;
    }
    if (auto *at = dyn_cast<ArrayType>(t))
    {
        Type *e = mapType(at->getElementType());
        if (e != at->getElementType())
            return ArrayType::get(e, at->getNumElements());
        return t;
    }
    if (auto *st = dyn_cast<StructType>(t))
    {
        bool changed = false;
        SmallVector<Type *, 8> elems;
        for (Type *e : st->elements())
        {
            Type *m = mapType(e);
            changed |= m != e;
            elems.push_back(m);
        }
        if (!changed)
            return t;
        if (!st->isLiteral())
            fail("named struct with pointer members is not supported: " + st->getName());
        return StructType::get(t->getContext(), elems, st->isPacked());
    }
    if (auto *ft = dyn_cast<FunctionType>(t))
    {
        SmallVector<Type *, 8> params;
        for (Type *p : ft->params())
            params.push_back(mapType(p));
        return FunctionType::get(mapType(ft->getReturnType()), params, ft->isVarArg());
    }
    return t;
}

// ---------------------------------------------------------------------------
// Pass pipeline helpers
// ---------------------------------------------------------------------------
void runCleanup(Module &m, bool inlineAll, bool inferAddressSpaces)
{
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;
    PassBuilder pb;
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    ModulePassManager mpm;
    if (inlineAll)
        mpm.addPass(AlwaysInlinerPass());
    FunctionPassManager fpm;
    fpm.addPass(SROAPass(SROAOptions::ModifyCFG));
    fpm.addPass(EarlyCSEPass());
    fpm.addPass(InstCombinePass());
    fpm.addPass(SimplifyCFGPass());
    if (inferAddressSpaces)
    {
        fpm.addPass(InferAddressSpacesPass(amd::Flat));
        fpm.addPass(InstCombinePass());
    }
    fpm.addPass(DCEPass());
    mpm.addPass(createModuleToFunctionPassAdaptor(std::move(fpm)));
    mpm.addPass(GlobalDCEPass());
    mpm.run(m, mam);
}

// ---------------------------------------------------------------------------
// Integer pointer arithmetic -> GEPs
//
// PTX does pointer arithmetic on integers (cvta, add.s64, ...). The translator
// faithfully emits ptrtoint/add/inttoptr. Metal cannot turn an integer into a
// threadgroup or thread pointer, so we recover the pointer provenance: any
// integer that is (ptrtoint P) + offsets, possibly through phi/select, is
// rewritten as a GEP chain on P.
// ---------------------------------------------------------------------------
// Thread-block clusters (.reqnctapercluster, function attribute "zluda-cluster-dims" of the translator): a cluster of
// C CTAs runs as one AGX threadgroup ("agx-cluster-dims", llvm/docs/AGXUsage.rst: the launch divides the grid by the
// cluster dims and multiplies the threadgroup's z by C). The special registers of the helper library derive from
// the placeholders __zluda_metal_cluster_dim(d) / __zluda_metal_explicit_cluster(), resolved here per kernel; each CTA
// gets its own copy of the kernel's .shared variables: they are laid out in one block of `stride` bytes, repeated C
// times (__zluda_cta_shared), and CTA `rank` uses the copy at rank * stride (rank = tid.z / (ntid.z / C), the
// threadgroup's z). mapa / getctarank use __zluda_metal_cluster_base() / __zluda_metal_cluster_stride(). Variables of
// the helper library named __zluda_cluster_* stay shared by the cluster.
static void clusterDimsOf(Function &f, unsigned d[3], bool *expl)
{
    d[0] = d[1] = d[2] = 1;
    *expl = f.hasFnAttribute("zluda-explicit-cluster");
    Attribute a = f.getFnAttribute("zluda-cluster-dims");
    if (!a.isValid())
        return;
    SmallVector<StringRef, 3> parts;
    a.getValueAsString().split(parts, ',');
    for (unsigned i = 0; i < 3 && i < parts.size(); i++)
        parts[i].trim().getAsInteger(10, d[i]);
    *expl = true;
}

static void resolveCluster(Module &m)
{
    LLVMContext &c = m.getContext();
    Type *i32 = Type::getInt32Ty(c);
    // the cluster of each function (after inlining, the kernels)
    auto dimsOf = [&](Function *f, unsigned d[3], bool *e) { clusterDimsOf(*f, d, e); };
    // per-CTA .shared variables of the kernels with clusters of more than one CTA
    std::vector<GlobalVariable *> vars;
    unsigned cd[3] = {1, 1, 1};
    bool expl = false;
    Function *kernel = nullptr;
    for (Function &f : m)
        if (!f.isDeclaration() && f.hasFnAttribute("zluda-cluster-dims"))
        {
            unsigned d[3];
            bool e;
            dimsOf(&f, d, &e);
            if (d[0] * d[1] * d[2] > 1 && !kernel)
            {
                kernel = &f;
                cd[0] = d[0], cd[1] = d[1], cd[2] = d[2];
                expl = e;
            }
        }
    unsigned C = cd[0] * cd[1] * cd[2];
    GlobalVariable *block = nullptr;
    uint64_t stride = 0;
    if (kernel)
    {
        const DataLayout &dl = m.getDataLayout();
        for (GlobalVariable &gv : m.globals())
        {
            if (gv.getAddressSpace() != amd::Shared || gv.getName().contains("__zluda_cluster_") || gv.use_empty())
                continue;
            if (auto *at = dyn_cast<ArrayType>(gv.getValueType()); at && at->getNumElements() == 0)
                continue; // dynamic shared memory: not per CTA (not supported with clusters)
            vars.push_back(&gv);
        }
        uint64_t off = 0, maxAlign = 16;
        std::vector<uint64_t> offs;
        for (GlobalVariable *gv : vars)
        {
            uint64_t al = std::max<uint64_t>(gv->getAlign() ? gv->getAlign()->value() : 1,
                                             dl.getABITypeAlign(gv->getValueType()).value());
            maxAlign = std::max(maxAlign, al);
            off = alignTo(off, al);
            offs.push_back(off);
            off += dl.getTypeAllocSize(gv->getValueType());
        }
        stride = alignTo(std::max<uint64_t>(off, 1), maxAlign);
        ArrayType *bt = ArrayType::get(Type::getInt8Ty(c), stride * C);
        block = new GlobalVariable(m, bt, false, GlobalValue::InternalLinkage, UndefValue::get(bt), "__zluda_cta_shared",
                                   nullptr, GlobalValue::NotThreadLocal, amd::Shared);
        block->setAlignment(Align(maxAlign));
        block->setMetadata("zluda.cluster", MDNode::get(c, {ConstantAsMetadata::get(ConstantInt::get(i32, stride)),
                                                           ConstantAsMetadata::get(ConstantInt::get(i32, C))}));
        FunctionCallee tpos = m.getOrInsertFunction("__zluda_metal_thread_position_in_threadgroup", i32, i32);
        FunctionCallee tptg = m.getOrInsertFunction("__zluda_metal_threads_per_threadgroup", i32, i32);
        std::map<Function *, Value *> rankOf;
        auto rank = [&](Function *f) -> Value * {
            auto it = rankOf.find(f);
            if (it != rankOf.end())
                return it->second;
            IRBuilder<> b(&*f->getEntryBlock().getFirstInsertionPt());
            Value *tz = b.CreateCall(tpos, {b.getInt32(2)});
            Value *nz = b.CreateUDiv(b.CreateCall(tptg, {b.getInt32(2)}), b.getInt32(C));
            Value *r = b.CreateUDiv(tz, nz, "cta.rank");
            rankOf[f] = r;
            return r;
        };
        for (size_t k = 0; k < vars.size(); k++)
        {
            GlobalVariable *gv = vars[k];
            convertUsersOfConstantsToInstructions({gv});
            std::vector<Use *> uses;
            for (Use &u : gv->uses())
                uses.push_back(&u);
            for (Use *u : uses)
            {
                auto *i = dyn_cast<Instruction>(u->getUser());
                if (!i)
                    continue;
                Function *f = i->getFunction();
                Value *r = rank(f);
                IRBuilder<> b(cast<Instruction>(r)->getNextNode());
                Value *o = b.CreateAdd(b.CreateMul(b.CreateZExt(r, b.getInt64Ty()), b.getInt64(stride)), b.getInt64(offs[k]));
                Value *p = b.CreateGEP(b.getInt8Ty(), block, o);
                if (p->getType() != gv->getType())
                    p = b.CreateAddrSpaceCast(p, gv->getType());
                u->set(p);
            }
        }
        for (GlobalVariable *gv : vars)
            if (gv->use_empty())
                gv->eraseFromParent();
    }
    // placeholders
    for (const char *name : {"__zluda_metal_cluster_dim", "__zluda_metal_explicit_cluster", "__zluda_metal_cluster_stride",
                             "__zluda_metal_cluster_base"})
    {
        Function *ph = m.getFunction(name);
        if (!ph)
            continue;
        std::vector<CallInst *> calls;
        for (User *u : ph->users())
            if (auto *ci = dyn_cast<CallInst>(u))
                calls.push_back(ci);
        for (CallInst *ci : calls)
        {
            unsigned d[3];
            bool e;
            dimsOf(ci->getFunction(), d, &e);
            bool mine = ci->getFunction() == kernel;
            Value *v;
            std::string n = name;
            if (n == "__zluda_metal_cluster_dim")
            {
                auto *k = dyn_cast<ConstantInt>(ci->getArgOperand(0));
                unsigned dim = k ? (unsigned)k->getZExtValue() : 0;
                v = ConstantInt::get(ci->getType(), dim < 3 ? d[dim] : 1);
            }
            else if (n == "__zluda_metal_explicit_cluster")
                v = ConstantInt::get(ci->getType(), e ? 1 : 0);
            else if (n == "__zluda_metal_cluster_stride")
                v = ConstantInt::get(ci->getType(), mine ? stride : 0);
            else
                v = mine && block ? (Value *)ConstantExpr::getPointerCast(block, ci->getType())
                                  : (Value *)Constant::getNullValue(ci->getType());
            ci->replaceAllUsesWith(v);
            ci->eraseFromParent();
        }
    }
    if (kernel)
    {
        std::string v = std::to_string(cd[0]) + "," + std::to_string(cd[1]) + "," + std::to_string(cd[2]);
        kernel->addFnAttr("zluda-cluster-dims-agx", v);
    }
    (void)expl;
}

// NaN constants: PTX arithmetic producing a NaN gives the canonical NaN (all ones but the sign: 0x7fffffff for .f32),
// which the AGX hardware produces at run time (+canonical-nan); LLVM's constant folding of f32 gives its default quiet NaN
// (0x7fc00000) instead. NaN constants of the program (bit patterns of the PTX source, the helper library) are hidden
// behind opaque calls during the cleanups; afterwards every default NaN constant comes from folding and is replaced by
// the canonical NaN, and the program's constants are restored. (f64 keeps NaN payloads in PTX, as LLVM's folding does.)
static void hideSourceNaNs(Module &m)
{
    std::vector<std::pair<Instruction *, unsigned>> work;
    for (Function &f : m)
        for (Instruction &i : instructions(f))
            for (unsigned k = 0; k < i.getNumOperands(); k++)
                if (auto *cf = dyn_cast<ConstantFP>(i.getOperand(k)); cf && cf->isNaN() && cf->getType()->isFloatTy())
                    work.push_back({&i, k});
    for (auto [i, k] : work)
    {
        auto *cf = cast<ConstantFP>(i->getOperand(k));
        Type *t = cf->getType();
        Type *it = IntegerType::get(m.getContext(), t->getPrimitiveSizeInBits());
        std::string name = "__zluda_nan_keep." + std::to_string(t->getTypeID()) + "." + std::to_string(it->getIntegerBitWidth());
        FunctionCallee keep = m.getOrInsertFunction(name, FunctionType::get(t, {it}, false));
        if (auto *fn = dyn_cast<Function>(keep.getCallee()))
        {
            fn->setDoesNotThrow();
            fn->setDoesNotAccessMemory();
            fn->setWillReturn();
        }
        Instruction *at = i;
        if (auto *phi = dyn_cast<PHINode>(i))
            at = phi->getIncomingBlock(k)->getTerminator();
        IRBuilder<> b(at);
        Value *v = b.CreateCall(keep, {ConstantInt::get(it, cf->getValueAPF().bitcastToAPInt())});
        i->setOperand(k, v);
    }
}

static void canonicalizeFoldedNaNs(Module &m)
{
    for (Function &f : m)
        for (Instruction &i : instructions(f))
            for (unsigned k = 0; k < i.getNumOperands(); k++)
                if (auto *cf = dyn_cast<ConstantFP>(i.getOperand(k)); cf && cf->isNaN() && cf->getType()->isFloatTy())
                {
                    const fltSemantics &sem = cf->getValueAPF().getSemantics();
                    APInt bits = cf->getValueAPF().bitcastToAPInt();
                    if (bits != APFloat::getQNaN(sem).bitcastToAPInt())
                        continue;
                    APInt canon = APInt::getSignedMaxValue(bits.getBitWidth());
                    i.setOperand(k, ConstantFP::get(cf->getType(), APFloat(sem, canon)));
                }
    std::vector<Function *> keeps;
    for (Function &f : m)
        if (f.getName().starts_with("__zluda_nan_keep."))
            keeps.push_back(&f);
    for (Function *fn : keeps)
    {
        std::vector<CallInst *> calls;
        for (User *u : fn->users())
            if (auto *ci = dyn_cast<CallInst>(u))
                calls.push_back(ci);
        for (CallInst *ci : calls)
        {
            auto *bits = cast<ConstantInt>(ci->getArgOperand(0));
            ci->replaceAllUsesWith(ConstantFP::get(ci->getType(), APFloat(ci->getType()->getFltSemantics(), bits->getValue())));
            ci->eraseFromParent();
        }
        if (fn->use_empty())
            fn->eraseFromParent();
    }
}

// alloca / stacksave / stackrestore (ptx/lib/metal_impl/memory.inc): a per-thread stack, an entry-block alloca of
// each kernel, and its stack offset (the stack grows down from its end)
static void resolveLocalStack(Module &m)
{
    Function *base = m.getFunction("__zluda_metal_local_stack");
    Function *sp = m.getFunction("__zluda_metal_local_sp");
    const unsigned size = 2048;
    std::map<Function *, std::pair<Value *, Value *>> perKernel;
    for (Function *ph : {base, sp})
    {
        if (!ph)
            continue;
        std::vector<CallInst *> calls;
        for (User *u : ph->users())
            if (auto *ci = dyn_cast<CallInst>(u))
                calls.push_back(ci);
        for (CallInst *ci : calls)
        {
            Function *f = ci->getFunction();
            auto it = perKernel.find(f);
            if (it == perKernel.end())
            {
                IRBuilder<> b(&*f->getEntryBlock().getFirstInsertionPt());
                AllocaInst *stack = b.CreateAlloca(ArrayType::get(b.getInt8Ty(), size), amd::Private, nullptr, "ptx.stack");
                stack->setAlignment(Align(64));
                AllocaInst *off = b.CreateAlloca(b.getInt32Ty(), amd::Private, nullptr, "ptx.sp");
                b.CreateStore(b.getInt32(size), off);
                it = perKernel.emplace(f, std::make_pair((Value *)stack, (Value *)off)).first;
            }
            Value *v = ph == base ? it->second.first : it->second.second;
            if (v->getType() != ci->getType())
                v = new AddrSpaceCastInst(v, ci->getType(), "", ci);
            ci->replaceAllUsesWith(v);
            ci->eraseFromParent();
        }
    }
}

// Generic addresses that stay unresolved (PTX ISA 6.4.1.1 generic windows): a cvta of a .shared or .local address whose
// generic value escapes static resolution (it goes through memory, or through integer arithmetic PointerRecovery
// cannot follow) is the object's 32-bit address in a window of the generic address space, and the generic loads and
// stores of the function then test the windows at run time. Only functions with such an escaping address pay for the
// test; ordinary global-memory code is unchanged. A local window access is relative to one private object of the
// function (the one whose address escapes).
static const uint64_t GW_SHARED = 0x01000000, GW_LOCAL = 0x02000000, GW_SIZE = 0x01000000;
static std::set<Function *> g_windowed;
static void scanGenericWindows(Module &m)
{
    g_windowed.clear();
    for (Function &f : m)
        for (Instruction &i : instructions(f))
            if (auto *asc = dyn_cast<AddrSpaceCastInst>(&i))
                if (asc->getDestAddressSpace() == amd::Flat &&
                    (asc->getSrcAddressSpace() == amd::Shared || asc->getSrcAddressSpace() == amd::Private))
                    g_windowed.insert(&f);
}
static void resolveGenericWindows(Module &m)
{
    LLVMContext &c = m.getContext();
    Type *i64 = Type::getInt64Ty(c), *i32 = Type::getInt32Ty(c), *i8 = Type::getInt8Ty(c);
    for (Function *f : g_windowed)
    {
        Value *shBase = nullptr, *loBase = nullptr;
        SmallVector<AddrSpaceCastInst *, 8> casts;
        for (Instruction &i : instructions(*f))
            if (auto *asc = dyn_cast<AddrSpaceCastInst>(&i))
                if (asc->getDestAddressSpace() == amd::Flat &&
                    (asc->getSrcAddressSpace() == amd::Shared || asc->getSrcAddressSpace() == amd::Private))
                    casts.push_back(asc);
        for (AddrSpaceCastInst *asc : casts)
        {
            Value *src = asc->getPointerOperand();
            bool sh = asc->getSrcAddressSpace() == amd::Shared;
            Value *obj = getUnderlyingObject(src);
            if (sh && !shBase)
                shBase = obj;
            if (!sh && !loBase)
                loBase = obj;
            IRBuilder<> b(asc);
            Value *off = b.CreateZExt(b.CreatePtrToInt(src, i32), i64);
            Value *g = b.CreateIntToPtr(b.CreateAdd(off, ConstantInt::get(i64, sh ? GW_SHARED : GW_LOCAL)), asc->getType());
            asc->replaceAllUsesWith(g);
            asc->eraseFromParent();
        }
        // the generic accesses: the window of the address at run time
        SmallVector<Instruction *, 16> acc;
        for (Instruction &i : instructions(*f))
        {
            Value *p = nullptr;
            if (auto *ld = dyn_cast<LoadInst>(&i))
                p = ld->getPointerOperand();
            else if (auto *st = dyn_cast<StoreInst>(&i))
                p = st->getPointerOperand();
            else if (auto *rmw = dyn_cast<AtomicRMWInst>(&i))
                p = rmw->getPointerOperand();
            else if (auto *cx = dyn_cast<AtomicCmpXchgInst>(&i))
                p = cx->getPointerOperand();
            if (p && p->getType()->getPointerAddressSpace() == amd::Flat)
                acc.push_back(&i);
        }
        for (Instruction *i : acc)
        {
            unsigned opIdx = isa<StoreInst>(i) ? 1 : 0;
            Value *p = i->getOperand(opIdx);
            BasicBlock *bb = i->getParent();
            BasicBlock *join = bb->splitBasicBlock(i, "gw.join");
            bb->getTerminator()->eraseFromParent();
            IRBuilder<> b(bb);
            Value *x = b.CreatePtrToInt(p, i64);
            // the global access stays the original instruction (moved into its own block)
            BasicBlock *glBB = BasicBlock::Create(c, "gw.global", f, join);
            i->removeFromParent();
            i->insertInto(glBB, glBB->end());
            BranchInst::Create(join, glBB);
            SmallVector<std::pair<Value *, BasicBlock *>, 3> results;
            if (!i->getType()->isVoidTy())
                results.push_back({i, glBB});
            BasicBlock *next = glBB;
            for (int w = 0; w < 2; w++)
            {
                Value *base = w == 0 ? loBase : shBase;
                if (!base)
                    continue;
                uint64_t wb = w == 0 ? GW_LOCAL : GW_SHARED;
                unsigned as = w == 0 ? amd::Private : amd::Shared;
                BasicBlock *test = BasicBlock::Create(c, "gw.test", f, next);
                BasicBlock *hit = BasicBlock::Create(c, w == 0 ? "gw.local" : "gw.shared", f, next);
                IRBuilder<> tb(test);
                Value *rel = tb.CreateSub(x, ConstantInt::get(i64, wb));
                tb.CreateCondBr(tb.CreateICmpULT(rel, ConstantInt::get(i64, GW_SIZE)), hit, next);
                IRBuilder<> hb(hit);
                Value *boff = hb.CreateSub(hb.CreateTrunc(rel, i32), hb.CreatePtrToInt(base, i32));
                Value *q = hb.CreateGEP(i8, base, hb.CreateSExt(boff, i64));
                if (q->getType()->getPointerAddressSpace() != as)
                    q = hb.CreateAddrSpaceCast(q, PointerType::get(c, as));
                Instruction *cl = i->clone();
                cl->setOperand(opIdx, q);
                hb.Insert(cl);
                hb.CreateBr(join);
                if (!i->getType()->isVoidTy())
                    results.push_back({cl, hit});
                next = test;
            }
            BranchInst::Create(next, bb);
            if (!i->getType()->isVoidTy())
            {
                IRBuilder<> jb(&*join->getFirstInsertionPt());
                PHINode *phi = jb.CreatePHI(i->getType(), results.size());
                SmallVector<Use *, 8> uses;
                for (Use &u : i->uses())
                    uses.push_back(&u);
                for (auto &[v, from] : results)
                    phi->addIncoming(v, from);
                for (Use *u : uses)
                    if (u->getUser() != phi)
                        u->set(phi);
            }
        }
    }
}

// isspacep (ptx/lib/metal_impl/memory.inc): `__zluda_metal_isspacep(generic pointer, kind)` is decided from the
// pointer's provenance after PointerRecovery (AGX has no generic addressing: every generic access must resolve
// statically anyway). kind: 0 global, 1 shared::cta, 2 local, 3 const, 4 param, 5 shared::cluster. A pointer
// without a known object (an integer from memory or a parameter) is a global address.
static void resolveIsSpacep(Module &m)
{
    Function *f = m.getFunction("__zluda_metal_isspacep");
    if (!f)
        return;
    std::vector<CallInst *> calls;
    for (User *u : f->users())
        if (auto *ci = dyn_cast<CallInst>(u))
            calls.push_back(ci);
    for (CallInst *ci : calls)
    {
        Value *p = ci->getArgOperand(0);
        for (int guard = 0; guard < 64; guard++)
        {
            p = p->stripPointerCasts();
            if (auto *g = dyn_cast<GEPOperator>(p))
                p = g->getPointerOperand();
            else
                break;
        }
        unsigned as = p->getType()->getPointerAddressSpace();
        int space = 0;
        if (as == amd::Shared)
            space = 1;
        else if (as == amd::Private)
            space = 2;
        else if (as == amd::Constant || as == amd::Constant32)
            space = isa<Argument>(p) ? 4 : 3;
        uint64_t kind = cast<ConstantInt>(ci->getArgOperand(1))->getZExtValue();
        bool r = kind == 5 ? space == 1 : (uint64_t)space == kind;
        Value *arg = ci->getArgOperand(0);
        if (space == 0 && !isa<GlobalValue>(p) && !isa<Argument>(p) && g_windowed.count(ci->getFunction()) && kind <= 2)
        {
            // an address of unknown origin in a function with generic windows: test the windows
            IRBuilder<> b(ci);
            Value *x = b.CreatePtrToInt(arg, b.getInt64Ty());
            auto in = [&](uint64_t base) {
                return b.CreateICmpULT(b.CreateSub(x, b.getInt64(base)), b.getInt64(GW_SIZE));
            };
            Value *v = kind == 1 ? in(GW_SHARED) : kind == 2 ? in(GW_LOCAL) : b.CreateNot(b.CreateOr(in(GW_SHARED), in(GW_LOCAL)));
            ci->replaceAllUsesWith(b.CreateZExt(v, ci->getType()));
            ci->eraseFromParent();
            continue;
        }
        GlobalVariable *block = m.getGlobalVariable("__zluda_cta_shared", true);
        MDNode *md = block ? block->getMetadata("zluda.cluster") : nullptr;
        Value *repl = ConstantInt::get(ci->getType(), r);
        if (md && space == 1 && (kind == 1 || kind == 5))
        {
            // with clusters, the CTA of a shared address is decided at run time: rebuild the address as a
            // threadgroup pointer (the GEPs above the object) and compare its offset in the cluster's copies
            uint64_t stride = mdconst::extract<ConstantInt>(md->getOperand(0))->getZExtValue();
            uint64_t n = mdconst::extract<ConstantInt>(md->getOperand(1))->getZExtValue();
            SmallVector<GEPOperator *, 4> geps;
            Value *q = arg;
            for (int guard = 0; guard < 64; guard++)
            {
                if (auto *o = dyn_cast<Operator>(q); o && (o->getOpcode() == Instruction::AddrSpaceCast || o->getOpcode() == Instruction::BitCast))
                    q = o->getOperand(0);
                else if (auto *g = dyn_cast<GEPOperator>(q))
                {
                    geps.push_back(g);
                    q = g->getPointerOperand();
                }
                else
                    break;
            }
            IRBuilder<> b(ci);
            Value *sp = q;
            for (auto it = geps.rbegin(); it != geps.rend(); ++it)
            {
                SmallVector<Value *, 4> idx((*it)->idx_begin(), (*it)->idx_end());
                sp = b.CreateGEP((*it)->getSourceElementType(), sp, idx);
            }
            Value *off = b.CreateSub(b.CreatePtrToInt(sp, b.getInt32Ty()), b.CreatePtrToInt(block, b.getInt32Ty()));
            Value *in = b.CreateICmpULT(off, b.getInt32(stride * n));
            if (kind == 1)
            {
                Type *i32 = b.getInt32Ty();
                FunctionCallee tpos = m.getOrInsertFunction("__zluda_metal_thread_position_in_threadgroup", i32, i32);
                FunctionCallee tptg = m.getOrInsertFunction("__zluda_metal_threads_per_threadgroup", i32, i32);
                Value *self = b.CreateUDiv(b.CreateCall(tpos, {b.getInt32(2)}),
                                           b.CreateUDiv(b.CreateCall(tptg, {b.getInt32(2)}), b.getInt32(n)));
                in = b.CreateAnd(in, b.CreateICmpEQ(b.CreateUDiv(off, b.getInt32(stride)), self));
            }
            repl = b.CreateZExt(in, ci->getType());
        }
        ci->replaceAllUsesWith(repl);
        ci->eraseFromParent();
        if (auto *i = dyn_cast<Instruction>(arg))
            RecursivelyDeleteTriviallyDeadInstructions(i);
    }
}

class PointerRecovery
{
public:
    explicit PointerRecovery(Function &f) : f(f), dl(f.getParent()->getDataLayout()) {}

    // integer pointer arithmetic that folded into constant expressions (on module-scope variables) is expanded
    // into instructions, which the recovery below handles
    static bool hasIntToPtr(Constant *c)
    {
        auto *ce = dyn_cast<ConstantExpr>(c);
        if (!ce)
            return false;
        if (ce->getOpcode() == Instruction::IntToPtr)
            return true;
        for (Value *op : ce->operands())
            if (auto *oc = dyn_cast<Constant>(op))
                if (hasIntToPtr(oc))
                    return true;
        return false;
    }
    static Instruction *expandConstant(ConstantExpr *ce, Instruction *before)
    {
        Instruction *ni = ce->getAsInstruction();
        ni->insertBefore(before);
        for (unsigned i = 0; i < ni->getNumOperands(); i++)
            if (auto *oc = dyn_cast<ConstantExpr>(ni->getOperand(i)))
                if (hasIntToPtr(oc))
                    ni->setOperand(i, expandConstant(oc, ni));
        return ni;
    }
    bool expandConstants()
    {
        bool changed = false;
        SmallVector<Instruction *, 32> users;
        for (Instruction &i : instructions(f))
            users.push_back(&i);
        for (Instruction *i : users)
            for (unsigned k = 0; k < i->getNumOperands(); k++)
                if (auto *ce = dyn_cast<ConstantExpr>(i->getOperand(k)))
                    if (hasIntToPtr(ce))
                    {
                        Instruction *at = i;
                        if (auto *phi = dyn_cast<PHINode>(i))
                            at = phi->getIncomingBlock(k)->getTerminator();
                        i->setOperand(k, expandConstant(ce, at));
                        changed = true;
                    }
        return changed;
    }

    bool run()
    {
        bool changed = expandConstants();
        SmallVector<Instruction *, 32> worklist;
        for (Instruction &i : instructions(f))
            if (isa<IntToPtrInst>(i) || isa<AddrSpaceCastInst>(i) || isa<ICmpInst>(i))
                worklist.push_back(&i);
        for (Instruction *i : worklist)
        {
            if (auto *cmp = dyn_cast<ICmpInst>(i))
            {
                // compare pointers instead of their integer values
                Value *a = cmp->getOperand(0), *c = cmp->getOperand(1);
                if (!recoverable(a) || !recoverable(c))
                    continue;
                Value *pa = recover(a), *pc = recover(c);
                if (!pa || !pc || pa->getType() != pc->getType())
                    continue;
                IRBuilder<> b(cmp);
                Value *repl = b.CreateICmp(cmp->getPredicate(), pa, pc);
                cmp->replaceAllUsesWith(repl);
                cmp->eraseFromParent();
                changed = true;
            }
            else if (auto *itp = dyn_cast<IntToPtrInst>(i))
            {
                if (!recoverable(itp->getOperand(0)))
                    continue;
                Value *p = recover(itp->getOperand(0));
                if (!p)
                    continue;
                IRBuilder<> b(itp);
                Value *repl = castTo(b, p, cast<PointerType>(itp->getType()));
                itp->replaceAllUsesWith(repl);
                itp->eraseFromParent();
                changed = true;
            }
            else if (auto *asc = dyn_cast<AddrSpaceCastInst>(i))
            {
                // addrspacecast (inttoptr x to ptr) to ptr addrspace(N) -> inttoptr x to ptr addrspace(N)
                if (auto *itp = dyn_cast<IntToPtrInst>(asc->getOperand(0)))
                {
                    IRBuilder<> b(asc);
                    Value *repl = b.CreateIntToPtr(itp->getOperand(0), asc->getType());
                    asc->replaceAllUsesWith(repl);
                    asc->eraseFromParent();
                    changed = true;
                }
            }
        }
        return changed;
    }

private:
    Function &f;
    const DataLayout &dl;
    std::map<Value *, Value *> memo;

    // the pointer under addrspacecasts (cvta: the generic address of a shared or local object keeps the
    // provenance of that object)
    static Value *stripCasts(Value *p)
    {
        while (auto *o = dyn_cast<AddrSpaceCastOperator>(p))
            p = o->getPointerOperand();
        return p;
    }

    Value *castTo(IRBuilder<> &b, Value *p, PointerType *t)
    {
        if (p->getType() == t)
            return p;
        return b.CreateAddrSpaceCast(p, t);
    }

    static Instruction *insertionPointAfter(Value *v, Function &f)
    {
        if (auto *i = dyn_cast<Instruction>(v))
        {
            if (auto *phi = dyn_cast<PHINode>(i))
                return &*phi->getParent()->getFirstInsertionPt();
            return i->getNextNode();
        }
        return &*f.getEntryBlock().getFirstInsertionPt();
    }

    // Address space of the pointer an integer value derives from (without
    // creating any IR); ~0u if it does not derive from a single pointer
    unsigned baseAddressSpace(Value *v, SmallPtrSetImpl<Value *> &visiting)
    {
        if (auto *pti = dyn_cast<PtrToIntOperator>(v))
            return stripCasts(pti->getPointerOperand())->getType()->getPointerAddressSpace();
        if (!visiting.insert(v).second)
            return ~1u; // cycle: neutral element
        if (Value *x = identityAnd(v))
            return baseAddressSpace(x, visiting);
        if (Value *x = addressZExt(v))
        {
            unsigned a = baseAddressSpace(x, visiting);
            return a == amd::Shared || a == amd::Private || a == ~1u ? a : ~0u;
        }
        if (auto *bo = dyn_cast<BinaryOperator>(v))
        {
            if (bo->getOpcode() == Instruction::Add || bo->getOpcode() == Instruction::Sub ||
                (bo->getOpcode() == Instruction::Or && cast<PossiblyDisjointInst>(bo)->isDisjoint()))
            {
                // pointer + offset: one side carries the provenance (a cycle
                // through a phi counts as the pointer side)
                unsigned a = baseAddressSpace(bo->getOperand(0), visiting);
                if (a < ~1u)
                    return a;
                unsigned b = bo->getOpcode() == Instruction::Sub ? ~0u : baseAddressSpace(bo->getOperand(1), visiting);
                if (b < ~1u)
                    return b;
                if (a == ~1u || b == ~1u)
                    return ~1u;
            }
            return ~0u;
        }
        if (auto *ce = dyn_cast<ConstantExpr>(v))
        {
            if (ce->getOpcode() == Instruction::Add)
            {
                unsigned a = baseAddressSpace(ce->getOperand(0), visiting);
                return a < ~1u ? a : baseAddressSpace(ce->getOperand(1), visiting);
            }
            return ~0u;
        }
        if (auto *phi = dyn_cast<PHINode>(v))
        {
            unsigned as = ~1u;
            for (Value *in : phi->incoming_values())
            {
                unsigned a = baseAddressSpace(in, visiting);
                if (a == ~0u)
                    return ~0u;
                if (a == ~1u)
                    continue;
                if (as != ~1u && as != a)
                    return ~0u;
                as = a;
            }
            return as;
        }
        if (auto *sel = dyn_cast<SelectInst>(v))
        {
            unsigned a = baseAddressSpace(sel->getTrueValue(), visiting);
            unsigned b = baseAddressSpace(sel->getFalseValue(), visiting);
            return a == b ? a : ~0u;
        }
        return ~0u;
    }

    // `zext i32 x`: threadgroup and thread addresses are 32-bit (PTX's .shared / .local addresses, `mov.u32 %r,
    // sdata`), widened for a 64-bit inttoptr; on every address in the window the widening is the identity (the
    // checks of the caller keep it to those address spaces)
    static Value *addressZExt(Value *v)
    {
        auto *z = dyn_cast<ZExtInst>(v);
        if (!z || z->getSrcTy()->getIntegerBitWidth() != 32)
            return nullptr;
        return z->getOperand(0);
    }

    // `x & C` that keeps every bit x can have (a 32-bit pointer zero-extended, known alignment): x
    Value *identityAnd(Value *v)
    {
        auto *bo = dyn_cast<BinaryOperator>(v);
        if (!bo || bo->getOpcode() != Instruction::And)
            return nullptr;
        auto *c = dyn_cast<ConstantInt>(bo->getOperand(1));
        if (!c)
            return nullptr;
        KnownBits kb = computeKnownBits(bo->getOperand(0), dl);
        // threadgroup and thread addresses are 32-bit
        SmallPtrSet<Value *, 16> visiting;
        unsigned as = baseAddressSpace(bo->getOperand(0), visiting);
        if ((as == amd::Shared || as == amd::Private) && kb.getBitWidth() > 32)
            kb.Zero.setBitsFrom(32);
        if ((~c->getValue() & ~kb.Zero).isZero())
            return bo->getOperand(0);
        return nullptr;
    }

    bool recoverable(Value *v)
    {
        SmallPtrSet<Value *, 16> visiting;
        return baseAddressSpace(v, visiting) < ~1u;
    }

    // Returns a pointer equivalent of integer `v`, or nullptr
    Value *recover(Value *v)
    {
        auto it = memo.find(v);
        if (it != memo.end())
            return it->second;
        memo[v] = nullptr; // cycle guard for non-phi values
        Value *result = nullptr;
        if (auto *pti = dyn_cast<PtrToIntInst>(v))
        {
            result = stripCasts(pti->getOperand(0));
        }
        else if (auto *ce = dyn_cast<ConstantExpr>(v))
        {
            if (ce->getOpcode() == Instruction::PtrToInt)
                result = stripCasts(ce->getOperand(0));
            else if (ce->getOpcode() == Instruction::Add)
            {
                for (unsigned side = 0; side < 2 && !result; side++)
                    if (auto *base = dyn_cast_or_null<Constant>(recover(ce->getOperand(side))))
                        result = ConstantExpr::getGetElementPtr(Type::getInt8Ty(f.getContext()), base,
                                                                ce->getOperand(1 - side));
            }
        }
        else if (Value *x = identityAnd(v))
        {
            result = recover(x);
        }
        else if (Value *x = addressZExt(v))
        {
            SmallPtrSet<Value *, 16> visiting;
            unsigned as = baseAddressSpace(v, visiting);
            if (as == amd::Shared || as == amd::Private)
                result = recover(x);
        }
        else if (auto *bo = dyn_cast<BinaryOperator>(v))
        {
            if (bo->getOpcode() == Instruction::Add || bo->getOpcode() == Instruction::Sub ||
                bo->getOpcode() == Instruction::Or)
            {
                for (unsigned side = 0; side < 2 && !result; side++)
                {
                    if (side == 1 && bo->getOpcode() == Instruction::Sub)
                        break;
                    // `or` is only an add if the operands have no common bits; we
                    // accept it only with the `disjoint` flag
                    if (bo->getOpcode() == Instruction::Or && !cast<PossiblyDisjointInst>(bo)->isDisjoint())
                        break;
                    Value *base = recover(bo->getOperand(side));
                    if (!base)
                        continue;
                    Value *offset = bo->getOperand(1 - side);
                    IRBuilder<> b(bo);
                    if (bo->getOpcode() == Instruction::Sub)
                        offset = b.CreateNeg(offset);
                    result = b.CreateGEP(b.getInt8Ty(), base, offset);
                }
            }
        }
        else if (auto *phi = dyn_cast<PHINode>(v))
        {
            SmallPtrSet<Value *, 16> visiting;
            unsigned as = baseAddressSpace(phi, visiting);
            if (as >= ~1u)
                return nullptr;
            PHINode *np = PHINode::Create(PointerType::get(f.getContext(), as), phi->getNumIncomingValues(),
                                          phi->getName() + ".ptr", phi);
            memo[v] = np;
            for (unsigned i = 0; i < phi->getNumIncomingValues(); i++)
            {
                Value *in = recover(phi->getIncomingValue(i));
                if (!in || in->getType() != np->getType())
                    fail("inconsistent pointer provenance through phi: " + toString(phi));
                np->addIncoming(in, phi->getIncomingBlock(i));
            }
            result = np;
        }
        else if (auto *sel = dyn_cast<SelectInst>(v))
        {
            Value *t = recover(sel->getTrueValue());
            Value *fv = recover(sel->getFalseValue());
            if (t && fv && t->getType() == fv->getType())
            {
                IRBuilder<> b(sel);
                result = b.CreateSelect(sel->getCondition(), t, fv);
            }
        }
        memo[v] = result;
        return result;
    }
};

// ---------------------------------------------------------------------------
// Address space remapping (AMDGPU numbering -> AIR numbering), done in place
// ---------------------------------------------------------------------------
Constant *remapConstant(Constant *c)
{
    Type *nt = mapType(c->getType());
    if (nt == c->getType())
        return c;
    if (isa<GlobalValue>(c))
        return c; // globals are mutated in place
    if (isa<PoisonValue>(c))
        return PoisonValue::get(nt);
    if (isa<UndefValue>(c))
        return UndefValue::get(nt);
    if (c->isNullValue())
        return Constant::getNullValue(nt);
    fail("cannot remap address space of constant: " + toString(c));
}

void remapAddressSpaces(Module &m)
{
    LLVMContext &ctx = m.getContext();
    // 1. No pointer-typed constant expressions may survive: turn them into instructions
    {
        SmallVector<Constant *, 16> consts;
        for (GlobalVariable &gv : m.globals())
            consts.push_back(&gv);
        for (Function &f : m)
            consts.push_back(&f);
        convertUsersOfConstantsToInstructions(consts);
    }
    for (GlobalVariable &gv : m.globals())
    {
        if (gv.hasInitializer() && mapType(gv.getInitializer()->getType()) != gv.getInitializer()->getType())
            fail("global initializers holding pointers are not supported: " + gv.getName());
        if (mapType(gv.getValueType()) != gv.getValueType())
            fail("globals holding pointers are not supported: " + gv.getName());
    }
    // 2. Functions: new signatures
    std::vector<Function *> functions;
    for (Function &f : m)
        functions.push_back(&f);
    std::map<Function *, Function *> replaced;
    for (Function *f : functions)
    {
        FunctionType *ft = f->getFunctionType();
        FunctionType *nft = cast<FunctionType>(mapType(ft));
        if (nft == ft)
            continue;
        Function *nf = Function::Create(nft, f->getLinkage(), f->getAddressSpace(), "", &m);
        nf->copyAttributesFrom(f);
        nf->setCallingConv(f->getCallingConv());
        nf->takeName(f);
        // byref/byval attribute types are value types, they don't contain pointers here
        nf->splice(nf->end(), f);
        for (unsigned i = 0; i < f->arg_size(); i++)
        {
            Argument *oa = f->getArg(i);
            Argument *na = nf->getArg(i);
            na->takeName(oa);
            oa->mutateType(na->getType());
            oa->replaceAllUsesWith(na);
        }
        replaced[f] = nf;
    }
    // 3. Globals: mutate the pointer type in place (globals are not uniqued)
    for (GlobalVariable &gv : m.globals())
        gv.mutateType(PointerType::get(ctx, mapAddressSpace(gv.getAddressSpace())));
    // 4. Instructions
    for (Function &f : m)
    {
        for (Instruction &i : instructions(f))
        {
            for (Use &u : i.operands())
            {
                if (auto *c = dyn_cast<Constant>(u.get()))
                {
                    if (auto *oldF = dyn_cast<Function>(c))
                    {
                        auto it = replaced.find(oldF);
                        if (it != replaced.end())
                            u.set(it->second);
                        continue;
                    }
                    Constant *nc = remapConstant(c);
                    if (nc != c)
                        u.set(nc);
                }
            }
            i.mutateType(mapType(i.getType()));
            if (auto *ci = dyn_cast<CallBase>(&i))
                ci->mutateFunctionType(cast<FunctionType>(mapType(ci->getFunctionType())));
            if (auto *ai = dyn_cast<AllocaInst>(&i))
                ai->setAllocatedType(mapType(ai->getAllocatedType()));
            if (auto *gep = dyn_cast<GetElementPtrInst>(&i))
            {
                gep->setSourceElementType(mapType(gep->getSourceElementType()));
                gep->setResultElementType(mapType(gep->getResultElementType()));
            }
        }
    }
    for (auto &[of, nf] : replaced)
    {
        if (!of->use_empty())
        {
            of->mutateType(nf->getType());
            of->replaceAllUsesWith(nf);
        }
        of->eraseFromParent();
    }
    // 5. Address space casts that became no-ops
    for (Function &f : m)
    {
        SmallVector<AddrSpaceCastInst *, 8> dead;
        for (Instruction &i : instructions(f))
            if (auto *asc = dyn_cast<AddrSpaceCastInst>(&i))
                if (asc->getSrcAddressSpace() == asc->getDestAddressSpace())
                    dead.push_back(asc);
        for (auto *asc : dead)
        {
            asc->replaceAllUsesWith(asc->getOperand(0));
            asc->eraseFromParent();
        }
    }
    // 6. Re-mangle overloaded intrinsics (llvm.memcpy.p0.p5.i64 -> llvm.memcpy.p0.p0.i64 ...)
    std::vector<Function *> intrinsics;
    for (Function &f : m)
        if (f.isIntrinsic())
            intrinsics.push_back(&f);
    for (Function *f : intrinsics)
    {
        if (auto nf = Intrinsic::remangleIntrinsicFunction(f))
        {
            f->replaceAllUsesWith(*nf);
            f->eraseFromParent();
        }
    }
    m.setDataLayout(AIR_DATALAYOUT);
    m.setTargetTriple(AIR_TRIPLE);
}

// ---------------------------------------------------------------------------
// Instruction / intrinsic lowering to AIR
// ---------------------------------------------------------------------------
std::string airTypeSuffix(Type *t)
{
    if (auto *vt = dyn_cast<FixedVectorType>(t))
        return "v" + std::to_string(vt->getNumElements()) + airTypeSuffix(vt->getElementType());
    if (t->isHalfTy())
        return "f16";
    if (t->isFloatTy())
        return "f32";
    if (t->isDoubleTy())
        return "f64";
    if (t->isBFloatTy())
        return "bf16";
    if (t->isIntegerTy())
        return "i" + std::to_string(t->getIntegerBitWidth());
    fail("no AIR type suffix for type");
}

FunctionCallee airFunction(Module &m, StringRef name, Type *ret, ArrayRef<Type *> params)
{
    FunctionCallee callee = m.getOrInsertFunction(name, FunctionType::get(ret, params, false));
    if (auto *f = dyn_cast<Function>(callee.getCallee()))
    {
        f->addFnAttr(Attribute::NoUnwind);
        f->addFnAttr(Attribute::WillReturn);
    }
    return callee;
}

Value *callAir(IRBuilder<> &b, StringRef name, Type *ret, ArrayRef<Value *> args)
{
    SmallVector<Type *, 4> types;
    for (Value *a : args)
        types.push_back(a->getType());
    Module &m = *b.GetInsertBlock()->getModule();
    return b.CreateCall(airFunction(m, name, ret, types), args);
}

// MSL memory_order / thread_scope encodings used by the air.atomic.* intrinsics
constexpr unsigned MEMORY_ORDER_RELAXED = 0;
constexpr unsigned MEMORY_ORDER_SEQ_CST = 5;
constexpr unsigned SCOPE_THREADGROUP = 1;
constexpr unsigned SCOPE_DEVICE = 2;

const char *atomicSpace(unsigned as)
{
    switch (as)
    {
    case air::Device:
        return "global";
    case air::Threadgroup:
        return "local";
    default:
        fail("atomic operation on unsupported address space " + Twine(as));
    }
}

unsigned atomicScope(unsigned as) { return as == air::Threadgroup ? SCOPE_THREADGROUP : SCOPE_DEVICE; }

// CAS-based emulation of an atomic read-modify-write that AIR has no intrinsic for
Value *emitCasLoop(IRBuilder<> &b, Value *ptr, Type *valTy, function_ref<Value *(IRBuilder<> &, Value *)> op);

Value *emitCasLoop(IRBuilder<> &b, Value *ptr, Type *valTy, function_ref<Value *(IRBuilder<> &, Value *)> op);

Value *emitCmpXchgStrong(IRBuilder<> &b, Value *ptr, Value *cmp, Value *desired, Value **successOut)
{
    if (cmp->getType()->isIntegerTy(8) || cmp->getType()->isIntegerTy(16))
    {
        // no 8 / 16-bit compare-exchange: a 32-bit one on the containing word
        Type *t = cmp->getType();
        Value *off = b.CreateAnd(b.CreatePtrToInt(ptr, b.getInt64Ty()), 3);
        Value *base = b.CreateGEP(b.getInt8Ty(), ptr, b.CreateNeg(off));
        Value *shift = b.CreateTrunc(b.CreateShl(off, 3), b.getInt32Ty());
        Value *mask = b.CreateShl(b.getInt32((1u << t->getIntegerBitWidth()) - 1), shift);
        Value *old32 = emitCasLoop(b, base, b.getInt32Ty(), [&](IRBuilder<> &lb, Value *old) -> Value * {
            Value *cur = lb.CreateTrunc(lb.CreateLShr(old, shift), t);
            Value *upd = lb.CreateOr(lb.CreateAnd(old, lb.CreateNot(mask)),
                                     lb.CreateShl(lb.CreateZExt(desired, lb.getInt32Ty()), shift));
            return lb.CreateSelect(lb.CreateICmpEQ(cur, cmp), upd, old);
        });
        Value *observed = b.CreateTrunc(b.CreateLShr(old32, shift), t);
        if (successOut)
            *successOut = b.CreateICmpEQ(observed, cmp);
        return observed;
    }
    // air.atomic.*.cmpxchg.weak.iN(ptr object, ptr(thread) expected, iN desired,
    //                               i32 order_success, i32 order_failure, i32 scope, i1 volatile)
    // returns i1 and writes the observed value to *expected. Wrap it into a
    // strong compare-exchange loop.
    Function *f = b.GetInsertBlock()->getParent();
    LLVMContext &ctx = f->getContext();
    Type *t = cmp->getType();
    unsigned as = ptr->getType()->getPointerAddressSpace();
    IRBuilder<> entryB(&*f->getEntryBlock().getFirstInsertionPt());
    AllocaInst *expected = entryB.CreateAlloca(t, nullptr, "cmpxchg.expected");
    BasicBlock *pre = b.GetInsertBlock();
    BasicBlock *rest;
    if (b.GetInsertPoint() == pre->end())
    {
        // a block under construction (emitCasLoop's loop): nothing to split off
        rest = BasicBlock::Create(ctx, "cmpxchg.end", f, pre->getNextNode());
    }
    else
    {
        rest = pre->splitBasicBlock(b.GetInsertPoint(), "cmpxchg.end");
        pre->getTerminator()->eraseFromParent();
    }
    BasicBlock *loop = BasicBlock::Create(ctx, "cmpxchg.loop", f, rest);
    b.SetInsertPoint(pre);
    b.CreateBr(loop);
    b.SetInsertPoint(loop);
    b.CreateStore(cmp, expected);
    std::string name = std::string("air.atomic.") + atomicSpace(as) + ".cmpxchg.weak." + airTypeSuffix(t);
    Value *ok = callAir(b, name, b.getInt1Ty(),
                        {ptr, expected, desired, b.getInt32(MEMORY_ORDER_RELAXED), b.getInt32(MEMORY_ORDER_RELAXED),
                         b.getInt32(atomicScope(as)), b.getTrue()});
    Value *observed = b.CreateLoad(t, expected);
    // retry only on a spurious failure (observed value equals the expected one)
    Value *spurious = b.CreateAnd(b.CreateNot(ok), b.CreateICmpEQ(observed, cmp));
    b.CreateCondBr(spurious, loop, rest);
    b.SetInsertPoint(rest, rest->begin());
    if (successOut)
        *successOut = ok;
    return observed;
}

Value *emitCasLoop(IRBuilder<> &b, Value *ptr, Type *valTy, function_ref<Value *(IRBuilder<> &, Value *)> op)
{
    Function *f = b.GetInsertBlock()->getParent();
    LLVMContext &ctx = f->getContext();
    Type *intTy = IntegerType::get(ctx, valTy->getPrimitiveSizeInBits());
    if (intTy->getIntegerBitWidth() < 32)
    {
        // 8 / 16-bit read-modify-write: on the containing 32-bit word
        Value *off = b.CreateAnd(b.CreatePtrToInt(ptr, b.getInt64Ty()), 3);
        Value *base = b.CreateGEP(b.getInt8Ty(), ptr, b.CreateNeg(off));
        Value *shift = b.CreateTrunc(b.CreateShl(off, 3), b.getInt32Ty());
        Value *mask = b.CreateShl(b.getInt32((1u << intTy->getIntegerBitWidth()) - 1), shift);
        Value *old32 = emitCasLoop(b, base, b.getInt32Ty(), [&](IRBuilder<> &lb, Value *old) -> Value * {
            Value *cur = lb.CreateBitCast(lb.CreateTrunc(lb.CreateLShr(old, shift), intTy), valTy);
            Value *nv = lb.CreateBitCast(op(lb, cur), intTy);
            return lb.CreateOr(lb.CreateAnd(old, lb.CreateNot(mask)), lb.CreateShl(lb.CreateZExt(nv, lb.getInt32Ty()), shift));
        });
        return b.CreateBitCast(b.CreateTrunc(b.CreateLShr(old32, shift), intTy), valTy);
    }
    unsigned as = ptr->getType()->getPointerAddressSpace();
    Value *initial = callAir(b, std::string("air.atomic.") + atomicSpace(as) + ".load." + airTypeSuffix(intTy), intTy,
                             {ptr, b.getInt32(MEMORY_ORDER_RELAXED), b.getInt32(atomicScope(as)), b.getTrue()});
    BasicBlock *pre = b.GetInsertBlock();
    BasicBlock *rest = pre->splitBasicBlock(b.GetInsertPoint(), "atomic.end");
    pre->getTerminator()->eraseFromParent();
    BasicBlock *loop = BasicBlock::Create(ctx, "atomic.loop", f, rest);
    b.SetInsertPoint(pre);
    b.CreateBr(loop);
    b.SetInsertPoint(loop);
    PHINode *old = b.CreatePHI(intTy, 2);
    old->addIncoming(initial, pre);
    Value *oldVal = b.CreateBitCast(old, valTy);
    Value *newVal = b.CreateBitCast(op(b, oldVal), intTy);
    Value *success = nullptr;
    // emitCmpXchgStrong splits the current block, which needs a terminator: a branch to the rest, which ends
    // in the "done" block below
    b.SetInsertPoint(b.CreateBr(rest));
    Value *observed = emitCmpXchgStrong(b, ptr, old, newVal, &success);
    BasicBlock *loopEnd = b.GetInsertBlock();
    BasicBlock *done;
    if (b.GetInsertPoint() == loopEnd->end())
        done = rest;
    else
    {
        done = loopEnd->splitBasicBlock(b.GetInsertPoint(), "atomic.done");
        loopEnd->getTerminator()->eraseFromParent();
    }
    b.SetInsertPoint(loopEnd);
    old->addIncoming(observed, loopEnd);
    b.CreateCondBr(success, done, loop);
    b.SetInsertPoint(done, done->begin());
    (void)rest;
    return b.CreateBitCast(old, valTy);
}

void lowerAtomicRMW(AtomicRMWInst *rmw)
{
    IRBuilder<> b(rmw);
    Value *ptr = rmw->getPointerOperand();
    Value *val = rmw->getValOperand();
    Type *t = val->getType();
    unsigned as = rmw->getPointerAddressSpace();
    std::string prefix = std::string("air.atomic.") + atomicSpace(as) + ".";
    std::string op;
    std::string sign;
    switch (rmw->getOperation())
    {
    case AtomicRMWInst::Xchg:
        op = "xchg";
        break;
    case AtomicRMWInst::Add:
        op = "add";
        sign = "u.";
        break;
    case AtomicRMWInst::Sub:
        op = "sub";
        sign = "u.";
        break;
    case AtomicRMWInst::And:
        op = "and";
        sign = "u.";
        break;
    case AtomicRMWInst::Or:
        op = "or";
        sign = "u.";
        break;
    case AtomicRMWInst::Xor:
        op = "xor";
        sign = "u.";
        break;
    case AtomicRMWInst::Max:
        op = "max";
        sign = "s.";
        break;
    case AtomicRMWInst::Min:
        op = "min";
        sign = "s.";
        break;
    case AtomicRMWInst::UMax:
        op = "max";
        sign = "u.";
        break;
    case AtomicRMWInst::UMin:
        op = "min";
        sign = "u.";
        break;
    case AtomicRMWInst::FAdd:
        op = "add";
        break;
    case AtomicRMWInst::FSub:
        op = "sub";
        break;
    default:
        break;
    }
    Value *result = nullptr;
    bool native = !op.empty() && (t->isIntegerTy(32) || t->isIntegerTy(64) || t->isFloatTy());
    // the hardware's float atomic add flushes subnormals; PTX's atom / red .shared.add.f32 keeps them (only the
    // .global form without .noftz flushes): a compare-exchange loop with an IEEE add
    bool keepDenormals = t->isFloatTy() && as == air::Threadgroup &&
                         (rmw->getOperation() == AtomicRMWInst::FAdd || rmw->getOperation() == AtomicRMWInst::FSub);
    if (keepDenormals)
        native = false;
    if (native && t->isFloatTy() && rmw->getOperation() == AtomicRMWInst::Xchg)
        native = true;
    if (native)
    {
        std::string name = prefix + op + "." + sign + airTypeSuffix(t);
        result = callAir(b, name, t, {ptr, val, b.getInt32(MEMORY_ORDER_RELAXED), b.getInt32(atomicScope(as)), b.getTrue()});
    }
    else
    {
        auto opcode = rmw->getOperation();
        result = emitCasLoop(b, ptr, t, [&](IRBuilder<> &lb, Value *old) -> Value * {
            switch (opcode)
            {
            case AtomicRMWInst::Nand:
                return lb.CreateNot(lb.CreateAnd(old, val));
            case AtomicRMWInst::UIncWrap:
                return lb.CreateSelect(lb.CreateICmpUGE(old, val), ConstantInt::get(t, 0),
                                       lb.CreateAdd(old, ConstantInt::get(t, 1)));
            case AtomicRMWInst::UDecWrap:
                return lb.CreateSelect(
                    lb.CreateOr(lb.CreateICmpEQ(old, ConstantInt::get(t, 0)), lb.CreateICmpUGT(old, val)), val,
                    lb.CreateSub(old, ConstantInt::get(t, 1)));
            case AtomicRMWInst::FMax:
                return lb.CreateMaxNum(old, val);
            case AtomicRMWInst::FMin:
                return lb.CreateMinNum(old, val);
            case AtomicRMWInst::FAdd:
            case AtomicRMWInst::FSub:
            {
                Value *r = opcode == AtomicRMWInst::FAdd ? lb.CreateFAdd(old, val) : lb.CreateFSub(old, val);
                if (keepDenormals)
                    if (auto *ri = dyn_cast<Instruction>(r))
                        ri->setMetadata("agx.keep-denormals", MDNode::get(ri->getContext(), {}));
                return r;
            }
            case AtomicRMWInst::Add:
                return lb.CreateAdd(old, val);
            case AtomicRMWInst::Sub:
                return lb.CreateSub(old, val);
            case AtomicRMWInst::And:
                return lb.CreateAnd(old, val);
            case AtomicRMWInst::Or:
                return lb.CreateOr(old, val);
            case AtomicRMWInst::Xor:
                return lb.CreateXor(old, val);
            case AtomicRMWInst::Xchg:
                return val;
            case AtomicRMWInst::Max:
                return lb.CreateSelect(lb.CreateICmpSGT(old, val), old, val);
            case AtomicRMWInst::Min:
                return lb.CreateSelect(lb.CreateICmpSLT(old, val), old, val);
            case AtomicRMWInst::UMax:
                return lb.CreateSelect(lb.CreateICmpUGT(old, val), old, val);
            case AtomicRMWInst::UMin:
                return lb.CreateSelect(lb.CreateICmpULT(old, val), old, val);
            default:
                fail("unsupported atomicrmw operation");
            }
        });
    }
    rmw->replaceAllUsesWith(result);
    rmw->eraseFromParent();
}

void lowerCmpXchg(AtomicCmpXchgInst *cx)
{
    IRBuilder<> b(cx);
    Value *success = nullptr;
    Value *cmp = cx->getCompareOperand();
    Value *nv = cx->getNewValOperand();
    Type *t = cmp->getType();
    Value *ptr = cx->getPointerOperand();
    if (t->isPointerTy())
        fail("cmpxchg on pointers is not supported");
    Value *observed = emitCmpXchgStrong(b, ptr, cmp, nv, &success);
    Value *agg = UndefValue::get(cx->getType());
    agg = b.CreateInsertValue(agg, observed, 0);
    agg = b.CreateInsertValue(agg, success, 1);
    cx->replaceAllUsesWith(agg);
    cx->eraseFromParent();
}

void lowerAtomicLoadStore(Instruction *i)
{
    IRBuilder<> b(i);
    if (auto *ld = dyn_cast<LoadInst>(i))
    {
        Type *t = ld->getType();
        unsigned as = ld->getPointerAddressSpace();
        if (as != air::Device && as != air::Threadgroup)
        {
            ld->setAtomic(AtomicOrdering::NotAtomic);
            return;
        }
        Type *it = t->isFloatTy() ? b.getInt32Ty() : t;
        Value *v = callAir(b, std::string("air.atomic.") + atomicSpace(as) + ".load." + airTypeSuffix(it), it,
                           {ld->getPointerOperand(), b.getInt32(MEMORY_ORDER_RELAXED), b.getInt32(atomicScope(as)),
                            b.getTrue()});
        if (it != t)
            v = b.CreateBitCast(v, t);
        ld->replaceAllUsesWith(v);
        ld->eraseFromParent();
    }
    else if (auto *st = dyn_cast<StoreInst>(i))
    {
        Value *v = st->getValueOperand();
        unsigned as = st->getPointerAddressSpace();
        if (as != air::Device && as != air::Threadgroup)
        {
            st->setAtomic(AtomicOrdering::NotAtomic);
            return;
        }
        if (v->getType()->isFloatTy())
            v = b.CreateBitCast(v, b.getInt32Ty());
        callAir(b, std::string("air.atomic.") + atomicSpace(as) + ".store." + airTypeSuffix(v->getType()),
                b.getVoidTy(),
                {st->getPointerOperand(), v, b.getInt32(MEMORY_ORDER_RELAXED), b.getInt32(atomicScope(as)), b.getTrue()});
        st->eraseFromParent();
    }
}

// Portable AIR for llvm-air2agx: doubles and directed-rounding constrained intrinsics are
// kept (the AIR target's air-emulate-f64 lowers them), instead of rejected / rounded to nearest.
static bool g_portable_air = false;
// floating point mode fast (ZLUDA_FP_MODE, compile.rs FpMode): the hardware's operations, within a few ulp
static bool g_fp_fast = false;
// and binary32 denormals flushed (ZLUDA_FP_MODE=fast-ftz)
static bool g_fp_ftz = false;

// Maps an LLVM / AMDGPU intrinsic call onto AIR. Returns true when handled.
bool lowerIntrinsic(CallInst *ci)
{
    Function *callee = ci->getCalledFunction();
    if (!callee)
        return false;
    StringRef name = callee->getName();
    IRBuilder<> b(ci);
    Type *rt = ci->getType();
    auto arg = [&](unsigned i) { return ci->getArgOperand(i); };
    auto replace = [&](Value *v) {
        if (auto *vi = dyn_cast_or_null<Instruction>(v))
            if (MDNode *md = ci->getMetadata("agx.keep-denormals"))
                vi->setMetadata("agx.keep-denormals", md);
        if (v)
            ci->replaceAllUsesWith(v);
        ci->eraseFromParent();
        return true;
    };
    auto airMath = [&](StringRef base, unsigned nargs) {
        SmallVector<Value *, 3> args;
        for (unsigned i = 0; i < nargs; i++)
            args.push_back(arg(i));
        return replace(callAir(b, ("air." + base + "." + airTypeSuffix(rt)).str(), rt, args));
    };
    auto intSuffix = [&](bool isSigned) { return std::string(isSigned ? "s." : "u.") + airTypeSuffix(rt); };

    Intrinsic::ID id = callee->getIntrinsicID();
    switch (id)
    {
    case Intrinsic::not_intrinsic:
        break;
    // Kept as-is: Metal AIR uses these standard LLVM intrinsics too
    case Intrinsic::memcpy:
    case Intrinsic::memmove:
    case Intrinsic::memset:
        return false;
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::assume:
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_value:
    case Intrinsic::dbg_label:
    case Intrinsic::trap:
    case Intrinsic::debugtrap:
    case Intrinsic::experimental_noalias_scope_decl:
    case Intrinsic::sideeffect:
    case Intrinsic::donothing:
        return replace(nullptr);
    case Intrinsic::expect:
        return replace(arg(0));
    case Intrinsic::fma:
    case Intrinsic::fmuladd:
        return airMath("fma", 3);
    case Intrinsic::fabs:
        return airMath("fabs", 1);
    case Intrinsic::sqrt:
        return airMath("sqrt", 1);
    case Intrinsic::minnum:
        return airMath("fmin", 2);
    case Intrinsic::maxnum:
        return airMath("fmax", 2);
    case Intrinsic::minimum:
    case Intrinsic::maximum:
    {
        // LLVM's minimum / maximum: a NaN operand gives NaN, and -0 < +0. AIR's fmin / fmax are
        // minnum / maxnum, so the NaN and equal-zero cases are selected explicitly.
        bool isMax = id == Intrinsic::maximum;
        Value *a = arg(0), *c = arg(1);
        Value *m = callAir(b, (std::string(isMax ? "air.fmax." : "air.fmin.") + airTypeSuffix(rt)), rt, {a, c});
        Type *it = rt->getWithNewType(IntegerType::get(rt->getContext(), rt->getScalarSizeInBits()));
        Value *ai = b.CreateBitCast(a, it), *ci2 = b.CreateBitCast(c, it);
        Value *bits = isMax ? b.CreateAnd(ai, ci2) : b.CreateOr(ai, ci2);
        Value *eq = b.CreateFCmpOEQ(a, c);
        m = b.CreateSelect(eq, b.CreateBitCast(bits, rt), m);
        Value *nan = b.CreateFCmpUNO(a, c);
        m = b.CreateSelect(nan, b.CreateFAdd(a, c), m);
        return replace(m);
    }
    case Intrinsic::copysign:
        // Metal accepts llvm.copysign as it is (there is no air.copysign)
        return false;
    case Intrinsic::floor:
        return airMath("floor", 1);
    case Intrinsic::ceil:
        return airMath("ceil", 1);
    case Intrinsic::trunc:
        return airMath("trunc", 1);
    case Intrinsic::rint:
    case Intrinsic::nearbyint:
    case Intrinsic::roundeven:
        return airMath("rint", 1);
    case Intrinsic::round:
        return airMath("round", 1);
    case Intrinsic::exp2:
        return airMath("exp2", 1);
    case Intrinsic::exp:
        return airMath("exp", 1);
    case Intrinsic::log2:
        return airMath("log2", 1);
    case Intrinsic::log:
        return airMath("log", 1);
    case Intrinsic::log10:
        return airMath("log10", 1);
    case Intrinsic::sin:
        return airMath("sin", 1);
    case Intrinsic::cos:
        return airMath("cos", 1);
    case Intrinsic::pow:
        return airMath("pow", 2);
    case Intrinsic::ctpop:
        return replace(callAir(b, "air.popcount." + airTypeSuffix(rt), rt, {arg(0)}));
    case Intrinsic::ctlz:
        return replace(callAir(b, "air.clz." + airTypeSuffix(rt), rt, {arg(0)}));
    case Intrinsic::cttz:
        return replace(callAir(b, "air.ctz." + airTypeSuffix(rt), rt, {arg(0)}));
    case Intrinsic::bitreverse:
        return replace(callAir(b, "air.reverse_bits." + airTypeSuffix(rt), rt, {arg(0)}));
    case Intrinsic::smin:
        return replace(callAir(b, "air.min." + intSuffix(true), rt, {arg(0), arg(1)}));
    case Intrinsic::smax:
        return replace(callAir(b, "air.max." + intSuffix(true), rt, {arg(0), arg(1)}));
    case Intrinsic::umin:
        return replace(callAir(b, "air.min." + intSuffix(false), rt, {arg(0), arg(1)}));
    case Intrinsic::umax:
        return replace(callAir(b, "air.max." + intSuffix(false), rt, {arg(0), arg(1)}));
    case Intrinsic::abs:
        return replace(callAir(b, "air.abs." + intSuffix(true), rt, {arg(0)}));
    case Intrinsic::sadd_sat:
    case Intrinsic::uadd_sat:
    case Intrinsic::ssub_sat:
    case Intrinsic::usub_sat:
        // Metal accepts the saturating integer intrinsics as they are
        return false;
    case Intrinsic::fshl:
    case Intrinsic::fshr:
    {
        // fshl(a, b, s) = (a << s) | (b >> (bw - s)), shift amount modulo bw
        unsigned bw = rt->getScalarSizeInBits();
        Value *bwv = ConstantInt::get(rt, bw);
        Value *s = b.CreateURem(arg(2), bwv);
        Value *inv = b.CreateSub(bwv, s);
        Value *zero = b.CreateICmpEQ(s, ConstantInt::get(rt, 0));
        Value *res;
        if (id == Intrinsic::fshl)
            res = b.CreateSelect(zero, arg(0), b.CreateOr(b.CreateShl(arg(0), s), b.CreateLShr(arg(1), inv)));
        else
            res = b.CreateSelect(zero, arg(1), b.CreateOr(b.CreateLShr(arg(1), s), b.CreateShl(arg(0), inv)));
        return replace(res);
    }
    case Intrinsic::bswap:
    {
        unsigned bytes = rt->getScalarSizeInBits() / 8;
        Value *res = ConstantInt::get(rt, 0);
        for (unsigned i = 0; i < bytes; i++)
        {
            Value *byte = b.CreateAnd(b.CreateLShr(arg(0), i * 8), ConstantInt::get(rt, 0xFF));
            res = b.CreateOr(res, b.CreateShl(byte, (bytes - 1 - i) * 8));
        }
        return replace(res);
    }
    case Intrinsic::uadd_with_overflow:
    case Intrinsic::sadd_with_overflow:
    case Intrinsic::usub_with_overflow:
    case Intrinsic::ssub_with_overflow:
    case Intrinsic::umul_with_overflow:
    case Intrinsic::smul_with_overflow:
    {
        Value *x = arg(0), *y = arg(1);
        Type *t = x->getType();
        Value *r, *ov;
        unsigned bw = t->getScalarSizeInBits();
        Type *wide = IntegerType::get(t->getContext(), bw * 2);
        switch (id)
        {
        case Intrinsic::uadd_with_overflow:
            r = b.CreateAdd(x, y);
            ov = b.CreateICmpULT(r, x);
            break;
        case Intrinsic::usub_with_overflow:
            r = b.CreateSub(x, y);
            ov = b.CreateICmpULT(x, y);
            break;
        case Intrinsic::sadd_with_overflow:
        case Intrinsic::ssub_with_overflow:
        {
            Value *w = id == Intrinsic::sadd_with_overflow ? b.CreateAdd(b.CreateSExt(x, wide), b.CreateSExt(y, wide))
                                                           : b.CreateSub(b.CreateSExt(x, wide), b.CreateSExt(y, wide));
            r = b.CreateTrunc(w, t);
            ov = b.CreateICmpNE(b.CreateSExt(r, wide), w);
            break;
        }
        case Intrinsic::umul_with_overflow:
        {
            Value *w = b.CreateMul(b.CreateZExt(x, wide), b.CreateZExt(y, wide));
            r = b.CreateTrunc(w, t);
            ov = b.CreateICmpNE(b.CreateLShr(w, bw), ConstantInt::get(wide, 0));
            break;
        }
        default:
        {
            Value *w = b.CreateMul(b.CreateSExt(x, wide), b.CreateSExt(y, wide));
            r = b.CreateTrunc(w, t);
            ov = b.CreateICmpNE(b.CreateSExt(r, wide), w);
            break;
        }
        }
        Value *agg = UndefValue::get(rt);
        agg = b.CreateInsertValue(agg, r, 0);
        agg = b.CreateInsertValue(agg, ov, 1);
        return replace(agg);
    }
    default:
        break;
    }

    // Constrained FP intrinsics (emitted when PTX uses explicit rounding modes):
    // Metal has a single rounding mode, so we drop the constraints.
    if (auto *cfp = dyn_cast<ConstrainedFPIntrinsic>(ci))
    {
        // Portable AIR (for llvm-air2agx): directed roundings stay constrained intrinsics,
        // which the AIR target's air-emulate-f64 computes exactly.
        if (g_portable_air)
        {
            auto rm = cfp->getRoundingMode();
            if (rm && *rm != RoundingMode::NearestTiesToEven && *rm != RoundingMode::Dynamic)
                return true;
        }
        Value *res = nullptr;
        switch (cfp->getIntrinsicID())
        {
        case Intrinsic::experimental_constrained_fadd:
            res = b.CreateFAdd(arg(0), arg(1));
            break;
        case Intrinsic::experimental_constrained_fsub:
            res = b.CreateFSub(arg(0), arg(1));
            break;
        case Intrinsic::experimental_constrained_fmul:
            res = b.CreateFMul(arg(0), arg(1));
            break;
        case Intrinsic::experimental_constrained_fdiv:
            res = b.CreateFDiv(arg(0), arg(1));
            break;
        case Intrinsic::experimental_constrained_frem:
            res = b.CreateFRem(arg(0), arg(1));
            break;
        case Intrinsic::experimental_constrained_fma:
        case Intrinsic::experimental_constrained_fmuladd:
            res = callAir(b, "air.fma." + airTypeSuffix(rt), rt, {arg(0), arg(1), arg(2)});
            break;
        case Intrinsic::experimental_constrained_sqrt:
            res = callAir(b, "air.sqrt." + airTypeSuffix(rt), rt, {arg(0)});
            break;
        case Intrinsic::experimental_constrained_fptosi:
            res = b.CreateFPToSI(arg(0), rt);
            break;
        case Intrinsic::experimental_constrained_fptoui:
            res = b.CreateFPToUI(arg(0), rt);
            break;
        case Intrinsic::experimental_constrained_sitofp:
            res = b.CreateSIToFP(arg(0), rt);
            break;
        case Intrinsic::experimental_constrained_uitofp:
            res = b.CreateUIToFP(arg(0), rt);
            break;
        case Intrinsic::experimental_constrained_fptrunc:
            res = b.CreateFPTrunc(arg(0), rt);
            break;
        case Intrinsic::experimental_constrained_fpext:
            res = b.CreateFPExt(arg(0), rt);
            break;
        case Intrinsic::experimental_constrained_fcmp:
        case Intrinsic::experimental_constrained_fcmps:
            res = b.CreateFCmp(cast<ConstrainedFPCmpIntrinsic>(cfp)->getPredicate(), arg(0), arg(1));
            break;
        case Intrinsic::experimental_constrained_rint:
        case Intrinsic::experimental_constrained_nearbyint:
        case Intrinsic::experimental_constrained_roundeven:
            res = callAir(b, "air.rint." + airTypeSuffix(rt), rt, {arg(0)});
            break;
        case Intrinsic::experimental_constrained_floor:
            res = callAir(b, "air.floor." + airTypeSuffix(rt), rt, {arg(0)});
            break;
        case Intrinsic::experimental_constrained_ceil:
            res = callAir(b, "air.ceil." + airTypeSuffix(rt), rt, {arg(0)});
            break;
        case Intrinsic::experimental_constrained_trunc:
            res = callAir(b, "air.trunc." + airTypeSuffix(rt), rt, {arg(0)});
            break;
        case Intrinsic::experimental_constrained_minnum:
        case Intrinsic::experimental_constrained_minimum:
            res = callAir(b, "air.fmin." + airTypeSuffix(rt), rt, {arg(0), arg(1)});
            break;
        case Intrinsic::experimental_constrained_maxnum:
        case Intrinsic::experimental_constrained_maximum:
            res = callAir(b, "air.fmax." + airTypeSuffix(rt), rt, {arg(0), arg(1)});
            break;
        default:
            fail("unsupported constrained FP intrinsic: " + name);
        }
        return replace(res);
    }

    if (name.starts_with("llvm.amdgcn."))
    {
        StringRef rest = name.drop_front(strlen("llvm.amdgcn."));
        if (rest.consume_front("constrained."))
        {
        }
        if (rest.starts_with("rsq"))
            return replace(callAir(b, "air.rsqrt." + airTypeSuffix(rt), rt, {arg(0)}));
        if (rest.starts_with("sqrt"))
            return replace(callAir(b, "air.sqrt." + airTypeSuffix(rt), rt, {arg(0)}));
        if (rest.starts_with("rcp"))
            return replace(b.CreateFDiv(ConstantFP::get(rt, 1.0), arg(0)));
        if (rest.starts_with("exp2"))
            return replace(callAir(b, "air.exp2." + airTypeSuffix(rt), rt, {arg(0)}));
        if (rest.starts_with("log"))
            return replace(callAir(b, "air.log2." + airTypeSuffix(rt), rt, {arg(0)}));
        if (rest == "wave.barrier")
        {
            callAir(b, "air.simdgroup.barrier", b.getVoidTy(), {b.getInt32(3), b.getInt32(4)});
            return replace(nullptr);
        }
        if (rest == "s.dcache.inv" || rest.starts_with("s.setreg") || rest.starts_with("s.sleep"))
            return replace(nullptr);
        if (rest.starts_with("mul.u24") || rest.starts_with("mul.i24") || rest.starts_with("mulhi.u24") ||
            rest.starts_with("mulhi.i24"))
        {
            bool isSigned = rest.contains(".i24");
            bool hi = rest.starts_with("mulhi");
            Type *i64 = b.getInt64Ty();
            auto ext24 = [&](Value *v) -> Value * {
                if (isSigned)
                    return b.CreateAShr(b.CreateShl(b.CreateSExt(v, i64), 40), 40);
                return b.CreateAnd(b.CreateZExt(v, i64), 0xFFFFFF);
            };
            Value *p = b.CreateMul(ext24(arg(0)), ext24(arg(1)));
            if (hi)
                p = b.CreateLShr(p, 32);
            return replace(b.CreateTrunc(p, rt));
        }
        if (rest.starts_with("udot4") || rest.starts_with("sdot4"))
        {
            bool isSigned = rest.starts_with("sdot4");
            Value *acc = arg(2);
            for (unsigned i = 0; i < 4; i++)
            {
                auto byte = [&](Value *v) -> Value * {
                    Value *x = b.CreateTrunc(b.CreateLShr(v, i * 8), b.getInt8Ty());
                    return isSigned ? b.CreateSExt(x, rt) : b.CreateZExt(x, rt);
                };
                acc = b.CreateAdd(acc, b.CreateMul(byte(arg(0)), byte(arg(1))));
            }
            return replace(acc);
        }
        fail("unsupported AMDGPU intrinsic in Metal backend: " + name);
    }
    // Newer LLVM intrinsics the emitter may name that LLVM 18 does not know
    // portable AIR: llvm-air2agx implements minimumnum / maximumnum exactly (-0 < +0, NaN ignored,
    // subnormals with !agx.keep-denormals)
    if (g_portable_air && (name.starts_with("llvm.minimumnum.") || name.starts_with("llvm.maximumnum.")))
        return false;
    if (name.starts_with("llvm.minimumnum."))
        return airMath("fmin", 2);
    if (name.starts_with("llvm.maximumnum."))
        return airMath("fmax", 2);
    // lifetime markers of a newer clang (one pointer operand) that LLVM 18 does not recognize
    if (name.starts_with("llvm.lifetime."))
        return replace(nullptr);
    if (name.starts_with("llvm."))
        fail("unsupported LLVM intrinsic in Metal backend: " + name);
    return false;
}

// ---------------------------------------------------------------------------
// Kernel wrappers
// ---------------------------------------------------------------------------
enum Builtin
{
    TidInTg,
    TgInGrid,
    TgSize,
    GridInTg,
    SimdLane,
    BuiltinCount
};

struct KernelParam
{
    uint64_t offset;
    uint64_t size;
};

struct KernelInfo
{
    std::string name;
    std::vector<KernelParam> params;
    uint64_t paramsSize = 0;
    bool dynamicShared = false;
    uint64_t staticShared = 0;
    uint64_t globalsParamOffset = UINT64_MAX;
    // offset of the hidden u32 holding the launch's dynamic shared memory size (%dynamic_smem_size), or UINT64_MAX
    uint64_t dynSmemParamOffset = UINT64_MAX;
};

// names of the PTX module's .shared variables (before the helper library is linked): the static shared memory of
// %total_smem_size counts these only
static std::set<std::string> g_ptxShared;

struct DeviceGlobal
{
    GlobalVariable *gv;
    std::string name;
    uint64_t offset;
    uint64_t size;
};

struct DeviceGlobals
{
    std::vector<DeviceGlobal> vars;
    std::vector<uint8_t> init;
};

void constantToBytes(const DataLayout &dl, Constant *c, uint8_t *out)
{
    Type *t = c->getType();
    uint64_t size = dl.getTypeStoreSize(t);
    if (isa<UndefValue>(c) || c->isNullValue())
    {
        memset(out, 0, size);
        return;
    }
    if (auto *ci = dyn_cast<ConstantInt>(c))
    {
        APInt v = ci->getValue();
        for (uint64_t i = 0; i < size; i++)
            out[i] = (uint8_t)v.extractBitsAsZExtValue(std::min<unsigned>(8, v.getBitWidth() - i * 8), i * 8);
        return;
    }
    if (auto *cf = dyn_cast<ConstantFP>(c))
    {
        APInt v = cf->getValueAPF().bitcastToAPInt();
        for (uint64_t i = 0; i < size; i++)
            out[i] = (uint8_t)v.extractBitsAsZExtValue(8, i * 8);
        return;
    }
    if (auto *cds = dyn_cast<ConstantDataSequential>(c))
    {
        uint64_t elemSize = dl.getTypeAllocSize(cds->getElementType());
        for (unsigned i = 0; i < cds->getNumElements(); i++)
            constantToBytes(dl, cds->getElementAsConstant(i), out + i * elemSize);
        return;
    }
    if (auto *ca = dyn_cast<ConstantAggregate>(c))
    {
        if (auto *st = dyn_cast<StructType>(t))
        {
            const StructLayout *sl = dl.getStructLayout(st);
            memset(out, 0, size);
            for (unsigned i = 0; i < ca->getNumOperands(); i++)
                constantToBytes(dl, ca->getOperand(i), out + sl->getElementOffset(i));
            return;
        }
        Type *elem = isa<ArrayType>(t) ? t->getArrayElementType() : cast<VectorType>(t)->getElementType();
        uint64_t elemSize = dl.getTypeAllocSize(elem);
        for (unsigned i = 0; i < ca->getNumOperands(); i++)
            constantToBytes(dl, ca->getOperand(i), out + i * elemSize);
        return;
    }
    fail("unsupported initializer of a device global: " + toString(c));
}

// Writable module-scope device variables: Metal has no program-scope device
// memory, so they live in one buffer allocated by the runtime per module.
DeviceGlobals collectDeviceGlobals(Module &m)
{
    const DataLayout &dl = m.getDataLayout();
    DeviceGlobals result;
    uint64_t offset = 0;
    for (GlobalVariable &gv : m.globals())
    {
        if (gv.getAddressSpace() != air::Device)
            continue;
        if (gv.isDeclaration())
            fail("external device variable " + gv.getName() + " is not supported");
        uint64_t align = std::max<uint64_t>(gv.getAlign().valueOrOne().value(), 16);
        uint64_t size = dl.getTypeAllocSize(gv.getValueType());
        offset = alignTo(offset, align);
        result.vars.push_back({&gv, gv.getName().str(), offset, size});
        offset += size;
    }
    // uses inside constant expressions (getelementptr (@x, 8)) become instructions, which the
    // kernel wrappers rewrite to the device-globals buffer
    std::vector<Constant *> cs;
    for (auto &v : result.vars)
        cs.push_back(v.gv);
    convertUsersOfConstantsToInstructions(cs);
    result.init.assign(offset, 0);
    for (auto &v : result.vars)
        if (v.gv->hasInitializer())
            constantToBytes(dl, v.gv->getInitializer(), result.init.data() + v.offset);
    return result;
}

MDString *mds(LLVMContext &c, StringRef s) { return MDString::get(c, s); }
Metadata *mdi(LLVMContext &c, uint32_t v) { return ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(c), v)); }

bool isDynamicSharedGlobal(const GlobalVariable &gv)
{
    if (gv.getAddressSpace() != air::Threadgroup)
        return false;
    Type *t = gv.getValueType();
    if (auto *at = dyn_cast<ArrayType>(t))
        return at->getNumElements() == 0;
    return false;
}

KernelInfo buildKernelWrapper(Module &m, Function *body, const DeviceGlobals &globals)
{
    LLVMContext &c = m.getContext();
    const DataLayout &dl = m.getDataLayout();
    KernelInfo info;
    info.name = body->getName().str();
    body->setName(info.name + ".zluda_body");
    body->setCallingConv(CallingConv::C);
    body->setLinkage(GlobalValue::InternalLinkage);

    // CUDA parameter buffer layout: every parameter at its natural (or
    // explicitly requested) alignment, in declaration order.
    uint64_t offset = 0;
    std::vector<Type *> paramTypes;
    for (Argument &a : body->args())
    {
        Type *t = a.getParamByRefType();
        if (!t)
            fail("kernel parameter without byref type in " + info.name);
        uint64_t align = std::max<uint64_t>(dl.getABITypeAlign(t).value(), a.getParamAlign().valueOrOne().value());
        uint64_t size = dl.getTypeAllocSize(t);
        offset = alignTo(offset, align);
        info.params.push_back({offset, size});
        offset += size;
        paramTypes.push_back(t);
    }
    // Device globals used by this kernel: their base address is appended to
    // the parameter buffer
    bool usesGlobals = false;
    {
        std::vector<Constant *> cs;
        for (const DeviceGlobal &g : globals.vars)
            cs.push_back(g.gv);
        convertUsersOfConstantsToInstructions(cs);
    }
    for (const DeviceGlobal &g : globals.vars)
        for (User *u : g.gv->users())
            if (auto *i = dyn_cast<Instruction>(u); i && i->getFunction() == body)
                usesGlobals = true;
    if (usesGlobals)
    {
        offset = alignTo(offset, 8);
        info.globalsParamOffset = offset;
        offset += 8;
    }
    bool usesDynSmemSize = false;
    if (Function *ph = m.getFunction("__zluda_metal_dynamic_smem_size"))
        for (User *u : ph->users())
            if (auto *i = dyn_cast<Instruction>(u); i && i->getFunction() == body)
                usesDynSmemSize = true;
    if (usesDynSmemSize)
    {
        offset = alignTo(offset, 4);
        info.dynSmemParamOffset = offset;
        offset += 4;
    }
    info.paramsSize = offset;

    // Does the kernel use dynamic shared memory?
    GlobalVariable *dynShared = nullptr;
    for (GlobalVariable &gv : m.globals())
    {
        if (!isDynamicSharedGlobal(gv))
            continue;
        for (User *u : gv.users())
            if (auto *i = dyn_cast<Instruction>(u))
                if (i->getFunction() == body)
                    dynShared = &gv;
    }
    info.dynamicShared = dynShared != nullptr;

    // Per-CTA global memory (llvm-air2agx's hidden argument "perctamemory", AGXUsage.rst extension 8):
    // the helpers call __zluda_metal_percta_memory(bytes) for the base of the executing CTA's region
    // (the tensor family's tensor memory and wgmma scratch); the region is the largest size asked for.
    SmallVector<CallInst *, 4> perCtaCalls;
    uint64_t perCtaBytes = 0;
    for (Instruction &i : instructions(body))
        if (auto *ci = dyn_cast<CallInst>(&i))
            if (Function *f = ci->getCalledFunction(); f && f->getName() == "__zluda_metal_percta_memory")
            {
                auto *n = dyn_cast<ConstantInt>(ci->getArgOperand(0));
                if (!n)
                    fail("__zluda_metal_percta_memory with a non-constant size in " + info.name);
                perCtaBytes = std::max<uint64_t>(perCtaBytes, n->getZExtValue());
                perCtaCalls.push_back(ci);
            }
    perCtaBytes = alignTo(perCtaBytes, 1024);
    if (!perCtaCalls.empty() && !g_portable_air)
        fail("kernel " + info.name + " needs per-CTA global memory, which only the AGX backend provides");

    Type *i32 = Type::getInt32Ty(c);
    Type *v3 = FixedVectorType::get(i32, 3);
    PointerType *constPtr = PointerType::get(c, air::Constant);
    PointerType *tgPtr = PointerType::get(c, air::Threadgroup);
    std::vector<Type *> wrapperParams = {constPtr, v3, v3, v3, v3, i32};
    if (dynShared)
        wrapperParams.push_back(tgPtr);
    const unsigned perCtaArg = wrapperParams.size();
    if (!perCtaCalls.empty())
        wrapperParams.push_back(PointerType::get(c, air::Device));
    Function *w = Function::Create(FunctionType::get(Type::getVoidTy(c), wrapperParams, false),
                                   GlobalValue::ExternalLinkage, info.name, &m);
    w->addFnAttr(Attribute::NoUnwind);
    w->addFnAttr("frame-pointer", "none");
    if (body->hasFnAttribute("zluda-cluster-dims-agx"))
        w->addFnAttr("agx-cluster-dims", body->getFnAttribute("zluda-cluster-dims-agx").getValueAsString());
    w->addParamAttr(0, Attribute::NoUndef);
    const char *argNames[] = {"zluda_params", "tid", "ctaid", "ntid", "nctaid", "laneid", "zluda_dynamic_shared"};
    for (unsigned i = 0; i < w->arg_size() && i < 7; i++)
        w->getArg(i)->setName(argNames[i]);
    if (!perCtaCalls.empty())
    {
        w->getArg(perCtaArg)->setName("zluda_percta");
        w->addParamAttr(perCtaArg, Attribute::get(c, "agx-hidden", "perctamemory:" + std::to_string(perCtaBytes)));
    }

    BasicBlock *entry = BasicBlock::Create(c, "entry", w);
    IRBuilder<> b(entry);
    SmallVector<Value *, 8> args;
    for (unsigned i = 0; i < info.params.size(); i++)
        args.push_back(b.CreateConstInBoundsGEP1_64(b.getInt8Ty(), w->getArg(0), info.params[i].offset));
    CallInst *call = b.CreateCall(body, args);
    b.CreateRetVoid();
    InlineFunctionInfo ifi;
    InlineResult ir = InlineFunction(*call, ifi);
    if (!ir.isSuccess())
        fail("failed to inline kernel body of " + info.name + ": " + ir.getFailureReason());
    if (body->use_empty())
        body->eraseFromParent();
    if (usesGlobals)
    {
        IRBuilder<> gb(&*w->getEntryBlock().getFirstInsertionPt());
        Value *addr = gb.CreateLoad(gb.getInt64Ty(),
                                    gb.CreateConstInBoundsGEP1_64(gb.getInt8Ty(), w->getArg(0), info.globalsParamOffset),
                                    "zluda_globals_address");
        Value *base = gb.CreateIntToPtr(addr, PointerType::get(c, air::Device), "zluda_globals");
        {
            std::vector<Constant *> cs;
            for (const DeviceGlobal &g : globals.vars)
                cs.push_back(g.gv);
            convertUsersOfConstantsToInstructions(cs);
        }
        for (const DeviceGlobal &g : globals.vars)
        {
            SmallVector<Use *, 8> uses;
            for (Use &u : g.gv->uses())
                if (auto *i = dyn_cast<Instruction>(u.getUser()); i && i->getFunction() == w)
                    uses.push_back(&u);
            if (uses.empty())
                continue;
            Value *ptr = gb.CreateConstInBoundsGEP1_64(gb.getInt8Ty(), base, g.offset, g.name);
            for (Use *u : uses)
                u->set(ptr);
        }
    }

    // %dynamic_smem_size / the static .shared bytes of %total_smem_size (ptx/lib/metal_impl/sreg.inc)
    for (const char *name : {"__zluda_metal_dynamic_smem_size", "__zluda_metal_static_smem_size"})
    {
        Function *ph = m.getFunction(name);
        if (!ph)
            continue;
        SmallVector<CallInst *, 4> calls;
        for (User *u : ph->users())
            if (auto *ci = dyn_cast<CallInst>(u); ci && ci->getFunction() == w)
                calls.push_back(ci);
        if (calls.empty())
            continue;
        Value *v;
        if (StringRef(name) == "__zluda_metal_dynamic_smem_size")
        {
            IRBuilder<> gb(&*w->getEntryBlock().getFirstInsertionPt());
            v = gb.CreateLoad(gb.getInt32Ty(),
                              gb.CreateConstInBoundsGEP1_64(gb.getInt8Ty(), w->getArg(0), info.dynSmemParamOffset),
                              "zluda_dynamic_smem_size");
        }
        else
        {
            uint64_t total = 0;
            for (GlobalVariable &gv : m.globals())
            {
                if (gv.getAddressSpace() != air::Threadgroup || isDynamicSharedGlobal(gv))
                    continue;
                bool used = false;
                for (User *u : gv.users())
                    if (auto *i = dyn_cast<Instruction>(u); i && i->getFunction() == w)
                        used = true;
                if (!used)
                    continue;
                uint64_t size = dl.getTypeAllocSize(gv.getValueType());
                if (gv.getName() == "__zluda_cta_shared")
                {
                    if (MDNode *md = gv.getMetadata("zluda.cluster"))
                        size = mdconst::extract<ConstantInt>(md->getOperand(0))->getZExtValue();
                }
                else if (!g_ptxShared.count(gv.getName().str()))
                    continue;
                uint64_t al = gv.getAlign() ? gv.getAlign()->value() : 1;
                total = alignTo(total, al) + size;
            }
            v = ConstantInt::get(Type::getInt32Ty(c), total);
        }
        for (CallInst *ci : calls)
        {
            ci->replaceAllUsesWith(v);
            ci->eraseFromParent();
        }
    }

    // Builtins
    std::map<std::string, unsigned> builtinArg = {
        {"__zluda_metal_thread_position_in_threadgroup", 1},
        {"__zluda_metal_threadgroup_position_in_grid", 2},
        {"__zluda_metal_threads_per_threadgroup", 3},
        {"__zluda_metal_threadgroups_per_grid", 4},
        {"__zluda_metal_thread_index_in_simdgroup", 5},
    };
    SmallVector<CallInst *, 16> calls;
    for (Instruction &i : instructions(w))
        if (auto *ci = dyn_cast<CallInst>(&i))
            if (Function *f = ci->getCalledFunction())
                if (builtinArg.count(f->getName().str()))
                    calls.push_back(ci);
    for (CallInst *ci : calls)
    {
        unsigned idx = builtinArg[ci->getCalledFunction()->getName().str()];
        IRBuilder<> cb(ci);
        Value *v = w->getArg(idx);
        if (idx != 5)
            v = cb.CreateExtractElement(v, ci->getArgOperand(0));
        ci->replaceAllUsesWith(v);
        ci->eraseFromParent();
    }
    if (!perCtaCalls.empty())
    {
        // CTA c = x + nx * (y + ny * z) has the bytes from c * size
        SmallVector<CallInst *, 4> pc;
        for (Instruction &i : instructions(w))
            if (auto *ci = dyn_cast<CallInst>(&i))
                if (Function *f = ci->getCalledFunction(); f && f->getName() == "__zluda_metal_percta_memory")
                    pc.push_back(ci);
        IRBuilder<> pb(&*w->getEntryBlock().getFirstInsertionPt());
        Value *ct = w->getArg(2), *nct = w->getArg(4);
        Value *lin = pb.CreateAdd(pb.CreateExtractElement(ct, (uint64_t)0),
                                  pb.CreateMul(pb.CreateExtractElement(nct, (uint64_t)0),
                                               pb.CreateAdd(pb.CreateExtractElement(ct, 1),
                                                            pb.CreateMul(pb.CreateExtractElement(nct, 1),
                                                                         pb.CreateExtractElement(ct, 2)))));
        Value *base = pb.CreateAdd(pb.CreatePtrToInt(w->getArg(perCtaArg), pb.getInt64Ty()),
                                   pb.CreateMul(pb.CreateZExt(lin, pb.getInt64Ty()), pb.getInt64(perCtaBytes)),
                                   "zluda_percta_base");
        for (CallInst *ci : pc)
        {
            ci->replaceAllUsesWith(base);
            ci->eraseFromParent();
        }
    }
    if (dynShared)
    {
        SmallVector<Use *, 8> uses;
        for (Use &u : dynShared->uses())
            if (auto *i = dyn_cast<Instruction>(u.getUser()))
                if (i->getFunction() == w)
                    uses.push_back(&u);
        for (Use *u : uses)
            u->set(w->getArg(6));
    }

    // !air.kernel metadata
    std::vector<Metadata *> argMd;
    argMd.push_back(MDNode::get(
        c, {mdi(c, 0), mds(c, "air.buffer"), mds(c, "air.location_index"), mdi(c, 0), mdi(c, 1), mds(c, "air.read"),
            mds(c, "air.address_space"), mdi(c, air::Constant), mds(c, "air.arg_type_size"), mdi(c, 1),
            mds(c, "air.arg_type_align_size"), mdi(c, 1), mds(c, "air.arg_type_name"), mds(c, "uchar"),
            mds(c, "air.arg_name"), mds(c, "zluda_params")}));
    const char *builtinMd[] = {"air.thread_position_in_threadgroup", "air.threadgroup_position_in_grid",
                               "air.threads_per_threadgroup", "air.threadgroups_per_grid"};
    for (unsigned i = 0; i < 4; i++)
        argMd.push_back(MDNode::get(c, {mdi(c, i + 1), mds(c, builtinMd[i]), mds(c, "air.arg_type_name"),
                                        mds(c, "uint3"), mds(c, "air.arg_name"), mds(c, argNames[i + 1])}));
    argMd.push_back(MDNode::get(c, {mdi(c, 5), mds(c, "air.thread_index_in_simdgroup"), mds(c, "air.arg_type_name"),
                                    mds(c, "uint"), mds(c, "air.arg_name"), mds(c, "laneid")}));
    if (dynShared)
    {
        uint64_t align = std::max<uint64_t>(16, dynShared->getAlign().valueOrOne().value());
        argMd.push_back(MDNode::get(
            c, {mdi(c, 6), mds(c, "air.threadgroup"), mds(c, "air.location_index"), mdi(c, 0), mdi(c, 1),
                mds(c, "air.read_write"), mds(c, "air.address_space"), mdi(c, air::Threadgroup),
                mds(c, "air.arg_type_size"), mdi(c, 1), mds(c, "air.arg_type_align_size"), mdi(c, (uint32_t)align),
                mds(c, "air.arg_type_name"), mds(c, "uchar"), mds(c, "air.arg_name"), mds(c, argNames[6])}));
    }
    if (!perCtaCalls.empty())
        argMd.push_back(MDNode::get(
            c, {mdi(c, perCtaArg), mds(c, "air.buffer"), mds(c, "air.location_index"), mdi(c, 1), mdi(c, 1),
                mds(c, "air.read_write"), mds(c, "air.address_space"), mdi(c, air::Device), mds(c, "air.arg_type_size"),
                mdi(c, 4), mds(c, "air.arg_type_align_size"), mdi(c, 4), mds(c, "air.arg_type_name"), mds(c, "uint"),
                mds(c, "air.arg_name"), mds(c, "zluda_percta")}));
    MDNode *kernelMd =
        MDNode::get(c, {ValueAsMetadata::get(w), MDNode::get(c, {}), MDNode::get(c, argMd)});
    m.getOrInsertNamedMetadata("air.kernel")->addOperand(kernelMd);
    return info;
}

void addModuleMetadata(Module &m)
{
    LLVMContext &c = m.getContext();
    for (const char *name : {"llvm.module.flags", "llvm.ident", "opencl.ocl.version"})
        if (NamedMDNode *n = m.getNamedMetadata(name))
            m.eraseNamedMetadata(n);
    m.addModuleFlag(Module::Max, "wchar_size", 4);
    m.addModuleFlag(Module::Max, "frame-pointer", 2);
    m.addModuleFlag(Module::Max, "air.max_device_buffers", 31);
    m.addModuleFlag(Module::Max, "air.max_constant_buffers", 31);
    m.addModuleFlag(Module::Max, "air.max_threadgroup_buffers", 31);
    m.addModuleFlag(Module::Max, "air.max_textures", 128);
    m.addModuleFlag(Module::Max, "air.max_read_write_textures", 8);
    m.addModuleFlag(Module::Max, "air.max_samplers", 16);
    auto *opts = m.getOrInsertNamedMetadata("air.compile_options");
    opts->addOperand(MDNode::get(c, {mds(c, "air.compile.denorms_disable")}));
    opts->addOperand(MDNode::get(c, {mds(c, "air.compile.fast_math_disable")}));
    opts->addOperand(MDNode::get(c, {mds(c, "air.compile.framebuffer_fetch_enable")}));
    m.getOrInsertNamedMetadata("air.version")->addOperand(MDNode::get(c, {mdi(c, 2), mdi(c, 6), mdi(c, 0)}));
    m.getOrInsertNamedMetadata("air.language_version")
        ->addOperand(MDNode::get(c, {mds(c, "Metal"), mdi(c, 3), mdi(c, 1), mdi(c, 0)}));
    m.getOrInsertNamedMetadata("llvm.ident")->addOperand(MDNode::get(c, {mds(c, "ZLUDA Metal backend")}));
}

void stripTargetAttributes(Module &m)
{
    for (Function &f : m)
    {
        AttributeList al = f.getAttributes();
        AttrBuilder ab(m.getContext(), al.getFnAttrs());
        for (const Attribute &a : al.getFnAttrs())
            if (a.isStringAttribute() && !(g_portable_air && a.getKindAsString().starts_with("denormal-fp-math")) &&
                a.getKindAsString() != "agx-cluster-dims")
                ab.removeAttribute(a.getKindAsString());
        ab.removeAttribute(Attribute::StrictFP);
        f.setAttributes(al.removeFnAttributes(m.getContext()).addFnAttributes(m.getContext(), ab));
        for (Instruction &i : instructions(f))
            if (auto *cb = dyn_cast<CallBase>(&i))
            {
                AttributeList cal = cb->getAttributes();
                AttrBuilder cab(m.getContext(), cal.getFnAttrs());
                for (const Attribute &a : cal.getFnAttrs())
                    if (a.isStringAttribute())
                        cab.removeAttribute(a.getKindAsString());
                cab.removeAttribute(Attribute::StrictFP);
                cab.removeAttribute(Attribute::NoBuiltin);
                cb->setAttributes(cal.removeFnAttributes(m.getContext()).addFnAttributes(m.getContext(), cab));
            }
        f.setVisibility(GlobalValue::DefaultVisibility);
        f.setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    }
    std::vector<Function *> unused;
    for (Function &f : m)
        if (f.isDeclaration() && f.use_empty())
            unused.push_back(&f);
    for (Function *f : unused)
        f->eraseFromParent();
}


// The translator declares a helper with PTX's types ({i32, i1} for `d|p`, i1 predicates)
// while C++ spells them with its own ({i32, i8} or <2 x i32>, i8 / i1 booleans): calls whose
// type differs from the linked definition are rewritten with element-wise conversions, so
// that the definitions inline.
static Value *adaptValue(IRBuilder<> &b, Value *v, Type *to)
{
    Type *from = v->getType();
    if (from == to)
        return v;
    if (from->isIntegerTy() && to->isIntegerTy())
    {
        if (from->getIntegerBitWidth() == 1)
            return b.CreateZExt(v, to);
        if (to->getIntegerBitWidth() == 1)
            return b.CreateICmpNE(v, ConstantInt::get(from, 0));
        return b.CreateZExtOrTrunc(v, to);
    }
    if (from->isPointerTy() && to->isPointerTy())
        return b.CreateAddrSpaceCast(v, to);
    auto elems = [](Type *t) -> unsigned {
        if (auto *st = dyn_cast<StructType>(t))
            return st->getNumElements();
        if (auto *vt = dyn_cast<FixedVectorType>(t))
            return vt->getNumElements();
        return 0;
    };
    auto elemType = [](Type *t, unsigned i) -> Type * {
        if (auto *st = dyn_cast<StructType>(t))
            return st->getElementType(i);
        return cast<FixedVectorType>(t)->getElementType();
    };
    if (elems(from) && elems(from) == elems(to))
    {
        Value *r = PoisonValue::get(to);
        for (unsigned i = 0; i < elems(from); i++)
        {
            Value *e = from->isStructTy() ? b.CreateExtractValue(v, i) : b.CreateExtractElement(v, (uint64_t)i);
            e = adaptValue(b, e, elemType(to, i));
            r = to->isStructTy() ? b.CreateInsertValue(r, e, i) : b.CreateInsertElement(r, e, (uint64_t)i);
        }
        return r;
    }
    if (from->getPrimitiveSizeInBits() == to->getPrimitiveSizeInBits() && from->getPrimitiveSizeInBits() != 0)
        return b.CreateBitCast(v, to);
    return nullptr;
}

static void adaptHelperCalls(Module &m)
{
    std::vector<CallInst *> calls;
    for (Function &f : m)
        for (Instruction &i : instructions(f))
            if (auto *ci = dyn_cast<CallInst>(&i))
                if (auto *callee = dyn_cast<Function>(ci->getCalledOperand()->stripPointerCasts()))
                    if (!callee->isDeclaration() && ci->getFunctionType() != callee->getFunctionType())
                        calls.push_back(ci);
    for (CallInst *ci : calls)
    {
        auto *callee = cast<Function>(ci->getCalledOperand()->stripPointerCasts());
        FunctionType *ft = callee->getFunctionType();
        if (ft->getNumParams() != ci->arg_size())
            continue;
        IRBuilder<> b(ci);
        std::vector<Value *> args;
        bool ok = true;
        for (unsigned i = 0; i < ci->arg_size() && ok; i++)
        {
            Value *a = adaptValue(b, ci->getArgOperand(i), ft->getParamType(i));
            ok = a != nullptr;
            args.push_back(a);
        }
        if (!ok)
            continue;
        CallInst *nc = b.CreateCall(ft, callee, args);
        if (!ci->getType()->isVoidTy())
        {
            Value *r = ft->getReturnType()->isVoidTy() ? nullptr : adaptValue(b, nc, ci->getType());
            if (!r)
            {
                nc->eraseFromParent();
                continue;
            }
            ci->replaceAllUsesWith(r);
        }
        ci->eraseFromParent();
    }
}


// ---- Floating point modes for portable AIR ----
// ZLUDA expresses PTX's per-instruction rounding (.rz, .rm, .rp) and .ftz as AMDGPU mode
// register writes (llvm.amdgcn.s.setreg of the rounding and denormal fields) around
// constrained intrinsics with dynamic rounding. AGX has no mode register: a forward dataflow
// gives each instruction its mode, the constrained intrinsics get static roundings (the AIR
// target's air-emulate-f64 computes them exactly), every function flushes float denormals,
// as the hardware does, and the float operations that must keep them carry
// !agx.keep-denormals (the AGX backend's IEEE lowering, llvm/docs/AGXUsage.rst).
struct FPMode
{
    int r32 = 0, r16 = 0; // AMDGPU rounding field: 0 nearest, 1 +inf, 2 -inf, 3 zero
    bool ieee32 = true, ieee16 = true;
    bool operator==(const FPMode &o) const
    {
        return r32 == o.r32 && r16 == o.r16 && ieee32 == o.ieee32 && ieee16 == o.ieee16;
    }
};

static const char *roundingName(int r)
{
    switch (r)
    {
    case 1: return "round.upward";
    case 2: return "round.downward";
    case 3: return "round.towardzero";
    default: return "round.tonearest";
    }
}

static bool isF32Op(Instruction &i)
{
    auto f32 = [](Type *t) { return t->getScalarType()->isFloatTy(); };
    if (isa<FCmpInst>(i))
        return f32(i.getOperand(0)->getType());
    if (!f32(i.getType()))
        return false;
    if (i.isBinaryOp() || isa<UnaryOperator>(i))
        return true;
    if (auto *ci = dyn_cast<CallInst>(&i))
        if (Function *f = ci->getCalledFunction())
        {
            if (f->getName().starts_with("air.") || f->getName().starts_with("llvm.minimumnum.") ||
                f->getName().starts_with("llvm.maximumnum."))
                return true; // AIR math (air.sqrt.f32, air.fma.f32, ...), intrinsics LLVM 18 does not know
            switch (f->getIntrinsicID())
            {
            case Intrinsic::fma:
            case Intrinsic::fmuladd:
            case Intrinsic::sqrt:
            case Intrinsic::minnum:
            case Intrinsic::maxnum:
            case Intrinsic::minimum:
            case Intrinsic::maximum:
            case Intrinsic::floor:
            case Intrinsic::ceil:
            case Intrinsic::trunc:
            case Intrinsic::rint:
            case Intrinsic::nearbyint:
            case Intrinsic::round:
            case Intrinsic::roundeven:
            case Intrinsic::canonicalize:
                return true;
            default:
                return isa<ConstrainedFPIntrinsic>(ci);
            }
        }
    return false;
}

static bool resolveFPModes(Function &f)
{
    if (f.isDeclaration())
        return true;
    LLVMContext &c = f.getContext();
    FPMode entry;
    if (Attribute a = f.getFnAttribute("denormal-fp-math-f32"); a.isValid())
        entry.ieee32 = a.getValueAsString().starts_with("ieee");
    if (Attribute a = f.getFnAttribute("denormal-fp-math"); a.isValid())
        entry.ieee16 = a.getValueAsString().starts_with("ieee") || a.getValueAsString().starts_with("dynamic");
    std::map<BasicBlock *, FPMode> in;
    std::vector<BasicBlock *> work{&f.getEntryBlock()};
    in[&f.getEntryBlock()] = entry;
    MDNode *keep = MDNode::get(c, {});
    std::set<BasicBlock *> done;
    std::vector<std::pair<Instruction *, Value *>> pendingErase;
    while (!work.empty())
    {
        BasicBlock *bb = work.back();
        work.pop_back();
        if (!done.insert(bb).second)
            continue;
        FPMode m = in[bb];
        for (Instruction &i : *bb)
        {
            if (auto *ci = dyn_cast<CallInst>(&i))
            {
                Function *callee = ci->getCalledFunction();
                if (callee && callee->getName() == "llvm.amdgcn.s.setreg")
                {
                    auto *hw = dyn_cast<ConstantInt>(ci->getArgOperand(0));
                    auto *v = dyn_cast<ConstantInt>(ci->getArgOperand(1));
                    if (hw && v)
                    {
                        uint64_t val = v->getZExtValue();
                        if (hw->getZExtValue() == 6145) // rounding field
                        {
                            m.r32 = val & 3;
                            m.r16 = (val >> 2) & 3;
                        }
                        else if (hw->getZExtValue() == 6401) // denormal field
                        {
                            m.ieee32 = (val & 3) != 0;
                            m.ieee16 = ((val >> 2) & 3) != 0;
                        }
                    }
                    continue;
                }
                if (auto *cfp = dyn_cast<ConstrainedFPIntrinsic>(ci))
                {
                    auto rm = cfp->getRoundingMode();
                    if (rm && *rm == RoundingMode::Dynamic)
                    {
                        Type *t = cfp->getType()->getScalarType();
                        if (cfp->getIntrinsicID() == Intrinsic::experimental_constrained_fcmp ||
                            cfp->getIntrinsicID() == Intrinsic::experimental_constrained_fcmps)
                            t = cfp->getArgOperand(0)->getType()->getScalarType();
                        int r = t->isFloatTy() ? m.r32 : m.r16;
                        unsigned idx = cfp->arg_size() - 2; // rounding, then exception behaviour
                        ci->setArgOperand(idx, MetadataAsValue::get(c, MDString::get(c, roundingName(r))));
                    }
                }
            }
            // (ZLUDA_AGX_FTZ_ALL: a measurement aid, not PTX: every float operation flushes)
            // (floating point mode fast-ftz: the hardware's flushing operations)
            if (m.ieee32 && isF32Op(i) && !g_fp_ftz && !getenv("ZLUDA_AGX_FTZ_ALL"))
                i.setMetadata("agx.keep-denormals", keep);
            // plain operations in a directed-rounding region (the helper library's) round in it too
            if (auto *bo = dyn_cast<BinaryOperator>(&i))
            {
                Type *t = bo->getType()->getScalarType();
                int r = t->isFloatTy() ? m.r32 : m.r16;
                Intrinsic::ID cid = Intrinsic::not_intrinsic;
                switch (bo->getOpcode())
                {
                case Instruction::FAdd: cid = Intrinsic::experimental_constrained_fadd; break;
                case Instruction::FSub: cid = Intrinsic::experimental_constrained_fsub; break;
                case Instruction::FMul: cid = Intrinsic::experimental_constrained_fmul; break;
                case Instruction::FDiv: cid = Intrinsic::experimental_constrained_fdiv; break;
                default: break;
                }
                if (r != 0 && cid != Intrinsic::not_intrinsic && t->isFloatingPointTy() && !bo->getType()->isVectorTy())
                {
                    IRBuilder<> b(bo);
                    Function *cf = Intrinsic::getDeclaration(f.getParent(), cid, {bo->getType()});
                    Value *nv = b.CreateCall(cf, {bo->getOperand(0), bo->getOperand(1),
                                                  MetadataAsValue::get(c, MDString::get(c, roundingName(r))),
                                                  MetadataAsValue::get(c, MDString::get(c, "fpexcept.ignore"))});
                    if (MDNode *md = bo->getMetadata("agx.keep-denormals"))
                        cast<Instruction>(nv)->setMetadata("agx.keep-denormals", md);
                    pendingErase.push_back({bo, nv});
                }
            }
        }
        for (BasicBlock *s : successors(bb))
            if (!in.count(s))
            {
                in[s] = m;
                work.push_back(s);
            }
    }
    for (auto &[old, nv] : pendingErase)
    {
        old->replaceAllUsesWith(nv);
        old->eraseFromParent();
    }
    return entry.ieee32;
}

// Fast floating point mode: the .approx transcendentals of the library (the formal model's definitions, in
// software) become short float evaluations: the hardware's approximations (air.fast_<fn>, agx-lower-air: the
// LLVM intrinsic with afn), tanh by tanhF. Half precision forms are computed in float and rounded to nearest.
// Division and square root stay correctly rounded (a reciprocal-based division is wrong by far more than a few
// ulp for divisors beyond 2^126), as does div.approx (whose range behavior PTX defines). Control words as in
// ptx_parser/src/helper_table/float.rs: bits 4-7 type (1 f32, 2 f16, 3 bf16), 8 packed, 12 .ftz; fsincos_f32
// bit 0 cos.
static void fastFPHelpers(Module &m)
{
    LLVMContext &c = m.getContext();
    Type *f32 = Type::getFloatTy(c), *i32 = Type::getInt32Ty(c), *i16 = Type::getInt16Ty(c);
    auto fast = [&](IRBuilder<> &b, StringRef fn, Value *x) -> Value * {
        FunctionCallee callee = m.getOrInsertFunction(("air.fast_" + fn + ".f32").str(), FunctionType::get(f32, {f32}, false));
        if (auto *d = dyn_cast<Function>(callee.getCallee()))
        {
            d->setDoesNotAccessMemory();
            d->setDoesNotThrow();
            d->addFnAttr(Attribute::WillReturn);
        }
        return b.CreateCall(callee, {x});
    };
    // tanh x within ~2 ulp: |x| < 0.6 x + x^3 q(x^2) (degree-4 least-squares fit, 0.9 ulp in float), else
    // 1 - 2 / (2^(2x log2 e) + 1) (1.5 ulp, and the hardware's exp2 and reciprocal), +-1 from |x| >= 9; sign
    // restored (tanh is odd), NaN propagated
    auto tanhF = [&](IRBuilder<> &b, Value *x) -> Value * {
        auto k = [&](double v) { return ConstantFP::get(f32, v); };
        Value *ax = b.CreateUnaryIntrinsic(Intrinsic::fabs, x);
        Value *z = b.CreateFMul(x, x);
        Value *p = k(-0.006339305080473423);
        for (double cf : {0.02116510644555092, -0.05388752743601799, 0.1333303451538086, -0.3333333134651184})
            p = b.CreateIntrinsic(Intrinsic::fma, {f32}, {p, z, k(cf)});
        Value *small = b.CreateIntrinsic(Intrinsic::fma, {f32}, {b.CreateFMul(x, z), p, x});
        Value *e = fast(b, "exp2", b.CreateFMul(ax, k(2.0 * 1.4426950408889634)));
        Value *q = b.CreateFDiv(k(2.0), b.CreateFAdd(e, k(1.0)));
        FastMathFlags fmf;
        fmf.setApproxFunc();
        fmf.setAllowReciprocal();
        cast<Instruction>(q)->setFastMathFlags(fmf);
        Value *big = b.CreateSelect(b.CreateFCmpOGE(ax, k(9.0)), k(1.0), b.CreateFSub(k(1.0), q));
        big = b.CreateIntrinsic(Intrinsic::copysign, {f32}, {big, x});
        Value *r = b.CreateSelect(b.CreateFCmpOLT(ax, k(0.6)), small, big);
        return b.CreateSelect(b.CreateFCmpUNO(x, x), x, r);
    };
    auto fn1 = [&](IRBuilder<> &b, StringRef fn, Value *x) -> Value * {
        return fn == "tanh" ? tanhF(b, x) : fast(b, fn, x);
    };
    // a 16-bit lane (bits in the low half of an i32) as float, and back (nearest even; NaN canonical)
    auto toF = [&](IRBuilder<> &b, Value *lane, unsigned t) -> Value * {
        Value *h = b.CreateTrunc(lane, i16);
        if (t == 2)
            return b.CreateFPExt(b.CreateBitCast(h, Type::getHalfTy(c)), f32);
        return b.CreateBitCast(b.CreateShl(b.CreateZExt(h, i32), 16), f32);
    };
    auto fromF = [&](IRBuilder<> &b, Value *v, unsigned t) -> Value * {
        if (t == 2)
            return b.CreateZExt(b.CreateBitCast(b.CreateFPTrunc(v, Type::getHalfTy(c)), i16), i32);
        Value *u = b.CreateBitCast(v, i32);
        Value *odd = b.CreateAnd(b.CreateLShr(u, 16), 1);
        Value *r = b.CreateLShr(b.CreateAdd(u, b.CreateAdd(b.getInt32(0x7FFF), odd)), 16);
        return b.CreateSelect(b.CreateFCmpUNO(v, v), b.getInt32(0x7FFF), r);
    };
    std::vector<CallInst *> calls;
    for (Function &f : m)
        for (Instruction &i : instructions(f))
            if (auto *ci = dyn_cast<CallInst>(&i))
                if (auto *callee = dyn_cast<Function>(ci->getCalledOperand()->stripPointerCasts()))
                    if (callee->getName().starts_with("__zluda_ptx_impl_"))
                        calls.push_back(ci);
    for (CallInst *ci : calls)
    {
        StringRef n = cast<Function>(ci->getCalledOperand()->stripPointerCasts())->getName().drop_front(strlen("__zluda_ptx_impl_"));
        auto ctrlAt = [&](unsigned k) -> int64_t {
            auto *k_ = k < ci->arg_size() ? dyn_cast<ConstantInt>(ci->getArgOperand(k)) : nullptr;
            return k_ ? (int64_t)k_->getZExtValue() : -1;
        };
        IRBuilder<> b(ci);
        Value *r = nullptr;
        Type *rt = ci->getType();
        if (!rt->isIntegerTy())
            continue;
        if (n == "fsincos_f32" && ctrlAt(1) >= 0)
            r = b.CreateBitCast(fast(b, ctrlAt(1) & 1 ? "cos" : "sin", b.CreateBitCast(ci->getArgOperand(0), f32)), i32);
        else if (n == "frsqrt_f32" && ctrlAt(1) >= 0)
            r = b.CreateBitCast(fast(b, "rsqrt", b.CreateBitCast(ci->getArgOperand(0), f32)), i32);
        else if ((n == "ftanh_32" || n == "ftanh_16" || n == "fex2h_32" || n == "fex2h_16") && ctrlAt(1) >= 0)
        {
            int64_t ctrl = ctrlAt(1);
            unsigned t = (ctrl >> 4) & 15;
            bool pk = ctrl & 256;
            StringRef fn = n.starts_with("ftanh") ? "tanh" : "exp2";
            Value *a = b.CreateZExtOrTrunc(ci->getArgOperand(0), i32);
            if (t == 1 && !pk)
                r = b.CreateBitCast(fn1(b, fn, b.CreateBitCast(a, f32)), i32);
            else if (t == 2 || t == 3)
            {
                r = fromF(b, fn1(b, fn, toF(b, a, t)), t);
                if (pk)
                    r = b.CreateOr(r, b.CreateShl(fromF(b, fn1(b, fn, toF(b, b.CreateLShr(a, 16), t)), t), 16));
            }
            if (r)
                r = b.CreateZExtOrTrunc(r, rt);
        }
        if (!r)
            continue;
        ci->replaceAllUsesWith(r);
        ci->eraseFromParent();
    }
}

static void portableFPModes(Module &m)
{
    for (Function &f : m)
    {
        // the function's entry mode (LLVM folds constants by it; IEEE operations are tagged, and
        // the untagged ones of .ftz regions are the hardware's flushing ones)
        bool ieee = resolveFPModes(f) && !g_fp_ftz;
        f.addFnAttr("denormal-fp-math-f32", ieee ? "ieee,ieee" : "preserve-sign,preserve-sign");

        f.removeFnAttr(Attribute::StrictFP);
        for (Instruction &i : instructions(f))
            if (auto *cb = dyn_cast<CallBase>(&i))
                cb->removeFnAttr(Attribute::StrictFP);
    }
}

static bool usedIn(const Value *v, const Function *f, int depth = 0)
{
    for (const User *u : v->users())
    {
        if (auto *i = dyn_cast<Instruction>(u))
        {
            if (i->getFunction() == f)
                return true;
        }
        else if (isa<Constant>(u) && depth < 8 && usedIn(u, f, depth + 1))
            return true;
    }
    return false;
}

// A threadgroup address computed from an integer of unknown provenance (the shared memory matrix
// descriptors of wgmma / tcgen05, a constant address) may point into any threadgroup variable. LLVM
// only lets such a pointer alias variables whose address is exposed, so the kernel exposes those
// it uses (a volatile store of the address), or their stores would be dead.
static void exposeThreadgroupGlobals(Module &m)
{
    GlobalVariable *sink = nullptr;
    for (Function &f : m)
    {
        if (f.isDeclaration() || f.getCallingConv() != CallingConv::AMDGPU_KERNEL)
            continue;
        bool unknown = false;
        for (Instruction &i : instructions(f))
            if (auto *p = dyn_cast<IntToPtrInst>(&i); p && p->getType()->getPointerAddressSpace() == amd::Shared)
                unknown = true;
        if (!unknown)
            continue;
        IRBuilder<> b(&*f.getEntryBlock().getFirstInsertionPt());
        for (GlobalVariable &gv : m.globals())
        {
            if (gv.getAddressSpace() != amd::Shared || &gv == sink || !usedIn(&gv, &f))
                continue;
            if (auto *at = dyn_cast<ArrayType>(gv.getValueType()); at && at->getNumElements() == 0)
                continue;
            if (!sink)
                sink = new GlobalVariable(m, b.getInt32Ty(), false, GlobalValue::InternalLinkage,
                                          UndefValue::get(b.getInt32Ty()), "zluda_tg_expose", nullptr,
                                          GlobalValue::NotThreadLocal, amd::Shared);
            b.CreateStore(b.CreatePtrToInt(&gv, b.getInt32Ty()), sink, true);
        }
    }
}

// Helpers that need IEEE binary32 operations whatever the kernel's denormal mode call
// __zluda_ieee_{fadd,fsub,fmul,fdiv,sqrt,fma}_f32 (declared in zluda_ptx_metal_impl.cpp): the LLVM
// operation, tagged !agx.keep-denormals for llvm-air2agx (subnormals kept, correctly rounded).
static void lowerIEEEHelperCalls(Module &m)
{
    LLVMContext &c = m.getContext();
    MDNode *keep = MDNode::get(c, {});
    std::vector<CallInst *> calls;
    for (Function &f : m)
        if (f.isDeclaration() && f.getName().starts_with("__zluda_ieee_"))
            for (User *u : f.users())
                if (auto *ci = dyn_cast<CallInst>(u))
                    calls.push_back(ci);
    for (CallInst *ci : calls)
    {
        StringRef n = ci->getCalledFunction()->getName().drop_front(strlen("__zluda_ieee_"));
        IRBuilder<> b(ci);
        Value *a0 = ci->getArgOperand(0);
        Value *r = nullptr;
        if (n == "fadd_f32")
            r = b.CreateFAdd(a0, ci->getArgOperand(1));
        else if (n == "fsub_f32")
            r = b.CreateFSub(a0, ci->getArgOperand(1));
        else if (n == "fmul_f32")
            r = b.CreateFMul(a0, ci->getArgOperand(1));
        else if (n == "fdiv_f32")
            r = b.CreateFDiv(a0, ci->getArgOperand(1));
        else if (n == "sqrt_f32")
            r = b.CreateUnaryIntrinsic(Intrinsic::sqrt, a0);
        else if (n == "fma_f32")
            r = b.CreateIntrinsic(Intrinsic::fma, {a0->getType()}, {a0, ci->getArgOperand(1), ci->getArgOperand(2)});
        else
            fail("unknown IEEE helper operation " + n.str());
        if (g_portable_air && !g_fp_ftz)
            if (auto *ri = dyn_cast<Instruction>(r))
                ri->setMetadata("agx.keep-denormals", keep);
        ci->replaceAllUsesWith(r);
        ci->eraseFromParent();
    }
}

void lowerModuleToAir(Module &m, std::vector<KernelInfo> &kernels, DeviceGlobals &globals)
{
    lowerIEEEHelperCalls(m);
    // a PTX body that falls off its end returns
    for (Function &f : m)
        for (BasicBlock &bb : f)
            if (!bb.getTerminator())
            {
                IRBuilder<> rb(&bb);
                if (f.getReturnType()->isVoidTy())
                    rb.CreateRetVoid();
                else
                    rb.CreateUnreachable();
            }
    // portable AIR: strictfp would keep the helpers from inlining before the modes are resolved
    if (g_portable_air)
        for (Function &f : m)
        {
            f.removeFnAttr(Attribute::StrictFP);
            for (Instruction &i : instructions(f))
                if (auto *cb = dyn_cast<CallBase>(&i))
                    cb->removeFnAttr(Attribute::StrictFP);
        }
    // ---- Phase 1: AMDGPU-numbered cleanup ----
    std::vector<Function *> kernelBodies;
    for (Function &f : m)
    {
        if (f.isDeclaration())
            continue;
        if (f.getCallingConv() == CallingConv::AMDGPU_KERNEL)
        {
            kernelBodies.push_back(&f);
            continue;
        }
        f.setLinkage(GlobalValue::InternalLinkage);
        f.removeFnAttr(Attribute::NoInline);
        f.removeFnAttr(Attribute::OptimizeNone);
        f.addFnAttr(Attribute::AlwaysInline);
    }
    for (Function *k : kernelBodies)
        k->removeFnAttr(Attribute::OptimizeNone);
    // Threadgroup globals: Metal's threadgroup variables are module-scope
    // globals without an initializer value (undef).
    for (GlobalVariable &gv : m.globals())
    {
        if (gv.getAddressSpace() == amd::Shared)
        {
            // PTX .shared variables have no initial value (the helper library's zero
            // initializers are only C++ syntax)
            if (auto *at = dyn_cast<ArrayType>(gv.getValueType()); at && at->getNumElements() == 0)
                continue; // dynamic shared memory, bound as a kernel argument
            gv.setInitializer(UndefValue::get(gv.getValueType()));
            gv.setLinkage(GlobalValue::InternalLinkage);
        }
        else if (!gv.isDeclaration())
        {
            gv.setLinkage(GlobalValue::InternalLinkage);
        }
        gv.setVisibility(GlobalValue::DefaultVisibility);
    }
    hideSourceNaNs(m);
    runCleanup(m, true, true);
    resolveCluster(m);
    // the per-thread stack before PointerRecovery: alloca's integer results keep their local provenance (isspacep)
    resolveLocalStack(m);
    runCleanup(m, false, true);
    for (int iter = 0; iter < 4; iter++)
    {
        // address arithmetic folded into constant expressions (ptrtoint (addrspacecast @shared)
        // + 4) is invisible to PointerRecovery: make it instructions first
        {
            std::vector<Constant *> cs;
            for (GlobalVariable &gv : m.globals())
                if (!gv.use_empty())
                    cs.push_back(&gv);
            convertUsersOfConstantsToInstructions(cs);
        }
        bool changed = false;
        for (Function &f : m)
            if (!f.isDeclaration())
                changed |= PointerRecovery(f).run();
        if (!changed)
            break;
        runCleanup(m, false, true);
    }
    canonicalizeFoldedNaNs(m);
    if (getenv("ZLUDA_AIR_DEBUG_RECOVERED"))
        m.print(errs(), nullptr);
    scanGenericWindows(m);
    resolveIsSpacep(m);
    resolveGenericWindows(m);
    for (Function &f : m)
        if (!f.isDeclaration() && f.getCallingConv() != CallingConv::AMDGPU_KERNEL && !f.use_empty())
            fail("function could not be inlined into its kernel (recursion is not supported): " + f.getName());
    if (getenv("ZLUDA_AIR_DEBUG_RESOLVED"))
        m.print(errs(), nullptr);
    for (Function &f : m)
        for (Instruction &i : instructions(f))
            if (auto *asc = dyn_cast<AddrSpaceCastInst>(&i))
            {
                unsigned s = asc->getSrcAddressSpace(), d = asc->getDestAddressSpace();
                bool sGenericish = s == amd::Flat || s == amd::Global;
                bool dGenericish = d == amd::Flat || d == amd::Global;
                if (!(sGenericish && dGenericish))
                    fail("generic pointer to non-global memory could not be resolved statically in " + f.getName() +
                         ": " + toString(asc));
            }

    exposeThreadgroupGlobals(m);
    if (g_portable_air)
        portableFPModes(m);

    // ---- Phase 2: AIR address spaces ----
    remapAddressSpaces(m);

    // ---- Phase 3: kernel entry points ----
    std::vector<Function *> bodies;
    for (Function &f : m)
        if (f.getCallingConv() == CallingConv::AMDGPU_KERNEL && !f.isDeclaration())
            bodies.push_back(&f);
    globals = collectDeviceGlobals(m);
    for (Function *f : bodies)
        kernels.push_back(buildKernelWrapper(m, f, globals));
    for (DeviceGlobal &g : globals.vars)
    {
        if (!g.gv->use_empty())
            fail("device variable " + g.name + " is used outside of a kernel");
        g.gv->eraseFromParent();
        g.gv = nullptr;
    }

    // ---- Phase 4: instruction / intrinsic lowering ----
    for (Function &f : m)
    {
        SmallVector<Instruction *, 32> work;
        for (Instruction &i : instructions(f))
            work.push_back(&i);
        for (Instruction *i : work)
        {
            if (auto *rmw = dyn_cast<AtomicRMWInst>(i))
                lowerAtomicRMW(rmw);
            else if (auto *cx = dyn_cast<AtomicCmpXchgInst>(i))
                lowerCmpXchg(cx);
            else if (auto *fence = dyn_cast<FenceInst>(i))
            {
                IRBuilder<> b(fence);
                unsigned scope = fence->getSyncScopeID() == SyncScope::SingleThread ? SCOPE_THREADGROUP : SCOPE_DEVICE;
                callAir(b, "air.atomic.fence", b.getVoidTy(),
                        {b.getInt32(3), b.getInt32(MEMORY_ORDER_SEQ_CST), b.getInt32(scope)});
                fence->eraseFromParent();
            }
            else if (auto *ld = dyn_cast<LoadInst>(i); ld && ld->isAtomic())
                lowerAtomicLoadStore(ld);
            else if (auto *st = dyn_cast<StoreInst>(i); st && st->isAtomic())
                lowerAtomicLoadStore(st);
            else if (auto *ci = dyn_cast<CallInst>(i))
                lowerIntrinsic(ci);
        }
    }
    for (Function &f : m)
        if (f.isDeclaration() && !f.use_empty() && !f.getName().starts_with("air.") && !f.isIntrinsic())
            fail("unresolved external function: " + f.getName());
    for (Function &f : m)
        f.setCallingConv(CallingConv::C);
    // Metal GPUs have no double precision arithmetic (portable AIR keeps it: air-emulate-f64)
    if (!getenv("ZLUDA_METAL_ALLOW_F64") && !g_portable_air)
        for (Function &f : m)
            for (Instruction &i : instructions(f))
            {
                bool usesF64 = i.getType()->getScalarType()->isDoubleTy();
                for (Value *op : i.operands())
                    usesF64 |= op->getType()->getScalarType()->isDoubleTy();
                if (usesF64)
                    fail("kernel " + f.getName() + " uses 64-bit floating point, which Metal does not support: " +
                         toString(&i));
            }
    stripTargetAttributes(m);
    // static shared memory size, for reporting
    const DataLayout &dl = m.getDataLayout();
    for (KernelInfo &k : kernels)
        for (GlobalVariable &gv : m.globals())
            if (gv.getAddressSpace() == air::Threadgroup && !isDynamicSharedGlobal(gv))
                for (User *u : gv.users())
                    if (auto *i = dyn_cast<Instruction>(u); i && i->getFunction()->getName() == k.name)
                    {
                        k.staticShared += dl.getTypeAllocSize(gv.getValueType());
                        break;
                    }
    addModuleMetadata(m);
}

std::unique_ptr<Module> parse(LLVMContext &ctx, const void *data, size_t len, StringRef name)
{
    auto buf = MemoryBuffer::getMemBuffer(StringRef((const char *)data, len), name, false);
    auto mod = parseBitcodeFile(buf->getMemBufferRef(), ctx);
    if (!mod)
        fail("cannot parse bitcode " + name + ": " + toString(mod.takeError()));
    return std::move(*mod);
}

std::string writeBitcode(const Module &m)
{
    std::string out;
    raw_string_ostream os(out);
    WriteBitcodeToFile(m, os);
    os.flush();
    return out;
}

// Make a copy of `m` that only keeps kernel `name` (Metal libraries store one
// module per function).
std::unique_ptr<Module> extractKernel(const Module &m, StringRef name)
{
    ValueToValueMapTy vmap;
    std::unique_ptr<Module> copy = CloneModule(m, vmap);
    for (Function &f : *copy)
        if (!f.isDeclaration() && f.getName() != name)
            f.setLinkage(GlobalValue::InternalLinkage);
    NamedMDNode *kernelsMd = copy->getNamedMetadata("air.kernel");
    std::vector<MDNode *> keep;
    for (MDNode *n : kernelsMd->operands())
    {
        auto *fmd = cast<ValueAsMetadata>(n->getOperand(0));
        if (fmd->getValue()->getName() == name)
            keep.push_back(n);
    }
    kernelsMd->clearOperands();
    for (MDNode *n : keep)
        kernelsMd->addOperand(n);
    // drop other kernels
    std::vector<Function *> dead;
    for (Function &f : *copy)
        if (!f.isDeclaration() && f.getName() != name)
            dead.push_back(&f);
    for (Function *f : dead)
    {
        f->replaceAllUsesWith(PoisonValue::get(f->getType()));
        f->eraseFromParent();
    }
    ModuleAnalysisManager mam;
    PassBuilder pb;
    pb.registerModuleAnalyses(mam);
    GlobalDCEPass().run(*copy, mam);
    return copy;
}

char *dupString(const std::string &s)
{
    char *p = (char *)malloc(s.size() + 1);
    memcpy(p, s.data(), s.size());
    p[s.size()] = 0;
    return p;
}

} // namespace

extern "C" int zluda_air_lower(const ZludaAirInput *input, ZludaAirOutput *output, char **error)
{
    *error = nullptr;
    memset(output, 0, sizeof(*output));
    try
    {
        LLVMContext ctx;
        g_portable_air = input->portable_air != 0;
        g_fp_fast = input->fp_fast != 0;
        g_fp_ftz = input->fp_ftz != 0;
        std::unique_ptr<Module> main = parse(ctx, input->main_bc, input->main_len, "main");
        if (getenv("ZLUDA_AIR_DEBUG_INPUT"))
            main->print(errs(), nullptr);
        // PTX registers are allocas of the translator; the formal model reads an unwritten register as 0
        for (Function &f : *main)
            if (!f.isDeclaration())
                for (Instruction &i : f.getEntryBlock())
                    if (auto *ai = dyn_cast<AllocaInst>(&i))
                        if (ai->isStaticAlloca() && ai->getAllocatedType()->isSingleValueType() && ai->getNextNode())
                            new StoreInst(Constant::getNullValue(ai->getAllocatedType()), ai, ai->getNextNode());
        if (input->attributes_bc)
        {
            auto attrs = parse(ctx, input->attributes_bc, input->attributes_len, "attributes");
            if (Linker::linkModules(*main, std::move(attrs), Linker::LinkOnlyNeeded))
                fail("failed to link attributes module");
        }
        g_ptxShared.clear();
        for (GlobalVariable &gv : main->globals())
            if (gv.getAddressSpace() == amd::Shared)
                g_ptxShared.insert(gv.getName().str());
        auto impl = parse(ctx, input->impl_bc, input->impl_len, "zluda_ptx_metal_impl");
        impl->setDataLayout(main->getDataLayout());
        impl->setTargetTriple(main->getTargetTriple());
        if (Linker::linkModules(*main, std::move(impl), Linker::LinkOnlyNeeded))
            fail("failed to link Metal implementation library");
        adaptHelperCalls(*main);
        if (g_fp_fast)
            fastFPHelpers(*main);
        if (getenv("ZLUDA_AIR_DEBUG"))
            main->print(errs(), nullptr);
        std::vector<KernelInfo> kernels;
        DeviceGlobals globals;
        lowerModuleToAir(*main, kernels, globals);
        std::string verifyErrors;
        raw_string_ostream vos(verifyErrors);
        if (verifyModule(*main, &vos))
        {
            std::string ir;
            raw_string_ostream irs(ir);
            main->print(irs, nullptr);
            fail("AIR module failed verification: " + vos.str() + "\n" + irs.str());
        }
        if (input->want_text)
        {
            std::string ir;
            raw_string_ostream irs(ir);
            main->print(irs, nullptr);
            output->module_text = dupString(irs.str());
        }
        output->kernel_count = kernels.size();
        output->kernels = (ZludaAirKernel *)calloc(kernels.size() ? kernels.size() : 1, sizeof(ZludaAirKernel));
        for (size_t i = 0; i < kernels.size(); i++)
        {
            ZludaAirKernel &k = output->kernels[i];
            k.name = dupString(kernels[i].name);
            k.param_count = kernels[i].params.size();
            k.param_offsets = (uint64_t *)calloc(k.param_count + 1, sizeof(uint64_t));
            k.param_sizes = (uint64_t *)calloc(k.param_count + 1, sizeof(uint64_t));
            for (size_t p = 0; p < k.param_count; p++)
            {
                k.param_offsets[p] = kernels[i].params[p].offset;
                k.param_sizes[p] = kernels[i].params[p].size;
            }
            k.params_size = kernels[i].paramsSize;
            k.dynamic_shared = kernels[i].dynamicShared;
            k.static_shared = kernels[i].staticShared;
            k.globals_param_offset = kernels[i].globalsParamOffset;
            k.dyn_smem_param_offset = kernels[i].dynSmemParamOffset;
            std::string bc = writeBitcode(*extractKernel(*main, kernels[i].name));
            k.air_len = bc.size();
            k.air = (uint8_t *)malloc(bc.size());
            memcpy(k.air, bc.data(), bc.size());
        }
        output->global_count = globals.vars.size();
        output->globals = (ZludaAirGlobal *)calloc(globals.vars.size() + 1, sizeof(ZludaAirGlobal));
        for (size_t i = 0; i < globals.vars.size(); i++)
            output->globals[i] = {dupString(globals.vars[i].name), globals.vars[i].offset, globals.vars[i].size};
        output->globals_size = globals.init.size();
        output->globals_init = (uint8_t *)malloc(globals.init.size() + 1);
        memcpy(output->globals_init, globals.init.data(), globals.init.size());
        return 0;
    }
    catch (const std::exception &e)
    {
        *error = dupString(e.what());
        return 1;
    }
}

extern "C" void zluda_air_free(ZludaAirOutput *output)
{
    for (size_t i = 0; i < output->kernel_count; i++)
    {
        free(output->kernels[i].name);
        free(output->kernels[i].param_offsets);
        free(output->kernels[i].param_sizes);
        free(output->kernels[i].air);
    }
    free(output->kernels);
    for (size_t i = 0; i < output->global_count; i++)
        free(output->globals[i].name);
    free(output->globals);
    free(output->globals_init);
    free(output->module_text);
    memset(output, 0, sizeof(*output));
}

extern "C" void zluda_air_free_string(char *s) { free(s); }
