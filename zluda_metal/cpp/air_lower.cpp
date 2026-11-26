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
class PointerRecovery
{
public:
    explicit PointerRecovery(Function &f) : f(f), dl(f.getParent()->getDataLayout()) {}

    bool run()
    {
        bool changed = false;
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
            return pti->getPointerAddressSpace();
        if (!visiting.insert(v).second)
            return ~1u; // cycle: neutral element
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
            result = pti->getOperand(0);
        }
        else if (auto *ce = dyn_cast<ConstantExpr>(v))
        {
            if (ce->getOpcode() == Instruction::PtrToInt)
                result = ce->getOperand(0);
            else if (ce->getOpcode() == Instruction::Add)
            {
                for (unsigned side = 0; side < 2 && !result; side++)
                    if (auto *base = dyn_cast_or_null<Constant>(recover(ce->getOperand(side))))
                        result = ConstantExpr::getGetElementPtr(Type::getInt8Ty(f.getContext()), base,
                                                                ce->getOperand(1 - side));
            }
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

Value *emitCmpXchgStrong(IRBuilder<> &b, Value *ptr, Value *cmp, Value *desired, Value **successOut)
{
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
    BasicBlock *rest = pre->splitBasicBlock(b.GetInsertPoint(), "cmpxchg.end");
    pre->getTerminator()->eraseFromParent();
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
    Value *observed = emitCmpXchgStrong(b, ptr, old, newVal, &success);
    BasicBlock *loopEnd = b.GetInsertBlock();
    BasicBlock *done = loopEnd->splitBasicBlock(b.GetInsertPoint(), "atomic.done");
    loopEnd->getTerminator()->eraseFromParent();
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
                return lb.CreateFAdd(old, val);
            case AtomicRMWInst::FSub:
                return lb.CreateFSub(old, val);
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
    case Intrinsic::minimum:
        return airMath("fmin", 2);
    case Intrinsic::maxnum:
    case Intrinsic::maximum:
        return airMath("fmax", 2);
    case Intrinsic::copysign:
        return airMath("copysign", 2);
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
        return replace(callAir(b, "air.add_sat." + intSuffix(true), rt, {arg(0), arg(1)}));
    case Intrinsic::uadd_sat:
        return replace(callAir(b, "air.add_sat." + intSuffix(false), rt, {arg(0), arg(1)}));
    case Intrinsic::ssub_sat:
        return replace(callAir(b, "air.sub_sat." + intSuffix(true), rt, {arg(0), arg(1)}));
    case Intrinsic::usub_sat:
        return replace(callAir(b, "air.sub_sat." + intSuffix(false), rt, {arg(0), arg(1)}));
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
    if (name.starts_with("llvm.minimumnum."))
        return airMath("fmin", 2);
    if (name.starts_with("llvm.maximumnum."))
        return airMath("fmax", 2);
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
};

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

    Type *i32 = Type::getInt32Ty(c);
    Type *v3 = FixedVectorType::get(i32, 3);
    PointerType *constPtr = PointerType::get(c, air::Constant);
    PointerType *tgPtr = PointerType::get(c, air::Threadgroup);
    std::vector<Type *> wrapperParams = {constPtr, v3, v3, v3, v3, i32};
    if (dynShared)
        wrapperParams.push_back(tgPtr);
    Function *w = Function::Create(FunctionType::get(Type::getVoidTy(c), wrapperParams, false),
                                   GlobalValue::ExternalLinkage, info.name, &m);
    w->addFnAttr(Attribute::NoUnwind);
    w->addFnAttr("frame-pointer", "none");
    w->addParamAttr(0, Attribute::NoUndef);
    const char *argNames[] = {"zluda_params", "tid", "ctaid", "ntid", "nctaid", "laneid", "zluda_dynamic_shared"};
    for (unsigned i = 0; i < w->arg_size(); i++)
        w->getArg(i)->setName(argNames[i]);

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
            if (a.isStringAttribute())
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

void lowerModuleToAir(Module &m, std::vector<KernelInfo> &kernels, DeviceGlobals &globals)
{
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
        if (gv.getAddressSpace() == amd::Shared && !gv.hasInitializer())
        {
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
    runCleanup(m, true, true);
    for (int iter = 0; iter < 4; iter++)
    {
        bool changed = false;
        for (Function &f : m)
            if (!f.isDeclaration())
                changed |= PointerRecovery(f).run();
        if (!changed)
            break;
        runCleanup(m, false, true);
    }
    for (Function &f : m)
        if (!f.isDeclaration() && f.getCallingConv() != CallingConv::AMDGPU_KERNEL && !f.use_empty())
            fail("function could not be inlined into its kernel (recursion is not supported): " + f.getName());
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
    // Metal GPUs have no double precision arithmetic
    if (!getenv("ZLUDA_METAL_ALLOW_F64"))
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
        std::unique_ptr<Module> main = parse(ctx, input->main_bc, input->main_len, "main");
        if (input->attributes_bc)
        {
            auto attrs = parse(ctx, input->attributes_bc, input->attributes_len, "attributes");
            if (Linker::linkModules(*main, std::move(attrs), Linker::LinkOnlyNeeded))
                fail("failed to link attributes module");
        }
        auto impl = parse(ctx, input->impl_bc, input->impl_len, "zluda_ptx_metal_impl");
        impl->setDataLayout(main->getDataLayout());
        impl->setTargetTriple(main->getTargetTriple());
        if (Linker::linkModules(*main, std::move(impl), Linker::LinkOnlyNeeded))
            fail("failed to link Metal implementation library");
        adaptHelperCalls(*main);
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
