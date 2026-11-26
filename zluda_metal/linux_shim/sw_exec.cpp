// Software execution of Metal AIR kernels (Linux test device).
//
// This executor reads AIR the way a Metal driver would: it takes the kernel
// signature from the !air.kernel argument metadata (buffers by location
// index, threadgroup buffers, thread position builtins) and gives the air.*
// intrinsics their Metal Shading Language meaning. It shares no code with the
// AIR producer (cpp/air_lower.cpp), so running ZLUDA's output through it is a
// real consistency check of the generated AIR.
//
// Execution model: the threads of one threadgroup run as OS threads, so
// threadgroup barriers and SIMD-group operations (shuffle, ballot) behave as on
// a GPU with 32-wide SIMD groups. Threadgroups of a dispatch run one after the
// other.

#include "sw_exec.h"

#include "../cpp/metallib.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ReplaceConstant.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <thread>

using namespace llvm;

namespace zsw
{

// ---------------------------------------------------------------------------
// Synchronization primitives for one threadgroup
// ---------------------------------------------------------------------------
namespace
{

// Barrier whose participant count shrinks when threads exit the kernel
struct DynBarrier
{
    std::mutex m;
    std::condition_variable cv;
    unsigned expected = 0;
    unsigned arrived = 0;
    uint64_t generation = 0;

    void reset(unsigned n)
    {
        std::lock_guard<std::mutex> l(m);
        expected = n;
        arrived = 0;
    }
    void releaseLocked()
    {
        arrived = 0;
        generation++;
        cv.notify_all();
    }
    void arriveAndWait()
    {
        std::unique_lock<std::mutex> l(m);
        uint64_t gen = generation;
        if (++arrived >= expected)
        {
            releaseLocked();
            return;
        }
        cv.wait(l, [&] { return generation != gen; });
    }
    void drop()
    {
        std::lock_guard<std::mutex> l(m);
        expected--;
        if (expected > 0 && arrived >= expected)
            releaseLocked();
    }
};

constexpr unsigned SIMD_WIDTH = 32;

struct SimdGroup
{
    DynBarrier barrier;
    uint32_t values[SIMD_WIDTH];
    uint32_t predicates[SIMD_WIDTH];
    bool active[SIMD_WIDTH];
    unsigned size = 0;
};

struct GroupState
{
    DynBarrier tgBarrier;
    std::vector<std::unique_ptr<SimdGroup>> simds;
};

struct ThreadContext
{
    GroupState *group;
    SimdGroup *simd;
    unsigned lane;
};

thread_local ThreadContext t_ctx;

} // namespace
} // namespace zsw

using namespace zsw;

// Host functions the JIT-compiled AIR calls into
extern "C"
{
    void zsw_tg_barrier() { t_ctx.group->tgBarrier.arriveAndWait(); }
    void zsw_simd_barrier() { t_ctx.simd->barrier.arriveAndWait(); }
    uint32_t zsw_simd_shuffle(uint32_t value, uint32_t srcLane)
    {
        SimdGroup *s = t_ctx.simd;
        s->values[t_ctx.lane] = value;
        s->barrier.arriveAndWait();
        uint32_t r = (srcLane < s->size && s->active[srcLane]) ? s->values[srcLane] : value;
        s->barrier.arriveAndWait();
        return r;
    }
    uint64_t zsw_simd_ballot(uint32_t predicate)
    {
        SimdGroup *s = t_ctx.simd;
        s->predicates[t_ctx.lane] = predicate;
        s->barrier.arriveAndWait();
        uint64_t mask = 0;
        for (unsigned i = 0; i < s->size; i++)
            if (s->active[i] && s->predicates[i])
                mask |= uint64_t(1) << i;
        s->barrier.arriveAndWait();
        return mask;
    }
    float zsw_tanhf(float x) { return std::tanh(x); }
    double zsw_tanh(double x) { return std::tanh(x); }
}

namespace zsw
{

// ---------------------------------------------------------------------------
// metallib parsing
// ---------------------------------------------------------------------------
bool parseLibrary(const void *data, size_t len, std::vector<LibraryFunction> &out, std::string &error)
{
    ZludaMetallibEntry *entries = nullptr;
    size_t count = 0;
    char *err = nullptr;
    if (zluda_metallib_read((const uint8_t *)data, len, &entries, &count, &err) != 0)
    {
        error = err ? err : "metallib parse error";
        free(err);
        return false;
    }
    for (size_t i = 0; i < count; i++)
        out.push_back({entries[i].name, std::vector<uint8_t>(entries[i].air, entries[i].air + entries[i].air_len)});
    zluda_metallib_free_entries(entries, count);
    return true;
}

namespace
{

[[noreturn]] void fail(const Twine &msg) { throw std::runtime_error(msg.str()); }

// --- address spaces -> host (everything is a plain pointer on the CPU) ---
Type *flatType(Type *t)
{
    if (auto *pt = dyn_cast<PointerType>(t))
        return pt->getAddressSpace() == 0 ? t : PointerType::get(t->getContext(), 0);
    if (auto *vt = dyn_cast<VectorType>(t))
        return VectorType::get(flatType(vt->getElementType()), vt->getElementCount());
    if (auto *at = dyn_cast<ArrayType>(t))
        return ArrayType::get(flatType(at->getElementType()), at->getNumElements());
    if (auto *st = dyn_cast<StructType>(t))
    {
        if (!st->isLiteral())
            return t;
        SmallVector<Type *, 8> e;
        for (Type *x : st->elements())
            e.push_back(flatType(x));
        return StructType::get(t->getContext(), e, st->isPacked());
    }
    if (auto *ft = dyn_cast<FunctionType>(t))
    {
        SmallVector<Type *, 8> p;
        for (Type *x : ft->params())
            p.push_back(flatType(x));
        return FunctionType::get(flatType(ft->getReturnType()), p, ft->isVarArg());
    }
    return t;
}

void flattenAddressSpaces(Module &m)
{
    LLVMContext &c = m.getContext();
    {
        SmallVector<Constant *, 16> consts;
        for (GlobalVariable &gv : m.globals())
            consts.push_back(&gv);
        for (Function &f : m)
            consts.push_back(&f);
        convertUsersOfConstantsToInstructions(consts);
    }
    std::vector<Function *> fns;
    for (Function &f : m)
        fns.push_back(&f);
    std::map<Function *, Function *> repl;
    for (Function *f : fns)
    {
        auto *nft = cast<FunctionType>(flatType(f->getFunctionType()));
        if (nft == f->getFunctionType())
            continue;
        Function *nf = Function::Create(nft, f->getLinkage(), 0, "", &m);
        nf->copyAttributesFrom(f);
        nf->takeName(f);
        nf->splice(nf->end(), f);
        for (unsigned i = 0; i < f->arg_size(); i++)
        {
            nf->getArg(i)->takeName(f->getArg(i));
            f->getArg(i)->mutateType(nf->getArg(i)->getType());
            f->getArg(i)->replaceAllUsesWith(nf->getArg(i));
        }
        repl[f] = nf;
    }
    for (GlobalVariable &gv : m.globals())
        gv.mutateType(PointerType::get(c, 0));
    for (Function &f : m)
        for (Instruction &i : instructions(f))
        {
            for (Use &u : i.operands())
            {
                auto *k = dyn_cast<Constant>(u.get());
                if (!k)
                    continue;
                if (auto *of = dyn_cast<Function>(k))
                {
                    if (repl.count(of))
                        u.set(repl[of]);
                    continue;
                }
                if (isa<GlobalValue>(k))
                    continue;
                Type *nt = flatType(k->getType());
                if (nt == k->getType())
                    continue;
                if (isa<PoisonValue>(k))
                    u.set(PoisonValue::get(nt));
                else if (isa<UndefValue>(k))
                    u.set(UndefValue::get(nt));
                else if (k->isNullValue())
                    u.set(Constant::getNullValue(nt));
                else
                    fail("unsupported pointer constant in AIR");
            }
            i.mutateType(flatType(i.getType()));
            if (auto *cb = dyn_cast<CallBase>(&i))
                cb->mutateFunctionType(cast<FunctionType>(flatType(cb->getFunctionType())));
            if (auto *ai = dyn_cast<AllocaInst>(&i))
                ai->setAllocatedType(flatType(ai->getAllocatedType()));
        }
    for (auto &[of, nf] : repl)
    {
        if (!of->use_empty())
        {
            of->mutateType(nf->getType());
            of->replaceAllUsesWith(nf);
        }
        of->eraseFromParent();
    }
    for (Function &f : m)
    {
        SmallVector<Instruction *, 8> dead;
        for (Instruction &i : instructions(f))
            if (auto *asc = dyn_cast<AddrSpaceCastInst>(&i))
            {
                asc->replaceAllUsesWith(asc->getOperand(0));
                dead.push_back(asc);
            }
        for (auto *i : dead)
            i->eraseFromParent();
    }
    std::vector<Function *> intr;
    for (Function &f : m)
        if (f.isIntrinsic())
            intr.push_back(&f);
    for (Function *f : intr)
        if (auto nf = Intrinsic::remangleIntrinsicFunction(f))
        {
            f->replaceAllUsesWith(*nf);
            f->eraseFromParent();
        }
}

SmallVector<StringRef, 8> splitName(StringRef name)
{
    SmallVector<StringRef, 8> parts;
    name.split(parts, '.');
    return parts;
}

// Gives an air.* declaration a host implementation
void defineAirFunction(Function &f)
{
    Module &m = *f.getParent();
    LLVMContext &c = m.getContext();
    StringRef name = f.getName();
    auto parts = splitName(name);
    BasicBlock *bb = BasicBlock::Create(c, "entry", &f);
    IRBuilder<> b(bb);
    Type *rt = f.getReturnType();
    auto arg = [&](unsigned i) -> Value * { return f.getArg(i); };
    auto host = [&](const char *fname, Type *ret, ArrayRef<Type *> params) {
        return m.getOrInsertFunction(fname, FunctionType::get(ret, params, false));
    };
    auto ret = [&](Value *v) {
        if (rt->isVoidTy())
            b.CreateRetVoid();
        else
            b.CreateRet(v);
    };
    auto unary = [&](Intrinsic::ID id) { ret(b.CreateUnaryIntrinsic(id, arg(0))); };
    auto binary = [&](Intrinsic::ID id) { ret(b.CreateBinaryIntrinsic(id, arg(0), arg(1))); };
    f.setLinkage(GlobalValue::InternalLinkage);
    f.addFnAttr(Attribute::AlwaysInline);

    if (name == "air.wg.barrier")
    {
        b.CreateCall(host("zsw_tg_barrier", b.getVoidTy(), {}));
        return ret(nullptr);
    }
    if (name == "air.simdgroup.barrier")
    {
        b.CreateCall(host("zsw_simd_barrier", b.getVoidTy(), {}));
        return ret(nullptr);
    }
    if (name.starts_with("air.simd_shuffle."))
    {
        // simd_shuffle(value, ushort lane)
        Type *t = rt;
        unsigned bits = t->getPrimitiveSizeInBits();
        if (bits > 32)
            fail("unsupported simd_shuffle type in " + name);
        Value *v = b.CreateBitCast(arg(0), b.getIntNTy(bits));
        v = b.CreateZExt(v, b.getInt32Ty());
        Value *lane = b.CreateZExt(arg(1), b.getInt32Ty());
        Value *r = b.CreateCall(host("zsw_simd_shuffle", b.getInt32Ty(), {b.getInt32Ty(), b.getInt32Ty()}), {v, lane});
        r = b.CreateTrunc(r, b.getIntNTy(bits));
        return ret(b.CreateBitCast(r, t));
    }
    if (name.starts_with("air.simd_ballot."))
    {
        Value *p = b.CreateZExt(arg(0), b.getInt32Ty());
        Value *r = b.CreateCall(host("zsw_simd_ballot", b.getInt64Ty(), {b.getInt32Ty()}), {p});
        return ret(b.CreateZExtOrTrunc(r, rt));
    }
    if (name == "air.atomic.fence")
    {
        b.CreateFence(AtomicOrdering::SequentiallyConsistent);
        return ret(nullptr);
    }
    if (name.starts_with("air.atomic.") && parts.size() >= 4)
    {
        // air.atomic.<global|local>.<op>[.<s|u>|.weak].<type>
        StringRef op = parts[3];
        bool isSigned = parts.size() > 5 && parts[4] == "s";
        Value *ptr = arg(0);
        if (op == "load")
        {
            LoadInst *l = b.CreateLoad(rt, ptr);
            l->setAtomic(AtomicOrdering::Monotonic);
            l->setAlignment(Align(rt->getPrimitiveSizeInBits() / 8));
            return ret(l);
        }
        if (op == "store")
        {
            Type *vt = arg(1)->getType();
            StoreInst *s = b.CreateStore(arg(1), ptr);
            s->setAtomic(AtomicOrdering::Monotonic);
            s->setAlignment(Align(vt->getPrimitiveSizeInBits() / 8));
            return ret(nullptr);
        }
        if (op == "cmpxchg")
        {
            // (ptr object, ptr expected, T desired, order_s, order_f, scope, volatile) -> i1
            Type *t = arg(2)->getType();
            Value *expected = b.CreateLoad(t, arg(1));
            auto *cx = b.CreateAtomicCmpXchg(ptr, expected, arg(2), MaybeAlign(), AtomicOrdering::Monotonic,
                                             AtomicOrdering::Monotonic);
            b.CreateStore(b.CreateExtractValue(cx, 0), arg(1));
            return ret(b.CreateExtractValue(cx, 1));
        }
        AtomicRMWInst::BinOp bop;
        bool fp = rt->isFloatingPointTy();
        if (op == "add")
            bop = fp ? AtomicRMWInst::FAdd : AtomicRMWInst::Add;
        else if (op == "sub")
            bop = fp ? AtomicRMWInst::FSub : AtomicRMWInst::Sub;
        else if (op == "and")
            bop = AtomicRMWInst::And;
        else if (op == "or")
            bop = AtomicRMWInst::Or;
        else if (op == "xor")
            bop = AtomicRMWInst::Xor;
        else if (op == "xchg")
            bop = AtomicRMWInst::Xchg;
        else if (op == "min")
            bop = fp ? AtomicRMWInst::FMin : (isSigned ? AtomicRMWInst::Min : AtomicRMWInst::UMin);
        else if (op == "max")
            bop = fp ? AtomicRMWInst::FMax : (isSigned ? AtomicRMWInst::Max : AtomicRMWInst::UMax);
        else
            fail("unsupported AIR atomic: " + name);
        return ret(b.CreateAtomicRMW(bop, ptr, arg(1), MaybeAlign(), AtomicOrdering::Monotonic));
    }
    if (parts.size() >= 3)
    {
        StringRef base = parts[1];
        bool isSigned = parts.size() >= 4 && parts[2] == "s";
        if (base == "fma")
            return ret(b.CreateIntrinsic(Intrinsic::fma, {rt}, {arg(0), arg(1), arg(2)}));
        if (base == "fabs")
            return unary(Intrinsic::fabs);
        if (base == "sqrt" || base == "fast_sqrt")
            return unary(Intrinsic::sqrt);
        if (base == "rsqrt" || base == "fast_rsqrt")
            return ret(b.CreateFDiv(ConstantFP::get(rt, 1.0), b.CreateUnaryIntrinsic(Intrinsic::sqrt, arg(0))));
        if (base == "exp2" || base == "fast_exp2")
            return unary(Intrinsic::exp2);
        if (base == "log2" || base == "fast_log2")
            return unary(Intrinsic::log2);
        if (base == "exp" || base == "fast_exp")
            return unary(Intrinsic::exp);
        if (base == "log" || base == "fast_log")
            return unary(Intrinsic::log);
        if (base == "log10" || base == "fast_log10")
            return unary(Intrinsic::log10);
        if (base == "sin" || base == "fast_sin")
            return unary(Intrinsic::sin);
        if (base == "cos" || base == "fast_cos")
            return unary(Intrinsic::cos);
        if (base == "pow" || base == "fast_pow")
            return binary(Intrinsic::pow);
        if (base == "floor")
            return unary(Intrinsic::floor);
        if (base == "ceil")
            return unary(Intrinsic::ceil);
        if (base == "trunc")
            return unary(Intrinsic::trunc);
        if (base == "rint")
            return unary(Intrinsic::rint);
        if (base == "round")
            return unary(Intrinsic::round);
        if (base == "fmin" || base == "fast_fmin")
            return binary(Intrinsic::minnum);
        if (base == "fmax" || base == "fast_fmax")
            return binary(Intrinsic::maxnum);
        if (base == "copysign")
            return binary(Intrinsic::copysign);
        if ((base == "tanh" || base == "fast_tanh") && rt->isFloatingPointTy())
        {
            Value *x = b.CreateFPCast(arg(0), b.getDoubleTy());
            Value *r = b.CreateCall(host("zsw_tanh", b.getDoubleTy(), {b.getDoubleTy()}), {x});
            return ret(b.CreateFPCast(r, rt));
        }
        if (base == "popcount")
            return unary(Intrinsic::ctpop);
        if (base == "clz")
            return ret(b.CreateBinaryIntrinsic(Intrinsic::ctlz, arg(0), b.getFalse()));
        if (base == "ctz")
            return ret(b.CreateBinaryIntrinsic(Intrinsic::cttz, arg(0), b.getFalse()));
        if (base == "reverse_bits")
            return unary(Intrinsic::bitreverse);
        if (base == "min")
            return binary(isSigned ? Intrinsic::smin : Intrinsic::umin);
        if (base == "max")
            return binary(isSigned ? Intrinsic::smax : Intrinsic::umax);
        if (base == "abs")
            return ret(b.CreateBinaryIntrinsic(Intrinsic::abs, arg(0), b.getFalse()));
        if (base == "add_sat")
            return binary(isSigned ? Intrinsic::sadd_sat : Intrinsic::uadd_sat);
        if (base == "sub_sat")
            return binary(isSigned ? Intrinsic::ssub_sat : Intrinsic::usub_sat);
    }
    fail("the software Metal device does not implement AIR intrinsic " + name);
}

// Builtin slots in the per-thread array passed to the entry trampoline
enum BuiltinSlot : unsigned
{
    TID = 0,         // thread_position_in_threadgroup (3)
    CTAID = 3,       // threadgroup_position_in_grid (3)
    NTID = 6,        // threads_per_threadgroup (3)
    NCTAID = 9,      // threadgroups_per_grid (3)
    GID = 12,        // thread_position_in_grid (3)
    GRID = 15,       // threads_per_grid (3)
    LANE = 18,       // thread_index_in_simdgroup
    SIMD_INDEX = 19, // simdgroup_index_in_threadgroup
    FLAT_TID = 20,   // thread_index_in_threadgroup
    SIMD_WIDTH_SLOT = 21,
    SIMD_COUNT = 22,
    BUILTIN_SLOTS = 23
};

struct ArgBinding
{
    enum Kind
    {
        Buffer,
        Threadgroup,
        Builtin
    } kind;
    unsigned index; // location index or builtin slot
};

std::string mdString(const MDOperand &op)
{
    if (auto *s = dyn_cast<MDString>(op.get()))
        return s->getString().str();
    return "";
}

} // namespace

struct CompiledKernel
{
    std::string name;
    void (*entry)(void **bindings, const uint32_t *builtins) = nullptr;
    uint64_t staticThreadgroup = 0;
};

namespace
{
std::mutex g_jitMutex;
std::unique_ptr<orc::LLJIT> g_jit;
unsigned g_kernelCounter = 0;

orc::LLJIT &jit()
{
    if (!g_jit)
    {
        InitializeNativeTarget();
        InitializeNativeTargetAsmPrinter();
        auto j = orc::LLJITBuilder().create();
        if (!j)
            fail("cannot create LLJIT: " + toString(j.takeError()));
        g_jit = std::move(*j);
    }
    return *g_jit;
}
} // namespace

std::shared_ptr<CompiledKernel> compileKernel(const LibraryFunction &fn, std::string &error)
{
    try
    {
        std::lock_guard<std::mutex> lock(g_jitMutex);
        orc::LLJIT &J = jit();
        auto tsc = std::make_unique<LLVMContext>();
        LLVMContext &c = *tsc;
        auto buf = MemoryBuffer::getMemBuffer(StringRef((const char *)fn.air.data(), fn.air.size()), fn.name, false);
        auto parsed = parseBitcodeFile(buf->getMemBufferRef(), c);
        if (!parsed)
            fail("invalid AIR bitcode: " + toString(parsed.takeError()));
        std::unique_ptr<Module> m = std::move(*parsed);
        if (!StringRef(m->getTargetTriple()).starts_with("air64"))
            fail("not an AIR module (triple " + m->getTargetTriple() + ")");

        // Kernel signature from !air.kernel
        NamedMDNode *kernels = m->getNamedMetadata("air.kernel");
        if (!kernels)
            fail("module has no !air.kernel metadata");
        Function *kernel = nullptr;
        MDNode *argList = nullptr;
        for (MDNode *k : kernels->operands())
        {
            auto *fmd = dyn_cast<ValueAsMetadata>(k->getOperand(0).get());
            if (fmd && fmd->getValue()->getName() == fn.name)
            {
                kernel = cast<Function>(fmd->getValue());
                argList = cast<MDNode>(k->getOperand(2).get());
            }
        }
        if (!kernel)
            fail("kernel " + fn.name + " not found in !air.kernel");
        std::vector<ArgBinding> bindings(kernel->arg_size(), ArgBinding{ArgBinding::Builtin, ~0u});
        static const std::map<std::string, unsigned> builtinSlots = {
            {"air.thread_position_in_threadgroup", TID},
            {"air.threadgroup_position_in_grid", CTAID},
            {"air.threads_per_threadgroup", NTID},
            {"air.threadgroups_per_grid", NCTAID},
            {"air.thread_position_in_grid", GID},
            {"air.threads_per_grid", GRID},
            {"air.thread_index_in_simdgroup", LANE},
            {"air.simdgroup_index_in_threadgroup", SIMD_INDEX},
            {"air.thread_index_in_threadgroup", FLAT_TID},
            {"air.threads_per_simdgroup", SIMD_WIDTH_SLOT},
            {"air.simdgroups_per_threadgroup", SIMD_COUNT},
        };
        for (const MDOperand &op : argList->operands())
        {
            auto *a = cast<MDNode>(op.get());
            unsigned idx = mdconst::extract<ConstantInt>(a->getOperand(0))->getZExtValue();
            std::string kind = mdString(a->getOperand(1));
            if (idx >= bindings.size())
                fail("argument metadata index out of range");
            if (kind == "air.buffer" || kind == "air.threadgroup")
            {
                if (mdString(a->getOperand(2)) != "air.location_index")
                    fail("buffer argument without air.location_index");
                unsigned loc = mdconst::extract<ConstantInt>(a->getOperand(3))->getZExtValue();
                if (loc >= MAX_BUFFERS)
                    fail("buffer location index out of range");
                bindings[idx] = {kind == "air.buffer" ? ArgBinding::Buffer : ArgBinding::Threadgroup, loc};
            }
            else if (builtinSlots.count(kind))
                bindings[idx] = {ArgBinding::Builtin, builtinSlots.at(kind)};
            else
                fail("unsupported kernel argument kind " + kind);
        }
        for (auto &bnd : bindings)
            if (bnd.kind == ArgBinding::Builtin && bnd.index == ~0u)
                fail("kernel argument without metadata");

        auto result = std::make_shared<CompiledKernel>();
        result->name = fn.name;
        for (GlobalVariable &gv : m->globals())
            if (gv.getAddressSpace() == 3)
            {
                result->staticThreadgroup += m->getDataLayout().getTypeAllocSize(gv.getValueType());
                if (!gv.hasInitializer() || isa<UndefValue>(gv.getInitializer()))
                    gv.setInitializer(Constant::getNullValue(gv.getValueType()));
            }

        // Host retargeting
        auto jtmb = orc::JITTargetMachineBuilder::detectHost();
        if (!jtmb)
            fail(toString(jtmb.takeError()));
        auto hostDL = jtmb->getDefaultDataLayoutForTarget();
        if (!hostDL)
            fail(toString(hostDL.takeError()));
        flattenAddressSpaces(*m);
        kernel = m->getFunction(fn.name);
        m->setDataLayout(*hostDL);
        m->setTargetTriple(jtmb->getTargetTriple().str());
        std::vector<Function *> airDecls;
        for (Function &f : *m)
            if (f.isDeclaration() && f.getName().starts_with("air."))
                airDecls.push_back(&f);
        for (Function *f : airDecls)
            defineAirFunction(*f);
        for (Function &f : *m)
            if (f.isDeclaration() && !f.isIntrinsic() && !f.use_empty() && !f.getName().starts_with("zsw_"))
                fail("AIR module references unknown external function " + f.getName());

        // Entry trampoline: void entry(void **bindings, const uint32_t *builtins)
        unsigned id = g_kernelCounter++;
        std::string entryName = "zsw_entry_" + std::to_string(id);
        PointerType *ptrTy = PointerType::get(c, 0);
        Function *entry = Function::Create(FunctionType::get(Type::getVoidTy(c), {ptrTy, ptrTy}, false),
                                           GlobalValue::ExternalLinkage, entryName, m.get());
        IRBuilder<> b(BasicBlock::Create(c, "entry", entry));
        SmallVector<Value *, 8> args;
        for (unsigned i = 0; i < kernel->arg_size(); i++)
        {
            Type *t = kernel->getArg(i)->getType();
            const ArgBinding &bnd = bindings[i];
            if (bnd.kind != ArgBinding::Builtin)
            {
                unsigned slot = bnd.index + (bnd.kind == ArgBinding::Threadgroup ? MAX_BUFFERS : 0);
                Value *p = b.CreateLoad(ptrTy, b.CreateConstGEP1_32(ptrTy, entry->getArg(0), slot));
                args.push_back(p);
                continue;
            }
            auto load = [&](unsigned slot) {
                return b.CreateLoad(b.getInt32Ty(), b.CreateConstGEP1_32(b.getInt32Ty(), entry->getArg(1), slot));
            };
            Type *elem = t->isVectorTy() ? cast<VectorType>(t)->getElementType() : t;
            if (!elem->isIntegerTy())
                fail("unsupported builtin argument type");
            if (auto *vt = dyn_cast<FixedVectorType>(t))
            {
                Value *v = PoisonValue::get(vt);
                for (unsigned e = 0; e < vt->getNumElements(); e++)
                    v = b.CreateInsertElement(v, b.CreateZExtOrTrunc(load(bnd.index + e), elem), e);
                args.push_back(v);
            }
            else
                args.push_back(b.CreateZExtOrTrunc(load(bnd.index), t));
        }
        b.CreateCall(kernel, args);
        b.CreateRetVoid();
        kernel->setLinkage(GlobalValue::InternalLinkage);

        std::string verr;
        raw_string_ostream vos(verr);
        if (verifyModule(*m, &vos))
            fail("host module failed verification: " + vos.str());
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
            ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(OptimizationLevel::O2);
            mpm.run(*m, mam);
        }

        orc::JITDylib &jd = J.getExecutionSession().createBareJITDylib("zsw_kernel_" + std::to_string(id));
        orc::SymbolMap host;
        auto def = [&](const char *n, void *p) {
            host[J.mangleAndIntern(n)] = {orc::ExecutorAddr::fromPtr(p), JITSymbolFlags::Exported};
        };
        def("zsw_tg_barrier", (void *)&zsw_tg_barrier);
        def("zsw_simd_barrier", (void *)&zsw_simd_barrier);
        def("zsw_simd_shuffle", (void *)&zsw_simd_shuffle);
        def("zsw_simd_ballot", (void *)&zsw_simd_ballot);
        def("zsw_tanhf", (void *)&zsw_tanhf);
        def("zsw_tanh", (void *)&zsw_tanh);
        if (auto e = jd.define(orc::absoluteSymbols(std::move(host))))
            fail(toString(std::move(e)));
        auto gen = orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(J.getDataLayout().getGlobalPrefix());
        if (!gen)
            fail(toString(gen.takeError()));
        jd.addGenerator(std::move(*gen));
        if (auto e = J.addIRModule(jd, orc::ThreadSafeModule(std::move(m), std::move(tsc))))
            fail(toString(std::move(e)));
        auto sym = J.lookup(jd, entryName);
        if (!sym)
            fail(toString(sym.takeError()));
        result->entry = sym->toPtr<void (*)(void **, const uint32_t *)>();
        return result;
    }
    catch (const std::exception &e)
    {
        error = e.what();
        return nullptr;
    }
}

uint64_t staticThreadgroupMemory(const CompiledKernel &k) { return k.staticThreadgroup; }

bool dispatch(CompiledKernel &k, const DispatchArgs &args, std::string &error)
{
    const uint32_t *tg = args.threadsPerThreadgroup;
    const uint32_t *grid = args.threadgroups;
    uint64_t threads = uint64_t(tg[0]) * tg[1] * tg[2];
    if (threads == 0 || threads > 1024)
    {
        error = "invalid threadgroup size";
        return false;
    }
    uint64_t groups = uint64_t(grid[0]) * grid[1] * grid[2];
    if (groups == 0)
        return true;
    // threadgroup memory buffers (shared by the threads of the group in flight)
    std::vector<std::vector<uint8_t>> tgMemory(MAX_BUFFERS);
    void *bindings[2 * MAX_BUFFERS] = {};
    for (unsigned i = 0; i < MAX_BUFFERS; i++)
    {
        bindings[i] = args.buffers[i];
        if (args.threadgroupLength[i])
        {
            tgMemory[i].assign(args.threadgroupLength[i], 0);
            bindings[MAX_BUFFERS + i] = tgMemory[i].data();
        }
    }
    unsigned simdCount = unsigned((threads + SIMD_WIDTH - 1) / SIMD_WIDTH);
    GroupState state;
    for (unsigned s = 0; s < simdCount; s++)
    {
        auto sg = std::make_unique<SimdGroup>();
        sg->size = unsigned(std::min<uint64_t>(SIMD_WIDTH, threads - uint64_t(s) * SIMD_WIDTH));
        state.simds.push_back(std::move(sg));
    }
    // End-of-group barrier: the last thread to arrive re-arms the per-group state
    std::mutex endMutex;
    std::condition_variable endCv;
    unsigned endArrived = 0;
    uint64_t endGeneration = 0;
    auto rearm = [&]() {
        state.tgBarrier.reset(unsigned(threads));
        for (auto &sg : state.simds)
        {
            sg->barrier.reset(sg->size);
            for (unsigned l = 0; l < sg->size; l++)
                sg->active[l] = true;
        }
    };
    rearm();

    auto worker = [&](unsigned flat) {
        uint32_t b[BUILTIN_SLOTS] = {};
        b[TID + 0] = flat % tg[0];
        b[TID + 1] = (flat / tg[0]) % tg[1];
        b[TID + 2] = flat / (tg[0] * tg[1]);
        for (int d = 0; d < 3; d++)
        {
            b[NTID + d] = tg[d];
            b[NCTAID + d] = grid[d];
            b[GRID + d] = tg[d] * grid[d];
        }
        b[FLAT_TID] = flat;
        b[LANE] = flat % SIMD_WIDTH;
        b[SIMD_INDEX] = flat / SIMD_WIDTH;
        b[SIMD_WIDTH_SLOT] = SIMD_WIDTH;
        b[SIMD_COUNT] = simdCount;
        SimdGroup *sg = state.simds[flat / SIMD_WIDTH].get();
        t_ctx = ThreadContext{&state, sg, flat % SIMD_WIDTH};
        for (uint64_t g = 0; g < groups; g++)
        {
            b[CTAID + 0] = uint32_t(g % grid[0]);
            b[CTAID + 1] = uint32_t((g / grid[0]) % grid[1]);
            b[CTAID + 2] = uint32_t(g / (uint64_t(grid[0]) * grid[1]));
            for (int d = 0; d < 3; d++)
                b[GID + d] = b[CTAID + d] * tg[d] + b[TID + d];
            k.entry(bindings, b);
            // leave this group's barriers
            sg->active[t_ctx.lane] = false;
            sg->barrier.drop();
            state.tgBarrier.drop();
            std::unique_lock<std::mutex> l(endMutex);
            uint64_t gen = endGeneration;
            if (++endArrived == threads)
            {
                endArrived = 0;
                rearm();
                endGeneration++;
                endCv.notify_all();
            }
            else
                endCv.wait(l, [&] { return endGeneration != gen; });
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (unsigned t = 0; t < threads; t++)
        pool.emplace_back(worker, t);
    for (auto &t : pool)
        t.join();
    return true;
}

} // namespace zsw
