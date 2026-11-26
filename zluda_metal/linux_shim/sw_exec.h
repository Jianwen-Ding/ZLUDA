// Software execution of Metal AIR kernels on the host CPU (Linux test device).
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace zsw
{

constexpr unsigned MAX_BUFFERS = 31;

struct LibraryFunction
{
    std::string name;
    std::vector<uint8_t> air; // raw AIR bitcode
};

// Parses a .metallib (see cpp/metallib.h)
bool parseLibrary(const void *data, size_t len, std::vector<LibraryFunction> &out, std::string &error);

struct CompiledKernel;

// JIT compiles one AIR kernel for the host. Returns nullptr and sets `error`
// if the AIR uses something the software device does not implement.
std::shared_ptr<CompiledKernel> compileKernel(const LibraryFunction &fn, std::string &error);
uint64_t staticThreadgroupMemory(const CompiledKernel &k);

struct DispatchArgs
{
    void *buffers[MAX_BUFFERS] = {};
    size_t threadgroupLength[MAX_BUFFERS] = {};
    uint32_t threadgroups[3] = {1, 1, 1};
    uint32_t threadsPerThreadgroup[3] = {1, 1, 1};
};

// Runs a whole dispatch, returns false and sets error on failure
bool dispatch(CompiledKernel &k, const DispatchArgs &args, std::string &error);

} // namespace zsw
