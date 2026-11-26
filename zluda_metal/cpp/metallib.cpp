// Open .metallib writer/reader, see metallib.h
#include "metallib.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "sha256.h"

namespace
{

constexpr uint16_t PLATFORM_MACOS = 0x8001;
constexpr uint16_t FILE_VERSION_MAJOR = 1;
constexpr uint16_t FILE_VERSION_MINOR = 2;
constexpr uint8_t FILE_TYPE_EXECUTABLE = 0;
constexpr uint8_t TARGET_OS_MACOS = 0x81;
constexpr uint16_t OS_VERSION_MAJOR = 14;
constexpr uint16_t OS_VERSION_MINOR = 0;
constexpr uint8_t FUNCTION_TYPE_KERNEL = 2;
constexpr size_t HEADER_SIZE = 0x58;

// Bitcode wrapper (the same "0x0B17C0DE" wrapper LLVM understands, see
// llvm/Bitcode/BitcodeReader.cpp: isBitcodeWrapper)
constexpr uint32_t WRAPPER_MAGIC = 0x0B17C0DE;
constexpr uint32_t WRAPPER_HEADER_SIZE = 20;
constexpr uint32_t CPU_TYPE_AIR64 = 0x01000017;

struct Writer
{
    std::vector<uint8_t> bytes;
    void u8(uint8_t v) { bytes.push_back(v); }
    void u16(uint16_t v)
    {
        for (int i = 0; i < 2; i++)
            bytes.push_back(uint8_t(v >> (8 * i)));
    }
    void u32(uint32_t v)
    {
        for (int i = 0; i < 4; i++)
            bytes.push_back(uint8_t(v >> (8 * i)));
    }
    void u64(uint64_t v)
    {
        for (int i = 0; i < 8; i++)
            bytes.push_back(uint8_t(v >> (8 * i)));
    }
    void raw(const void *p, size_t n)
    {
        const uint8_t *b = (const uint8_t *)p;
        bytes.insert(bytes.end(), b, b + n);
    }
    void tag(const char name[4], uint16_t size)
    {
        raw(name, 4);
        u16(size);
    }
    void patch64(size_t at, uint64_t v)
    {
        for (int i = 0; i < 8; i++)
            bytes[at + i] = uint8_t(v >> (8 * i));
    }
};

struct Reader
{
    const uint8_t *data;
    size_t len;
    size_t pos = 0;
    bool ok = true;
    bool need(size_t n)
    {
        if (pos + n > len)
            ok = false;
        return ok;
    }
    uint16_t u16()
    {
        if (!need(2))
            return 0;
        uint16_t v = data[pos] | (data[pos + 1] << 8);
        pos += 2;
        return v;
    }
    uint32_t u32()
    {
        if (!need(4))
            return 0;
        uint32_t v = 0;
        for (int i = 0; i < 4; i++)
            v |= uint32_t(data[pos + i]) << (8 * i);
        pos += 4;
        return v;
    }
    uint64_t u64()
    {
        if (!need(8))
            return 0;
        uint64_t v = 0;
        for (int i = 0; i < 8; i++)
            v |= uint64_t(data[pos + i]) << (8 * i);
        pos += 8;
        return v;
    }
};

uint32_t read32(const uint8_t *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (uint32_t(p[3]) << 24); }

char *dupString(const std::string &s)
{
    char *p = (char *)malloc(s.size() + 1);
    memcpy(p, s.data(), s.size());
    p[s.size()] = 0;
    return p;
}

void freeNames(std::vector<ZludaMetallibEntry> &v)
{
    for (auto &e : v)
        free(e.name);
}

} // namespace

extern "C" int zluda_metallib_write(const ZludaMetallibFunction *functions, size_t count, uint8_t **out, size_t *out_len)
{
    // wrapped bitcode for each function
    std::vector<std::vector<uint8_t>> wrapped(count);
    for (size_t i = 0; i < count; i++)
    {
        const uint8_t *bc = functions[i].air;
        size_t bcLen = functions[i].air_len;
        // LLVM already wraps bitcode for Darwin-style triples; unwrap and re-wrap
        if (bcLen >= WRAPPER_HEADER_SIZE && read32(bc) == WRAPPER_MAGIC)
        {
            uint32_t off = read32(bc + 8), size = read32(bc + 12);
            if ((size_t)off + size > bcLen)
                return 1;
            bc += off;
            bcLen = size;
        }
        if (bcLen < 4 || bc[0] != 'B' || bc[1] != 'C')
            return 1;
        Writer w;
        w.u32(WRAPPER_MAGIC);
        w.u32(0);
        w.u32(WRAPPER_HEADER_SIZE);
        w.u32((uint32_t)bcLen);
        w.u32(CPU_TYPE_AIR64);
        w.raw(bc, bcLen);
        while (w.bytes.size() % 16)
            w.u8(0);
        wrapped[i] = std::move(w.bytes);
    }

    // function list
    Writer list;
    list.u32((uint32_t)count);
    uint64_t bitcodeOffset = 0;
    for (size_t i = 0; i < count; i++)
    {
        Writer entry;
        size_t nameLen = strlen(functions[i].name) + 1;
        entry.tag("NAME", (uint16_t)nameLen);
        entry.raw(functions[i].name, nameLen);
        entry.tag("TYPE", 1);
        entry.u8(FUNCTION_TYPE_KERNEL);
        entry.tag("HASH", 32);
        auto hash = zluda::sha256(wrapped[i].data(), wrapped[i].size());
        entry.raw(hash.data(), 32);
        entry.tag("MDSZ", 8);
        entry.u64(wrapped[i].size());
        entry.tag("OFFT", 24);
        entry.u64(i * 8); // public metadata offset
        entry.u64(i * 8); // private metadata offset
        entry.u64(bitcodeOffset);
        entry.tag("VERS", 8);
        entry.u16(2); // AIR 2.6
        entry.u16(6);
        entry.u16(3); // Metal 3.1
        entry.u16(1);
        entry.raw("ENDT", 4);
        list.u32((uint32_t)entry.bytes.size() + 4);
        list.raw(entry.bytes.data(), entry.bytes.size());
        bitcodeOffset += wrapped[i].size();
    }

    // per-function public / private metadata (empty tag groups)
    Writer publicMd, privateMd;
    for (size_t i = 0; i < count; i++)
    {
        publicMd.u32(8);
        publicMd.raw("ENDT", 4);
        privateMd.u32(8);
        privateMd.raw("ENDT", 4);
    }

    Writer file;
    file.raw("MTLB", 4);
    file.u16(PLATFORM_MACOS);
    file.u16(FILE_VERSION_MAJOR);
    file.u16(FILE_VERSION_MINOR);
    file.u8(FILE_TYPE_EXECUTABLE);
    file.u8(TARGET_OS_MACOS);
    file.u16(OS_VERSION_MAJOR);
    file.u16(OS_VERSION_MINOR);
    size_t sizesAt = file.bytes.size();
    for (int i = 0; i < 9; i++)
        file.u64(0);
    if (file.bytes.size() != HEADER_SIZE)
        return 1;
    uint64_t listOff = file.bytes.size();
    file.raw(list.bytes.data(), list.bytes.size());
    uint64_t pubOff = file.bytes.size();
    file.raw(publicMd.bytes.data(), publicMd.bytes.size());
    uint64_t privOff = file.bytes.size();
    file.raw(privateMd.bytes.data(), privateMd.bytes.size());
    while (file.bytes.size() % 16)
        file.u8(0);
    uint64_t bcOff = file.bytes.size();
    for (auto &w : wrapped)
        file.raw(w.data(), w.size());
    file.patch64(sizesAt + 0, file.bytes.size());
    file.patch64(sizesAt + 8, listOff);
    file.patch64(sizesAt + 16, list.bytes.size());
    file.patch64(sizesAt + 24, pubOff);
    file.patch64(sizesAt + 32, publicMd.bytes.size());
    file.patch64(sizesAt + 40, privOff);
    file.patch64(sizesAt + 48, privateMd.bytes.size());
    file.patch64(sizesAt + 56, bcOff);
    file.patch64(sizesAt + 64, bitcodeOffset);

    *out_len = file.bytes.size();
    *out = (uint8_t *)malloc(file.bytes.size());
    memcpy(*out, file.bytes.data(), file.bytes.size());
    return 0;
}

extern "C" void zluda_metallib_free(uint8_t *data) { free(data); }

extern "C" int zluda_metallib_read(const uint8_t *data, size_t len, ZludaMetallibEntry **entries, size_t *count,
                                   char **error)
{
    *entries = nullptr;
    *count = 0;
    *error = nullptr;
    auto err = [&](const char *msg) {
        *error = dupString(msg);
        return 1;
    };
    if (len < HEADER_SIZE || memcmp(data, "MTLB", 4) != 0)
        return err("not a metallib (bad magic)");
    Reader h{data, len, 0x10};
    uint64_t fileSize = h.u64();
    uint64_t listOff = h.u64(), listSize = h.u64();
    h.u64(); // public metadata
    h.u64();
    h.u64(); // private metadata
    h.u64();
    uint64_t bcOff = h.u64(), bcSize = h.u64();
    if (fileSize > len || listOff + listSize > len || bcOff + bcSize > len)
        return err("metallib sections out of bounds");
    Reader r{data, listOff + listSize, listOff};
    uint32_t n = r.u32();
    std::vector<ZludaMetallibEntry> out;
    for (uint32_t i = 0; i < n && r.ok; i++)
    {
        size_t entryStart = r.pos;
        uint32_t entrySize = r.u32();
        size_t entryEnd = entryStart + entrySize;
        ZludaMetallibEntry e{};
        std::string name;
        uint64_t mdsz = 0, bcRel = 0;
        bool haveOffsets = false;
        while (r.ok && r.pos + 4 <= entryEnd)
        {
            const uint8_t *tag = data + r.pos;
            r.pos += 4;
            if (memcmp(tag, "ENDT", 4) == 0)
                break;
            uint16_t size = r.u16();
            if (!r.need(size))
                break;
            const uint8_t *payload = data + r.pos;
            if (memcmp(tag, "NAME", 4) == 0)
                name.assign((const char *)payload, strnlen((const char *)payload, size));
            else if (memcmp(tag, "TYPE", 4) == 0 && size >= 1)
                e.type = payload[0];
            else if (memcmp(tag, "HASH", 4) == 0 && size == 32)
                memcpy(e.hash, payload, 32);
            else if (memcmp(tag, "MDSZ", 4) == 0 && size == 8)
            {
                Reader t{payload, 8};
                mdsz = t.u64();
            }
            else if (memcmp(tag, "OFFT", 4) == 0 && size == 24)
            {
                Reader t{payload, 24};
                t.u64();
                t.u64();
                bcRel = t.u64();
                haveOffsets = true;
            }
            r.pos += size;
        }
        r.pos = entryEnd;
        if (!haveOffsets || bcRel + mdsz > bcSize)
        {
            freeNames(out);
            return err("function entry without valid bitcode offsets");
        }
        const uint8_t *bc = data + bcOff + bcRel;
        size_t bcLen = mdsz;
        if (bcLen >= WRAPPER_HEADER_SIZE && read32(bc) == WRAPPER_MAGIC)
        {
            uint32_t off = read32(bc + 8), size = read32(bc + 12);
            if (off + size > bcLen)
            {
                freeNames(out);
                return err("bad bitcode wrapper");
            }
            auto hash = zluda::sha256(bc, bcLen);
            if (memcmp(hash.data(), e.hash, 32) != 0)
            {
                freeNames(out);
                return err("function hash mismatch");
            }
            bc += off;
            bcLen = size;
        }
        e.name = dupString(name);
        e.air = bc;
        e.air_len = bcLen;
        out.push_back(e);
    }
    if (!r.ok)
    {
        freeNames(out);
        return err("truncated function list");
    }
    *count = out.size();
    *entries = (ZludaMetallibEntry *)calloc(out.size() ? out.size() : 1, sizeof(ZludaMetallibEntry));
    if (!out.empty())
        memcpy(*entries, out.data(), out.size() * sizeof(ZludaMetallibEntry));
    return 0;
}

extern "C" void zluda_metallib_free_entries(ZludaMetallibEntry *entries, size_t count)
{
    if (!entries)
        return;
    for (size_t i = 0; i < count; i++)
        free(entries[i].name);
    free(entries);
}
