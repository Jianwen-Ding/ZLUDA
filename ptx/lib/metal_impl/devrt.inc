// Included inside extern "C" of zluda_ptx_metal_impl.cpp: CUDA's device runtime services: printf (vprintf), the device
// heap (malloc / free), the clocks (%clock, %clock64, %globaltimer) and __nanosleep. Their state lives in buffers the
// host owns (zluda_metal/src/device_rt.rs), found through module variables the host fills before each launch:
//
//   __zluda_devrt_printf  the printf FIFO's device address (0: no FIFO)
//   __zluda_devrt_heap    the device heap's device address (0: no heap)
//   __zluda_devrt_clock   [0] the cycle counter (advanced by the reads), [1] its value at the launch, [2] the
//                         launch's %globaltimer in ns
//
// Every layout below is shared with device_rt.rs.
#define DEVICE_SPACE __attribute__((address_space(1)))

    DEVICE_SPACE uint64_t __zluda_devrt_printf;
    DEVICE_SPACE uint64_t __zluda_devrt_heap;
    DEVICE_SPACE uint64_t __zluda_devrt_clock[3];

    // ---------------------------------------------------------------------
    // Clocks
    // ---------------------------------------------------------------------
    // One cycle counter per launch, in device memory: every read by a SIMD group advances it by 25 cycles (a plausible issue-to-issue
    // distance of dependent special-register reads), __nanosleep(t) by t ns at the reported clock rate (1.398 GHz,
    // cuDeviceGetAttribute CLOCK_RATE). It starts at the host's running count (the cycles of every earlier launch,
    // from the simulator's instruction counts), so %clock64 is monotonic within and across launches; %globaltimer is
    // the launch's start time in ns plus the cycles since the launch at that rate.
    // the SIMD group's active lanes read (and sleep) together: its lowest active lane advances the counter once and
    // the others get its value, as the lanes of a warp read one %clock64 value
    static inline uint64_t clock_advance(uint64_t cycles)
    {
        uint32_t leader = __builtin_ctz(ballot32(true));
        uint64_t v = 0;
        if (__zluda_ptx_impl_sreg_laneid() == leader)
            v = __atomic_fetch_add(&__zluda_devrt_clock[0], cycles, __ATOMIC_RELAXED) + cycles;
        uint32_t lo = air_simd_shuffle((uint32_t)v, (uint16_t)leader);
        uint32_t hi = air_simd_shuffle((uint32_t)(v >> 32), (uint16_t)leader);
        return (uint64_t)hi << 32 | lo;
    }
    static inline uint64_t clock_tick() { return clock_advance(25); }
    static inline uint64_t timer_ns(uint64_t cycles)
    {
        return __zluda_devrt_clock[2] + (cycles - __zluda_devrt_clock[1]) * 1000u / 1398u;
    }
    uint64_t FUNC(sreg_clock64)() { return clock_tick(); }
    uint32_t FUNC(sreg_clock)() { return (uint32_t)clock_tick(); }
    uint64_t FUNC(sreg_globaltimer)() { return timer_ns(clock_tick()); }
    uint32_t FUNC(sreg_globaltimer_lo)() { return (uint32_t)timer_ns(clock_tick()); }
    uint32_t FUNC(sreg_globaltimer_hi)() { return (uint32_t)(timer_ns(clock_tick()) >> 32); }
    // PTX: the thread sleeps for about t ns (at most 1 ms); the functional model has no time, the clocks advance
    void FUNC(nanosleep_u32)(uint32_t t)
    {
        if (t > 1000000u)
            t = 1000000u;
        clock_advance((uint64_t)t * 1398u / 1000u + 1u);
    }

    // ---------------------------------------------------------------------
    // printf
    // ---------------------------------------------------------------------
    // FIFO: u32 head (bytes ever reserved since the last host flush), u32 capacity (data bytes, a multiple of 8), 8
    // reserved bytes, then the circular data. A record is 8-aligned: u32 its absolute start, u32 its length, u64 the
    // format string's address, then one 8-byte slot per argument consumed (an integer zero-extended from its size in
    // the argument buffer, a double, a pointer), a %s argument as its length (~0: a null pointer) and its bytes padded
    // to 8. As CUDA's, the buffer is circular: a launch that prints more than it holds overwrites the oldest output.
    // The host (device_rt.rs) parses the format again to print each record.
    struct printf_spec
    {
        uint32_t stars;  // '*' width / precision arguments (ints)
        uint32_t size;   // bytes of the argument in the argument buffer: 4 or 8, 0 for none (%%), 1 for %s
        uint32_t length; // characters of the specifier after '%'
    };
    static inline bool is_flag(char c) { return c == '-' || c == '+' || c == ' ' || c == '#' || c == '0' || c == '\''; }
    // the specifier starting after a '%': size 0 and stars 0 for %%, or an unknown conversion (printed as is)
    static printf_spec parse_spec(const char *p)
    {
        printf_spec s = {0, 0, 0};
        uint32_t i = 0;
        while (is_flag(p[i]))
            i++;
        if (p[i] == '*')
            s.stars++, i++;
        else
            while (p[i] >= '0' && p[i] <= '9')
                i++;
        if (p[i] == '.')
        {
            i++;
            if (p[i] == '*')
                s.stars++, i++;
            else
                while (p[i] >= '0' && p[i] <= '9')
                    i++;
        }
        bool wide = false;
        for (;;)
        {
            char c = p[i];
            if (c == 'h')
                i++;
            else if (c == 'l' || c == 'L' || c == 'q' || c == 'j' || c == 'z' || c == 't')
                wide = true, i++;
            else
                break;
        }
        char c = p[i];
        if (c == 0)
        {
            s.length = i;
            s.stars = 0;
            return s;
        }
        i++;
        s.length = i;
        switch (c)
        {
        case 'd': case 'i': case 'u': case 'x': case 'X': case 'o': case 'c':
            s.size = wide ? 8 : 4;
            break;
        case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': case 'a': case 'A':
        case 'p': case 'n':
            s.size = 8;
            break;
        case 's':
            s.size = 1;
            break;
        default: // %% and unknown conversions consume nothing
            s.stars = 0;
            break;
        }
        return s;
    }
    static inline uint64_t load_arg(const uint8_t *args, uint32_t &off, uint32_t size)
    {
        off = (off + size - 1) & ~(size - 1);
        if (!args) // no argument buffer (a format without arguments): zeros
            return 0;
        uint64_t v = size == 8 ? *(const uint64_t *)(args + off) : (uint64_t) * (const uint32_t *)(args + off);
        off += size;
        return v;
    }
    static inline uint32_t bounded_strlen(const char *s)
    {
        uint32_t n = 0;
        while (n < 4096u && s[n])
            n++;
        return n;
    }
    // CUDA: at most 32 arguments are printed
    enum { PRINTF_MAX_ARGS = 32 };

    // (PTX passes the format and the argument buffer as .b64 generic addresses)
    int FUNC(vprintf)(uint64_t fmt_addr, uint64_t args_addr)
    {
        const char *fmt = (const char *)fmt_addr;
        if (!fmt)
            return -1;
        const uint8_t *args = (const uint8_t *)args_addr;
        // the record's size: 16 bytes of header, 8 per argument, a string's bytes
        uint32_t size = 16, off = 0, nargs = 0;
        for (const char *p = fmt; *p; p++)
        {
            if (*p != '%')
                continue;
            printf_spec s = parse_spec(p + 1);
            p += s.length;
            if (!s.size)
                continue;
            for (uint32_t k = 0; k < s.stars && nargs < PRINTF_MAX_ARGS; k++, nargs++)
            {
                load_arg(args, off, 4);
                size += 8;
            }
            if (nargs >= PRINTF_MAX_ARGS)
                break;
            nargs++;
            uint64_t v = load_arg(args, off, s.size == 1 ? 8 : s.size);
            size += 8;
            if (s.size == 1 && v)
                size += (bounded_strlen((const char *)v) + 7u) & ~7u;
            if (!*p)
                break;
        }
        uint64_t fifo = __zluda_devrt_printf;
        if (!fifo)
            return -2;
        DEVICE_SPACE uint32_t *hdr = (DEVICE_SPACE uint32_t *)fifo;
        DEVICE_SPACE uint8_t *data = (DEVICE_SPACE uint8_t *)fifo + 16;
        uint32_t cap = hdr[1];
        if (size > cap)
            return -2;
        uint32_t start = __atomic_fetch_add(&hdr[0], size, __ATOMIC_RELAXED);
        auto put64 = [&](uint32_t at, uint64_t v) { *(DEVICE_SPACE uint64_t *)(data + (start + at) % cap) = v; };
        put64(0, (uint64_t)start | (uint64_t)size << 32);
        put64(8, (uint64_t)fmt);
        uint32_t w = 16;
        off = 0;
        nargs = 0;
        for (const char *p = fmt; *p; p++)
        {
            if (*p != '%')
                continue;
            printf_spec s = parse_spec(p + 1);
            p += s.length;
            if (!s.size)
                continue;
            for (uint32_t k = 0; k < s.stars && nargs < PRINTF_MAX_ARGS; k++, nargs++, w += 8)
                put64(w, load_arg(args, off, 4));
            if (nargs >= PRINTF_MAX_ARGS)
                break;
            nargs++;
            uint64_t v = load_arg(args, off, s.size == 1 ? 8 : s.size);
            if (s.size == 1)
            {
                const char *str = (const char *)v;
                uint32_t n = str ? bounded_strlen(str) : 0;
                put64(w, str ? n : ~0ull);
                w += 8;
                for (uint32_t k = 0; k < ((n + 7u) & ~7u); k += 8, w += 8)
                {
                    uint64_t word = 0;
                    for (uint32_t b = 0; b < 8 && k + b < n; b++)
                        word |= (uint64_t)(uint8_t)str[k + b] << (8 * b);
                    put64(w, word);
                }
            }
            else
            {
                put64(w, v);
                w += 8;
            }
            if (!*p)
                break;
        }
        return (int)nargs;
    }

    // ---------------------------------------------------------------------
    // Device heap (malloc / free)
    // ---------------------------------------------------------------------
    // Heap: u64 bump offset, u64 heap size (bytes, from the start of the heap buffer), then HEAP_CLASSES free-list
    // heads, then the arena (at HEAP_ARENA). Blocks are size classes of four steps per power of two (32 bytes up,
    // at most 25% waste), with a 16-byte header (u32 class, u32 magic; then the next free block while free), so
    // the user pointer is 16-byte aligned as CUDA's. The lists are lock-free stacks (64-bit CAS, a 24-bit tag over a
    // 40-bit offset against ABA): no thread ever waits for another, which divergent SIMD groups could not honor.
    // Freed blocks are reused by allocations of the same class; blocks are not split or coalesced.
    enum { HEAP_CLASSES = 144, HEAP_ARENA = 1280 };
    static const uint32_t HEAP_MAGIC = 0x5a4d414cu;
    static inline uint64_t heap_class_size(uint32_t c) { return (uint64_t)(4 + (c & 3)) << ((c >> 2) + 3); }
    static inline uint32_t heap_class(uint64_t bytes)
    {
        if (bytes <= 32)
            return 0;
        uint32_t k = 63 - __builtin_clzll(bytes); // 2^k <= bytes < 2^(k+1), k >= 5
        uint64_t q = (bytes + (1ull << (k - 2)) - 1) >> (k - 2); // 4..8 quarters
        if (q == 8)
            k++, q = 4;
        return (k - 5) * 4 + (uint32_t)(q - 4);
    }

    // (PTX passes and returns the pointers as .b64)
    uint64_t FUNC(malloc)(uint64_t size)
    {
        uint64_t heap = __zluda_devrt_heap;
        if (!heap || size > (1ull << 39))
            return 0;
        DEVICE_SPACE uint64_t *h = (DEVICE_SPACE uint64_t *)heap;
        uint32_t c = heap_class(size + 16);
        if (c >= HEAP_CLASSES)
            return 0;
        DEVICE_SPACE uint64_t *head = h + 2 + c;
        const uint64_t mask = (1ull << 40) - 1;
        uint64_t block = 0;
        uint64_t old = __atomic_load_n(head, __ATOMIC_ACQUIRE);
        while (old & mask)
        {
            uint64_t next = *(DEVICE_SPACE uint64_t *)(heap + (old & mask) + 8);
            uint64_t nv = (next & mask) | ((old >> 40) + 1) << 40;
            if (__atomic_compare_exchange_n(head, &old, nv, true, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            {
                block = old & mask;
                break;
            }
        }
        if (!block)
        {
            uint64_t bytes = heap_class_size(c);
            uint64_t limit = h[1];
            uint64_t cur = __atomic_load_n(&h[0], __ATOMIC_RELAXED);
            do
            {
                if (cur < HEAP_ARENA || cur + bytes > limit)
                    return 0;
            } while (!__atomic_compare_exchange_n(&h[0], &cur, cur + bytes, true, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED));
            block = cur;
        }
        DEVICE_SPACE uint32_t *hdr = (DEVICE_SPACE uint32_t *)(heap + block);
        hdr[0] = c;
        hdr[1] = HEAP_MAGIC;
        return heap + block + 16;
    }

    void FUNC(free)(uint64_t a)
    {
        uint64_t heap = __zluda_devrt_heap;
        if (!a || !heap || a < heap + HEAP_ARENA + 16)
            return;
        uint64_t block = a - 16 - heap;
        DEVICE_SPACE uint32_t *hdr = (DEVICE_SPACE uint32_t *)(heap + block);
        uint32_t c = hdr[0];
        if (hdr[1] != HEAP_MAGIC || c >= HEAP_CLASSES)
            return; // not a block of this heap (CUDA: undefined)
        hdr[1] = 0;
        DEVICE_SPACE uint64_t *head = (DEVICE_SPACE uint64_t *)heap + 2 + c;
        const uint64_t mask = (1ull << 40) - 1;
        uint64_t old = __atomic_load_n(head, __ATOMIC_RELAXED);
        for (;;)
        {
            *(DEVICE_SPACE uint64_t *)(heap + block + 8) = old & mask;
            uint64_t nv = block | ((old >> 40) + 1) << 40;
            if (__atomic_compare_exchange_n(head, &old, nv, true, __ATOMIC_RELEASE, __ATOMIC_RELAXED))
                break;
        }
    }
#undef DEVICE_SPACE
