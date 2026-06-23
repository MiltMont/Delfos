---
name: llvm-cpp-toolchain
description: LLVM-based toolchain for debugging and profiling C++ code in this project. Covers sanitizers, profiling, optimization, and benchmarking — all Clang/LLVM, no GCC or Valgrind.
---

# C++ Debugging & Profiling Toolchain (LLVM)

All debugging and performance work uses the LLVM/Clang toolchain exclusively.

## Sanitizers (Debug/CI)

| Sanitizer | Flag | Catches | Overhead |
|---|---|---|---|
| AddressSanitizer | `-fsanitize=address` | Buffer overflows, use-after-free, double-free, leaks | ~2x |
| UndefinedBehaviorSanitizer | `-fsanitize=undefined` | Signed overflow, null deref, misaligned access | ~1.2x |
| ThreadSanitizer | `-fsanitize=thread` | Data races, lock-order inversions | ~5–15x |
| MemorySanitizer | `-fsanitize=memory` | Reads of uninitialized memory | ~3x |

Always compile with `-fno-omit-frame-pointer -g` alongside sanitizers. ASan+UBSan should run in CI on every commit. TSan when concurrency is involved.

## Profiling

| Tool | Purpose | Usage |
|---|---|---|
| `perf record -g` + flamegraphs | Find where CPU time goes | `perf record -g --call-graph=fp ./binary && perf script \| flamegraph.pl > flame.svg` |
| `llvm-xray` | Precise per-function instrumentation tracing | Compile with `-fxray-instrument`, run with `XRAY_OPTIONS="patch_premain=true xray_mode=xray-basic"`, analyze with `llvm-xray account` or convert to Perfetto/Chrome trace |
| `perf stat` hw counters | Cache misses, branch mispredicts | `perf stat -e cache-misses,branch-misses,LLC-load-misses ./binary` |
| MemProf | Allocation hot/cold analysis | Compile with `-fmemory-profile -gmlt`, merge with `llvm-profdata` |

Build for profiling: `CMAKE_BUILD_TYPE=RelWithDebInfo` + `-fno-omit-frame-pointer`.

## Optimization

| Tool | What it does | When to use |
|---|---|---|
| PGO (`-fprofile-generate` / `-fprofile-use`) | Profile-guided code generation (10–20% gains) | Once code is stable, before release |
| LTO (`-flto=thin`) | Link-time optimization across translation units | Always in release builds |
| BOLT (`llvm-bolt`) | Post-link binary layout optimization (5–15% gains) | Final production binary |

PGO workflow:
```bash
clang++ -fprofile-generate -O2 ...       # instrumented build
./binary --workload                       # run representative workload
llvm-profdata merge -o default.profdata *.profraw
clang++ -fprofile-use=default.profdata -O3 -flto=thin ...  # optimized build
```

## Benchmarking

Use [nanobench](https://github.com/martinus/nanobench) (single header). It reads perf counters directly and reports median, percentiles, instructions/op, branch misses/op, cache misses/op.

```cpp
#include <nanobench.h>

ankerl::nanobench::Bench().run("name", [&] {
    auto result = function_under_test();
    ankerl::nanobench::doNotOptimizeAway(result);
});
```

## CMake Presets

```json
{
  "configurePresets": [
    {
      "name": "debug",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "CMAKE_CXX_COMPILER": "clang++",
        "CMAKE_CXX_FLAGS": "-fsanitize=address,undefined -fno-omit-frame-pointer"
      }
    },
    {
      "name": "profile",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "RelWithDebInfo",
        "CMAKE_CXX_COMPILER": "clang++",
        "CMAKE_CXX_FLAGS": "-fno-omit-frame-pointer -fxray-instrument"
      }
    },
    {
      "name": "release",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CXX_COMPILER": "clang++",
        "CMAKE_CXX_FLAGS": "-O3 -flto=thin -march=native"
      }
    }
  ]
}
```

## Workflow

```
1. Develop with ASan+UBSan (catch memory/UB bugs immediately)
2. Write nanobench micro-benchmarks (establish baseline)
3. perf → flamegraph (identify hotspots)
4. llvm-xray → Perfetto (precise function-level timing)
5. perf stat (validate cache/branch behavior)
6. PGO + LTO (profile-guided release build)
7. BOLT (final binary layout optimization)
```
