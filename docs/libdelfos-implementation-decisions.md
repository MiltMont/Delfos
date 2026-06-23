# libdelfos Implementation Decision Log

This document records implementation decisions that were required during
execution but were **not explicitly specified** in `docs/libdelfos-plan.md`.

Use this as an append-only log during implementation phases.

## How to update this log

- Add one entry per decision.
- Keep entries factual and concise.
- Include the impact so later contributors know what may need revision.
- Prefer adding new entries over rewriting old ones.

## Entries

| Date (UTC) | Area | Decision | Why this was needed | Impact |
|---|---|---|---|---|
| 2026-06-23 | Build system | Set `cmake_minimum_required(VERSION 3.28)` in top-level `CMakeLists.txt`. | The plan showed 3.20 in one example and 3.28 in cloud/toolchain guidance; a single baseline was needed for local consistency. | Requires CMake 3.28+ for configure in this repo. |
| 2026-06-23 | Testing framework | Pin Catch2 via `FetchContent` to `v3.6.0` for Phase 1 tests. | Plan specified Catch2 `v3.x` but did not pick an exact tag. | Reproducible test dependency resolution for current setup. |
| 2026-06-23 | Clang/CMake compatibility | Set `CMAKE_CXX_SCAN_FOR_MODULES OFF`. | The environment lacked `clang-scan-deps` wiring, causing Ninja builds to fail during dependency scanning. | Allows debug build/test execution with current Clang toolchain; can be revisited once module-scanning tooling is available. |
| 2026-06-23 | Phase 1 scope interpretation | Implemented a functional `Graph` skeleton (`upsert`, delete, `find`, `rebuild`, adjacency queries) instead of pure stubs. | "Setup" could be interpreted as scaffolding only; choosing a runnable baseline made the phase testable immediately. | Phase 1 now starts from compilable behavior, with later phases able to iterate on invariants/perf. |

