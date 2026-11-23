# polyglot-agent-runtime-rs

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2021%20Edition-orange.svg)](Cargo.toml)
[![C11 FFI](https://img.shields.io/badge/FFI-ISO%20C11%20--Werror-blue.svg)](c_src/ring_crc.c)
[![External Deps](https://img.shields.io/badge/Runtime%20Dependencies-0-brightgreen.svg)](Cargo.toml)
[![WASM32](https://img.shields.io/badge/Target-wasm32--unknown--unknown-purple.svg)](scripts/verify_wasm_p2.mjs)

Deterministic, zero-dependency **Rust 2021 + ISO C11 FFI + `wasm32`** systems runtime engineered for sub-microsecond AI agent telemetry, zero-float-drift token-bucket rate limiting, capability-fenced DAG orchestration, tamper-evident audit rings, and constant-memory (`O(1)`) streaming tail-latency estimation.

---

## RFC System Architecture

```mermaid
flowchart LR
    Agent["Agent Step / Tool Invocation"] --> Fence["Capability Fence & 64B C11 Token Bucket"]
    Fence --> DAG["Kahn Topological DAG + Retry & Circuit Breaker"]
    DAG --> Vault["BoundedTelemetryVault<N> / CapabilityAuditRing"]
    Vault -->|"#[repr(C, align(64))] FFI"| C11["C11 ring_crc.c (CRC32C Seal + Rational Bucket)"]
    DAG --> P2["Jain-Chlamtac P2 Estimator (O(1) p50/p95/p99)"]
```

---

## Empirical Release Benchmarks (`cargo run --release --bin bench_runtime`)

| Subsystem | Workload | Throughput | Unit Cost / Latency | $p_{50}$ Latency | $p_{95}$ Latency | $p_{99}$ Latency | Memory Invariant |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **C11 CRC32C Ring Vault (`seal + validate`)** | `1,000,000` ops | **`3,138,485 ops/s`** | **`318.6 ns/op`** | **`0.3169 µs`** | **`0.3643 µs`** | **`0.4712 µs`** | `0` heap allocs post-init (`256 B/slot`) |
| **C11 Rational Token Bucket (`try_acquire`)** | `5,000,000` ops | **`12,760,307 ops/s`** | **`78.4 ns/op`** | — | — | — | **`64 B` (`1` cache line, `0` float drift)** |
| **Capability-Fenced DAG (`6` nodes)** | `50,000` DAGs | **`409,696 DAGs/s`** | **`2.44 µs/DAG`** | **`2.051 µs`** | **`5.807 µs`** | **`6.020 µs`** | Deterministic `BTreeSet` Kahn queue |
| **Jain-Chlamtac $P^2$ (`p50/p95/p99` Native)** | `2,000,000` obs | **`37,379,099 obs/s`** | **`26.8 ns/obs`** | — | — | — | **`560 B` (`O(1)` 64-bit state)** |
| **Jain-Chlamtac $P^2$ (`wasm32-unknown-unknown`)** | `2,048,000` obs | **`30,784,214 obs/s`** | **`32.5 ns/obs`** | — | — | — | **`512 B` (`O(1)` WASM state, `33.5 KB` `.wasm`)** |

---

## C11 FFI & Memory Safety Invariants

1. **Exact Cache-Line ABI Parity (`64` Bytes)**:
   - C11 `ParRingSlotHeader` enforces `_Alignas(64)` and compile-time `_Static_assert(sizeof(ParRingSlotHeader) == 64U)` + `_Static_assert(offsetof(ParRingSlotHeader, header_crc32c) == 40U)`.
   - Rust `RingSlotHeader` mirrors every field under `#[repr(C, align(64))]` with matching `const` size and alignment assertions, eliminating false sharing across adjacent cache lines.
2. **Strict Compiler Hygiene**:
   - `build.rs` compiles `c_src/ring_crc.c` with `-std=c11 -O3 -Wall -Wextra -Werror -Wpedantic` and links statically with zero external crate dependencies.
3. **Bounded Pointer Provenance & Zero Ownership Transfer**:
   - Every `unsafe` FFI call in `src/lib.rs` passes borrows whose lifetimes strictly outlive the synchronous C11 checksum/header routine; C11 never allocates, frees, or retains Rust pointers.
4. **Dual-Stage Integrity Verification & Sub-Nanosecond Rational Rate Conservation**:
   - `par_ring_header_validate` checks non-null pointers, `64`-byte pointer alignment, magic `0x50415231` (`"PAR1"`), version `1`, payload length bounds (`<= 192 B`), zeroed `20`-byte reserved region, header CRC32C, and Castagnoli payload CRC32C (verified against RFC 3720 check vector `0xE3069283`).
   - `par_token_bucket_try_acquire` maintains exact milli-token balances and sub-nanosecond rational remainders (`rem_ns_num < 10^9`) inside a single `64`-byte cache line (`ParTokenBucket` / `TokenBucketState`) with `0` floating-point drift and exact `1 ns` ceiling retry-after bounds.

---

## Quickstart

### 1. Build, Lint & Run Full Test Suite
```bash
cargo clippy --all-targets -- -D warnings
cargo test --verbose
cargo run --release --bin bench_runtime
```

### 2. Rust Library Usage
```rust
use polyglot_agent_runtime::{
    capabilities, BoundedTelemetryVault, CapabilityDagOrchestrator, CircuitBreaker, DagNode,
    RetryPolicy, StreamingQuantileSuite,
};

let mut vault = BoundedTelemetryVault::<1024>::new();
vault.record(1_700_000_000, 1, capabilities::TELEMETRY_EMIT, 0, b"step:ok")?;
assert_eq!(vault.verify_all()?, 1);

let mut quantiles = StreamingQuantileSuite::new();
for sample_us in [12.4, 15.1, 11.8, 19.2, 14.0, 42.5] {
    quantiles.observe(sample_us);
}
let snap = quantiles.snapshot();
println!("p50={:.2}us p95={:.2}us bytes={}", snap.p50, snap.p95, snap.state_bytes);
```

### 3. WebAssembly (`wasm32-unknown-unknown`) Browser & V8 Telemetry Export
```bash
rustup target add wasm32-unknown-unknown
cargo build --release --lib --target wasm32-unknown-unknown
node scripts/verify_wasm_p2.mjs
```
Exports zero-glue C-ABI symbols (`par_p2_suite_*`, `par_bucket_*`, `par_audit_ring_*`, `par_wasm_crc32c`) for zero-copy `Float64Array` linear-memory streaming, deterministic `64 B` token-bucket rate limiting, and `256 B/slot` CRC32C capability audit rings in browser dashboards and edge workers.

### 4. Containerized Execution
```bash
docker build -t polyglot-agent-runtime-rs:1.2.0 .
docker run --rm polyglot-agent-runtime-rs:1.2.0
```

---

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Nahian Akhtar.
