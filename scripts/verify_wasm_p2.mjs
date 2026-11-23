import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { resolve } from "node:path";

const wasmPath = resolve(
  import.meta.dirname,
  "../target/wasm32-unknown-unknown/release/polyglot_agent_runtime.wasm"
);
const wasmBytes = readFileSync(wasmPath);
const { instance } = await WebAssembly.instantiate(wasmBytes, {});
const ex = instance.exports;

const stateBytes = ex.par_p2_suite_state_bytes();
assert.equal(stateBytes, 512, "wasm32 P2 state footprint must equal exact 512 bytes (8 cache lines)");

// Verify RFC 3720 Castagnoli CRC32C ("123456789" -> 0xE3069283) inside WASM linear memory
const asciiVec = new TextEncoder().encode("123456789");
const bytePtr = ex.par_wasm_alloc_bytes(asciiVec.length);
new Uint8Array(ex.memory.buffer, bytePtr, asciiVec.length).set(asciiVec);
const crcUnsigned = ex.par_wasm_crc32c(bytePtr, asciiVec.length) >>> 0;
ex.par_wasm_free_bytes(bytePtr, asciiVec.length);
assert.equal(crcUnsigned, 0xE3069283, "RFC 3720 CRC32C check vector mismatch in WASM");

// Benchmark 2,000,000 streaming observations across zero-copy Float64Array in WASM linear memory
const suitePtr = ex.par_p2_suite_new();
assert.ok(suitePtr !== 0, "WASM suite allocation failed");

const batchSize = 4096;
const totalBatches = 500; // 2,048,000 observations
const totalObs = batchSize * totalBatches;
const bufPtr = ex.par_p2_alloc_f64_buffer(batchSize);
assert.ok(bufPtr !== 0, "WASM f64 scratch buffer allocation failed");

// Pre-fill deterministic Weyl golden-ratio batch in WASM linear memory
{
  const view = new Float64Array(ex.memory.buffer, bufPtr, batchSize);
  for (let i = 0; i < batchSize; i += 1) {
    const frac = (i * 0.6180339887498949) % 1.0;
    view[i] = frac * 1000.0;
  }
}

const t0 = performance.now();
let count = 0n;
for (let b = 0; b < totalBatches; b += 1) {
  count = ex.par_p2_suite_observe_batch(suitePtr, bufPtr, batchSize);
}
const elapsedSec = (performance.now() - t0) / 1000.0;
const obsPerSec = totalObs / elapsedSec;

const p50 = ex.par_p2_suite_p50(suitePtr);
const p95 = ex.par_p2_suite_p95(suitePtr);
const p99 = ex.par_p2_suite_p99(suitePtr);
const mean = ex.par_p2_suite_mean(suitePtr);
const min = ex.par_p2_suite_min(suitePtr);
const max = ex.par_p2_suite_max(suitePtr);

assert.equal(count, BigInt(totalObs));
assert.ok(Math.abs(p50 - 500.0) < 6.0, `WASM p50=${p50} out of tolerance`);
assert.ok(Math.abs(p95 - 950.0) < 6.0, `WASM p95=${p95} out of tolerance`);
assert.ok(Math.abs(p99 - 990.0) < 4.0, `WASM p99=${p99} out of tolerance`);
assert.ok(Math.abs(mean - 500.0) < 2.0, `WASM mean=${mean} out of tolerance`);
assert.ok(min >= 0.0 && max <= 1000.0);

ex.par_p2_free_f64_buffer(bufPtr, batchSize);
ex.par_p2_suite_free(suitePtr);

// Verify v1.2.0 Deterministic 64B Token-Bucket Rate Limiter in WASM linear memory
const bucketBytes = ex.par_bucket_state_bytes();
assert.equal(bucketBytes, 64, "WASM token bucket state must equal exact 64 bytes (1 cache line)");
const bucketPtr = ex.par_bucket_new(5000n, 3333n, 1000000n);
assert.ok(bucketPtr !== 0 && bucketPtr % 64 === 0, "WASM token bucket must be 64B-aligned");
assert.equal(ex.par_bucket_try_acquire(bucketPtr, 5000n, 1000000n), 1, "Initial burst acquire must succeed");
let wasmNowNs = 1000000n;
for (let i = 0; i < 1000; i += 1) {
  wasmNowNs += 1000000n;
  ex.par_bucket_try_acquire(bucketPtr, 5000n, wasmNowNs);
}
assert.equal(
  ex.par_bucket_tokens_milli(bucketPtr),
  3333n,
  "WASM token bucket must conserve exact 3333 milli-tokens over 1000x 1ms steps"
);
const waitNs = ex.par_bucket_retry_after_ns(bucketPtr, 4000n);
assert.ok(waitNs > 0n);
assert.equal(ex.par_bucket_try_acquire(bucketPtr, 4000n, wasmNowNs + waitNs - 1n), 0);
assert.equal(ex.par_bucket_try_acquire(bucketPtr, 4000n, wasmNowNs + waitNs), 1);
assert.equal(ex.par_bucket_allowed_count(bucketPtr), 2n);
ex.par_bucket_free(bucketPtr);

// Verify v1.2.0 Capability Audit Ring (64 x 256B = 16,384B) & linear-memory CRC32C tamper detection
assert.equal(ex.par_audit_ring_slot_bytes(), 256);
assert.equal(ex.par_audit_ring_capacity(), 64);
const ringPtr = ex.par_audit_ring_new();
assert.ok(ringPtr !== 0, "WASM capability audit ring allocation failed");
const auditMsg = new TextEncoder().encode("mcp.tool:rag_hybrid_search");
const msgPtr = ex.par_wasm_alloc_bytes(auditMsg.length);
new Uint8Array(ex.memory.buffer, msgPtr, auditMsg.length).set(auditMsg);
// 1) Allowed (required=0x09, granted=0x0F -> 1)
assert.equal(ex.par_audit_ring_record(ringPtr, 1000n, 7, 0x09, 0x0f, msgPtr, auditMsg.length), 1);
// 2) Denied capability (required=0x10 GPU_DISPATCH, granted=0x09 -> 0, sealed in ring)
assert.equal(ex.par_audit_ring_record(ringPtr, 2000n, 7, 0x10, 0x09, msgPtr, auditMsg.length), 0);
assert.equal(ex.par_audit_ring_len(ringPtr), 2);
assert.equal(ex.par_audit_ring_verify_all(ringPtr), 2);

// Flip 1 payload bit in WASM linear memory (offset 64 inside 256B RingSlot #0) -> must return -8 (PAR_ERR_PAYLOAD_CRC)
const slot0Ptr = ex.par_audit_ring_slot_ptr(ringPtr, 0);
assert.ok(slot0Ptr !== 0 && slot0Ptr % 64 === 0, "WASM RingSlot must be 64B-aligned");
const payloadByteView = new Uint8Array(ex.memory.buffer, slot0Ptr + 64, 1);
payloadByteView[0] ^= 0x01;
assert.equal(ex.par_audit_ring_verify_all(ringPtr), -8, "Bit-flip in WASM audit ring payload must trigger -8 CRC32C error");
payloadByteView[0] ^= 0x01;
assert.equal(ex.par_audit_ring_verify_all(ringPtr), 2, "Restored WASM audit ring payload must re-verify");

ex.par_wasm_free_bytes(msgPtr, auditMsg.length);
ex.par_audit_ring_free(ringPtr);

console.log(
  JSON.stringify(
    {
      target: "wasm32-unknown-unknown",
      version: "1.2.0",
      wasmBytes: wasmBytes.byteLength,
      stateBytes,
      bucketStateBytes: bucketBytes,
      auditRingVerifiedSlots: 2,
      observations: totalObs,
      obsPerSec: Number(obsPerSec.toFixed(1)),
      nsPerObs: Number(((elapsedSec * 1e9) / totalObs).toFixed(2)),
      p50: Number(p50.toFixed(2)),
      p95: Number(p95.toFixed(2)),
      p99: Number(p99.toFixed(2)),
      crc32cRfc3720Hex: "0x" + crcUnsigned.toString(16).toUpperCase(),
      ok: true
    },
    null,
    2
  )
);
