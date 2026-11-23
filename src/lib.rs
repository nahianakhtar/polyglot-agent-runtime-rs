//! # polyglot-agent-runtime-rs
//!
//! Production-grade systems runtime combining:
//! - **C11 FFI Cache-Line Ring Telemetry Vault** (`64`-byte aligned `ParRingSlotHeader`, Castagnoli CRC32C, zero heap allocation after initialization)
//! - **Deterministic Zero-Float-Drift Token-Bucket Rate Limiter** (`64`-byte aligned `ParTokenBucket` / `TokenBucketState`, exact sub-nanosecond rational remainder conservation across C11/Rust/WASM FFI)
//! - **Capability-Fenced DAG Task Orchestrator & Tamper-Evident Audit Ring** (deterministic Kahn topological sort, cycle detection, bounded retry, `Closed`/`Open`/`HalfOpen` circuit breaker, and C-ABI/WASM `CapabilityAuditRing` export)
//! - **Streaming Jain-Chlamtac P² Quantile Estimator** (`p50`, `p95`, `p99` in strict `O(1)` memory)

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

pub const RING_MAGIC: u32 = 0x5041_5231;
pub const RING_VERSION: u16 = 1;
pub const MAX_PAYLOAD_LEN: usize = 192;
pub const BUCKET_MAGIC: u32 = 0x5054_4231;
pub const BUCKET_VERSION: u16 = 1;
pub const NS_PER_SEC: u64 = 1_000_000_000;
pub const DEFAULT_AUDIT_RING_SLOTS: usize = 64;

/// Exact C11 ABI mirror of `ParRingSlotHeader` (`64` bytes, `64`-byte cache-line aligned).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RingSlotHeader {
    pub magic: u32,
    pub version: u16,
    pub flags: u16,
    pub sequence: u64,
    pub timestamp_ns: u64,
    pub agent_id: u32,
    pub capability_mask: u32,
    pub payload_len: u32,
    pub payload_crc32c: u32,
    pub header_crc32c: u32,
    pub reserved: [u8; 20],
}

const _: () = {
    assert!(core::mem::size_of::<RingSlotHeader>() == 64);
    assert!(core::mem::align_of::<RingSlotHeader>() == 64);
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfiRingError {
    NullPointer,
    MisalignedHeader,
    PayloadOutOfBounds,
    InvalidMagic,
    UnsupportedVersion,
    NonZeroReserved,
    HeaderCrcMismatch,
    PayloadCrcMismatch,
    InvalidRateLimitParam,
    UnknownCode(i32),
}

impl fmt::Display for FfiRingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for FfiRingError {}

fn decode_ffi_status(code: i32) -> Result<(), FfiRingError> {
    match code {
        0 => Ok(()),
        -1 => Err(FfiRingError::NullPointer),
        -2 => Err(FfiRingError::MisalignedHeader),
        -3 => Err(FfiRingError::PayloadOutOfBounds),
        -4 => Err(FfiRingError::InvalidMagic),
        -5 => Err(FfiRingError::UnsupportedVersion),
        -6 => Err(FfiRingError::NonZeroReserved),
        -7 => Err(FfiRingError::HeaderCrcMismatch),
        -8 => Err(FfiRingError::PayloadCrcMismatch),
        -9 => Err(FfiRingError::InvalidRateLimitParam),
        other => Err(FfiRingError::UnknownCode(other)),
    }
}

const fn encode_ffi_error(err: FfiRingError) -> i32 {
    match err {
        FfiRingError::NullPointer => -1,
        FfiRingError::MisalignedHeader => -2,
        FfiRingError::PayloadOutOfBounds => -3,
        FfiRingError::InvalidMagic => -4,
        FfiRingError::UnsupportedVersion => -5,
        FfiRingError::NonZeroReserved => -6,
        FfiRingError::HeaderCrcMismatch => -7,
        FfiRingError::PayloadCrcMismatch => -8,
        FfiRingError::InvalidRateLimitParam => -9,
        FfiRingError::UnknownCode(c) => c,
    }
}

/// Exact C11 ABI mirror of `ParTokenBucket` (`64` bytes, `64`-byte cache-line aligned).
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenBucketState {
    pub magic: u32,
    pub version: u16,
    pub flags: u16,
    pub capacity_milli: u64,
    pub tokens_milli: u64,
    pub refill_per_sec_milli: u64,
    pub last_refill_ns: u64,
    pub rem_ns_num: u64,
    pub allowed_count: u64,
    pub denied_count: u64,
}

const _: () = {
    assert!(core::mem::size_of::<TokenBucketState>() == 64);
    assert!(core::mem::align_of::<TokenBucketState>() == 64);
};

#[cfg(not(target_arch = "wasm32"))]
mod raw_ffi {
    use super::{RingSlotHeader, TokenBucketState};

    extern "C" {
        pub fn par_crc32c(data: *const u8, len: usize) -> u32;
        pub fn par_ring_header_seal(
            hdr: *mut RingSlotHeader,
            payload: *const u8,
            payload_len: usize,
        ) -> i32;
        pub fn par_ring_header_validate(
            hdr: *const RingSlotHeader,
            payload: *const u8,
            payload_len: usize,
        ) -> i32;
        pub fn par_token_bucket_init(
            tb: *mut TokenBucketState,
            capacity_milli: u64,
            refill_per_sec_milli: u64,
            initial_ns: u64,
        ) -> i32;
        pub fn par_token_bucket_try_acquire(
            tb: *mut TokenBucketState,
            cost_milli: u64,
            now_ns: u64,
            retry_after_ns_out: *mut u64,
        ) -> i32;
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm_ring_fallback {
    use super::{
        RingSlotHeader, TokenBucketState, BUCKET_MAGIC, BUCKET_VERSION, MAX_PAYLOAD_LEN,
        NS_PER_SEC, RING_MAGIC, RING_VERSION,
    };

    const PAR_HDR_HASH_BYTES: usize = 40;

    #[inline]
    pub fn crc32c_portable(data: &[u8]) -> u32 {
        if data.is_empty() {
            return 0;
        }
        let mut crc: u32 = 0xFFFF_FFFF;
        for &byte in data {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0x82F6_3B78 & mask);
            }
        }
        !crc
    }

    #[inline]
    fn header_prefix_bytes(hdr: &RingSlotHeader) -> [u8; PAR_HDR_HASH_BYTES] {
        let mut buf = [0u8; PAR_HDR_HASH_BYTES];
        buf[0..4].copy_from_slice(&hdr.magic.to_le_bytes());
        buf[4..6].copy_from_slice(&hdr.version.to_le_bytes());
        buf[6..8].copy_from_slice(&hdr.flags.to_le_bytes());
        buf[8..16].copy_from_slice(&hdr.sequence.to_le_bytes());
        buf[16..24].copy_from_slice(&hdr.timestamp_ns.to_le_bytes());
        buf[24..28].copy_from_slice(&hdr.agent_id.to_le_bytes());
        buf[28..32].copy_from_slice(&hdr.capability_mask.to_le_bytes());
        buf[32..36].copy_from_slice(&hdr.payload_len.to_le_bytes());
        buf[36..40].copy_from_slice(&hdr.payload_crc32c.to_le_bytes());
        buf
    }

    pub fn seal(hdr: &mut RingSlotHeader, payload: &[u8]) -> i32 {
        if ((hdr as *const RingSlotHeader as usize) & 63) != 0 {
            return -2;
        }
        if payload.len() > MAX_PAYLOAD_LEN {
            return -3;
        }
        hdr.magic = RING_MAGIC;
        hdr.version = RING_VERSION;
        hdr.payload_len = payload.len() as u32;
        hdr.payload_crc32c = crc32c_portable(payload);
        hdr.reserved = [0u8; 20];
        let prefix = header_prefix_bytes(hdr);
        hdr.header_crc32c = crc32c_portable(&prefix);
        0
    }

    pub fn validate(hdr: &RingSlotHeader, payload: &[u8]) -> i32 {
        if ((hdr as *const RingSlotHeader as usize) & 63) != 0 {
            return -2;
        }
        if hdr.magic != RING_MAGIC {
            return -4;
        }
        if hdr.version != RING_VERSION {
            return -5;
        }
        if (hdr.payload_len as usize) > MAX_PAYLOAD_LEN || (hdr.payload_len as usize) != payload.len() {
            return -3;
        }
        if hdr.reserved.iter().any(|&b| b != 0) {
            return -6;
        }
        let prefix = header_prefix_bytes(hdr);
        if crc32c_portable(&prefix) != hdr.header_crc32c {
            return -7;
        }
        if crc32c_portable(payload) != hdr.payload_crc32c {
            return -8;
        }
        0
    }

    pub fn bucket_init(
        tb: &mut TokenBucketState,
        capacity_milli: u64,
        refill_per_sec_milli: u64,
        initial_ns: u64,
    ) -> i32 {
        if ((tb as *const TokenBucketState as usize) & 63) != 0 {
            return -2;
        }
        if capacity_milli == 0 {
            return -9;
        }
        tb.magic = BUCKET_MAGIC;
        tb.version = BUCKET_VERSION;
        tb.flags = 0;
        tb.capacity_milli = capacity_milli;
        tb.tokens_milli = capacity_milli;
        tb.refill_per_sec_milli = refill_per_sec_milli;
        tb.last_refill_ns = initial_ns;
        tb.rem_ns_num = 0;
        tb.allowed_count = 0;
        tb.denied_count = 0;
        0
    }

    pub fn bucket_try_acquire(
        tb: &mut TokenBucketState,
        cost_milli: u64,
        now_ns: u64,
        retry_after_ns_out: &mut u64,
    ) -> i32 {
        if ((tb as *const TokenBucketState as usize) & 63) != 0 {
            return -2;
        }
        if tb.magic != BUCKET_MAGIC {
            return -4;
        }
        if tb.version != BUCKET_VERSION {
            return -5;
        }
        if tb.capacity_milli == 0 || cost_milli > tb.capacity_milli || tb.rem_ns_num >= NS_PER_SEC {
            return -9;
        }

        if now_ns > tb.last_refill_ns {
            let elapsed_ns = now_ns - tb.last_refill_ns;
            tb.last_refill_ns = now_ns;
            if tb.refill_per_sec_milli > 0 && tb.tokens_milli < tb.capacity_milli {
                let headroom = tb.capacity_milli - tb.tokens_milli;
                let total_num = (elapsed_ns as u128)
                    .saturating_mul(tb.refill_per_sec_milli as u128)
                    .saturating_add(tb.rem_ns_num as u128);
                let add_milli = total_num / (NS_PER_SEC as u128);
                let new_rem = (total_num % (NS_PER_SEC as u128)) as u64;
                if add_milli >= (headroom as u128) {
                    tb.tokens_milli = tb.capacity_milli;
                    tb.rem_ns_num = 0;
                } else {
                    tb.tokens_milli += add_milli as u64;
                    tb.rem_ns_num = new_rem;
                }
            } else if tb.tokens_milli >= tb.capacity_milli {
                tb.tokens_milli = tb.capacity_milli;
                tb.rem_ns_num = 0;
            }
        }

        if tb.tokens_milli >= cost_milli {
            tb.tokens_milli -= cost_milli;
            tb.allowed_count += 1;
            *retry_after_ns_out = 0;
            return 1;
        }

        tb.denied_count += 1;
        if tb.refill_per_sec_milli == 0 {
            *retry_after_ns_out = u64::MAX;
        } else {
            let deficit = (cost_milli - tb.tokens_milli) as u128;
            let num = deficit * (NS_PER_SEC as u128) - (tb.rem_ns_num as u128);
            let dt = num.div_ceil(tb.refill_per_sec_milli as u128);
            *retry_after_ns_out = u64::try_from(dt).unwrap_or(u64::MAX);
        }
        0
    }
}

/// Safe wrapper over Castagnoli CRC32C (`0x82F63B78` reflected polynomial).
#[inline]
pub fn crc32c(data: &[u8]) -> u32 {
    if data.is_empty() {
        return 0;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        // SAFETY: `data.as_ptr()` is non-null, valid for reads of `data.len()` bytes,
        // and C11 `par_crc32c` neither retains the pointer nor mutates memory.
        unsafe { raw_ffi::par_crc32c(data.as_ptr(), data.len()) }
    }
    #[cfg(target_arch = "wasm32")]
    {
        wasm_ring_fallback::crc32c_portable(data)
    }
}

/// Seals a 64-byte cache-line-aligned ring header against `payload` via C11 FFI (or WASM fallback).
#[inline]
pub fn seal_ring_header(hdr: &mut RingSlotHeader, payload: &[u8]) -> Result<(), FfiRingError> {
    #[cfg(not(target_arch = "wasm32"))]
    let rc = {
        let ptr = if payload.is_empty() {
            core::ptr::null()
        } else {
            payload.as_ptr()
        };
        // SAFETY: `hdr` is a uniquely borrowed `#[repr(C, align(64))]` struct matching C11 layout,
        // and `ptr` is valid for `payload.len()` bytes across the synchronous FFI call.
        unsafe { raw_ffi::par_ring_header_seal(hdr as *mut RingSlotHeader, ptr, payload.len()) }
    };
    #[cfg(target_arch = "wasm32")]
    let rc = wasm_ring_fallback::seal(hdr, payload);
    decode_ffi_status(rc)
}

/// Validates a 64-byte cache-line-aligned ring header and its `payload` via C11 FFI (or WASM fallback).
#[inline]
pub fn validate_ring_header(hdr: &RingSlotHeader, payload: &[u8]) -> Result<(), FfiRingError> {
    #[cfg(not(target_arch = "wasm32"))]
    let rc = {
        let ptr = if payload.is_empty() {
            core::ptr::null()
        } else {
            payload.as_ptr()
        };
        // SAFETY: `hdr` and `payload` are shared borrows valid for the duration of the pure C11 check.
        unsafe {
            raw_ffi::par_ring_header_validate(hdr as *const RingSlotHeader, ptr, payload.len())
        }
    };
    #[cfg(target_arch = "wasm32")]
    let rc = wasm_ring_fallback::validate(hdr, payload);
    decode_ffi_status(rc)
}

/// Four-cache-line (`256`-byte) telemetry slot pairing the 64B header with inline payload storage.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct RingSlot {
    pub header: RingSlotHeader,
    pub payload: [u8; MAX_PAYLOAD_LEN],
}

const _: () = {
    assert!(core::mem::size_of::<RingSlot>() == 256);
    assert!(core::mem::align_of::<RingSlot>() == 64);
};

impl Default for RingSlot {
    fn default() -> Self {
        Self {
            header: RingSlotHeader::default(),
            payload: [0u8; MAX_PAYLOAD_LEN],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TelemetryRecordView<'a> {
    pub sequence: u64,
    pub timestamp_ns: u64,
    pub agent_id: u32,
    pub capability_mask: u32,
    pub flags: u16,
    pub payload: &'a [u8],
}

/// Bounded zero-allocation ring-buffer telemetry vault backed by C11 CRC32C slot sealing.
///
/// All `N` cache-line-aligned slots are allocated once in `BoundedTelemetryVault::new()`.
/// Hot-path `record` and `read_slot` calls perform **zero heap allocations**.
pub struct BoundedTelemetryVault<const N: usize> {
    slots: Box<[RingSlot; N]>,
    write_seq: u64,
    overwritten_slots: u64,
}

impl<const N: usize> Default for BoundedTelemetryVault<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> BoundedTelemetryVault<N> {
    pub fn new() -> Self {
        assert!(N > 0, "BoundedTelemetryVault capacity N must be > 0");
        let slots_vec = vec![RingSlot::default(); N];
        let slots: Box<[RingSlot; N]> = slots_vec
            .into_boxed_slice()
            .try_into()
            .expect("exact slot slice length N");
        Self {
            slots,
            write_seq: 0,
            overwritten_slots: 0,
        }
    }

    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    #[inline]
    pub const fn total_written(&self) -> u64 {
        self.write_seq
    }

    #[inline]
    pub const fn overwritten_slots(&self) -> u64 {
        self.overwritten_slots
    }

    #[inline]
    pub fn len(&self) -> usize {
        (self.write_seq as usize).min(N)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.write_seq == 0
    }

    /// Appends an event into the next ring slot with zero heap allocations.
    #[inline]
    pub fn record(
        &mut self,
        timestamp_ns: u64,
        agent_id: u32,
        capability_mask: u32,
        flags: u16,
        payload: &[u8],
    ) -> Result<u64, FfiRingError> {
        if payload.len() > MAX_PAYLOAD_LEN {
            return Err(FfiRingError::PayloadOutOfBounds);
        }
        let seq = self.write_seq + 1;
        let idx = (self.write_seq as usize) % N;
        if self.write_seq >= N as u64 {
            self.overwritten_slots += 1;
        }

        let slot = &mut self.slots[idx];
        slot.payload[..payload.len()].copy_from_slice(payload);
        if payload.len() < MAX_PAYLOAD_LEN {
            slot.payload[payload.len()..].fill(0);
        }
        slot.header.flags = flags;
        slot.header.sequence = seq;
        slot.header.timestamp_ns = timestamp_ns;
        slot.header.agent_id = agent_id;
        slot.header.capability_mask = capability_mask;

        seal_ring_header(&mut slot.header, payload)?;
        self.write_seq = seq;
        Ok(seq)
    }

    /// Reads and C11-validates a live slot by physical index `0..self.len()`.
    #[inline]
    pub fn read_slot(&self, slot_index: usize) -> Result<TelemetryRecordView<'_>, FfiRingError> {
        if slot_index >= self.len() {
            return Err(FfiRingError::PayloadOutOfBounds);
        }
        let slot = &self.slots[slot_index];
        let payload_len = slot.header.payload_len as usize;
        if payload_len > MAX_PAYLOAD_LEN {
            return Err(FfiRingError::PayloadOutOfBounds);
        }
        let slice = &slot.payload[..payload_len];
        validate_ring_header(&slot.header, slice)?;
        Ok(TelemetryRecordView {
            sequence: slot.header.sequence,
            timestamp_ns: slot.header.timestamp_ns,
            agent_id: slot.header.agent_id,
            capability_mask: slot.header.capability_mask,
            flags: slot.header.flags,
            payload: slice,
        })
    }

    /// Audits every populated slot via C11 CRC32C header + payload verification.
    pub fn verify_all(&self) -> Result<usize, FfiRingError> {
        let populated = self.len();
        for idx in 0..populated {
            let _ = self.read_slot(idx)?;
        }
        Ok(populated)
    }

    #[cfg(test)]
    pub(crate) fn raw_slot_mut(&mut self, slot_index: usize) -> &mut RingSlot {
        &mut self.slots[slot_index]
    }
}

/// Default `64`-slot (`16,384`-byte) cache-line-aligned capability & rate-limit audit ring.
pub type CapabilityAuditRing = BoundedTelemetryVault<DEFAULT_AUDIT_RING_SLOTS>;

/// Audit decision code stored in `RingSlotHeader::flags` inside [`BoundedTelemetryVault`].
#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditDecision {
    Allowed = 1,
    DeniedCapability = 2,
    DeniedRateLimit = 3,
    DeniedCircuitOpen = 4,
}

// ============================================================================
// Deterministic Zero-Float-Drift Token-Bucket Rate Limiter (C11 / Rust / WASM)
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateLimitDecision {
    Allowed {
        remaining_milli: u64,
    },
    Denied {
        deficit_milli: u64,
        retry_after_ns: u64,
    },
}

/// Cache-line-aligned (`64`-byte) deterministic integer token-bucket rate limiter.
///
/// Tracks capacity and refill rates in exact milli-tokens (`1_000 == 1` token) with
/// sub-nanosecond rational remainder conservation (`rem_ns_num < 1_000_000_000`),
/// eliminating IEEE-754 floating-point drift across arbitrary tick frequencies.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeterministicTokenBucket {
    state: TokenBucketState,
}

const _: () = {
    assert!(core::mem::size_of::<DeterministicTokenBucket>() == 64);
    assert!(core::mem::align_of::<DeterministicTokenBucket>() == 64);
};

impl DeterministicTokenBucket {
    pub fn new(
        capacity_milli: u64,
        refill_per_sec_milli: u64,
        initial_ns: u64,
    ) -> Result<Self, FfiRingError> {
        let mut bucket = Self {
            state: TokenBucketState::default(),
        };
        #[cfg(not(target_arch = "wasm32"))]
        let rc = unsafe {
            // SAFETY: `bucket.state` is a uniquely owned `#[repr(C, align(64))]` struct matching C11 `ParTokenBucket`.
            raw_ffi::par_token_bucket_init(
                &mut bucket.state as *mut TokenBucketState,
                capacity_milli,
                refill_per_sec_milli,
                initial_ns,
            )
        };
        #[cfg(target_arch = "wasm32")]
        let rc = wasm_ring_fallback::bucket_init(
            &mut bucket.state,
            capacity_milli,
            refill_per_sec_milli,
            initial_ns,
        );
        decode_ffi_status(rc)?;
        Ok(bucket)
    }

    #[inline]
    pub const fn state(&self) -> &TokenBucketState {
        &self.state
    }

    #[inline]
    pub const fn tokens_milli(&self) -> u64 {
        self.state.tokens_milli
    }

    #[inline]
    pub const fn allowed_count(&self) -> u64 {
        self.state.allowed_count
    }

    #[inline]
    pub const fn denied_count(&self) -> u64 {
        self.state.denied_count
    }

    /// Computes exact ceiling nanoseconds until `cost_milli` tokens are available from current state without mutating counters.
    #[inline]
    pub fn retry_after_ns(&self, cost_milli: u64) -> u64 {
        if cost_milli > self.state.capacity_milli {
            return u64::MAX;
        }
        if self.state.tokens_milli >= cost_milli {
            return 0;
        }
        if self.state.refill_per_sec_milli == 0 {
            return u64::MAX;
        }
        let deficit = (cost_milli - self.state.tokens_milli) as u128;
        let num = deficit * (NS_PER_SEC as u128) - (self.state.rem_ns_num as u128);
        let dt = num.div_ceil(self.state.refill_per_sec_milli as u128);
        u64::try_from(dt).unwrap_or(u64::MAX)
    }

    /// Attempts to acquire `cost_milli` milli-tokens at monotonic timestamp `now_ns` in `O(1)` time and `0` heap allocations.
    #[inline]
    pub fn try_acquire(
        &mut self,
        cost_milli: u64,
        now_ns: u64,
    ) -> Result<RateLimitDecision, FfiRingError> {
        let mut retry_ns = 0u64;
        #[cfg(not(target_arch = "wasm32"))]
        let rc = unsafe {
            // SAFETY: `self.state` and `retry_ns` are uniquely borrowed stack/struct memory valid across synchronous C11 call.
            raw_ffi::par_token_bucket_try_acquire(
                &mut self.state as *mut TokenBucketState,
                cost_milli,
                now_ns,
                &mut retry_ns as *mut u64,
            )
        };
        #[cfg(target_arch = "wasm32")]
        let rc = wasm_ring_fallback::bucket_try_acquire(
            &mut self.state,
            cost_milli,
            now_ns,
            &mut retry_ns,
        );

        match rc {
            1 => Ok(RateLimitDecision::Allowed {
                remaining_milli: self.state.tokens_milli,
            }),
            0 => Ok(RateLimitDecision::Denied {
                deficit_milli: cost_milli.saturating_sub(self.state.tokens_milli),
                retry_after_ns: retry_ns,
            }),
            err_code => {
                decode_ffi_status(err_code)?;
                Err(FfiRingError::UnknownCode(err_code))
            }
        }
    }
}

// ============================================================================
// Capability-Fenced DAG Task Orchestrator
// ============================================================================

pub mod capabilities {
    pub const NONE: u32 = 0;
    pub const FS_READ: u32 = 1 << 0;
    pub const KV_WRITE: u32 = 1 << 1;
    pub const NET_HTTP: u32 = 1 << 2;
    pub const TOOL_EXEC: u32 = 1 << 3;
    pub const GPU_DISPATCH: u32 = 1 << 4;
    pub const TELEMETRY_EMIT: u32 = 1 << 5;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub base_backoff_us: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 2,
            base_backoff_us: 50,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BreakerState {
    Closed,
    Open { opened_at_tick: u64 },
    HalfOpen,
}

#[derive(Clone, Debug)]
pub struct CircuitBreaker {
    failure_threshold: u32,
    cooldown_ticks: u64,
    consecutive_failures: u32,
    state: BreakerState,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, cooldown_ticks: u64) -> Self {
        assert!(failure_threshold > 0, "failure_threshold must be > 0");
        Self {
            failure_threshold,
            cooldown_ticks,
            consecutive_failures: 0,
            state: BreakerState::Closed,
        }
    }

    pub fn state(&self) -> BreakerState {
        self.state
    }

    pub fn allow(&mut self, now_tick: u64) -> bool {
        match self.state {
            BreakerState::Closed | BreakerState::HalfOpen => true,
            BreakerState::Open { opened_at_tick } => {
                if now_tick.saturating_sub(opened_at_tick) >= self.cooldown_ticks {
                    self.state = BreakerState::HalfOpen;
                    true
                } else {
                    false
                }
            }
        }
    }

    pub fn on_success(&mut self) {
        self.consecutive_failures = 0;
        self.state = BreakerState::Closed;
    }

    pub fn on_failure(&mut self, now_tick: u64) {
        self.consecutive_failures += 1;
        match self.state {
            BreakerState::HalfOpen => {
                self.state = BreakerState::Open {
                    opened_at_tick: now_tick,
                };
            }
            BreakerState::Closed if self.consecutive_failures >= self.failure_threshold => {
                self.state = BreakerState::Open {
                    opened_at_tick: now_tick,
                };
            }
            _ => {}
        }
    }
}

pub type TaskHandler = Box<dyn FnMut(&BTreeMap<String, i64>, u32) -> Result<i64, String>>;

pub struct DagNode {
    pub id: String,
    pub deps: Vec<String>,
    pub required_caps: u32,
    pub retry: RetryPolicy,
    pub handler: TaskHandler,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrchestratorError {
    DuplicateTask(String),
    MissingDependency { task: String, missing: String },
    CycleDetected(Vec<String>),
    CapabilityDenied {
        task: String,
        required: u32,
        granted: u32,
    },
    RateLimited {
        task: String,
        cost_milli: u64,
        retry_after_ns: u64,
    },
    CircuitOpen { task: String },
    TaskExhausted {
        task: String,
        attempts: u32,
        last_error: String,
    },
}

impl fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for OrchestratorError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DagExecutionSummary {
    pub topological_order: Vec<String>,
    pub outputs: BTreeMap<String, i64>,
    pub total_attempts: u32,
    pub simulated_backoff_us: u64,
}

pub struct CapabilityDagOrchestrator {
    granted_caps: u32,
    breaker: CircuitBreaker,
    nodes: BTreeMap<String, DagNode>,
}

impl CapabilityDagOrchestrator {
    pub fn new(granted_caps: u32, breaker: CircuitBreaker) -> Self {
        Self {
            granted_caps,
            breaker,
            nodes: BTreeMap::new(),
        }
    }

    pub fn breaker(&self) -> &CircuitBreaker {
        &self.breaker
    }

    pub fn add_task(&mut self, node: DagNode) -> Result<(), OrchestratorError> {
        if self.nodes.contains_key(&node.id) {
            return Err(OrchestratorError::DuplicateTask(node.id));
        }
        self.nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Computes a deterministic lexicographical Kahn topological order or detects cycles.
    pub fn topological_sort(&self) -> Result<Vec<String>, OrchestratorError> {
        let mut indegree: BTreeMap<String, usize> = BTreeMap::new();
        let mut children: BTreeMap<String, Vec<String>> = BTreeMap::new();

        for id in self.nodes.keys() {
            indegree.insert(id.clone(), 0);
            children.insert(id.clone(), Vec::new());
        }

        for (id, node) in &self.nodes {
            let mut unique_deps = BTreeSet::new();
            for dep in &node.deps {
                if !self.nodes.contains_key(dep) {
                    return Err(OrchestratorError::MissingDependency {
                        task: id.clone(),
                        missing: dep.clone(),
                    });
                }
                if unique_deps.insert(dep.clone()) {
                    *indegree.get_mut(id).expect("node registered") += 1;
                    children
                        .get_mut(dep)
                        .expect("dep registered")
                        .push(id.clone());
                }
            }
        }

        for list in children.values_mut() {
            list.sort();
        }

        // BTreeSet preserves deterministic lexicographical tie-breaking for ready nodes.
        let mut ready: BTreeSet<String> = indegree
            .iter()
            .filter_map(|(k, &deg)| if deg == 0 { Some(k.clone()) } else { None })
            .collect();

        let mut order = Vec::with_capacity(self.nodes.len());
        let mut queue = VecDeque::new();
        while let Some(first) = ready.pop_first() {
            queue.push_back(first);
            while let Some(curr) = queue.pop_front() {
                order.push(curr.clone());
                if let Some(next_list) = children.get(&curr) {
                    for nxt in next_list {
                        let deg = indegree.get_mut(nxt).expect("child registered");
                        *deg -= 1;
                        if *deg == 0 {
                            ready.insert(nxt.clone());
                        }
                    }
                }
                if let Some(next_lex) = ready.pop_first() {
                    queue.push_back(next_lex);
                }
            }
        }

        if order.len() != self.nodes.len() {
            let unresolved: Vec<String> = indegree
                .into_iter()
                .filter_map(|(k, deg)| if deg > 0 { Some(k) } else { None })
                .collect();
            return Err(OrchestratorError::CycleDetected(unresolved));
        }

        Ok(order)
    }

    /// Validates capability fences prior to execution, then runs tasks in topological order.
    pub fn execute(&mut self, now_tick: u64) -> Result<DagExecutionSummary, OrchestratorError> {
        let order = self.topological_sort()?;

        // Preflight capability fence across the entire plan before mutating state.
        for task_id in &order {
            let node = self.nodes.get(task_id).expect("sorted node exists");
            if (node.required_caps & self.granted_caps) != node.required_caps {
                return Err(OrchestratorError::CapabilityDenied {
                    task: task_id.clone(),
                    required: node.required_caps,
                    granted: self.granted_caps,
                });
            }
        }

        let mut outputs = BTreeMap::new();
        let mut total_attempts = 0u32;
        let mut simulated_backoff_us = 0u64;

        for task_id in &order {
            if !self.breaker.allow(now_tick) {
                return Err(OrchestratorError::CircuitOpen {
                    task: task_id.clone(),
                });
            }

            let node = self.nodes.get_mut(task_id).expect("sorted node exists");
            let max_attempts = node.retry.max_attempts.max(1);
            let mut attempt = 0u32;
            let mut last_err = String::new();
            let mut succeeded = false;

            while attempt < max_attempts {
                if attempt > 0 && !self.breaker.allow(now_tick) {
                    return Err(OrchestratorError::CircuitOpen {
                        task: task_id.clone(),
                    });
                }
                attempt += 1;
                total_attempts += 1;

                match (node.handler)(&outputs, attempt) {
                    Ok(val) => {
                        self.breaker.on_success();
                        outputs.insert(task_id.clone(), val);
                        succeeded = true;
                        break;
                    }
                    Err(err_msg) => {
                        self.breaker.on_failure(now_tick);
                        last_err = err_msg;
                        if attempt < max_attempts {
                            simulated_backoff_us +=
                                node.retry.base_backoff_us * (1u64 << (attempt - 1).min(10));
                        }
                    }
                }
            }

            if !succeeded {
                return Err(OrchestratorError::TaskExhausted {
                    task: task_id.clone(),
                    attempts: attempt,
                    last_error: last_err,
                });
            }
        }

        Ok(DagExecutionSummary {
            topological_order: order,
            outputs,
            total_attempts,
            simulated_backoff_us,
        })
    }

    /// Executes the DAG while enforcing an optional deterministic token-bucket rate limiter
    /// and sealing every capability/rate/circuit decision into a C11/WASM CRC32C `BoundedTelemetryVault`.
    pub fn execute_audited<const N: usize>(
        &mut self,
        ctx: AuditExecutionContext,
        mut bucket: Option<&mut DeterministicTokenBucket>,
        audit_ring: &mut BoundedTelemetryVault<N>,
    ) -> Result<DagExecutionSummary, OrchestratorError> {
        let order = self.topological_sort()?;

        // Preflight capability fence across the entire plan and seal denial into the audit ring if rejected.
        for task_id in &order {
            let node = self.nodes.get(task_id).expect("sorted node exists");
            if (node.required_caps & self.granted_caps) != node.required_caps {
                let mut frame = [0u8; MAX_PAYLOAD_LEN];
                let n = write_audit_payload(&mut frame, "deny:cap:", task_id);
                let _ = audit_ring.record(
                    ctx.now_ns,
                    ctx.agent_id,
                    node.required_caps & self.granted_caps,
                    AuditDecision::DeniedCapability as u16,
                    &frame[..n],
                );
                return Err(OrchestratorError::CapabilityDenied {
                    task: task_id.clone(),
                    required: node.required_caps,
                    granted: self.granted_caps,
                });
            }
        }

        let mut outputs = BTreeMap::new();
        let mut total_attempts = 0u32;
        let mut simulated_backoff_us = 0u64;

        for (step_idx, task_id) in order.iter().enumerate() {
            let step_ns = ctx.now_ns.saturating_add(step_idx as u64);
            if !self.breaker.allow(ctx.now_tick) {
                let mut frame = [0u8; MAX_PAYLOAD_LEN];
                let n = write_audit_payload(&mut frame, "deny:circuit:", task_id);
                let _ = audit_ring.record(
                    step_ns,
                    ctx.agent_id,
                    self.granted_caps,
                    AuditDecision::DeniedCircuitOpen as u16,
                    &frame[..n],
                );
                return Err(OrchestratorError::CircuitOpen {
                    task: task_id.clone(),
                });
            }

            if let Some(tb) = bucket.as_deref_mut() {
                match tb.try_acquire(ctx.cost_per_task_milli, step_ns) {
                    Ok(RateLimitDecision::Allowed { .. }) => {}
                    Ok(RateLimitDecision::Denied { retry_after_ns, .. }) => {
                        let mut frame = [0u8; MAX_PAYLOAD_LEN];
                        let n = write_audit_payload(&mut frame, "deny:rate:", task_id);
                        let _ = audit_ring.record(
                            step_ns,
                            ctx.agent_id,
                            self.granted_caps,
                            AuditDecision::DeniedRateLimit as u16,
                            &frame[..n],
                        );
                        return Err(OrchestratorError::RateLimited {
                            task: task_id.clone(),
                            cost_milli: ctx.cost_per_task_milli,
                            retry_after_ns,
                        });
                    }
                    Err(_) => {
                        return Err(OrchestratorError::RateLimited {
                            task: task_id.clone(),
                            cost_milli: ctx.cost_per_task_milli,
                            retry_after_ns: u64::MAX,
                        });
                    }
                }
            }

            let node = self.nodes.get_mut(task_id).expect("sorted node exists");
            let req_caps = node.required_caps;
            let max_attempts = node.retry.max_attempts.max(1);
            let mut attempt = 0u32;
            let mut last_err = String::new();
            let mut succeeded = false;

            while attempt < max_attempts {
                if attempt > 0 && !self.breaker.allow(ctx.now_tick) {
                    return Err(OrchestratorError::CircuitOpen {
                        task: task_id.clone(),
                    });
                }
                attempt += 1;
                total_attempts += 1;

                match (node.handler)(&outputs, attempt) {
                    Ok(val) => {
                        self.breaker.on_success();
                        outputs.insert(task_id.clone(), val);
                        let mut frame = [0u8; MAX_PAYLOAD_LEN];
                        let n = write_audit_payload(&mut frame, "allow:ok:", task_id);
                        let _ = audit_ring.record(
                            step_ns,
                            ctx.agent_id,
                            req_caps,
                            AuditDecision::Allowed as u16,
                            &frame[..n],
                        );
                        succeeded = true;
                        break;
                    }
                    Err(err_msg) => {
                        self.breaker.on_failure(ctx.now_tick);
                        last_err = err_msg;
                        if attempt < max_attempts {
                            simulated_backoff_us +=
                                node.retry.base_backoff_us * (1u64 << (attempt - 1).min(10));
                        }
                    }
                }
            }

            if !succeeded {
                return Err(OrchestratorError::TaskExhausted {
                    task: task_id.clone(),
                    attempts: attempt,
                    last_error: last_err,
                });
            }
        }

        Ok(DagExecutionSummary {
            topological_order: order,
            outputs,
            total_attempts,
            simulated_backoff_us,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AuditExecutionContext {
    pub now_tick: u64,
    pub now_ns: u64,
    pub agent_id: u32,
    pub cost_per_task_milli: u64,
}

#[inline]
fn write_audit_payload(dst: &mut [u8; MAX_PAYLOAD_LEN], prefix: &str, task_id: &str) -> usize {
    let p = prefix.as_bytes();
    let p_len = p.len().min(MAX_PAYLOAD_LEN);
    dst[..p_len].copy_from_slice(&p[..p_len]);
    let rem = MAX_PAYLOAD_LEN - p_len;
    let t = task_id.as_bytes();
    let t_len = t.len().min(rem);
    dst[p_len..(p_len + t_len)].copy_from_slice(&t[..t_len]);
    p_len + t_len
}

// ============================================================================
// Streaming Jain-Chlamtac P² Quantile Estimator (O(1) Memory)
// ============================================================================

/// Single-quantile Jain & Chlamtac (1985) piecewise-parabolic (`P²`) streaming estimator.
///
/// Uses exactly 5 markers (`O(1)` memory, zero heap allocation).
#[derive(Clone, Copy, Debug)]
pub struct P2Quantile {
    p: f64,
    count: u64,
    q: [f64; 5],
    n: [isize; 5],
    np: [f64; 5],
    dn: [f64; 5],
}

impl P2Quantile {
    pub fn new(p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p) && p > 0.0 && p < 1.0,
            "quantile p must lie strictly inside (0.0, 1.0)"
        );
        Self {
            p,
            count: 0,
            q: [0.0; 5],
            n: [1, 2, 3, 4, 5],
            np: [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0],
            dn: [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
        }
    }

    #[inline]
    pub const fn count(&self) -> u64 {
        self.count
    }

    #[inline]
    pub fn observe(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        let idx = self.count as usize;
        if idx < 5 {
            self.q[idx] = x;
            self.count += 1;
            if self.count == 5 {
                // Insertion sort on 5 elements (zero allocation).
                for i in 1..5 {
                    let key = self.q[i];
                    let mut j = i;
                    while j > 0 && self.q[j - 1] > key {
                        self.q[j] = self.q[j - 1];
                        j -= 1;
                    }
                    self.q[j] = key;
                }
            }
            return;
        }

        self.count += 1;
        let k = if x < self.q[0] {
            self.q[0] = x;
            0
        } else if x < self.q[1] {
            0
        } else if x < self.q[2] {
            1
        } else if x < self.q[3] {
            2
        } else if x <= self.q[4] {
            3
        } else {
            self.q[4] = x;
            3
        };

        for pos in &mut self.n[(k + 1)..5] {
            *pos += 1;
        }
        for (np_val, &dn_val) in self.np.iter_mut().zip(self.dn.iter()) {
            *np_val += dn_val;
        }

        for i in 1..=3 {
            let d = self.np[i] - (self.n[i] as f64);
            if (d >= 1.0 && self.n[i + 1] - self.n[i] > 1)
                || (d <= -1.0 && self.n[i - 1] - self.n[i] < -1)
            {
                let d_sign_i: isize = if d >= 0.0 { 1 } else { -1 };
                let d_sign = d_sign_i as f64;

                let n_i = self.n[i] as f64;
                let n_im1 = self.n[i - 1] as f64;
                let n_ip1 = self.n[i + 1] as f64;

                let term1 = (n_i - n_im1 + d_sign) * (self.q[i + 1] - self.q[i]) / (n_ip1 - n_i);
                let term2 = (n_ip1 - n_i - d_sign) * (self.q[i] - self.q[i - 1]) / (n_i - n_im1);
                let qp = self.q[i] + (d_sign / (n_ip1 - n_im1)) * (term1 + term2);

                if self.q[i - 1] < qp && qp < self.q[i + 1] {
                    self.q[i] = qp;
                } else {
                    let adj = (i as isize + d_sign_i) as usize;
                    self.q[i] += d_sign * (self.q[adj] - self.q[i]) / ((self.n[adj] - self.n[i]) as f64);
                }
                self.n[i] += d_sign_i;
            }
        }
    }

    #[inline]
    pub fn estimate(&self) -> f64 {
        match self.count {
            0 => 0.0,
            n @ 1..=4 => {
                let len = n as usize;
                let mut tmp = [0.0; 4];
                tmp[..len].copy_from_slice(&self.q[..len]);
                for i in 1..len {
                    let key = tmp[i];
                    let mut j = i;
                    while j > 0 && tmp[j - 1] > key {
                        tmp[j] = tmp[j - 1];
                        j -= 1;
                    }
                    tmp[j] = key;
                }
                let rank = self.p * ((len - 1) as f64);
                let lo = rank.floor() as usize;
                let hi = rank.ceil() as usize;
                if lo == hi {
                    tmp[lo]
                } else {
                    tmp[lo] + (rank - lo as f64) * (tmp[hi] - tmp[lo])
                }
            }
            _ => self.q[2],
        }
    }
}

/// Multi-quantile `O(1)` latency summary tracking `p50`, `p95`, `p99`, `min`, `max`, and Welford `mean`.
#[derive(Clone, Copy, Debug)]
pub struct StreamingQuantileSuite {
    p50: P2Quantile,
    p95: P2Quantile,
    p99: P2Quantile,
    count: u64,
    mean: f64,
    min: f64,
    max: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantileSnapshot {
    pub count: u64,
    pub min: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub mean: f64,
    pub state_bytes: usize,
}

impl Default for StreamingQuantileSuite {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingQuantileSuite {
    pub fn new() -> Self {
        Self {
            p50: P2Quantile::new(0.50),
            p95: P2Quantile::new(0.95),
            p99: P2Quantile::new(0.99),
            count: 0,
            mean: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    #[inline]
    pub fn observe(&mut self, sample: f64) {
        if !sample.is_finite() {
            return;
        }
        self.count += 1;
        let delta = sample - self.mean;
        self.mean += delta / (self.count as f64);
        if sample < self.min {
            self.min = sample;
        }
        if sample > self.max {
            self.max = sample;
        }
        self.p50.observe(sample);
        self.p95.observe(sample);
        self.p99.observe(sample);
    }

    pub fn snapshot(&self) -> QuantileSnapshot {
        if self.count == 0 {
            return QuantileSnapshot {
                count: 0,
                min: 0.0,
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
                max: 0.0,
                mean: 0.0,
                state_bytes: core::mem::size_of::<Self>(),
            };
        }
        QuantileSnapshot {
            count: self.count,
            min: self.min,
            p50: self.p50.estimate(),
            p95: self.p95.estimate(),
            p99: self.p99.estimate(),
            max: self.max,
            mean: self.mean,
            state_bytes: core::mem::size_of::<Self>(),
        }
    }
}


// ============================================================================
// WebAssembly (`wasm32-unknown-unknown`) & C-ABI Streaming Quantile Exports
// ============================================================================

/// Returns the exact constant byte footprint (`560` bytes) of `StreamingQuantileSuite`.
#[no_mangle]
pub extern "C" fn par_p2_suite_state_bytes() -> usize {
    core::mem::size_of::<StreamingQuantileSuite>()
}

/// Allocates a new constant-memory (`560` B) `StreamingQuantileSuite` on heap / WASM linear memory.
///
/// Caller owns the returned pointer and must release it with [`par_p2_suite_free`].
#[no_mangle]
pub extern "C" fn par_p2_suite_new() -> *mut StreamingQuantileSuite {
    Box::into_raw(Box::new(StreamingQuantileSuite::new()))
}

/// Releases a `StreamingQuantileSuite` previously allocated by [`par_p2_suite_new`].
///
/// # Safety
/// `suite` must either be null (no-op) or a valid pointer returned by [`par_p2_suite_new`]
/// that has not yet been freed and is not aliased by any active reference.
#[no_mangle]
pub unsafe extern "C" fn par_p2_suite_free(suite: *mut StreamingQuantileSuite) {
    if !suite.is_null() {
        drop(unsafe { Box::from_raw(suite) });
    }
}

/// Resets an existing `StreamingQuantileSuite` in place with zero heap allocations.
///
/// # Safety
/// `suite` must be null (no-op) or a valid, uniquely writable pointer from [`par_p2_suite_new`].
#[no_mangle]
pub unsafe extern "C" fn par_p2_suite_reset(suite: *mut StreamingQuantileSuite) {
    if let Some(s) = unsafe { suite.as_mut() } {
        *s = StreamingQuantileSuite::new();
    }
}

/// Observes a single finite `f64` latency sample in `O(1)` time and `0` allocations.
///
/// # Safety
/// `suite` must be null (no-op) or a valid, uniquely writable pointer from [`par_p2_suite_new`].
#[no_mangle]
pub unsafe extern "C" fn par_p2_suite_observe(suite: *mut StreamingQuantileSuite, sample: f64) {
    if let Some(s) = unsafe { suite.as_mut() } {
        s.observe(sample);
    }
}

/// Batch-ingests `len` contiguous `f64` samples from WASM linear memory (`Float64Array`)
/// in a single FFI/WASM call and returns the updated total observation count.
///
/// # Safety
/// - `suite` must be null (returns `0`) or a valid, uniquely writable pointer from [`par_p2_suite_new`].
/// - When `len > 0`, `samples_ptr` must be non-null, properly `f64`-aligned, and valid for reads
///   of `len` contiguous `f64` elements without aliasing `suite`.
#[no_mangle]
pub unsafe extern "C" fn par_p2_suite_observe_batch(
    suite: *mut StreamingQuantileSuite,
    samples_ptr: *const f64,
    len: usize,
) -> u64 {
    let Some(s) = (unsafe { suite.as_mut() }) else {
        return 0;
    };
    if len == 0 || samples_ptr.is_null() {
        return s.snapshot().count;
    }
    let slice = unsafe { core::slice::from_raw_parts(samples_ptr, len) };
    for &sample in slice {
        s.observe(sample);
    }
    s.snapshot().count
}

/// Allocates a contiguous `f64` scratch buffer in WASM linear memory for zero-copy JS `Float64Array` writes.
#[no_mangle]
pub extern "C" fn par_p2_alloc_f64_buffer(len: usize) -> *mut f64 {
    if len == 0 {
        return core::ptr::null_mut();
    }
    let mut buf = vec![0.0_f64; len].into_boxed_slice();
    let ptr = buf.as_mut_ptr();
    core::mem::forget(buf);
    ptr
}

/// Frees a contiguous `f64` scratch buffer previously allocated by [`par_p2_alloc_f64_buffer`].
///
/// # Safety
/// `ptr` must either be null (no-op) or a pointer returned by [`par_p2_alloc_f64_buffer`]
/// with the exact same `len`, and must not be used after this call.
#[no_mangle]
pub unsafe extern "C" fn par_p2_free_f64_buffer(ptr: *mut f64, len: usize) {
    if !ptr.is_null() && len > 0 {
        let raw_slice = core::ptr::slice_from_raw_parts_mut(ptr, len);
        drop(unsafe { Box::from_raw(raw_slice) });
    }
}

/// Allocates a contiguous `u8` scratch buffer in WASM linear memory for checksum/frame verification.
#[no_mangle]
pub extern "C" fn par_wasm_alloc_bytes(len: usize) -> *mut u8 {
    if len == 0 {
        return core::ptr::null_mut();
    }
    let mut buf = vec![0_u8; len].into_boxed_slice();
    let ptr = buf.as_mut_ptr();
    core::mem::forget(buf);
    ptr
}

/// Frees a byte buffer previously allocated by [`par_wasm_alloc_bytes`].
///
/// # Safety
/// `ptr` must either be null (no-op) or a pointer returned by [`par_wasm_alloc_bytes`] with the exact `len`.
#[no_mangle]
pub unsafe extern "C" fn par_wasm_free_bytes(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        let raw_slice = core::ptr::slice_from_raw_parts_mut(ptr, len);
        drop(unsafe { Box::from_raw(raw_slice) });
    }
}

/// Returns total observations ingested into `suite` (`0` if `suite` is null).
///
/// # Safety
/// `suite` must be null or a valid shared pointer to a live `StreamingQuantileSuite`.
#[no_mangle]
pub unsafe extern "C" fn par_p2_suite_count(suite: *const StreamingQuantileSuite) -> u64 {
    unsafe { suite.as_ref() }.map_or(0, |s| s.snapshot().count)
}

/// Returns minimum observed sample (`0.0` if empty or null).
///
/// # Safety
/// `suite` must be null or a valid shared pointer to a live `StreamingQuantileSuite`.
#[no_mangle]
pub unsafe extern "C" fn par_p2_suite_min(suite: *const StreamingQuantileSuite) -> f64 {
    unsafe { suite.as_ref() }.map_or(0.0, |s| s.snapshot().min)
}

/// Returns streaming Jain-Chlamtac `P²` median (`p50`) estimate (`0.0` if empty or null).
///
/// # Safety
/// `suite` must be null or a valid shared pointer to a live `StreamingQuantileSuite`.
#[no_mangle]
pub unsafe extern "C" fn par_p2_suite_p50(suite: *const StreamingQuantileSuite) -> f64 {
    unsafe { suite.as_ref() }.map_or(0.0, |s| s.snapshot().p50)
}

/// Returns streaming Jain-Chlamtac `P²` `p95` tail-latency estimate (`0.0` if empty or null).
///
/// # Safety
/// `suite` must be null or a valid shared pointer to a live `StreamingQuantileSuite`.
#[no_mangle]
pub unsafe extern "C" fn par_p2_suite_p95(suite: *const StreamingQuantileSuite) -> f64 {
    unsafe { suite.as_ref() }.map_or(0.0, |s| s.snapshot().p95)
}

/// Returns streaming Jain-Chlamtac `P²` `p99` tail-latency estimate (`0.0` if empty or null).
///
/// # Safety
/// `suite` must be null or a valid shared pointer to a live `StreamingQuantileSuite`.
#[no_mangle]
pub unsafe extern "C" fn par_p2_suite_p99(suite: *const StreamingQuantileSuite) -> f64 {
    unsafe { suite.as_ref() }.map_or(0.0, |s| s.snapshot().p99)
}

/// Returns maximum observed sample (`0.0` if empty or null).
///
/// # Safety
/// `suite` must be null or a valid shared pointer to a live `StreamingQuantileSuite`.
#[no_mangle]
pub unsafe extern "C" fn par_p2_suite_max(suite: *const StreamingQuantileSuite) -> f64 {
    unsafe { suite.as_ref() }.map_or(0.0, |s| s.snapshot().max)
}

/// Returns Welford streaming arithmetic mean (`0.0` if empty or null).
///
/// # Safety
/// `suite` must be null or a valid shared pointer to a live `StreamingQuantileSuite`.
#[no_mangle]
pub unsafe extern "C" fn par_p2_suite_mean(suite: *const StreamingQuantileSuite) -> f64 {
    unsafe { suite.as_ref() }.map_or(0.0, |s| s.snapshot().mean)
}

/// Computes RFC 3720 Castagnoli CRC32C over `len` bytes in WASM / C-ABI linear memory.
///
/// # Safety
/// When `len > 0`, `data_ptr` must be non-null and valid for reads of `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn par_wasm_crc32c(data_ptr: *const u8, len: usize) -> u32 {
    if len == 0 || data_ptr.is_null() {
        return 0;
    }
    let bytes = unsafe { core::slice::from_raw_parts(data_ptr, len) };
    crc32c(bytes)
}

// ============================================================================
// C-ABI & WebAssembly Token-Bucket Rate Limiter + Capability Audit Ring Exports
// ============================================================================

/// Returns the exact constant byte footprint (`64` bytes = `1` cache line) of [`DeterministicTokenBucket`].
#[no_mangle]
pub extern "C" fn par_bucket_state_bytes() -> usize {
    core::mem::size_of::<DeterministicTokenBucket>()
}

/// Allocates a new `64`-byte cache-line-aligned [`DeterministicTokenBucket`] in heap / WASM linear memory.
///
/// Returns null if `capacity_milli == 0`. Caller must release with [`par_bucket_free`].
#[no_mangle]
pub extern "C" fn par_bucket_new(
    capacity_milli: u64,
    refill_per_sec_milli: u64,
    initial_ns: u64,
) -> *mut DeterministicTokenBucket {
    match DeterministicTokenBucket::new(capacity_milli, refill_per_sec_milli, initial_ns) {
        Ok(b) => Box::into_raw(Box::new(b)),
        Err(_) => core::ptr::null_mut(),
    }
}

/// Releases a [`DeterministicTokenBucket`] previously allocated by [`par_bucket_new`].
///
/// # Safety
/// `bucket` must either be null (no-op) or a valid unaliased pointer returned by [`par_bucket_new`].
#[no_mangle]
pub unsafe extern "C" fn par_bucket_free(bucket: *mut DeterministicTokenBucket) {
    if !bucket.is_null() {
        drop(unsafe { Box::from_raw(bucket) });
    }
}

/// Attempts to acquire `cost_milli` milli-tokens at `now_ns`.
///
/// Returns `1` if allowed, `0` if rate-limited, or a negative `PAR_ERR_*` code on invalid input.
///
/// # Safety
/// `bucket` must be null (returns `-1`) or a valid uniquely writable pointer from [`par_bucket_new`].
#[no_mangle]
pub unsafe extern "C" fn par_bucket_try_acquire(
    bucket: *mut DeterministicTokenBucket,
    cost_milli: u64,
    now_ns: u64,
) -> i32 {
    let Some(tb) = (unsafe { bucket.as_mut() }) else {
        return -1;
    };
    match tb.try_acquire(cost_milli, now_ns) {
        Ok(RateLimitDecision::Allowed { .. }) => 1,
        Ok(RateLimitDecision::Denied { .. }) => 0,
        Err(err) => encode_ffi_error(err),
    }
}

/// Returns exact ceiling nanoseconds until `cost_milli` milli-tokens become available from current state (`0` if immediately available or null).
///
/// # Safety
/// `bucket` must be null or a valid shared pointer to a live [`DeterministicTokenBucket`].
#[no_mangle]
pub unsafe extern "C" fn par_bucket_retry_after_ns(
    bucket: *const DeterministicTokenBucket,
    cost_milli: u64,
) -> u64 {
    unsafe { bucket.as_ref() }.map_or(0, |tb| tb.retry_after_ns(cost_milli))
}

/// Returns currently available milli-tokens (`0` if `bucket` is null).
///
/// # Safety
/// `bucket` must be null or a valid shared pointer to a live [`DeterministicTokenBucket`].
#[no_mangle]
pub unsafe extern "C" fn par_bucket_tokens_milli(bucket: *const DeterministicTokenBucket) -> u64 {
    unsafe { bucket.as_ref() }.map_or(0, |tb| tb.tokens_milli())
}

/// Returns cumulative allowed acquisitions (`0` if `bucket` is null).
///
/// # Safety
/// `bucket` must be null or a valid shared pointer to a live [`DeterministicTokenBucket`].
#[no_mangle]
pub unsafe extern "C" fn par_bucket_allowed_count(bucket: *const DeterministicTokenBucket) -> u64 {
    unsafe { bucket.as_ref() }.map_or(0, |tb| tb.allowed_count())
}

/// Returns cumulative rate-limited rejections (`0` if `bucket` is null).
///
/// # Safety
/// `bucket` must be null or a valid shared pointer to a live [`DeterministicTokenBucket`].
#[no_mangle]
pub unsafe extern "C" fn par_bucket_denied_count(bucket: *const DeterministicTokenBucket) -> u64 {
    unsafe { bucket.as_ref() }.map_or(0, |tb| tb.denied_count())
}

/// Returns the exact byte size (`256` bytes = `4` cache lines) of a single [`RingSlot`].
#[no_mangle]
pub extern "C" fn par_audit_ring_slot_bytes() -> usize {
    core::mem::size_of::<RingSlot>()
}

/// Returns the fixed slot capacity (`64` slots = `16,384` bytes) of [`CapabilityAuditRing`].
#[no_mangle]
pub extern "C" fn par_audit_ring_capacity() -> usize {
    DEFAULT_AUDIT_RING_SLOTS
}

/// Allocates a new zero-post-init-allocation `64`-slot [`CapabilityAuditRing`] (`16,384` B).
///
/// Caller owns the returned pointer and must release it with [`par_audit_ring_free`].
#[no_mangle]
pub extern "C" fn par_audit_ring_new() -> *mut CapabilityAuditRing {
    Box::into_raw(Box::new(CapabilityAuditRing::new()))
}

/// Releases a [`CapabilityAuditRing`] previously allocated by [`par_audit_ring_new`].
///
/// # Safety
/// `ring` must either be null (no-op) or a valid unaliased pointer returned by [`par_audit_ring_new`].
#[no_mangle]
pub unsafe extern "C" fn par_audit_ring_free(ring: *mut CapabilityAuditRing) {
    if !ring.is_null() {
        drop(unsafe { Box::from_raw(ring) });
    }
}

/// Evaluates capability bitmask fence `(required_caps & granted_caps) == required_caps`,
/// seals a `64`-byte Castagnoli CRC32C audit slot in `ring`, and returns:
/// - `1` if capability check passed (`AuditDecision::Allowed`),
/// - `0` if capability check was denied (`AuditDecision::DeniedCapability`, still sealed in audit ring),
/// - negative `PAR_ERR_*` code on null ring or payload bounds violation.
///
/// # Safety
/// - `ring` must be null (returns `-1`) or a valid uniquely writable pointer from [`par_audit_ring_new`].
/// - When `payload_len > 0`, `payload_ptr` must be non-null and valid for reads of `payload_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn par_audit_ring_record(
    ring: *mut CapabilityAuditRing,
    timestamp_ns: u64,
    agent_id: u32,
    required_caps: u32,
    granted_caps: u32,
    payload_ptr: *const u8,
    payload_len: usize,
) -> i32 {
    let Some(r) = (unsafe { ring.as_mut() }) else {
        return -1;
    };
    if payload_len > 0 && payload_ptr.is_null() {
        return -1;
    }
    let payload = if payload_len == 0 {
        &[][..]
    } else {
        unsafe { core::slice::from_raw_parts(payload_ptr, payload_len) }
    };
    let allowed = (required_caps & granted_caps) == required_caps;
    let flags = if allowed {
        AuditDecision::Allowed as u16
    } else {
        AuditDecision::DeniedCapability as u16
    };
    match r.record(
        timestamp_ns,
        agent_id,
        required_caps & granted_caps,
        flags,
        payload,
    ) {
        Ok(_) => i32::from(allowed),
        Err(err) => encode_ffi_error(err),
    }
}

/// Verifies header + payload Castagnoli CRC32C across all populated slots in `ring`.
///
/// Returns the verified slot count (`0..=64`) or a negative `PAR_ERR_*` tamper error code.
///
/// # Safety
/// `ring` must be null (returns `-1`) or a valid shared pointer to a live [`CapabilityAuditRing`].
#[no_mangle]
pub unsafe extern "C" fn par_audit_ring_verify_all(ring: *const CapabilityAuditRing) -> i32 {
    let Some(r) = (unsafe { ring.as_ref() }) else {
        return -1;
    };
    match r.verify_all() {
        Ok(count) => count as i32,
        Err(err) => encode_ffi_error(err),
    }
}

/// Returns current populated slot count (`0..=64`, or `0` if `ring` is null).
///
/// # Safety
/// `ring` must be null or a valid shared pointer to a live [`CapabilityAuditRing`].
#[no_mangle]
pub unsafe extern "C" fn par_audit_ring_len(ring: *const CapabilityAuditRing) -> usize {
    unsafe { ring.as_ref() }.map_or(0, BoundedTelemetryVault::len)
}

/// Returns monotonic total audit records written since initialization (`0` if `ring` is null).
///
/// # Safety
/// `ring` must be null or a valid shared pointer to a live [`CapabilityAuditRing`].
#[no_mangle]
pub unsafe extern "C" fn par_audit_ring_total_written(ring: *const CapabilityAuditRing) -> u64 {
    unsafe { ring.as_ref() }.map_or(0, BoundedTelemetryVault::total_written)
}

/// Returns a read-only pointer to `RingSlot` at `slot_index` in linear memory (or null if out of bounds).
///
/// # Safety
/// `ring` must be null or a valid shared pointer to a live [`CapabilityAuditRing`].
#[no_mangle]
pub unsafe extern "C" fn par_audit_ring_slot_ptr(
    ring: *const CapabilityAuditRing,
    slot_index: usize,
) -> *const RingSlot {
    let Some(r) = (unsafe { ring.as_ref() }) else {
        return core::ptr::null();
    };
    if slot_index >= r.len() {
        return core::ptr::null();
    }
    &r.slots[slot_index] as *const RingSlot
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn test_crc32c_rfc3720_standard_vector() {
        assert_eq!(crc32c(b""), 0x0000_0000);
        // Canonical RFC 3720 / Castagnoli check value for ASCII "123456789"
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        let msg = b"deterministic-polyglot-ffi-telemetry-frame";
        assert_eq!(crc32c(msg), crc32c(msg));
    }

    #[test]
    fn test_c11_ring_header_alignment_and_tamper_detection() {
        assert_eq!(core::mem::size_of::<RingSlotHeader>(), 64);
        assert_eq!(core::mem::align_of::<RingSlotHeader>(), 64);

        let payload = b"agent.step.completed:ok";
        let mut hdr = RingSlotHeader {
            flags: 0x01,
            sequence: 42,
            timestamp_ns: 1_700_000_000,
            agent_id: 7,
            capability_mask: capabilities::FS_READ | capabilities::TELEMETRY_EMIT,
            ..RingSlotHeader::default()
        };

        seal_ring_header(&mut hdr, payload).expect("seal must succeed");
        validate_ring_header(&hdr, payload).expect("freshly sealed header must validate");

        // 1. Tamper with header field -> HeaderCrcMismatch
        let mut bad_hdr = hdr;
        bad_hdr.capability_mask ^= 0xFF;
        assert_eq!(
            validate_ring_header(&bad_hdr, payload),
            Err(FfiRingError::HeaderCrcMismatch)
        );

        // 2. Tamper with reserved padding -> NonZeroReserved
        let mut bad_res = hdr;
        bad_res.reserved[3] = 1;
        assert_eq!(
            validate_ring_header(&bad_res, payload),
            Err(FfiRingError::NonZeroReserved)
        );

        // 3. Tamper with payload byte -> PayloadCrcMismatch
        let mut bad_payload = payload.to_vec();
        bad_payload[0] ^= 0x01;
        assert_eq!(
            validate_ring_header(&hdr, &bad_payload),
            Err(FfiRingError::PayloadCrcMismatch)
        );
    }

    #[test]
    fn test_bounded_telemetry_vault_zero_alloc_wraparound() {
        let mut vault = BoundedTelemetryVault::<8>::new();
        assert!(vault.is_empty());

        for seq in 0..20u64 {
            let msg = format!("evt-{seq:02}");
            let written_seq = vault
                .record(
                    1_000 + seq * 10,
                    (seq % 4) as u32,
                    capabilities::TELEMETRY_EMIT,
                    0,
                    msg.as_bytes(),
                )
                .expect("record must succeed");
            assert_eq!(written_seq, seq + 1);
        }

        assert_eq!(vault.len(), 8);
        assert_eq!(vault.total_written(), 20);
        assert_eq!(vault.overwritten_slots(), 12);
        assert_eq!(vault.verify_all(), Ok(8));

        // Corrupt a live slot payload directly and confirm C11 audit flags it.
        vault.raw_slot_mut(2).payload[0] ^= 0x80;
        assert_eq!(vault.verify_all(), Err(FfiRingError::PayloadCrcMismatch));
    }

    #[test]
    fn test_dag_topological_sort_deterministic_diamond() {
        let granted = capabilities::FS_READ | capabilities::KV_WRITE | capabilities::TOOL_EXEC;
        let mut dag = CapabilityDagOrchestrator::new(granted, CircuitBreaker::new(3, 5));

        dag.add_task(DagNode {
            id: "ingest".into(),
            deps: vec![],
            required_caps: capabilities::FS_READ,
            retry: RetryPolicy::default(),
            handler: Box::new(|_, _| Ok(10)),
        })
        .unwrap();

        dag.add_task(DagNode {
            id: "branch_b".into(),
            deps: vec!["ingest".into()],
            required_caps: capabilities::TOOL_EXEC,
            retry: RetryPolicy::default(),
            handler: Box::new(|out, _| Ok(out["ingest"] * 3)),
        })
        .unwrap();

        dag.add_task(DagNode {
            id: "branch_a".into(),
            deps: vec!["ingest".into()],
            required_caps: capabilities::TOOL_EXEC,
            retry: RetryPolicy::default(),
            handler: Box::new(|out, _| Ok(out["ingest"] + 5)),
        })
        .unwrap();

        dag.add_task(DagNode {
            id: "commit".into(),
            deps: vec!["branch_a".into(), "branch_b".into()],
            required_caps: capabilities::KV_WRITE,
            retry: RetryPolicy::default(),
            handler: Box::new(|out, _| Ok(out["branch_a"] + out["branch_b"])),
        })
        .unwrap();

        let summary = dag.execute(1).expect("diamond DAG must succeed");
        assert_eq!(
            summary.topological_order,
            vec!["ingest", "branch_a", "branch_b", "commit"]
        );
        assert_eq!(summary.outputs["commit"], 45);
        assert_eq!(summary.total_attempts, 4);
    }

    #[test]
    fn test_dag_cycle_and_missing_dependency_detection() {
        let mut cyclic = CapabilityDagOrchestrator::new(u32::MAX, CircuitBreaker::new(3, 5));
        for (id, dep) in [("a", "c"), ("b", "a"), ("c", "b")] {
            cyclic
                .add_task(DagNode {
                    id: id.into(),
                    deps: vec![dep.into()],
                    required_caps: 0,
                    retry: RetryPolicy::default(),
                    handler: Box::new(|_, _| Ok(1)),
                })
                .unwrap();
        }
        assert_eq!(
            cyclic.topological_sort(),
            Err(OrchestratorError::CycleDetected(vec![
                "a".into(),
                "b".into(),
                "c".into()
            ]))
        );

        let mut missing = CapabilityDagOrchestrator::new(u32::MAX, CircuitBreaker::new(3, 5));
        missing
            .add_task(DagNode {
                id: "sink".into(),
                deps: vec!["nonexistent".into()],
                required_caps: 0,
                retry: RetryPolicy::default(),
                handler: Box::new(|_, _| Ok(1)),
            })
            .unwrap();
        assert_eq!(
            missing.topological_sort(),
            Err(OrchestratorError::MissingDependency {
                task: "sink".into(),
                missing: "nonexistent".into()
            })
        );
    }

    #[test]
    fn test_dag_capability_fence_denies_unauthorized_task() {
        let ran = Rc::new(Cell::new(false));
        let ran_clone = Rc::clone(&ran);
        let mut dag =
            CapabilityDagOrchestrator::new(capabilities::FS_READ, CircuitBreaker::new(3, 5));

        dag.add_task(DagNode {
            id: "fetch_remote".into(),
            deps: vec![],
            required_caps: capabilities::FS_READ | capabilities::NET_HTTP,
            retry: RetryPolicy::default(),
            handler: Box::new(move |_, _| {
                ran_clone.set(true);
                Ok(200)
            }),
        })
        .unwrap();

        let err = dag.execute(1).expect_err("must reject unauthorized capability");
        assert_eq!(
            err,
            OrchestratorError::CapabilityDenied {
                task: "fetch_remote".into(),
                required: capabilities::FS_READ | capabilities::NET_HTTP,
                granted: capabilities::FS_READ,
            }
        );
        assert!(!ran.get(), "unauthorized handler must never be invoked");
    }

    #[test]
    fn test_dag_retry_recovery_and_circuit_breaker_trip() {
        // Part A: Transient failure recovers on attempt 2.
        let mut retry_dag = CapabilityDagOrchestrator::new(u32::MAX, CircuitBreaker::new(4, 10));
        retry_dag
            .add_task(DagNode {
                id: "flaky_rpc".into(),
                deps: vec![],
                required_caps: 0,
                retry: RetryPolicy {
                    max_attempts: 3,
                    base_backoff_us: 25,
                },
                handler: Box::new(|_, attempt| {
                    if attempt < 2 {
                        Err("transient timeout".into())
                    } else {
                        Ok(99)
                    }
                }),
            })
            .unwrap();

        let ok = retry_dag.execute(1).expect("retry must recover");
        assert_eq!(ok.outputs["flaky_rpc"], 99);
        assert_eq!(ok.total_attempts, 2);
        assert_eq!(ok.simulated_backoff_us, 25);

        // Part B: Consecutive failures trip circuit breaker Open, then HalfOpen after cooldown.
        let should_fail = Rc::new(Cell::new(true));
        let fail_flag = Rc::clone(&should_fail);
        let mut cb_dag = CapabilityDagOrchestrator::new(u32::MAX, CircuitBreaker::new(2, 10));
        cb_dag
            .add_task(DagNode {
                id: "downstream".into(),
                deps: vec![],
                required_caps: 0,
                retry: RetryPolicy {
                    max_attempts: 3,
                    base_backoff_us: 10,
                },
                handler: Box::new(move |_, _| {
                    if fail_flag.get() {
                        Err("upstream 503".into())
                    } else {
                        Ok(7)
                    }
                }),
            })
            .unwrap();

        // After 2 failed attempts, breaker trips Open and short-circuits attempt 3!
        assert_eq!(
            cb_dag.execute(100),
            Err(OrchestratorError::CircuitOpen {
                task: "downstream".into()
            })
        );
        assert_eq!(
            cb_dag.breaker().state(),
            BreakerState::Open {
                opened_at_tick: 100
            }
        );

        // Before cooldown (tick 105 < 110), immediate CircuitOpen.
        assert_eq!(
            cb_dag.execute(105),
            Err(OrchestratorError::CircuitOpen {
                task: "downstream".into()
            })
        );

        // At tick 110, HalfOpen probe succeeds and closes breaker!
        should_fail.set(false);
        let recovered = cb_dag.execute(110).expect("HalfOpen probe must recover");
        assert_eq!(recovered.outputs["downstream"], 7);
        assert_eq!(cb_dag.breaker().state(), BreakerState::Closed);
    }

    #[test]
    fn test_p2_quantile_streaming_accuracy_o1_memory() {
        let mut suite = StreamingQuantileSuite::new();
        let initial_bytes = core::mem::size_of_val(&suite);
        assert_eq!(initial_bytes, 560, "state footprint must remain exact 560B O(1)");

        // Deterministic pseudo-random permutation over [0.0, 1000.0) using golden-ratio Weyl sequence.
        let n = 50_000u64;
        for i in 0..n {
            let frac = ((i as f64) * 0.618_033_988_749_894_9).fract();
            suite.observe(frac * 1000.0);
        }

        let snap = suite.snapshot();
        assert_eq!(snap.count, n);
        assert_eq!(snap.state_bytes, initial_bytes);
        assert!((snap.p50 - 500.0).abs() < 5.0, "p50={} out of tolerance", snap.p50);
        assert!((snap.p95 - 950.0).abs() < 5.0, "p95={} out of tolerance", snap.p95);
        assert!((snap.p99 - 990.0).abs() < 3.0, "p99={} out of tolerance", snap.p99);
        assert!((snap.mean - 500.0).abs() < 2.0);
    }

    #[test]
    fn test_wasm_c_abi_p2_suite_lifecycle_and_batch_ingest() {
        assert_eq!(par_p2_suite_state_bytes(), 560);
        let suite = par_p2_suite_new();
        assert!(!suite.is_null());

        let batch_len = 1_000usize;
        let buf = par_p2_alloc_f64_buffer(batch_len);
        assert!(!buf.is_null());

        // SAFETY: `suite` and `buf` were freshly allocated with exact capacities above.
        unsafe {
            for round in 0..25usize {
                let slice = core::slice::from_raw_parts_mut(buf, batch_len);
                for (idx, slot) in slice.iter_mut().enumerate() {
                    let i = (round * batch_len + idx) as f64;
                    *slot = (i * 0.618_033_988_749_894_9).fract() * 1000.0;
                }
                let count = par_p2_suite_observe_batch(suite, buf, batch_len);
                assert_eq!(count, ((round + 1) * batch_len) as u64);
            }

            assert_eq!(par_p2_suite_count(suite), 25_000);
            assert!((par_p2_suite_p50(suite) - 500.0).abs() < 6.0);
            assert!((par_p2_suite_p95(suite) - 950.0).abs() < 6.0);
            assert!((par_p2_suite_p99(suite) - 990.0).abs() < 4.0);
            assert!((par_p2_suite_mean(suite) - 500.0).abs() < 2.0);
            assert!(par_p2_suite_min(suite) >= 0.0 && par_p2_suite_max_helper(suite) <= 1000.0);

            par_p2_suite_reset(suite);
            assert_eq!(par_p2_suite_count(suite), 0);
            par_p2_suite_observe(suite, 42.25);
            assert_eq!(par_p2_suite_count(suite), 1);
            assert!((par_p2_suite_p50(suite) - 42.25).abs() < 1e-9);

            let check_vec = b"123456789";
            let bptr = par_wasm_alloc_bytes(check_vec.len());
            core::ptr::copy_nonoverlapping(check_vec.as_ptr(), bptr, check_vec.len());
            assert_eq!(par_wasm_crc32c(bptr, check_vec.len()), 0xE306_9283);
            par_wasm_free_bytes(bptr, check_vec.len());

            par_p2_free_f64_buffer(buf, batch_len);
            par_p2_suite_free(suite);
        }
    }

    unsafe fn par_p2_suite_max_helper(suite: *const StreamingQuantileSuite) -> f64 {
        unsafe { par_p2_suite_max(suite) }
    }

    #[test]
    fn test_wasm_c_abi_null_pointer_guards() {
        assert_eq!(par_p2_alloc_f64_buffer(0), core::ptr::null_mut());
        assert_eq!(par_wasm_alloc_bytes(0), core::ptr::null_mut());
        // SAFETY: Testing explicit null-pointer defensive no-op contracts.
        unsafe {
            assert_eq!(par_p2_suite_count(core::ptr::null()), 0);
            assert_eq!(par_p2_suite_p50(core::ptr::null()), 0.0);
            assert_eq!(par_p2_suite_p95(core::ptr::null()), 0.0);
            assert_eq!(par_p2_suite_p99(core::ptr::null()), 0.0);
            assert_eq!(par_p2_suite_observe_batch(core::ptr::null_mut(), core::ptr::null(), 8), 0);
            assert_eq!(par_wasm_crc32c(core::ptr::null(), 16), 0);
            par_p2_suite_observe(core::ptr::null_mut(), 10.0);
            par_p2_suite_reset(core::ptr::null_mut());
            par_p2_suite_free(core::ptr::null_mut());
            par_p2_free_f64_buffer(core::ptr::null_mut(), 0);
            par_wasm_free_bytes(core::ptr::null_mut(), 0);
            assert_eq!(par_bucket_try_acquire(core::ptr::null_mut(), 1000, 0), -1);
            assert_eq!(par_bucket_retry_after_ns(core::ptr::null(), 1000), 0);
            assert_eq!(par_bucket_tokens_milli(core::ptr::null()), 0);
            assert_eq!(par_bucket_allowed_count(core::ptr::null()), 0);
            assert_eq!(par_bucket_denied_count(core::ptr::null()), 0);
            par_bucket_free(core::ptr::null_mut());
            assert_eq!(
                par_audit_ring_record(core::ptr::null_mut(), 0, 0, 1, 1, core::ptr::null(), 0),
                -1
            );
            assert_eq!(par_audit_ring_verify_all(core::ptr::null()), -1);
            assert_eq!(par_audit_ring_len(core::ptr::null()), 0);
            assert_eq!(par_audit_ring_total_written(core::ptr::null()), 0);
            assert_eq!(par_audit_ring_slot_ptr(core::ptr::null(), 0), core::ptr::null());
            par_audit_ring_free(core::ptr::null_mut());
        }
        assert_eq!(par_bucket_new(0, 1000, 0), core::ptr::null_mut());
    }

    #[test]
    fn test_token_bucket_exact_nanosecond_conservation_and_c11_parity() {
        assert_eq!(par_bucket_state_bytes(), 64);
        assert_eq!(
            DeterministicTokenBucket::new(0, 1_000, 0),
            Err(FfiRingError::InvalidRateLimitParam)
        );

        // Prime refill rate (3,333 milli-tokens/sec) to stress fractional nanosecond remainders.
        let mut tb = DeterministicTokenBucket::new(5_000, 3_333, 1_000_000).unwrap();
        assert_eq!(
            tb.try_acquire(5_001, 1_000_000),
            Err(FfiRingError::InvalidRateLimitParam)
        );

        // Drain full 5,000 milli-token burst at t = 1_000_000 ns.
        assert_eq!(
            tb.try_acquire(5_000, 1_000_000),
            Ok(RateLimitDecision::Allowed { remaining_milli: 0 })
        );

        // Non-monotonic timestamp (clock skew stepback) must neither panic nor mint tokens.
        let skew = tb.try_acquire(1_000, 999_500).unwrap();
        let expected_wait = tb.retry_after_ns(1_000);
        assert_eq!(
            skew,
            RateLimitDecision::Denied {
                deficit_milli: 1_000,
                retry_after_ns: expected_wait,
            }
        );

        // Step 1,000 ticks of 1_000_000 ns (1 ms each = 1.000s total) -> must conserve exact 3,333 milli-tokens!
        let mut now = 1_000_000u64;
        for _ in 0..1_000 {
            now += 1_000_000;
            let _ = tb.try_acquire(5_000, now).unwrap(); // denied while accumulating exact remainder
        }
        assert_eq!(
            tb.tokens_milli(),
            3_333,
            "1000 x 1ms steps at 3333 milli/s must conserve exact 3333 milli-tokens with zero float drift"
        );

        // Exact 1-nanosecond boundary proof for acquiring 4,000 milli-tokens (deficit = 667 milli-tokens):
        let wait_ns = tb.retry_after_ns(4_000);
        assert!(wait_ns > 0);
        let one_ns_early = tb.try_acquire(4_000, now + wait_ns - 1).unwrap();
        assert!(
            matches!(one_ns_early, RateLimitDecision::Denied { retry_after_ns: 1, .. }),
            "1ns prior to exact ceiling retry_after_ns must deny with remaining wait == 1ns, got {one_ns_early:?}"
        );
        assert_eq!(
            tb.try_acquire(4_000, now + wait_ns),
            Ok(RateLimitDecision::Allowed { remaining_milli: 0 }),
            "exact now + retry_after_ns must succeed with 0 deficit"
        );
    }

    #[test]
    fn test_audited_dag_and_c_abi_capability_audit_ring_export() {
        let mut ring = BoundedTelemetryVault::<16>::new();
        let mut tb = DeterministicTokenBucket::new(2_000, 1_000, 0).unwrap();
        let mut dag = CapabilityDagOrchestrator::new(
            capabilities::FS_READ | capabilities::TOOL_EXEC,
            CircuitBreaker::new(3, 5),
        );
        for id in ["t1", "t2", "t3"] {
            dag.add_task(DagNode {
                id: id.into(),
                deps: vec![],
                required_caps: capabilities::FS_READ,
                retry: RetryPolicy::default(),
                handler: Box::new(|_, _| Ok(1)),
            })
            .unwrap();
        }

        // Cost = 1,000 milli per task, burst = 2,000 milli -> t1 & t2 allowed, t3 rate-limited!
        let err = dag
            .execute_audited(
                AuditExecutionContext {
                    now_tick: 1,
                    now_ns: 10_000,
                    agent_id: 42,
                    cost_per_task_milli: 1_000,
                },
                Some(&mut tb),
                &mut ring,
            )
            .expect_err("3rd task must hit token bucket limit");
        assert!(matches!(err, OrchestratorError::RateLimited { ref task, .. } if task == "t3"));
        assert_eq!(ring.verify_all(), Ok(3));
        assert_eq!(ring.read_slot(0).unwrap().flags, AuditDecision::Allowed as u16);
        assert_eq!(ring.read_slot(1).unwrap().flags, AuditDecision::Allowed as u16);
        assert_eq!(ring.read_slot(2).unwrap().flags, AuditDecision::DeniedRateLimit as u16);

        // Verify C-ABI capability audit ring lifecycle & CRC32C tamper detection.
        assert_eq!(par_audit_ring_slot_bytes(), 256);
        assert_eq!(par_audit_ring_capacity(), 64);
        let c_ring = par_audit_ring_new();
        assert!(!c_ring.is_null());
        unsafe {
            let allow_msg = b"tool:rag_search";
            let rc_ok = par_audit_ring_record(
                c_ring,
                50_000,
                7,
                capabilities::FS_READ,
                capabilities::FS_READ | capabilities::NET_HTTP,
                allow_msg.as_ptr(),
                allow_msg.len(),
            );
            assert_eq!(rc_ok, 1);

            let deny_msg = b"tool:gpu_dispatch";
            let rc_deny = par_audit_ring_record(
                c_ring,
                50_100,
                7,
                capabilities::GPU_DISPATCH,
                capabilities::FS_READ,
                deny_msg.as_ptr(),
                deny_msg.len(),
            );
            assert_eq!(rc_deny, 0);

            assert_eq!(par_audit_ring_len(c_ring), 2);
            assert_eq!(par_audit_ring_total_written(c_ring), 2);
            assert_eq!(par_audit_ring_verify_all(c_ring), 2);

            let slot1_ptr = par_audit_ring_slot_ptr(c_ring, 1);
            assert!(!slot1_ptr.is_null());
            assert_eq!((*slot1_ptr).header.flags, AuditDecision::DeniedCapability as u16);

            par_audit_ring_free(c_ring);
        }
    }
}
