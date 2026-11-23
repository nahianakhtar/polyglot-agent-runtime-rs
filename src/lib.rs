//! # polyglot-agent-runtime-rs
//!
//! Production-grade systems runtime combining:
//! - **C11 FFI Cache-Line Ring Telemetry Vault** (`64`-byte aligned `ParRingSlotHeader`, Castagnoli CRC32C, zero heap allocation after initialization)
//! - **Capability-Fenced DAG Task Orchestrator** (deterministic Kahn topological sort, cycle detection, bounded retry, and `Closed`/`Open`/`HalfOpen` circuit breaker)
//! - **Streaming Jain-Chlamtac P² Quantile Estimator** (`p50`, `p95`, `p99` in strict `O(1)` memory)

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

pub const RING_MAGIC: u32 = 0x5041_5231;
pub const RING_VERSION: u16 = 1;
pub const MAX_PAYLOAD_LEN: usize = 192;

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
        other => Err(FfiRingError::UnknownCode(other)),
    }
}

mod raw_ffi {
    use super::RingSlotHeader;

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
    }
}

/// Safe wrapper over C11 Castagnoli CRC32C (`0x82F63B78` reflected polynomial).
#[inline]
pub fn crc32c(data: &[u8]) -> u32 {
    if data.is_empty() {
        return 0;
    }
    // SAFETY: `data.as_ptr()` is non-null, valid for reads of `data.len()` bytes,
    // and C11 `par_crc32c` neither retains the pointer nor mutates memory.
    unsafe { raw_ffi::par_crc32c(data.as_ptr(), data.len()) }
}

/// Seals a 64-byte cache-line-aligned ring header against `payload` via C11 FFI.
#[inline]
pub fn seal_ring_header(hdr: &mut RingSlotHeader, payload: &[u8]) -> Result<(), FfiRingError> {
    let ptr = if payload.is_empty() {
        core::ptr::null()
    } else {
        payload.as_ptr()
    };
    // SAFETY: `hdr` is a uniquely borrowed `#[repr(C, align(64))]` struct matching C11 layout,
    // and `ptr` is valid for `payload.len()` bytes across the synchronous FFI call.
    let rc = unsafe { raw_ffi::par_ring_header_seal(hdr as *mut RingSlotHeader, ptr, payload.len()) };
    decode_ffi_status(rc)
}

/// Validates a 64-byte cache-line-aligned ring header and its `payload` via C11 FFI.
#[inline]
pub fn validate_ring_header(hdr: &RingSlotHeader, payload: &[u8]) -> Result<(), FfiRingError> {
    let ptr = if payload.is_empty() {
        core::ptr::null()
    } else {
        payload.as_ptr()
    };
    // SAFETY: `hdr` and `payload` are shared borrows valid for the duration of the pure C11 check.
    let rc = unsafe {
        raw_ffi::par_ring_header_validate(hdr as *const RingSlotHeader, ptr, payload.len())
    };
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
      }
