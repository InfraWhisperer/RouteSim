//! Simulated GPU backend with queue management, compute modeling, and KV cache.
//!
//! Each [`SimulatedBackend`] represents a single GPU worker that processes
//! inference requests with realistic timing based on its hardware profile
//! and current load.

use crate::kv_cache::KvCacheSimulator;
use crate::request::{ActiveRequest, InferenceRequest, QueuedRequest};
use crate::topology::{BackendRole, BackendState, GpuProfile};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

/// Compute model parameters for a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeModel {
    /// Prefill throughput in tokens/sec.
    pub prefill_tokens_per_sec: f64,
    /// Decode throughput at batch size 1 in tokens/sec.
    pub decode_tokens_per_sec_batch1: f64,
    /// Batch size at which decode throughput saturates.
    pub decode_throughput_saturation_batch: u32,
    /// Decode throughput at saturation in tokens/sec.
    pub decode_tokens_per_sec_saturated: f64,
}

impl ComputeModel {
    /// Get decode throughput for a given batch size using piecewise-linear interpolation.
    ///
    /// Throughput increases linearly from batch1 rate to saturated rate,
    /// then stays constant (or slightly degrades) beyond the saturation point.
    pub fn decode_throughput_at_batch(&self, batch_size: u32) -> f64 {
        if batch_size == 0 {
            return 0.0;
        }
        if batch_size >= self.decode_throughput_saturation_batch {
            return self.decode_tokens_per_sec_saturated;
        }
        let t =
            (batch_size - 1) as f64 / (self.decode_throughput_saturation_batch - 1).max(1) as f64;
        self.decode_tokens_per_sec_batch1
            + t * (self.decode_tokens_per_sec_saturated - self.decode_tokens_per_sec_batch1)
    }

    /// Estimate prefill latency in milliseconds for a given number of tokens.
    pub fn prefill_latency_ms(&self, tokens: u32) -> f64 {
        if self.prefill_tokens_per_sec <= 0.0 {
            return 0.0;
        }
        (tokens as f64 / self.prefill_tokens_per_sec) * 1000.0
    }

    /// Estimate time between tokens for current batch size (in milliseconds).
    pub fn inter_token_latency_ms(&self, batch_size: u32) -> f64 {
        let throughput = self.decode_throughput_at_batch(batch_size);
        if throughput <= 0.0 {
            return f64::MAX;
        }
        // Total tokens/sec across batch; per-request TBT = batch_size / throughput
        (batch_size as f64 / throughput) * 1000.0
    }
}

impl Default for ComputeModel {
    fn default() -> Self {
        // Default H100-like parameters
        Self {
            prefill_tokens_per_sec: 50_000.0,
            decode_tokens_per_sec_batch1: 80.0,
            decode_throughput_saturation_batch: 64,
            decode_tokens_per_sec_saturated: 3200.0,
        }
    }
}

/// Information about a batch of requests currently being processed.
#[derive(Debug, Clone, Default)]
pub struct ActiveBatch {
    /// Requests in the batch.
    pub requests: Vec<ActiveRequest>,
    /// Total tokens being processed.
    pub total_tokens: u32,
}

impl ActiveBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn size(&self) -> u32 {
        self.requests.len() as u32
    }

    /// Add a request to the batch. Returns false if the batch would exceed max_tokens.
    pub fn try_add(&mut self, request: ActiveRequest, max_tokens: u32) -> bool {
        let req_tokens = request.request.total_tokens();
        if self.total_tokens + req_tokens > max_tokens && !self.requests.is_empty() {
            return false;
        }
        self.total_tokens += req_tokens;
        self.requests.push(request);
        true
    }

    /// Remove completed requests, returning them.
    pub fn remove_completed(&mut self) -> Vec<ActiveRequest> {
        let (completed, active): (Vec<_>, Vec<_>) =
            self.requests.drain(..).partition(|r| r.is_complete());
        self.total_tokens = active.iter().map(|r| r.request.total_tokens()).sum();
        self.requests = active;
        completed
    }
}

/// A simulated GPU backend that processes LLM inference requests.
#[derive(Debug, Clone)]
pub struct SimulatedBackend {
    /// Unique backend identifier.
    pub id: u32,
    /// GPU hardware profile.
    pub gpu_type: GpuProfile,
    /// Compute model parameters.
    pub compute_model: ComputeModel,
    /// KV cache simulator.
    pub kv_cache: KvCacheSimulator,
    /// Pending request queue.
    pub queue: VecDeque<QueuedRequest>,
    /// Currently active batch (requests being processed).
    pub active_batch: ActiveBatch,
    /// Maximum tokens per batch.
    pub max_batch_tokens: u32,
    /// Maximum queue depth.
    pub max_queue_depth: u32,
    /// Backend role in the cluster.
    pub role: BackendRole,
    /// Current state.
    pub state: BackendState,
    /// Loaded LoRA adapters.
    pub lora_adapters: Vec<String>,
    // --- Counters ---
    /// Total requests served since start.
    pub total_requests_served: u64,
    /// Total tokens generated since start.
    pub total_tokens_generated: u64,
    /// Total prompt tokens processed.
    pub total_prompt_tokens: u64,
    /// Total time spent processing (for utilization calculation).
    pub busy_time_ms: u64,
}

impl SimulatedBackend {
    /// Create a new simulated backend.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u32,
        gpu_type: GpuProfile,
        compute_model: ComputeModel,
        kv_cache_blocks: u32,
        kv_block_size: u32,
        max_batch_tokens: u32,
        max_queue_depth: u32,
        role: BackendRole,
    ) -> Self {
        Self {
            id,
            gpu_type,
            compute_model,
            kv_cache: KvCacheSimulator::new(kv_cache_blocks, kv_block_size),
            queue: VecDeque::new(),
            active_batch: ActiveBatch::new(),
            max_batch_tokens,
            max_queue_depth,
            role,
            state: BackendState::Idle,
            lora_adapters: Vec::new(),
            total_requests_served: 0,
            total_tokens_generated: 0,
            total_prompt_tokens: 0,
            busy_time_ms: 0,
        }
    }

    /// Whether the queue has room for another request.
    pub fn can_accept(&self) -> bool {
        self.state != BackendState::Draining
            && self.state != BackendState::Offline
            && (self.queue.len() as u32) < self.max_queue_depth
    }

    /// Enqueue a request. Returns false if the queue is full.
    pub fn enqueue(&mut self, request: InferenceRequest, enqueue_time_ms: u64) -> bool {
        if !self.can_accept() {
            return false;
        }
        self.queue.push_back(QueuedRequest {
            request,
            enqueue_time_ms,
        });
        true
    }

    /// Dequeue the next request from the queue.
    pub fn dequeue(&mut self) -> Option<QueuedRequest> {
        self.queue.pop_front()
    }

    /// Current queue depth.
    pub fn queue_depth(&self) -> u32 {
        self.queue.len() as u32
    }

    /// Number of requests actively being processed.
    pub fn active_batch_size(&self) -> u32 {
        self.active_batch.size()
    }

    /// Total tokens in the active batch.
    pub fn active_batch_tokens(&self) -> u32 {
        self.active_batch.total_tokens
    }

    /// Estimated time to first token for a new request, in milliseconds.
    pub fn estimated_ttft_ms(&self) -> f64 {
        let queue_wait = self.estimated_queue_wait_ms();
        let avg_prompt = if self.queue.is_empty() {
            512.0
        } else {
            let total: u32 = self.queue.iter().map(|q| q.request.prompt_tokens).sum();
            total as f64 / self.queue.len() as f64
        };
        let prefill = self.compute_model.prefill_latency_ms(avg_prompt as u32);
        queue_wait + prefill
    }

    /// Estimated queue wait time in milliseconds.
    fn estimated_queue_wait_ms(&self) -> f64 {
        if self.queue.is_empty() {
            return 0.0;
        }
        let batch_size = self.active_batch.size().max(1);
        let tbt = self.compute_model.inter_token_latency_ms(batch_size);
        let avg_remaining = if self.active_batch.requests.is_empty() {
            50.0
        } else {
            let total_remaining: u32 = self
                .active_batch
                .requests
                .iter()
                .map(|r| {
                    r.request
                        .actual_gen_tokens
                        .saturating_sub(r.tokens_generated)
                })
                .sum();
            total_remaining as f64 / self.active_batch.size() as f64
        };
        self.queue.len() as f64 * tbt * avg_remaining
    }

    /// Current decode tokens/sec for the active batch.
    pub fn current_tokens_per_sec(&self) -> f64 {
        self.compute_model
            .decode_throughput_at_batch(self.active_batch.size())
    }

    /// Create a read-only snapshot for routing algorithms.
    pub fn snapshot(&self) -> BackendSnapshot {
        BackendSnapshot {
            id: self.id,
            queue_depth: self.queue_depth(),
            active_batch_size: self.active_batch.size(),
            active_batch_tokens: self.active_batch.total_tokens,
            kv_cache_utilization: self.kv_cache.utilization(),
            prefix_hashes_cached: self.kv_cache.cached_prefix_hashes(),
            cached_block_hashes: self.kv_cache.cached_content_block_hashes(),
            estimated_ttft_ms: self.estimated_ttft_ms(),
            tokens_per_sec_current: self.current_tokens_per_sec(),
            role: self.role,
            state: self.state,
            lora_adapters_loaded: self.lora_adapters.clone(),
            total_requests_served: self.total_requests_served,
            total_tokens_generated: self.total_tokens_generated,
            max_queue_depth: self.max_queue_depth,
        }
    }
}

/// Read-only snapshot of a backend's state, provided to routing algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSnapshot {
    /// Backend identifier.
    pub id: u32,
    /// Number of requests waiting in queue.
    pub queue_depth: u32,
    /// Number of requests in the active batch.
    pub active_batch_size: u32,
    /// Total tokens in the active batch.
    pub active_batch_tokens: u32,
    /// KV cache utilization (0.0 - 1.0).
    pub kv_cache_utilization: f32,
    /// Set of prefix hashes currently cached.
    pub prefix_hashes_cached: HashSet<u64>,
    /// Set of individual cache block hashes currently held by this backend.
    pub cached_block_hashes: HashSet<u64>,
    /// Estimated time to first token for a new request (ms).
    pub estimated_ttft_ms: f64,
    /// Current decode throughput (tokens/sec).
    pub tokens_per_sec_current: f64,
    /// Backend role (Prefill, Decode, Both).
    pub role: BackendRole,
    /// Current backend state.
    pub state: BackendState,
    /// Loaded LoRA adapters.
    pub lora_adapters_loaded: Vec<String>,
    /// Total requests served.
    pub total_requests_served: u64,
    /// Total tokens generated.
    pub total_tokens_generated: u64,
    /// Maximum queue depth.
    pub max_queue_depth: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_backend() -> SimulatedBackend {
        SimulatedBackend::new(
            0,
            GpuProfile::H100Sxm,
            ComputeModel::default(),
            1000,
            16,
            16384,
            256,
            BackendRole::Both,
        )
    }

    fn sample_request(id: u64) -> InferenceRequest {
        InferenceRequest {
            id,
            arrival_time_ms: 0,
            prompt_tokens: 512,
            max_gen_tokens: 256,
            actual_gen_tokens: 128,
            prefix_hash: None,
            prefix_token_length: None,
            cache_block_hashes: Vec::new(),
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
            metadata: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn test_compute_model_throughput() {
        let model = ComputeModel::default();
        assert_eq!(model.decode_throughput_at_batch(0), 0.0);
        assert_eq!(model.decode_throughput_at_batch(1), 80.0);
        assert_eq!(model.decode_throughput_at_batch(64), 3200.0);
        assert_eq!(model.decode_throughput_at_batch(128), 3200.0);

        // Midpoint should be between batch1 and saturated
        let mid = model.decode_throughput_at_batch(32);
        assert!(mid > 80.0 && mid < 3200.0);
    }

    #[test]
    fn test_prefill_latency() {
        let model = ComputeModel::default();
        let latency = model.prefill_latency_ms(1000);
        assert!((latency - 20.0).abs() < 0.01); // 1000 / 50000 * 1000 = 20ms
    }

    #[test]
    fn test_enqueue_dequeue() {
        let mut backend = default_backend();
        let req = sample_request(1);
        assert!(backend.enqueue(req, 0));
        assert_eq!(backend.queue_depth(), 1);
        let dequeued = backend.dequeue().unwrap();
        assert_eq!(dequeued.request.id, 1);
        assert_eq!(backend.queue_depth(), 0);
    }

    #[test]
    fn test_queue_full() {
        let mut backend = SimulatedBackend::new(
            0,
            GpuProfile::H100Sxm,
            ComputeModel::default(),
            1000,
            16,
            16384,
            2, // max queue depth 2
            BackendRole::Both,
        );
        assert!(backend.enqueue(sample_request(1), 0));
        assert!(backend.enqueue(sample_request(2), 0));
        assert!(!backend.enqueue(sample_request(3), 0)); // queue full
    }

    #[test]
    fn test_cannot_accept_when_draining() {
        let mut backend = default_backend();
        backend.state = BackendState::Draining;
        assert!(!backend.can_accept());
    }

    #[test]
    fn test_snapshot() {
        let backend = default_backend();
        let snap = backend.snapshot();
        assert_eq!(snap.id, 0);
        assert_eq!(snap.queue_depth, 0);
        assert_eq!(snap.active_batch_size, 0);
        assert_eq!(snap.role, BackendRole::Both);
    }

    #[test]
    fn test_active_batch() {
        let mut batch = ActiveBatch::new();
        let req = ActiveRequest::new(sample_request(1));
        assert!(batch.try_add(req, 16384));
        assert_eq!(batch.size(), 1);
    }
}
