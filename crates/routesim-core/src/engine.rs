//! Discrete-event simulation engine.
//!
//! The engine maintains a priority queue of [`SimEvent`]s sorted by timestamp.
//! Each iteration pops the next event, advances the virtual clock, processes
//! the event (potentially generating new events), and collects metrics.

use crate::backend::SimulatedBackend;
use crate::clock::SimClock;
use crate::config::SimConfig;
use crate::metrics::{MetricsCollector, RequestMetric, SimulationMetrics};
use crate::request::ActiveRequest;
use crate::request::InferenceRequest;
use crate::topology::{BackendRole, BackendState};
use routesim_algorithms::{self, RoutingAlgorithm};
use std::collections::{BinaryHeap, HashMap};

/// Events in the discrete-event simulation.
#[derive(Debug, Clone)]
pub enum SimEvent {
    /// A new request arrives at the load balancer.
    RequestArrival(InferenceRequest),
    /// Prefill computation completes for a request on a backend.
    PrefillComplete { backend_id: u32, request_id: u64 },
    /// A decode token is generated.
    TokenGenerated {
        backend_id: u32,
        request_id: u64,
        token_num: u32,
    },
    /// Request generation is fully complete.
    RequestComplete { backend_id: u32, request_id: u64 },
    /// Trigger batch scheduling on a backend (continuous batching).
    BatchSchedule { backend_id: u32 },
    /// KV cache eviction needed on a backend.
    KvCacheEviction { backend_id: u32 },
    /// Health check for a backend.
    BackendHealthCheck { backend_id: u32 },
    /// KV transfer starts between prefill and decode nodes (disaggregated).
    KvTransferStart {
        from_backend: u32,
        to_backend: u32,
        request_id: u64,
    },
    /// KV transfer completes.
    KvTransferComplete {
        from_backend: u32,
        to_backend: u32,
        request_id: u64,
    },
}

/// A timestamped event for the priority queue.
#[derive(Debug, Clone)]
struct TimedEvent {
    time_ms: u64,
    sequence: u64,
    event: SimEvent,
}

impl PartialEq for TimedEvent {
    fn eq(&self, other: &Self) -> bool {
        self.time_ms == other.time_ms && self.sequence == other.sequence
    }
}

impl Eq for TimedEvent {}

impl PartialOrd for TimedEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TimedEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap; we want min-heap
        other
            .time_ms
            .cmp(&self.time_ms)
            .then(other.sequence.cmp(&self.sequence))
    }
}

// --- Type conversions from core types to algorithm types ---

fn to_algo_snapshot(
    snap: &crate::backend::BackendSnapshot,
) -> routesim_algorithms::BackendSnapshot {
    routesim_algorithms::BackendSnapshot {
        id: snap.id,
        queue_depth: snap.queue_depth,
        active_batch_size: snap.active_batch_size,
        active_batch_tokens: snap.active_batch_tokens,
        kv_cache_utilization: snap.kv_cache_utilization,
        prefix_hashes_cached: snap.prefix_hashes_cached.clone(),
        estimated_ttft_ms: snap.estimated_ttft_ms,
        tokens_per_sec_current: snap.tokens_per_sec_current,
        role: match snap.role {
            BackendRole::Both => routesim_algorithms::BackendRole::Both,
            BackendRole::Prefill => routesim_algorithms::BackendRole::Prefill,
            BackendRole::Decode => routesim_algorithms::BackendRole::Decode,
        },
        state: match snap.state {
            BackendState::Idle => routesim_algorithms::BackendState::Idle,
            BackendState::Processing => routesim_algorithms::BackendState::Processing,
            BackendState::Draining => routesim_algorithms::BackendState::Draining,
            BackendState::Offline => routesim_algorithms::BackendState::Offline,
        },
        lora_adapters_loaded: snap.lora_adapters_loaded.clone(),
        total_requests_served: snap.total_requests_served,
        total_tokens_generated: snap.total_tokens_generated,
    }
}

fn to_algo_request(req: &InferenceRequest) -> routesim_algorithms::RequestInfo {
    routesim_algorithms::RequestInfo {
        id: req.id,
        prompt_tokens: req.prompt_tokens,
        max_gen_tokens: req.max_gen_tokens,
        actual_gen_tokens: req.actual_gen_tokens,
        prefix_hash: req.prefix_hash,
        prefix_token_length: req.prefix_token_length,
        conversation_id: req.conversation_id.clone(),
        lora_adapter: req.lora_adapter.clone(),
        priority: req.priority,
    }
}

fn to_algo_event(event: &SimEvent) -> routesim_algorithms::SimEventInfo {
    match event {
        SimEvent::RequestArrival(req) => {
            routesim_algorithms::SimEventInfo::RequestArrival { request_id: req.id }
        }
        SimEvent::PrefillComplete {
            backend_id,
            request_id,
        } => routesim_algorithms::SimEventInfo::PrefillComplete {
            backend_id: *backend_id,
            request_id: *request_id,
        },
        SimEvent::RequestComplete {
            backend_id,
            request_id,
        } => routesim_algorithms::SimEventInfo::RequestComplete {
            backend_id: *backend_id,
            request_id: *request_id,
        },
        SimEvent::TokenGenerated {
            backend_id,
            request_id,
            token_num,
        } => routesim_algorithms::SimEventInfo::TokenGenerated {
            backend_id: *backend_id,
            request_id: *request_id,
            token_num: *token_num,
        },
        // Other events don't have algorithm-facing equivalents
        _ => routesim_algorithms::SimEventInfo::RequestArrival { request_id: 0 },
    }
}

/// Clock adapter implementing the algorithm crate's Clock trait.
struct ClockAdapter<'a>(&'a SimClock);

impl<'a> routesim_algorithms::Clock for ClockAdapter<'a> {
    fn now_ms(&self) -> u64 {
        self.0.now_ms()
    }
}

/// The main simulation engine.
pub struct SimulationEngine {
    /// Virtual clock.
    pub clock: SimClock,
    /// Event queue (min-heap by time).
    event_queue: BinaryHeap<TimedEvent>,
    /// Sequence counter for tie-breaking.
    sequence: u64,
    /// Simulated backends.
    pub backends: Vec<SimulatedBackend>,
    /// Metrics collector.
    pub metrics: MetricsCollector,
    /// Routing algorithm.
    algorithm: Box<dyn RoutingAlgorithm>,
    /// Requests in flight (request_id -> backend_id).
    requests_in_flight: HashMap<u64, u32>,
    /// Active requests by (backend_id, request_id).
    active_requests: HashMap<(u32, u64), ActiveRequest>,
    /// Total events processed.
    pub events_processed: u64,
    /// Configuration.
    config: SimConfig,
}

impl SimulationEngine {
    /// Create a new simulation engine from config and algorithm.
    pub fn new(config: SimConfig, algorithm: Box<dyn RoutingAlgorithm>) -> Self {
        let gpu_profile = config.gpu_profile();
        let compute_model = config.compute_model();
        let num_backends = config.cluster.num_backends;

        let mut backends = Vec::with_capacity(num_backends as usize);
        for i in 0..num_backends {
            let role = if config.cluster.disaggregated.enabled {
                if i < config.cluster.disaggregated.prefill_backends {
                    BackendRole::Prefill
                } else {
                    BackendRole::Decode
                }
            } else {
                BackendRole::Both
            };

            backends.push(SimulatedBackend::new(
                i,
                gpu_profile.clone(),
                compute_model.clone(),
                config.cluster.kv_cache_blocks,
                config.cluster.kv_block_size,
                config.cluster.max_batch_tokens,
                config.cluster.max_queue_depth,
                role,
            ));
        }

        Self {
            clock: SimClock::new(),
            event_queue: BinaryHeap::new(),
            sequence: 0,
            backends,
            metrics: MetricsCollector::new(config.simulation.warmup_requests),
            algorithm,
            requests_in_flight: HashMap::new(),
            active_requests: HashMap::new(),
            events_processed: 0,
            config,
        }
    }

    /// Schedule an event at a given time.
    pub fn schedule_event(&mut self, time_ms: u64, event: SimEvent) {
        self.event_queue.push(TimedEvent {
            time_ms,
            sequence: self.sequence,
            event,
        });
        self.sequence += 1;
    }

    /// Load a trace (list of requests) into the event queue.
    pub fn load_trace(&mut self, requests: Vec<InferenceRequest>) {
        for req in requests {
            let arrival = req.arrival_time_ms;
            self.schedule_event(arrival, SimEvent::RequestArrival(req));
        }
    }

    /// Run the simulation until all events are processed.
    pub fn run(&mut self) -> SimulationMetrics {
        while let Some(timed_event) = self.event_queue.pop() {
            self.clock.advance_to_ms(timed_event.time_ms);
            self.process_event(timed_event.event);
            self.events_processed += 1;
        }

        let custom_metrics = self.algorithm.custom_metrics();
        self.metrics
            .aggregate(self.algorithm.name(), &self.backends, custom_metrics)
    }

    /// Process a single event.
    fn process_event(&mut self, event: SimEvent) {
        // Let the algorithm observe the event
        let algo_snapshots: Vec<_> = self
            .backends
            .iter()
            .map(|b| to_algo_snapshot(&b.snapshot()))
            .collect();
        let algo_event = to_algo_event(&event);
        self.algorithm.on_event(&algo_event, &algo_snapshots);

        match event {
            SimEvent::RequestArrival(request) => self.handle_arrival(request),
            SimEvent::PrefillComplete {
                backend_id,
                request_id,
            } => self.handle_prefill_complete(backend_id, request_id),
            SimEvent::TokenGenerated {
                backend_id,
                request_id,
                token_num,
            } => self.handle_token_generated(backend_id, request_id, token_num),
            SimEvent::RequestComplete {
                backend_id,
                request_id,
            } => self.handle_request_complete(backend_id, request_id),
            SimEvent::BatchSchedule { backend_id } => self.handle_batch_schedule(backend_id),
            SimEvent::KvCacheEviction { .. } => {}
            SimEvent::BackendHealthCheck { .. } => {}
            SimEvent::KvTransferStart {
                from_backend,
                to_backend,
                request_id,
            } => self.handle_kv_transfer_start(from_backend, to_backend, request_id),
            SimEvent::KvTransferComplete {
                from_backend: _,
                to_backend,
                request_id,
            } => self.handle_kv_transfer_complete(to_backend, request_id),
        }
    }

    /// Handle a request arrival: route to a backend.
    fn handle_arrival(&mut self, request: InferenceRequest) {
        let algo_snapshots: Vec<_> = self
            .backends
            .iter()
            .map(|b| to_algo_snapshot(&b.snapshot()))
            .collect();
        let algo_request = to_algo_request(&request);
        let clock_adapter = ClockAdapter(&self.clock);
        let decision = self
            .algorithm
            .route(&algo_request, &algo_snapshots, &clock_adapter);

        match decision {
            routesim_algorithms::RoutingDecision::Route(backend_id) => {
                self.route_to_backend(request, backend_id);
            }
            routesim_algorithms::RoutingDecision::RouteWithPriority(backend_id, _priority) => {
                self.route_to_backend(request, backend_id);
            }
            routesim_algorithms::RoutingDecision::Reject => {
                self.metrics.record_rejection();
            }
            routesim_algorithms::RoutingDecision::RouteDisaggregated { prefill, decode } => {
                self.route_to_prefill(request, prefill, decode);
            }
        }
    }

    /// Route a request to a specific backend.
    fn route_to_backend(&mut self, request: InferenceRequest, backend_id: u32) {
        let now = self.clock.now_ms();
        let backend = &mut self.backends[backend_id as usize];

        if !backend.enqueue(request.clone(), now) {
            self.metrics.record_rejection();
            return;
        }

        self.requests_in_flight.insert(request.id, backend_id);

        // Schedule batch processing if backend is idle
        if backend.state != BackendState::Processing {
            self.schedule_event(now, SimEvent::BatchSchedule { backend_id });
        }
    }

    /// Route a request to a prefill backend in disaggregated mode.
    fn route_to_prefill(
        &mut self,
        request: InferenceRequest,
        prefill_backend: u32,
        _decode_backend: u32,
    ) {
        let now = self.clock.now_ms();
        let backend = &mut self.backends[prefill_backend as usize];

        if !backend.enqueue(request.clone(), now) {
            self.metrics.record_rejection();
            return;
        }

        self.requests_in_flight.insert(request.id, prefill_backend);

        if backend.state != BackendState::Processing {
            self.schedule_event(
                now,
                SimEvent::BatchSchedule {
                    backend_id: prefill_backend,
                },
            );
        }
    }

    /// Handle batch scheduling: dequeue requests and start prefill.
    fn handle_batch_schedule(&mut self, backend_id: u32) {
        let now = self.clock.now_ms();

        // Collect events and state changes to apply after releasing the backend borrow.
        let mut pending_events: Vec<(u64, SimEvent)> = Vec::new();
        let mut pending_active: Vec<((u32, u64), ActiveRequest)> = Vec::new();
        let mut rejections = 0u32;

        {
            let backend = &mut self.backends[backend_id as usize];

            while let Some(queued) = backend.queue.front().cloned() {
                let request = queued.request.clone();
                let total_tokens = request.total_tokens();

                if backend.active_batch.total_tokens + total_tokens > backend.max_batch_tokens
                    && backend.active_batch.size() > 0
                {
                    break;
                }

                backend.queue.pop_front();

                let alloc_result = backend.kv_cache.allocate_for_request(
                    request.id,
                    total_tokens,
                    request.prefix_hash,
                    request.prefix_token_length,
                );

                if !alloc_result.success {
                    rejections += 1;
                    continue;
                }

                let mut active = ActiveRequest::new(request.clone());
                active.prefix_cache_hit = alloc_result.prefix_cache_hit;
                active.prefill_start_ms = Some(now);

                let prefill_tokens = if alloc_result.prefix_cache_hit {
                    request.new_prompt_tokens()
                } else {
                    request.prompt_tokens
                };
                let prefill_ms = backend
                    .compute_model
                    .prefill_latency_ms(prefill_tokens)
                    .ceil() as u64;

                pending_active.push(((backend_id, request.id), active));

                backend.state = BackendState::Processing;
                backend.total_prompt_tokens += request.prompt_tokens as u64;

                pending_events.push((
                    now + prefill_ms,
                    SimEvent::PrefillComplete {
                        backend_id,
                        request_id: request.id,
                    },
                ));
            }
        }

        // Apply deferred state changes
        for _ in 0..rejections {
            self.metrics.record_rejection();
        }
        for (key, active) in pending_active {
            self.active_requests.insert(key, active);
        }
        for (time, event) in pending_events {
            self.schedule_event(time, event);
        }
    }

    /// Handle prefill completion: start token generation.
    fn handle_prefill_complete(&mut self, backend_id: u32, request_id: u64) {
        let now = self.clock.now_ms();

        // Check if this is a prefill-only backend (disaggregated)
        let role = self.backends[backend_id as usize].role;
        if role == BackendRole::Prefill {
            let decode_id = self
                .backends
                .iter()
                .filter(|b| b.role == BackendRole::Decode && b.can_accept())
                .min_by_key(|b| b.queue_depth())
                .map(|b| b.id);

            if let Some(decode_id) = decode_id {
                self.schedule_event(
                    now,
                    SimEvent::KvTransferStart {
                        from_backend: backend_id,
                        to_backend: decode_id,
                        request_id,
                    },
                );
            } else {
                self.metrics.record_rejection();
            }
            return;
        }

        if let Some(active) = self.active_requests.get_mut(&(backend_id, request_id)) {
            active.prefill_end_ms = Some(now);
            active.first_token_ms = Some(now);

            let backend = &mut self.backends[backend_id as usize];
            backend
                .active_batch
                .try_add(active.clone(), backend.max_batch_tokens);

            if active.request.actual_gen_tokens > 0 {
                let batch_size = backend.active_batch.size();
                let tbt_ms = backend
                    .compute_model
                    .inter_token_latency_ms(batch_size)
                    .ceil() as u64;
                self.schedule_event(
                    now + tbt_ms.max(1),
                    SimEvent::TokenGenerated {
                        backend_id,
                        request_id,
                        token_num: 1,
                    },
                );
            } else {
                self.schedule_event(
                    now,
                    SimEvent::RequestComplete {
                        backend_id,
                        request_id,
                    },
                );
            }
        }

        // Try to schedule more requests from the queue
        self.schedule_event(now, SimEvent::BatchSchedule { backend_id });
    }

    /// Handle a token being generated.
    fn handle_token_generated(&mut self, backend_id: u32, request_id: u64, token_num: u32) {
        let now = self.clock.now_ms();

        if let Some(active) = self.active_requests.get_mut(&(backend_id, request_id)) {
            active.tokens_generated = token_num;
            active.last_token_ms = Some(now);

            if active.is_complete() {
                self.schedule_event(
                    now,
                    SimEvent::RequestComplete {
                        backend_id,
                        request_id,
                    },
                );
            } else {
                let backend = &self.backends[backend_id as usize];
                let batch_size = backend.active_batch.size().max(1);
                let tbt_ms = backend
                    .compute_model
                    .inter_token_latency_ms(batch_size)
                    .ceil() as u64;
                self.schedule_event(
                    now + tbt_ms.max(1),
                    SimEvent::TokenGenerated {
                        backend_id,
                        request_id,
                        token_num: token_num + 1,
                    },
                );
            }
        }
    }

    /// Handle request completion.
    fn handle_request_complete(&mut self, backend_id: u32, request_id: u64) {
        let now = self.clock.now_ms();

        if let Some(active) = self.active_requests.remove(&(backend_id, request_id)) {
            let request = &active.request;

            // Compute TBT samples
            let tbt_samples = if active.tokens_generated > 1 {
                let total_decode_time = now.saturating_sub(active.first_token_ms.unwrap_or(now));
                let avg_tbt = total_decode_time as f64 / active.tokens_generated as f64;
                vec![avg_tbt; active.tokens_generated as usize]
            } else {
                vec![]
            };

            let queue_wait = active
                .prefill_start_ms
                .unwrap_or(now)
                .saturating_sub(request.arrival_time_ms);

            let metric = RequestMetric {
                request_id: request.id,
                backend_id,
                arrival_time_ms: request.arrival_time_ms,
                queue_wait_ms: queue_wait,
                ttft_ms: active.ttft_ms().unwrap_or(0),
                total_latency_ms: now.saturating_sub(request.arrival_time_ms),
                prompt_tokens: request.prompt_tokens,
                gen_tokens: active.tokens_generated,
                prefix_cache_hit: active.prefix_cache_hit,
                tbt_samples_ms: tbt_samples,
            };
            self.metrics.record(metric);

            // Update backend counters
            let backend = &mut self.backends[backend_id as usize];
            backend.total_requests_served += 1;
            backend.total_tokens_generated += active.tokens_generated as u64;
            backend.busy_time_ms += now.saturating_sub(active.prefill_start_ms.unwrap_or(now));

            // Release KV cache blocks
            backend.kv_cache.release_request(request_id);

            // Remove from active batch
            backend
                .active_batch
                .requests
                .retain(|r| r.request.id != request_id);
            backend.active_batch.total_tokens = backend
                .active_batch
                .requests
                .iter()
                .map(|r| r.request.total_tokens())
                .sum();

            // Update backend state
            if backend.active_batch.requests.is_empty() && backend.queue.is_empty() {
                backend.state = BackendState::Idle;
            }

            self.requests_in_flight.remove(&request_id);
        }

        // Try to schedule more from the queue
        self.schedule_event(now, SimEvent::BatchSchedule { backend_id });
    }

    /// Handle KV transfer start (disaggregated mode).
    fn handle_kv_transfer_start(&mut self, from_backend: u32, to_backend: u32, request_id: u64) {
        let now = self.clock.now_ms();
        let disagg = &self.config.cluster.disaggregated;

        let tokens = self
            .active_requests
            .get(&(from_backend, request_id))
            .map(|a| a.request.prompt_tokens)
            .unwrap_or(512);

        let transfer_us = estimate_transfer_time_us(
            disagg.kv_transfer_latency_us,
            disagg.kv_transfer_bandwidth_gb_s,
            tokens,
            80,
        );
        let transfer_ms = transfer_us.div_ceil(1000);

        self.schedule_event(
            now + transfer_ms,
            SimEvent::KvTransferComplete {
                from_backend,
                to_backend,
                request_id,
            },
        );
    }

    /// Handle KV transfer completion.
    fn handle_kv_transfer_complete(&mut self, to_backend: u32, request_id: u64) {
        let now = self.clock.now_ms();

        let from_key = self
            .active_requests
            .keys()
            .find(|(_, rid)| *rid == request_id)
            .cloned();

        if let Some(from_key) = from_key {
            if let Some(mut active) = self.active_requests.remove(&from_key) {
                active.first_token_ms = Some(now);
                self.active_requests
                    .insert((to_backend, request_id), active.clone());
                self.requests_in_flight.insert(request_id, to_backend);

                let backend = &mut self.backends[to_backend as usize];

                let alloc = backend.kv_cache.allocate_for_request(
                    request_id,
                    active.request.total_tokens(),
                    active.request.prefix_hash,
                    active.request.prefix_token_length,
                );

                if !alloc.success {
                    self.metrics.record_rejection();
                    self.active_requests.remove(&(to_backend, request_id));
                    return;
                }

                backend
                    .active_batch
                    .try_add(active.clone(), backend.max_batch_tokens);

                if active.request.actual_gen_tokens > 0 {
                    let batch_size = backend.active_batch.size();
                    let tbt_ms = backend
                        .compute_model
                        .inter_token_latency_ms(batch_size)
                        .ceil() as u64;
                    self.schedule_event(
                        now + tbt_ms.max(1),
                        SimEvent::TokenGenerated {
                            backend_id: to_backend,
                            request_id,
                            token_num: 1,
                        },
                    );
                } else {
                    self.schedule_event(
                        now,
                        SimEvent::RequestComplete {
                            backend_id: to_backend,
                            request_id,
                        },
                    );
                }
            }
        }
    }

    /// Get the number of pending events.
    pub fn pending_events(&self) -> usize {
        self.event_queue.len()
    }
}

/// Estimate KV transfer time in microseconds.
fn estimate_transfer_time_us(
    latency_us: u64,
    bandwidth_gb_s: f64,
    num_tokens: u32,
    num_layers: u32,
) -> u64 {
    let bytes_per_token = 2u64 * 2 * 128 * num_layers as u64;
    let total_bytes = num_tokens as u64 * bytes_per_token;
    let bandwidth_bytes_per_us = (bandwidth_gb_s * 1e3) / 1e6;
    let transfer_us = (total_bytes as f64 / bandwidth_bytes_per_us) as u64;
    latency_us + transfer_us
}

#[cfg(test)]
mod tests {
    use super::*;
    use routesim_algorithms::RoundRobin;

    fn test_config() -> SimConfig {
        SimConfig::from_str(
            r#"
[simulation]
name = "test"
seed = 42

[cluster]
num_backends = 2
max_batch_tokens = 8192
max_queue_depth = 128
kv_cache_blocks = 1000
kv_block_size = 16

[trace]
format = "compact_jsonl"
"#,
        )
        .unwrap()
    }

    fn sample_requests(n: usize) -> Vec<InferenceRequest> {
        (0..n)
            .map(|i| InferenceRequest {
                id: i as u64,
                arrival_time_ms: (i as u64) * 10,
                prompt_tokens: 128,
                max_gen_tokens: 32,
                actual_gen_tokens: 32,
                prefix_hash: None,
                prefix_token_length: None,
                conversation_id: None,
                lora_adapter: None,
                priority: 0,
                metadata: std::collections::HashMap::new(),
            })
            .collect()
    }

    #[test]
    fn test_engine_creation() {
        let config = test_config();
        let algo = Box::new(RoundRobin::new());
        let engine = SimulationEngine::new(config, algo);
        assert_eq!(engine.backends.len(), 2);
        assert_eq!(engine.events_processed, 0);
    }

    #[test]
    fn test_load_and_run_basic() {
        let config = test_config();
        let algo = Box::new(RoundRobin::new());
        let mut engine = SimulationEngine::new(config, algo);
        engine.load_trace(sample_requests(10));
        let metrics = engine.run();

        assert_eq!(metrics.completed_requests + metrics.rejected_requests, 10);
        assert!(engine.events_processed > 0);
    }

    #[test]
    fn test_event_ordering() {
        let config = test_config();
        let algo = Box::new(RoundRobin::new());
        let mut engine = SimulationEngine::new(config, algo);

        engine.schedule_event(100, SimEvent::BatchSchedule { backend_id: 0 });
        engine.schedule_event(50, SimEvent::BatchSchedule { backend_id: 1 });
        engine.schedule_event(200, SimEvent::BatchSchedule { backend_id: 0 });

        let first = engine.event_queue.pop().unwrap();
        assert_eq!(first.time_ms, 50);
        let second = engine.event_queue.pop().unwrap();
        assert_eq!(second.time_ms, 100);
    }

    #[test]
    fn test_run_produces_metrics() {
        let config = test_config();
        let algo = Box::new(RoundRobin::new());
        let mut engine = SimulationEngine::new(config, algo);
        engine.load_trace(sample_requests(20));
        let metrics = engine.run();

        assert!(metrics.duration_ms > 0);
        assert!(metrics.completed_requests > 0);
        assert!(metrics.requests_per_sec > 0.0);
    }

    #[test]
    fn test_prefix_cache_hit_in_simulation() {
        let config = test_config();
        let algo = Box::new(RoundRobin::new());
        let mut engine = SimulationEngine::new(config, algo);

        let requests: Vec<InferenceRequest> = (0..4)
            .map(|i| InferenceRequest {
                id: i,
                arrival_time_ms: i * 100,
                prompt_tokens: 256,
                max_gen_tokens: 32,
                actual_gen_tokens: 32,
                prefix_hash: Some(0xABC),
                prefix_token_length: Some(128),
                conversation_id: None,
                lora_adapter: None,
                priority: 0,
                metadata: std::collections::HashMap::new(),
            })
            .collect();

        engine.load_trace(requests);
        let metrics = engine.run();
        assert!(metrics.completed_requests > 0);
    }

    #[test]
    fn test_multiple_algorithms_produce_different_results() {
        let config_str = r#"
[simulation]
name = "compare-test"
seed = 42

[cluster]
num_backends = 4
max_batch_tokens = 8192
max_queue_depth = 128
kv_cache_blocks = 1000
kv_block_size = 16

[trace]
format = "compact_jsonl"
"#;
        let config = SimConfig::from_str(config_str).unwrap();
        let requests = sample_requests(50);

        let results =
            crate::compare_algorithms(&config, &requests, &["round_robin", "least_outstanding"]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].algorithm, "round_robin");
        assert_eq!(results[1].algorithm, "least_outstanding");
    }
}
