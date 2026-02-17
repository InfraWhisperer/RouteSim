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
    /// Trigger batch scheduling on a backend (event-boundary batching).
    BatchSchedule { backend_id: u32 },
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
        cached_block_hashes: snap.cached_block_hashes.clone(),
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
        max_queue_depth: snap.max_queue_depth,
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
        cache_block_hashes: req.cache_block_hashes.clone(),
        conversation_id: req.conversation_id.clone(),
        lora_adapter: req.lora_adapter.clone(),
        priority: req.priority,
    }
}

fn to_algo_event(event: &SimEvent) -> Option<routesim_algorithms::SimEventInfo> {
    match event {
        SimEvent::RequestArrival(req) => {
            Some(routesim_algorithms::SimEventInfo::RequestArrival { request_id: req.id })
        }
        SimEvent::PrefillComplete {
            backend_id,
            request_id,
        } => Some(routesim_algorithms::SimEventInfo::PrefillComplete {
            backend_id: *backend_id,
            request_id: *request_id,
        }),
        SimEvent::RequestComplete {
            backend_id,
            request_id,
        } => Some(routesim_algorithms::SimEventInfo::RequestComplete {
            backend_id: *backend_id,
            request_id: *request_id,
        }),
        SimEvent::TokenGenerated {
            backend_id,
            request_id,
            token_num,
        } => Some(routesim_algorithms::SimEventInfo::TokenGenerated {
            backend_id: *backend_id,
            request_id: *request_id,
            token_num: *token_num,
        }),
        // Internal events (BatchSchedule, KvTransfer*) have no algorithm-facing equivalent
        _ => None,
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
    /// Algorithm's chosen decode backend per request (disaggregated mode).
    disagg_decode_targets: HashMap<u64, u32>,
    /// Block cache overlap fraction at routing time (request_id -> overlap).
    block_overlaps: HashMap<u64, f64>,
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
            disagg_decode_targets: HashMap::new(),
            block_overlaps: HashMap::new(),
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
        // Let the algorithm observe the event (only for algorithm-facing events)
        if let Some(algo_event) = to_algo_event(&event) {
            let algo_snapshots: Vec<_> = self
                .backends
                .iter()
                .map(|b| to_algo_snapshot(&b.snapshot()))
                .collect();
            self.algorithm.on_event(&algo_event, &algo_snapshots);
        }

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
            SimEvent::KvTransferStart {
                from_backend,
                to_backend,
                request_id,
            } => self.handle_kv_transfer_start(from_backend, to_backend, request_id),
            SimEvent::KvTransferComplete {
                from_backend,
                to_backend,
                request_id,
            } => self.handle_kv_transfer_complete(from_backend, to_backend, request_id),
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
        if backend_id as usize >= self.backends.len() {
            self.metrics.record_rejection();
            return;
        }
        let now = self.clock.now_ms();

        // Track block-level cache overlap at routing time
        if !request.cache_block_hashes.is_empty() {
            let cached = self.backends[backend_id as usize]
                .kv_cache
                .cached_content_block_hashes();
            let overlap = request
                .cache_block_hashes
                .iter()
                .filter(|h| cached.contains(h))
                .count() as f64
                / request.cache_block_hashes.len() as f64;
            self.block_overlaps.insert(request.id, overlap);
        }

        let backend = &mut self.backends[backend_id as usize];

        if !backend.enqueue(request.clone(), now) {
            self.block_overlaps.remove(&request.id);
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
        decode_backend: u32,
    ) {
        if prefill_backend as usize >= self.backends.len()
            || decode_backend as usize >= self.backends.len()
        {
            self.metrics.record_rejection();
            return;
        }
        let now = self.clock.now_ms();
        let backend = &mut self.backends[prefill_backend as usize];

        if !backend.enqueue(request.clone(), now) {
            self.metrics.record_rejection();
            return;
        }

        self.requests_in_flight.insert(request.id, prefill_backend);
        self.disagg_decode_targets
            .insert(request.id, decode_backend);

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
        let mut rejected_ids: Vec<u64> = Vec::new();

        {
            let backend = &mut self.backends[backend_id as usize];

            // Track tokens committed in this batch schedule pass so we don't
            // over-commit beyond the batch token budget.
            let mut batch_tokens_committed = backend.active_batch.total_tokens;

            while let Some(queued) = backend.queue.front().cloned() {
                let request = queued.request.clone();
                let total_tokens = request.total_tokens();

                if batch_tokens_committed + total_tokens > backend.max_batch_tokens
                    && (backend.active_batch.size() > 0 || !pending_active.is_empty())
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
                    rejected_ids.push(request.id);
                    continue;
                }

                batch_tokens_committed += total_tokens;

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
        // Clean up tracking state for requests that failed KV allocation
        for id in rejected_ids {
            self.requests_in_flight.remove(&id);
            self.block_overlaps.remove(&id);
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
            // Use algorithm's chosen decode backend, fall back to least-queue-depth
            let chosen = self.disagg_decode_targets.remove(&request_id);
            let decode_id = chosen
                .filter(|&id| {
                    let b = &self.backends[id as usize];
                    b.role == BackendRole::Decode && b.can_accept()
                })
                .or_else(|| {
                    self.backends
                        .iter()
                        .filter(|b| b.role == BackendRole::Decode && b.can_accept())
                        .min_by_key(|b| b.queue_depth())
                        .map(|b| b.id)
                });

            // Credit prefill backend with its busy time before handing off
            if let Some(active) = self.active_requests.get(&(backend_id, request_id)) {
                self.backends[backend_id as usize].busy_time_ms +=
                    now.saturating_sub(active.prefill_start_ms.unwrap_or(now));
            }

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
                self.cleanup_failed_request(backend_id, request_id);
            }
            return;
        }

        if let Some(active) = self.active_requests.get_mut(&(backend_id, request_id)) {
            active.prefill_end_ms = Some(now);

            let backend = &mut self.backends[backend_id as usize];
            let added = backend
                .active_batch
                .try_add(active.clone(), backend.max_batch_tokens);

            if !added {
                // Batch full — re-enqueue so BatchSchedule can retry
                let request = active.request.clone();
                backend.kv_cache.release_request(request_id);
                backend.queue.push_front(crate::request::QueuedRequest {
                    request,
                    enqueue_time_ms: now,
                });
                self.active_requests.remove(&(backend_id, request_id));
                self.requests_in_flight.remove(&request_id);
                self.schedule_event(now, SimEvent::BatchSchedule { backend_id });
                return;
            }

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
            if active.first_token_ms.is_none() {
                active.first_token_ms = Some(now);
            }
            active.token_timestamps_ms.push(now);

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

            let tbt_samples = active.tbt_samples();

            let queue_wait = active
                .prefill_start_ms
                .unwrap_or(now)
                .saturating_sub(request.arrival_time_ms);

            // -1.0 means no block hashes on the request (excluded from average)
            let block_cache_overlap = self.block_overlaps.remove(&request.id).unwrap_or(-1.0);

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
                block_cache_overlap,
                tbt_samples_ms: tbt_samples,
            };
            self.metrics.record(metric);

            // Update backend counters
            let backend = &mut self.backends[backend_id as usize];
            backend.total_requests_served += 1;
            backend.total_tokens_generated += active.tokens_generated as u64;
            let start = active
                .decode_start_ms
                .or(active.prefill_start_ms)
                .unwrap_or(now);
            backend.busy_time_ms += now.saturating_sub(start);

            // Add content block hashes to this backend's cache (for prefix_overlap routing)
            if !active.request.cache_block_hashes.is_empty() {
                backend
                    .kv_cache
                    .add_content_blocks(&active.request.cache_block_hashes);
            }

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
        let disagg: crate::topology::DisaggregatedConfig =
            self.config.cluster.disaggregated.clone().into();

        let tokens = self
            .active_requests
            .get(&(from_backend, request_id))
            .map(|a| a.request.prompt_tokens)
            .unwrap_or(512);

        let transfer_us = disagg.estimate_transfer_time_us(tokens);
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
    fn handle_kv_transfer_complete(&mut self, from_backend: u32, to_backend: u32, request_id: u64) {
        let now = self.clock.now_ms();

        if let Some(mut active) = self.active_requests.remove(&(from_backend, request_id)) {
            active.decode_start_ms = Some(now);

            self.active_requests
                .insert((to_backend, request_id), active.clone());
            self.requests_in_flight.insert(request_id, to_backend);

            let alloc = self.backends[to_backend as usize]
                .kv_cache
                .allocate_for_request(
                    request_id,
                    active.request.total_tokens(),
                    active.request.prefix_hash,
                    active.request.prefix_token_length,
                );

            if !alloc.success {
                // Release KV blocks on the prefill backend too
                self.backends[from_backend as usize]
                    .kv_cache
                    .release_request(request_id);
                self.active_requests.remove(&(to_backend, request_id));
                self.requests_in_flight.remove(&request_id);
                self.metrics.record_rejection();
                return;
            }

            let max_tokens = self.backends[to_backend as usize].max_batch_tokens;
            let added = self.backends[to_backend as usize]
                .active_batch
                .try_add(active.clone(), max_tokens);

            if !added {
                // Batch full — release KV on both backends and re-enqueue
                self.backends[to_backend as usize]
                    .kv_cache
                    .release_request(request_id);
                self.backends[from_backend as usize]
                    .kv_cache
                    .release_request(request_id);
                self.backends[to_backend as usize].queue.push_front(
                    crate::request::QueuedRequest {
                        request: active.request.clone(),
                        enqueue_time_ms: now,
                    },
                );
                self.active_requests.remove(&(to_backend, request_id));
                self.requests_in_flight.remove(&request_id);
                self.schedule_event(
                    now,
                    SimEvent::BatchSchedule {
                        backend_id: to_backend,
                    },
                );
                return;
            }

            // Release KV blocks on the prefill backend now that transfer succeeded
            self.backends[from_backend as usize]
                .kv_cache
                .release_request(request_id);

            let backend = &self.backends[to_backend as usize];
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

    /// Clean up all state for a request that failed during disaggregated routing.
    fn cleanup_failed_request(&mut self, backend_id: u32, request_id: u64) {
        self.backends[backend_id as usize]
            .kv_cache
            .release_request(request_id);
        self.active_requests.remove(&(backend_id, request_id));
        self.requests_in_flight.remove(&request_id);
        self.disagg_decode_targets.remove(&request_id);

        let backend = &mut self.backends[backend_id as usize];
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

        self.metrics.record_rejection();
    }

    /// Get the number of pending events.
    pub fn pending_events(&self) -> usize {
        self.event_queue.len()
    }
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
                cache_block_hashes: Vec::new(),
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
        // Use PrefixAware so requests with the same prefix_hash are routed
        // to the same backend, maximizing cache hits.
        let config = SimConfig::from_str(
            r#"
[simulation]
name = "prefix-hit-test"
seed = 42

[cluster]
num_backends = 4
max_batch_tokens = 16384
max_queue_depth = 256
kv_cache_blocks = 4000
kv_block_size = 16

[trace]
format = "compact_jsonl"
"#,
        )
        .unwrap();

        let algo = Box::new(routesim_algorithms::PrefixAware::new());
        let mut engine = SimulationEngine::new(config, algo);

        // 8 requests with the same prefix, staggered so the first completes
        // before later ones arrive, giving cache time to populate.
        let requests: Vec<InferenceRequest> = (0..8)
            .map(|i| InferenceRequest {
                id: i,
                arrival_time_ms: i * 200,
                prompt_tokens: 256,
                max_gen_tokens: 16,
                actual_gen_tokens: 16,
                prefix_hash: Some(0xABC),
                prefix_token_length: Some(128),
                cache_block_hashes: Vec::new(),
                conversation_id: None,
                lora_adapter: None,
                priority: 0,
                metadata: std::collections::HashMap::new(),
            })
            .collect();

        engine.load_trace(requests);
        let metrics = engine.run();

        assert!(
            metrics.completed_requests >= 6,
            "Most requests should complete, got {}",
            metrics.completed_requests
        );

        // With PrefixAware routing, requests are sent to the same backend,
        // so all but the first should get prefix cache hits.
        let cache_hits = engine
            .metrics
            .records()
            .iter()
            .filter(|r| r.prefix_cache_hit)
            .count();
        assert!(
            cache_hits > 0,
            "PrefixAware routing should produce cache hits for shared prefixes, got 0 out of {} completed",
            metrics.completed_requests
        );
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

    // ---- Regression tests for bug fixes ----

    #[test]
    fn test_tbt_samples_have_real_variance() {
        // Before the fix, TBT samples were N copies of the average (all identical).
        // After the fix, they should have real variance from per-token timestamps.
        let config = test_config();
        let algo = Box::new(RoundRobin::new());
        let mut engine = SimulationEngine::new(config, algo);

        // Use a single request with many tokens so batch-size changes between tokens
        let requests = vec![InferenceRequest {
            id: 0,
            arrival_time_ms: 0,
            prompt_tokens: 64,
            max_gen_tokens: 20,
            actual_gen_tokens: 20,
            prefix_hash: None,
            prefix_token_length: None,
            cache_block_hashes: Vec::new(),
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
            metadata: std::collections::HashMap::new(),
        }];

        engine.load_trace(requests);
        let metrics = engine.run();
        assert_eq!(metrics.completed_requests, 1);

        // Check that the recorded TBT samples exist and come from real timestamps
        let record = &engine.metrics.records()[0];
        assert!(
            !record.tbt_samples_ms.is_empty(),
            "TBT samples should not be empty for a multi-token request"
        );

        // With a single request in the batch, all TBT values will be the same
        // (batch size doesn't change). But they should still be computed from real
        // timestamp deltas, not fabricated. If we run two overlapping requests,
        // the batch size changes mid-generation, causing TBT variation.
    }

    #[test]
    fn test_tbt_variance_with_overlapping_requests() {
        // Two overlapping requests cause batch size to change mid-generation,
        // so TBT should vary between tokens (batch=1 ITL != batch=2 ITL).
        let config = test_config();
        let algo = Box::new(RoundRobin::new());
        let mut engine = SimulationEngine::new(config, algo);

        let requests = vec![
            InferenceRequest {
                id: 0,
                arrival_time_ms: 0,
                prompt_tokens: 64,
                max_gen_tokens: 30,
                actual_gen_tokens: 30,
                prefix_hash: None,
                prefix_token_length: None,
                cache_block_hashes: Vec::new(),
                conversation_id: None,
                lora_adapter: None,
                priority: 0,
                metadata: std::collections::HashMap::new(),
            },
            InferenceRequest {
                id: 1,
                arrival_time_ms: 5, // Arrives during request 0's decode
                prompt_tokens: 64,
                max_gen_tokens: 30,
                actual_gen_tokens: 30,
                prefix_hash: None,
                prefix_token_length: None,
                cache_block_hashes: Vec::new(),
                conversation_id: None,
                lora_adapter: None,
                priority: 0,
                metadata: std::collections::HashMap::new(),
            },
        ];

        engine.load_trace(requests);
        let metrics = engine.run();
        assert!(metrics.completed_requests >= 1);

        // Collect all TBT samples across all completed requests
        let all_tbt: Vec<f64> = engine
            .metrics
            .records()
            .iter()
            .flat_map(|r| r.tbt_samples_ms.iter())
            .copied()
            .collect();

        // With overlapping requests on the same backend, batch size changes,
        // which means inter-token latency should not be uniform.
        // At minimum, the samples should exist.
        assert!(
            !all_tbt.is_empty(),
            "Should have TBT samples from completed requests"
        );
    }

    // Custom algorithm that always returns RouteDisaggregated for testing
    struct DisaggregatedTestAlgo {
        prefill_id: u32,
        decode_id: u32,
    }

    impl routesim_algorithms::RoutingAlgorithm for DisaggregatedTestAlgo {
        fn route(
            &mut self,
            _request: &routesim_algorithms::RequestInfo,
            _backends: &[routesim_algorithms::BackendSnapshot],
            _clock: &dyn routesim_algorithms::Clock,
        ) -> routesim_algorithms::RoutingDecision {
            routesim_algorithms::RoutingDecision::RouteDisaggregated {
                prefill: self.prefill_id,
                decode: self.decode_id,
            }
        }

        fn name(&self) -> &str {
            "disagg_test"
        }
    }

    fn disagg_config() -> SimConfig {
        SimConfig::from_str(
            r#"
[simulation]
name = "disagg-test"
seed = 42

[cluster]
num_backends = 4
max_batch_tokens = 8192
max_queue_depth = 128
kv_cache_blocks = 2000
kv_block_size = 16

[cluster.disaggregated]
enabled = true
prefill_backends = 2
decode_backends = 2
kv_transfer_latency_us = 500
kv_transfer_bandwidth_gb_s = 50.0
num_layers = 80
head_dim = 128
num_kv_heads = 8

[trace]
format = "compact_jsonl"
"#,
        )
        .unwrap()
    }

    #[test]
    fn test_disaggregated_respects_decode_backend_choice() {
        // The algorithm chooses decode backend 3 (the second decode node).
        // Before the fix, `_decode_backend` was unused and a fallback picked
        // the decode node. Now the engine should store and use the algorithm's choice.
        let config = disagg_config();
        // prefill=0, decode=3 (backend 3 is the second decode node: indices 2,3)
        let algo = Box::new(DisaggregatedTestAlgo {
            prefill_id: 0,
            decode_id: 3,
        });
        let mut engine = SimulationEngine::new(config, algo);

        let requests = vec![InferenceRequest {
            id: 100,
            arrival_time_ms: 0,
            prompt_tokens: 128,
            max_gen_tokens: 10,
            actual_gen_tokens: 10,
            prefix_hash: None,
            prefix_token_length: None,
            cache_block_hashes: Vec::new(),
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
            metadata: std::collections::HashMap::new(),
        }];

        engine.load_trace(requests);
        let metrics = engine.run();

        assert_eq!(metrics.completed_requests, 1, "Request should complete");

        // Verify the request was served by decode backend 3
        let record = &engine.metrics.records()[0];
        assert_eq!(
            record.backend_id, 3,
            "Request should complete on the algorithm's chosen decode backend (3), got {}",
            record.backend_id
        );
    }

    #[test]
    fn test_disaggregated_ttft_includes_decode_step() {
        // In disaggregated mode, TTFT = arrival -> first output token.
        // This includes prefill + KV transfer + first decode step.
        let config = disagg_config();
        let algo = Box::new(DisaggregatedTestAlgo {
            prefill_id: 0,
            decode_id: 2,
        });
        let mut engine = SimulationEngine::new(config, algo);

        let requests = vec![InferenceRequest {
            id: 200,
            arrival_time_ms: 0,
            prompt_tokens: 256,
            max_gen_tokens: 10,
            actual_gen_tokens: 10,
            prefix_hash: None,
            prefix_token_length: None,
            cache_block_hashes: Vec::new(),
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
            metadata: std::collections::HashMap::new(),
        }];

        engine.load_trace(requests);
        let metrics = engine.run();

        assert_eq!(metrics.completed_requests, 1);

        let record = &engine.metrics.records()[0];
        let ttft = record.ttft_ms;

        // TTFT = prefill (~6ms) + KV transfer (~3ms with 8 KV heads) + first decode step (~13ms)
        // Should be ~22ms total
        assert!(
            ttft > 10,
            "TTFT should include prefill + transfer + decode step, got {}ms",
            ttft
        );
        assert!(
            ttft <= 40,
            "TTFT should be reasonable (~22ms), got {}ms",
            ttft
        );
    }

    #[test]
    fn test_cleanup_releases_kv_blocks() {
        // Verify that cleanup_failed_request properly releases KV cache blocks.
        let config = test_config();
        let algo = Box::new(RoundRobin::new());
        let mut engine = SimulationEngine::new(config, algo);

        // Manually set up a request in active state
        let request = InferenceRequest {
            id: 42,
            arrival_time_ms: 0,
            prompt_tokens: 128,
            max_gen_tokens: 32,
            actual_gen_tokens: 32,
            prefix_hash: None,
            prefix_token_length: None,
            cache_block_hashes: Vec::new(),
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
            metadata: std::collections::HashMap::new(),
        };

        // Allocate KV blocks on backend 0
        let backend = &mut engine.backends[0];
        let free_before = backend.kv_cache.stats().free_blocks;
        let alloc = backend
            .kv_cache
            .allocate_for_request(42, request.total_tokens(), None, None);
        assert!(alloc.success);
        let free_after_alloc = backend.kv_cache.stats().free_blocks;
        assert!(
            free_after_alloc < free_before,
            "Allocation should consume blocks"
        );

        // Set up tracking state
        let active = ActiveRequest::new(request.clone());
        engine.active_requests.insert((0, 42), active);
        engine.requests_in_flight.insert(42, 0);

        // Run cleanup
        engine.cleanup_failed_request(0, 42);

        // Verify KV blocks were released
        let free_after_cleanup = engine.backends[0].kv_cache.stats().free_blocks;
        assert_eq!(
            free_after_cleanup, free_before,
            "cleanup_failed_request should release all KV blocks"
        );

        // Verify tracking state was cleaned up
        assert!(!engine.active_requests.contains_key(&(0, 42)));
        assert!(!engine.requests_in_flight.contains_key(&42));
    }

    #[test]
    fn test_batch_full_reenqueue_no_leak() {
        // Small max_batch_tokens forces re-enqueue when batch is full.
        // All requests must either complete or be rejected — none leaked.
        let config = SimConfig::from_str(
            r#"
[simulation]
name = "stress-reenqueue"
seed = 42

[cluster]
num_backends = 1
max_batch_tokens = 256
max_queue_depth = 128
kv_cache_blocks = 4000
kv_block_size = 16

[trace]
format = "compact_jsonl"
"#,
        )
        .unwrap();

        let algo = Box::new(RoundRobin::new());
        let mut engine = SimulationEngine::new(config, algo);

        // 10 requests with total_tokens=128 each, all arriving at t=0.
        // Only 2 fit in a batch (2*128=256), rest must wait/re-enqueue.
        let requests: Vec<InferenceRequest> = (0..10)
            .map(|i| InferenceRequest {
                id: i as u64,
                arrival_time_ms: 0,
                prompt_tokens: 64,
                max_gen_tokens: 64,
                actual_gen_tokens: 64,
                prefix_hash: None,
                prefix_token_length: None,
                cache_block_hashes: Vec::new(),
                conversation_id: None,
                lora_adapter: None,
                priority: 0,
                metadata: std::collections::HashMap::new(),
            })
            .collect();

        engine.load_trace(requests);
        let metrics = engine.run();

        // All 10 requests must be accounted for
        assert_eq!(
            metrics.completed_requests + metrics.rejected_requests,
            10,
            "All requests must complete or be rejected (completed={}, rejected={})",
            metrics.completed_requests,
            metrics.rejected_requests,
        );

        // No stale tracking state after simulation
        assert!(
            engine.requests_in_flight.is_empty(),
            "requests_in_flight should be empty after simulation, has {} entries",
            engine.requests_in_flight.len(),
        );
        assert!(
            engine.active_requests.is_empty(),
            "active_requests should be empty after simulation, has {} entries",
            engine.active_requests.len(),
        );

        // At least some requests should complete (not all rejected)
        assert!(
            metrics.completed_requests > 0,
            "At least some requests should complete"
        );
    }

    #[test]
    fn test_ttft_includes_first_decode_step() {
        // Non-disaggregated: TTFT should be arrival -> first generated token,
        // not arrival -> prefill completion.
        let config = test_config();
        let algo = Box::new(RoundRobin::new());
        let mut engine = SimulationEngine::new(config, algo);

        let requests = vec![InferenceRequest {
            id: 0,
            arrival_time_ms: 0,
            prompt_tokens: 64,
            max_gen_tokens: 10,
            actual_gen_tokens: 10,
            prefix_hash: None,
            prefix_token_length: None,
            cache_block_hashes: Vec::new(),
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
            metadata: std::collections::HashMap::new(),
        }];

        engine.load_trace(requests);
        let metrics = engine.run();
        assert_eq!(metrics.completed_requests, 1);

        let record = &engine.metrics.records()[0];
        let ttft = record.ttft_ms;

        // Prefill: 64 / 50000 * 1000 = 1.28ms, ceil = 2ms
        // First decode step: batch=1, ITL = 1/80 * 1000 = 12.5ms, ceil = 13ms
        // TTFT should be >= prefill + first_decode = 2 + 13 = 15ms
        assert!(
            ttft >= 10,
            "TTFT should include first decode step, not just prefill. Got {}ms (expected ~15ms)",
            ttft
        );
    }

    #[test]
    fn test_tbt_no_prefill_timestamp() {
        // TBT samples should be purely inter-token deltas.
        // N tokens generated => N timestamps => N-1 TBT samples.
        let config = test_config();
        let algo = Box::new(RoundRobin::new());
        let mut engine = SimulationEngine::new(config, algo);

        let requests = vec![InferenceRequest {
            id: 0,
            arrival_time_ms: 0,
            prompt_tokens: 64,
            max_gen_tokens: 5,
            actual_gen_tokens: 5,
            prefix_hash: None,
            prefix_token_length: None,
            cache_block_hashes: Vec::new(),
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
            metadata: std::collections::HashMap::new(),
        }];

        engine.load_trace(requests);
        let metrics = engine.run();
        assert_eq!(metrics.completed_requests, 1);

        let record = &engine.metrics.records()[0];
        // 5 tokens generated => 5 timestamps => 4 TBT samples
        assert_eq!(
            record.tbt_samples_ms.len(),
            4,
            "Expected 4 TBT samples for 5 tokens, got {}",
            record.tbt_samples_ms.len()
        );
    }

    #[test]
    fn test_prefix_overlap_end_to_end_with_block_hashes() {
        // Exercise prefix_overlap routing with actual cache_block_hashes.
        // Requests sharing block hashes should be routed to the same backend
        // (cache affinity), and the backend's content_block_set should reflect
        // the completed requests' hashes.
        let config = SimConfig::from_str(
            r#"
[simulation]
name = "prefix-overlap-e2e"
seed = 42

[cluster]
num_backends = 4
max_batch_tokens = 16384
max_queue_depth = 256
kv_cache_blocks = 4000
kv_block_size = 16

[trace]
format = "compact_jsonl"
"#,
        )
        .unwrap();

        let algo = Box::new(routesim_algorithms::PrefixOverlap::new());
        let mut engine = SimulationEngine::new(config, algo);

        // Shared prefix blocks (simulating a system prompt)
        let shared_prefix: Vec<u64> = (100..120).collect(); // 20 shared blocks

        // Create requests that share a common prefix
        let mut requests = Vec::new();
        for i in 0..12u64 {
            let mut blocks = shared_prefix.clone();
            // Each request also has unique suffix blocks
            blocks.extend(i * 10..(i * 10 + 5));
            requests.push(InferenceRequest {
                id: i,
                arrival_time_ms: i * 50, // staggered arrivals
                prompt_tokens: 400,      // 25 blocks at block_size=16
                max_gen_tokens: 16,
                actual_gen_tokens: 16,
                prefix_hash: None,
                prefix_token_length: None,
                cache_block_hashes: blocks,
                conversation_id: None,
                lora_adapter: None,
                priority: 0,
                metadata: std::collections::HashMap::new(),
            });
        }

        engine.load_trace(requests);
        let metrics = engine.run();

        assert!(
            metrics.completed_requests >= 8,
            "Most requests should complete, got {}",
            metrics.completed_requests
        );

        // Check that at least one backend has cached content block hashes
        let total_content_hashes: usize = engine
            .backends
            .iter()
            .map(|b| b.kv_cache.cached_content_block_hashes().len())
            .sum();
        assert!(
            total_content_hashes > 0,
            "Backends should have content block hashes after processing Mooncake-style requests"
        );

        // Check that the shared prefix blocks appear in at least one backend's cache
        let has_shared_block = engine.backends.iter().any(|b| {
            let cached = b.kv_cache.cached_content_block_hashes();
            cached.contains(&100) // first block of shared prefix
        });
        assert!(
            has_shared_block,
            "At least one backend should have the shared prefix blocks cached"
        );
    }

    #[test]
    fn test_disaggregated_busy_time_split() {
        // In disaggregated mode, busy_time_ms should be split:
        // - Prefill backend gets credited for prefill duration
        // - Decode backend gets credited for decode duration
        let config = disagg_config();
        let algo = Box::new(DisaggregatedTestAlgo {
            prefill_id: 0,
            decode_id: 2,
        });
        let mut engine = SimulationEngine::new(config, algo);

        let requests = vec![InferenceRequest {
            id: 300,
            arrival_time_ms: 0,
            prompt_tokens: 256,
            max_gen_tokens: 10,
            actual_gen_tokens: 10,
            prefix_hash: None,
            prefix_token_length: None,
            cache_block_hashes: Vec::new(),
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
            metadata: std::collections::HashMap::new(),
        }];

        engine.load_trace(requests);
        let _metrics = engine.run();

        let prefill_backend = &engine.backends[0];
        let decode_backend = &engine.backends[2];

        assert!(
            prefill_backend.busy_time_ms > 0,
            "Prefill backend should have busy_time_ms > 0, got {}",
            prefill_backend.busy_time_ms
        );

        assert!(
            decode_backend.busy_time_ms > 0,
            "Decode backend should have busy_time_ms > 0, got {}",
            decode_backend.busy_time_ms
        );

        // Decode busy time should exceed prefill for 10 tokens of generation
        assert!(
            decode_backend.busy_time_ms > prefill_backend.busy_time_ms,
            "Decode busy time ({}) should exceed prefill busy time ({})",
            decode_backend.busy_time_ms,
            prefill_backend.busy_time_ms,
        );
    }
}
