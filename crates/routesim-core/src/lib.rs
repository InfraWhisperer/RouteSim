//! RouteSim — Discrete-event simulator for LLM inference load balancing.
//!
//! This crate provides the core simulation engine that models GPU backends,
//! KV caches, request queues, and the interactions between them. Routing
//! algorithms from `routesim-algorithms` are plugged in to make routing
//! decisions for each incoming request.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────┐     ┌───────────┐     ┌──────────────┐
//! │  Trace   │────▶│  Engine   │────▶│   Metrics    │
//! │ Ingestion│     │ (Events)  │     │  Collection  │
//! └──────────┘     └─────┬─────┘     └──────────────┘
//!                        │
//!                ┌───────┴───────┐
//!                │   Algorithm   │
//!                │   (Routing)   │
//!                └───────┬───────┘
//!                        │
//!          ┌─────────────┼─────────────┐
//!          ▼             ▼             ▼
//!    ┌──────────┐  ┌──────────┐  ┌──────────┐
//!    │ Backend 0│  │ Backend 1│  │ Backend N│
//!    │ KV Cache │  │ KV Cache │  │ KV Cache │
//!    │  Queue   │  │  Queue   │  │  Queue   │
//!    └──────────┘  └──────────┘  └──────────┘
//! ```

pub mod backend;
pub mod clock;
pub mod config;
pub mod engine;
pub mod kv_cache;
pub mod metrics;
pub mod request;
pub mod topology;
pub mod trace;

// Re-export key types for convenience.
pub use backend::{BackendSnapshot, ComputeModel, SimulatedBackend};
pub use clock::SimClock;
pub use config::SimConfig;
pub use engine::{SimEvent, SimulationEngine};
pub use kv_cache::KvCacheSimulator;
pub use metrics::{MetricsCollector, SimulationMetrics};
pub use request::InferenceRequest;
pub use topology::{BackendRole, BackendState, DisaggregatedConfig, GpuProfile};
pub use trace::{load_trace, write_compact_jsonl};

/// Run a complete simulation with the given config, trace, and algorithm.
pub fn run_simulation(
    config: SimConfig,
    requests: Vec<InferenceRequest>,
    algorithm: Box<dyn routesim_algorithms::RoutingAlgorithm>,
) -> SimulationMetrics {
    let mut engine = SimulationEngine::new(config, algorithm);
    engine.load_trace(requests);
    engine.run()
}

/// Run a comparison of multiple algorithms on the same trace and config.
pub fn compare_algorithms(
    config: &SimConfig,
    requests: &[InferenceRequest],
    algorithm_names: &[&str],
) -> Vec<SimulationMetrics> {
    algorithm_names
        .iter()
        .filter_map(|name| {
            let algo = routesim_algorithms::algorithm_by_name(name)?;
            let cfg = config.clone();
            let reqs = requests.to_vec();
            Some(run_simulation(cfg, reqs, algo))
        })
        .collect()
}
