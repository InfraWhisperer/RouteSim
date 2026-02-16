//! Routing algorithm trait definitions.
//!
//! All routing algorithms implement the [`RoutingAlgorithm`] trait, which
//! receives request information and backend snapshots to make routing decisions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Read-only snapshot of a backend's state, provided to routing algorithms.
///
/// This is the algorithms crate's view of a backend — it contains only the
/// information needed for routing decisions, not the full simulation state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSnapshot {
    pub id: u32,
    pub queue_depth: u32,
    pub active_batch_size: u32,
    pub active_batch_tokens: u32,
    pub kv_cache_utilization: f32,
    pub prefix_hashes_cached: std::collections::HashSet<u64>,
    pub estimated_ttft_ms: f64,
    pub tokens_per_sec_current: f64,
    pub role: BackendRole,
    pub state: BackendState,
    pub lora_adapters_loaded: Vec<String>,
    pub total_requests_served: u64,
    pub total_tokens_generated: u64,
}

/// Role a backend plays in the cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendRole {
    Both,
    Prefill,
    Decode,
}

/// State of a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendState {
    Idle,
    Processing,
    Draining,
    Offline,
}

/// Virtual simulation clock interface for algorithms.
pub trait Clock {
    fn now_ms(&self) -> u64;
}

/// Decision returned by a routing algorithm.
#[derive(Debug, Clone)]
pub enum RoutingDecision {
    /// Route to a specific backend.
    Route(u32),
    /// Route with a priority override.
    RouteWithPriority(u32, u8),
    /// Reject the request (no suitable backend).
    Reject,
    /// Route with disaggregated prefill/decode split.
    RouteDisaggregated { prefill: u32, decode: u32 },
}

/// Information about an incoming request, provided to routing algorithms.
#[derive(Debug, Clone)]
pub struct RequestInfo {
    pub id: u64,
    pub prompt_tokens: u32,
    pub max_gen_tokens: u32,
    pub actual_gen_tokens: u32,
    pub prefix_hash: Option<u64>,
    pub prefix_token_length: Option<u32>,
    pub conversation_id: Option<String>,
    pub lora_adapter: Option<String>,
    pub priority: u8,
}

/// Simulation event information passed to algorithms for state updates.
#[derive(Debug, Clone)]
pub enum SimEventInfo {
    RequestArrival {
        request_id: u64,
    },
    PrefillComplete {
        backend_id: u32,
        request_id: u64,
    },
    RequestComplete {
        backend_id: u32,
        request_id: u64,
    },
    TokenGenerated {
        backend_id: u32,
        request_id: u64,
        token_num: u32,
    },
}

/// The core routing algorithm trait.
///
/// Implement this trait to create custom load balancing strategies.
/// The simulator calls [`route`] for each incoming request and [`on_event`]
/// after each simulation event.
pub trait RoutingAlgorithm: Send + Sync {
    /// Called for each incoming request. Returns the backend ID to route to.
    fn route(
        &mut self,
        request: &RequestInfo,
        backends: &[BackendSnapshot],
        clock: &dyn Clock,
    ) -> RoutingDecision;

    /// Called after each event so the algorithm can update internal state.
    fn on_event(&mut self, _event: &SimEventInfo, _backends: &[BackendSnapshot]) {}

    /// Human-readable name for reports.
    fn name(&self) -> &str;

    /// Optional: algorithm-specific metrics to include in output.
    fn custom_metrics(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

/// Filter backends to only those that are available for routing.
pub fn available_backends(backends: &[BackendSnapshot]) -> Vec<&BackendSnapshot> {
    backends
        .iter()
        .filter(|b| b.state != BackendState::Draining && b.state != BackendState::Offline)
        .collect()
}

/// Filter backends to only those with a specific role.
pub fn backends_with_role(
    backends: &[BackendSnapshot],
    role: BackendRole,
) -> Vec<&BackendSnapshot> {
    backends
        .iter()
        .filter(|b| {
            b.role == role && b.state != BackendState::Draining && b.state != BackendState::Offline
        })
        .collect()
}
