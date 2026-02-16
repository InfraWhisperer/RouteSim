//! Built-in routing algorithms for RouteSim.
//!
//! This crate provides the [`RoutingAlgorithm`] trait and several built-in
//! implementations for LLM inference load balancing:
//!
//! | Algorithm | Strategy | Best For |
//! |-----------|----------|----------|
//! | [`RoundRobin`] | Cycle through backends | Uniform workloads |
//! | [`LeastOutstanding`] | Fewest queued requests | Variable request sizes |
//! | [`LeastKv`] | Lowest KV cache usage | Memory-constrained clusters |
//! | [`PrefixAware`] | Maximize cache hits | Shared system prompts |
//! | [`SessionAffinity`] | Sticky conversations | Multi-turn chat |
//! | [`CostEscalation`] | Dynamic cost model | Complex workloads |

pub mod cost_escalation;
pub mod least_kv;
pub mod least_outstanding;
pub mod prefix_aware;
pub mod round_robin;
pub mod session_affinity;
pub mod traits;

pub use cost_escalation::CostEscalation;
pub use least_kv::LeastKv;
pub use least_outstanding::LeastOutstanding;
pub use prefix_aware::PrefixAware;
pub use round_robin::RoundRobin;
pub use session_affinity::SessionAffinity;
pub use traits::*;

/// Create a routing algorithm by name.
pub fn algorithm_by_name(name: &str) -> Option<Box<dyn RoutingAlgorithm>> {
    match name {
        "round_robin" => Some(Box::new(RoundRobin::new())),
        "least_outstanding" => Some(Box::new(LeastOutstanding::new())),
        "least_kv" => Some(Box::new(LeastKv::new())),
        "prefix_aware" => Some(Box::new(PrefixAware::new())),
        "session_affinity" => Some(Box::new(SessionAffinity::new())),
        "cost_escalation" => Some(Box::new(CostEscalation::new())),
        _ => None,
    }
}

/// List all available built-in algorithm names.
pub fn available_algorithms() -> Vec<&'static str> {
    vec![
        "round_robin",
        "least_outstanding",
        "least_kv",
        "prefix_aware",
        "session_affinity",
        "cost_escalation",
    ]
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Helper to create N test backend snapshots.
    pub fn make_backends(n: u32) -> Vec<BackendSnapshot> {
        (0..n)
            .map(|i| BackendSnapshot {
                id: i,
                queue_depth: 0,
                active_batch_size: 0,
                active_batch_tokens: 0,
                kv_cache_utilization: 0.0,
                prefix_hashes_cached: HashSet::new(),
                estimated_ttft_ms: 10.0,
                tokens_per_sec_current: 100.0,
                role: BackendRole::Both,
                state: BackendState::Idle,
                lora_adapters_loaded: vec![],
                total_requests_served: 0,
                total_tokens_generated: 0,
            })
            .collect()
    }

    #[test]
    fn test_algorithm_by_name() {
        for name in available_algorithms() {
            assert!(algorithm_by_name(name).is_some(), "Missing: {}", name);
        }
        assert!(algorithm_by_name("nonexistent").is_none());
    }

    #[test]
    fn test_available_algorithms_not_empty() {
        assert!(!available_algorithms().is_empty());
    }
}
