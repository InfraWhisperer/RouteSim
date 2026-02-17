//! Least KV cache memory used routing algorithm.
//!
//! Routes requests to the backend with the lowest KV cache utilization.
//! This helps prevent memory pressure and reduces eviction rates.

use crate::traits::*;

/// Least KV cache utilization router.
pub struct LeastKv;

impl LeastKv {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LeastKv {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingAlgorithm for LeastKv {
    fn route(
        &mut self,
        _request: &RequestInfo,
        backends: &[BackendSnapshot],
        _clock: &dyn Clock,
    ) -> RoutingDecision {
        let available = available_backends(backends);
        if available.is_empty() {
            return RoutingDecision::Reject;
        }

        let best = available
            .iter()
            .min_by(|a, b| {
                a.kv_cache_utilization
                    .partial_cmp(&b.kv_cache_utilization)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        RoutingDecision::Route(best.id)
    }

    fn name(&self) -> &str {
        "least_kv"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_backends;

    struct FakeClock;
    impl Clock for FakeClock {
        fn now_ms(&self) -> u64 {
            0
        }
    }

    fn dummy_request() -> RequestInfo {
        RequestInfo {
            id: 0,
            prompt_tokens: 100,
            max_gen_tokens: 50,
            actual_gen_tokens: 50,
            prefix_hash: None,
            prefix_token_length: None,
            cache_block_hashes: Vec::new(),
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
        }
    }

    #[test]
    fn test_least_kv_picks_least_utilized() {
        let mut algo = LeastKv::new();
        let mut backends = make_backends(3);
        backends[0].kv_cache_utilization = 0.8;
        backends[1].kv_cache_utilization = 0.2;
        backends[2].kv_cache_utilization = 0.5;
        let clock = FakeClock;

        match algo.route(&dummy_request(), &backends, &clock) {
            RoutingDecision::Route(id) => assert_eq!(id, 1),
            _ => panic!("Expected Route"),
        }
    }
}
