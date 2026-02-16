//! Least outstanding requests routing algorithm.
//!
//! Routes each request to the backend with the fewest in-progress and queued
//! requests. This is a simple load-aware strategy that avoids hot spots.

use crate::traits::*;

/// Least outstanding requests router.
pub struct LeastOutstanding;

impl LeastOutstanding {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LeastOutstanding {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingAlgorithm for LeastOutstanding {
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
            .min_by_key(|b| b.queue_depth + b.active_batch_size)
            .unwrap();

        RoutingDecision::Route(best.id)
    }

    fn name(&self) -> &str {
        "least_outstanding"
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
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
        }
    }

    #[test]
    fn test_least_outstanding_picks_least_loaded() {
        let mut algo = LeastOutstanding::new();
        let mut backends = make_backends(3);
        backends[0].queue_depth = 10;
        backends[1].queue_depth = 2;
        backends[2].queue_depth = 5;
        let clock = FakeClock;

        match algo.route(&dummy_request(), &backends, &clock) {
            RoutingDecision::Route(id) => assert_eq!(id, 1),
            _ => panic!("Expected Route"),
        }
    }
}
