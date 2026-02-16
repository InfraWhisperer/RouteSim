//! Round-robin routing algorithm.
//!
//! The simplest routing strategy: distributes requests evenly across backends
//! in a circular fashion. Provides good fairness but ignores backend state
//! (queue depth, cache contents, etc.).

use crate::traits::*;

/// Round-robin router.
pub struct RoundRobin {
    next_index: usize,
}

impl RoundRobin {
    pub fn new() -> Self {
        Self { next_index: 0 }
    }
}

impl Default for RoundRobin {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingAlgorithm for RoundRobin {
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

        // Wrap around
        self.next_index %= available.len();
        let backend = available[self.next_index];
        self.next_index += 1;

        RoutingDecision::Route(backend.id)
    }

    fn name(&self) -> &str {
        "round_robin"
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
    fn test_round_robin_distributes_evenly() {
        let mut rr = RoundRobin::new();
        let backends = make_backends(4);
        let clock = FakeClock;

        let mut counts = [0u32; 4];
        for _ in 0..100 {
            match rr.route(&dummy_request(), &backends, &clock) {
                RoutingDecision::Route(id) => counts[id as usize] += 1,
                _ => panic!("Expected Route"),
            }
        }
        assert_eq!(counts, [25, 25, 25, 25]);
    }

    #[test]
    fn test_round_robin_rejects_no_backends() {
        let mut rr = RoundRobin::new();
        let clock = FakeClock;
        match rr.route(&dummy_request(), &[], &clock) {
            RoutingDecision::Reject => {}
            _ => panic!("Expected Reject"),
        }
    }
}
