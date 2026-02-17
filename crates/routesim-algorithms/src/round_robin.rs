//! Round-robin routing algorithm.
//!
//! The simplest routing strategy: distributes requests evenly across backends
//! in a circular fashion. Provides good fairness but ignores backend state
//! (queue depth, cache contents, etc.).

use crate::traits::*;

/// Round-robin router.
///
/// Tracks the last-used backend by ID rather than positional index, so the
/// rotation is stable even when backends go offline or start draining.
pub struct RoundRobin {
    /// ID of the last backend we routed to (None on first call).
    last_backend_id: Option<u32>,
}

impl RoundRobin {
    pub fn new() -> Self {
        Self {
            last_backend_id: None,
        }
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

        // Find the next available backend after the last one we used.
        // If no prior backend (first call), start from the first available.
        let chosen = match self.last_backend_id {
            Some(last_id) => {
                // Find the first available backend with id > last_id (wrap around)
                available
                    .iter()
                    .find(|b| b.id > last_id)
                    .or_else(|| available.first())
                    .unwrap()
            }
            None => available.first().unwrap(),
        };

        self.last_backend_id = Some(chosen.id);
        RoutingDecision::Route(chosen.id)
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
            cache_block_hashes: Vec::new(),
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

    #[test]
    fn test_round_robin_stable_with_backend_removal() {
        // When a backend goes offline, round-robin should not skip or
        // double-serve other backends. It advances by ID, not position.
        let mut rr = RoundRobin::new();
        let clock = FakeClock;

        // Start with 4 backends: [0, 1, 2, 3]
        let backends = make_backends(4);
        // Route 2 requests: should go to 0, 1
        let r1 = match rr.route(&dummy_request(), &backends, &clock) {
            RoutingDecision::Route(id) => id,
            _ => panic!("Expected Route"),
        };
        assert_eq!(r1, 0);
        let r2 = match rr.route(&dummy_request(), &backends, &clock) {
            RoutingDecision::Route(id) => id,
            _ => panic!("Expected Route"),
        };
        assert_eq!(r2, 1);

        // Backend 2 goes offline: available = [0, 1, 3]
        let mut backends_minus_2 = make_backends(4);
        backends_minus_2[2].state = BackendState::Offline;

        // Next request should go to backend 3 (next ID after 1), not skip to 0
        let r3 = match rr.route(&dummy_request(), &backends_minus_2, &clock) {
            RoutingDecision::Route(id) => id,
            _ => panic!("Expected Route"),
        };
        assert_eq!(
            r3, 3,
            "Should advance to backend 3, not skip due to backend 2 going offline"
        );

        // Next should wrap around to 0
        let r4 = match rr.route(&dummy_request(), &backends_minus_2, &clock) {
            RoutingDecision::Route(id) => id,
            _ => panic!("Expected Route"),
        };
        assert_eq!(r4, 0, "Should wrap around to backend 0");
    }
}
