//! Cost escalation routing algorithm (Monte Carlo Heap inspired).
//!
//! Each backend maintains an escalation cost that increases with load.
//! Requests are routed to the backend with the lowest current cost.
//! Costs decay over time, creating a natural load-balancing effect.

use crate::traits::*;
use std::collections::HashMap;

/// Cost escalation router.
///
/// Inspired by the Monte Carlo Heap scheduling approach. Each backend
/// has a cost that escalates with queue depth and KV cache pressure.
/// The algorithm routes to the minimum-cost backend.
pub struct CostEscalation {
    /// Cost accumulator per backend.
    costs: HashMap<u32, f64>,
    /// Base cost per request.
    base_cost: f64,
    /// Cost multiplier per queued request.
    queue_cost: f64,
    /// Cost multiplier for KV cache pressure.
    kv_pressure_cost: f64,
    /// Cost decay factor per second.
    decay_per_sec: f64,
    /// Last update time.
    last_update_ms: u64,
}

impl CostEscalation {
    pub fn new() -> Self {
        Self {
            costs: HashMap::new(),
            base_cost: 1.0,
            queue_cost: 0.5,
            kv_pressure_cost: 2.0,
            decay_per_sec: 0.1,
            last_update_ms: 0,
        }
    }

    pub fn with_params(
        base_cost: f64,
        queue_cost: f64,
        kv_pressure_cost: f64,
        decay_per_sec: f64,
    ) -> Self {
        assert!(
            decay_per_sec >= 0.0,
            "decay_per_sec must be >= 0.0, got {}",
            decay_per_sec
        );
        Self {
            costs: HashMap::new(),
            base_cost,
            queue_cost,
            kv_pressure_cost,
            decay_per_sec,
            last_update_ms: 0,
        }
    }

    fn update_costs(&mut self, backends: &[BackendSnapshot], now_ms: u64) {
        let elapsed_sec = (now_ms.saturating_sub(self.last_update_ms)) as f64 / 1000.0;
        let decay = (-self.decay_per_sec * elapsed_sec).exp();

        for backend in backends {
            let cost = self.costs.entry(backend.id).or_insert(0.0);
            *cost *= decay; // apply time decay
        }
        self.last_update_ms = now_ms;
    }

    fn escalate(&mut self, backend_id: u32, backend: &BackendSnapshot) {
        let cost = self.costs.entry(backend_id).or_insert(0.0);
        let escalation = self.base_cost
            + self.queue_cost * backend.queue_depth as f64
            + self.kv_pressure_cost * backend.kv_cache_utilization as f64;
        *cost += escalation;
    }
}

impl Default for CostEscalation {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingAlgorithm for CostEscalation {
    fn route(
        &mut self,
        _request: &RequestInfo,
        backends: &[BackendSnapshot],
        clock: &dyn Clock,
    ) -> RoutingDecision {
        let available = available_backends(backends);
        if available.is_empty() {
            return RoutingDecision::Reject;
        }

        self.update_costs(backends, clock.now_ms());

        // Pick backend with lowest cost
        let best = available
            .iter()
            .min_by(|a, b| {
                let cost_a = self.costs.get(&a.id).copied().unwrap_or(0.0);
                let cost_b = self.costs.get(&b.id).copied().unwrap_or(0.0);
                cost_a
                    .partial_cmp(&cost_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        // Escalate cost for chosen backend
        self.escalate(best.id, best);

        RoutingDecision::Route(best.id)
    }

    fn name(&self) -> &str {
        "cost_escalation"
    }

    fn custom_metrics(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        if !self.costs.is_empty() {
            let max_cost = self
                .costs
                .values()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let min_cost = self.costs.values().cloned().fold(f64::INFINITY, f64::min);
            m.insert("max_backend_cost".to_string(), max_cost);
            m.insert("min_backend_cost".to_string(), min_cost);
            m.insert("cost_spread".to_string(), max_cost - min_cost);
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::make_backends;

    struct FakeClock(u64);
    impl Clock for FakeClock {
        fn now_ms(&self) -> u64 {
            self.0
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
    fn test_cost_escalation_distributes() {
        let mut algo = CostEscalation::new();
        let backends = make_backends(4);
        let clock = FakeClock(0);

        let mut counts = [0u32; 4];
        for _ in 0..100 {
            match algo.route(&dummy_request(), &backends, &clock) {
                RoutingDecision::Route(id) => counts[id as usize] += 1,
                _ => panic!("Expected Route"),
            }
        }
        // Should be roughly evenly distributed
        for count in &counts {
            assert!(*count >= 15, "Count {} too low", count);
        }
    }

    #[test]
    fn test_cost_decay() {
        let mut algo = CostEscalation::new();
        let backends = make_backends(2);

        // Route at t=0 — picks backend 0, escalates its cost
        let clock0 = FakeClock(0);
        let first = match algo.route(&dummy_request(), &backends, &clock0) {
            RoutingDecision::Route(id) => id,
            _ => panic!(),
        };

        // Next route at t=0 should pick the other backend (cost of first is higher)
        let second = match algo.route(&dummy_request(), &backends, &clock0) {
            RoutingDecision::Route(id) => id,
            _ => panic!(),
        };
        assert_ne!(first, second);
    }

    #[test]
    fn test_custom_metrics() {
        let mut algo = CostEscalation::new();
        let backends = make_backends(2);
        let clock = FakeClock(0);

        algo.route(&dummy_request(), &backends, &clock);
        let metrics = algo.custom_metrics();
        assert!(metrics.contains_key("max_backend_cost"));
        assert!(metrics.contains_key("cost_spread"));
    }

    #[test]
    #[should_panic(expected = "decay_per_sec must be >= 0.0")]
    fn test_negative_decay_panics() {
        CostEscalation::with_params(1.0, 0.5, 2.0, -0.1);
    }
}
