//! Prefix-cache-aware routing algorithm.
//!
//! Maximizes KV cache hit rate by routing requests to backends that already
//! have the request's prefix cached. Falls back to least-loaded among
//! backends without the prefix.

use crate::traits::*;

/// Prefix-cache-aware router.
///
/// Strategy:
/// 1. If a backend has the request's prefix cached, route there (prefer least loaded among matches).
/// 2. If no backend has the prefix, route to the least loaded backend.
pub struct PrefixAware {
    /// Weight for cache hit bonus vs load penalty (0.0 = pure load, 1.0 = pure cache).
    cache_weight: f64,
}

impl PrefixAware {
    pub fn new() -> Self {
        Self { cache_weight: 0.8 }
    }

    pub fn with_cache_weight(cache_weight: f64) -> Self {
        Self {
            cache_weight: cache_weight.clamp(0.0, 1.0),
        }
    }
}

impl Default for PrefixAware {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingAlgorithm for PrefixAware {
    fn route(
        &mut self,
        request: &RequestInfo,
        backends: &[BackendSnapshot],
        _clock: &dyn Clock,
    ) -> RoutingDecision {
        let available = available_backends(backends);
        if available.is_empty() {
            return RoutingDecision::Reject;
        }

        // If no prefix, fall back to least outstanding
        let prefix_hash = match request.prefix_hash {
            Some(h) => h,
            None => {
                let best = available
                    .iter()
                    .min_by_key(|b| b.queue_depth + b.active_batch_size)
                    .unwrap();
                return RoutingDecision::Route(best.id);
            }
        };

        // Score each backend: higher is better
        let best = available
            .iter()
            .max_by(|a, b| {
                let score_a = self.score_backend(a, prefix_hash);
                let score_b = self.score_backend(b, prefix_hash);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        RoutingDecision::Route(best.id)
    }

    fn name(&self) -> &str {
        "prefix_aware"
    }
}

impl PrefixAware {
    fn score_backend(&self, backend: &BackendSnapshot, prefix_hash: u64) -> f64 {
        let has_prefix = backend.prefix_hashes_cached.contains(&prefix_hash);
        let cache_score = if has_prefix { 1.0 } else { 0.0 };

        // Load score: lower load = higher score
        let max_load = 256.0; // normalize against max queue depth
        let load = (backend.queue_depth + backend.active_batch_size) as f64;
        let load_score = 1.0 - (load / max_load).min(1.0);

        self.cache_weight * cache_score + (1.0 - self.cache_weight) * load_score
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

    fn request_with_prefix(hash: u64) -> RequestInfo {
        RequestInfo {
            id: 0,
            prompt_tokens: 512,
            max_gen_tokens: 128,
            actual_gen_tokens: 128,
            prefix_hash: Some(hash),
            prefix_token_length: Some(256),
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
        }
    }

    #[test]
    fn test_prefix_aware_prefers_cached() {
        let mut algo = PrefixAware::new();
        let mut backends = make_backends(3);

        // Backend 1 has the prefix cached
        backends[1].prefix_hashes_cached.insert(0xABC);

        let clock = FakeClock;
        match algo.route(&request_with_prefix(0xABC), &backends, &clock) {
            RoutingDecision::Route(id) => assert_eq!(id, 1),
            _ => panic!("Expected Route"),
        }
    }

    #[test]
    fn test_prefix_aware_fallback_to_least_loaded() {
        let mut algo = PrefixAware::new();
        let mut backends = make_backends(3);
        backends[0].queue_depth = 10;
        backends[1].queue_depth = 2;
        backends[2].queue_depth = 5;

        let clock = FakeClock;
        // No backend has the prefix, so fall back to least loaded
        match algo.route(&request_with_prefix(0xDEAD), &backends, &clock) {
            RoutingDecision::Route(id) => assert_eq!(id, 1),
            _ => panic!("Expected Route"),
        }
    }

    #[test]
    fn test_prefix_aware_no_prefix() {
        let mut algo = PrefixAware::new();
        let mut backends = make_backends(3);
        backends[0].queue_depth = 5;
        backends[1].queue_depth = 1;
        backends[2].queue_depth = 3;

        let req = RequestInfo {
            id: 0,
            prompt_tokens: 100,
            max_gen_tokens: 50,
            actual_gen_tokens: 50,
            prefix_hash: None,
            prefix_token_length: None,
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
        };
        let clock = FakeClock;
        match algo.route(&req, &backends, &clock) {
            RoutingDecision::Route(id) => assert_eq!(id, 1),
            _ => panic!("Expected Route"),
        }
    }
}
