//! Block-level prefix overlap routing algorithm.
//!
//! Routes requests to the backend that has the most cache block hashes in common
//! with the request's `cache_block_hashes`. This is how real prefix-cache-aware
//! routers work (SGLang's RadixAttention, Mooncake's scheduler): the overlap
//! count directly determines how many prefill tokens can be skipped.
//!
//! Falls back to least outstanding when no backend has any overlap or when
//! the request has no block hashes.

use crate::traits::*;

/// Block-level prefix overlap router.
///
/// Strategy:
/// 1. Compute overlap = |request.cache_block_hashes ∩ backend.cached_block_hashes|
///    for each backend.
/// 2. Score = cache_weight * (overlap / total_blocks) + (1 - cache_weight) * (1 - load_ratio)
/// 3. Route to highest-scoring backend.
/// 4. If no block hashes on the request, fall back to least outstanding.
pub struct PrefixOverlap {
    /// Weight for cache overlap vs load (0.0 = pure load, 1.0 = pure cache).
    cache_weight: f64,
}

impl PrefixOverlap {
    pub fn new() -> Self {
        Self { cache_weight: 0.7 }
    }

    pub fn with_cache_weight(cache_weight: f64) -> Self {
        Self {
            cache_weight: cache_weight.clamp(0.0, 1.0),
        }
    }
}

impl Default for PrefixOverlap {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingAlgorithm for PrefixOverlap {
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

        // If no block hashes, fall back to least outstanding
        if request.cache_block_hashes.is_empty() {
            let best = available
                .iter()
                .min_by_key(|b| b.queue_depth + b.active_batch_size)
                .unwrap();
            return RoutingDecision::Route(best.id);
        }

        let total_blocks = request.cache_block_hashes.len() as f64;

        let best = available
            .iter()
            .max_by(|a, b| {
                let score_a = self.score_backend(a, &request.cache_block_hashes, total_blocks);
                let score_b = self.score_backend(b, &request.cache_block_hashes, total_blocks);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        RoutingDecision::Route(best.id)
    }

    fn name(&self) -> &str {
        "prefix_overlap"
    }
}

impl PrefixOverlap {
    fn score_backend(
        &self,
        backend: &BackendSnapshot,
        request_blocks: &[u64],
        total_blocks: f64,
    ) -> f64 {
        let max_load = backend.max_queue_depth.max(1) as f64;
        let load = (backend.queue_depth + backend.active_batch_size) as f64;
        let load_ratio = (load / max_load).min(1.0);

        // Circuit breaker: if load exceeds 90% of capacity, ignore cache affinity
        if load_ratio >= 0.9 {
            return (1.0 - self.cache_weight) * (1.0 - load_ratio);
        }

        // Count how many of the request's blocks are cached on this backend
        let overlap = request_blocks
            .iter()
            .filter(|h| backend.cached_block_hashes.contains(h))
            .count() as f64;

        let cache_score = overlap / total_blocks;
        let load_score = 1.0 - load_ratio;

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

    fn request_with_blocks(blocks: Vec<u64>) -> RequestInfo {
        RequestInfo {
            id: 0,
            prompt_tokens: 512,
            max_gen_tokens: 128,
            actual_gen_tokens: 128,
            prefix_hash: Some(0xABC),
            prefix_token_length: Some(256),
            cache_block_hashes: blocks,
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
        }
    }

    #[test]
    fn test_prefix_overlap_prefers_most_overlap() {
        let mut algo = PrefixOverlap::new();
        let mut backends = make_backends(3);

        // Backend 0: has 1 matching block
        backends[0].cached_block_hashes.insert(10);

        // Backend 1: has 3 matching blocks
        backends[1].cached_block_hashes.insert(10);
        backends[1].cached_block_hashes.insert(20);
        backends[1].cached_block_hashes.insert(30);

        // Backend 2: no matching blocks
        // (leave empty)

        let clock = FakeClock;
        let req = request_with_blocks(vec![10, 20, 30, 40, 50]);
        match algo.route(&req, &backends, &clock) {
            RoutingDecision::Route(id) => {
                assert_eq!(id, 1, "Should route to backend with most overlap")
            }
            _ => panic!("Expected Route"),
        }
    }

    #[test]
    fn test_prefix_overlap_fallback_to_least_loaded() {
        let mut algo = PrefixOverlap::new();
        let mut backends = make_backends(3);
        backends[0].queue_depth = 10;
        backends[1].queue_depth = 2;
        backends[2].queue_depth = 5;

        let clock = FakeClock;
        // No block hashes — should fall back to least outstanding
        let req = RequestInfo {
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
        };
        match algo.route(&req, &backends, &clock) {
            RoutingDecision::Route(id) => assert_eq!(id, 1),
            _ => panic!("Expected Route"),
        }
    }

    #[test]
    fn test_prefix_overlap_circuit_breaker() {
        let mut algo = PrefixOverlap::new();
        let mut backends = make_backends(2);

        // Backend 0: has all blocks cached but is overloaded
        for i in 0..5 {
            backends[0].cached_block_hashes.insert(i);
        }
        backends[0].queue_depth = 240;
        backends[0].active_batch_size = 10; // 250/256 = 97.7% > 90%

        // Backend 1: no blocks but idle
        backends[1].queue_depth = 0;

        let clock = FakeClock;
        let req = request_with_blocks(vec![0, 1, 2, 3, 4]);
        match algo.route(&req, &backends, &clock) {
            RoutingDecision::Route(id) => {
                assert_ne!(id, 0, "Should not route to overloaded backend")
            }
            _ => panic!("Expected Route"),
        }
    }

    #[test]
    fn test_prefix_overlap_no_overlap_prefers_least_loaded() {
        let mut algo = PrefixOverlap::new();
        let mut backends = make_backends(3);
        // No backend has any of the request's blocks
        backends[0].queue_depth = 10;
        backends[1].queue_depth = 2;
        backends[2].queue_depth = 5;

        let clock = FakeClock;
        let req = request_with_blocks(vec![100, 200, 300]);
        match algo.route(&req, &backends, &clock) {
            RoutingDecision::Route(id) => {
                assert_eq!(id, 1, "With no overlap, should prefer least loaded")
            }
            _ => panic!("Expected Route"),
        }
    }
}
