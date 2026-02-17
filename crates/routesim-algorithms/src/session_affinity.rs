//! Session affinity (sticky sessions) routing algorithm.
//!
//! Routes requests from the same conversation to the same backend,
//! maximizing KV cache reuse for multi-turn conversations.
//! Falls back to least outstanding for new conversations.

use crate::traits::*;
use std::collections::HashMap;

/// Session affinity router.
pub struct SessionAffinity {
    /// conversation_id -> (backend_id, last_access_ms) mapping.
    session_map: HashMap<String, (u32, u64)>,
    /// Maximum entries in session map before cleanup.
    max_sessions: usize,
}

impl SessionAffinity {
    pub fn new() -> Self {
        Self {
            session_map: HashMap::new(),
            max_sessions: 100_000,
        }
    }

    pub fn with_max_sessions(max_sessions: usize) -> Self {
        Self {
            session_map: HashMap::new(),
            max_sessions,
        }
    }

    /// Remove least-recently-used entries if map is too large.
    fn maybe_cleanup(&mut self) {
        if self.session_map.len() > self.max_sessions {
            // LRU eviction: sort by last access time, remove oldest half
            let mut entries: Vec<(String, u64)> = self
                .session_map
                .iter()
                .map(|(k, &(_, ts))| (k.clone(), ts))
                .collect();
            entries.sort_by_key(|(_, ts)| *ts);
            let to_remove = entries.len() / 2;
            for (key, _) in entries.into_iter().take(to_remove) {
                self.session_map.remove(&key);
            }
        }
    }
}

impl Default for SessionAffinity {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingAlgorithm for SessionAffinity {
    fn route(
        &mut self,
        request: &RequestInfo,
        backends: &[BackendSnapshot],
        clock: &dyn Clock,
    ) -> RoutingDecision {
        let available = available_backends(backends);
        if available.is_empty() {
            return RoutingDecision::Reject;
        }

        // Check if we have a session mapping
        if let Some(conv_id) = &request.conversation_id {
            if let Some(entry) = self.session_map.get_mut(conv_id) {
                let backend_id = entry.0;
                // Check if the backend is still available and has queue capacity.
                // available_backends already filters out draining/offline/queue-full
                // backends, so checking membership is sufficient.
                if available.iter().any(|b| b.id == backend_id) {
                    entry.1 = clock.now_ms(); // Refresh timestamp on access
                    return RoutingDecision::Route(backend_id);
                }
            }
            // Backend went offline, full, or no mapping — remove stale mapping
            self.session_map.remove(conv_id);
        }

        // No existing mapping: route to least loaded
        let best = available
            .iter()
            .min_by_key(|b| b.queue_depth + b.active_batch_size)
            .unwrap();

        // Store the mapping
        if let Some(conv_id) = &request.conversation_id {
            self.maybe_cleanup();
            self.session_map
                .insert(conv_id.clone(), (best.id, clock.now_ms()));
        }

        RoutingDecision::Route(best.id)
    }

    fn name(&self) -> &str {
        "session_affinity"
    }

    fn custom_metrics(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("active_sessions".to_string(), self.session_map.len() as f64);
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

    fn request_with_session(conv_id: &str) -> RequestInfo {
        RequestInfo {
            id: 0,
            prompt_tokens: 100,
            max_gen_tokens: 50,
            actual_gen_tokens: 50,
            prefix_hash: None,
            prefix_token_length: None,
            cache_block_hashes: Vec::new(),
            conversation_id: Some(conv_id.to_string()),
            lora_adapter: None,
            priority: 0,
        }
    }

    #[test]
    fn test_session_affinity_sticky() {
        let mut algo = SessionAffinity::new();
        let backends = make_backends(4);
        let clock = FakeClock(0);

        // First request creates a mapping
        let first = match algo.route(&request_with_session("conv-1"), &backends, &clock) {
            RoutingDecision::Route(id) => id,
            _ => panic!("Expected Route"),
        };

        // Second request with same conversation should go to same backend
        match algo.route(&request_with_session("conv-1"), &backends, &clock) {
            RoutingDecision::Route(id) => assert_eq!(id, first),
            _ => panic!("Expected Route"),
        }
    }

    #[test]
    fn test_session_affinity_different_conversations() {
        let mut algo = SessionAffinity::new();
        let mut backends = make_backends(4);
        // Make backend 0 loaded so first request goes elsewhere
        backends[0].queue_depth = 100;
        let clock = FakeClock(0);

        let first = match algo.route(&request_with_session("conv-1"), &backends, &clock) {
            RoutingDecision::Route(id) => id,
            _ => panic!("Expected Route"),
        };

        // Different conversation may go to different backend
        // (both go to least loaded, which should be same since no state changes)
        let second = match algo.route(&request_with_session("conv-2"), &backends, &clock) {
            RoutingDecision::Route(id) => id,
            _ => panic!("Expected Route"),
        };

        // Both should go to least loaded (same backend here)
        assert_eq!(first, second);
    }

    #[test]
    fn test_session_affinity_no_conversation_id() {
        let mut algo = SessionAffinity::new();
        let backends = make_backends(3);
        let clock = FakeClock(0);

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
            RoutingDecision::Route(_) => {}
            _ => panic!("Expected Route"),
        }
    }

    #[test]
    fn test_session_affinity_lru_eviction() {
        // Verify that oldest sessions (by access time) are evicted, not random ones.
        // maybe_cleanup() is called BEFORE insert and uses strict >, so we need
        // max_sessions+2 inserts to trigger: the (max+1)th insert brings len to
        // max+1 without cleanup, then the (max+2)th insert sees len > max.
        let mut algo = SessionAffinity::with_max_sessions(5);
        let backends = make_backends(4);

        // Insert 7 sessions with increasing timestamps
        for i in 0..7 {
            let clock = FakeClock(i * 100);
            let req = request_with_session(&format!("conv-{}", i));
            algo.route(&req, &backends, &clock);
        }

        // When inserting conv-6, cleanup sees len=6 > 5, evicts oldest 3 (conv-0,1,2).
        // Then conv-6 is inserted → map has conv-3,4,5,6 (4 entries).
        assert!(algo.session_map.len() <= 5);
        assert!(
            !algo.session_map.contains_key("conv-0"),
            "Oldest session conv-0 should have been evicted"
        );
        assert!(
            !algo.session_map.contains_key("conv-2"),
            "Session conv-2 should have been evicted"
        );
        assert!(
            algo.session_map.contains_key("conv-3"),
            "Session conv-3 should remain (newer half)"
        );
        assert!(
            algo.session_map.contains_key("conv-6"),
            "Newest session conv-6 should remain"
        );
    }
}
