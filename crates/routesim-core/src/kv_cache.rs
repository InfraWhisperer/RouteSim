//! KV cache simulation with block-based allocation, prefix tracking, and LRU eviction.
//!
//! Models the memory management of a real vLLM-style paged KV cache, including:
//! - Block-level allocation and deallocation
//! - Shared prefix blocks with reference counting
//! - LRU eviction (non-prefix blocks first, then prefix blocks)
//! - Cache hit/miss tracking for routing algorithm feedback

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// A single block in the KV cache.
#[derive(Debug, Clone)]
pub struct CacheBlock {
    /// Unique block identifier.
    pub block_id: u64,
    /// The request that owns this block (None for prefix blocks shared across requests).
    pub owner_request_id: Option<u64>,
    /// If this block is part of a prefix, its prefix hash.
    pub prefix_hash: Option<u64>,
    /// Reference count (>1 for shared prefix blocks).
    pub ref_count: u32,
    /// Number of tokens stored in this block.
    pub tokens_stored: u32,
}

/// Block-based KV cache simulator for a single backend.
#[derive(Debug, Clone)]
pub struct KvCacheSimulator {
    /// Total number of blocks available.
    pub total_blocks: u32,
    /// Tokens per block.
    pub block_size: u32,
    /// Currently allocated blocks.
    allocated_blocks: HashMap<u64, CacheBlock>,
    /// Prefix hash -> block IDs containing that prefix's KV data.
    prefix_index: HashMap<u64, Vec<u64>>,
    /// Request ID -> block IDs allocated to that request.
    request_blocks: HashMap<u64, Vec<u64>>,
    /// LRU order for eviction (front = least recently used).
    lru_order: VecDeque<u64>,
    /// Set of free block IDs.
    free_blocks: Vec<u64>,
    /// Content-addressed block hashes cached on this backend.
    /// Tracks which Mooncake-style block hashes are present for `prefix_overlap` routing.
    /// Bounded by `total_blocks` capacity; evicts in FIFO order (oldest first).
    content_block_set: HashSet<u64>,
    /// Insertion order for FIFO eviction of content block hashes.
    content_block_order: VecDeque<u64>,
    /// Cache hit counter.
    pub hits: u64,
    /// Cache miss counter.
    pub misses: u64,
    /// Total evictions performed.
    pub evictions: u64,
}

/// Result of attempting to allocate blocks for a request.
#[derive(Debug, Clone)]
pub struct AllocationResult {
    /// Whether the allocation succeeded.
    pub success: bool,
    /// Number of blocks allocated.
    pub blocks_allocated: u32,
    /// Whether prefix blocks were found and reused (cache hit).
    pub prefix_cache_hit: bool,
    /// Number of prefix blocks reused.
    pub prefix_blocks_reused: u32,
    /// Number of blocks that had to be evicted to make room.
    pub blocks_evicted: u32,
}

/// Statistics snapshot for reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheStats {
    pub total_blocks: u32,
    pub used_blocks: u32,
    pub free_blocks: u32,
    pub utilization: f32,
    pub prefix_entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub evictions: u64,
}

impl KvCacheSimulator {
    /// Create a new KV cache simulator.
    pub fn new(total_blocks: u32, block_size: u32) -> Self {
        let free_blocks = (0..total_blocks as u64).collect();
        Self {
            total_blocks,
            block_size,
            allocated_blocks: HashMap::new(),
            prefix_index: HashMap::new(),
            request_blocks: HashMap::new(),
            lru_order: VecDeque::new(),
            free_blocks,
            content_block_set: HashSet::new(),
            content_block_order: VecDeque::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Number of blocks currently in use.
    pub fn used_blocks(&self) -> u32 {
        self.allocated_blocks.len() as u32
    }

    /// Cache utilization as a fraction (0.0 - 1.0).
    pub fn utilization(&self) -> f32 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        self.used_blocks() as f32 / self.total_blocks as f32
    }

    /// Cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    /// Check whether a given prefix hash has cached blocks.
    pub fn has_prefix(&self, prefix_hash: u64) -> bool {
        self.prefix_index
            .get(&prefix_hash)
            .is_some_and(|blocks| !blocks.is_empty())
    }

    /// Get the set of all cached prefix hashes.
    pub fn cached_prefix_hashes(&self) -> HashSet<u64> {
        self.prefix_index
            .iter()
            .filter(|(_, blocks)| !blocks.is_empty())
            .map(|(hash, _)| *hash)
            .collect()
    }

    /// Get the set of all content-addressed block hashes currently cached.
    /// These are the original hash_ids (e.g., from Mooncake traces), not
    /// internal block allocation identifiers.
    pub fn cached_content_block_hashes(&self) -> HashSet<u64> {
        self.content_block_set.clone()
    }

    /// Record that a set of content block hashes are cached on this backend.
    /// New hashes are appended; when the set exceeds `total_blocks` capacity,
    /// the oldest entries are evicted in FIFO order.
    pub fn add_content_blocks(&mut self, hashes: &[u64]) {
        for &hash in hashes {
            if self.content_block_set.insert(hash) {
                self.content_block_order.push_back(hash);
            }
        }
        // Evict oldest entries if over capacity
        let capacity = self.total_blocks as usize;
        while self.content_block_set.len() > capacity {
            if let Some(old) = self.content_block_order.pop_front() {
                self.content_block_set.remove(&old);
            } else {
                break;
            }
        }
    }

    /// Allocate blocks for a request, including prefix block reuse.
    ///
    /// Returns an `AllocationResult` describing what happened.
    pub fn allocate_for_request(
        &mut self,
        request_id: u64,
        total_tokens: u32,
        prefix_hash: Option<u64>,
        prefix_token_length: Option<u32>,
    ) -> AllocationResult {
        let total_blocks_needed = self.tokens_to_blocks(total_tokens);

        // Check prefix cache hit
        let (prefix_cache_hit, prefix_blocks_reused) =
            if let (Some(hash), Some(_prefix_len)) = (prefix_hash, prefix_token_length) {
                // Collect block IDs first to avoid borrow conflict
                let existing_block_ids: Option<Vec<u64>> = self
                    .prefix_index
                    .get(&hash)
                    .filter(|blocks| !blocks.is_empty())
                    .cloned();

                if let Some(block_ids) = existing_block_ids {
                    self.hits += 1;
                    let reused = block_ids.len() as u32;
                    for &block_id in &block_ids {
                        if let Some(block) = self.allocated_blocks.get_mut(&block_id) {
                            block.ref_count += 1;
                        }
                        self.touch_lru(block_id);
                    }
                    (true, reused)
                } else {
                    self.misses += 1;
                    (false, 0)
                }
            } else {
                // No prefix to look up — not counted as hit or miss
                (false, 0)
            };

        let new_blocks_needed = total_blocks_needed.saturating_sub(prefix_blocks_reused);

        // Evict if necessary
        let mut blocks_evicted = 0u32;
        while (self.free_blocks.len() as u32) < new_blocks_needed {
            if self.evict_one_block() {
                blocks_evicted += 1;
            } else {
                // Cannot free enough space
                return AllocationResult {
                    success: false,
                    blocks_allocated: 0,
                    prefix_cache_hit,
                    prefix_blocks_reused,
                    blocks_evicted,
                };
            }
        }

        // Allocate new blocks
        let mut allocated_block_ids = Vec::with_capacity(new_blocks_needed as usize);
        for _ in 0..new_blocks_needed {
            if let Some(block_id) = self.free_blocks.pop() {
                let block = CacheBlock {
                    block_id,
                    owner_request_id: Some(request_id),
                    prefix_hash: None,
                    ref_count: 1,
                    tokens_stored: self.block_size,
                };
                self.allocated_blocks.insert(block_id, block);
                self.lru_order.push_back(block_id);
                allocated_block_ids.push(block_id);
            }
        }

        // If this request has a new prefix (cache miss), mark prefix blocks
        if let (Some(hash), Some(prefix_len)) = (prefix_hash, prefix_token_length) {
            if !prefix_cache_hit {
                let prefix_block_count = self.tokens_to_blocks(prefix_len) as usize;
                let prefix_blocks: Vec<u64> = allocated_block_ids
                    .iter()
                    .take(prefix_block_count)
                    .copied()
                    .collect();
                for &block_id in &prefix_blocks {
                    if let Some(block) = self.allocated_blocks.get_mut(&block_id) {
                        block.prefix_hash = Some(hash);
                    }
                }
                self.prefix_index.insert(hash, prefix_blocks);
            }
        }

        // Track which blocks belong to this request
        let mut req_blocks = allocated_block_ids;
        if prefix_cache_hit {
            if let Some(hash) = prefix_hash {
                if let Some(prefix_block_ids) = self.prefix_index.get(&hash) {
                    req_blocks.extend(prefix_block_ids.iter());
                }
            }
        }
        self.request_blocks.insert(request_id, req_blocks);

        AllocationResult {
            success: true,
            blocks_allocated: new_blocks_needed + prefix_blocks_reused,
            prefix_cache_hit,
            prefix_blocks_reused,
            blocks_evicted,
        }
    }

    /// Release blocks held by a completed request.
    ///
    /// Prefix blocks are kept (with decremented ref count) for potential reuse.
    /// Non-prefix blocks are freed immediately.
    pub fn release_request(&mut self, request_id: u64) {
        if let Some(block_ids) = self.request_blocks.remove(&request_id) {
            for block_id in block_ids {
                if let Some(block) = self.allocated_blocks.get_mut(&block_id) {
                    block.ref_count = block.ref_count.saturating_sub(1);
                    if block.ref_count == 0 {
                        if block.prefix_hash.is_some() {
                            // Keep prefix blocks around for reuse, but they're evictable
                            block.owner_request_id = None;
                        } else {
                            // Free non-prefix blocks immediately
                            let block_id = block.block_id;
                            self.allocated_blocks.remove(&block_id);
                            self.free_blocks.push(block_id);
                            self.remove_from_lru(block_id);
                        }
                    }
                }
            }
        }
    }

    /// Get a statistics snapshot.
    pub fn stats(&self) -> KvCacheStats {
        KvCacheStats {
            total_blocks: self.total_blocks,
            used_blocks: self.used_blocks(),
            free_blocks: self.free_blocks.len() as u32,
            utilization: self.utilization(),
            prefix_entries: self.prefix_index.len(),
            hits: self.hits,
            misses: self.misses,
            hit_rate: self.hit_rate(),
            evictions: self.evictions,
        }
    }

    /// Convert token count to block count, rounding up.
    fn tokens_to_blocks(&self, tokens: u32) -> u32 {
        tokens.div_ceil(self.block_size)
    }

    /// Evict a single block using LRU policy.
    /// Non-prefix, zero-refcount blocks are evicted first, then prefix blocks.
    fn evict_one_block(&mut self) -> bool {
        // First pass: find non-prefix block with ref_count == 0
        let mut evict_id = None;
        for &block_id in &self.lru_order {
            if let Some(block) = self.allocated_blocks.get(&block_id) {
                if block.prefix_hash.is_none() && block.ref_count == 0 {
                    evict_id = Some(block_id);
                    break;
                }
            }
        }

        // Second pass: evict prefix block with ref_count == 0
        if evict_id.is_none() {
            for &block_id in &self.lru_order {
                if let Some(block) = self.allocated_blocks.get(&block_id) {
                    if block.ref_count == 0 {
                        evict_id = Some(block_id);
                        break;
                    }
                }
            }
        }

        if let Some(block_id) = evict_id {
            // Remove from prefix index if it's a prefix block
            if let Some(block) = self.allocated_blocks.get(&block_id) {
                if let Some(hash) = block.prefix_hash {
                    if let Some(prefix_blocks) = self.prefix_index.get_mut(&hash) {
                        prefix_blocks.retain(|&id| id != block_id);
                        if prefix_blocks.is_empty() {
                            self.prefix_index.remove(&hash);
                        }
                    }
                }
            }
            self.allocated_blocks.remove(&block_id);
            self.free_blocks.push(block_id);
            self.remove_from_lru(block_id);
            self.evictions += 1;
            true
        } else {
            false
        }
    }

    /// Move a block to the back (most recently used) of the LRU list.
    fn touch_lru(&mut self, block_id: u64) {
        self.remove_from_lru(block_id);
        self.lru_order.push_back(block_id);
    }

    /// Remove a block from the LRU list.
    fn remove_from_lru(&mut self, block_id: u64) {
        self.lru_order.retain(|&id| id != block_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_cache() {
        let cache = KvCacheSimulator::new(1000, 16);
        assert_eq!(cache.total_blocks, 1000);
        assert_eq!(cache.used_blocks(), 0);
        assert_eq!(cache.utilization(), 0.0);
    }

    #[test]
    fn test_allocate_simple() {
        let mut cache = KvCacheSimulator::new(100, 16);
        let result = cache.allocate_for_request(1, 64, None, None); // 4 blocks needed
        assert!(result.success);
        assert_eq!(result.blocks_allocated, 4);
        assert!(!result.prefix_cache_hit);
        assert_eq!(cache.used_blocks(), 4);
    }

    #[test]
    fn test_prefix_cache_hit() {
        let mut cache = KvCacheSimulator::new(100, 16);

        // First request creates prefix blocks
        let r1 = cache.allocate_for_request(1, 256, Some(0xABC), Some(128));
        assert!(r1.success);
        assert!(!r1.prefix_cache_hit);

        // Release request but prefix blocks should remain
        cache.release_request(1);

        // Second request with same prefix should hit cache
        let r2 = cache.allocate_for_request(2, 256, Some(0xABC), Some(128));
        assert!(r2.success);
        assert!(r2.prefix_cache_hit);
        assert!(r2.prefix_blocks_reused > 0);
    }

    #[test]
    fn test_has_prefix() {
        let mut cache = KvCacheSimulator::new(100, 16);
        assert!(!cache.has_prefix(0xABC));

        cache.allocate_for_request(1, 128, Some(0xABC), Some(64));
        assert!(cache.has_prefix(0xABC));
    }

    #[test]
    fn test_eviction() {
        let mut cache = KvCacheSimulator::new(10, 16);

        // Fill the cache with prefix blocks that are retained after release
        cache.allocate_for_request(1, 80, Some(0xAA), Some(80)); // 5 prefix blocks
        cache.release_request(1); // prefix blocks stay (ref_count=0 but kept for reuse)
        cache.allocate_for_request(2, 80, Some(0xBB), Some(80)); // 5 prefix blocks
        cache.release_request(2); // prefix blocks stay

        // All 10 blocks used by prefix blocks; allocating more forces eviction
        let result = cache.allocate_for_request(3, 32, None, None); // needs 2 blocks
        assert!(result.success);
        assert!(result.blocks_evicted > 0);
        assert!(cache.evictions > 0);
    }

    #[test]
    fn test_allocation_failure() {
        let mut cache = KvCacheSimulator::new(4, 16);

        // Allocate all blocks with active requests (ref_count > 0)
        cache.allocate_for_request(1, 64, None, None); // 4 blocks, all in use

        // Try to allocate more — should fail because blocks are still referenced
        let result = cache.allocate_for_request(2, 16, None, None);
        assert!(!result.success);
    }

    #[test]
    fn test_stats() {
        let mut cache = KvCacheSimulator::new(100, 16);
        cache.allocate_for_request(1, 64, Some(0xABC), Some(32));
        let stats = cache.stats();
        assert_eq!(stats.total_blocks, 100);
        assert!(stats.used_blocks > 0);
        assert!(stats.utilization > 0.0);
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = KvCacheSimulator::new(100, 16);
        cache.allocate_for_request(1, 128, Some(0xABC), Some(64));
        cache.release_request(1);
        cache.allocate_for_request(2, 128, Some(0xABC), Some(64)); // hit
        cache.allocate_for_request(3, 128, Some(0xDEF), Some(64)); // miss

        assert_eq!(cache.hits, 1);
        assert_eq!(cache.misses, 2); // first request + DEF miss
        assert!(cache.hit_rate() > 0.0);
    }

    #[test]
    fn test_prefix_refcount_no_double_decrement() {
        // Regression test: verify that on a prefix cache hit, the prefix block
        // ref_count is incremented once (in the hit path) and decremented once
        // (in release_request). No double-decrement or premature eviction.
        let mut cache = KvCacheSimulator::new(100, 16);

        // Request A: creates prefix blocks (cache miss)
        let r1 = cache.allocate_for_request(1, 128, Some(0xABC), Some(64));
        assert!(r1.success);
        assert!(!r1.prefix_cache_hit);
        let _total_blocks = r1.blocks_allocated;

        // Release A — prefix blocks stay (ref_count=0, kept for reuse)
        cache.release_request(1);
        assert!(
            cache.has_prefix(0xABC),
            "Prefix should survive after release"
        );

        // Request B: same prefix → cache hit
        let r2 = cache.allocate_for_request(2, 128, Some(0xABC), Some(64));
        assert!(r2.success);
        assert!(r2.prefix_cache_hit);
        assert!(r2.prefix_blocks_reused > 0);

        // Request C: same prefix, concurrent with B → cache hit
        let r3 = cache.allocate_for_request(3, 128, Some(0xABC), Some(64));
        assert!(r3.success);
        assert!(r3.prefix_cache_hit);

        // Verify prefix block ref_counts: should be 2 (B + C)
        let prefix_block_ids: Vec<u64> = cache.prefix_index.get(&0xABC).unwrap().clone();
        for &bid in &prefix_block_ids {
            let block = cache.allocated_blocks.get(&bid).unwrap();
            assert_eq!(
                block.ref_count, 2,
                "Prefix block {} should have ref_count=2 (B+C), got {}",
                bid, block.ref_count
            );
        }

        // Release B — prefix ref_count should go 2→1
        cache.release_request(2);
        for &bid in &prefix_block_ids {
            let block = cache.allocated_blocks.get(&bid).unwrap();
            assert_eq!(
                block.ref_count, 1,
                "After releasing B, prefix block {} should have ref_count=1, got {}",
                bid, block.ref_count
            );
        }
        assert!(
            cache.has_prefix(0xABC),
            "Prefix should survive after B released"
        );

        // Release C — prefix ref_count should go 1→0, blocks kept for reuse
        cache.release_request(3);
        for &bid in &prefix_block_ids {
            let block = cache.allocated_blocks.get(&bid).unwrap();
            assert_eq!(
                block.ref_count, 0,
                "After releasing C, prefix block {} should have ref_count=0, got {}",
                bid, block.ref_count
            );
        }
        assert!(
            cache.has_prefix(0xABC),
            "Prefix should survive with ref_count=0 (kept for reuse)"
        );

        // Request D: should still get a cache hit on the surviving prefix
        let r4 = cache.allocate_for_request(4, 128, Some(0xABC), Some(64));
        assert!(r4.success);
        assert!(
            r4.prefix_cache_hit,
            "Prefix should still be available for cache hit after all prior requests released"
        );
    }
}
