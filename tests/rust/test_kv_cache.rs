/// Integration tests for KV cache simulation.
use routesim_core::kv_cache::KvCacheSimulator;

#[test]
fn test_cache_lifecycle() {
    let mut cache = KvCacheSimulator::new(1000, 16);

    // Allocate for multiple requests
    for i in 0..10 {
        let result = cache.allocate_for_request(
            i,
            256, // 16 blocks needed
            Some(i % 3),
            Some(128),
        );
        assert!(result.success);
    }

    assert!(cache.utilization() > 0.0);

    // Release all requests
    for i in 0..10 {
        cache.release_request(i);
    }

    // Prefix blocks should remain, non-prefix blocks freed
    let stats = cache.stats();
    assert!(stats.prefix_entries > 0);
}

#[test]
fn test_cache_prefix_reuse() {
    let mut cache = KvCacheSimulator::new(500, 16);

    // First request creates prefix blocks
    let r1 = cache.allocate_for_request(1, 256, Some(0xABC), Some(128));
    assert!(r1.success);
    assert!(!r1.prefix_cache_hit);

    // Release and reallocate with same prefix
    cache.release_request(1);
    let r2 = cache.allocate_for_request(2, 256, Some(0xABC), Some(128));
    assert!(r2.success);
    assert!(r2.prefix_cache_hit);
    assert!(r2.prefix_blocks_reused > 0);

    // Hit rate should be positive
    assert!(cache.hit_rate() > 0.0);
}

#[test]
fn test_cache_eviction_under_pressure() {
    let mut cache = KvCacheSimulator::new(20, 16); // Small cache

    // Fill with different prefix blocks
    for i in 0..5 {
        cache.allocate_for_request(i, 64, Some(i), Some(64)); // 4 blocks each, 20 total
        cache.release_request(i);
    }

    // This should force eviction of old prefix blocks
    let result = cache.allocate_for_request(100, 64, Some(100), Some(64));
    assert!(result.success);
    assert!(cache.stats().evictions > 0);
}

#[test]
fn test_cache_utilization_tracking() {
    let mut cache = KvCacheSimulator::new(100, 16);

    assert_eq!(cache.utilization(), 0.0);

    cache.allocate_for_request(1, 160, None, None); // 10 blocks
    assert!((cache.utilization() - 0.1).abs() < 0.01);

    cache.allocate_for_request(2, 160, None, None); // 10 more blocks
    assert!((cache.utilization() - 0.2).abs() < 0.01);

    cache.release_request(1);
    assert!((cache.utilization() - 0.1).abs() < 0.01);
}
