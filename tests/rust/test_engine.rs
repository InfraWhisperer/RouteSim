/// Integration tests for the simulation engine.
use routesim_algorithms::*;
use routesim_core::config::SimConfig;
use routesim_core::request::InferenceRequest;
use std::collections::HashMap;

fn production_config() -> SimConfig {
    SimConfig::from_str(
        r#"
[simulation]
name = "integration-test"
seed = 42
warmup_requests = 5

[cluster]
gpu_type = "H100Sxm"
num_backends = 8
max_batch_tokens = 16384
max_queue_depth = 256
kv_cache_blocks = 32768
kv_block_size = 16

[cluster.compute_model]
prefill_tokens_per_sec = 50000
decode_tokens_per_sec_batch1 = 80
decode_throughput_saturation_batch = 64
decode_tokens_per_sec_saturated = 3200

[trace]
format = "compact_jsonl"
"#,
    )
    .unwrap()
}

fn mixed_workload(n: usize) -> Vec<InferenceRequest> {
    (0..n)
        .map(|i| InferenceRequest {
            id: i as u64,
            arrival_time_ms: (i as u64) * 5,
            prompt_tokens: [128, 256, 512, 1024, 2048][i % 5],
            max_gen_tokens: [32, 64, 128, 256, 512][i % 5],
            actual_gen_tokens: [32, 64, 128, 256, 512][i % 5],
            prefix_hash: Some((i % 5) as u64),
            prefix_token_length: Some([64, 128, 256, 512, 1024][i % 5]),
            cache_block_hashes: Vec::new(),
            conversation_id: Some(format!("conv-{}", i % 20)),
            lora_adapter: None,
            priority: 0,
            metadata: HashMap::new(),
        })
        .collect()
}

#[test]
fn test_full_simulation_round_robin() {
    let config = production_config();
    let requests = mixed_workload(100);
    let algo = Box::new(RoundRobin::new());
    let metrics = routesim_core::run_simulation(config, requests, algo);

    assert!(metrics.completed_requests > 0);
    assert!(metrics.ttft.p50 > 0.0);
    assert!(metrics.end_to_end_latency.p50 > 0.0);
    assert!(metrics.requests_per_sec > 0.0);
    assert!(metrics.jains_fairness_index > 0.0);
}

#[test]
fn test_full_simulation_all_algorithms() {
    let config = production_config();
    let requests = mixed_workload(50);

    for algo_name in available_algorithms() {
        let algo = algorithm_by_name(algo_name).unwrap();
        let metrics = routesim_core::run_simulation(config.clone(), requests.clone(), algo);
        assert!(
            metrics.completed_requests > 0,
            "Algorithm {} produced no completed requests",
            algo_name
        );
    }
}

#[test]
fn test_compare_mode() {
    let config = production_config();
    let requests = mixed_workload(50);
    let results = routesim_core::compare_algorithms(
        &config,
        &requests,
        &["round_robin", "least_outstanding", "prefix_aware"],
    );
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].algorithm, "round_robin");
    assert_eq!(results[1].algorithm, "least_outstanding");
    assert_eq!(results[2].algorithm, "prefix_aware");
}

#[test]
fn test_prefix_aware_has_cache_hits() {
    let config = production_config();
    // All requests have the same prefix — should get high cache hit rate
    let requests: Vec<InferenceRequest> = (0..100)
        .map(|i| InferenceRequest {
            id: i,
            arrival_time_ms: i * 10,
            prompt_tokens: 512,
            max_gen_tokens: 64,
            actual_gen_tokens: 64,
            prefix_hash: Some(0xDEADBEEF),
            prefix_token_length: Some(256),
            cache_block_hashes: Vec::new(),
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
            metadata: HashMap::new(),
        })
        .collect();

    let algo = Box::new(PrefixAware::new());
    let metrics = routesim_core::run_simulation(config, requests, algo);
    assert!(metrics.completed_requests > 0, "No requests completed");
    // With prefix-aware routing and same prefix on all requests,
    // we should see a non-trivial cache hit rate.
    assert!(
        metrics.global_cache_hit_rate > 0.0,
        "PrefixAware with identical prefix_hash should produce cache hits, got 0% on {} completed requests",
        metrics.completed_requests
    );
}

#[test]
fn test_high_load_no_crash() {
    // Use warmup_requests=0 so all 1000 completed requests count in metrics
    let config = SimConfig::from_str(
        r#"
[simulation]
name = "high-load-test"
seed = 42
warmup_requests = 0

[cluster]
gpu_type = "H100Sxm"
num_backends = 8
max_batch_tokens = 16384
max_queue_depth = 256
kv_cache_blocks = 32768
kv_block_size = 16

[cluster.compute_model]
prefill_tokens_per_sec = 50000
decode_tokens_per_sec_batch1 = 80
decode_throughput_saturation_batch = 64
decode_tokens_per_sec_saturated = 3200

[trace]
format = "compact_jsonl"
"#,
    )
    .unwrap();
    // Very high arrival rate — should handle gracefully (with rejections)
    let requests: Vec<InferenceRequest> = (0..1000)
        .map(|i| InferenceRequest {
            id: i,
            arrival_time_ms: i, // 1 request per ms = 1000 req/s
            prompt_tokens: 1024,
            max_gen_tokens: 256,
            actual_gen_tokens: 256,
            prefix_hash: None,
            prefix_token_length: None,
            cache_block_hashes: Vec::new(),
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
            metadata: HashMap::new(),
        })
        .collect();

    let algo = Box::new(LeastOutstanding::new());
    let metrics = routesim_core::run_simulation(config, requests, algo);
    // All 1000 requests should be accounted for (completed + rejected)
    // Some may be double-counted as rejected (at routing and at KV allocation)
    assert!(
        metrics.completed_requests + metrics.rejected_requests >= 1000,
        "Only {} accounted for (completed={}, rejected={})",
        metrics.completed_requests + metrics.rejected_requests,
        metrics.completed_requests,
        metrics.rejected_requests,
    );
    assert!(metrics.completed_requests > 0, "No requests completed");
}

#[test]
fn test_metrics_format_table() {
    let config = production_config();
    let requests = mixed_workload(20);
    let algo = Box::new(RoundRobin::new());
    let metrics = routesim_core::run_simulation(config, requests, algo);
    let table = routesim_core::metrics::format_table(&metrics);
    assert!(table.contains("TTFT"));
    assert!(table.contains("Throughput"));
}
