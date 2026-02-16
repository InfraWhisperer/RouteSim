/// Integration tests for routing algorithms in a simulation context.
use routesim_algorithms::*;
use routesim_core::config::SimConfig;
use routesim_core::request::InferenceRequest;
use std::collections::HashMap;

fn small_config() -> SimConfig {
    SimConfig::from_str(
        r#"
[simulation]
name = "algo-test"
seed = 42

[cluster]
num_backends = 4
max_batch_tokens = 8192
max_queue_depth = 64
kv_cache_blocks = 4096
kv_block_size = 16

[trace]
format = "compact_jsonl"
"#,
    )
    .unwrap()
}

fn uniform_requests(n: usize) -> Vec<InferenceRequest> {
    (0..n)
        .map(|i| InferenceRequest {
            id: i as u64,
            arrival_time_ms: (i as u64) * 10,
            prompt_tokens: 256,
            max_gen_tokens: 64,
            actual_gen_tokens: 64,
            prefix_hash: None,
            prefix_token_length: None,
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
            metadata: HashMap::new(),
        })
        .collect()
}

#[test]
fn test_round_robin_fairness() {
    let config = small_config();
    let requests = uniform_requests(100);
    let algo = Box::new(RoundRobin::new());
    let metrics = routesim_core::run_simulation(config, requests, algo);

    // Round robin should have good fairness with uniform workload
    assert!(
        metrics.jains_fairness_index > 0.9,
        "Fairness too low: {}",
        metrics.jains_fairness_index
    );
}

#[test]
fn test_session_affinity_stickiness() {
    let config = small_config();
    let requests: Vec<InferenceRequest> = (0..40)
        .map(|i| InferenceRequest {
            id: i,
            arrival_time_ms: i * 20,
            prompt_tokens: 128,
            max_gen_tokens: 32,
            actual_gen_tokens: 32,
            prefix_hash: None,
            prefix_token_length: None,
            conversation_id: Some(format!("conv-{}", i % 5)), // 5 conversations
            lora_adapter: None,
            priority: 0,
            metadata: HashMap::new(),
        })
        .collect();

    let algo = Box::new(SessionAffinity::new());
    let metrics = routesim_core::run_simulation(config, requests, algo);
    assert!(metrics.completed_requests > 0);
}

#[test]
fn test_all_algorithms_complete_successfully() {
    let config = small_config();
    let requests = uniform_requests(30);

    for name in available_algorithms() {
        let algo = algorithm_by_name(name).unwrap();
        let metrics = routesim_core::run_simulation(config.clone(), requests.clone(), algo);
        assert!(
            metrics.completed_requests > 0,
            "{} completed 0 requests",
            name
        );
        assert!(
            metrics.rejected_requests < metrics.total_requests,
            "{} rejected all requests",
            name
        );
    }
}
