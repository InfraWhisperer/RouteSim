use criterion::{black_box, criterion_group, criterion_main, Criterion};
use routesim_algorithms::RoundRobin;
use routesim_core::config::SimConfig;
use routesim_core::request::InferenceRequest;
use std::collections::HashMap;

fn sample_requests(n: usize) -> Vec<InferenceRequest> {
    (0..n)
        .map(|i| InferenceRequest {
            id: i as u64,
            arrival_time_ms: (i as u64) * 5,
            prompt_tokens: 256,
            max_gen_tokens: 64,
            actual_gen_tokens: 64,
            prefix_hash: Some((i % 10) as u64),
            prefix_token_length: Some(128),
            cache_block_hashes: Vec::new(),
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
            metadata: HashMap::new(),
        })
        .collect()
}

fn test_config(num_backends: u32) -> SimConfig {
    SimConfig::from_str(&format!(
        r#"
[simulation]
name = "bench"
seed = 42

[cluster]
num_backends = {}
max_batch_tokens = 16384
max_queue_depth = 256
kv_cache_blocks = 32768
kv_block_size = 16

[trace]
format = "compact_jsonl"
"#,
        num_backends
    ))
    .unwrap()
}

fn bench_simulation_1k(c: &mut Criterion) {
    let config = test_config(8);
    let requests = sample_requests(1_000);

    c.bench_function("simulate_1k_requests_8_backends", |b| {
        b.iter(|| {
            let algo = Box::new(RoundRobin::new());
            routesim_core::run_simulation(
                black_box(config.clone()),
                black_box(requests.clone()),
                algo,
            )
        })
    });
}

fn bench_simulation_10k(c: &mut Criterion) {
    let config = test_config(8);
    let requests = sample_requests(10_000);

    c.bench_function("simulate_10k_requests_8_backends", |b| {
        b.iter(|| {
            let algo = Box::new(RoundRobin::new());
            routesim_core::run_simulation(
                black_box(config.clone()),
                black_box(requests.clone()),
                algo,
            )
        })
    });
}

criterion_group!(benches, bench_simulation_1k, bench_simulation_10k);
criterion_main!(benches);
