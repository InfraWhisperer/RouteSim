/// Integration tests for trace ingestion.
use routesim_core::trace;

#[test]
fn test_load_example_trace() {
    let path = std::path::Path::new("traces/example_trace.jsonl");
    if !path.exists() {
        return; // Skip if trace file doesn't exist (CI may not have it)
    }
    let requests = trace::load_trace(path, "compact_jsonl").unwrap();
    assert!(!requests.is_empty());

    // Verify sorted by arrival time
    for i in 1..requests.len() {
        assert!(requests[i].arrival_time_ms >= requests[i - 1].arrival_time_ms);
    }
}

#[test]
fn test_write_and_read_trace() {
    let tmp_dir = std::env::temp_dir();
    let tmp_path = tmp_dir.join("routesim_test_trace.jsonl");

    let requests = vec![
        routesim_core::InferenceRequest {
            id: 0,
            arrival_time_ms: 0,
            prompt_tokens: 256,
            max_gen_tokens: 64,
            actual_gen_tokens: 64,
            prefix_hash: Some(0xABC),
            prefix_token_length: Some(128),
            cache_block_hashes: Vec::new(),
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
            metadata: std::collections::HashMap::new(),
        },
        routesim_core::InferenceRequest {
            id: 1,
            arrival_time_ms: 100,
            prompt_tokens: 512,
            max_gen_tokens: 128,
            actual_gen_tokens: 128,
            prefix_hash: None,
            prefix_token_length: None,
            cache_block_hashes: Vec::new(),
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
            metadata: std::collections::HashMap::new(),
        },
    ];

    trace::write_compact_jsonl(&requests, &tmp_path).unwrap();
    let loaded = trace::load_trace(&tmp_path, "compact_jsonl").unwrap();

    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].prompt_tokens, 256);
    assert_eq!(loaded[1].prompt_tokens, 512);

    // Cleanup
    let _ = std::fs::remove_file(&tmp_path);
}

#[test]
fn test_invalid_format() {
    let path = std::path::Path::new("traces/example_trace.jsonl");
    let result = trace::load_trace(path, "invalid_format");
    assert!(result.is_err());
}
