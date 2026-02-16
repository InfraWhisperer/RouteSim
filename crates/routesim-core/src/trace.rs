//! Trace ingestion for RouteSim.
//!
//! Supports two input formats:
//! - **Compact JSONL**: One JSON object per line with minimal fields.
//! - **OpenTelemetry JSON**: Parsed from OTEL trace exports, extracting
//!   LLM inference spans with token counts.

use crate::request::InferenceRequest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TraceError {
    #[error("Failed to read trace file: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse JSON at line {line}: {source}")]
    JsonParse {
        line: usize,
        source: serde_json::Error,
    },
    #[error("Unsupported trace format: {0}")]
    UnsupportedFormat(String),
    #[error("Missing required field: {0}")]
    MissingField(String),
}

/// A compact JSONL trace record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactTraceRecord {
    /// Arrival timestamp in milliseconds.
    pub ts: u64,
    /// Prompt token count.
    pub prompt_tokens: u32,
    /// Generation token count.
    pub gen_tokens: u32,
    /// Optional prefix hash (string, will be hashed to u64).
    pub prefix_hash: Option<String>,
    /// Optional prefix length in tokens.
    pub prefix_len: Option<u32>,
    /// Optional conversation ID.
    pub conversation_id: Option<String>,
    /// Optional LoRA adapter name.
    pub lora_adapter: Option<String>,
    /// Optional priority.
    pub priority: Option<u8>,
}

/// An OpenTelemetry span record (simplified).
#[derive(Debug, Clone, Deserialize)]
struct OtelSpan {
    #[serde(rename = "startTimeUnixNano")]
    start_time_unix_nano: Option<String>,
    #[serde(default)]
    attributes: Vec<OtelAttribute>,
}

#[derive(Debug, Clone, Deserialize)]
struct OtelAttribute {
    key: String,
    value: OtelValue,
}

#[derive(Debug, Clone, Deserialize)]
struct OtelValue {
    #[serde(rename = "intValue")]
    int_value: Option<String>,
    #[serde(rename = "stringValue")]
    string_value: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct OtelResourceSpans {
    #[serde(rename = "scopeSpans")]
    scope_spans: Vec<OtelScopeSpans>,
}

#[derive(Debug, Clone, Deserialize)]
struct OtelScopeSpans {
    spans: Vec<OtelSpan>,
}

#[derive(Debug, Clone, Deserialize)]
struct OtelTrace {
    #[serde(rename = "resourceSpans")]
    resource_spans: Vec<OtelResourceSpans>,
}

/// Load a trace from a file, auto-detecting format.
pub fn load_trace(path: &Path, format: &str) -> Result<Vec<InferenceRequest>, TraceError> {
    match format {
        "compact_jsonl" | "jsonl" => load_compact_jsonl(path),
        "otel" | "opentelemetry" => load_otel_json(path),
        other => Err(TraceError::UnsupportedFormat(other.to_string())),
    }
}

/// Load a compact JSONL trace file.
pub fn load_compact_jsonl(path: &Path) -> Result<Vec<InferenceRequest>, TraceError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    parse_compact_jsonl(reader)
}

/// Parse compact JSONL from any reader.
pub fn parse_compact_jsonl<R: Read>(
    reader: BufReader<R>,
) -> Result<Vec<InferenceRequest>, TraceError> {
    let mut requests = Vec::new();
    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let record: CompactTraceRecord =
            serde_json::from_str(trimmed).map_err(|e| TraceError::JsonParse {
                line: line_num + 1,
                source: e,
            })?;
        requests.push(record_to_request(requests.len() as u64, record));
    }

    // Sort by arrival time
    requests.sort_by_key(|r| r.arrival_time_ms);
    Ok(requests)
}

/// Load an OpenTelemetry JSON trace.
pub fn load_otel_json(path: &Path) -> Result<Vec<InferenceRequest>, TraceError> {
    let content = std::fs::read_to_string(path)?;
    let trace: OtelTrace =
        serde_json::from_str(&content).map_err(|e| TraceError::JsonParse { line: 0, source: e })?;

    let mut requests = Vec::new();
    for resource_spans in &trace.resource_spans {
        for scope_spans in &resource_spans.scope_spans {
            for span in &scope_spans.spans {
                if let Some(req) = otel_span_to_request(requests.len() as u64, span) {
                    requests.push(req);
                }
            }
        }
    }

    requests.sort_by_key(|r| r.arrival_time_ms);
    Ok(requests)
}

/// Convert a compact trace record to an InferenceRequest.
fn record_to_request(id: u64, record: CompactTraceRecord) -> InferenceRequest {
    let prefix_hash = record.prefix_hash.as_ref().map(|s| hash_string(s));

    InferenceRequest {
        id,
        arrival_time_ms: record.ts,
        prompt_tokens: record.prompt_tokens,
        max_gen_tokens: record.gen_tokens,
        actual_gen_tokens: record.gen_tokens,
        prefix_hash,
        prefix_token_length: record.prefix_len,
        conversation_id: record.conversation_id,
        lora_adapter: record.lora_adapter,
        priority: record.priority.unwrap_or(0),
        metadata: HashMap::new(),
    }
}

/// Convert an OTEL span to an InferenceRequest, if it has the right attributes.
fn otel_span_to_request(id: u64, span: &OtelSpan) -> Option<InferenceRequest> {
    let attrs: HashMap<&str, &OtelAttribute> = span
        .attributes
        .iter()
        .map(|a| (a.key.as_str(), a))
        .collect();

    let prompt_tokens = attrs
        .get("llm.prompt_tokens")
        .or_else(|| attrs.get("gen_ai.prompt.tokens"))
        .and_then(|a| a.value.int_value.as_ref())
        .and_then(|v| v.parse::<u32>().ok())?;

    let gen_tokens = attrs
        .get("llm.completion_tokens")
        .or_else(|| attrs.get("gen_ai.completion.tokens"))
        .and_then(|a| a.value.int_value.as_ref())
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(64);

    let arrival_ms = span
        .start_time_unix_nano
        .as_ref()
        .and_then(|s| s.parse::<u64>().ok())
        .map(|ns| ns / 1_000_000) // nano -> ms
        .unwrap_or(0);

    let prefix_hash = attrs
        .get("llm.system_prompt_hash")
        .and_then(|a| a.value.string_value.as_ref())
        .map(|s| hash_string(s));

    Some(InferenceRequest {
        id,
        arrival_time_ms: arrival_ms,
        prompt_tokens,
        max_gen_tokens: gen_tokens,
        actual_gen_tokens: gen_tokens,
        prefix_hash,
        prefix_token_length: None,
        conversation_id: attrs
            .get("llm.conversation_id")
            .and_then(|a| a.value.string_value.clone()),
        lora_adapter: attrs
            .get("llm.model")
            .and_then(|a| a.value.string_value.clone()),
        priority: 0,
        metadata: HashMap::new(),
    })
}

/// Simple string hash using FNV-1a.
fn hash_string(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Write requests to compact JSONL format.
pub fn write_compact_jsonl(requests: &[InferenceRequest], path: &Path) -> Result<(), TraceError> {
    use std::io::Write;
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);

    for req in requests {
        let record = serde_json::json!({
            "ts": req.arrival_time_ms,
            "prompt_tokens": req.prompt_tokens,
            "gen_tokens": req.actual_gen_tokens,
            "prefix_hash": req.prefix_hash.map(|h| format!("{:x}", h)),
            "prefix_len": req.prefix_token_length,
        });
        serde_json::to_writer(&mut writer, &record)
            .map_err(|e| TraceError::JsonParse { line: 0, source: e })?;
        writeln!(writer)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_compact_jsonl() {
        let data = r#"{"ts": 1000, "prompt_tokens": 512, "gen_tokens": 128, "prefix_hash": "abc123", "prefix_len": 256}
{"ts": 1050, "prompt_tokens": 1024, "gen_tokens": 64, "prefix_hash": "abc123", "prefix_len": 256}
{"ts": 1200, "prompt_tokens": 128, "gen_tokens": 256, "prefix_hash": "def456", "prefix_len": 64}
"#;
        let reader = BufReader::new(data.as_bytes());
        let requests = parse_compact_jsonl(reader).unwrap();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[0].arrival_time_ms, 1000);
        assert_eq!(requests[0].prompt_tokens, 512);
        assert_eq!(requests[0].actual_gen_tokens, 128);
        assert!(requests[0].prefix_hash.is_some());
        // Same prefix hash string should produce same hash
        assert_eq!(requests[0].prefix_hash, requests[1].prefix_hash);
        assert_ne!(requests[0].prefix_hash, requests[2].prefix_hash);
    }

    #[test]
    fn test_parse_empty_lines() {
        let data = "\n\n{\"ts\": 100, \"prompt_tokens\": 32, \"gen_tokens\": 16}\n\n";
        let reader = BufReader::new(data.as_bytes());
        let requests = parse_compact_jsonl(reader).unwrap();
        assert_eq!(requests.len(), 1);
    }

    #[test]
    fn test_parse_comments() {
        let data =
            "# This is a comment\n{\"ts\": 100, \"prompt_tokens\": 32, \"gen_tokens\": 16}\n";
        let reader = BufReader::new(data.as_bytes());
        let requests = parse_compact_jsonl(reader).unwrap();
        assert_eq!(requests.len(), 1);
    }

    #[test]
    fn test_hash_string_deterministic() {
        assert_eq!(hash_string("abc123"), hash_string("abc123"));
        assert_ne!(hash_string("abc123"), hash_string("def456"));
    }

    #[test]
    fn test_sorted_by_arrival_time() {
        let data = r#"{"ts": 200, "prompt_tokens": 32, "gen_tokens": 16}
{"ts": 100, "prompt_tokens": 32, "gen_tokens": 16}
{"ts": 300, "prompt_tokens": 32, "gen_tokens": 16}
"#;
        let reader = BufReader::new(data.as_bytes());
        let requests = parse_compact_jsonl(reader).unwrap();
        assert_eq!(requests[0].arrival_time_ms, 100);
        assert_eq!(requests[1].arrival_time_ms, 200);
        assert_eq!(requests[2].arrival_time_ms, 300);
    }
}
