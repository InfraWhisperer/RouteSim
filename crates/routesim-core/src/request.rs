//! Request model for LLM inference simulation.
//!
//! Each [`InferenceRequest`] represents a single inference call with token counts,
//! optional prefix information for cache-aware routing, and metadata for
//! session affinity and multi-model routing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single LLM inference request flowing through the simulated system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// Unique request identifier.
    pub id: u64,
    /// Arrival time in simulation milliseconds.
    pub arrival_time_ms: u64,
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Maximum generation tokens the client will accept.
    pub max_gen_tokens: u32,
    /// Actual generation tokens (known from trace, or sampled from a distribution).
    pub actual_gen_tokens: u32,
    /// Hash of the system prompt / shared prefix for cache-aware routing.
    pub prefix_hash: Option<u64>,
    /// Length of the shared prefix in tokens.
    pub prefix_token_length: Option<u32>,
    /// Conversation identifier for session affinity.
    pub conversation_id: Option<String>,
    /// LoRA adapter name for multi-model routing.
    pub lora_adapter: Option<String>,
    /// Priority level (0 = lowest, 255 = highest).
    pub priority: u8,
    /// Arbitrary key-value metadata.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl InferenceRequest {
    /// Total tokens this request will consume (prompt + generation).
    pub fn total_tokens(&self) -> u32 {
        self.prompt_tokens + self.actual_gen_tokens
    }

    /// Number of new (non-prefix) prompt tokens that need prefill computation.
    pub fn new_prompt_tokens(&self) -> u32 {
        match self.prefix_token_length {
            Some(prefix_len) => self.prompt_tokens.saturating_sub(prefix_len),
            None => self.prompt_tokens,
        }
    }
}

/// A request sitting in a backend's queue, annotated with queue entry time.
#[derive(Debug, Clone)]
pub struct QueuedRequest {
    pub request: InferenceRequest,
    pub enqueue_time_ms: u64,
}

/// Tracking state for a request actively being processed by a backend.
#[derive(Debug, Clone)]
pub struct ActiveRequest {
    pub request: InferenceRequest,
    pub prefill_start_ms: Option<u64>,
    pub prefill_end_ms: Option<u64>,
    pub first_token_ms: Option<u64>,
    pub tokens_generated: u32,
    pub last_token_ms: Option<u64>,
    pub prefix_cache_hit: bool,
}

impl ActiveRequest {
    pub fn new(request: InferenceRequest) -> Self {
        Self {
            request,
            prefill_start_ms: None,
            prefill_end_ms: None,
            first_token_ms: None,
            tokens_generated: 0,
            last_token_ms: None,
            prefix_cache_hit: false,
        }
    }

    /// Time to first token, if prefill has completed.
    pub fn ttft_ms(&self) -> Option<u64> {
        match (self.first_token_ms, Some(self.request.arrival_time_ms)) {
            (Some(first), Some(arrival)) => Some(first.saturating_sub(arrival)),
            _ => None,
        }
    }

    /// Whether generation is complete.
    pub fn is_complete(&self) -> bool {
        self.tokens_generated >= self.request.actual_gen_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> InferenceRequest {
        InferenceRequest {
            id: 1,
            arrival_time_ms: 1000,
            prompt_tokens: 512,
            max_gen_tokens: 256,
            actual_gen_tokens: 128,
            prefix_hash: Some(0xABCD),
            prefix_token_length: Some(256),
            conversation_id: None,
            lora_adapter: None,
            priority: 0,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_total_tokens() {
        let req = sample_request();
        assert_eq!(req.total_tokens(), 640);
    }

    #[test]
    fn test_new_prompt_tokens_with_prefix() {
        let req = sample_request();
        assert_eq!(req.new_prompt_tokens(), 256);
    }

    #[test]
    fn test_new_prompt_tokens_without_prefix() {
        let mut req = sample_request();
        req.prefix_token_length = None;
        assert_eq!(req.new_prompt_tokens(), 512);
    }

    #[test]
    fn test_active_request_ttft() {
        let req = sample_request();
        let mut active = ActiveRequest::new(req);
        assert_eq!(active.ttft_ms(), None);
        active.first_token_ms = Some(1050);
        assert_eq!(active.ttft_ms(), Some(50));
    }

    #[test]
    fn test_active_request_completion() {
        let req = sample_request();
        let mut active = ActiveRequest::new(req);
        assert!(!active.is_complete());
        active.tokens_generated = 128;
        assert!(active.is_complete());
    }
}
