//! Cluster topology definitions.
//!
//! Supports both unified backends (each node handles prefill + decode) and
//! disaggregated topologies where prefill and decode are handled by separate
//! node pools with KV cache transfer between them.

use serde::{Deserialize, Serialize};

/// Role a backend plays in the cluster.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendRole {
    /// Handles both prefill and decode (unified serving).
    #[default]
    Both,
    /// Prefill-only node in a disaggregated topology.
    Prefill,
    /// Decode-only node in a disaggregated topology.
    Decode,
}

/// State of a backend.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendState {
    /// Ready to accept requests.
    #[default]
    Idle,
    /// Actively processing a batch.
    Processing,
    /// Draining: finishing current work but not accepting new requests.
    Draining,
    /// Offline / unhealthy.
    Offline,
}

/// GPU hardware profile that determines compute characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuProfile {
    /// NVIDIA H100 SXM (80GB HBM3, ~3.9 TB/s bandwidth).
    H100Sxm,
    /// NVIDIA A100 SXM (80GB HBM2e, ~2.0 TB/s bandwidth).
    A100Sxm80,
    /// NVIDIA L40S (48GB GDDR6, ~864 GB/s bandwidth).
    L40S,
    /// Custom GPU profile with user-specified parameters.
    Custom {
        hbm_gb: f32,
        bandwidth_tb_s: f32,
        flops_fp16_tflops: f32,
    },
}

impl GpuProfile {
    /// Memory capacity in GB.
    pub fn memory_gb(&self) -> f32 {
        match self {
            GpuProfile::H100Sxm => 80.0,
            GpuProfile::A100Sxm80 => 80.0,
            GpuProfile::L40S => 48.0,
            GpuProfile::Custom { hbm_gb, .. } => *hbm_gb,
        }
    }

    /// Memory bandwidth in TB/s.
    pub fn bandwidth_tb_s(&self) -> f32 {
        match self {
            GpuProfile::H100Sxm => 3.35,
            GpuProfile::A100Sxm80 => 2.0,
            GpuProfile::L40S => 0.864,
            GpuProfile::Custom { bandwidth_tb_s, .. } => *bandwidth_tb_s,
        }
    }

    /// FP16 compute in TFLOPS.
    pub fn flops_fp16_tflops(&self) -> f32 {
        match self {
            GpuProfile::H100Sxm => 989.0,
            GpuProfile::A100Sxm80 => 312.0,
            GpuProfile::L40S => 362.0,
            GpuProfile::Custom {
                flops_fp16_tflops, ..
            } => *flops_fp16_tflops,
        }
    }

    /// Hourly cost estimate in USD (rough cloud pricing).
    pub fn estimated_cost_per_hour(&self) -> f64 {
        match self {
            GpuProfile::H100Sxm => 3.50,
            GpuProfile::A100Sxm80 => 2.00,
            GpuProfile::L40S => 1.00,
            GpuProfile::Custom { .. } => 2.00,
        }
    }
}

/// Configuration for disaggregated prefill/decode topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisaggregatedConfig {
    /// Whether disaggregated mode is enabled.
    pub enabled: bool,
    /// Number of prefill-dedicated backends.
    pub prefill_backends: u32,
    /// Number of decode-dedicated backends.
    pub decode_backends: u32,
    /// Latency for KV cache transfer in microseconds.
    pub kv_transfer_latency_us: u64,
    /// KV transfer bandwidth in GB/s.
    pub kv_transfer_bandwidth_gb_s: f64,
}

impl Default for DisaggregatedConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            prefill_backends: 0,
            decode_backends: 0,
            kv_transfer_latency_us: 500,
            kv_transfer_bandwidth_gb_s: 50.0,
        }
    }
}

impl DisaggregatedConfig {
    /// Estimate KV transfer time in microseconds for a given number of tokens.
    ///
    /// Assumes ~2 bytes per element, 2 KV heads, and a model dimension of 128 per head.
    /// Total bytes = num_tokens * 2 * 2 * 128 * num_layers (default 80 for large models).
    pub fn estimate_transfer_time_us(&self, num_tokens: u32, num_layers: u32) -> u64 {
        let bytes_per_token = 2u64 * 2 * 128 * num_layers as u64; // KV for all layers
        let total_bytes = num_tokens as u64 * bytes_per_token;
        let bandwidth_bytes_per_us = (self.kv_transfer_bandwidth_gb_s * 1e3) / 1e6; // GB/s -> bytes/us
        let transfer_us = (total_bytes as f64 / bandwidth_bytes_per_us) as u64;
        self.kv_transfer_latency_us + transfer_us
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_profile_memory() {
        assert_eq!(GpuProfile::H100Sxm.memory_gb(), 80.0);
        assert_eq!(GpuProfile::L40S.memory_gb(), 48.0);
    }

    #[test]
    fn test_backend_role_default() {
        assert_eq!(BackendRole::default(), BackendRole::Both);
    }

    #[test]
    fn test_disaggregated_transfer_time() {
        let config = DisaggregatedConfig {
            enabled: true,
            prefill_backends: 2,
            decode_backends: 6,
            kv_transfer_latency_us: 500,
            kv_transfer_bandwidth_gb_s: 50.0,
        };
        let time = config.estimate_transfer_time_us(1024, 80);
        // Should be latency + transfer time
        assert!(time > 500);
    }

    #[test]
    fn test_backend_state_default() {
        assert_eq!(BackendState::default(), BackendState::Idle);
    }
}
