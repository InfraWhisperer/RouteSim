//! TOML configuration parsing for RouteSim.
//!
//! Defines the complete configuration schema for simulation runs, including
//! cluster topology, compute model, trace source, and simulation parameters.

use crate::backend::ComputeModel;
use crate::topology::{DisaggregatedConfig, GpuProfile};
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse TOML: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("Invalid configuration: {0}")]
    Validation(String),
}

/// Top-level simulation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimConfig {
    pub simulation: SimulationSection,
    pub cluster: ClusterSection,
    pub trace: TraceSection,
}

/// General simulation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSection {
    /// Human-readable name for this simulation.
    #[serde(default = "default_sim_name")]
    pub name: String,
    /// Random seed for reproducibility.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Number of warmup requests to discard from metrics.
    #[serde(default)]
    pub warmup_requests: u64,
}

fn default_sim_name() -> String {
    "simulation".to_string()
}

fn default_seed() -> u64 {
    42
}

/// Cluster configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSection {
    /// GPU type for all backends (unless overridden per-backend).
    #[serde(default = "default_gpu_type")]
    pub gpu_type: String,
    /// Number of backends.
    pub num_backends: u32,
    /// Maximum tokens per batch.
    #[serde(default = "default_max_batch_tokens")]
    pub max_batch_tokens: u32,
    /// Maximum queue depth per backend.
    #[serde(default = "default_max_queue_depth")]
    pub max_queue_depth: u32,
    /// KV cache blocks per backend.
    #[serde(default = "default_kv_cache_blocks")]
    pub kv_cache_blocks: u32,
    /// Tokens per KV cache block.
    #[serde(default = "default_kv_block_size")]
    pub kv_block_size: u32,
    /// Compute model parameters.
    #[serde(default)]
    pub compute_model: ComputeModelSection,
    /// Disaggregated topology config.
    #[serde(default)]
    pub disaggregated: DisaggregatedSection,
}

fn default_gpu_type() -> String {
    "H100Sxm".to_string()
}
fn default_max_batch_tokens() -> u32 {
    16384
}
fn default_max_queue_depth() -> u32 {
    256
}
fn default_kv_cache_blocks() -> u32 {
    32768
}
fn default_kv_block_size() -> u32 {
    16
}

/// Compute model parameters from config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeModelSection {
    #[serde(default = "default_prefill_tps")]
    pub prefill_tokens_per_sec: f64,
    #[serde(default = "default_decode_batch1")]
    pub decode_tokens_per_sec_batch1: f64,
    #[serde(default = "default_decode_sat_batch")]
    pub decode_throughput_saturation_batch: u32,
    #[serde(default = "default_decode_sat_tps")]
    pub decode_tokens_per_sec_saturated: f64,
}

fn default_prefill_tps() -> f64 {
    50_000.0
}
fn default_decode_batch1() -> f64 {
    80.0
}
fn default_decode_sat_batch() -> u32 {
    64
}
fn default_decode_sat_tps() -> f64 {
    3200.0
}

impl Default for ComputeModelSection {
    fn default() -> Self {
        Self {
            prefill_tokens_per_sec: default_prefill_tps(),
            decode_tokens_per_sec_batch1: default_decode_batch1(),
            decode_throughput_saturation_batch: default_decode_sat_batch(),
            decode_tokens_per_sec_saturated: default_decode_sat_tps(),
        }
    }
}

impl From<ComputeModelSection> for ComputeModel {
    fn from(s: ComputeModelSection) -> Self {
        ComputeModel {
            prefill_tokens_per_sec: s.prefill_tokens_per_sec,
            decode_tokens_per_sec_batch1: s.decode_tokens_per_sec_batch1,
            decode_throughput_saturation_batch: s.decode_throughput_saturation_batch,
            decode_tokens_per_sec_saturated: s.decode_tokens_per_sec_saturated,
        }
    }
}

/// Disaggregated topology section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisaggregatedSection {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub prefill_backends: u32,
    #[serde(default)]
    pub decode_backends: u32,
    #[serde(default = "default_kv_transfer_latency")]
    pub kv_transfer_latency_us: u64,
    #[serde(default = "default_kv_transfer_bw")]
    pub kv_transfer_bandwidth_gb_s: f64,
    #[serde(default = "default_num_layers")]
    pub num_layers: u32,
    #[serde(default = "default_head_dim")]
    pub head_dim: u32,
    #[serde(default = "default_num_kv_heads")]
    pub num_kv_heads: u32,
}

fn default_kv_transfer_latency() -> u64 {
    500
}
fn default_kv_transfer_bw() -> f64 {
    50.0
}
fn default_num_layers() -> u32 {
    80
}
fn default_head_dim() -> u32 {
    128
}
fn default_num_kv_heads() -> u32 {
    8
}

impl Default for DisaggregatedSection {
    fn default() -> Self {
        Self {
            enabled: false,
            prefill_backends: 0,
            decode_backends: 0,
            kv_transfer_latency_us: default_kv_transfer_latency(),
            kv_transfer_bandwidth_gb_s: default_kv_transfer_bw(),
            num_layers: default_num_layers(),
            head_dim: default_head_dim(),
            num_kv_heads: default_num_kv_heads(),
        }
    }
}

impl From<DisaggregatedSection> for DisaggregatedConfig {
    fn from(s: DisaggregatedSection) -> Self {
        DisaggregatedConfig {
            enabled: s.enabled,
            prefill_backends: s.prefill_backends,
            decode_backends: s.decode_backends,
            kv_transfer_latency_us: s.kv_transfer_latency_us,
            kv_transfer_bandwidth_gb_s: s.kv_transfer_bandwidth_gb_s,
            num_layers: s.num_layers,
            head_dim: s.head_dim,
            num_kv_heads: s.num_kv_heads,
        }
    }
}

/// Trace source configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSection {
    /// Format: "compact_jsonl", "otel", or "synthetic".
    #[serde(default = "default_trace_format")]
    pub format: String,
    /// Path to trace file (for compact_jsonl and otel).
    pub path: Option<String>,
    /// Synthetic generator name.
    pub generator: Option<String>,
    /// Synthetic: request rate per second.
    pub rate: Option<f64>,
    /// Synthetic: duration in seconds.
    pub duration_sec: Option<u64>,
    /// Synthetic: mean prompt tokens.
    pub prompt_tokens_mean: Option<f64>,
    /// Synthetic: std dev of prompt tokens.
    pub prompt_tokens_std: Option<f64>,
    /// Synthetic: mean generation tokens.
    pub gen_tokens_mean: Option<f64>,
    /// Synthetic: std dev of generation tokens.
    pub gen_tokens_std: Option<f64>,
    /// Synthetic: number of distinct prefixes.
    pub num_prefixes: Option<u32>,
    /// Synthetic: mean prefix length in tokens.
    pub prefix_len_mean: Option<f64>,
}

fn default_trace_format() -> String {
    "compact_jsonl".to_string()
}

impl SimConfig {
    /// Load configuration from a TOML file.
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        Self::from_str(&content)
    }

    /// Parse configuration from a TOML string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, ConfigError> {
        let config: SimConfig = toml::from_str(s)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration consistency.
    fn validate(&self) -> Result<(), ConfigError> {
        if self.cluster.num_backends == 0 {
            return Err(ConfigError::Validation(
                "num_backends must be > 0".to_string(),
            ));
        }
        if self.cluster.disaggregated.enabled {
            let total = self.cluster.disaggregated.prefill_backends
                + self.cluster.disaggregated.decode_backends;
            if total != self.cluster.num_backends {
                return Err(ConfigError::Validation(format!(
                    "Disaggregated prefill ({}) + decode ({}) must equal num_backends ({})",
                    self.cluster.disaggregated.prefill_backends,
                    self.cluster.disaggregated.decode_backends,
                    self.cluster.num_backends,
                )));
            }
            if self.cluster.disaggregated.kv_transfer_bandwidth_gb_s <= 0.0 {
                return Err(ConfigError::Validation(
                    "kv_transfer_bandwidth_gb_s must be > 0 when disaggregated is enabled"
                        .to_string(),
                ));
            }
        }
        if self.cluster.compute_model.prefill_tokens_per_sec <= 0.0 {
            return Err(ConfigError::Validation(
                "prefill_tokens_per_sec must be > 0".to_string(),
            ));
        }
        if self.cluster.compute_model.decode_tokens_per_sec_batch1 <= 0.0 {
            return Err(ConfigError::Validation(
                "decode_tokens_per_sec_batch1 must be > 0".to_string(),
            ));
        }
        if self.cluster.compute_model.decode_tokens_per_sec_saturated <= 0.0 {
            return Err(ConfigError::Validation(
                "decode_tokens_per_sec_saturated must be > 0".to_string(),
            ));
        }
        if self.cluster.max_batch_tokens == 0 {
            return Err(ConfigError::Validation(
                "max_batch_tokens must be > 0".to_string(),
            ));
        }
        if self.cluster.kv_cache_blocks == 0 {
            return Err(ConfigError::Validation(
                "kv_cache_blocks must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Parse the GPU type string into a GpuProfile.
    pub fn gpu_profile(&self) -> GpuProfile {
        match self.cluster.gpu_type.as_str() {
            "H100Sxm" | "H100" | "h100" => GpuProfile::H100Sxm,
            "A100Sxm80" | "A100" | "a100" => GpuProfile::A100Sxm80,
            "L40S" | "l40s" => GpuProfile::L40S,
            _ => GpuProfile::H100Sxm,
        }
    }

    /// Convert compute model section to the engine's ComputeModel.
    pub fn compute_model(&self) -> ComputeModel {
        self.cluster.compute_model.clone().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CONFIG: &str = r#"
[simulation]
name = "test-sim"
seed = 123

[cluster]
gpu_type = "H100Sxm"
num_backends = 4
max_batch_tokens = 8192
max_queue_depth = 128
kv_cache_blocks = 16384
kv_block_size = 16

[cluster.compute_model]
prefill_tokens_per_sec = 50000
decode_tokens_per_sec_batch1 = 80
decode_throughput_saturation_batch = 64
decode_tokens_per_sec_saturated = 3200

[trace]
format = "compact_jsonl"
path = "traces/test.jsonl"
"#;

    #[test]
    fn test_parse_config() {
        let config = SimConfig::from_str(SAMPLE_CONFIG).unwrap();
        assert_eq!(config.simulation.name, "test-sim");
        assert_eq!(config.simulation.seed, 123);
        assert_eq!(config.cluster.num_backends, 4);
        assert_eq!(config.cluster.max_batch_tokens, 8192);
    }

    #[test]
    fn test_gpu_profile_parsing() {
        let config = SimConfig::from_str(SAMPLE_CONFIG).unwrap();
        match config.gpu_profile() {
            GpuProfile::H100Sxm => {}
            _ => panic!("Expected H100Sxm"),
        }
    }

    #[test]
    fn test_validation_zero_backends() {
        let toml = r#"
[simulation]
name = "test"

[cluster]
num_backends = 0

[trace]
format = "compact_jsonl"
"#;
        assert!(SimConfig::from_str(toml).is_err());
    }

    #[test]
    fn test_validation_disaggregated_mismatch() {
        let toml = r#"
[simulation]
name = "test"

[cluster]
num_backends = 8

[cluster.disaggregated]
enabled = true
prefill_backends = 2
decode_backends = 4

[trace]
format = "compact_jsonl"
"#;
        assert!(SimConfig::from_str(toml).is_err());
    }

    #[test]
    fn test_defaults() {
        let toml = r#"
[simulation]

[cluster]
num_backends = 4

[trace]
format = "compact_jsonl"
"#;
        let config = SimConfig::from_str(toml).unwrap();
        assert_eq!(config.simulation.seed, 42);
        assert_eq!(config.cluster.max_batch_tokens, 16384);
    }

    #[test]
    fn test_validation_zero_prefill_tps() {
        let toml = r#"
[simulation]
[cluster]
num_backends = 4
[cluster.compute_model]
prefill_tokens_per_sec = 0
[trace]
format = "compact_jsonl"
"#;
        assert!(SimConfig::from_str(toml).is_err());
    }

    #[test]
    fn test_validation_zero_decode_batch1() {
        let toml = r#"
[simulation]
[cluster]
num_backends = 4
[cluster.compute_model]
decode_tokens_per_sec_batch1 = 0
[trace]
format = "compact_jsonl"
"#;
        assert!(SimConfig::from_str(toml).is_err());
    }

    #[test]
    fn test_validation_zero_decode_saturated() {
        let toml = r#"
[simulation]
[cluster]
num_backends = 4
[cluster.compute_model]
decode_tokens_per_sec_saturated = 0
[trace]
format = "compact_jsonl"
"#;
        assert!(SimConfig::from_str(toml).is_err());
    }

    #[test]
    fn test_validation_zero_max_batch_tokens() {
        let toml = r#"
[simulation]
[cluster]
num_backends = 4
max_batch_tokens = 0
[trace]
format = "compact_jsonl"
"#;
        assert!(SimConfig::from_str(toml).is_err());
    }

    #[test]
    fn test_validation_zero_kv_cache_blocks() {
        let toml = r#"
[simulation]
[cluster]
num_backends = 4
kv_cache_blocks = 0
[trace]
format = "compact_jsonl"
"#;
        assert!(SimConfig::from_str(toml).is_err());
    }

    #[test]
    fn test_validation_zero_bandwidth_disaggregated() {
        let toml = r#"
[simulation]
[cluster]
num_backends = 4
[cluster.disaggregated]
enabled = true
prefill_backends = 2
decode_backends = 2
kv_transfer_bandwidth_gb_s = 0
[trace]
format = "compact_jsonl"
"#;
        assert!(SimConfig::from_str(toml).is_err());
    }

    #[test]
    fn test_validation_zero_bandwidth_non_disaggregated_ok() {
        let toml = r#"
[simulation]
[cluster]
num_backends = 4
[cluster.disaggregated]
enabled = false
kv_transfer_bandwidth_gb_s = 0
[trace]
format = "compact_jsonl"
"#;
        assert!(SimConfig::from_str(toml).is_ok());
    }
}
