//! Metrics collection and aggregation for simulation runs.
//!
//! Tracks per-request latency metrics (TTFT, TBT, end-to-end), throughput,
//! cache statistics, fairness metrics, and cost estimates.

use crate::backend::SimulatedBackend;
use crate::kv_cache::KvCacheStats;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-request completion record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetric {
    pub request_id: u64,
    pub backend_id: u32,
    pub arrival_time_ms: u64,
    pub queue_wait_ms: u64,
    pub ttft_ms: u64,
    pub total_latency_ms: u64,
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    pub prefix_cache_hit: bool,
    /// Fraction of the request's cache_block_hashes that were already cached
    /// on the routed backend (0.0 if no block hashes). Measures block-level
    /// cache affinity for prefix_overlap routing.
    pub block_cache_overlap: f64,
    pub tbt_samples_ms: Vec<f64>,
}

impl RequestMetric {
    /// Average time between tokens.
    pub fn avg_tbt_ms(&self) -> f64 {
        if self.tbt_samples_ms.is_empty() {
            return 0.0;
        }
        self.tbt_samples_ms.iter().sum::<f64>() / self.tbt_samples_ms.len() as f64
    }
}

/// Percentile values for a distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Percentiles {
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

impl Percentiles {
    /// Compute percentiles from a slice of values.
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                p50: 0.0,
                p75: 0.0,
                p90: 0.0,
                p95: 0.0,
                p99: 0.0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
            };
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let mean = sorted.iter().sum::<f64>() / n as f64;

        Self {
            p50: percentile_sorted(&sorted, 50.0),
            p75: percentile_sorted(&sorted, 75.0),
            p90: percentile_sorted(&sorted, 90.0),
            p95: percentile_sorted(&sorted, 95.0),
            p99: percentile_sorted(&sorted, 99.0),
            min: sorted[0],
            max: sorted[n - 1],
            mean,
        }
    }
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Aggregated metrics for an entire simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationMetrics {
    /// Algorithm name.
    pub algorithm: String,
    /// Total simulation duration in ms.
    pub duration_ms: u64,
    /// Total requests processed.
    pub total_requests: u64,
    /// Requests that completed successfully.
    pub completed_requests: u64,
    /// Requests that were rejected.
    pub rejected_requests: u64,

    // Latency
    pub ttft: Percentiles,
    pub tbt: Percentiles,
    pub end_to_end_latency: Percentiles,
    pub queue_wait: Percentiles,

    // Throughput
    pub requests_per_sec: f64,
    pub prompt_tokens_per_sec: f64,
    pub gen_tokens_per_sec: f64,
    pub total_tokens_per_sec: f64,

    // Cache
    pub global_cache_hit_rate: f64,
    /// Average block-level cache overlap at routing time (0.0–1.0).
    /// Measures how well the routing algorithm exploits content-addressed
    /// block caching (relevant for `prefix_overlap` and Mooncake traces).
    pub block_cache_reuse_rate: f64,
    pub per_backend_cache_stats: Vec<KvCacheStats>,

    // Fairness
    pub load_cv: f64,
    pub jains_fairness_index: f64,
    pub max_min_queue_ratio: f64,

    // Cost
    pub gpu_seconds_per_request: f64,
    pub estimated_cost_per_1k_tokens: f64,

    // Per-backend summary
    pub per_backend_requests: Vec<u64>,

    // Custom algorithm metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Collector that accumulates per-request metrics during simulation.
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    /// Per-request records.
    records: Vec<RequestMetric>,
    /// Number of requests to discard at the start (warmup).
    warmup_count: u64,
    /// Rejected request count.
    rejected: u64,
    /// Timestamps of KV cache utilization samples: (time_ms, backend_id, utilization).
    cache_utilization_samples: Vec<(u64, u32, f32)>,
}

impl MetricsCollector {
    /// Create a new collector.
    pub fn new(warmup_count: u64) -> Self {
        Self {
            records: Vec::new(),
            warmup_count,
            rejected: 0,
            cache_utilization_samples: Vec::new(),
        }
    }

    /// Record a completed request.
    pub fn record(&mut self, metric: RequestMetric) {
        self.records.push(metric);
    }

    /// Record a rejected request.
    pub fn record_rejection(&mut self) {
        self.rejected += 1;
    }

    /// Record a cache utilization sample.
    pub fn sample_cache_utilization(&mut self, time_ms: u64, backend_id: u32, utilization: f32) {
        self.cache_utilization_samples
            .push((time_ms, backend_id, utilization));
    }

    /// Number of completed requests recorded.
    pub fn completed_count(&self) -> u64 {
        self.records.len() as u64
    }

    /// Get all per-request records (post-warmup).
    pub fn records(&self) -> &[RequestMetric] {
        let skip = self.warmup_count as usize;
        if skip >= self.records.len() {
            return &[];
        }
        &self.records[skip..]
    }

    /// Aggregate all metrics into a summary.
    ///
    /// If `warmup_requests` exceeds the number of completed requests, all
    /// post-warmup records are empty and metrics will be zero. A warning is
    /// printed to stderr in this case.
    pub fn aggregate(
        &self,
        algorithm: &str,
        backends: &[SimulatedBackend],
        custom_metrics: HashMap<String, f64>,
    ) -> SimulationMetrics {
        let records = self.records();

        if records.is_empty() && !self.records.is_empty() {
            eprintln!(
                "WARNING: warmup_requests ({}) >= completed requests ({}). \
                 All latency/throughput metrics will be zero. \
                 Reduce warmup_requests or increase the number of requests.",
                self.warmup_count,
                self.records.len(),
            );
        }

        // Latency distributions
        let ttft_values: Vec<f64> = records.iter().map(|r| r.ttft_ms as f64).collect();
        let e2e_values: Vec<f64> = records.iter().map(|r| r.total_latency_ms as f64).collect();
        let queue_values: Vec<f64> = records.iter().map(|r| r.queue_wait_ms as f64).collect();
        let tbt_values: Vec<f64> = records
            .iter()
            .flat_map(|r| r.tbt_samples_ms.iter())
            .copied()
            .collect();

        // Duration
        let duration_ms = if records.is_empty() {
            0
        } else {
            let first = records.iter().map(|r| r.arrival_time_ms).min().unwrap_or(0);
            let last = records
                .iter()
                .map(|r| r.arrival_time_ms + r.total_latency_ms)
                .max()
                .unwrap_or(0);
            last.saturating_sub(first)
        };

        // Throughput
        let duration_sec = duration_ms as f64 / 1000.0;
        let total_prompt_tokens: u64 = records.iter().map(|r| r.prompt_tokens as u64).sum();
        let total_gen_tokens: u64 = records.iter().map(|r| r.gen_tokens as u64).sum();
        let completed = records.len() as u64;

        // Cache
        let cache_hits = records.iter().filter(|r| r.prefix_cache_hit).count() as f64;
        let cache_lookups = records.len() as f64;
        let global_cache_hit_rate = if cache_lookups > 0.0 {
            cache_hits / cache_lookups
        } else {
            0.0
        };

        // Block-level cache reuse (for prefix_overlap / Mooncake traces).
        // block_cache_overlap is -1.0 for requests without block hashes;
        // only average over requests that actually had block hashes.
        let block_overlaps: Vec<f64> = records
            .iter()
            .map(|r| r.block_cache_overlap)
            .filter(|&v| v >= 0.0)
            .collect();
        let block_cache_reuse_rate = if block_overlaps.is_empty() {
            0.0
        } else {
            block_overlaps.iter().sum::<f64>() / block_overlaps.len() as f64
        };

        let per_backend_cache_stats: Vec<KvCacheStats> =
            backends.iter().map(|b| b.kv_cache.stats()).collect();

        // Fairness
        let per_backend_requests: Vec<u64> =
            backends.iter().map(|b| b.total_requests_served).collect();
        let load_cv = coefficient_of_variation(&per_backend_requests);
        let jains = jains_fairness_index(&per_backend_requests);
        let max_queue = backends.iter().map(|b| b.queue_depth()).max().unwrap_or(0) as f64;
        let min_queue = backends
            .iter()
            .map(|b| b.queue_depth())
            .min()
            .unwrap_or(0)
            .max(1) as f64;

        // Cost
        let total_gpu_seconds: f64 = backends
            .iter()
            .map(|b| b.busy_time_ms as f64 / 1000.0)
            .sum();
        let gpu_seconds_per_request = if completed > 0 {
            total_gpu_seconds / completed as f64
        } else {
            0.0
        };

        let total_cost_per_hour: f64 = backends
            .iter()
            .map(|b| b.gpu_type.estimated_cost_per_hour())
            .sum();
        let cost_per_sec = total_cost_per_hour / 3600.0;
        let total_tokens = total_prompt_tokens + total_gen_tokens;
        let estimated_cost_per_1k_tokens = if total_tokens > 0 && duration_sec > 0.0 {
            (cost_per_sec * duration_sec) / (total_tokens as f64 / 1000.0)
        } else {
            0.0
        };

        SimulationMetrics {
            algorithm: algorithm.to_string(),
            duration_ms,
            total_requests: completed + self.rejected,
            completed_requests: completed,
            rejected_requests: self.rejected,
            ttft: Percentiles::from_values(&ttft_values),
            tbt: Percentiles::from_values(&tbt_values),
            end_to_end_latency: Percentiles::from_values(&e2e_values),
            queue_wait: Percentiles::from_values(&queue_values),
            requests_per_sec: if duration_sec > 0.0 {
                completed as f64 / duration_sec
            } else {
                0.0
            },
            prompt_tokens_per_sec: if duration_sec > 0.0 {
                total_prompt_tokens as f64 / duration_sec
            } else {
                0.0
            },
            gen_tokens_per_sec: if duration_sec > 0.0 {
                total_gen_tokens as f64 / duration_sec
            } else {
                0.0
            },
            total_tokens_per_sec: if duration_sec > 0.0 {
                total_tokens as f64 / duration_sec
            } else {
                0.0
            },
            global_cache_hit_rate,
            block_cache_reuse_rate,
            per_backend_cache_stats,
            load_cv,
            jains_fairness_index: jains,
            max_min_queue_ratio: if max_queue == 0.0 {
                1.0 // All queues empty means perfectly balanced
            } else {
                max_queue / min_queue
            },
            gpu_seconds_per_request,
            estimated_cost_per_1k_tokens,
            per_backend_requests,
            custom_metrics,
        }
    }
}

/// Coefficient of variation (std / mean).
fn coefficient_of_variation(values: &[u64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<u64>() as f64 / n;
    if mean == 0.0 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    variance.sqrt() / mean
}

/// Jain's fairness index: (sum(x_i))^2 / (n * sum(x_i^2)).
fn jains_fairness_index(values: &[u64]) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    let n = values.len() as f64;
    let sum: f64 = values.iter().map(|&v| v as f64).sum();
    let sum_sq: f64 = values.iter().map(|&v| (v as f64).powi(2)).sum();
    if sum_sq == 0.0 {
        return 1.0;
    }
    (sum * sum) / (n * sum_sq)
}

/// Format metrics as a pretty-printed table string.
pub fn format_table(metrics: &SimulationMetrics) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "\n{:=<70}\n",
        format!("  {} Results  ", metrics.algorithm)
    ));
    out.push_str(&format!(
        "  Duration: {:.1}s | Requests: {} ({} rejected)\n",
        metrics.duration_ms as f64 / 1000.0,
        metrics.total_requests,
        metrics.rejected_requests
    ));
    out.push_str(&format!("{:-<70}\n", "  Latency  "));
    out.push_str(&format!(
        "  TTFT (ms)       P50={:>8.1}  P90={:>8.1}  P99={:>8.1}\n",
        metrics.ttft.p50, metrics.ttft.p90, metrics.ttft.p99
    ));
    out.push_str(&format!(
        "  TBT (ms)        P50={:>8.1}  P90={:>8.1}  P99={:>8.1}\n",
        metrics.tbt.p50, metrics.tbt.p90, metrics.tbt.p99
    ));
    out.push_str(&format!(
        "  E2E (ms)        P50={:>8.1}  P90={:>8.1}  P99={:>8.1}\n",
        metrics.end_to_end_latency.p50,
        metrics.end_to_end_latency.p90,
        metrics.end_to_end_latency.p99
    ));
    out.push_str(&format!(
        "  Queue wait (ms) P50={:>8.1}  P90={:>8.1}  P99={:>8.1}\n",
        metrics.queue_wait.p50, metrics.queue_wait.p90, metrics.queue_wait.p99
    ));
    out.push_str(&format!("{:-<70}\n", "  Throughput  "));
    out.push_str(&format!(
        "  Requests/sec: {:.1}  Tokens/sec: {:.0} (prompt: {:.0}, gen: {:.0})\n",
        metrics.requests_per_sec,
        metrics.total_tokens_per_sec,
        metrics.prompt_tokens_per_sec,
        metrics.gen_tokens_per_sec,
    ));
    out.push_str(&format!("{:-<70}\n", "  Cache  "));
    out.push_str(&format!(
        "  Global cache hit rate: {:.1}%\n",
        metrics.global_cache_hit_rate * 100.0
    ));
    if metrics.block_cache_reuse_rate > 0.0 {
        out.push_str(&format!(
            "  Block cache reuse:     {:.1}%\n",
            metrics.block_cache_reuse_rate * 100.0
        ));
    }
    out.push_str(&format!("{:-<70}\n", "  Fairness  "));
    out.push_str(&format!(
        "  Load CV: {:.3}  Jain's index: {:.4}  Max/min queue: {:.1}\n",
        metrics.load_cv, metrics.jains_fairness_index, metrics.max_min_queue_ratio,
    ));
    out.push_str(&format!("{:-<70}\n", "  Cost  "));
    out.push_str(&format!(
        "  GPU-sec/req: {:.3}  Est. $/1K tokens: {:.4}\n",
        metrics.gpu_seconds_per_request, metrics.estimated_cost_per_1k_tokens,
    ));
    out.push_str(&format!("{:=<70}\n", ""));
    out
}

/// Format a comparison table of multiple algorithm results.
pub fn format_comparison_table(results: &[SimulationMetrics]) -> String {
    if results.is_empty() {
        return String::from("No results to compare.\n");
    }

    let mut out = String::new();
    out.push_str(&format!("\n{:=<90}\n", "  Algorithm Comparison  "));
    out.push_str(&format!(
        "{:<22} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}\n",
        "Algorithm", "TTFT p50", "TTFT p99", "E2E p50", "E2E p99", "Req/s", "Cache%", "Jain's"
    ));
    out.push_str(&format!("{:-<90}\n", ""));

    for m in results {
        out.push_str(&format!(
            "{:<22} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>7.1}% {:>8.4}\n",
            m.algorithm,
            m.ttft.p50,
            m.ttft.p99,
            m.end_to_end_latency.p50,
            m.end_to_end_latency.p99,
            m.requests_per_sec,
            m.global_cache_hit_rate * 100.0,
            m.jains_fairness_index,
        ));
    }
    out.push_str(&format!("{:=<90}\n", ""));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentiles_empty() {
        let p = Percentiles::from_values(&[]);
        assert_eq!(p.p50, 0.0);
        assert_eq!(p.mean, 0.0);
    }

    #[test]
    fn test_percentiles_single() {
        let p = Percentiles::from_values(&[42.0]);
        assert_eq!(p.p50, 42.0);
        assert_eq!(p.p99, 42.0);
        assert_eq!(p.mean, 42.0);
    }

    #[test]
    fn test_percentiles_distribution() {
        let values: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let p = Percentiles::from_values(&values);
        assert!((p.p50 - 50.0).abs() < 2.0);
        assert!((p.p99 - 99.0).abs() < 2.0);
        assert_eq!(p.min, 1.0);
        assert_eq!(p.max, 100.0);
    }

    #[test]
    fn test_coefficient_of_variation() {
        assert_eq!(coefficient_of_variation(&[]), 0.0);
        assert_eq!(coefficient_of_variation(&[5, 5, 5, 5]), 0.0);
        let cv = coefficient_of_variation(&[1, 2, 3, 4, 5]);
        assert!(cv > 0.0);
    }

    #[test]
    fn test_jains_fairness_equal() {
        let j = jains_fairness_index(&[100, 100, 100, 100]);
        assert!((j - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_jains_fairness_unequal() {
        let j = jains_fairness_index(&[100, 0, 0, 0]);
        assert!(j < 0.5);
    }

    #[test]
    fn test_request_metric_avg_tbt() {
        let m = RequestMetric {
            request_id: 1,
            backend_id: 0,
            arrival_time_ms: 0,
            queue_wait_ms: 10,
            ttft_ms: 50,
            total_latency_ms: 200,
            prompt_tokens: 512,
            gen_tokens: 128,
            prefix_cache_hit: false,
            block_cache_overlap: -1.0,
            tbt_samples_ms: vec![10.0, 12.0, 11.0, 13.0],
        };
        assert!((m.avg_tbt_ms() - 11.5).abs() < 0.01);
    }

    #[test]
    fn test_max_min_queue_ratio_all_empty() {
        use crate::backend::SimulatedBackend;
        use crate::topology::{BackendRole, GpuProfile};

        let backends: Vec<SimulatedBackend> = (0..4)
            .map(|id| {
                SimulatedBackend::new(
                    id,
                    GpuProfile::H100Sxm,
                    Default::default(),
                    1024,
                    16,
                    16384,
                    256,
                    BackendRole::Both,
                )
            })
            .collect();

        // All queues are empty
        assert!(backends.iter().all(|b| b.queue_depth() == 0));

        let collector = MetricsCollector::new(0);
        let metrics = collector.aggregate("test", &backends, HashMap::new());
        assert_eq!(
            metrics.max_min_queue_ratio, 1.0,
            "All-empty queues should give ratio 1.0 (perfectly balanced), got {}",
            metrics.max_min_queue_ratio,
        );
    }

    #[test]
    fn test_format_table_no_panic() {
        let metrics = SimulationMetrics {
            algorithm: "test".to_string(),
            duration_ms: 10000,
            total_requests: 100,
            completed_requests: 95,
            rejected_requests: 5,
            ttft: Percentiles::from_values(&[10.0, 20.0, 30.0]),
            tbt: Percentiles::from_values(&[5.0, 6.0, 7.0]),
            end_to_end_latency: Percentiles::from_values(&[100.0, 200.0, 300.0]),
            queue_wait: Percentiles::from_values(&[1.0, 2.0, 3.0]),
            requests_per_sec: 9.5,
            prompt_tokens_per_sec: 5000.0,
            gen_tokens_per_sec: 1500.0,
            total_tokens_per_sec: 6500.0,
            global_cache_hit_rate: 0.45,
            block_cache_reuse_rate: 0.0,
            per_backend_cache_stats: vec![],
            load_cv: 0.1,
            jains_fairness_index: 0.98,
            max_min_queue_ratio: 1.5,
            gpu_seconds_per_request: 0.5,
            estimated_cost_per_1k_tokens: 0.01,
            per_backend_requests: vec![50, 45],
            custom_metrics: HashMap::new(),
        };
        let table = format_table(&metrics);
        assert!(table.contains("test"));
        assert!(table.contains("TTFT"));
    }
}
