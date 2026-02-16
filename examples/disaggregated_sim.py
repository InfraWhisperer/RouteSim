#!/usr/bin/env python3
"""Simulate a prefill/decode disaggregated topology.

Compares unified serving vs disaggregated prefill/decode split to
measure the impact on TTFT and throughput.
"""

import routesim

# Unified topology (all backends do both prefill + decode)
unified = routesim.run(
    config="configs/production_h100x8.toml",
    trace="traces/example_trace.jsonl",
    algorithm="prefix_aware",
)

# Disaggregated topology (2 prefill + 6 decode)
disagg = routesim.run(
    config="configs/disaggregated_pd.toml",
    trace="traces/example_trace.jsonl",
    algorithm="prefix_aware",
)

print("=" * 70)
print("Unified vs Disaggregated Comparison")
print("=" * 70)
print(f"{'Metric':<25} {'Unified':>15} {'Disaggregated':>15} {'Delta':>10}")
print("-" * 70)

metrics = [
    ("TTFT P50 (ms)", unified.ttft_p50, disagg.ttft_p50),
    ("TTFT P99 (ms)", unified.ttft_p99, disagg.ttft_p99),
    ("E2E P50 (ms)", unified.e2e_p50, disagg.e2e_p50),
    ("E2E P99 (ms)", unified.e2e_p99, disagg.e2e_p99),
    ("Req/s", unified.requests_per_sec, disagg.requests_per_sec),
    ("Cache Hit %", unified.cache_hit_rate * 100, disagg.cache_hit_rate * 100),
]

for name, u, d in metrics:
    delta = d - u
    sign = "+" if delta >= 0 else ""
    print(f"{name:<25} {u:>15.1f} {d:>15.1f} {sign}{delta:>9.1f}")
print("=" * 70)
