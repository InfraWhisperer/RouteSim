#!/usr/bin/env python3
"""Replay a production trace and analyze the results.

This example shows how to load a real trace, run it through the simulator,
and analyze the detailed per-request metrics.
"""

import json

import routesim

# Run simulation with prefix-aware routing
result = routesim.run(
    config="configs/production_h100x8.toml",
    trace="traces/example_trace.jsonl",
    algorithm="prefix_aware",
)

# Print summary
print(result.summary())

# Access individual metrics
print(f"Algorithm: {result.algorithm}")
print(f"Completed: {result.completed_requests} requests")
print(f"Rejected:  {result.rejected_requests} requests")
print(f"Duration:  {result.duration_ms / 1000:.1f}s")
print(f"TTFT P50:  {result.ttft_p50:.1f}ms")
print(f"TTFT P99:  {result.ttft_p99:.1f}ms")
print(f"E2E P50:   {result.e2e_p50:.1f}ms")
print(f"E2E P99:   {result.e2e_p99:.1f}ms")
print(f"Throughput: {result.requests_per_sec:.1f} req/s")
print(f"Cache hit:  {result.cache_hit_rate * 100:.1f}%")
print(f"Fairness:   {result.jains_fairness:.4f}")
