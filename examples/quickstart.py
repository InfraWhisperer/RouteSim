#!/usr/bin/env python3
"""RouteSim Quickstart — 10 lines to benchmark routing algorithms."""

import routesim

# Compare four algorithms on a production cluster config
results = routesim.compare(
    config="configs/production_h100x8.toml",
    trace="traces/example_trace.jsonl",
    algorithms=["round_robin", "least_outstanding", "prefix_aware", "session_affinity"],
)

# Print summary for each algorithm
for r in results:
    print(r.summary())
