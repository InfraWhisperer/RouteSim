#!/usr/bin/env python3
"""Side-by-side comparison of all built-in routing algorithms.

Generates both terminal output and an HTML report.
"""

import json
from pathlib import Path

import routesim

# Compare all built-in algorithms
all_algos = routesim.list_algorithms()
print(f"Comparing {len(all_algos)} algorithms: {', '.join(all_algos)}")

results = routesim.compare(
    config="configs/production_h100x8.toml",
    trace="traces/example_trace.jsonl",
    algorithms=all_algos,
)

# Print comparison table
print("\n" + "=" * 90)
print(f"{'Algorithm':<22} {'TTFT p50':>8} {'TTFT p99':>8} {'E2E p50':>8} "
      f"{'E2E p99':>8} {'Req/s':>8} {'Cache%':>8} {'Jains':>8}")
print("-" * 90)
for r in results:
    print(f"{r.algorithm:<22} {r.ttft_p50:>8.1f} {r.ttft_p99:>8.1f} "
          f"{r.e2e_p50:>8.1f} {r.e2e_p99:>8.1f} "
          f"{r.requests_per_sec:>8.1f} {r.cache_hit_rate*100:>7.1f}% "
          f"{r.jains_fairness:>8.4f}")
print("=" * 90)

# Save JSON results
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
all_json = [json.loads(r.to_json()) for r in results]
(output_dir / "comparison.json").write_text(json.dumps(all_json, indent=2))
print(f"\nJSON results saved to {output_dir / 'comparison.json'}")

# Generate HTML report
try:
    from routesim.report import generate_html_report
    generate_html_report(results, str(output_dir / "report.html"))
    print(f"HTML report saved to {output_dir / 'report.html'}")
except ImportError:
    print("Install plotly for HTML reports: pip install plotly")
