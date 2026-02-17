#!/usr/bin/env python3
"""Compare routing algorithms on Mooncake production traces.

Demonstrates how prefix_overlap routing exploits block-level KV cache
sharing in Mooncake traces from Moonshot AI's Kimi chatbot.

Usage:
    python examples/mooncake_comparison.py
    python examples/mooncake_comparison.py --config configs/mooncake_demo.toml
    python examples/mooncake_comparison.py --trace traces/trace_a.jsonl
"""

import argparse
import json
from pathlib import Path

import routesim


def main():
    parser = argparse.ArgumentParser(description="Compare routing on Mooncake traces")
    parser.add_argument(
        "--config",
        default="configs/mooncake_demo.toml",
        help="Path to config TOML (default: configs/mooncake_demo.toml)",
    )
    parser.add_argument(
        "--trace",
        default="traces/mooncake_sample.jsonl",
        help="Path to Mooncake trace JSONL (default: traces/mooncake_sample.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON results (optional)",
    )
    args = parser.parse_args()

    # Algorithms to compare: general-purpose vs prefix-aware
    algorithms = [
        "round_robin",
        "least_outstanding",
        "prefix_aware",
        "prefix_overlap",
    ]

    print(f"Config: {args.config}")
    print(f"Trace:  {args.trace}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print()

    results = routesim.compare(
        config=args.config,
        trace=args.trace,
        algorithms=algorithms,
    )

    # Print comparison table
    print("=" * 100)
    print(
        f"{'Algorithm':<20} {'TTFT p50':>9} {'TTFT p99':>9} {'E2E p50':>9} "
        f"{'E2E p99':>9} {'Req/s':>8} {'Tok/s':>9} {'Cache%':>8} {'Jains':>8}"
    )
    print("-" * 100)
    for r in results:
        print(
            f"{r.algorithm:<20} {r.ttft_p50:>8.1f}ms {r.ttft_p99:>8.1f}ms "
            f"{r.e2e_p50:>8.1f}ms {r.e2e_p99:>8.1f}ms "
            f"{r.requests_per_sec:>8.1f} {r.tokens_per_sec:>9.1f} "
            f"{r.cache_hit_rate * 100:>7.1f}% {r.jains_fairness:>8.4f}"
        )
    print("=" * 100)

    # Highlight prefix_overlap vs baseline
    rr = next((r for r in results if r.algorithm == "round_robin"), None)
    po = next((r for r in results if r.algorithm == "prefix_overlap"), None)
    if rr and po:
        print()
        print("prefix_overlap vs round_robin:")
        ttft_improvement = (1 - po.ttft_p50 / rr.ttft_p50) * 100 if rr.ttft_p50 > 0 else 0
        cache_improvement = (po.cache_hit_rate - rr.cache_hit_rate) * 100
        print(f"  TTFT p50:    {ttft_improvement:+.1f}%")
        print(f"  Cache rate:  {cache_improvement:+.1f} pp")
        print(f"  Throughput:  {po.tokens_per_sec:.0f} vs {rr.tokens_per_sec:.0f} tok/s")

    # Save JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_json = [json.loads(r.to_json()) for r in results]
        output_path.write_text(json.dumps(all_json, indent=2))
        print(f"\nJSON results saved to {output_path}")


if __name__ == "__main__":
    main()
