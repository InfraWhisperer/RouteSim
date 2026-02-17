#!/usr/bin/env python3
"""Analyze Mooncake trace characteristics.

Standalone script (no RouteSim dependency) that reads a Mooncake JSONL trace
and prints statistics about request sizes, prefix sharing, and block overlap.

Usage:
    python examples/mooncake_trace_stats.py traces/mooncake_sample.jsonl
    python examples/mooncake_trace_stats.py traces/trace_a.jsonl --top-blocks 20
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def percentile(values, p):
    """Compute the p-th percentile of a sorted list."""
    if not values:
        return 0.0
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(values):
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


def main():
    parser = argparse.ArgumentParser(description="Analyze Mooncake trace statistics")
    parser.add_argument("trace", help="Path to Mooncake JSONL trace file")
    parser.add_argument(
        "--top-blocks",
        type=int,
        default=10,
        help="Number of top block hashes to show (default: 10)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Tokens per KV cache block (default: 16)",
    )
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        print(f"Error: trace file not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    records = []
    block_counter = Counter()
    all_input_lengths = []
    all_output_lengths = []
    all_block_counts = []
    empty_hash_count = 0
    timestamps = []

    with open(trace_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping invalid JSON at line {line_num}: {e}", file=sys.stderr)
                continue

            records.append(record)
            timestamps.append(record.get("timestamp", 0))
            all_input_lengths.append(record.get("input_length", 0))
            all_output_lengths.append(record.get("output_length", 0))

            hash_ids = record.get("hash_ids", [])
            all_block_counts.append(len(hash_ids))
            if not hash_ids:
                empty_hash_count += 1
            for h in hash_ids:
                block_counter[h] += 1

    if not records:
        print("Error: no valid records found in trace", file=sys.stderr)
        sys.exit(1)

    # Sort for percentile calculations
    all_input_lengths.sort()
    all_output_lengths.sort()
    all_block_counts.sort()
    timestamps.sort()

    n = len(records)
    duration_ms = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
    unique_blocks = len(block_counter)
    total_block_refs = sum(block_counter.values())

    print(f"Mooncake Trace Analysis: {trace_path.name}")
    print("=" * 60)

    print(f"\nRequests:           {n:,}")
    print(f"Duration:           {duration_ms:,} ms ({duration_ms / 1000:.1f} s)")
    if duration_ms > 0:
        print(f"Request rate:       {n / (duration_ms / 1000):.1f} req/s")

    print(f"\n--- Input Length (tokens) ---")
    print(f"  Mean:             {sum(all_input_lengths) / n:.0f}")
    print(f"  P50:              {percentile(all_input_lengths, 50):.0f}")
    print(f"  P90:              {percentile(all_input_lengths, 90):.0f}")
    print(f"  P99:              {percentile(all_input_lengths, 99):.0f}")
    print(f"  Min/Max:          {all_input_lengths[0]} / {all_input_lengths[-1]}")

    print(f"\n--- Output Length (tokens) ---")
    print(f"  Mean:             {sum(all_output_lengths) / n:.0f}")
    print(f"  P50:              {percentile(all_output_lengths, 50):.0f}")
    print(f"  P90:              {percentile(all_output_lengths, 90):.0f}")
    print(f"  P99:              {percentile(all_output_lengths, 99):.0f}")
    print(f"  Min/Max:          {all_output_lengths[0]} / {all_output_lengths[-1]}")

    print(f"\n--- Block-Level KV Cache Hashes ---")
    print(f"  Unique blocks:    {unique_blocks:,}")
    print(f"  Total refs:       {total_block_refs:,}")
    print(f"  Avg reuse:        {total_block_refs / unique_blocks:.1f}x" if unique_blocks else "  Avg reuse:        N/A")
    print(f"  Empty hash_ids:   {empty_hash_count} ({empty_hash_count / n * 100:.1f}%)")
    print(f"  Blocks per req:   mean={sum(all_block_counts) / n:.1f}, "
          f"p50={percentile(all_block_counts, 50):.0f}, "
          f"p99={percentile(all_block_counts, 99):.0f}")

    # Prefix sharing analysis: how many requests share blocks?
    blocks_seen_by_2plus = sum(1 for c in block_counter.values() if c >= 2)
    if unique_blocks > 0:
        sharing_ratio = blocks_seen_by_2plus / unique_blocks * 100
    else:
        sharing_ratio = 0
    print(f"  Shared blocks:    {blocks_seen_by_2plus:,} / {unique_blocks:,} ({sharing_ratio:.1f}%)")

    # Estimated prefill savings
    if total_block_refs > 0:
        cacheable_refs = total_block_refs - unique_blocks  # refs beyond first occurrence
        savings = cacheable_refs / total_block_refs * 100
        print(f"  Prefill savings:  {savings:.1f}% (blocks cacheable after first occurrence)")
        print(f"  Saved tokens:     ~{cacheable_refs * args.block_size:,} tokens")

    # Top N most frequent blocks
    print(f"\n--- Top {args.top_blocks} Most Frequent Blocks ---")
    for i, (block_hash, count) in enumerate(block_counter.most_common(args.top_blocks)):
        pct = count / n * 100
        print(f"  {i + 1:>3}. 0x{block_hash:X}  count={count:>5}  ({pct:.1f}% of requests)")


if __name__ == "__main__":
    main()
