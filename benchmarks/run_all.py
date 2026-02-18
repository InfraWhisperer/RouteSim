#!/usr/bin/env python3
"""RouteSim benchmark runner.

Runs a matrix of simulation experiments and saves raw results as JSON.
Charts are generated separately by generate_charts.py.

Usage:
    python benchmarks/run_all.py                          # Run all experiments
    python benchmarks/run_all.py --experiment scaling      # Run one experiment
    python benchmarks/run_all.py --dry-run                 # Print what would run
    python benchmarks/run_all.py --force                   # Re-run even if results exist
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "target" / "release" / "routesim"
RESULTS_DIR = ROOT / "benchmarks" / "results"
TRACE_DIR = ROOT / "traces"

# Pick the best available Mooncake trace.
MOONCAKE_TRACE = TRACE_DIR / "conversation_trace.jsonl"
if not MOONCAKE_TRACE.exists():
    for fallback in ("mooncake_trace.jsonl", "mooncake_sample.jsonl"):
        candidate = TRACE_DIR / fallback
        if candidate.exists():
            MOONCAKE_TRACE = candidate
            break

# ---------------------------------------------------------------------------
# Defaults shared across experiments
# ---------------------------------------------------------------------------
BASE_CONFIG = {
    "gpu_type": "H100Sxm",
    "num_backends": 8,
    "max_batch_tokens": 65536,
    "max_queue_depth": 256,
    "kv_cache_blocks": 32768,
    "kv_block_size": 16,
    "warmup_requests": 50,
    "prefill_tokens_per_sec": 50000,
    "decode_tokens_per_sec_batch1": 80,
    "decode_throughput_saturation_batch": 64,
    "decode_tokens_per_sec_saturated": 3200,
}

ALL_ALGORITHMS = [
    "round_robin",
    "least_outstanding",
    "least_kv",
    "prefix_aware",
    "session_affinity",
    "cost_escalation",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def file_hash(path: Path) -> str:
    """SHA-256 of a file (first 12 hex chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def write_toml(cfg: dict, path: Path) -> None:
    """Write a RouteSim TOML config from a flat dict."""
    lines = [
        "[simulation]",
        f'name = "benchmark"',
        f"seed = {cfg['seed']}",
        f"warmup_requests = {cfg.get('warmup_requests', 50)}",
        "",
        "[cluster]",
        f'gpu_type = "{cfg["gpu_type"]}"',
        f"num_backends = {cfg['num_backends']}",
        f"max_batch_tokens = {cfg['max_batch_tokens']}",
        f"max_queue_depth = {cfg['max_queue_depth']}",
        f"kv_cache_blocks = {cfg['kv_cache_blocks']}",
        f"kv_block_size = {cfg['kv_block_size']}",
        "",
        "[cluster.compute_model]",
        f"prefill_tokens_per_sec = {cfg['prefill_tokens_per_sec']}",
        f"decode_tokens_per_sec_batch1 = {cfg['decode_tokens_per_sec_batch1']}",
        f"decode_throughput_saturation_batch = {cfg['decode_throughput_saturation_batch']}",
        f"decode_tokens_per_sec_saturated = {cfg['decode_tokens_per_sec_saturated']}",
        "",
        "[trace]",
        f'format = "mooncake"',
    ]
    path.write_text("\n".join(lines) + "\n")


def run_simulation(cfg: dict, algorithm: str, trace: Path, tmp_dir: Path) -> dict:
    """Run one simulation via the CLI and return the parsed JSON metrics."""
    config_path = tmp_dir / "config.toml"
    output_path = tmp_dir / "result.json"
    write_toml(cfg, config_path)

    cmd = [
        str(BINARY),
        "run",
        "-c", str(config_path),
        "-t", str(trace),
        "-a", algorithm,
        "-o", str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        return None

    with open(output_path) as f:
        return json.load(f)


def make_record(experiment: str, algorithm: str, seed: int, cfg: dict,
                trace_path: str, metrics: dict) -> dict:
    """Build a flat result record from raw SimulationMetrics JSON."""
    return {
        "experiment": experiment,
        "algorithm": algorithm,
        "seed": seed,
        "config": {
            "num_backends": cfg["num_backends"],
            "kv_cache_blocks": cfg["kv_cache_blocks"],
            "max_batch_tokens": cfg["max_batch_tokens"],
            "kv_block_size": cfg["kv_block_size"],
        },
        "trace": trace_path,
        "metrics": {
            "ttft_p50_ms": metrics["ttft"]["p50"],
            "ttft_p75_ms": metrics["ttft"]["p75"],
            "ttft_p90_ms": metrics["ttft"]["p90"],
            "ttft_p95_ms": metrics["ttft"]["p95"],
            "ttft_p99_ms": metrics["ttft"]["p99"],
            "ttft_mean_ms": metrics["ttft"]["mean"],
            "tbt_p50_ms": metrics["tbt"]["p50"],
            "tbt_p99_ms": metrics["tbt"]["p99"],
            "e2e_p50_ms": metrics["end_to_end_latency"]["p50"],
            "e2e_p90_ms": metrics["end_to_end_latency"]["p90"],
            "e2e_p95_ms": metrics["end_to_end_latency"]["p95"],
            "e2e_p99_ms": metrics["end_to_end_latency"]["p99"],
            "e2e_mean_ms": metrics["end_to_end_latency"]["mean"],
            "queue_wait_p50_ms": metrics["queue_wait"]["p50"],
            "queue_wait_p99_ms": metrics["queue_wait"]["p99"],
            "throughput_rps": metrics["requests_per_sec"],
            "throughput_tps": metrics["total_tokens_per_sec"],
            "cache_hit_rate": metrics["global_cache_hit_rate"],
            "block_cache_reuse": metrics["block_cache_reuse_rate"],
            "fairness_jain": metrics["jains_fairness_index"],
            "load_cv": metrics["load_cv"],
            "max_min_queue_ratio": metrics["max_min_queue_ratio"],
            "gpu_sec_per_req": metrics["gpu_seconds_per_request"],
            "cost_per_1k_tokens": metrics["estimated_cost_per_1k_tokens"],
            "requests_completed": metrics["completed_requests"],
            "requests_rejected": metrics["rejected_requests"],
            "per_backend_requests": metrics.get("per_backend_requests", []),
        },
    }


def scale_mooncake_trace(src: Path, dst: Path, scale_factor: float) -> None:
    """Read a Mooncake trace and write a copy with scaled timestamps.

    scale_factor > 1.0 compresses time (higher request rate).
    scale_factor < 1.0 stretches time (lower request rate).
    """
    records = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            obj["timestamp"] = int(obj["timestamp"] / scale_factor)
            records.append(obj)
    records.sort(key=lambda r: r["timestamp"])
    with open(dst, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
def estimate_runs(experiments: list[str]) -> int:
    """Count total simulation runs for time estimation."""
    counts = {
        "algorithm_comparison": len(ALL_ALGORITHMS) * 5,
        "scaling": 4 * 3 * 3,          # 4 sizes * 3 algos * 3 seeds
        "cache_pressure": 4 * 3 * 3,   # 4 sizes * 3 algos * 3 seeds
        "rate_sweep": 5 * 2 * 3,       # 5 rates * 2 algos * 3 seeds
        "prefix_granularity": 3 * 5,   # 3 granularities * 5 seeds
    }
    return sum(counts.get(e, 0) for e in experiments)


def run_algorithm_comparison(dry_run: bool) -> list[dict]:
    """Experiment 1: Compare all algorithms with multiple seeds."""
    algorithms = ALL_ALGORITHMS
    seeds = [42, 43, 44, 45, 46]
    records = []

    for algo in algorithms:
        for seed in seeds:
            label = f"algorithm_comparison [{algo}, seed={seed}]"
            if dry_run:
                print(f"  [dry-run] {label}")
                continue
            print(f"  Running {label} ...", end=" ", flush=True)
            cfg = {**BASE_CONFIG, "seed": seed}
            t0 = time.time()
            with tempfile.TemporaryDirectory() as tmp:
                metrics = run_simulation(cfg, algo, MOONCAKE_TRACE, Path(tmp))
            if metrics is None:
                print("FAILED")
                continue
            elapsed = time.time() - t0
            rec = make_record("algorithm_comparison", algo, seed, cfg,
                              str(MOONCAKE_TRACE.relative_to(ROOT)), metrics)
            records.append(rec)
            print(f"done ({elapsed:.1f}s)")

    return records


def run_scaling(dry_run: bool) -> list[dict]:
    """Experiment 2: Vary cluster size."""
    backend_counts = [4, 8, 16, 32]
    algorithms = ["round_robin", "least_outstanding", "prefix_aware"]
    seeds = [42, 43, 44]
    records = []

    for n in backend_counts:
        for algo in algorithms:
            for seed in seeds:
                label = f"scaling [{algo}, backends={n}, seed={seed}]"
                if dry_run:
                    print(f"  [dry-run] {label}")
                    continue
                print(f"  Running {label} ...", end=" ", flush=True)
                cfg = {**BASE_CONFIG, "num_backends": n, "seed": seed}
                t0 = time.time()
                with tempfile.TemporaryDirectory() as tmp:
                    metrics = run_simulation(cfg, algo, MOONCAKE_TRACE, Path(tmp))
                if metrics is None:
                    print("FAILED")
                    continue
                elapsed = time.time() - t0
                rec = make_record("scaling", algo, seed, cfg,
                                  str(MOONCAKE_TRACE.relative_to(ROOT)), metrics)
                rec["config"]["num_backends"] = n
                records.append(rec)
                print(f"done ({elapsed:.1f}s)")

    return records


def run_cache_pressure(dry_run: bool) -> list[dict]:
    """Experiment 3: Vary KV cache capacity."""
    cache_sizes = [8192, 16384, 32768, 65536]
    algorithms = ["round_robin", "least_outstanding", "prefix_aware"]
    seeds = [42, 43, 44]
    records = []

    for blocks in cache_sizes:
        for algo in algorithms:
            for seed in seeds:
                label = f"cache_pressure [{algo}, blocks={blocks}, seed={seed}]"
                if dry_run:
                    print(f"  [dry-run] {label}")
                    continue
                print(f"  Running {label} ...", end=" ", flush=True)
                cfg = {**BASE_CONFIG, "kv_cache_blocks": blocks, "seed": seed}
                t0 = time.time()
                with tempfile.TemporaryDirectory() as tmp:
                    metrics = run_simulation(cfg, algo, MOONCAKE_TRACE, Path(tmp))
                if metrics is None:
                    print("FAILED")
                    continue
                elapsed = time.time() - t0
                rec = make_record("cache_pressure", algo, seed, cfg,
                                  str(MOONCAKE_TRACE.relative_to(ROOT)), metrics)
                rec["config"]["kv_cache_blocks"] = blocks
                records.append(rec)
                print(f"done ({elapsed:.1f}s)")

    return records


def run_rate_sweep(dry_run: bool) -> list[dict]:
    """Experiment 4: Vary request arrival rate via timestamp scaling."""
    rate_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
    algorithms = ["round_robin", "prefix_aware"]
    seeds = [42, 43, 44]
    records = []

    for mult in rate_multipliers:
        for algo in algorithms:
            for seed in seeds:
                label = f"rate_sweep [{algo}, rate={mult}x, seed={seed}]"
                if dry_run:
                    print(f"  [dry-run] {label}")
                    continue
                print(f"  Running {label} ...", end=" ", flush=True)
                cfg = {**BASE_CONFIG, "seed": seed}
                t0 = time.time()
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    # Scale the trace timestamps
                    scaled_trace = tmp_path / "scaled_trace.jsonl"
                    scale_mooncake_trace(MOONCAKE_TRACE, scaled_trace, mult)
                    metrics = run_simulation(cfg, algo, scaled_trace, tmp_path)
                if metrics is None:
                    print("FAILED")
                    continue
                elapsed = time.time() - t0
                rec = make_record("rate_sweep", algo, seed, cfg,
                                  str(MOONCAKE_TRACE.relative_to(ROOT)), metrics)
                rec["config"]["rate_multiplier"] = mult
                records.append(rec)
                print(f"done ({elapsed:.1f}s)")

    return records


def run_prefix_granularity(dry_run: bool) -> list[dict]:
    """Experiment 5: Compare prefix matching granularity.

    Maps to existing algorithms:
      - least_outstanding : no cache awareness (baseline)
      - prefix_aware      : coarse prefix matching (single hash)
      - prefix_overlap    : block-level overlap (fine-grained, Mooncake)
    """
    algorithms = [
        ("least_outstanding", "none"),
        ("prefix_aware", "coarse"),
        ("prefix_overlap", "block_level"),
    ]
    seeds = [42, 43, 44, 45, 46]
    records = []

    for algo, granularity in algorithms:
        for seed in seeds:
            label = f"prefix_granularity [{granularity} ({algo}), seed={seed}]"
            if dry_run:
                print(f"  [dry-run] {label}")
                continue
            print(f"  Running {label} ...", end=" ", flush=True)
            cfg = {**BASE_CONFIG, "seed": seed}
            t0 = time.time()
            with tempfile.TemporaryDirectory() as tmp:
                metrics = run_simulation(cfg, algo, MOONCAKE_TRACE, Path(tmp))
            if metrics is None:
                print("FAILED")
                continue
            elapsed = time.time() - t0
            rec = make_record("prefix_granularity", algo, seed, cfg,
                              str(MOONCAKE_TRACE.relative_to(ROOT)), metrics)
            rec["granularity"] = granularity
            records.append(rec)
            print(f"done ({elapsed:.1f}s)")

    return records


EXPERIMENTS = {
    "algorithm_comparison": run_algorithm_comparison,
    "scaling": run_scaling,
    "cache_pressure": run_cache_pressure,
    "rate_sweep": run_rate_sweep,
    "prefix_granularity": run_prefix_granularity,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RouteSim benchmark runner")
    parser.add_argument("--experiment", "-e", type=str, default=None,
                        choices=list(EXPERIMENTS.keys()),
                        help="Run a single experiment (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results exist")
    args = parser.parse_args()

    # Validate binary exists
    if not args.dry_run and not BINARY.exists():
        print(f"ERROR: Binary not found at {BINARY}", file=sys.stderr)
        print("Run: cargo build --release", file=sys.stderr)
        sys.exit(1)

    if not MOONCAKE_TRACE.exists():
        print(f"ERROR: Trace not found at {MOONCAKE_TRACE}", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = [args.experiment] if args.experiment else list(EXPERIMENTS.keys())
    total_runs = estimate_runs(experiments)
    # Rough estimate: ~15s per run (based on observed ~55s for 4 parallel algos)
    est_minutes = total_runs * 15 / 60
    print(f"Benchmark plan: {len(experiments)} experiment(s), ~{total_runs} simulation runs")
    if not args.dry_run:
        print(f"Estimated time: ~{est_minutes:.0f} minutes")
    print()

    trace_hash = file_hash(MOONCAKE_TRACE) if not args.dry_run else "dry-run"
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_file": str(MOONCAKE_TRACE.relative_to(ROOT)),
        "trace_hash": trace_hash,
        "binary": str(BINARY.relative_to(ROOT)),
    }

    for i, name in enumerate(experiments, 1):
        output_path = RESULTS_DIR / f"{name}.json"

        if output_path.exists() and not args.force and not args.dry_run:
            print(f"[{i}/{len(experiments)}] {name}: results exist, skipping "
                  f"(use --force to re-run)")
            continue

        print(f"[{i}/{len(experiments)}] {name}")
        runner = EXPERIMENTS[name]
        records = runner(args.dry_run)

        if args.dry_run:
            print()
            continue

        output = {"metadata": metadata, "records": records}
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  -> Saved {len(records)} records to {output_path.relative_to(ROOT)}\n")

    if not args.dry_run:
        print("Done.")


if __name__ == "__main__":
    main()
