"""Synthetic trace generators for RouteSim.

Generate realistic LLM inference traffic patterns without needing real traces.
"""

import json
import math
import random
from pathlib import Path
from typing import Optional, List, Dict, Any


def poisson(
    rate: float,
    duration_sec: float,
    prompt_tokens_mean: float = 500,
    prompt_tokens_std: float = 200,
    gen_tokens_mean: float = 150,
    gen_tokens_std: float = 50,
    num_prefixes: int = 10,
    prefix_len_mean: float = 256,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate a Poisson-arrival trace.

    Args:
        rate: Average requests per second.
        duration_sec: Total trace duration in seconds.
        prompt_tokens_mean: Mean prompt token count.
        prompt_tokens_std: Standard deviation of prompt tokens.
        gen_tokens_mean: Mean generation token count.
        gen_tokens_std: Standard deviation of generation tokens.
        num_prefixes: Number of distinct prefix hashes.
        prefix_len_mean: Mean prefix length in tokens.
        seed: Random seed for reproducibility.

    Returns:
        List of trace records (dicts).
    """
    rng = random.Random(seed)
    records = []
    t_ms = 0.0

    while t_ms < duration_sec * 1000:
        # Poisson inter-arrival time
        inter_arrival = -math.log(1 - rng.random()) / rate * 1000  # ms
        t_ms += inter_arrival

        if t_ms >= duration_sec * 1000:
            break

        prompt = max(1, int(rng.gauss(prompt_tokens_mean, prompt_tokens_std)))
        gen = max(1, int(rng.gauss(gen_tokens_mean, gen_tokens_std)))
        prefix_idx = rng.randint(0, num_prefixes - 1) if num_prefixes > 0 else None
        prefix_len = max(1, int(rng.gauss(prefix_len_mean, prefix_len_mean * 0.2)))

        record = {
            "ts": int(t_ms),
            "prompt_tokens": prompt,
            "gen_tokens": gen,
        }
        if prefix_idx is not None:
            record["prefix_hash"] = f"prefix_{prefix_idx}"
            record["prefix_len"] = min(prefix_len, prompt)

        records.append(record)

    return records


def bursty(
    base_rate: float,
    burst_rate: float,
    burst_duration_sec: float = 10,
    burst_interval_sec: float = 60,
    total_duration_sec: float = 300,
    prompt_tokens_mean: float = 500,
    gen_tokens_mean: float = 150,
    num_prefixes: int = 10,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate a bursty traffic trace with periodic high-rate bursts.

    Args:
        base_rate: Normal request rate (req/s).
        burst_rate: Request rate during bursts (req/s).
        burst_duration_sec: Duration of each burst.
        burst_interval_sec: Time between burst starts.
        total_duration_sec: Total trace duration.
        prompt_tokens_mean: Mean prompt tokens.
        gen_tokens_mean: Mean generation tokens.
        num_prefixes: Number of distinct prefixes.
        seed: Random seed.

    Returns:
        List of trace records.
    """
    rng = random.Random(seed)
    records = []
    t_ms = 0.0

    while t_ms < total_duration_sec * 1000:
        # Determine current rate
        cycle_pos = (t_ms / 1000) % burst_interval_sec
        rate = burst_rate if cycle_pos < burst_duration_sec else base_rate

        inter_arrival = -math.log(1 - rng.random()) / rate * 1000
        t_ms += inter_arrival

        if t_ms >= total_duration_sec * 1000:
            break

        prompt = max(1, int(rng.gauss(prompt_tokens_mean, prompt_tokens_mean * 0.3)))
        gen = max(1, int(rng.gauss(gen_tokens_mean, gen_tokens_mean * 0.3)))
        prefix_idx = rng.randint(0, num_prefixes - 1) if num_prefixes > 0 else None

        record = {
            "ts": int(t_ms),
            "prompt_tokens": prompt,
            "gen_tokens": gen,
        }
        if prefix_idx is not None:
            record["prefix_hash"] = f"prefix_{prefix_idx}"
            record["prefix_len"] = min(256, prompt)

        records.append(record)

    return records


def diurnal(
    peak_rate: float,
    trough_rate: float,
    period_hours: float = 24,
    duration_hours: float = 24,
    prompt_tokens_mean: float = 500,
    gen_tokens_mean: float = 150,
    num_prefixes: int = 10,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate a diurnal (day/night) traffic pattern.

    Args:
        peak_rate: Maximum request rate (req/s).
        trough_rate: Minimum request rate (req/s).
        period_hours: Period of the diurnal cycle.
        duration_hours: Total trace duration in hours.
        prompt_tokens_mean: Mean prompt tokens.
        gen_tokens_mean: Mean generation tokens.
        num_prefixes: Number of distinct prefixes.
        seed: Random seed.

    Returns:
        List of trace records.
    """
    rng = random.Random(seed)
    records = []
    t_ms = 0.0
    duration_ms = duration_hours * 3600 * 1000
    period_ms = period_hours * 3600 * 1000

    while t_ms < duration_ms:
        # Sinusoidal rate variation
        phase = (t_ms / period_ms) * 2 * math.pi
        rate = trough_rate + (peak_rate - trough_rate) * (1 + math.sin(phase)) / 2
        rate = max(rate, 0.1)

        inter_arrival = -math.log(1 - rng.random()) / rate * 1000
        t_ms += inter_arrival

        if t_ms >= duration_ms:
            break

        prompt = max(1, int(rng.gauss(prompt_tokens_mean, prompt_tokens_mean * 0.3)))
        gen = max(1, int(rng.gauss(gen_tokens_mean, gen_tokens_mean * 0.3)))
        prefix_idx = rng.randint(0, num_prefixes - 1) if num_prefixes > 0 else None

        record = {
            "ts": int(t_ms),
            "prompt_tokens": prompt,
            "gen_tokens": gen,
        }
        if prefix_idx is not None:
            record["prefix_hash"] = f"prefix_{prefix_idx}"
            record["prefix_len"] = min(256, prompt)

        records.append(record)

    return records


def replay_with_scaling(
    trace_path: str,
    scale_factor: float = 1.0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Replay a real trace at a different speed.

    Args:
        trace_path: Path to a compact JSONL trace file.
        scale_factor: Speed multiplier (2.0 = 2x faster, 0.5 = half speed).
        seed: Random seed (unused for replay, included for API consistency).

    Returns:
        List of trace records with adjusted timestamps.
    """
    records = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            record = json.loads(line)
            record["ts"] = int(record["ts"] / scale_factor)
            records.append(record)

    records.sort(key=lambda r: r["ts"])
    return records


def write_trace(records: List[Dict[str, Any]], output_path: str) -> None:
    """Write trace records to a compact JSONL file.

    Args:
        records: List of trace record dicts.
        output_path: Path to write the JSONL file.
    """
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
