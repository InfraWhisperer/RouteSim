"""Tests for synthetic trace generators."""

import json
import pytest
from routesim.trace_gen import poisson, bursty, diurnal, write_trace, replay_with_scaling


def test_poisson_rate():
    """Test that Poisson generator produces approximately the right rate."""
    records = poisson(rate=100, duration_sec=100, seed=42)
    # Should be approximately 10000 requests
    assert 8000 < len(records) < 12000


def test_poisson_prefix_hashes():
    """Test that prefix hashes are distributed across N prefixes."""
    records = poisson(rate=50, duration_sec=10, num_prefixes=5, seed=42)
    prefix_hashes = set()
    for r in records:
        if "prefix_hash" in r:
            prefix_hashes.add(r["prefix_hash"])
    # Should see all 5 prefixes (with enough samples)
    assert len(prefix_hashes) == 5


def test_bursty_has_bursts():
    """Test that bursty generator has varying inter-arrival times."""
    records = bursty(
        base_rate=10,
        burst_rate=100,
        burst_duration_sec=5,
        burst_interval_sec=30,
        total_duration_sec=120,
        seed=42,
    )
    assert len(records) > 0
    # Calculate inter-arrival times
    timestamps = [r["ts"] for r in records]
    iats = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    # Should have some very short inter-arrivals (during bursts)
    assert min(iats) < 50  # Less than 50ms during burst


def test_diurnal_pattern():
    """Test that diurnal generator creates varying rates."""
    records = diurnal(
        peak_rate=100,
        trough_rate=10,
        period_hours=1,
        duration_hours=2,
        seed=42,
    )
    assert len(records) > 0


def test_replay_with_scaling(tmp_path):
    """Test trace replay with speed scaling."""
    # Create a trace
    original = [
        {"ts": 0, "prompt_tokens": 100, "gen_tokens": 50},
        {"ts": 1000, "prompt_tokens": 200, "gen_tokens": 100},
        {"ts": 2000, "prompt_tokens": 150, "gen_tokens": 75},
    ]
    trace_file = tmp_path / "test.jsonl"
    with open(trace_file, "w") as f:
        for r in original:
            f.write(json.dumps(r) + "\n")

    # Replay at 2x speed
    replayed = replay_with_scaling(str(trace_file), scale_factor=2.0)
    assert len(replayed) == 3
    assert replayed[0]["ts"] == 0
    assert replayed[1]["ts"] == 500  # 1000 / 2
    assert replayed[2]["ts"] == 1000  # 2000 / 2


def test_write_trace(tmp_path):
    """Test writing traces to JSONL."""
    records = [
        {"ts": 0, "prompt_tokens": 100, "gen_tokens": 50},
        {"ts": 100, "prompt_tokens": 200, "gen_tokens": 100},
    ]
    output = tmp_path / "out.jsonl"
    write_trace(records, str(output))

    with open(output) as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 2
    assert lines[0]["ts"] == 0
    assert lines[1]["prompt_tokens"] == 200
