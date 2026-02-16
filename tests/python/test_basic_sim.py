"""Basic simulation tests for the Python interface."""

import pytest


def test_trace_gen_poisson():
    """Test Poisson trace generator produces reasonable output."""
    from routesim.trace_gen import poisson

    records = poisson(rate=100, duration_sec=10, seed=42)
    assert len(records) > 0
    # ~1000 requests at 100 req/s for 10s
    assert 500 < len(records) < 1500

    # Check record structure
    r = records[0]
    assert "ts" in r
    assert "prompt_tokens" in r
    assert "gen_tokens" in r
    assert r["prompt_tokens"] > 0
    assert r["gen_tokens"] > 0


def test_trace_gen_bursty():
    """Test bursty trace generator."""
    from routesim.trace_gen import bursty

    records = bursty(
        base_rate=50,
        burst_rate=200,
        burst_duration_sec=5,
        burst_interval_sec=30,
        total_duration_sec=60,
        seed=42,
    )
    assert len(records) > 0


def test_trace_gen_diurnal():
    """Test diurnal trace generator."""
    from routesim.trace_gen import diurnal

    records = diurnal(
        peak_rate=100,
        trough_rate=10,
        period_hours=1,
        duration_hours=1,
        seed=42,
    )
    assert len(records) > 0


def test_trace_gen_deterministic():
    """Test that same seed produces same trace."""
    from routesim.trace_gen import poisson

    r1 = poisson(rate=50, duration_sec=5, seed=123)
    r2 = poisson(rate=50, duration_sec=5, seed=123)
    assert len(r1) == len(r2)
    for a, b in zip(r1, r2):
        assert a == b


def test_trace_gen_write_read(tmp_path):
    """Test writing and reading a trace file."""
    from routesim.trace_gen import poisson, write_trace

    records = poisson(rate=10, duration_sec=5, seed=42)
    output = tmp_path / "test_trace.jsonl"
    write_trace(records, str(output))

    # Read back
    import json

    with open(output) as f:
        read_back = [json.loads(line) for line in f if line.strip()]
    assert len(read_back) == len(records)


def test_trace_sorted():
    """Test that generated traces are sorted by timestamp."""
    from routesim.trace_gen import poisson

    records = poisson(rate=100, duration_sec=10, seed=42)
    timestamps = [r["ts"] for r in records]
    assert timestamps == sorted(timestamps)
