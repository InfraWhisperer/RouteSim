# RouteSim Usage Guide

RouteSim has two interfaces: a Rust CLI (compiled binary) and a Python API (requires building with `maturin`). This guide covers both.

## Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- For Python bindings: Python 3.9+ and [maturin](https://www.maturin.rs/)

Build the Rust binary:

```bash
cargo build --release
```

The binary will be at `target/release/routesim`.

## Core Concepts

RouteSim takes three inputs:

1. **Config** (TOML) -- describes your GPU cluster (how many backends, GPU type, KV cache size, compute speeds)
2. **Trace** (JSONL) -- the workload to simulate (a sequence of inference requests with arrival times, token counts, and optional prefix info)
3. **Algorithm** -- the routing strategy to benchmark

It runs a discrete-event simulation and outputs latency percentiles, throughput, cache hit rates, fairness metrics, and cost estimates.

## Trace Format

Each line in a JSONL trace is one inference request:

```json
{"ts": 0, "prompt_tokens": 512, "gen_tokens": 128, "prefix_hash": "system_prompt_v1", "prefix_len": 256}
{"ts": 5, "prompt_tokens": 256, "gen_tokens": 64, "prefix_hash": "system_prompt_v1", "prefix_len": 256}
```

| Field | Description |
|-------|-------------|
| `ts` | Arrival time in milliseconds |
| `prompt_tokens` | Number of input tokens |
| `gen_tokens` | Number of tokens to generate |
| `prefix_hash` | Shared prefix identifier (for KV cache reuse). Optional. |
| `prefix_len` | Length of shared prefix in tokens. Optional. |

If your requests share system prompts (e.g., all chat requests use the same system prompt), give them the same `prefix_hash` and `prefix_len`. This lets the simulator model KV cache prefix reuse, which is how prefix-aware routing algorithms gain their advantage.

## Configuration

Configs are TOML files. Here's what the fields mean:

```toml
[simulation]
name = "my-experiment"        # Label for output
seed = 42                     # RNG seed for reproducibility
warmup_requests = 100         # Exclude first N requests from metrics (transient behavior)

[cluster]
gpu_type = "H100Sxm"          # GPU model: H100Sxm, A100Sxm80, L40S
num_backends = 8               # Number of GPU workers
max_batch_tokens = 16384       # Max tokens in a single batch
max_queue_depth = 256          # Max pending requests per backend
kv_cache_blocks = 32768        # Total KV cache blocks per backend
kv_block_size = 16             # Tokens per KV cache block

[cluster.compute_model]
prefill_tokens_per_sec = 50000         # Prefill throughput
decode_tokens_per_sec_batch1 = 80      # Decode throughput at batch size 1
decode_throughput_saturation_batch = 64 # Batch size where decode throughput plateaus
decode_tokens_per_sec_saturated = 3200  # Peak decode throughput

[trace]
format = "compact_jsonl"
path = "traces/my_trace.jsonl"  # Optional: embed trace path in config
```

The included configs are:

| Config | Description |
|--------|-------------|
| `configs/production_h100x8.toml` | 8x H100 production cluster |
| `configs/default.toml` | 4x H100 small cluster |
| `configs/disaggregated_pd.toml` | 2 prefill + 6 decode (disaggregated) |
| `configs/multi_model.toml` | 6x A100 |

### Warmup

The `warmup_requests` parameter excludes the first N completed requests from metrics. This avoids polluting results with cold-start behavior (empty caches, empty queues). Set it to 0 if you want to include everything, or to something like 100 for steady-state analysis.

**Gotcha**: If your trace has fewer requests than `warmup_requests`, all metrics will be zero. The example trace only has 20 requests, so using it with `warmup_requests = 100` will produce empty results. Either use a larger trace or lower the warmup.

## CLI Reference

### List algorithms

```bash
routesim list-algorithms
```

### Run a single simulation

```bash
routesim run -c configs/production_h100x8.toml -t traces/my_trace.jsonl -a prefix_aware
```

If the trace path is set in the config's `[trace]` section, you can omit `-t`:

```bash
routesim run -c configs/production_h100x8.toml -a round_robin
```

Save results as JSON:

```bash
routesim run -c configs/production_h100x8.toml -a prefix_aware -o results.json
```

### Compare algorithms

```bash
routesim compare -c configs/production_h100x8.toml \
    -t traces/my_trace.jsonl \
    -A round_robin,least_outstanding,prefix_aware,session_affinity
```

This runs the same trace through each algorithm and prints a comparison table followed by detailed per-algorithm breakdowns.

If you omit `-A`, it runs all available algorithms.

### Generate synthetic traces

```bash
routesim gen-trace --rate 200 --duration 60 --num-prefixes 5 -o traces/synthetic.jsonl
```

| Option | Description |
|--------|-------------|
| `--rate` | Requests per second |
| `--duration` | Duration in seconds |
| `--prompt-mean` / `--prompt-std` | Prompt token distribution |
| `--gen-mean` / `--gen-std` | Generation token distribution |
| `--num-prefixes` | Number of distinct prefix groups |
| `--prefix-len-mean` | Mean prefix length in tokens |

### Rate sweep

Find your cluster's saturation point:

```bash
routesim sweep -c configs/production_h100x8.toml -a prefix_aware \
    --rates 50,100,200,400,800 --duration 30
```

This generates a synthetic trace at each rate, runs the simulation, and prints throughput/latency at each point. You'll see throughput climb linearly until it saturates.

### Convert traces

```bash
routesim convert -i traces/otel_export.json -f otel -o traces/converted.jsonl
```

## Understanding the Output

Here's what a typical result looks like:

```
  prefix_aware Results  ==============================================
  Duration: 13.9s | Requests: 1900 (0 rejected)
  Latency  -----------------------------------------------------------
  TTFT (ms)       P50=   185.0  P90=  1316.0  P99=  1994.0
  TBT (ms)        P50=    20.0  P90=    20.0  P99=    20.0
  E2E (ms)        P50=  3414.0  P90=  4478.0  P99=  5564.0
  Queue wait (ms) P50=   180.0  P90=  1311.0  P99=  1991.0
  Throughput  --------------------------------------------------------
  Requests/sec: 137.1  Tokens/sec: 88966 (prompt: 68322, gen: 20644)
  Cache  -------------------------------------------------------------
  Global cache hit rate: 100.0%
  Fairness  ----------------------------------------------------------
  Load CV: 1.150  Jain's index: 0.4305  Max/min queue: 0.0
  Cost  --------------------------------------------------------------
  GPU-sec/req: 3.131  Est. $/1K tokens: 0.0001
======================================================================
```

### Metrics explained

**Latency**
- **TTFT** (Time To First Token): How long from request arrival until the first token is generated. Dominated by queue wait + prefill time.
- **TBT** (Time Between Tokens): Inter-token latency during decode. Depends on batch size and GPU throughput.
- **E2E** (End-to-End): Total time from arrival to completion.
- **Queue wait**: Time spent waiting in the backend queue before processing starts.

**Throughput**
- Requests/sec, tokens/sec broken down by prompt (prefill) vs. generation (decode).

**Cache**
- Global cache hit rate: Fraction of requests that found their prefix already in KV cache.

**Fairness**
- **Load CV** (Coefficient of Variation): Standard deviation / mean of per-backend request counts. 0 = perfectly balanced.
- **Jain's fairness index**: Ranges from 1/n (maximally unfair) to 1.0 (perfectly fair). Generally you want > 0.9.
- **Max/min queue ratio**: Ratio of longest to shortest queue at simulation end.

**Cost**
- **GPU-sec/req**: Total GPU busy time divided by completed requests.
- **Est. $/1K tokens**: Rough cost estimate based on GPU type pricing.

### What to look for when comparing algorithms

The comparison table makes trade-offs visible:

- **prefix_aware** tends to get the best cache hit rate and lowest tail latency, but sacrifices fairness (it concentrates requests on backends with cached prefixes).
- **round_robin** gives perfect fairness but ignores cache locality.
- **cost_escalation** often sits in the middle -- good latency with reasonable fairness.
- **session_affinity** is best for multi-turn conversations where you want requests from the same conversation to hit the same backend.

## Python API

After building with `maturin develop --release`:

```python
import routesim

# Single run
result = routesim.run(
    config="configs/production_h100x8.toml",
    trace="traces/my_trace.jsonl",
    algorithm="prefix_aware",
)
print(result.summary())

# Compare algorithms
results = routesim.compare(
    config="configs/production_h100x8.toml",
    trace="traces/my_trace.jsonl",
    algorithms=["round_robin", "least_outstanding", "prefix_aware"],
)
for r in results:
    print(r.summary())

# List algorithms
print(routesim.list_algorithms())
```

The Python CLI wrapper (requires `click`):

```bash
pip install click
python -m routesim.cli run -c configs/production_h100x8.toml -t traces/my_trace.jsonl -a prefix_aware
python -m routesim.cli compare -c configs/production_h100x8.toml -t traces/my_trace.jsonl
```

## Typical Workflow

1. **Pick or write a config** matching your target cluster (GPU count, type, KV cache size).
2. **Get a trace** -- use the example, generate one with `gen-trace`, or convert production logs to JSONL.
3. **Compare algorithms** to find which routing strategy gives the best latency/throughput for your workload.
4. **Sweep rates** to find where your cluster saturates under each algorithm.
5. **Export results** as JSON for further analysis or visualization.

## Tips

- Start with a small trace (1000-2000 requests) while iterating on configs. The simulator is fast but large traces with many backends can take a few seconds.
- Use `warmup_requests = 0` when debugging or working with small traces.
- The `seed` parameter makes runs reproducible. Change it to get variance estimates across different random orderings.
- If you're benchmarking for production, use a trace that matches your real traffic pattern (arrival rate, prompt size distribution, prefix sharing ratio).
