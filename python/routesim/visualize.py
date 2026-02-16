"""Visualization helpers for RouteSim results.

Optional dependency on matplotlib/plotly for generating charts.
"""

import json
from typing import List, Any, Optional


def plot_latency_comparison(
    results: List[Any],
    metric: str = "ttft",
    output_path: Optional[str] = None,
) -> Any:
    """Plot latency percentiles across algorithms.

    Args:
        results: List of Results objects or JSON dicts.
        metric: "ttft", "tbt", or "end_to_end_latency".
        output_path: If provided, save the plot to this file.

    Returns:
        matplotlib Figure if matplotlib is available, None otherwise.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting: pip install matplotlib")
        return None

    data = _to_dicts(results)
    algorithms = [d["algorithm"] for d in data]
    percentiles = ["p50", "p75", "p90", "p95", "p99"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(algorithms))
    width = 0.15

    for i, p in enumerate(percentiles):
        values = [d[metric][p] for d in data]
        offset = (i - len(percentiles) / 2) * width
        ax.bar([xi + offset for xi in x], values, width, label=p.upper())

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"{metric.upper()} Latency Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45)
    ax.legend()
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)

    return fig


def plot_throughput_comparison(
    results: List[Any],
    output_path: Optional[str] = None,
) -> Any:
    """Plot throughput comparison across algorithms.

    Args:
        results: List of Results objects or JSON dicts.
        output_path: If provided, save the plot to this file.

    Returns:
        matplotlib Figure if available.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting: pip install matplotlib")
        return None

    data = _to_dicts(results)
    algorithms = [d["algorithm"] for d in data]
    req_per_sec = [d["requests_per_sec"] for d in data]
    tok_per_sec = [d["total_tokens_per_sec"] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(algorithms, req_per_sec, color="#2196F3")
    ax1.set_ylabel("Requests/sec")
    ax1.set_title("Request Throughput")
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(algorithms, tok_per_sec, color="#FF9800")
    ax2.set_ylabel("Tokens/sec")
    ax2.set_title("Token Throughput")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)

    return fig


def _to_dicts(results: List[Any]) -> List[dict]:
    """Convert Results objects to dicts."""
    out = []
    for r in results:
        if isinstance(r, dict):
            out.append(r)
        elif hasattr(r, "to_json"):
            out.append(json.loads(r.to_json()))
        else:
            out.append(r)
    return out
