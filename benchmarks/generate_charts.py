#!/usr/bin/env python3
"""Generate publication-quality charts from RouteSim benchmark results.

Reads JSON results from benchmarks/results/ and produces charts in
benchmarks/charts/.  Handles missing experiments gracefully — only
generates charts for data that exists.

Usage:
    python benchmarks/generate_charts.py
"""

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results"
CHARTS_DIR = ROOT / "benchmarks" / "charts"

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
ALGO_COLORS = {
    "round_robin":       "#6b7280",
    "least_outstanding": "#3b82f6",
    "least_kv":          "#06b6d4",
    "prefix_aware":      "#22c55e",
    "session_affinity":  "#f59e0b",
    "cost_escalation":   "#8b5cf6",
    "prefix_overlap":    "#ef4444",
}

ALGO_LABELS = {
    "round_robin":       "Round Robin",
    "least_outstanding": "Least Outstanding",
    "least_kv":          "Least KV",
    "prefix_aware":      "Prefix Aware",
    "session_affinity":  "Session Affinity",
    "cost_escalation":   "Cost Escalation",
    "prefix_overlap":    "Prefix Overlap",
}

GRANULARITY_LABELS = {
    "none": "No Cache Awareness",
    "coarse": "Coarse Prefix Hash",
    "block_level": "Block-Level Overlap",
}

GRANULARITY_COLORS = {
    "none": "#6b7280",
    "coarse": "#3b82f6",
    "block_level": "#22c55e",
}


def algo_color(name: str) -> str:
    return ALGO_COLORS.get(name, "#9ca3af")


def algo_label(name: str) -> str:
    return ALGO_LABELS.get(name, name)


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "#9ca3af",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_experiment(name: str) -> list[dict] | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        print(f"  WARNING: {path.name} not found, skipping related charts")
        return None
    with open(path) as f:
        data = json.load(f)
    return data["records"]


def group_by(records: list[dict], key: str) -> dict[str, list[dict]]:
    groups = {}
    for r in records:
        k = r.get(key, r.get("config", {}).get(key, ""))
        groups.setdefault(k, []).append(r)
    return groups


def metric_stats(records: list[dict], metric_key: str):
    """Return (mean, std) for a metric across records."""
    vals = [r["metrics"][metric_key] for r in records if metric_key in r["metrics"]]
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def save(fig, name: str):
    for ext in ("png", "svg"):
        fig.savefig(CHARTS_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {name}.png / .svg")


# ---------------------------------------------------------------------------
# Chart 1: Algorithm Comparison — TTFT
# ---------------------------------------------------------------------------
def chart_algorithm_comparison_ttft(records):
    by_algo = group_by(records, "algorithm")
    # Sort algorithms by P99 TTFT (ascending = best first)
    algo_order = sorted(
        by_algo.keys(),
        key=lambda a: metric_stats(by_algo[a], "ttft_p99_ms")[0],
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(algo_order))
    width = 0.25

    for i, (pkey, plabel) in enumerate([
        ("ttft_p50_ms", "P50"),
        ("ttft_p95_ms", "P95"),
        ("ttft_p99_ms", "P99"),
    ]):
        means, stds = [], []
        for algo in algo_order:
            m, s = metric_stats(by_algo[algo], pkey)
            means.append(m)
            stds.append(s)
        bars = ax.bar(x + i * width, means, width, yerr=stds,
                      label=plabel, color=[algo_color(a) for a in algo_order],
                      alpha=0.5 + i * 0.2, edgecolor="white", linewidth=0.5,
                      capsize=3)

    # Annotate improvement of best over round_robin at P99
    if "round_robin" in by_algo and len(algo_order) > 1:
        rr_p99, _ = metric_stats(by_algo["round_robin"], "ttft_p99_ms")
        best_algo = algo_order[0]
        best_p99, _ = metric_stats(by_algo[best_algo], "ttft_p99_ms")
        if rr_p99 > 0 and best_algo != "round_robin":
            pct = (rr_p99 - best_p99) / rr_p99 * 100
            ax.annotate(
                f"{pct:.0f}% lower P99\nvs Round Robin",
                xy=(0, best_p99), xytext=(0.5, best_p99 * 1.4),
                fontsize=10, ha="center", color=algo_color(best_algo),
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=algo_color(best_algo), lw=1.2),
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels([algo_label(a) for a in algo_order], rotation=20, ha="right")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to First Token by Routing Algorithm\n"
                 "(Mooncake Production Trace, 8×H100 SXM)")
    ax.legend(title="Percentile")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "algorithm_comparison_ttft")


# ---------------------------------------------------------------------------
# Chart 2: Cache Hit Rate & Throughput
# ---------------------------------------------------------------------------
def chart_algorithm_comparison_cache(records):
    by_algo = group_by(records, "algorithm")
    algo_order = sorted(by_algo.keys(), key=lambda a: algo_label(a))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(algo_order))
    width = 0.35

    cache_means, cache_stds = [], []
    rps_means, rps_stds = [], []
    for algo in algo_order:
        m, s = metric_stats(by_algo[algo], "cache_hit_rate")
        cache_means.append(m * 100)
        cache_stds.append(s * 100)
        m, s = metric_stats(by_algo[algo], "throughput_rps")
        rps_means.append(m)
        rps_stds.append(s)

    bars1 = ax1.bar(x - width / 2, cache_means, width, yerr=cache_stds,
                    label="Cache Hit Rate", capsize=3,
                    color=[algo_color(a) for a in algo_order], alpha=0.8)
    ax1.set_ylabel("Cache Hit Rate (%)")
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, rps_means, width, yerr=rps_stds,
                    label="Throughput", capsize=3,
                    color=[algo_color(a) for a in algo_order], alpha=0.4,
                    hatch="//")
    ax2.set_ylabel("Throughput (req/s)")

    ax1.set_xticks(x)
    ax1.set_xticklabels([algo_label(a) for a in algo_order], rotation=20, ha="right")
    ax1.set_title("Cache Hit Rate and Throughput by Algorithm\n"
                  "(Mooncake Production Trace, 8×H100 SXM)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    save(fig, "algorithm_comparison_cache")


# ---------------------------------------------------------------------------
# Chart 3: Radar chart
# ---------------------------------------------------------------------------
def chart_algorithm_comparison_radar(records):
    by_algo = group_by(records, "algorithm")
    algos = sorted(by_algo.keys())

    # Axes: lower-is-better metrics are inverted
    axes_defs = [
        ("ttft_p99_ms",      "TTFT P99",          True),   # invert: lower = better
        ("throughput_rps",   "Throughput",         False),
        ("cache_hit_rate",   "Cache Hit Rate",     False),
        ("fairness_jain",    "Fairness",           False),
        ("load_cv",          "Queue Stability",    True),   # invert: lower CV = better
        ("cost_per_1k_tokens", "Cost Efficiency",  True),   # invert: lower = better
    ]

    # Gather raw values
    raw = {algo: [] for algo in algos}
    for algo in algos:
        for mkey, _, _ in axes_defs:
            m, _ = metric_stats(by_algo[algo], mkey)
            raw[algo].append(m)

    # Normalize to 0–1 per axis
    n_axes = len(axes_defs)
    norm = {algo: [] for algo in algos}
    for j in range(n_axes):
        col = [raw[a][j] for a in algos]
        lo, hi = min(col), max(col)
        span = hi - lo if hi != lo else 1.0
        _, _, invert = axes_defs[j]
        for algo in algos:
            v = (raw[algo][j] - lo) / span
            norm[algo].append(1.0 - v if invert else v)

    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for algo in algos:
        values = norm[algo] + norm[algo][:1]
        ax.plot(angles, values, "o-", linewidth=2, label=algo_label(algo),
                color=algo_color(algo))
        ax.fill(angles, values, alpha=0.1, color=algo_color(algo))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d[1] for d in axes_defs], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=9, alpha=0.6)
    ax.set_title("Algorithm Performance Profile\n"
                 "(Mooncake Production Trace, 8×H100 SXM)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    save(fig, "algorithm_comparison_radar")


# ---------------------------------------------------------------------------
# Chart 4: Scaling — P99 TTFT vs cluster size
# ---------------------------------------------------------------------------
def chart_scaling_ttft(records):
    by_algo = group_by(records, "algorithm")

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo, recs in sorted(by_algo.items()):
        by_size = group_by(recs, "num_backends")
        sizes = sorted(by_size.keys(), key=int)
        means, stds = [], []
        for s in sizes:
            m, sd = metric_stats(by_size[s], "ttft_p99_ms")
            means.append(m)
            stds.append(sd)
        sizes_int = [int(s) for s in sizes]
        ax.errorbar(sizes_int, means, yerr=stds, marker="o", linewidth=2,
                    capsize=4, label=algo_label(algo), color=algo_color(algo))
        ax.fill_between(sizes_int,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.1, color=algo_color(algo))

    ax.set_xlabel("Number of Backends")
    ax.set_ylabel("P99 TTFT (ms)")
    ax.set_title("P99 TTFT vs Cluster Size\n"
                 "(Mooncake Trace, 65536 max batch tokens)")
    ax.legend()
    ax.set_xticks([4, 8, 16, 32])
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "scaling_ttft")


# ---------------------------------------------------------------------------
# Chart 5: Scaling — Cache Hit Rate
# ---------------------------------------------------------------------------
def chart_scaling_cache_hit(records):
    by_algo = group_by(records, "algorithm")

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo, recs in sorted(by_algo.items()):
        by_size = group_by(recs, "num_backends")
        sizes = sorted(by_size.keys(), key=int)
        means, stds = [], []
        for s in sizes:
            m, sd = metric_stats(by_size[s], "cache_hit_rate")
            means.append(m * 100)
            stds.append(sd * 100)
        sizes_int = [int(s) for s in sizes]
        ax.errorbar(sizes_int, means, yerr=stds, marker="o", linewidth=2,
                    capsize=4, label=algo_label(algo), color=algo_color(algo))
        ax.fill_between(sizes_int,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.1, color=algo_color(algo))

    ax.set_xlabel("Number of Backends")
    ax.set_ylabel("Cache Hit Rate (%)")
    ax.set_title("Cache Hit Rate vs Cluster Size\n"
                 "(Mooncake Trace, 65536 max batch tokens)")
    ax.legend()
    ax.set_xticks([4, 8, 16, 32])
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "scaling_cache_hit")


# ---------------------------------------------------------------------------
# Chart 6: Cache Pressure
# ---------------------------------------------------------------------------
def chart_cache_pressure(records):
    by_algo = group_by(records, "algorithm")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for algo, recs in sorted(by_algo.items()):
        by_blocks = group_by(recs, "kv_cache_blocks")
        sizes = sorted(by_blocks.keys(), key=int)
        cache_means, cache_stds = [], []
        ttft_means, ttft_stds = [], []
        for s in sizes:
            m, sd = metric_stats(by_blocks[s], "cache_hit_rate")
            cache_means.append(m * 100)
            cache_stds.append(sd * 100)
            m, sd = metric_stats(by_blocks[s], "ttft_p99_ms")
            ttft_means.append(m)
            ttft_stds.append(sd)
        sizes_int = [int(s) for s in sizes]
        ax1.errorbar(sizes_int, cache_means, yerr=cache_stds, marker="o",
                     linewidth=2, capsize=4, label=algo_label(algo),
                     color=algo_color(algo))
        ax2.errorbar(sizes_int, ttft_means, yerr=ttft_stds, marker="o",
                     linewidth=2, capsize=4, label=algo_label(algo),
                     color=algo_color(algo))

    ax1.set_xlabel("KV Cache Blocks")
    ax1.set_ylabel("Cache Hit Rate (%)")
    ax1.set_title("Cache Hit Rate vs KV Cache Capacity")
    ax1.legend()
    ax1.set_ylim(bottom=0)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax2.set_xlabel("KV Cache Blocks")
    ax2.set_ylabel("P99 TTFT (ms)")
    ax2.set_title("P99 TTFT vs KV Cache Capacity")
    ax2.legend()
    ax2.set_ylim(bottom=0)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle("Performance Sensitivity to KV Cache Capacity\n"
                 "(Mooncake Trace, 8×H100 SXM)", fontsize=16, y=1.02)
    fig.tight_layout()
    save(fig, "cache_pressure")


# ---------------------------------------------------------------------------
# Chart 7: Rate Sweep
# ---------------------------------------------------------------------------
def chart_rate_sweep(records):
    by_algo = group_by(records, "algorithm")

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo, recs in sorted(by_algo.items()):
        by_rate = group_by(recs, "rate_multiplier")
        rates = sorted(by_rate.keys(), key=float)
        means, stds = [], []
        for r in rates:
            m, sd = metric_stats(by_rate[r], "ttft_p99_ms")
            means.append(m)
            stds.append(sd)
        rates_f = [float(r) for r in rates]
        ax.errorbar(rates_f, means, yerr=stds, marker="o", linewidth=2,
                    capsize=4, label=algo_label(algo), color=algo_color(algo))
        ax.fill_between(rates_f,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.1, color=algo_color(algo))

    # Find and mark the knee point (where latency doubles from the min)
    for algo, recs in by_algo.items():
        by_rate = group_by(recs, "rate_multiplier")
        rates = sorted(by_rate.keys(), key=float)
        means = [metric_stats(by_rate[r], "ttft_p99_ms")[0] for r in rates]
        if len(means) >= 2:
            base = min(means)
            for i, m in enumerate(means):
                if m > base * 2:
                    ax.axvline(x=float(rates[i]), color=algo_color(algo),
                               linestyle="--", alpha=0.5, linewidth=1)
                    ax.annotate(f"knee ({float(rates[i]):.2f}x)",
                                xy=(float(rates[i]), m),
                                xytext=(float(rates[i]) + 0.05, m * 0.8),
                                fontsize=9, color=algo_color(algo))
                    break

    ax.set_xlabel("Request Rate Multiplier")
    ax.set_ylabel("P99 TTFT (ms)")
    ax.set_title("Latency Under Increasing Load\n"
                 "(Mooncake Trace, 8×H100 SXM)")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "rate_sweep")


# ---------------------------------------------------------------------------
# Chart 8: Prefix Granularity
# ---------------------------------------------------------------------------
def chart_prefix_granularity(records):
    gran_order = ["none", "coarse", "block_level"]
    by_gran = {}
    for r in records:
        g = r.get("granularity", "")
        by_gran.setdefault(g, []).append(r)

    # Filter to granularities that have data
    gran_order = [g for g in gran_order if g in by_gran]
    if not gran_order:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(gran_order))
    width = 0.5

    # Cache hit rate
    means, stds = [], []
    for g in gran_order:
        m, s = metric_stats(by_gran[g], "block_cache_reuse")
        means.append(m * 100)
        stds.append(s * 100)
    colors = [GRANULARITY_COLORS.get(g, "#9ca3af") for g in gran_order]
    ax1.bar(x, means, width, yerr=stds, color=colors, capsize=5, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels([GRANULARITY_LABELS.get(g, g) for g in gran_order], fontsize=10)
    ax1.set_ylabel("Block Cache Reuse (%)")
    ax1.set_title("Cache Reuse by Matching Granularity")
    ax1.set_ylim(bottom=0)

    # Annotate delta
    if len(means) >= 2:
        delta = means[-1] - means[0]
        if delta > 0:
            ax1.annotate(
                f"+{delta:.1f}pp vs baseline",
                xy=(x[-1], means[-1]),
                xytext=(x[-1], means[-1] + max(stds[-1], means[-1] * 0.1) + 2),
                fontsize=10, ha="center", fontweight="bold",
                color=GRANULARITY_COLORS.get(gran_order[-1], "#333"),
            )

    # P99 TTFT
    means2, stds2 = [], []
    for g in gran_order:
        m, s = metric_stats(by_gran[g], "ttft_p99_ms")
        means2.append(m)
        stds2.append(s)
    ax2.bar(x, means2, width, yerr=stds2, color=colors, capsize=5, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels([GRANULARITY_LABELS.get(g, g) for g in gran_order], fontsize=10)
    ax2.set_ylabel("P99 TTFT (ms)")
    ax2.set_title("Tail Latency by Matching Granularity")
    ax2.set_ylim(bottom=0)

    fig.suptitle("Impact of Prefix Matching Granularity on Cache Efficiency\n"
                 "(Mooncake Trace, 8×H100 SXM)", fontsize=16, y=1.02)
    fig.tight_layout()
    save(fig, "prefix_granularity")


# ---------------------------------------------------------------------------
# Chart 9: Fairness Heatmap
# ---------------------------------------------------------------------------
def chart_fairness_heatmap(records):
    by_algo = group_by(records, "algorithm")
    algos = sorted(by_algo.keys())

    # Get per-backend request counts (averaged across seeds)
    data = {}
    max_backends = 0
    for algo in algos:
        all_per_backend = [r["metrics"].get("per_backend_requests", [])
                           for r in by_algo[algo]]
        all_per_backend = [p for p in all_per_backend if p]
        if not all_per_backend:
            continue
        max_len = max(len(p) for p in all_per_backend)
        max_backends = max(max_backends, max_len)
        # Pad shorter arrays and average
        padded = [p + [0] * (max_len - len(p)) for p in all_per_backend]
        data[algo] = np.mean(padded, axis=0)

    if not data:
        return

    algos = [a for a in algos if a in data]
    fig, ax = plt.subplots(figsize=(10, max(4, len(algos) * 0.8 + 2)))

    matrix = np.array([data[a] for a in algos])
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(max_backends))
    ax.set_xticklabels([f"GPU {i}" for i in range(max_backends)])
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels([algo_label(a) for a in algos])

    # Annotate cells
    for i in range(len(algos)):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=9, color="white" if val > matrix.max() * 0.6 else "black")

    plt.colorbar(im, ax=ax, label="Requests Served")
    ax.set_title("Request Distribution Across Backends\n"
                 "(Mooncake Trace, 8×H100 SXM)")
    fig.tight_layout()
    save(fig, "fairness_heatmap")


# ---------------------------------------------------------------------------
# Chart 10: Summary Table
# ---------------------------------------------------------------------------
def chart_summary_table(records):
    by_algo = group_by(records, "algorithm")
    # Sort by P99 TTFT
    algo_order = sorted(
        by_algo.keys(),
        key=lambda a: metric_stats(by_algo[a], "ttft_p99_ms")[0],
    )

    columns = ["Algorithm", "P50 TTFT\n(ms)", "P99 TTFT\n(ms)",
               "Cache Hit\nRate", "Throughput\n(req/s)", "Fairness\n(Jain's)"]

    rows = []
    col_values = {c: [] for c in columns[1:]}
    for algo in algo_order:
        p50, _ = metric_stats(by_algo[algo], "ttft_p50_ms")
        p99, _ = metric_stats(by_algo[algo], "ttft_p99_ms")
        cache, _ = metric_stats(by_algo[algo], "cache_hit_rate")
        rps, _ = metric_stats(by_algo[algo], "throughput_rps")
        fair, _ = metric_stats(by_algo[algo], "fairness_jain")
        row = [algo_label(algo), f"{p50:.0f}", f"{p99:.0f}",
               f"{cache * 100:.1f}%", f"{rps:.1f}", f"{fair:.4f}"]
        rows.append(row)
        col_values[columns[1]].append(p50)
        col_values[columns[2]].append(p99)
        col_values[columns[3]].append(cache)
        col_values[columns[4]].append(rps)
        col_values[columns[5]].append(fair)

    fig, ax = plt.subplots(figsize=(12, max(3, len(rows) * 0.5 + 2)))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=columns, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor("#374151")
        cell.set_text_props(color="white", fontweight="bold")

    # Highlight best value in each column (green background)
    best_fns = {
        columns[1]: min, columns[2]: min,      # lower TTFT = better
        columns[3]: max, columns[4]: max,       # higher cache/throughput = better
        columns[5]: max,                         # higher fairness = better
    }
    for col_idx, col_name in enumerate(columns[1:], 1):
        vals = col_values[col_name]
        if not vals:
            continue
        best_val = best_fns[col_name](vals)
        for row_idx in range(len(rows)):
            if vals[row_idx] == best_val:
                table[row_idx + 1, col_idx].set_facecolor("#dcfce7")
                table[row_idx + 1, col_idx].set_text_props(fontweight="bold")

    ax.set_title("Algorithm Comparison Summary\n"
                 "(Mooncake Trace, 8×H100 SXM Cluster)", fontsize=16, pad=20)
    fig.tight_layout()
    save(fig, "summary_table")


# ---------------------------------------------------------------------------
# RESULTS.md generation
# ---------------------------------------------------------------------------
def generate_results_md(all_data: dict):
    """Generate benchmarks/RESULTS.md from available experiment data."""
    lines = ["# RouteSim Benchmark Results\n"]

    # Configuration — pull metadata from any available results file
    lines.append("## Configuration\n")
    meta = {}
    for name in ("algorithm_comparison", "scaling", "cache_pressure",
                 "rate_sweep", "prefix_granularity"):
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            if "metadata" in raw:
                meta = raw["metadata"]
                break
    lines.append(f"- **Trace**: {meta.get('trace_file', 'N/A')} "
                 f"(hash: `{meta.get('trace_hash', 'N/A')}`)")
    lines.append(f"- **Cluster**: 8x H100 SXM, 32768 KV cache blocks, block size 16")
    lines.append(f"- **Date**: {meta.get('timestamp', 'N/A')}")
    lines.append("")

    # Key Findings (auto-generated from data)
    algo_records = all_data.get("algorithm_comparison")
    if algo_records:
        lines.append("## Key Findings\n")
        by_algo = group_by(algo_records, "algorithm")

        # Finding 1: Best vs round_robin on P99 TTFT
        if "round_robin" in by_algo:
            rr_p99, _ = metric_stats(by_algo["round_robin"], "ttft_p99_ms")
            best_algo = min(by_algo.keys(),
                            key=lambda a: metric_stats(by_algo[a], "ttft_p99_ms")[0])
            best_p99, _ = metric_stats(by_algo[best_algo], "ttft_p99_ms")
            if rr_p99 > 0 and best_algo != "round_robin":
                pct = (rr_p99 - best_p99) / rr_p99 * 100
                lines.append(
                    f"1. **{algo_label(best_algo)} reduces P99 TTFT by {pct:.0f}% "
                    f"compared to Round Robin** ({best_p99:.0f}ms vs {rr_p99:.0f}ms) "
                    f"on the Mooncake production trace."
                )

        # Finding 2: Cache hit rates
        cache_rates = {a: metric_stats(recs, "cache_hit_rate")[0]
                       for a, recs in by_algo.items()}
        best_cache_algo = max(cache_rates, key=cache_rates.get)
        lines.append(
            f"2. **{algo_label(best_cache_algo)} achieves the highest cache hit rate** "
            f"at {cache_rates[best_cache_algo] * 100:.1f}%, demonstrating effective "
            f"prefix-aware routing."
        )

        # Finding 3: Block cache reuse
        block_reuse = {a: metric_stats(recs, "block_cache_reuse")[0]
                       for a, recs in by_algo.items()}
        best_block_algo = max(block_reuse, key=block_reuse.get)
        worst_block = min(block_reuse.values())
        if block_reuse[best_block_algo] > 0:
            lines.append(
                f"3. **Block-level cache reuse reaches "
                f"{block_reuse[best_block_algo] * 100:.1f}%** with "
                f"{algo_label(best_block_algo)}, compared to "
                f"{worst_block * 100:.1f}% for the worst algorithm."
            )

        # Finding 4: Fairness
        fairness = {a: metric_stats(recs, "fairness_jain")[0]
                    for a, recs in by_algo.items()}
        best_fair = max(fairness, key=fairness.get)
        lines.append(
            f"4. **{algo_label(best_fair)} achieves the best load balance** with "
            f"Jain's fairness index of {fairness[best_fair]:.4f}."
        )
        lines.append("")

    # Charts
    charts = [
        ("Algorithm Comparison", [
            ("algorithm_comparison_ttft", "TTFT by Routing Algorithm"),
            ("algorithm_comparison_cache", "Cache Hit Rate and Throughput"),
            ("algorithm_comparison_radar", "Multi-Dimensional Performance Profile"),
            ("summary_table", "Summary Table"),
            ("fairness_heatmap", "Request Distribution Across Backends"),
        ]),
        ("Scaling Analysis", [
            ("scaling_ttft", "P99 TTFT vs Cluster Size"),
            ("scaling_cache_hit", "Cache Hit Rate vs Cluster Size"),
        ]),
        ("Cache Pressure Sensitivity", [
            ("cache_pressure", "Performance vs KV Cache Capacity"),
        ]),
        ("Load Sensitivity", [
            ("rate_sweep", "Latency Under Increasing Load"),
        ]),
        ("Prefix Matching Granularity", [
            ("prefix_granularity", "Impact of Matching Granularity"),
        ]),
    ]

    for section, chart_list in charts:
        lines.append(f"## {section}\n")
        for chart_name, caption in chart_list:
            chart_path = CHARTS_DIR / f"{chart_name}.png"
            if chart_path.exists():
                lines.append(f"### {caption}\n")
                lines.append(f"![{caption}](charts/{chart_name}.png)\n")
        lines.append("")

    # Summary table in markdown
    if algo_records:
        lines.append("## Summary Table\n")
        by_algo = group_by(algo_records, "algorithm")
        algo_order = sorted(
            by_algo.keys(),
            key=lambda a: metric_stats(by_algo[a], "ttft_p99_ms")[0],
        )
        lines.append("| Algorithm | P50 TTFT (ms) | P99 TTFT (ms) | Cache Hit Rate | "
                      "Throughput (req/s) | Fairness |")
        lines.append("|-----------|--------------|--------------|---------------|"
                      "-------------------|----------|")
        for algo in algo_order:
            p50, _ = metric_stats(by_algo[algo], "ttft_p50_ms")
            p99, _ = metric_stats(by_algo[algo], "ttft_p99_ms")
            cache, _ = metric_stats(by_algo[algo], "cache_hit_rate")
            rps, _ = metric_stats(by_algo[algo], "throughput_rps")
            fair, _ = metric_stats(by_algo[algo], "fairness_jain")
            lines.append(
                f"| {algo_label(algo)} | {p50:.0f} | {p99:.0f} | "
                f"{cache * 100:.1f}% | {rps:.1f} | {fair:.4f} |"
            )
        lines.append("")

    md_path = ROOT / "benchmarks" / "RESULTS.md"
    md_path.write_text("\n".join(lines))
    print(f"  -> Generated {md_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    setup_style()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating charts from benchmark results...\n")

    # Load all available experiment data
    all_raw = {}
    for name in ("algorithm_comparison", "scaling", "cache_pressure",
                 "rate_sweep", "prefix_granularity"):
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                all_raw[name] = json.load(f)
        else:
            all_raw[name] = None

    # --- Algorithm comparison charts ---
    records = load_experiment("algorithm_comparison")
    if records:
        print("  [algorithm_comparison]")
        chart_algorithm_comparison_ttft(records)
        chart_algorithm_comparison_cache(records)
        chart_algorithm_comparison_radar(records)
        chart_summary_table(records)
        chart_fairness_heatmap(records)

    # --- Scaling charts ---
    records = load_experiment("scaling")
    if records:
        print("  [scaling]")
        chart_scaling_ttft(records)
        chart_scaling_cache_hit(records)

    # --- Cache pressure chart ---
    records = load_experiment("cache_pressure")
    if records:
        print("  [cache_pressure]")
        chart_cache_pressure(records)

    # --- Rate sweep chart ---
    records = load_experiment("rate_sweep")
    if records:
        print("  [rate_sweep]")
        chart_rate_sweep(records)

    # --- Prefix granularity chart ---
    records = load_experiment("prefix_granularity")
    if records:
        print("  [prefix_granularity]")
        chart_prefix_granularity(records)

    # --- Generate RESULTS.md ---
    print("\n  Generating RESULTS.md...")
    md_data = {}
    for name, raw in all_raw.items():
        md_data[name] = raw.get("records") if raw else None
    generate_results_md(md_data)

    print("\nDone.")


if __name__ == "__main__":
    main()
