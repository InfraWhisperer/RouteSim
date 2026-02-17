#!/usr/bin/env bash
# Download Mooncake production traces from Moonshot AI.
#
# These traces are from Kimi, Moonshot AI's production chatbot, published
# alongside the FAST 2025 Best Paper: "Mooncake: A KVCache-centric
# Disaggregated Architecture for LLM Serving."
#
# Source: https://github.com/kvcache-ai/Mooncake
#
# Three FAST'25 release traces:
#   - conversation_trace.jsonl  (12,031 requests) — multi-turn chat
#   - synthetic_trace.jsonl     (3,993 requests)  — synthetic workload
#   - toolagent_trace.jsonl     (23,608 requests) — tool-use agent
#
# Plus the original arXiv trace:
#   - mooncake_trace.jsonl      (23,608 requests) — original release
#
# Citation:
#   @inproceedings{mooncake2025,
#     title     = {Mooncake: A KVCache-centric Disaggregated Architecture
#                  for LLM Serving},
#     author    = {Qin, Ruoyu and others},
#     booktitle = {USENIX FAST},
#     year      = {2025}
#   }

set -euo pipefail

TRACE_DIR="$(cd "$(dirname "$0")/../traces" && pwd)"
BASE_URL="https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release"

# FAST'25 release traces
FAST25_FILES=(
    "traces/conversation_trace.jsonl"
    "traces/synthetic_trace.jsonl"
    "traces/toolagent_trace.jsonl"
)

# Original arXiv trace
ARXIV_FILES=(
    "arxiv-trace/mooncake_trace.jsonl"
)

download_file() {
    local url="$1"
    local dest="$2"
    local name
    name="$(basename "$dest")"

    if [ -f "$dest" ]; then
        echo "  [skip] ${name} already exists"
        return 0
    fi

    echo "  [download] ${name}"
    if curl -fsSL "$url" -o "$dest"; then
        local lines
        lines=$(wc -l < "$dest" | tr -d ' ')
        echo "             ${lines} records"
    else
        echo "  [warn] Failed to download ${name}. The upstream URL may have changed."
        echo "         Check https://github.com/kvcache-ai/Mooncake for the latest trace paths."
        rm -f "$dest"
        return 1
    fi
}

echo "Downloading Mooncake FAST'25 traces to ${TRACE_DIR}/ ..."
echo ""

echo "FAST'25 release traces:"
for f in "${FAST25_FILES[@]}"; do
    dest="${TRACE_DIR}/$(basename "$f")"
    download_file "${BASE_URL}/${f}" "$dest"
done

echo ""
echo "Original arXiv trace:"
for f in "${ARXIV_FILES[@]}"; do
    dest="${TRACE_DIR}/$(basename "$f")"
    download_file "${BASE_URL}/${f}" "$dest"
done

echo ""
echo "Done. To run a simulation with Mooncake traces:"
echo ""
echo "  routesim run -c configs/mooncake_demo.toml \\"
echo "              -t traces/conversation_trace.jsonl \\"
echo "              -a prefix_overlap"
echo ""
echo "  # Or compare algorithms:"
echo "  routesim compare -c configs/mooncake_demo.toml \\"
echo "                   -t traces/conversation_trace.jsonl \\"
echo "                   -A round_robin,least_outstanding,prefix_aware,prefix_overlap"
echo ""
echo "  # Analyze trace characteristics:"
echo "  python examples/mooncake_trace_stats.py traces/conversation_trace.jsonl"
