#!/usr/bin/env python3
"""How to write a custom routing algorithm in Python.

This example implements a prefix-cache-aware router that breaks ties
by preferring backends with lower queue depth.
"""

import routesim


class SmartPrefixRouter(routesim.Algorithm):
    """Routes to backends that have the request's prefix cached.

    Falls back to least-loaded backend if no cache hit is possible.
    """

    def route(self, request, backends, clock_ms):
        # Sort backends by: (has prefix cached DESC, queue depth ASC)
        best = max(
            backends,
            key=lambda b: (
                request.prefix_hash in b.prefix_hashes_cached if request.prefix_hash else False,
                -b.queue_depth,
                -b.kv_cache_utilization,
            ),
        )
        return routesim.Route(best.id)

    def name(self):
        return "smart_prefix"


if __name__ == "__main__":
    # Run with custom algorithm
    result = routesim.run(
        config="configs/production_h100x8.toml",
        trace="traces/example_trace.jsonl",
        algorithm=SmartPrefixRouter(),
    )
    print(result.summary())
