"""RouteSim — Benchmark LLM routing algorithms without GPUs.

A discrete-event simulator for LLM inference load balancing that lets you
replay real or synthetic inference traffic against pluggable routing
algorithms and simulated GPU backends, without needing actual GPUs.
"""

__version__ = "0.1.0"

# Re-export from native module when available
try:
    from routesim.routesim import (
        run,
        compare,
        list_algorithms,
        Config,
        Results,
        Route,
        BackendInfo,
    )
except ImportError:
    # Native module not built yet — provide stubs for development
    pass

from routesim.algorithm import Algorithm
from routesim.trace_gen import poisson, bursty, diurnal, replay_with_scaling
