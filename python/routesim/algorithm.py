"""Base class for custom Python routing algorithms.

Subclass `Algorithm` and implement the `route` method to create
custom load balancing strategies.

Example::

    import routesim

    class MyRouter(routesim.Algorithm):
        def route(self, request, backends, clock_ms):
            # Pick the backend with most prefix cache hits
            best = min(backends, key=lambda b: b.queue_depth)
            return routesim.Route(best.id)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Algorithm(ABC):
    """Base class for custom routing algorithms.

    Override `route()` to implement your routing logic.
    Optionally override `on_event()` to track state changes.
    """

    @abstractmethod
    def route(self, request: Any, backends: List[Any], clock_ms: int) -> Any:
        """Route an incoming request to a backend.

        Args:
            request: RequestInfo with fields like prompt_tokens, prefix_hash, etc.
            backends: List of BackendInfo snapshots with queue_depth, kv_cache_utilization, etc.
            clock_ms: Current simulation time in milliseconds.

        Returns:
            A Route(backend_id) object indicating which backend to send the request to.
        """
        ...

    def on_event(self, event: Any, backends: List[Any]) -> None:
        """Called after each simulation event for state tracking.

        Override this if your algorithm needs to track events like
        request completions or prefill completions.

        Args:
            event: The simulation event that just occurred.
            backends: Current backend snapshots.
        """
        pass

    def name(self) -> str:
        """Human-readable algorithm name for reports."""
        return self.__class__.__name__

    def custom_metrics(self) -> Dict[str, float]:
        """Return algorithm-specific metrics for inclusion in results."""
        return {}
