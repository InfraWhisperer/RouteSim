"""Tests for custom Python algorithm base class."""

from routesim.algorithm import Algorithm


class DummyRouter(Algorithm):
    """Minimal algorithm for testing."""

    def route(self, request, backends, clock_ms):
        return backends[0].id if backends else None

    def name(self):
        return "dummy"


def test_algorithm_base_class():
    """Test that custom algorithms can be instantiated."""
    algo = DummyRouter()
    assert algo.name() == "dummy"
    assert algo.custom_metrics() == {}


def test_algorithm_on_event_default():
    """Test default on_event is a no-op."""
    algo = DummyRouter()
    algo.on_event(None, [])  # Should not raise


def test_algorithm_observes_events_default():
    """Test default observes_events returns False."""
    algo = DummyRouter()
    assert algo.observes_events() is False
