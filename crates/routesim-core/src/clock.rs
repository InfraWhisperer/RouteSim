//! Virtual clock for discrete-event simulation.
//!
//! The [`SimClock`] tracks simulation time independently of wall-clock time,
//! advancing only when events are processed. This enables deterministic,
//! repeatable simulations regardless of host machine speed.

use serde::{Deserialize, Serialize};

/// Virtual simulation clock.
///
/// Time is tracked in microseconds internally for precision, but most APIs
/// expose milliseconds for convenience (matching typical LLM latency scales).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimClock {
    /// Current simulation time in microseconds.
    current_us: u64,
    /// Wall-clock start time for optional real-time correlation.
    start_wall_ns: Option<u64>,
}

impl SimClock {
    /// Create a new clock starting at time zero.
    pub fn new() -> Self {
        Self {
            current_us: 0,
            start_wall_ns: None,
        }
    }

    /// Create a clock starting at a specific time in milliseconds.
    pub fn starting_at_ms(ms: u64) -> Self {
        Self {
            current_us: ms * 1000,
            start_wall_ns: None,
        }
    }

    /// Current time in milliseconds.
    pub fn now_ms(&self) -> u64 {
        self.current_us / 1000
    }

    /// Current time in microseconds.
    pub fn now_us(&self) -> u64 {
        self.current_us
    }

    /// Advance the clock to a specific time in milliseconds.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `ms` is in the past.
    pub fn advance_to_ms(&mut self, ms: u64) {
        let target_us = ms * 1000;
        debug_assert!(
            target_us >= self.current_us,
            "Cannot move clock backwards: current={}us, target={}us",
            self.current_us,
            target_us,
        );
        self.current_us = target_us;
    }

    /// Advance the clock to a specific time in microseconds.
    pub fn advance_to_us(&mut self, us: u64) {
        debug_assert!(
            us >= self.current_us,
            "Cannot move clock backwards: current={}us, target={}us",
            self.current_us,
            us,
        );
        self.current_us = us;
    }

    /// Advance the clock by a duration in milliseconds.
    pub fn advance_by_ms(&mut self, delta_ms: u64) {
        self.current_us += delta_ms * 1000;
    }

    /// Advance the clock by a duration in microseconds.
    pub fn advance_by_us(&mut self, delta_us: u64) {
        self.current_us += delta_us;
    }

    /// Elapsed time since clock creation, in milliseconds.
    pub fn elapsed_ms(&self) -> u64 {
        self.now_ms()
    }
}

impl Default for SimClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_clock_starts_at_zero() {
        let clock = SimClock::new();
        assert_eq!(clock.now_ms(), 0);
        assert_eq!(clock.now_us(), 0);
    }

    #[test]
    fn test_starting_at_ms() {
        let clock = SimClock::starting_at_ms(1000);
        assert_eq!(clock.now_ms(), 1000);
    }

    #[test]
    fn test_advance_to_ms() {
        let mut clock = SimClock::new();
        clock.advance_to_ms(500);
        assert_eq!(clock.now_ms(), 500);
    }

    #[test]
    fn test_advance_by_ms() {
        let mut clock = SimClock::new();
        clock.advance_by_ms(100);
        clock.advance_by_ms(200);
        assert_eq!(clock.now_ms(), 300);
    }

    #[test]
    fn test_microsecond_precision() {
        let mut clock = SimClock::new();
        clock.advance_by_us(1500);
        assert_eq!(clock.now_us(), 1500);
        assert_eq!(clock.now_ms(), 1); // truncation
    }

    #[test]
    #[should_panic(expected = "Cannot move clock backwards")]
    fn test_cannot_go_backwards() {
        let mut clock = SimClock::new();
        clock.advance_to_ms(100);
        clock.advance_to_ms(50);
    }
}
