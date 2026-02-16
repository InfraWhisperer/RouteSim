# Contributing to RouteSim

Thank you for your interest in contributing to RouteSim! This document provides guidelines for contributing.

## Development Setup

### Prerequisites

- Rust 1.75+ (install via [rustup](https://rustup.rs/))
- Python 3.9+ (for Python bindings and tests)
- [maturin](https://www.maturin.rs/) (`pip install maturin`)

### Building

```bash
# Build Rust crates
cargo build

# Build Python package (development mode)
maturin develop --release

# Run Rust tests
cargo test

# Run Python tests
pip install pytest click
pytest tests/python/ -v

# Run benchmarks
cargo bench -p routesim-core
```

### Code Quality

```bash
# Rust formatting
cargo fmt --all

# Rust linting
cargo clippy --workspace --exclude routesim-python -- -D warnings

# Python formatting
black python/ tests/python/

# Python linting
ruff check python/ tests/python/
```

## Adding a New Routing Algorithm

1. Create a new file in `crates/routesim-algorithms/src/your_algorithm.rs`
2. Implement the `RoutingAlgorithm` trait:

```rust
use crate::traits::*;
use std::collections::HashMap;

pub struct YourAlgorithm {
    // Internal state
}

impl YourAlgorithm {
    pub fn new() -> Self { Self {} }
}

impl RoutingAlgorithm for YourAlgorithm {
    fn route(
        &mut self,
        request: &RequestInfo,
        backends: &[BackendSnapshot],
        clock: &dyn Clock,
    ) -> RoutingDecision {
        // Your routing logic here
        let best = backends.iter()
            .filter(|b| b.state != BackendState::Offline)
            .min_by_key(|b| b.queue_depth)
            .unwrap();
        RoutingDecision::Route(best.id)
    }

    fn name(&self) -> &str { "your_algorithm" }
}
```

3. Register it in `crates/routesim-algorithms/src/lib.rs`:
   - Add `pub mod your_algorithm;`
   - Add `pub use your_algorithm::YourAlgorithm;`
   - Add the name to `algorithm_by_name()` and `available_algorithms()`

4. Add tests in the same file

5. Run the full test suite: `cargo test`

## Project Structure

```
crates/
├── routesim-core/       # Simulation engine, backend model, KV cache, metrics
├── routesim-algorithms/  # Routing algorithm trait + built-in implementations
└── routesim-python/      # PyO3 bindings
python/routesim/          # Python package (CLI, trace generators, reporting)
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure `cargo test` and `cargo clippy` pass
5. Submit a PR with a clear description

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
