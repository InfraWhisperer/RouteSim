//! Python bindings for RouteSim via PyO3.
//!
//! Exposes the simulation engine, routing algorithms, and configuration
//! to Python, allowing users to write custom algorithms in Python and
//! run simulations from Python scripts.

use pyo3::prelude::*;
use std::collections::HashSet;

use routesim_core::config::SimConfig;
use routesim_core::metrics::SimulationMetrics;

/// Python-accessible simulation configuration.
#[pyclass]
#[derive(Clone)]
struct Config {
    inner: SimConfig,
}

#[pymethods]
impl Config {
    /// Load configuration from a TOML file.
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let config = SimConfig::from_file(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: config })
    }

    /// Parse configuration from a TOML string.
    #[staticmethod]
    fn from_str(toml: &str) -> PyResult<Self> {
        let config = SimConfig::from_str(toml)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: config })
    }
}

/// Python-accessible simulation results.
#[pyclass]
#[derive(Clone)]
struct Results {
    inner: SimulationMetrics,
}

#[pymethods]
impl Results {
    #[getter]
    fn algorithm(&self) -> String {
        self.inner.algorithm.clone()
    }

    #[getter]
    fn duration_ms(&self) -> u64 {
        self.inner.duration_ms
    }

    #[getter]
    fn completed_requests(&self) -> u64 {
        self.inner.completed_requests
    }

    #[getter]
    fn rejected_requests(&self) -> u64 {
        self.inner.rejected_requests
    }

    #[getter]
    fn requests_per_sec(&self) -> f64 {
        self.inner.requests_per_sec
    }

    #[getter]
    fn tokens_per_sec(&self) -> f64 {
        self.inner.total_tokens_per_sec
    }

    #[getter]
    fn cache_hit_rate(&self) -> f64 {
        self.inner.global_cache_hit_rate
    }

    #[getter]
    fn jains_fairness(&self) -> f64 {
        self.inner.jains_fairness_index
    }

    #[getter]
    fn ttft_p50(&self) -> f64 {
        self.inner.ttft.p50
    }

    #[getter]
    fn ttft_p99(&self) -> f64 {
        self.inner.ttft.p99
    }

    #[getter]
    fn e2e_p50(&self) -> f64 {
        self.inner.end_to_end_latency.p50
    }

    #[getter]
    fn e2e_p99(&self) -> f64 {
        self.inner.end_to_end_latency.p99
    }

    /// Pretty-print a summary table.
    fn summary(&self) -> String {
        routesim_core::metrics::format_table(&self.inner)
    }

    /// Serialize results to JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Results(algorithm='{}', completed={}, ttft_p50={:.1}ms, req/s={:.1})",
            self.inner.algorithm,
            self.inner.completed_requests,
            self.inner.ttft.p50,
            self.inner.requests_per_sec,
        )
    }
}

/// Routing decision returned from Python algorithms.
#[pyclass]
#[derive(Clone)]
struct Route {
    #[pyo3(get)]
    backend_id: u32,
}

#[pymethods]
impl Route {
    #[new]
    fn new(backend_id: u32) -> Self {
        Self { backend_id }
    }
}

/// Backend snapshot exposed to Python routing algorithms.
#[pyclass]
#[derive(Clone)]
struct BackendInfo {
    #[pyo3(get)]
    id: u32,
    #[pyo3(get)]
    queue_depth: u32,
    #[pyo3(get)]
    active_batch_size: u32,
    #[pyo3(get)]
    active_batch_tokens: u32,
    #[pyo3(get)]
    kv_cache_utilization: f32,
    #[pyo3(get)]
    prefix_hashes_cached: HashSet<u64>,
    #[pyo3(get)]
    estimated_ttft_ms: f64,
    #[pyo3(get)]
    tokens_per_sec_current: f64,
    #[pyo3(get)]
    total_requests_served: u64,
    #[pyo3(get)]
    total_tokens_generated: u64,
}

/// Request info exposed to Python routing algorithms.
#[pyclass]
#[derive(Clone)]
struct PyRequestInfo {
    #[pyo3(get)]
    id: u64,
    #[pyo3(get)]
    prompt_tokens: u32,
    #[pyo3(get)]
    max_gen_tokens: u32,
    #[pyo3(get)]
    prefix_hash: Option<u64>,
    #[pyo3(get)]
    prefix_token_length: Option<u32>,
    #[pyo3(get)]
    conversation_id: Option<String>,
    #[pyo3(get)]
    priority: u8,
}

/// Run a simulation with a named algorithm.
#[pyfunction]
fn run(config: &str, trace: &str, algorithm: &str) -> PyResult<Results> {
    let sim_config = SimConfig::from_file(std::path::Path::new(config))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let requests = routesim_core::load_trace(std::path::Path::new(trace), &sim_config.trace.format)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let algo = routesim_algorithms::algorithm_by_name(algorithm).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("Unknown algorithm: {}", algorithm))
    })?;

    let metrics = routesim_core::run_simulation(sim_config, requests, algo);
    Ok(Results { inner: metrics })
}

/// Compare multiple algorithms on the same config and trace.
#[pyfunction]
fn compare(config: &str, trace: &str, algorithms: Vec<String>) -> PyResult<Vec<Results>> {
    let sim_config = SimConfig::from_file(std::path::Path::new(config))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let requests = routesim_core::load_trace(std::path::Path::new(trace), &sim_config.trace.format)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let algo_names: Vec<&str> = algorithms.iter().map(|s| s.as_str()).collect();
    let results = routesim_core::compare_algorithms(&sim_config, &requests, &algo_names);

    Ok(results.into_iter().map(|r| Results { inner: r }).collect())
}

/// List available built-in algorithm names.
#[pyfunction]
fn list_algorithms() -> Vec<String> {
    routesim_algorithms::available_algorithms()
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

/// Python module definition.
#[pymodule]
fn routesim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(compare, m)?)?;
    m.add_function(wrap_pyfunction!(list_algorithms, m)?)?;
    m.add_class::<Config>()?;
    m.add_class::<Results>()?;
    m.add_class::<Route>()?;
    m.add_class::<BackendInfo>()?;
    m.add_class::<PyRequestInfo>()?;
    Ok(())
}
