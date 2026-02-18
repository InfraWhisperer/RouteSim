//! Python bindings for RouteSim via PyO3.
//!
//! Exposes the simulation engine, routing algorithms, and configuration
//! to Python, allowing users to write custom algorithms in Python and
//! run simulations from Python scripts.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

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
    fn block_cache_reuse_rate(&self) -> f64 {
        self.inner.block_cache_reuse_rate
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

    // --- TTFT additional percentiles ---
    #[getter]
    fn ttft_p75(&self) -> f64 { self.inner.ttft.p75 }
    #[getter]
    fn ttft_p90(&self) -> f64 { self.inner.ttft.p90 }
    #[getter]
    fn ttft_p95(&self) -> f64 { self.inner.ttft.p95 }
    #[getter]
    fn ttft_min(&self) -> f64 { self.inner.ttft.min }
    #[getter]
    fn ttft_max(&self) -> f64 { self.inner.ttft.max }
    #[getter]
    fn ttft_mean(&self) -> f64 { self.inner.ttft.mean }

    // --- TBT percentiles ---
    #[getter]
    fn tbt_p50(&self) -> f64 { self.inner.tbt.p50 }
    #[getter]
    fn tbt_p75(&self) -> f64 { self.inner.tbt.p75 }
    #[getter]
    fn tbt_p90(&self) -> f64 { self.inner.tbt.p90 }
    #[getter]
    fn tbt_p95(&self) -> f64 { self.inner.tbt.p95 }
    #[getter]
    fn tbt_p99(&self) -> f64 { self.inner.tbt.p99 }
    #[getter]
    fn tbt_min(&self) -> f64 { self.inner.tbt.min }
    #[getter]
    fn tbt_max(&self) -> f64 { self.inner.tbt.max }
    #[getter]
    fn tbt_mean(&self) -> f64 { self.inner.tbt.mean }

    // --- E2E additional percentiles ---
    #[getter]
    fn e2e_p75(&self) -> f64 { self.inner.end_to_end_latency.p75 }
    #[getter]
    fn e2e_p90(&self) -> f64 { self.inner.end_to_end_latency.p90 }
    #[getter]
    fn e2e_p95(&self) -> f64 { self.inner.end_to_end_latency.p95 }
    #[getter]
    fn e2e_min(&self) -> f64 { self.inner.end_to_end_latency.min }
    #[getter]
    fn e2e_max(&self) -> f64 { self.inner.end_to_end_latency.max }
    #[getter]
    fn e2e_mean(&self) -> f64 { self.inner.end_to_end_latency.mean }

    // --- Queue wait percentiles ---
    #[getter]
    fn queue_wait_p50(&self) -> f64 { self.inner.queue_wait.p50 }
    #[getter]
    fn queue_wait_p75(&self) -> f64 { self.inner.queue_wait.p75 }
    #[getter]
    fn queue_wait_p90(&self) -> f64 { self.inner.queue_wait.p90 }
    #[getter]
    fn queue_wait_p95(&self) -> f64 { self.inner.queue_wait.p95 }
    #[getter]
    fn queue_wait_p99(&self) -> f64 { self.inner.queue_wait.p99 }
    #[getter]
    fn queue_wait_min(&self) -> f64 { self.inner.queue_wait.min }
    #[getter]
    fn queue_wait_max(&self) -> f64 { self.inner.queue_wait.max }
    #[getter]
    fn queue_wait_mean(&self) -> f64 { self.inner.queue_wait.mean }

    // --- Throughput ---
    #[getter]
    fn prompt_tokens_per_sec(&self) -> f64 { self.inner.prompt_tokens_per_sec }
    #[getter]
    fn gen_tokens_per_sec(&self) -> f64 { self.inner.gen_tokens_per_sec }

    // --- Fairness ---
    #[getter]
    fn load_cv(&self) -> f64 { self.inner.load_cv }
    #[getter]
    fn max_min_queue_ratio(&self) -> f64 { self.inner.max_min_queue_ratio }

    // --- Cost ---
    #[getter]
    fn gpu_seconds_per_request(&self) -> f64 { self.inner.gpu_seconds_per_request }
    #[getter]
    fn estimated_cost_per_1k_tokens(&self) -> f64 { self.inner.estimated_cost_per_1k_tokens }

    // --- Custom metrics ---
    #[getter]
    fn custom_metrics(&self) -> HashMap<String, f64> { self.inner.custom_metrics.clone() }

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
    cached_block_hashes: HashSet<u64>,
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
    cache_block_hashes: Vec<u64>,
    #[pyo3(get)]
    conversation_id: Option<String>,
    #[pyo3(get)]
    priority: u8,
}

// ---------------------------------------------------------------------------
// PyAlgorithmWrapper — bridges a Python Algorithm object into the Rust trait.
// ---------------------------------------------------------------------------

/// Convert a `routesim_algorithms::RequestInfo` to a `PyRequestInfo` pyclass.
fn request_info_to_py(req: &routesim_algorithms::RequestInfo) -> PyRequestInfo {
    PyRequestInfo {
        id: req.id,
        prompt_tokens: req.prompt_tokens,
        max_gen_tokens: req.max_gen_tokens,
        prefix_hash: req.prefix_hash,
        prefix_token_length: req.prefix_token_length,
        cache_block_hashes: req.cache_block_hashes.clone(),
        conversation_id: req.conversation_id.clone(),
        priority: req.priority,
    }
}

/// Convert a `routesim_algorithms::BackendSnapshot` to a `BackendInfo` pyclass.
fn backend_snapshot_to_py(snap: &routesim_algorithms::BackendSnapshot) -> BackendInfo {
    BackendInfo {
        id: snap.id,
        queue_depth: snap.queue_depth,
        active_batch_size: snap.active_batch_size,
        active_batch_tokens: snap.active_batch_tokens,
        kv_cache_utilization: snap.kv_cache_utilization,
        prefix_hashes_cached: snap.prefix_hashes_cached.clone(),
        cached_block_hashes: snap.cached_block_hashes.clone(),
        estimated_ttft_ms: snap.estimated_ttft_ms,
        tokens_per_sec_current: snap.tokens_per_sec_current,
        total_requests_served: snap.total_requests_served,
        total_tokens_generated: snap.total_tokens_generated,
    }
}

/// Wraps a Python `Algorithm` object so it can be used as a Rust `RoutingAlgorithm`.
///
/// The `Mutex` satisfies the `Sync` bound required by the trait. It is never
/// contended because the simulation engine is single-threaded; the Mutex exists
/// purely to satisfy the type system.
struct PyAlgorithmWrapper {
    py_obj: Mutex<Py<PyAny>>,
    cached_name: String,
    observes: bool,
}

impl routesim_algorithms::RoutingAlgorithm for PyAlgorithmWrapper {
    fn route(
        &mut self,
        request: &routesim_algorithms::RequestInfo,
        backends: &[routesim_algorithms::BackendSnapshot],
        clock: &dyn routesim_algorithms::Clock,
    ) -> routesim_algorithms::RoutingDecision {
        let py_obj = self.py_obj.lock().unwrap();

        Python::with_gil(|py| {
            let py_request = Py::new(py, request_info_to_py(request));
            let py_backends: PyResult<Vec<Py<BackendInfo>>> = backends
                .iter()
                .map(|b| Py::new(py, backend_snapshot_to_py(b)))
                .collect();
            let clock_ms = clock.now_ms();

            let (py_request, py_backends) = match (py_request, py_backends) {
                (Ok(r), Ok(b)) => (r, b),
                _ => return routesim_algorithms::RoutingDecision::Reject,
            };

            match py_obj.call_method1(py, "route", (py_request, py_backends, clock_ms)) {
                Ok(result) => {
                    // Try extracting backend_id from a Route object
                    if let Ok(bid) = result.getattr(py, "backend_id").and_then(|a| a.extract::<u32>(py)) {
                        routesim_algorithms::RoutingDecision::Route(bid)
                    // Also accept a plain integer
                    } else if let Ok(bid) = result.extract::<u32>(py) {
                        routesim_algorithms::RoutingDecision::Route(bid)
                    } else {
                        eprintln!(
                            "Python route() returned unexpected type; expected Route or int, rejecting request"
                        );
                        routesim_algorithms::RoutingDecision::Reject
                    }
                }
                Err(e) => {
                    eprintln!("Python route() raised exception: {e}");
                    routesim_algorithms::RoutingDecision::Reject
                }
            }
        })
    }

    fn on_event(
        &mut self,
        event: &routesim_algorithms::SimEventInfo,
        backends: &[routesim_algorithms::BackendSnapshot],
    ) {
        if !self.observes {
            return;
        }

        let py_obj = self.py_obj.lock().unwrap();

        Python::with_gil(|py| {
            let py_event = PyDict::new_bound(py);
            match event {
                routesim_algorithms::SimEventInfo::RequestArrival { request_id } => {
                    let _ = py_event.set_item("type", "request_arrival");
                    let _ = py_event.set_item("request_id", *request_id);
                }
                routesim_algorithms::SimEventInfo::PrefillComplete {
                    backend_id,
                    request_id,
                } => {
                    let _ = py_event.set_item("type", "prefill_complete");
                    let _ = py_event.set_item("backend_id", *backend_id);
                    let _ = py_event.set_item("request_id", *request_id);
                }
                routesim_algorithms::SimEventInfo::RequestComplete {
                    backend_id,
                    request_id,
                } => {
                    let _ = py_event.set_item("type", "request_complete");
                    let _ = py_event.set_item("backend_id", *backend_id);
                    let _ = py_event.set_item("request_id", *request_id);
                }
                routesim_algorithms::SimEventInfo::TokenGenerated {
                    backend_id,
                    request_id,
                    token_num,
                } => {
                    let _ = py_event.set_item("type", "token_generated");
                    let _ = py_event.set_item("backend_id", *backend_id);
                    let _ = py_event.set_item("request_id", *request_id);
                    let _ = py_event.set_item("token_num", *token_num);
                }
            }

            let py_backends: PyResult<Vec<Py<BackendInfo>>> = backends
                .iter()
                .map(|b| Py::new(py, backend_snapshot_to_py(b)))
                .collect();

            if let Ok(py_backends) = py_backends {
                if let Err(e) =
                    py_obj.call_method1(py, "on_event", (py_event, py_backends))
                {
                    eprintln!("Python on_event() raised exception: {e}");
                }
            }
        });
    }

    fn observes_events(&self) -> bool {
        self.observes
    }

    fn name(&self) -> &str {
        &self.cached_name
    }

    fn custom_metrics(&self) -> HashMap<String, f64> {
        let py_obj = self.py_obj.lock().unwrap();

        Python::with_gil(|py| {
            py_obj
                .call_method0(py, "custom_metrics")
                .ok()
                .and_then(|obj| obj.extract::<HashMap<String, f64>>(py).ok())
                .unwrap_or_default()
        })
    }
}

/// Run a simulation with a named algorithm or a Python Algorithm object.
#[pyfunction]
fn run(py: Python<'_>, config: &str, trace: &str, algorithm: &Bound<'_, PyAny>) -> PyResult<Results> {
    let sim_config = SimConfig::from_file(std::path::Path::new(config))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let requests = routesim_core::load_trace(std::path::Path::new(trace), &sim_config.trace.format)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let algo: Box<dyn routesim_algorithms::RoutingAlgorithm> =
        if let Ok(name) = algorithm.extract::<String>() {
            // String argument: look up built-in algorithm by name
            routesim_algorithms::algorithm_by_name(&name).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!("Unknown algorithm: {name}"))
            })?
        } else {
            // Python Algorithm object: wrap in PyAlgorithmWrapper
            let py_obj: Py<PyAny> = algorithm.clone().unbind();

            let cached_name = py_obj
                .call_method0(py, "name")
                .and_then(|n| n.extract::<String>(py))
                .unwrap_or_else(|_| "python_algorithm".to_string());

            let observes = py_obj
                .call_method0(py, "observes_events")
                .and_then(|o| o.extract::<bool>(py))
                .unwrap_or(false);

            Box::new(PyAlgorithmWrapper {
                py_obj: Mutex::new(py_obj),
                cached_name,
                observes,
            })
        };

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
