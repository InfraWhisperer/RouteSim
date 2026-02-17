//! RouteSim CLI — Benchmark LLM routing algorithms without GPUs.

use clap::{Parser, Subcommand};
use routesim_core::config::SimConfig;
use routesim_core::metrics;
use routesim_core::trace;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "routesim",
    about = "Benchmark LLM routing algorithms without GPUs",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a simulation with a single algorithm.
    Run {
        /// Path to TOML configuration file.
        #[arg(short, long)]
        config: PathBuf,
        /// Path to trace file.
        #[arg(short, long)]
        trace: Option<PathBuf>,
        /// Routing algorithm name.
        #[arg(short, long, default_value = "round_robin")]
        algorithm: String,
        /// Output results to JSON file.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Compare multiple algorithms on the same trace.
    Compare {
        /// Path to TOML configuration file.
        #[arg(short, long)]
        config: PathBuf,
        /// Path to trace file.
        #[arg(short, long)]
        trace: Option<PathBuf>,
        /// Comma-separated list of algorithm names.
        #[arg(short = 'A', long, value_delimiter = ',')]
        algorithms: Vec<String>,
        /// Output results to JSON file.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Generate a synthetic trace.
    GenTrace {
        /// Request rate (requests/sec).
        #[arg(long, default_value = "100")]
        rate: f64,
        /// Duration in seconds.
        #[arg(long, default_value = "300")]
        duration: u64,
        /// Mean prompt tokens.
        #[arg(long, default_value = "500")]
        prompt_mean: f64,
        /// Std dev of prompt tokens.
        #[arg(long, default_value = "200")]
        prompt_std: f64,
        /// Mean generation tokens.
        #[arg(long, default_value = "150")]
        gen_mean: f64,
        /// Std dev of generation tokens.
        #[arg(long, default_value = "50")]
        gen_std: f64,
        /// Number of distinct prefixes.
        #[arg(long, default_value = "10")]
        num_prefixes: u32,
        /// Mean prefix length in tokens.
        #[arg(long, default_value = "256")]
        prefix_len_mean: f64,
        /// Output file path.
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Convert traces between formats.
    Convert {
        /// Input trace file.
        #[arg(short, long)]
        input: PathBuf,
        /// Input format (otel, compact_jsonl, mooncake).
        #[arg(short, long)]
        format: String,
        /// Output file path.
        #[arg(short, long)]
        output: PathBuf,
        /// Tokens per KV cache block (for Mooncake format).
        #[arg(long, default_value = "16")]
        block_size: u32,
    },
    /// Sweep request rates to find saturation point.
    Sweep {
        /// Path to TOML configuration file.
        #[arg(short, long)]
        config: PathBuf,
        /// Routing algorithm name.
        #[arg(short, long, default_value = "round_robin")]
        algorithm: String,
        /// Comma-separated list of request rates.
        #[arg(long, value_delimiter = ',')]
        rates: Vec<f64>,
        /// Duration per rate point in seconds.
        #[arg(long, default_value = "60")]
        duration: u64,
        /// Output results to JSON file.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// List available algorithms.
    ListAlgorithms,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run {
            config,
            trace: trace_path,
            algorithm,
            output,
        } => {
            let sim_config = SimConfig::from_file(&config).unwrap_or_else(|e| {
                eprintln!("Error loading config: {}", e);
                std::process::exit(1);
            });

            let requests = load_requests(&sim_config, trace_path.as_deref());
            let algo = routesim_algorithms::algorithm_by_name(&algorithm).unwrap_or_else(|| {
                eprintln!(
                    "Unknown algorithm: {}. Available: {:?}",
                    algorithm,
                    routesim_algorithms::available_algorithms()
                );
                std::process::exit(1);
            });

            let result = routesim_core::run_simulation(sim_config, requests, algo);
            println!("{}", metrics::format_table(&result));

            if let Some(output_path) = output {
                let json = serde_json::to_string_pretty(&result).unwrap();
                std::fs::write(&output_path, json).unwrap_or_else(|e| {
                    eprintln!("Error writing output: {}", e);
                    std::process::exit(1);
                });
                println!("Results written to {}", output_path.display());
            }
        }
        Commands::Compare {
            config,
            trace: trace_path,
            algorithms,
            output,
        } => {
            let sim_config = SimConfig::from_file(&config).unwrap_or_else(|e| {
                eprintln!("Error loading config: {}", e);
                std::process::exit(1);
            });

            let requests = load_requests(&sim_config, trace_path.as_deref());
            let algo_names: Vec<&str> = if algorithms.is_empty() {
                routesim_algorithms::available_algorithms()
            } else {
                algorithms.iter().map(|s| s.as_str()).collect()
            };

            let results = routesim_core::compare_algorithms(&sim_config, &requests, &algo_names);
            println!("{}", metrics::format_comparison_table(&results));

            for result in &results {
                println!("{}", metrics::format_table(result));
            }

            if let Some(output_path) = output {
                let json = serde_json::to_string_pretty(&results).unwrap();
                std::fs::write(&output_path, json).unwrap_or_else(|e| {
                    eprintln!("Error writing output: {}", e);
                    std::process::exit(1);
                });
                println!("Results written to {}", output_path.display());
            }
        }
        Commands::GenTrace {
            rate,
            duration,
            prompt_mean,
            prompt_std,
            gen_mean,
            gen_std,
            num_prefixes,
            prefix_len_mean,
            output,
        } => {
            use rand::Rng;
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;

            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let mut requests = Vec::new();
            let total_requests = (rate * duration as f64) as u64;

            for i in 0..total_requests {
                let arrival_ms = (i as f64 / rate * 1000.0) as u64;
                let prompt_tokens = (prompt_mean + rng.gen::<f64>() * prompt_std * 2.0 - prompt_std)
                    .max(1.0) as u32;
                let gen_tokens =
                    (gen_mean + rng.gen::<f64>() * gen_std * 2.0 - gen_std).max(1.0) as u32;
                let prefix_idx = rng.gen_range(0..num_prefixes);
                let prefix_hash = Some(prefix_idx as u64);
                let prefix_len = Some(prefix_len_mean.max(1.0) as u32);

                requests.push(routesim_core::InferenceRequest {
                    id: i,
                    arrival_time_ms: arrival_ms,
                    prompt_tokens,
                    max_gen_tokens: gen_tokens,
                    actual_gen_tokens: gen_tokens,
                    prefix_hash,
                    prefix_token_length: prefix_len,
                    cache_block_hashes: Vec::new(),
                    conversation_id: None,
                    lora_adapter: None,
                    priority: 0,
                    metadata: std::collections::HashMap::new(),
                });
            }

            trace::write_compact_jsonl(&requests, &output).unwrap_or_else(|e| {
                eprintln!("Error writing trace: {}", e);
                std::process::exit(1);
            });
            println!(
                "Generated {} requests to {}",
                requests.len(),
                output.display()
            );
        }
        Commands::Convert {
            input,
            format,
            output,
            block_size,
        } => {
            let requests = trace::load_trace_with_block_size(&input, &format, block_size)
                .unwrap_or_else(|e| {
                    eprintln!("Error loading trace: {}", e);
                    std::process::exit(1);
                });
            trace::write_compact_jsonl(&requests, &output).unwrap_or_else(|e| {
                eprintln!("Error writing trace: {}", e);
                std::process::exit(1);
            });
            println!(
                "Converted {} requests to {}",
                requests.len(),
                output.display()
            );
        }
        Commands::Sweep {
            config,
            algorithm,
            rates,
            duration,
            output,
        } => {
            use rand::Rng;
            use rand::SeedableRng;
            use rand_chacha::ChaCha8Rng;

            let sim_config = SimConfig::from_file(&config).unwrap_or_else(|e| {
                eprintln!("Error loading config: {}", e);
                std::process::exit(1);
            });

            let mut all_results = Vec::new();
            for rate in &rates {
                let mut rng = ChaCha8Rng::seed_from_u64(42);
                let total = (rate * duration as f64) as u64;
                let requests: Vec<_> = (0..total)
                    .map(|i| {
                        let arrival_ms = (i as f64 / rate * 1000.0) as u64;
                        routesim_core::InferenceRequest {
                            id: i,
                            arrival_time_ms: arrival_ms,
                            prompt_tokens: (500.0 + rng.gen::<f64>() * 200.0) as u32,
                            max_gen_tokens: 150,
                            actual_gen_tokens: (150.0 + rng.gen::<f64>() * 50.0) as u32,
                            prefix_hash: Some(rng.gen_range(0..10u64)),
                            prefix_token_length: Some(256),
                            cache_block_hashes: Vec::new(),
                            conversation_id: None,
                            lora_adapter: None,
                            priority: 0,
                            metadata: std::collections::HashMap::new(),
                        }
                    })
                    .collect();

                let algo = routesim_algorithms::algorithm_by_name(&algorithm).unwrap();
                let result = routesim_core::run_simulation(sim_config.clone(), requests, algo);
                println!("Rate {:.0} req/s: TTFT p50={:.1}ms p99={:.1}ms | E2E p50={:.1}ms | {:.0} req/s throughput",
                    rate, result.ttft.p50, result.ttft.p99, result.end_to_end_latency.p50, result.requests_per_sec);
                all_results.push(result);
            }

            if let Some(output_path) = output {
                let json = serde_json::to_string_pretty(&all_results).unwrap();
                std::fs::write(&output_path, json).unwrap();
                println!("Sweep results written to {}", output_path.display());
            }
        }
        Commands::ListAlgorithms => {
            println!("Available routing algorithms:");
            for name in routesim_algorithms::available_algorithms() {
                println!("  - {}", name);
            }
        }
    }
}

fn load_requests(
    config: &SimConfig,
    trace_path: Option<&std::path::Path>,
) -> Vec<routesim_core::InferenceRequest> {
    let path = trace_path
        .map(PathBuf::from)
        .or_else(|| config.trace.path.as_ref().map(PathBuf::from));

    match path {
        Some(p) => trace::load_trace(&p, &config.trace.format).unwrap_or_else(|e| {
            eprintln!("Error loading trace: {}", e);
            std::process::exit(1);
        }),
        None => {
            eprintln!("No trace file specified. Use --trace or set trace.path in config.");
            std::process::exit(1);
        }
    }
}
