"""Click-based CLI wrapper for RouteSim Python interface."""

import json
import sys
from pathlib import Path

try:
    import click
except ImportError:
    print("CLI requires 'click' package: pip install click", file=sys.stderr)
    sys.exit(1)


@click.group()
@click.version_option(version="0.1.0")
def main():
    """RouteSim — Benchmark LLM routing algorithms without GPUs."""
    pass


@main.command()
@click.option("--config", "-c", required=True, help="Path to TOML config file")
@click.option("--trace", "-t", required=True, help="Path to trace file")
@click.option("--algorithm", "-a", default="round_robin", help="Algorithm name")
@click.option("--output", "-o", help="Output JSON file path")
@click.option("--report", help="Output HTML report path")
def run(config, trace, algorithm, output, report):
    """Run a simulation with a single algorithm."""
    import routesim

    result = routesim.run(config=config, trace=trace, algorithm=algorithm)
    click.echo(result.summary())

    if output:
        Path(output).write_text(result.to_json())
        click.echo(f"Results written to {output}")

    if report:
        from routesim.report import generate_html_report

        generate_html_report([result], report)
        click.echo(f"HTML report written to {report}")


@main.command()
@click.option("--config", "-c", required=True, help="Path to TOML config file")
@click.option("--trace", "-t", required=True, help="Path to trace file")
@click.option(
    "--algorithms",
    "-A",
    default="round_robin,least_outstanding,prefix_aware,session_affinity",
    help="Comma-separated algorithm names",
)
@click.option("--output", "-o", help="Output JSON file path")
@click.option("--report", help="Output HTML report path")
def compare(config, trace, algorithms, output, report):
    """Compare multiple algorithms on the same trace."""
    import routesim

    algo_list = [a.strip() for a in algorithms.split(",")]
    results = routesim.compare(config=config, trace=trace, algorithms=algo_list)

    for r in results:
        click.echo(r.summary())

    if output:
        all_json = json.dumps([json.loads(r.to_json()) for r in results], indent=2)
        Path(output).write_text(all_json)
        click.echo(f"Results written to {output}")

    if report:
        from routesim.report import generate_html_report

        generate_html_report(results, report)
        click.echo(f"HTML report written to {report}")


@main.command("gen-trace")
@click.option("--generator", default="poisson", help="Generator: poisson, bursty, diurnal")
@click.option("--rate", default=100.0, help="Request rate (req/s)")
@click.option("--duration", default=300, help="Duration in seconds")
@click.option("--prompt-mean", default=500.0, help="Mean prompt tokens")
@click.option("--gen-mean", default=150.0, help="Mean generation tokens")
@click.option("--num-prefixes", default=10, help="Number of distinct prefixes")
@click.option("--prefix-len-mean", default=256.0, help="Mean prefix length")
@click.option("--output", "-o", required=True, help="Output JSONL file path")
@click.option("--seed", default=42, help="Random seed")
def gen_trace(generator, rate, duration, prompt_mean, gen_mean, num_prefixes, prefix_len_mean, output, seed):
    """Generate a synthetic trace."""
    from routesim.trace_gen import poisson, bursty, diurnal, write_trace

    generators = {
        "poisson": lambda: poisson(
            rate=rate,
            duration_sec=duration,
            prompt_tokens_mean=prompt_mean,
            gen_tokens_mean=gen_mean,
            num_prefixes=num_prefixes,
            prefix_len_mean=prefix_len_mean,
            seed=seed,
        ),
        "bursty": lambda: bursty(
            base_rate=rate,
            burst_rate=rate * 3,
            total_duration_sec=duration,
            num_prefixes=num_prefixes,
            seed=seed,
        ),
        "diurnal": lambda: diurnal(
            peak_rate=rate,
            trough_rate=rate * 0.2,
            duration_hours=duration / 3600,
            num_prefixes=num_prefixes,
            seed=seed,
        ),
    }

    if generator not in generators:
        click.echo(f"Unknown generator: {generator}. Available: {list(generators.keys())}")
        sys.exit(1)

    records = generators[generator]()
    write_trace(records, output)
    click.echo(f"Generated {len(records)} requests to {output}")


@main.command("list-algorithms")
def list_algorithms():
    """List available routing algorithms."""
    import routesim

    click.echo("Available routing algorithms:")
    for name in routesim.list_algorithms():
        click.echo(f"  - {name}")


if __name__ == "__main__":
    main()
