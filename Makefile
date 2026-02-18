.PHONY: benchmark benchmark-charts benchmark-all benchmark-clean

benchmark:
	python benchmarks/run_all.py

benchmark-charts:
	python benchmarks/generate_charts.py

benchmark-all: benchmark benchmark-charts

benchmark-clean:
	rm -rf benchmarks/results/ benchmarks/charts/ benchmarks/RESULTS.md
