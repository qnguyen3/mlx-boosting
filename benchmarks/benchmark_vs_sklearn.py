#!/usr/bin/env python3
"""Benchmark MLX Decision Trees vs scikit-learn.

This script compares the training and prediction performance of
MLX-based decision trees against scikit-learn on various dataset sizes.

Usage:
    uv run python benchmarks/benchmark_vs_sklearn.py
    uv run python benchmarks/benchmark_vs_sklearn.py --quick
    uv run python benchmarks/benchmark_vs_sklearn.py --full
"""

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

# Import MLX boosting
from mlx_boosting.trees import DecisionTreeClassifier, DecisionTreeRegressor

# Import sklearn for comparison
try:
    from sklearn.tree import DecisionTreeClassifier as SklearnDTC
    from sklearn.tree import DecisionTreeRegressor as SklearnDTR

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. Install with: uv add scikit-learn")


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""

    name: str
    n_samples: int
    n_features: int
    max_depth: int
    sklearn_fit_ms: float
    sklearn_predict_ms: float
    mlx_fit_ms: float
    mlx_predict_ms: float

    @property
    def fit_speedup(self) -> float:
        """Speedup factor for fitting."""
        if self.mlx_fit_ms > 0:
            return self.sklearn_fit_ms / self.mlx_fit_ms
        return float("inf")

    @property
    def predict_speedup(self) -> float:
        """Speedup factor for prediction."""
        if self.mlx_predict_ms > 0:
            return self.sklearn_predict_ms / self.mlx_predict_ms
        return float("inf")


def time_function(func: Callable, warmup: int = 2, runs: int = 5) -> float:
    """Time a function with warmup and multiple runs.

    Args:
        func: Function to time.
        warmup: Number of warmup runs.
        runs: Number of timed runs.

    Returns:
        Mean execution time in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        func()
        mx.eval()  # Ensure MLX operations complete

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        mx.eval()  # Ensure MLX operations complete
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times)


def benchmark_regression(
    n_samples: int,
    n_features: int,
    max_depth: int,
    n_bins: int = 256,
    verbose: bool = True,
) -> BenchmarkResult:
    """Benchmark regression tree training and prediction.

    Args:
        n_samples: Number of training samples.
        n_features: Number of features.
        max_depth: Maximum tree depth.
        n_bins: Number of histogram bins for MLX.
        verbose: Print progress.

    Returns:
        BenchmarkResult with timing information.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"Regression: {n_samples:,} samples, {n_features} features, depth={max_depth}"
        )
        print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    y_np = np.random.randn(n_samples).astype(np.float32)

    # Convert to MLX
    X_mlx = mx.array(X_np)
    y_mlx = mx.array(y_np)
    mx.eval(X_mlx, y_mlx)

    # Generate test data
    X_test_np = np.random.randn(n_samples // 10, n_features).astype(np.float32)
    X_test_mlx = mx.array(X_test_np)
    mx.eval(X_test_mlx)

    sklearn_fit_ms = 0.0
    sklearn_predict_ms = 0.0

    # Sklearn benchmark
    if HAS_SKLEARN:
        sklearn_model = SklearnDTR(max_depth=max_depth)

        sklearn_fit_ms = time_function(lambda: sklearn_model.fit(X_np, y_np))
        sklearn_model.fit(X_np, y_np)  # Ensure fitted for prediction

        sklearn_predict_ms = time_function(lambda: sklearn_model.predict(X_test_np))

        if verbose:
            print(f"sklearn fit:     {sklearn_fit_ms:>8.2f} ms")
            print(f"sklearn predict: {sklearn_predict_ms:>8.2f} ms")

    # MLX benchmark
    mlx_model = DecisionTreeRegressor(max_depth=max_depth, n_bins=n_bins)

    mlx_fit_ms = time_function(lambda: mlx_model.fit(X_mlx, y_mlx))
    mlx_model.fit(X_mlx, y_mlx)  # Ensure fitted for prediction

    mlx_predict_ms = time_function(lambda: mlx_model.predict(X_test_mlx))

    if verbose:
        print(f"MLX fit:         {mlx_fit_ms:>8.2f} ms")
        print(f"MLX predict:     {mlx_predict_ms:>8.2f} ms")

        if HAS_SKLEARN:
            fit_speedup = (
                sklearn_fit_ms / mlx_fit_ms if mlx_fit_ms > 0 else float("inf")
            )
            pred_speedup = (
                sklearn_predict_ms / mlx_predict_ms
                if mlx_predict_ms > 0
                else float("inf")
            )
            print(f"\nFit speedup:     {fit_speedup:>8.2f}x")
            print(f"Predict speedup: {pred_speedup:>8.2f}x")

    return BenchmarkResult(
        name="regression",
        n_samples=n_samples,
        n_features=n_features,
        max_depth=max_depth,
        sklearn_fit_ms=sklearn_fit_ms,
        sklearn_predict_ms=sklearn_predict_ms,
        mlx_fit_ms=mlx_fit_ms,
        mlx_predict_ms=mlx_predict_ms,
    )


def benchmark_classification(
    n_samples: int,
    n_features: int,
    n_classes: int,
    max_depth: int,
    n_bins: int = 256,
    verbose: bool = True,
) -> BenchmarkResult:
    """Benchmark classification tree training and prediction.

    Args:
        n_samples: Number of training samples.
        n_features: Number of features.
        n_classes: Number of classes.
        max_depth: Maximum tree depth.
        n_bins: Number of histogram bins for MLX.
        verbose: Print progress.

    Returns:
        BenchmarkResult with timing information.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(
            f"Classification: {n_samples:,} samples, {n_features} features, "
            f"{n_classes} classes, depth={max_depth}"
        )
        print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    y_np = np.random.randint(0, n_classes, n_samples)

    # Convert to MLX
    X_mlx = mx.array(X_np)
    y_mlx = mx.array(y_np)
    mx.eval(X_mlx, y_mlx)

    # Generate test data
    X_test_np = np.random.randn(n_samples // 10, n_features).astype(np.float32)
    X_test_mlx = mx.array(X_test_np)
    mx.eval(X_test_mlx)

    sklearn_fit_ms = 0.0
    sklearn_predict_ms = 0.0

    # Sklearn benchmark
    if HAS_SKLEARN:
        sklearn_model = SklearnDTC(max_depth=max_depth)

        sklearn_fit_ms = time_function(lambda: sklearn_model.fit(X_np, y_np))
        sklearn_model.fit(X_np, y_np)

        sklearn_predict_ms = time_function(lambda: sklearn_model.predict(X_test_np))

        if verbose:
            print(f"sklearn fit:     {sklearn_fit_ms:>8.2f} ms")
            print(f"sklearn predict: {sklearn_predict_ms:>8.2f} ms")

    # MLX benchmark
    mlx_model = DecisionTreeClassifier(max_depth=max_depth, n_bins=n_bins)

    mlx_fit_ms = time_function(lambda: mlx_model.fit(X_mlx, y_mlx))
    mlx_model.fit(X_mlx, y_mlx)

    mlx_predict_ms = time_function(lambda: mlx_model.predict(X_test_mlx))

    if verbose:
        print(f"MLX fit:         {mlx_fit_ms:>8.2f} ms")
        print(f"MLX predict:     {mlx_predict_ms:>8.2f} ms")

        if HAS_SKLEARN:
            fit_speedup = (
                sklearn_fit_ms / mlx_fit_ms if mlx_fit_ms > 0 else float("inf")
            )
            pred_speedup = (
                sklearn_predict_ms / mlx_predict_ms
                if mlx_predict_ms > 0
                else float("inf")
            )
            print(f"\nFit speedup:     {fit_speedup:>8.2f}x")
            print(f"Predict speedup: {pred_speedup:>8.2f}x")

    return BenchmarkResult(
        name="classification",
        n_samples=n_samples,
        n_features=n_features,
        max_depth=max_depth,
        sklearn_fit_ms=sklearn_fit_ms,
        sklearn_predict_ms=sklearn_predict_ms,
        mlx_fit_ms=mlx_fit_ms,
        mlx_predict_ms=mlx_predict_ms,
    )


def run_quick_benchmark() -> list[BenchmarkResult]:
    """Run quick benchmark with small datasets."""
    print("\n" + "=" * 60)
    print("QUICK BENCHMARK")
    print("=" * 60)

    results = []

    # Small dataset
    results.append(benchmark_regression(1_000, 10, 6))
    results.append(benchmark_classification(1_000, 10, 2, 6))

    # Medium dataset
    results.append(benchmark_regression(10_000, 20, 8))
    results.append(benchmark_classification(10_000, 20, 3, 8))

    return results


def run_full_benchmark() -> list[BenchmarkResult]:
    """Run full benchmark with various dataset sizes."""
    print("\n" + "=" * 60)
    print("FULL BENCHMARK - MLX vs sklearn Decision Trees")
    print("=" * 60)

    results = []

    # Scaling with dataset size
    print("\n### Scaling with dataset size (20 features, depth=6) ###")
    for n_samples in [1_000, 10_000, 50_000, 100_000]:
        results.append(benchmark_regression(n_samples, 20, 6))

    # Scaling with features
    print("\n### Scaling with features (10000 samples, depth=6) ###")
    for n_features in [10, 50, 100, 200]:
        results.append(benchmark_regression(10_000, n_features, 6))

    # Scaling with depth
    print("\n### Scaling with tree depth (10000 samples, 20 features) ###")
    for max_depth in [4, 6, 8, 10]:
        results.append(benchmark_regression(10_000, 20, max_depth))

    # Classification benchmarks
    print("\n### Classification benchmarks ###")
    results.append(benchmark_classification(10_000, 20, 2, 6))
    results.append(benchmark_classification(10_000, 20, 5, 6))
    results.append(benchmark_classification(50_000, 20, 3, 8))

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print summary of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print(
        f"{'Task':<15} {'Samples':>10} {'Features':>10} {'Depth':>6} "
        f"{'Fit Speedup':>12} {'Pred Speedup':>12}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r.name:<15} {r.n_samples:>10,} {r.n_features:>10} {r.max_depth:>6} "
            f"{r.fit_speedup:>11.2f}x {r.predict_speedup:>11.2f}x"
        )

    # Calculate averages
    if results:
        avg_fit = np.mean([r.fit_speedup for r in results])
        avg_pred = np.mean([r.predict_speedup for r in results])
        print("-" * 80)
        print(
            f"{'AVERAGE':<15} {'-':>10} {'-':>10} {'-':>6} {avg_fit:>11.2f}x {avg_pred:>11.2f}x"
        )


def main() -> None:
    """Run benchmarks."""
    parser = argparse.ArgumentParser(
        description="Benchmark MLX vs sklearn Decision Trees"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--full", action="store_true", help="Run full benchmark")
    args = parser.parse_args()

    if not HAS_SKLEARN:
        print("\nRunning MLX-only benchmark (sklearn not installed)")
        print("Install sklearn for comparison: uv add scikit-learn")

    results = run_full_benchmark() if args.full else run_quick_benchmark()

    if HAS_SKLEARN:
        print_summary(results)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
