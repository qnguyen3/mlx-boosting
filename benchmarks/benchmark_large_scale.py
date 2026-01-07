#!/usr/bin/env python3
"""Benchmark MLX Decision Trees on large-scale synthetic data.

Tests the hypothesis that MLX becomes faster than sklearn at larger scales.

Usage:
    uv run python benchmarks/benchmark_large_scale.py
"""

import time

import mlx.core as mx
import numpy as np

from mlx_boosting.trees import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from sklearn.tree import DecisionTreeClassifier as SklearnDTC
    from sklearn.tree import DecisionTreeRegressor as SklearnDTR

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("sklearn required for comparison")


def benchmark_classification_scaling() -> None:
    """Benchmark classification with increasing sample sizes."""
    if not HAS_SKLEARN:
        return

    print("=" * 80)
    print("LARGE-SCALE CLASSIFICATION BENCHMARK")
    print("=" * 80)
    print()

    n_features = 100
    n_classes = 5
    max_depth = 8
    n_bins = 64

    print(
        f"Config: {n_features} features, {n_classes} classes, depth={max_depth}, bins={n_bins}"
    )
    print()
    print(
        f"{'Samples':>12} {'sklearn(ms)':>12} {'MLX(ms)':>12} {'Speedup':>10} {'Winner':>10}"
    )
    print("-" * 80)

    for n_samples in [10_000, 50_000, 100_000, 200_000, 500_000]:
        # Generate data
        np.random.seed(42)
        X_np = np.random.randn(n_samples, n_features).astype(np.float32)
        y_np = np.random.randint(0, n_classes, n_samples)

        X_mlx = mx.array(X_np)
        y_mlx = mx.array(y_np)
        mx.eval(X_mlx, y_mlx)

        # sklearn
        sklearn_model = SklearnDTC(max_depth=max_depth)
        start = time.perf_counter()
        sklearn_model.fit(X_np, y_np)
        sklearn_time = (time.perf_counter() - start) * 1000

        # MLX
        mlx_model = DecisionTreeClassifier(max_depth=max_depth, n_bins=n_bins)
        start = time.perf_counter()
        mlx_model.fit(X_mlx, y_mlx)
        mx.eval()
        mlx_time = (time.perf_counter() - start) * 1000

        speedup = sklearn_time / mlx_time
        winner = "MLX" if speedup > 1 else "sklearn"

        print(
            f"{n_samples:>12,} {sklearn_time:>12.1f} {mlx_time:>12.1f} {speedup:>10.2f}x {winner:>10}"
        )

        # Clear memory
        del X_np, y_np, X_mlx, y_mlx
        mx.metal.clear_cache()


def benchmark_regression_scaling() -> None:
    """Benchmark regression with increasing sample sizes."""
    if not HAS_SKLEARN:
        return

    print()
    print("=" * 80)
    print("LARGE-SCALE REGRESSION BENCHMARK")
    print("=" * 80)
    print()

    n_features = 100
    max_depth = 8
    n_bins = 64

    print(f"Config: {n_features} features, depth={max_depth}, bins={n_bins}")
    print()
    print(
        f"{'Samples':>12} {'sklearn(ms)':>12} {'MLX(ms)':>12} {'Speedup':>10} {'Winner':>10}"
    )
    print("-" * 80)

    for n_samples in [10_000, 50_000, 100_000, 200_000, 500_000]:
        # Generate data
        np.random.seed(42)
        X_np = np.random.randn(n_samples, n_features).astype(np.float32)
        y_np = np.random.randn(n_samples).astype(np.float32)

        X_mlx = mx.array(X_np)
        y_mlx = mx.array(y_np)
        mx.eval(X_mlx, y_mlx)

        # sklearn
        sklearn_model = SklearnDTR(max_depth=max_depth)
        start = time.perf_counter()
        sklearn_model.fit(X_np, y_np)
        sklearn_time = (time.perf_counter() - start) * 1000

        # MLX
        mlx_model = DecisionTreeRegressor(max_depth=max_depth, n_bins=n_bins)
        start = time.perf_counter()
        mlx_model.fit(X_mlx, y_mlx)
        mx.eval()
        mlx_time = (time.perf_counter() - start) * 1000

        speedup = sklearn_time / mlx_time
        winner = "MLX" if speedup > 1 else "sklearn"

        print(
            f"{n_samples:>12,} {sklearn_time:>12.1f} {mlx_time:>12.1f} {speedup:>10.2f}x {winner:>10}"
        )

        # Clear memory
        del X_np, y_np, X_mlx, y_mlx
        mx.metal.clear_cache()


def benchmark_high_dimensional() -> None:
    """Benchmark with high-dimensional data (many features)."""
    if not HAS_SKLEARN:
        return

    print()
    print("=" * 80)
    print("HIGH-DIMENSIONAL BENCHMARK (many features)")
    print("=" * 80)
    print()

    n_samples = 50_000
    max_depth = 6
    n_bins = 64

    print(f"Config: {n_samples:,} samples, depth={max_depth}, bins={n_bins}")
    print()
    print(
        f"{'Features':>12} {'sklearn(ms)':>12} {'MLX(ms)':>12} {'Speedup':>10} {'Winner':>10}"
    )
    print("-" * 80)

    for n_features in [50, 100, 200, 500, 1000]:
        # Generate data
        np.random.seed(42)
        X_np = np.random.randn(n_samples, n_features).astype(np.float32)
        y_np = np.random.randn(n_samples).astype(np.float32)

        X_mlx = mx.array(X_np)
        y_mlx = mx.array(y_np)
        mx.eval(X_mlx, y_mlx)

        # sklearn
        sklearn_model = SklearnDTR(max_depth=max_depth)
        start = time.perf_counter()
        sklearn_model.fit(X_np, y_np)
        sklearn_time = (time.perf_counter() - start) * 1000

        # MLX
        mlx_model = DecisionTreeRegressor(max_depth=max_depth, n_bins=n_bins)
        start = time.perf_counter()
        mlx_model.fit(X_mlx, y_mlx)
        mx.eval()
        mlx_time = (time.perf_counter() - start) * 1000

        speedup = sklearn_time / mlx_time
        winner = "MLX" if speedup > 1 else "sklearn"

        print(
            f"{n_features:>12,} {sklearn_time:>12.1f} {mlx_time:>12.1f} {speedup:>10.2f}x {winner:>10}"
        )

        # Clear memory
        del X_np, y_np, X_mlx, y_mlx
        mx.metal.clear_cache()


def benchmark_prediction_scaling() -> None:
    """Benchmark prediction speed at scale."""
    if not HAS_SKLEARN:
        return

    print()
    print("=" * 80)
    print("PREDICTION SPEED BENCHMARK")
    print("=" * 80)
    print()

    n_features = 100
    max_depth = 10
    n_bins = 64
    n_train = 10_000

    # Train models once
    np.random.seed(42)
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = np.random.randn(n_train).astype(np.float32)

    sklearn_model = SklearnDTR(max_depth=max_depth)
    sklearn_model.fit(X_train, y_train)

    mlx_model = DecisionTreeRegressor(max_depth=max_depth, n_bins=n_bins)
    mlx_model.fit(mx.array(X_train), mx.array(y_train))

    print(f"Config: {n_features} features, depth={max_depth}")
    print()
    print(
        f"{'Test Samples':>12} {'sklearn(ms)':>12} {'MLX(ms)':>12} {'Speedup':>10} {'Winner':>10}"
    )
    print("-" * 80)

    for n_test in [1_000, 10_000, 100_000, 500_000, 1_000_000]:
        X_test_np = np.random.randn(n_test, n_features).astype(np.float32)
        X_test_mlx = mx.array(X_test_np)
        mx.eval(X_test_mlx)

        # sklearn prediction
        start = time.perf_counter()
        _ = sklearn_model.predict(X_test_np)
        sklearn_time = (time.perf_counter() - start) * 1000

        # MLX prediction
        start = time.perf_counter()
        preds = mlx_model.predict(X_test_mlx)
        mx.eval(preds)
        mlx_time = (time.perf_counter() - start) * 1000

        speedup = sklearn_time / mlx_time
        winner = "MLX" if speedup > 1 else "sklearn"

        print(
            f"{n_test:>12,} {sklearn_time:>12.2f} {mlx_time:>12.2f} {speedup:>10.2f}x {winner:>10}"
        )

        del X_test_np, X_test_mlx
        mx.metal.clear_cache()


def main() -> None:
    """Run all large-scale benchmarks."""
    print()
    print("LARGE-SCALE MLX vs sklearn BENCHMARK")
    print("Testing hypothesis: MLX faster at scale")
    print()

    benchmark_regression_scaling()
    benchmark_classification_scaling()
    benchmark_high_dimensional()
    benchmark_prediction_scaling()

    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
