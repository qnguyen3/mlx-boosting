#!/usr/bin/env python3
"""Benchmark MLX Decision Trees on MNIST dataset.

MNIST has 60,000 training samples with 784 features (28x28 images),
making it a good test for GPU acceleration on real data.

Usage:
    uv run python benchmarks/benchmark_mnist.py
"""

import time

import mlx.core as mx
import numpy as np

from mlx_boosting.trees import DecisionTreeClassifier

# Try to import sklearn for comparison
try:
    from sklearn.datasets import fetch_openml
    from sklearn.tree import DecisionTreeClassifier as SklearnDTC

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not installed. Install with: uv add scikit-learn")


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST dataset.

    Returns:
        X_train, X_test, y_train, y_test as numpy arrays.
    """
    print("Loading MNIST dataset...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X, y = mnist.data.astype(np.float32), mnist.target.astype(np.int32)

    # Normalize to [0, 1]
    X = X / 255.0

    # Split into train/test (60k train, 10k test)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print(f"  Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]:,} samples")
    print(f"  Classes: {len(np.unique(y_train))} (digits 0-9)")

    return X_train, X_test, y_train, y_test


def time_function(func, warmup: int = 1, runs: int = 3) -> float:
    """Time a function with warmup and multiple runs."""
    for _ in range(warmup):
        func()
        mx.eval()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func()
        mx.eval()
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), result


def benchmark_mnist(max_depth: int = 10, n_bins: int = 64) -> None:
    """Run MNIST benchmark comparing MLX vs sklearn.

    Args:
        max_depth: Maximum tree depth.
        n_bins: Number of histogram bins for MLX.
    """
    if not HAS_SKLEARN:
        print("sklearn required for MNIST benchmark")
        return

    X_train, X_test, y_train, y_test = load_mnist()

    print(f"\n{'=' * 70}")
    print(f"MNIST Benchmark: max_depth={max_depth}, n_bins={n_bins}")
    print("=" * 70)

    # Convert to MLX
    X_train_mlx = mx.array(X_train)
    y_train_mlx = mx.array(y_train)
    X_test_mlx = mx.array(X_test)
    mx.eval(X_train_mlx, y_train_mlx, X_test_mlx)

    # sklearn benchmark
    print("\n--- scikit-learn ---")
    sklearn_model = SklearnDTC(max_depth=max_depth)

    start = time.perf_counter()
    sklearn_model.fit(X_train, y_train)
    sklearn_fit_time = time.perf_counter() - start
    print(f"Fit time:     {sklearn_fit_time * 1000:>10.2f} ms")

    start = time.perf_counter()
    sklearn_preds = sklearn_model.predict(X_test)
    sklearn_pred_time = time.perf_counter() - start
    print(f"Predict time: {sklearn_pred_time * 1000:>10.2f} ms")

    sklearn_accuracy = np.mean(sklearn_preds == y_test)
    print(f"Accuracy:     {sklearn_accuracy * 100:>10.2f}%")

    # MLX benchmark
    print("\n--- MLX ---")
    mlx_model = DecisionTreeClassifier(max_depth=max_depth, n_bins=n_bins)

    start = time.perf_counter()
    mlx_model.fit(X_train_mlx, y_train_mlx)
    mx.eval()
    mlx_fit_time = time.perf_counter() - start
    print(f"Fit time:     {mlx_fit_time * 1000:>10.2f} ms")

    start = time.perf_counter()
    mlx_preds = mlx_model.predict(X_test_mlx)
    mx.eval(mlx_preds)
    mlx_pred_time = time.perf_counter() - start
    print(f"Predict time: {mlx_pred_time * 1000:>10.2f} ms")

    mlx_preds_np = np.array(mlx_preds)
    mlx_accuracy = np.mean(mlx_preds_np == y_test)
    print(f"Accuracy:     {mlx_accuracy * 100:>10.2f}%")

    # Comparison
    print("\n--- Comparison ---")
    fit_speedup = sklearn_fit_time / mlx_fit_time if mlx_fit_time > 0 else float("inf")
    pred_speedup = (
        sklearn_pred_time / mlx_pred_time if mlx_pred_time > 0 else float("inf")
    )
    print(f"Fit speedup:     {fit_speedup:>10.2f}x")
    print(f"Predict speedup: {pred_speedup:>10.2f}x")


def benchmark_varying_depth() -> None:
    """Benchmark at different tree depths."""
    if not HAS_SKLEARN:
        print("sklearn required for benchmark")
        return

    X_train, _, y_train, _ = load_mnist()

    # Use subset for faster iteration
    n_train = 10000
    X_train_sub = X_train[:n_train]
    y_train_sub = y_train[:n_train]

    X_train_mlx = mx.array(X_train_sub)
    y_train_mlx = mx.array(y_train_sub)
    mx.eval(X_train_mlx, y_train_mlx)

    print(f"\n{'=' * 70}")
    print(f"MNIST Benchmark: Varying Depth ({n_train:,} samples)")
    print("=" * 70)
    print(f"{'Depth':>6} {'sklearn(ms)':>12} {'MLX(ms)':>12} {'Speedup':>10}")
    print("-" * 70)

    for max_depth in [4, 6, 8, 10, 12]:
        # sklearn
        sklearn_model = SklearnDTC(max_depth=max_depth)
        start = time.perf_counter()
        sklearn_model.fit(X_train_sub, y_train_sub)
        sklearn_time = (time.perf_counter() - start) * 1000

        # MLX
        mlx_model = DecisionTreeClassifier(max_depth=max_depth, n_bins=64)
        start = time.perf_counter()
        mlx_model.fit(X_train_mlx, y_train_mlx)
        mx.eval()
        mlx_time = (time.perf_counter() - start) * 1000

        speedup = sklearn_time / mlx_time if mlx_time > 0 else float("inf")
        print(
            f"{max_depth:>6} {sklearn_time:>12.2f} {mlx_time:>12.2f} {speedup:>10.2f}x"
        )


def benchmark_varying_samples() -> None:
    """Benchmark with different sample sizes."""
    if not HAS_SKLEARN:
        print("sklearn required for benchmark")
        return

    X_train_full, _, y_train_full, _ = load_mnist()

    print(f"\n{'=' * 70}")
    print("MNIST Benchmark: Varying Sample Size (depth=8)")
    print("=" * 70)
    print(f"{'Samples':>10} {'sklearn(ms)':>12} {'MLX(ms)':>12} {'Speedup':>10}")
    print("-" * 70)

    max_depth = 8
    n_bins = 64

    for n_samples in [1000, 5000, 10000, 30000, 60000]:
        X_train = X_train_full[:n_samples]
        y_train = y_train_full[:n_samples]

        X_train_mlx = mx.array(X_train)
        y_train_mlx = mx.array(y_train)
        mx.eval(X_train_mlx, y_train_mlx)

        # sklearn
        sklearn_model = SklearnDTC(max_depth=max_depth)
        start = time.perf_counter()
        sklearn_model.fit(X_train, y_train)
        sklearn_time = (time.perf_counter() - start) * 1000

        # MLX
        mlx_model = DecisionTreeClassifier(max_depth=max_depth, n_bins=n_bins)
        start = time.perf_counter()
        mlx_model.fit(X_train_mlx, y_train_mlx)
        mx.eval()
        mlx_time = (time.perf_counter() - start) * 1000

        speedup = sklearn_time / mlx_time if mlx_time > 0 else float("inf")
        print(
            f"{n_samples:>10,} {sklearn_time:>12.2f} {mlx_time:>12.2f} {speedup:>10.2f}x"
        )


def main() -> None:
    """Run MNIST benchmarks."""
    print("MNIST Decision Tree Benchmark")
    print("=" * 70)

    # Full MNIST benchmark
    benchmark_mnist(max_depth=10, n_bins=64)

    # Varying depth
    benchmark_varying_depth()

    # Varying sample size
    benchmark_varying_samples()

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
