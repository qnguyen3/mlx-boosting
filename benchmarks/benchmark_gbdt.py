"""Benchmark GBDT: MLX vs sklearn."""

import time
from typing import Any

import mlx.core as mx
import numpy as np


def benchmark_regression(
    n_samples: int, n_features: int, n_estimators: int
) -> dict[str, Any]:
    """Benchmark regression on synthetic data."""
    print(f"\n{'=' * 60}")
    print(
        f"Regression: {n_samples:,} samples, {n_features} features, {n_estimators} trees"
    )
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    y_np = np.random.randn(n_samples).astype(np.float32)

    results = {}

    # sklearn benchmark
    try:
        from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR

        sklearn_model = SklearnGBR(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
        )

        start = time.perf_counter()
        sklearn_model.fit(X_np, y_np)
        sklearn_time = time.perf_counter() - start
        results["sklearn_time"] = sklearn_time
        print(f"sklearn:     {sklearn_time:.3f}s")
    except ImportError:
        print("sklearn not installed, skipping sklearn benchmark")
        sklearn_time = None

    # MLX benchmark
    from mlx_boosting import GradientBoostingRegressor

    X_mx = mx.array(X_np)
    y_mx = mx.array(y_np)

    mlx_model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
    )

    # Warm-up
    mx.eval(X_mx, y_mx)

    start = time.perf_counter()
    mlx_model.fit(X_mx, y_mx)
    mlx_time = time.perf_counter() - start
    results["mlx_time"] = mlx_time
    print(f"MLX:         {mlx_time:.3f}s")

    if sklearn_time is not None:
        speedup = sklearn_time / mlx_time
        results["speedup"] = speedup
        print(f"Speedup:     {speedup:.2f}x")

    return results


def benchmark_classification(
    n_samples: int, n_features: int, n_estimators: int
) -> dict[str, Any]:
    """Benchmark classification on synthetic data."""
    print(f"\n{'=' * 60}")
    print(
        f"Classification: {n_samples:,} samples, {n_features} features, {n_estimators} trees"
    )
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    y_np = (np.random.randn(n_samples) > 0).astype(np.int32)

    results = {}

    # sklearn benchmark
    try:
        from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC

        sklearn_model = SklearnGBC(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
        )

        start = time.perf_counter()
        sklearn_model.fit(X_np, y_np)
        sklearn_time = time.perf_counter() - start
        results["sklearn_time"] = sklearn_time
        print(f"sklearn:     {sklearn_time:.3f}s")
    except ImportError:
        print("sklearn not installed, skipping sklearn benchmark")
        sklearn_time = None

    # MLX benchmark
    from mlx_boosting import GradientBoostingClassifier

    X_mx = mx.array(X_np)
    y_mx = mx.array(y_np)

    mlx_model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
    )

    # Warm-up
    mx.eval(X_mx, y_mx)

    start = time.perf_counter()
    mlx_model.fit(X_mx, y_mx)
    mlx_time = time.perf_counter() - start
    results["mlx_time"] = mlx_time
    print(f"MLX:         {mlx_time:.3f}s")

    if sklearn_time is not None:
        speedup = sklearn_time / mlx_time
        results["speedup"] = speedup
        print(f"Speedup:     {speedup:.2f}x")

    return results


def main() -> None:
    """Run all benchmarks."""
    print("\n" + "=" * 60)
    print("MLX-Boosting GBDT Benchmark vs sklearn")
    print("=" * 60)

    all_results = []

    # Test different scales
    configs = [
        # (n_samples, n_features, n_estimators)
        (1_000, 10, 50),
        (10_000, 20, 100),
        (50_000, 20, 100),
        (100_000, 20, 100),
    ]

    for n_samples, n_features, n_estimators in configs:
        reg_results = benchmark_regression(n_samples, n_features, n_estimators)
        reg_results["task"] = "regression"
        reg_results["n_samples"] = n_samples
        reg_results["n_features"] = n_features
        reg_results["n_estimators"] = n_estimators
        all_results.append(reg_results)

        cls_results = benchmark_classification(n_samples, n_features, n_estimators)
        cls_results["task"] = "classification"
        cls_results["n_samples"] = n_samples
        cls_results["n_features"] = n_features
        cls_results["n_estimators"] = n_estimators
        all_results.append(cls_results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Task':<15} {'Samples':>10} {'Features':>10} {'Trees':>8} {'Speedup':>10}")
    print("-" * 60)

    for r in all_results:
        if "speedup" in r:
            print(
                f"{r['task']:<15} {r['n_samples']:>10,} {r['n_features']:>10} "
                f"{r['n_estimators']:>8} {r['speedup']:>9.2f}x"
            )


if __name__ == "__main__":
    main()
