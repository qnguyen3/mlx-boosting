"""Benchmark XGBoost: MLX vs official xgboost library."""

import time
from typing import Any

import mlx.core as mx
import numpy as np


def benchmark_regression(
    n_samples: int, n_features: int, n_estimators: int
) -> dict[str, Any]:
    """Benchmark XGBoost regression on synthetic data."""
    print(f"\n{'=' * 60}")
    print(
        f"XGBoost Regression: {n_samples:,} samples, {n_features} features, {n_estimators} trees"
    )
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    y_np = np.random.randn(n_samples).astype(np.float32)

    results = {}

    # Official xgboost benchmark
    try:
        import xgboost as xgb

        xgb_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.3,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=1,  # Single thread for fair comparison
        )

        start = time.perf_counter()
        xgb_model.fit(X_np, y_np)
        xgb_time = time.perf_counter() - start
        results["xgboost_time"] = xgb_time
        print(f"xgboost:     {xgb_time:.3f}s")
    except ImportError:
        print("xgboost not installed, skipping xgboost benchmark")
        xgb_time = None

    # MLX benchmark
    from mlx_boosting import XGBoostRegressor

    X_mx = mx.array(X_np)
    y_mx = mx.array(y_np)

    mlx_model = XGBoostRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.3,
        reg_lambda=1.0,
    )

    # Warm-up
    mx.eval(X_mx, y_mx)

    start = time.perf_counter()
    mlx_model.fit(X_mx, y_mx)
    mlx_time = time.perf_counter() - start
    results["mlx_time"] = mlx_time
    print(f"MLX:         {mlx_time:.3f}s")

    if xgb_time is not None:
        speedup = xgb_time / mlx_time
        results["speedup"] = speedup
        print(f"Speedup:     {speedup:.2f}x")

    return results


def benchmark_classification_binary(
    n_samples: int, n_features: int, n_estimators: int
) -> dict[str, Any]:
    """Benchmark XGBoost binary classification on synthetic data."""
    print(f"\n{'=' * 60}")
    print(
        f"XGBoost Binary Classification: {n_samples:,} samples, {n_features} features, {n_estimators} trees"
    )
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    y_np = (np.random.randn(n_samples) > 0).astype(np.int32)

    results = {}

    # Official xgboost benchmark
    try:
        import xgboost as xgb

        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.3,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=1,
        )

        start = time.perf_counter()
        xgb_model.fit(X_np, y_np)
        xgb_time = time.perf_counter() - start
        results["xgboost_time"] = xgb_time
        print(f"xgboost:     {xgb_time:.3f}s")
    except ImportError:
        print("xgboost not installed, skipping xgboost benchmark")
        xgb_time = None

    # MLX benchmark
    from mlx_boosting import XGBoostClassifier

    X_mx = mx.array(X_np)
    y_mx = mx.array(y_np)

    mlx_model = XGBoostClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.3,
        reg_lambda=1.0,
    )

    # Warm-up
    mx.eval(X_mx, y_mx)

    start = time.perf_counter()
    mlx_model.fit(X_mx, y_mx)
    mlx_time = time.perf_counter() - start
    results["mlx_time"] = mlx_time
    print(f"MLX:         {mlx_time:.3f}s")

    if xgb_time is not None:
        speedup = xgb_time / mlx_time
        results["speedup"] = speedup
        print(f"Speedup:     {speedup:.2f}x")

    return results


def benchmark_classification_multiclass(
    n_samples: int, n_features: int, n_estimators: int, n_classes: int = 3
) -> dict[str, Any]:
    """Benchmark XGBoost multi-class classification on synthetic data."""
    print(f"\n{'=' * 60}")
    print(
        f"XGBoost {n_classes}-Class Classification: {n_samples:,} samples, {n_features} features, {n_estimators} trees"
    )
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    y_np = np.random.randint(0, n_classes, size=n_samples)

    results = {}

    # Official xgboost benchmark
    try:
        import xgboost as xgb

        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.3,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=1,
        )

        start = time.perf_counter()
        xgb_model.fit(X_np, y_np)
        xgb_time = time.perf_counter() - start
        results["xgboost_time"] = xgb_time
        print(f"xgboost:     {xgb_time:.3f}s")
    except ImportError:
        print("xgboost not installed, skipping xgboost benchmark")
        xgb_time = None

    # MLX benchmark
    from mlx_boosting import XGBoostClassifier

    X_mx = mx.array(X_np)
    y_mx = mx.array(y_np)

    mlx_model = XGBoostClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.3,
        reg_lambda=1.0,
    )

    # Warm-up
    mx.eval(X_mx, y_mx)

    start = time.perf_counter()
    mlx_model.fit(X_mx, y_mx)
    mlx_time = time.perf_counter() - start
    results["mlx_time"] = mlx_time
    print(f"MLX:         {mlx_time:.3f}s")

    if xgb_time is not None:
        speedup = xgb_time / mlx_time
        results["speedup"] = speedup
        print(f"Speedup:     {speedup:.2f}x")

    return results


def benchmark_with_missing_values(
    n_samples: int, n_features: int, n_estimators: int, missing_ratio: float = 0.1
) -> dict[str, Any]:
    """Benchmark XGBoost with missing values."""
    print(f"\n{'=' * 60}")
    print(
        f"XGBoost with {missing_ratio * 100:.0f}% Missing Values: {n_samples:,} samples, {n_features} features, {n_estimators} trees"
    )
    print("=" * 60)

    # Generate data with missing values
    np.random.seed(42)
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    y_np = np.random.randn(n_samples).astype(np.float32)

    # Introduce missing values
    missing_mask = np.random.random(X_np.shape) < missing_ratio
    X_np[missing_mask] = np.nan

    results = {}

    # Official xgboost benchmark
    try:
        import xgboost as xgb

        xgb_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.3,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=1,
        )

        start = time.perf_counter()
        xgb_model.fit(X_np, y_np)
        xgb_time = time.perf_counter() - start
        results["xgboost_time"] = xgb_time
        print(f"xgboost:     {xgb_time:.3f}s")
    except ImportError:
        print("xgboost not installed, skipping xgboost benchmark")
        xgb_time = None

    # MLX benchmark
    from mlx_boosting import XGBoostRegressor

    X_mx = mx.array(X_np)
    y_mx = mx.array(y_np)

    mlx_model = XGBoostRegressor(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.3,
        reg_lambda=1.0,
    )

    # Warm-up
    mx.eval(X_mx, y_mx)

    start = time.perf_counter()
    mlx_model.fit(X_mx, y_mx)
    mlx_time = time.perf_counter() - start
    results["mlx_time"] = mlx_time
    print(f"MLX:         {mlx_time:.3f}s")

    if xgb_time is not None:
        speedup = xgb_time / mlx_time
        results["speedup"] = speedup
        print(f"Speedup:     {speedup:.2f}x")

    return results


def main() -> None:
    """Run all benchmarks."""
    print("\n" + "=" * 60)
    print("MLX-Boosting XGBoost Benchmark vs Official xgboost")
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
        # Regression
        reg_results = benchmark_regression(n_samples, n_features, n_estimators)
        reg_results["task"] = "regression"
        reg_results["n_samples"] = n_samples
        reg_results["n_features"] = n_features
        reg_results["n_estimators"] = n_estimators
        all_results.append(reg_results)

        # Binary classification
        cls_results = benchmark_classification_binary(
            n_samples, n_features, n_estimators
        )
        cls_results["task"] = "binary_cls"
        cls_results["n_samples"] = n_samples
        cls_results["n_features"] = n_features
        cls_results["n_estimators"] = n_estimators
        all_results.append(cls_results)

    # Multi-class classification
    print("\n" + "=" * 60)
    print("Multi-class Classification Benchmarks")
    print("=" * 60)
    for n_classes in [3, 5, 10]:
        mc_results = benchmark_classification_multiclass(50_000, 20, 50, n_classes)
        mc_results["task"] = f"{n_classes}_class"
        mc_results["n_samples"] = 50_000
        mc_results["n_features"] = 20
        mc_results["n_estimators"] = 50
        all_results.append(mc_results)

    # Missing values benchmark
    print("\n" + "=" * 60)
    print("Missing Values Benchmarks")
    print("=" * 60)
    missing_results = benchmark_with_missing_values(50_000, 20, 50, missing_ratio=0.1)
    missing_results["task"] = "missing_10%"
    missing_results["n_samples"] = 50_000
    missing_results["n_features"] = 20
    missing_results["n_estimators"] = 50
    all_results.append(missing_results)

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
