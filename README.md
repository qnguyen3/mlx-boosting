# MLX-Boosting

GPU-accelerated gradient boosting algorithms for Apple Silicon, built on [Apple MLX](https://github.com/ml-explore/mlx).

[![PyPI version](https://badge.fury.io/py/mlx-boosting.svg)](https://badge.fury.io/py/mlx-boosting)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **XGBoost-style implementation** with second-order gradients
- **Gradient Boosted Decision Trees (GBDT)** for regression and classification
- **Decision Trees** as standalone estimators
- **Optimized for Apple Silicon** (M1/M2/M3/M4) using MLX
- **Numba JIT compilation** for fast tree building
- **scikit-learn compatible API**

## Installation

```bash
pip install mlx-boosting
```

**Requirements:**
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+

## Quick Start

### XGBoost Regressor

```python
import mlx.core as mx
from mlx_boosting import XGBoostRegressor

# Create sample data
X = mx.random.normal((1000, 10))
y = mx.random.normal((1000,))

# Train model
model = XGBoostRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
)
model.fit(X, y)

# Predict
predictions = model.predict(X)
```

### XGBoost Classifier

```python
import mlx.core as mx
from mlx_boosting import XGBoostClassifier

# Binary classification
X = mx.random.normal((1000, 10))
y = mx.array((mx.random.uniform((1000,)) > 0.5).astype(mx.int32))

model = XGBoostClassifier(n_estimators=100, max_depth=6)
model.fit(X, y)

# Predict probabilities
probs = model.predict_proba(X)

# Predict classes
classes = model.predict(X)
```

### Gradient Boosting

```python
from mlx_boosting import GradientBoostingRegressor, GradientBoostingClassifier

# Regression
reg = GradientBoostingRegressor(n_estimators=100, max_depth=4)
reg.fit(X, y)

# Classification
clf = GradientBoostingClassifier(n_estimators=100, max_depth=4)
clf.fit(X, y_class)
```

### Decision Trees

```python
from mlx_boosting import DecisionTreeRegressor, DecisionTreeClassifier

# Standalone decision tree
tree = DecisionTreeRegressor(max_depth=6)
tree.fit(X, y)
predictions = tree.predict(X)
```

## Parameters

### XGBoostRegressor / XGBoostClassifier

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 100 | Number of boosting rounds |
| `max_depth` | 6 | Maximum tree depth |
| `learning_rate` | 0.3 | Step size shrinkage |
| `min_child_weight` | 1.0 | Minimum sum of instance weight in a child |
| `reg_lambda` | 1.0 | L2 regularization term |
| `reg_alpha` | 0.0 | L1 regularization term |
| `gamma` | 0.0 | Minimum loss reduction for split |
| `subsample` | 1.0 | Subsample ratio of training instances |
| `colsample_bytree` | 1.0 | Subsample ratio of columns per tree |
| `n_bins` | 256 | Number of histogram bins |

### GradientBoostingRegressor / GradientBoostingClassifier

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 100 | Number of boosting rounds |
| `max_depth` | 3 | Maximum tree depth |
| `learning_rate` | 0.1 | Step size shrinkage |
| `min_samples_split` | 2 | Minimum samples required to split |
| `min_samples_leaf` | 1 | Minimum samples required in a leaf |

## Performance

MLX-Boosting is optimized for Apple Silicon and achieves excellent performance on high-volume datasets:

| Dataset Size | vs sklearn |
|--------------|------------|
| 10K samples | ~1.5x faster |
| 50K samples | ~2x faster |
| 100K samples | **up to 3x faster** |

MLX-Boosting achieves **up to 3x faster training** on high-volume data compared to sklearn's GradientBoosting, running natively on Apple Silicon.

## Working with NumPy

MLX-Boosting works seamlessly with NumPy arrays:

```python
import numpy as np
import mlx.core as mx
from mlx_boosting import XGBoostRegressor

# NumPy data
X_np = np.random.randn(1000, 10).astype(np.float32)
y_np = np.random.randn(1000).astype(np.float32)

# Convert to MLX
X = mx.array(X_np)
y = mx.array(y_np)

# Train
model = XGBoostRegressor(n_estimators=100)
model.fit(X, y)

# Predictions back to NumPy
preds = np.array(model.predict(X))
```

## API Reference

### Classes

- `XGBoostRegressor` - XGBoost-style regression
- `XGBoostClassifier` - XGBoost-style classification (binary and multiclass)
- `GradientBoostingRegressor` - GBDT regression
- `GradientBoostingClassifier` - GBDT classification
- `DecisionTreeRegressor` - Decision tree regression
- `DecisionTreeClassifier` - Decision tree classification

### Common Methods

- `fit(X, y)` - Train the model
- `predict(X)` - Make predictions
- `predict_proba(X)` - Predict probabilities (classifiers only)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Apple MLX](https://github.com/ml-explore/mlx) - The foundation for GPU acceleration
- [XGBoost](https://github.com/dmlc/xgboost) - Inspiration for the algorithm implementation
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) - API design patterns
