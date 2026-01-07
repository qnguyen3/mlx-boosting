"""Tree implementations for MLX Boosting.

GPU-accelerated decision tree implementations optimized for Apple Silicon.
"""

from mlx_boosting.trees.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

__all__ = ["DecisionTreeRegressor", "DecisionTreeClassifier"]
