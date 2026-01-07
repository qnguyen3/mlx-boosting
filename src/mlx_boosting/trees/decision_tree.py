"""GPU-accelerated Decision Tree estimators using MLX.

This module provides DecisionTreeRegressor and DecisionTreeClassifier
implementations optimized for Apple Silicon GPUs.
"""

from typing import TYPE_CHECKING, Literal

import mlx.core as mx
import numpy as np

from mlx_boosting.base import BaseEstimator
from mlx_boosting.trees._predictor import (
    predict_classification,
    predict_classification_proba,
    predict_regression,
)
from mlx_boosting.trees._split_finder import DEFAULT_N_BINS
from mlx_boosting.trees._tree_builder import (
    build_classification_tree,
    build_regression_tree,
)

if TYPE_CHECKING:
    from mlx_boosting.trees._tree_structure import TreeArrays


class DecisionTreeRegressor(BaseEstimator):
    """GPU-accelerated Decision Tree Regressor using MLX.

    This implementation uses histogram-based split finding and iterative
    tree building to maximize GPU utilization on Apple Silicon.

    Args:
        max_depth: Maximum depth of the tree. Default is 6.
        min_samples_split: Minimum samples required to split a node. Default is 2.
        min_samples_leaf: Minimum samples required in a leaf node. Default is 1.
        n_bins: Number of histogram bins for split finding. Default is 256.
            Higher values give more precise splits but use more memory.
        split_method: Method for finding splits.
            - "histogram": Fast histogram-based (best for large datasets)
            - "exact": Exact threshold search (best for small datasets)

    Attributes:
        tree_: Fitted tree structure (TreeArrays) after calling fit().
        n_features_in_: Number of features seen during fit.

    Example:
        >>> import mlx.core as mx
        >>> from mlx_boosting.trees import DecisionTreeRegressor
        >>> X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = mx.array([1.0, 2.0, 3.0, 4.0])
        >>> model = DecisionTreeRegressor(max_depth=3)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_bins: int = DEFAULT_N_BINS,
        split_method: Literal["histogram", "exact"] = "histogram",
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.split_method = split_method

        self.tree_: TreeArrays | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: mx.array, y: mx.array) -> "DecisionTreeRegressor":
        """Fit the decision tree to training data.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        X = self._validate_X(X)
        y = self._validate_y(y)

        self.n_features_in_ = X.shape[1]

        self.tree_ = build_regression_tree(
            X=X,
            y=y,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_bins=self.n_bins,
            split_method=self.split_method,
        )

        return self

    def predict(self, X: mx.array) -> mx.array:
        """Make predictions on new data.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).

        Raises:
            ValueError: If model has not been fitted.
        """
        if self.tree_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._validate_X(X)

        predictions = predict_regression(self.tree_, X, self.max_depth)
        mx.eval(predictions)
        return predictions

    def _validate_X(self, X: mx.array | np.ndarray | list) -> mx.array:
        """Validate and convert input features.

        Args:
            X: Input features.

        Returns:
            Validated MLX array.
        """
        if isinstance(X, (np.ndarray, list)):
            X = mx.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X.astype(mx.float32)

    def _validate_y(self, y: mx.array | np.ndarray | list) -> mx.array:
        """Validate and convert target values.

        Args:
            y: Target values.

        Returns:
            Validated MLX array.
        """
        if isinstance(y, (np.ndarray, list)):
            y = mx.array(y)

        return y.astype(mx.float32).flatten()


class DecisionTreeClassifier(BaseEstimator):
    """GPU-accelerated Decision Tree Classifier using MLX.

    This implementation uses histogram-based split finding with Gini
    impurity or entropy for optimal GPU utilization on Apple Silicon.

    Args:
        max_depth: Maximum depth of the tree. Default is 6.
        min_samples_split: Minimum samples required to split a node. Default is 2.
        min_samples_leaf: Minimum samples required in a leaf node. Default is 1.
        n_bins: Number of histogram bins for split finding. Default is 256.
        criterion: Split criterion.
            - "gini": Gini impurity (default)
            - "entropy": Information gain

    Attributes:
        tree_: Fitted tree structure (TreeArrays) after calling fit().
        n_features_in_: Number of features seen during fit.
        n_classes_: Number of classes.
        classes_: Array of unique class labels.

    Example:
        >>> import mlx.core as mx
        >>> from mlx_boosting.trees import DecisionTreeClassifier
        >>> X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = mx.array([0, 0, 1, 1])
        >>> model = DecisionTreeClassifier(max_depth=3)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
        >>> probabilities = model.predict_proba(X)
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_bins: int = DEFAULT_N_BINS,
        criterion: Literal["gini", "entropy"] = "gini",
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.criterion = criterion

        self.tree_: TreeArrays | None = None
        self.n_features_in_: int | None = None
        self.n_classes_: int | None = None
        self.classes_: mx.array | None = None

    def fit(self, X: mx.array, y: mx.array) -> "DecisionTreeClassifier":
        """Fit the decision tree classifier to training data.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Class labels of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        X = self._validate_X(X)
        y = self._validate_y(y)

        self.n_features_in_ = X.shape[1]

        # Determine classes using numpy (MLX doesn't have unique)
        y_np = np.array(y)
        classes_np = np.unique(y_np)
        self.classes_ = mx.array(classes_np)
        self.n_classes_ = len(classes_np)

        # Map y to indices 0, 1, ..., n_classes-1
        y_mapped = self._map_labels_to_indices(y)

        self.tree_ = build_classification_tree(
            X=X,
            y=y_mapped,
            n_classes=self.n_classes_,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_bins=self.n_bins,
            criterion=self.criterion,
        )

        return self

    def predict(self, X: mx.array) -> mx.array:
        """Predict class labels for new data.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predicted class labels of shape (n_samples,).

        Raises:
            ValueError: If model has not been fitted.
        """
        if self.tree_ is None or self.classes_ is None or self.n_classes_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._validate_X(X)

        predictions = predict_classification(
            self.tree_, X, self.max_depth, self.n_classes_, self.classes_
        )
        mx.eval(predictions)
        return predictions

    def predict_proba(self, X: mx.array) -> mx.array:
        """Predict class probabilities for new data.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Class probabilities of shape (n_samples, n_classes).

        Raises:
            ValueError: If model has not been fitted.
        """
        if self.tree_ is None or self.n_classes_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._validate_X(X)

        probabilities = predict_classification_proba(
            self.tree_, X, self.max_depth, self.n_classes_
        )
        mx.eval(probabilities)
        return probabilities

    def _validate_X(self, X: mx.array | np.ndarray | list) -> mx.array:
        """Validate and convert input features."""
        if isinstance(X, (np.ndarray, list)):
            X = mx.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X.astype(mx.float32)

    def _validate_y(self, y: mx.array | np.ndarray | list) -> mx.array:
        """Validate and convert class labels."""
        if isinstance(y, (np.ndarray, list)):
            y = mx.array(y)

        return y.flatten()

    def _map_labels_to_indices(self, y: mx.array) -> mx.array:
        """Map class labels to indices 0, 1, ..., n_classes-1.

        Args:
            y: Class labels.

        Returns:
            Indices corresponding to each label.
        """
        if self.classes_ is None or self.n_classes_ is None:
            raise ValueError("Classes not determined. Call fit() first.")

        # Use numpy for label mapping
        y_np = np.array(y)
        classes_np = np.array(self.classes_)

        # Create mapping using numpy searchsorted
        y_indices = np.searchsorted(classes_np, y_np).astype(np.int32)

        return mx.array(y_indices)
