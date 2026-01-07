"""Gradient Boosting Decision Trees using MLX.

This module implements GPU-accelerated gradient boosting for
both regression and classification tasks on Apple Silicon.
"""

import logging
from typing import TYPE_CHECKING, Literal

import mlx.core as mx
import numpy as np

from mlx_boosting.base import BaseEstimator
from mlx_boosting.losses.classification import LogLoss
from mlx_boosting.losses.regression import MSELoss
from mlx_boosting.trees._predictor import predict_regression
from mlx_boosting.trees._split_finder import DEFAULT_N_BINS
from mlx_boosting.trees._tree_builder import build_regression_tree_levelwise

if TYPE_CHECKING:
    from mlx_boosting.trees._tree_structure import TreeArrays

logger = logging.getLogger(__name__)


class GradientBoostingRegressor(BaseEstimator):
    """GPU-accelerated Gradient Boosting Regressor using MLX.

    Implements gradient boosting with decision tree weak learners,
    optimized for Apple Silicon GPUs using histogram subtraction
    and level-wise batched processing.

    Args:
        n_estimators: Number of boosting iterations. Default is 100.
        learning_rate: Shrinkage factor for each tree's contribution.
            Default is 0.1.
        max_depth: Maximum depth of individual trees. Default is 6.
        min_samples_split: Minimum samples to split a node. Default is 2.
        min_samples_leaf: Minimum samples in leaf node. Default is 1.
        subsample: Fraction of samples for each tree (1.0 = no subsampling).
            Default is 1.0.
        n_bins: Number of histogram bins. Default is 256.
        loss: Loss function. Default is "squared_error".
        verbose: Verbosity level. Default is 0.

    Attributes:
        trees_: List of fitted tree structures.
        train_score_: Training loss at each iteration.
        init_prediction_: Initial prediction value.

    Example:
        >>> import mlx.core as mx
        >>> from mlx_boosting import GradientBoostingRegressor
        >>> X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = mx.array([1.0, 2.0, 3.0, 4.0])
        >>> model = GradientBoostingRegressor(n_estimators=100)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        n_bins: int = DEFAULT_N_BINS,
        loss: Literal["squared_error"] = "squared_error",
        verbose: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.n_bins = n_bins
        self.loss = loss
        self.verbose = verbose

        self.trees_: list[TreeArrays] = []
        self.train_score_: list[float] = []
        self.init_prediction_: float = 0.0
        self.n_features_in_: int | None = None
        self._loss_fn = MSELoss()

    def fit(self, X: mx.array, y: mx.array) -> "GradientBoostingRegressor":
        """Fit the gradient boosting model.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        X = self._validate_X(X)
        y = self._validate_y(y)

        n_samples = X.shape[0]
        self.n_features_in_ = X.shape[1]

        # Initialize prediction with mean
        self.init_prediction_ = float(mx.mean(y))
        y_pred = mx.full((n_samples,), self.init_prediction_, dtype=mx.float32)

        self.trees_ = []
        self.train_score_ = []

        for iteration in range(self.n_estimators):
            # Compute negative gradients (pseudo-residuals)
            gradients = self._loss_fn.gradient(y, y_pred)
            residuals = -gradients  # Negative gradient direction

            # Subsample if needed
            if self.subsample < 1.0:
                subsample_indices = self._get_subsample_indices(n_samples)
                X_sub = X[subsample_indices]
                residuals_sub = residuals[subsample_indices]
            else:
                X_sub = X
                residuals_sub = residuals

            # Fit tree to residuals using optimized level-wise builder
            tree = build_regression_tree_levelwise(
                X=X_sub,
                y=residuals_sub,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_bins=self.n_bins,
            )

            # Update predictions
            tree_pred = predict_regression(tree, X, self.max_depth)
            y_pred = y_pred + self.learning_rate * tree_pred
            mx.eval(y_pred)

            self.trees_.append(tree)

            # Track training loss
            loss_value = float(self._loss_fn.loss(y, y_pred))
            self.train_score_.append(loss_value)

            if self.verbose > 0 and (iteration + 1) % 10 == 0:
                logger.info(
                    f"Iteration {iteration + 1}/{self.n_estimators}, Loss: {loss_value:.6f}"
                )

        return self

    def predict(self, X: mx.array) -> mx.array:
        """Make predictions.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """
        if not self.trees_:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._validate_X(X)
        n_samples = X.shape[0]

        # Start with initial prediction
        y_pred = mx.full((n_samples,), self.init_prediction_, dtype=mx.float32)

        # Add contribution from each tree
        for tree in self.trees_:
            tree_pred = predict_regression(tree, X, self.max_depth)
            y_pred = y_pred + self.learning_rate * tree_pred

        mx.eval(y_pred)
        return y_pred

    def _get_subsample_indices(self, n_samples: int) -> mx.array:
        """Generate random subsample indices."""
        n_subsample = int(n_samples * self.subsample)
        indices = np.random.choice(n_samples, n_subsample, replace=False)
        return mx.array(indices)

    def _validate_X(self, X: mx.array | np.ndarray | list) -> mx.array:
        """Validate input features."""
        if isinstance(X, (np.ndarray, list)):
            X = mx.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X.astype(mx.float32)

    def _validate_y(self, y: mx.array | np.ndarray | list) -> mx.array:
        """Validate target values."""
        if isinstance(y, (np.ndarray, list)):
            y = mx.array(y)
        return y.astype(mx.float32).flatten()


class GradientBoostingClassifier(BaseEstimator):
    """GPU-accelerated Gradient Boosting Classifier using MLX.

    Implements gradient boosting for binary classification using
    logistic loss (cross-entropy), optimized for Apple Silicon GPUs.

    Args:
        n_estimators: Number of boosting iterations. Default is 100.
        learning_rate: Shrinkage factor. Default is 0.1.
        max_depth: Maximum tree depth. Default is 6.
        min_samples_split: Minimum samples to split. Default is 2.
        min_samples_leaf: Minimum samples in leaf. Default is 1.
        subsample: Fraction of samples for each tree. Default is 1.0.
        n_bins: Number of histogram bins. Default is 256.
        verbose: Verbosity level. Default is 0.

    Attributes:
        trees_: List of fitted tree structures.
        train_score_: Training loss at each iteration.
        init_prediction_: Initial log-odds prediction.
        classes_: Unique class labels.

    Example:
        >>> import mlx.core as mx
        >>> from mlx_boosting import GradientBoostingClassifier
        >>> X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = mx.array([0, 0, 1, 1])
        >>> model = GradientBoostingClassifier(n_estimators=100)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
        >>> probabilities = model.predict_proba(X)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        n_bins: int = DEFAULT_N_BINS,
        verbose: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.n_bins = n_bins
        self.verbose = verbose

        self.trees_: list[TreeArrays] = []
        self.train_score_: list[float] = []
        self.init_prediction_: float = 0.0
        self.n_features_in_: int | None = None
        self.classes_: mx.array | None = None
        self._loss_fn = LogLoss()

    def fit(self, X: mx.array, y: mx.array) -> "GradientBoostingClassifier":
        """Fit the gradient boosting classifier.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Binary class labels (0 or 1) of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        X = self._validate_X(X)
        y = self._validate_y(y)

        # Determine classes
        y_np = np.array(y)
        classes_np = np.unique(y_np)
        if len(classes_np) != 2:
            raise ValueError(
                "GradientBoostingClassifier only supports binary classification."
            )
        self.classes_ = mx.array(classes_np)

        # Map labels to 0/1
        y_binary = self._map_to_binary(y)

        n_samples = X.shape[0]
        self.n_features_in_ = X.shape[1]

        # Initialize with log-odds
        p = float(mx.mean(y_binary.astype(mx.float32)))
        p = np.clip(p, 1e-15, 1 - 1e-15)
        self.init_prediction_ = float(np.log(p / (1 - p)))

        y_pred = mx.full((n_samples,), self.init_prediction_, dtype=mx.float32)

        self.trees_ = []
        self.train_score_ = []

        for iteration in range(self.n_estimators):
            # Compute negative gradients
            gradients = self._loss_fn.gradient(y_binary.astype(mx.float32), y_pred)
            residuals = -gradients

            # Subsample if needed
            if self.subsample < 1.0:
                subsample_indices = self._get_subsample_indices(n_samples)
                X_sub = X[subsample_indices]
                residuals_sub = residuals[subsample_indices]
            else:
                X_sub = X
                residuals_sub = residuals

            # Fit tree to residuals using optimized level-wise builder
            tree = build_regression_tree_levelwise(
                X=X_sub,
                y=residuals_sub,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_bins=self.n_bins,
            )

            # Update predictions (in log-odds space)
            tree_pred = predict_regression(tree, X, self.max_depth)
            y_pred = y_pred + self.learning_rate * tree_pred
            mx.eval(y_pred)

            self.trees_.append(tree)

            # Track training loss
            loss_value = float(self._loss_fn.loss(y_binary.astype(mx.float32), y_pred))
            self.train_score_.append(loss_value)

            if self.verbose > 0 and (iteration + 1) % 10 == 0:
                logger.info(
                    f"Iteration {iteration + 1}/{self.n_estimators}, Loss: {loss_value:.6f}"
                )

        return self

    def predict(self, X: mx.array) -> mx.array:
        """Predict class labels.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        proba = self.predict_proba(X)
        class_indices = mx.argmax(proba, axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, X: mx.array) -> mx.array:
        """Predict class probabilities.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Class probabilities of shape (n_samples, 2).
        """
        if not self.trees_:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._validate_X(X)
        n_samples = X.shape[0]

        # Get raw predictions (log-odds)
        y_pred = mx.full((n_samples,), self.init_prediction_, dtype=mx.float32)

        for tree in self.trees_:
            tree_pred = predict_regression(tree, X, self.max_depth)
            y_pred = y_pred + self.learning_rate * tree_pred

        # Convert to probabilities
        prob_positive = mx.sigmoid(y_pred)
        prob_negative = 1.0 - prob_positive

        proba = mx.stack([prob_negative, prob_positive], axis=1)
        mx.eval(proba)
        return proba

    def _map_to_binary(self, y: mx.array) -> mx.array:
        """Map class labels to binary 0/1."""
        if self.classes_ is None:
            raise ValueError("Classes not determined.")
        y_np = np.array(y)
        classes_np = np.array(self.classes_)
        return mx.array((y_np == classes_np[1]).astype(np.float32))

    def _get_subsample_indices(self, n_samples: int) -> mx.array:
        """Generate random subsample indices."""
        n_subsample = int(n_samples * self.subsample)
        indices = np.random.choice(n_samples, n_subsample, replace=False)
        return mx.array(indices)

    def _validate_X(self, X: mx.array | np.ndarray | list) -> mx.array:
        """Validate input features."""
        if isinstance(X, (np.ndarray, list)):
            X = mx.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X.astype(mx.float32)

    def _validate_y(self, y: mx.array | np.ndarray | list) -> mx.array:
        """Validate target values."""
        if isinstance(y, (np.ndarray, list)):
            y = mx.array(y)
        return y.flatten()
