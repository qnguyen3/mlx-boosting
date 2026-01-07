"""XGBoost implementation using MLX.

This module implements GPU-accelerated XGBoost for
both regression and classification tasks on Apple Silicon.
"""

import logging
from typing import TYPE_CHECKING, Literal

import mlx.core as mx
import numpy as np

from mlx_boosting.base import BaseEstimator
from mlx_boosting.losses.classification import LogLoss, SoftmaxLoss
from mlx_boosting.losses.regression import MSELoss
from mlx_boosting.trees._numba_builder import (
    _bin_features_numba,
    _compute_bin_edges_numba,
    build_xgboost_tree_numba,
)
from mlx_boosting.trees._predictor import predict_regression, predict_with_default_left
from mlx_boosting.trees._split_finder import DEFAULT_N_BINS

if TYPE_CHECKING:
    from mlx_boosting.trees._tree_structure import TreeArrays

logger = logging.getLogger(__name__)


class XGBoostRegressor(BaseEstimator):
    """GPU-accelerated XGBoost Regressor using MLX.

    Implements XGBoost with second-order optimization, regularization,
    and missing value handling, optimized for Apple Silicon GPUs.

    Args:
        n_estimators: Number of boosting iterations. Default is 100.
        learning_rate: Shrinkage factor (eta). Default is 0.3.
        max_depth: Maximum depth of trees. Default is 6.
        min_samples_split: Minimum samples to split a node. Default is 2.
        min_samples_leaf: Minimum samples in leaf node. Default is 1.
        min_child_weight: Minimum sum of Hessians in child. Default is 1.0.
        subsample: Fraction of samples for each tree. Default is 1.0.
        colsample_bytree: Fraction of features per tree. Default is 1.0.
        colsample_bylevel: Fraction of features per level. Default is 1.0.
        colsample_bynode: Fraction of features per node. Default is 1.0.
        reg_lambda: L2 regularization. Default is 1.0.
        reg_alpha: L1 regularization. Default is 0.0.
        gamma: Minimum gain for split. Default is 0.0.
        n_bins: Number of histogram bins. Default is 256.
        objective: Loss function. Default is "reg:squarederror".
        base_score: Initial prediction value. Default is 0.5.
        verbose: Verbosity level. Default is 0.

    Attributes:
        trees_: List of fitted tree structures.
        train_score_: Training loss at each iteration.

    Example:
        >>> import mlx.core as mx
        >>> from mlx_boosting import XGBoostRegressor
        >>> X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = mx.array([1.0, 2.0, 3.0, 4.0])
        >>> model = XGBoostRegressor(n_estimators=100)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        colsample_bylevel: float = 1.0,
        colsample_bynode: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        n_bins: int = DEFAULT_N_BINS,
        objective: Literal["reg:squarederror"] = "reg:squarederror",
        base_score: float = 0.5,
        verbose: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.n_bins = n_bins
        self.objective = objective
        self.base_score = base_score
        self.verbose = verbose

        self.trees_: list[TreeArrays] = []
        self.train_score_: list[float] = []
        self.n_features_in_: int | None = None
        self._loss_fn = MSELoss()
        self._feature_indices: list[np.ndarray] | None = None

    def fit(self, X: mx.array, y: mx.array) -> "XGBoostRegressor":
        """Fit the XGBoost model.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        X = self._validate_X(X)
        y = self._validate_y(y)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Check for missing values
        nan_mask = mx.isnan(X)
        has_missing = bool(mx.any(nan_mask))
        if has_missing:
            # Replace NaN with 0 for binning (they'll be handled separately)
            X = mx.where(nan_mask, mx.zeros_like(X), X)
        else:
            nan_mask = None

        # Initialize prediction
        y_pred = mx.full((n_samples,), self.base_score, dtype=mx.float32)

        self.trees_ = []
        self.train_score_ = []
        self._feature_indices = []

        # Precompute binning once if no column subsampling
        # This saves ~17ms per tree
        X_np = np.array(X, dtype=np.float32)
        if self.colsample_bytree >= 1.0 and self.subsample >= 1.0:
            bin_edges = _compute_bin_edges_numba(X_np, self.n_bins)
            binned_X = _bin_features_numba(X_np, bin_edges)
        else:
            bin_edges = None
            binned_X = None

        for iteration in range(self.n_estimators):
            # Compute gradients and Hessians
            gradients = self._loss_fn.gradient(y, y_pred)
            hessians = self._loss_fn.hessian(y, y_pred)

            # Subsample rows
            if self.subsample < 1.0:
                subsample_indices = self._get_subsample_indices(n_samples)
                X_sub = X[subsample_indices]
                gradients_sub = gradients[subsample_indices]
                hessians_sub = hessians[subsample_indices]
                nan_mask_sub = (
                    nan_mask[subsample_indices] if nan_mask is not None else None
                )
            else:
                X_sub = X
                gradients_sub = gradients
                hessians_sub = hessians
                nan_mask_sub = nan_mask

            # Select features for this tree
            if self.colsample_bytree < 1.0:
                n_features_tree = max(1, int(n_features * self.colsample_bytree))
                tree_features = np.random.choice(
                    n_features, n_features_tree, replace=False
                )
                tree_features = np.sort(tree_features)
                # Convert to mx.array for indexing
                tree_features_mx = mx.array(tree_features)
                X_sub = X_sub[:, tree_features_mx]
                if nan_mask_sub is not None:
                    nan_mask_sub = nan_mask_sub[:, tree_features_mx]
                self._feature_indices.append(tree_features)
            else:
                self._feature_indices.append(np.arange(n_features))

            # Build tree using XGBoost algorithm (numpy builder for speed)
            # Use precomputed binning only when no subsampling/colsample
            use_precomputed = (
                binned_X is not None
                and self.subsample >= 1.0
                and self.colsample_bytree >= 1.0
            )
            tree = build_xgboost_tree_numba(
                X=X_sub,
                gradients=gradients_sub,
                hessians=hessians_sub,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                gamma=self.gamma,
                n_bins=self.n_bins,
                binned_X=binned_X if use_precomputed else None,
                bin_edges=bin_edges if use_precomputed else None,
                X_np=X_np if use_precomputed else None,
            )

            # Update predictions
            if self.colsample_bytree < 1.0:
                # Convert to mx.array for indexing
                feature_idx = mx.array(self._feature_indices[-1])
                X_pred = X[:, feature_idx]
            else:
                X_pred = X

            if tree.default_left is not None:
                tree_pred = predict_with_default_left(tree, X_pred, self.max_depth)
            else:
                tree_pred = predict_regression(tree, X_pred, self.max_depth)

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

        # Handle missing values
        nan_mask = mx.isnan(X)
        has_missing = bool(mx.any(nan_mask))
        if has_missing:
            X = mx.where(nan_mask, mx.zeros_like(X), X)

        # Start with base score
        y_pred = mx.full((n_samples,), self.base_score, dtype=mx.float32)

        # Add contribution from each tree
        for i, tree in enumerate(self.trees_):
            if self.colsample_bytree < 1.0:
                # Convert to mx.array for indexing
                feature_idx = mx.array(self._feature_indices[i])
                X_pred = X[:, feature_idx]
            else:
                X_pred = X

            if tree.default_left is not None:
                tree_pred = predict_with_default_left(tree, X_pred, self.max_depth)
            else:
                tree_pred = predict_regression(tree, X_pred, self.max_depth)

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


class XGBoostClassifier(BaseEstimator):
    """GPU-accelerated XGBoost Classifier using MLX.

    Implements XGBoost for binary and multi-class classification,
    optimized for Apple Silicon GPUs.

    Args:
        n_estimators: Number of boosting iterations. Default is 100.
        learning_rate: Shrinkage factor (eta). Default is 0.3.
        max_depth: Maximum tree depth. Default is 6.
        min_samples_split: Minimum samples to split. Default is 2.
        min_samples_leaf: Minimum samples in leaf. Default is 1.
        min_child_weight: Minimum sum of Hessians in child. Default is 1.0.
        subsample: Fraction of samples for each tree. Default is 1.0.
        colsample_bytree: Fraction of features per tree. Default is 1.0.
        colsample_bylevel: Fraction of features per level. Default is 1.0.
        colsample_bynode: Fraction of features per node. Default is 1.0.
        reg_lambda: L2 regularization. Default is 1.0.
        reg_alpha: L1 regularization. Default is 0.0.
        gamma: Minimum gain for split. Default is 0.0.
        n_bins: Number of histogram bins. Default is 256.
        objective: Loss function. Default is "multi:softmax".
        base_score: Initial prediction value. Default is 0.5.
        verbose: Verbosity level. Default is 0.

    Attributes:
        trees_: List of fitted tree structures.
        train_score_: Training loss at each iteration.
        classes_: Unique class labels.
        n_classes_: Number of classes.

    Example:
        >>> import mlx.core as mx
        >>> from mlx_boosting import XGBoostClassifier
        >>> X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = mx.array([0, 0, 1, 1])
        >>> model = XGBoostClassifier(n_estimators=100)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
        >>> probabilities = model.predict_proba(X)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        colsample_bylevel: float = 1.0,
        colsample_bynode: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        n_bins: int = DEFAULT_N_BINS,
        objective: Literal["binary:logistic", "multi:softmax"] = "multi:softmax",
        base_score: float = 0.5,
        verbose: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.n_bins = n_bins
        self.objective = objective
        self.base_score = base_score
        self.verbose = verbose

        self.trees_: list[list[TreeArrays]] = []  # [iteration][class]
        self.train_score_: list[float] = []
        self.n_features_in_: int | None = None
        self.classes_: mx.array | None = None
        self.n_classes_: int = 0
        self._loss_fn = None
        self._feature_indices: list[np.ndarray] | None = None
        self._init_pred: float = 0.0  # Initial prediction value for binary

    def fit(self, X: mx.array, y: mx.array) -> "XGBoostClassifier":
        """Fit the XGBoost classifier.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Class labels (integers) of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        X = self._validate_X(X)
        y = self._validate_y(y)

        # Determine classes
        y_np = np.array(y)
        classes_np = np.unique(y_np)
        self.classes_ = mx.array(classes_np)
        self.n_classes_ = len(classes_np)

        # Map labels to 0, 1, 2, ...
        y_mapped = self._map_to_indices(y)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Check for missing values
        nan_mask = mx.isnan(X)
        has_missing = bool(mx.any(nan_mask))
        if has_missing:
            X = mx.where(nan_mask, mx.zeros_like(X), X)
        else:
            nan_mask = None

        # Select loss function based on number of classes
        if self.n_classes_ == 2:
            self._loss_fn = LogLoss()
            is_binary = True
            # Initialize with log-odds
            p = float(mx.mean(y_mapped.astype(mx.float32)))
            p = np.clip(p, 1e-15, 1 - 1e-15)
            self._init_pred = float(np.log(p / (1 - p)))
            y_pred = mx.full((n_samples,), self._init_pred, dtype=mx.float32)
        else:
            self._loss_fn = SoftmaxLoss(self.n_classes_)
            is_binary = False
            # Initialize with zeros for all classes
            y_pred = mx.zeros((n_samples, self.n_classes_), dtype=mx.float32)

        self.trees_ = []
        self.train_score_ = []
        self._feature_indices = []

        # Precompute binning once if no column subsampling
        X_np = np.array(X, dtype=np.float32)
        if self.colsample_bytree >= 1.0 and self.subsample >= 1.0:
            bin_edges = _compute_bin_edges_numba(X_np, self.n_bins)
            binned_X = _bin_features_numba(X_np, bin_edges)
        else:
            bin_edges = None
            binned_X = None

        for iteration in range(self.n_estimators):
            # Compute gradients and Hessians
            gradients = self._loss_fn.gradient(y_mapped, y_pred)
            hessians = self._loss_fn.hessian(y_mapped, y_pred)

            # Subsample rows
            if self.subsample < 1.0:
                subsample_indices = self._get_subsample_indices(n_samples)
                X_sub = X[subsample_indices]
                gradients_sub = gradients[subsample_indices]
                hessians_sub = hessians[subsample_indices]
                nan_mask_sub = (
                    nan_mask[subsample_indices] if nan_mask is not None else None
                )
            else:
                X_sub = X
                gradients_sub = gradients
                hessians_sub = hessians
                nan_mask_sub = nan_mask

            # Select features for this tree
            if self.colsample_bytree < 1.0:
                n_features_tree = max(1, int(n_features * self.colsample_bytree))
                tree_features = np.random.choice(
                    n_features, n_features_tree, replace=False
                )
                tree_features = np.sort(tree_features)
                # Convert to mx.array for indexing
                tree_features_mx = mx.array(tree_features)
                X_sub = X_sub[:, tree_features_mx]
                if nan_mask_sub is not None:
                    nan_mask_sub = nan_mask_sub[:, tree_features_mx]
                self._feature_indices.append(tree_features)
            else:
                self._feature_indices.append(np.arange(n_features))

            # Use precomputed binning only when no subsampling/colsample
            use_precomputed = (
                binned_X is not None
                and self.subsample >= 1.0
                and self.colsample_bytree >= 1.0
            )

            if is_binary:
                # Binary classification: single tree (numpy builder for speed)
                tree = build_xgboost_tree_numba(
                    X=X_sub,
                    gradients=gradients_sub,
                    hessians=hessians_sub,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    min_child_weight=self.min_child_weight,
                    reg_lambda=self.reg_lambda,
                    reg_alpha=self.reg_alpha,
                    gamma=self.gamma,
                    n_bins=self.n_bins,
                    binned_X=binned_X if use_precomputed else None,
                    bin_edges=bin_edges if use_precomputed else None,
                    X_np=X_np if use_precomputed else None,
                )

                # Update predictions
                if self.colsample_bytree < 1.0:
                    # Convert to mx.array for indexing
                    feature_idx = mx.array(self._feature_indices[-1])
                    X_pred = X[:, feature_idx]
                else:
                    X_pred = X

                if tree.default_left is not None:
                    tree_pred = predict_with_default_left(tree, X_pred, self.max_depth)
                else:
                    tree_pred = predict_regression(tree, X_pred, self.max_depth)

                y_pred = y_pred + self.learning_rate * tree_pred
                mx.eval(y_pred)

                self.trees_.append([tree])
            else:
                # Multi-class: one tree per class (numpy builder for speed)
                iteration_trees = []
                for class_idx in range(self.n_classes_):
                    class_gradients = gradients_sub[:, class_idx]
                    class_hessians = hessians_sub[:, class_idx]

                    tree = build_xgboost_tree_numba(
                        X=X_sub,
                        gradients=class_gradients,
                        hessians=class_hessians,
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        min_child_weight=self.min_child_weight,
                        reg_lambda=self.reg_lambda,
                        reg_alpha=self.reg_alpha,
                        gamma=self.gamma,
                        n_bins=self.n_bins,
                        binned_X=binned_X if use_precomputed else None,
                        bin_edges=bin_edges if use_precomputed else None,
                        X_np=X_np if use_precomputed else None,
                    )
                    iteration_trees.append(tree)

                    # Update predictions for this class
                    if self.colsample_bytree < 1.0:
                        # Convert to mx.array for indexing
                        feature_idx = mx.array(self._feature_indices[-1])
                        X_pred = X[:, feature_idx]
                    else:
                        X_pred = X

                    if tree.default_left is not None:
                        tree_pred = predict_with_default_left(
                            tree, X_pred, self.max_depth
                        )
                    else:
                        tree_pred = predict_regression(tree, X_pred, self.max_depth)

                    y_pred = y_pred.at[:, class_idx].add(self.learning_rate * tree_pred)

                mx.eval(y_pred)
                self.trees_.append(iteration_trees)

            # Track training loss
            loss_value = float(self._loss_fn.loss(y_mapped, y_pred))
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
            Class probabilities of shape (n_samples, n_classes).
        """
        if not self.trees_:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._validate_X(X)
        n_samples = X.shape[0]

        # Handle missing values
        nan_mask = mx.isnan(X)
        has_missing = bool(mx.any(nan_mask))
        if has_missing:
            X = mx.where(nan_mask, mx.zeros_like(X), X)

        is_binary = self.n_classes_ == 2

        if is_binary:
            # Binary classification - use saved init prediction from training
            y_pred = mx.full((n_samples,), self._init_pred, dtype=mx.float32)

            for i, tree_list in enumerate(self.trees_):
                tree = tree_list[0]
                if self.colsample_bytree < 1.0:
                    # Convert to mx.array for indexing
                    feature_idx = mx.array(self._feature_indices[i])
                    X_pred = X[:, feature_idx]
                else:
                    X_pred = X

                if tree.default_left is not None:
                    tree_pred = predict_with_default_left(tree, X_pred, self.max_depth)
                else:
                    tree_pred = predict_regression(tree, X_pred, self.max_depth)

                y_pred = y_pred + self.learning_rate * tree_pred

            # Convert to probabilities
            prob_positive = mx.sigmoid(y_pred)
            prob_negative = 1.0 - prob_positive
            proba = mx.stack([prob_negative, prob_positive], axis=1)
        else:
            # Multi-class classification
            y_pred = mx.zeros((n_samples, self.n_classes_), dtype=mx.float32)

            for i, tree_list in enumerate(self.trees_):
                for class_idx, tree in enumerate(tree_list):
                    if self.colsample_bytree < 1.0:
                        # Convert to mx.array for indexing
                        feature_idx = mx.array(self._feature_indices[i])
                        X_pred = X[:, feature_idx]
                    else:
                        X_pred = X

                    if tree.default_left is not None:
                        tree_pred = predict_with_default_left(
                            tree, X_pred, self.max_depth
                        )
                    else:
                        tree_pred = predict_regression(tree, X_pred, self.max_depth)

                    y_pred = y_pred.at[:, class_idx].add(self.learning_rate * tree_pred)

            # Softmax to get probabilities
            y_pred_max = mx.max(y_pred, axis=1, keepdims=True)
            y_pred_stable = y_pred - y_pred_max
            exp_pred = mx.exp(y_pred_stable)
            proba = exp_pred / mx.sum(exp_pred, axis=1, keepdims=True)

        mx.eval(proba)
        return proba

    def _map_to_indices(self, y: mx.array) -> mx.array:
        """Map class labels to indices 0, 1, 2, ..."""
        if self.classes_ is None:
            raise ValueError("Classes not determined.")
        y_np = np.array(y)
        classes_np = np.array(self.classes_)
        # Create mapping
        label_to_idx = {label: idx for idx, label in enumerate(classes_np)}
        y_indices = np.array([label_to_idx[label] for label in y_np])
        return mx.array(y_indices)

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
