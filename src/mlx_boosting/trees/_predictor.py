"""GPU-accelerated prediction functions for decision trees.

This module contains vectorized prediction implementations that process
all samples through the tree simultaneously for maximum GPU utilization.
"""

import mlx.core as mx

from mlx_boosting.trees._tree_structure import TreeArrays


def predict_regression(tree: TreeArrays, X: mx.array, max_depth: int) -> mx.array:
    """Predict regression values for all samples.

    This function traverses all samples through the tree simultaneously
    using vectorized operations for GPU efficiency.

    Args:
        tree: Fitted tree structure.
        X: Features of shape (n_samples, n_features).
        max_depth: Maximum depth to traverse.

    Returns:
        Predictions of shape (n_samples,).
    """
    n_samples = X.shape[0]

    # All samples start at root (node 0)
    current_nodes = mx.zeros((n_samples,), dtype=mx.int32)

    # Traverse tree depth by depth
    for _ in range(max_depth + 1):
        # Get split info for current nodes
        features = tree.feature_indices[current_nodes]
        thresholds = tree.thresholds[current_nodes]
        left = tree.left_children[current_nodes]
        right = tree.right_children[current_nodes]
        is_leaf = tree.is_leaf[current_nodes]

        # Gather feature values for comparison
        # Use advanced indexing: X[sample_idx, feature_idx]
        sample_indices = mx.arange(n_samples)

        # Clamp feature indices to valid range for leaf nodes
        safe_features = mx.clip(features, 0, X.shape[1] - 1)
        feature_values = X[sample_indices, safe_features]

        # Determine direction
        goes_left = feature_values <= thresholds

        # Compute next nodes
        next_nodes = mx.where(goes_left, left, right)

        # Only update non-leaf nodes (stay at leaf if already there)
        # Also handle -1 children by clamping
        next_nodes = mx.clip(next_nodes, 0, tree.feature_indices.shape[0] - 1)
        current_nodes = mx.where(is_leaf, current_nodes, next_nodes)

    # Return leaf values
    return tree.values[current_nodes]


def predict_classification_proba(
    tree: TreeArrays, X: mx.array, max_depth: int, n_classes: int
) -> mx.array:
    """Predict class probabilities for all samples.

    Args:
        tree: Fitted tree structure with class probabilities as values.
        X: Features of shape (n_samples, n_features).
        max_depth: Maximum depth to traverse.
        n_classes: Number of classes.

    Returns:
        Class probabilities of shape (n_samples, n_classes).
    """
    n_samples = X.shape[0]

    # All samples start at root
    current_nodes = mx.zeros((n_samples,), dtype=mx.int32)

    # Traverse tree
    for _ in range(max_depth + 1):
        features = tree.feature_indices[current_nodes]
        thresholds = tree.thresholds[current_nodes]
        left = tree.left_children[current_nodes]
        right = tree.right_children[current_nodes]
        is_leaf = tree.is_leaf[current_nodes]

        sample_indices = mx.arange(n_samples)
        safe_features = mx.clip(features, 0, X.shape[1] - 1)
        feature_values = X[sample_indices, safe_features]

        goes_left = feature_values <= thresholds
        next_nodes = mx.where(goes_left, left, right)
        next_nodes = mx.clip(next_nodes, 0, tree.feature_indices.shape[0] - 1)
        current_nodes = mx.where(is_leaf, current_nodes, next_nodes)

    # Return class probabilities
    # tree.values has shape (max_nodes, n_classes)
    return tree.values[current_nodes]


def predict_classification(
    tree: TreeArrays, X: mx.array, max_depth: int, n_classes: int, classes: mx.array
) -> mx.array:
    """Predict class labels for all samples.

    Args:
        tree: Fitted tree structure.
        X: Features of shape (n_samples, n_features).
        max_depth: Maximum depth to traverse.
        n_classes: Number of classes.
        classes: Array of class labels.

    Returns:
        Predicted class labels of shape (n_samples,).
    """
    proba = predict_classification_proba(tree, X, max_depth, n_classes)
    class_indices = mx.argmax(proba, axis=1)
    return classes[class_indices]


def predict_batch_compiled(
    X: mx.array,
    feature_indices: mx.array,
    thresholds: mx.array,
    left_children: mx.array,
    right_children: mx.array,
    values: mx.array,
    is_leaf: mx.array,
    max_depth: int,
) -> mx.array:
    """Batch prediction function designed for compilation.

    This function takes tree arrays as separate arguments to enable
    better JIT compilation with mx.compile.

    Args:
        X: Features of shape (n_samples, n_features).
        feature_indices: Tree feature indices array.
        thresholds: Tree thresholds array.
        left_children: Tree left children array.
        right_children: Tree right children array.
        values: Tree values array.
        is_leaf: Tree is_leaf array.
        max_depth: Maximum depth to traverse.

    Returns:
        Predictions of shape (n_samples,) or (n_samples, n_outputs).
    """
    n_samples = X.shape[0]
    max_nodes = feature_indices.shape[0]

    current_nodes = mx.zeros((n_samples,), dtype=mx.int32)

    for _ in range(max_depth + 1):
        features = feature_indices[current_nodes]
        thresh = thresholds[current_nodes]
        left = left_children[current_nodes]
        right = right_children[current_nodes]
        leaf_mask = is_leaf[current_nodes]

        sample_indices = mx.arange(n_samples)
        safe_features = mx.clip(features, 0, X.shape[1] - 1)
        feature_values = X[sample_indices, safe_features]

        goes_left = feature_values <= thresh
        next_nodes = mx.where(goes_left, left, right)
        next_nodes = mx.clip(next_nodes, 0, max_nodes - 1)
        current_nodes = mx.where(leaf_mask, current_nodes, next_nodes)

    return values[current_nodes]


# Create compiled version for repeated predictions
_compiled_predict = None


def get_compiled_predictor():
    """Get or create compiled prediction function.

    Returns:
        Compiled prediction function.
    """
    global _compiled_predict
    if _compiled_predict is None:
        _compiled_predict = mx.compile(predict_batch_compiled)
    return _compiled_predict


def predict_with_default_left(
    tree: TreeArrays, X: mx.array, max_depth: int
) -> mx.array:
    """Predict regression values with missing value handling.

    Uses the default_left array to determine which direction
    missing values (NaN) should go at each split.

    Args:
        tree: Fitted tree structure with default_left array.
        X: Features of shape (n_samples, n_features). May contain NaN.
        max_depth: Maximum depth to traverse.

    Returns:
        Predictions of shape (n_samples,).
    """
    n_samples = X.shape[0]

    # Check for NaN values
    nan_mask = mx.isnan(X)

    # All samples start at root (node 0)
    current_nodes = mx.zeros((n_samples,), dtype=mx.int32)

    # Traverse tree depth by depth
    for _ in range(max_depth + 1):
        # Get split info for current nodes
        features = tree.feature_indices[current_nodes]
        thresholds = tree.thresholds[current_nodes]
        left = tree.left_children[current_nodes]
        right = tree.right_children[current_nodes]
        is_leaf = tree.is_leaf[current_nodes]

        # Get default direction for missing values
        if tree.default_left is not None:
            default_left = tree.default_left[current_nodes]
        else:
            default_left = mx.ones((n_samples,), dtype=mx.bool_)

        # Gather feature values for comparison
        sample_indices = mx.arange(n_samples)

        # Clamp feature indices to valid range for leaf nodes
        safe_features = mx.clip(features, 0, X.shape[1] - 1)
        feature_values = X[sample_indices, safe_features]

        # Check if this feature value is missing
        is_missing = nan_mask[sample_indices, safe_features]

        # Determine direction based on threshold or default for missing
        goes_left_normal = feature_values <= thresholds

        # Use default direction for missing values
        goes_left = mx.where(is_missing, default_left, goes_left_normal)

        # Compute next nodes
        next_nodes = mx.where(goes_left, left, right)

        # Only update non-leaf nodes (stay at leaf if already there)
        next_nodes = mx.clip(next_nodes, 0, tree.feature_indices.shape[0] - 1)
        current_nodes = mx.where(is_leaf, current_nodes, next_nodes)

    # Return leaf values
    return tree.values[current_nodes]
