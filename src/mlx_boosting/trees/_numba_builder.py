"""Numba JIT-compiled XGBoost tree builder for maximum performance.

This module provides highly optimized tree building functions using Numba's
JIT compilation to achieve near-native performance.
"""

import mlx.core as mx
import numpy as np
from numba import njit, prange

from mlx_boosting.trees._tree_structure import TreeArrays


@njit(cache=True)
def _compute_bin_edges_numba(X: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute bin edges for histogram binning."""
    n_samples, n_features = X.shape
    bin_edges = np.empty((n_features, n_bins + 1), dtype=np.float32)

    for f in range(n_features):
        col = X[:, f]
        min_val = np.min(col)
        max_val = np.max(col)

        # Create uniform bins
        step = (max_val - min_val) / n_bins
        for b in range(n_bins + 1):
            bin_edges[f, b] = min_val + b * step

    return bin_edges


@njit(parallel=True, cache=True)
def _bin_features_numba(X: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Bin features into histogram bins using binary search."""
    n_samples, n_features = X.shape
    n_bins = bin_edges.shape[1] - 1
    binned = np.empty((n_samples, n_features), dtype=np.uint8)

    # Parallel over features for better cache usage
    for f in prange(n_features):
        edges = bin_edges[f]  # All edges for this feature
        for i in range(n_samples):
            val = X[i, f]
            # Binary search for the right bin
            lo, hi = 0, n_bins
            while lo < hi:
                mid = (lo + hi) // 2
                if val > edges[mid]:
                    lo = mid + 1
                else:
                    hi = mid
            # Clamp to valid range
            bin_idx = min(max(lo - 1, 0), n_bins - 1)
            binned[i, f] = bin_idx

    return binned


@njit(cache=True)
def _compute_histogram_parallel(
    binned_X: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    node_indices: np.ndarray,
    n_nodes: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute gradient and Hessian histograms for all nodes.

    Single pass over samples for better cache locality.
    """
    n_samples, n_features = binned_X.shape

    # Allocate output arrays: (n_nodes, n_features, n_bins)
    grad_hist = np.zeros((n_nodes, n_features, n_bins), dtype=np.float32)
    hess_hist = np.zeros((n_nodes, n_features, n_bins), dtype=np.float32)

    # Single pass over all samples - cache friendly
    for i in range(n_samples):
        node = node_indices[i]
        if node >= 0:
            g = gradients[i]
            h = hessians[i]
            for f in range(n_features):
                bin_idx = binned_X[i, f]
                grad_hist[node, f, bin_idx] += g
                hess_hist[node, f, bin_idx] += h

    return grad_hist, hess_hist


@njit(cache=True)
def _compute_histogram_sequential(
    binned_X: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    node_indices: np.ndarray,
    n_nodes: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential histogram computation (for small data or debugging)."""
    n_samples, n_features = binned_X.shape

    grad_hist = np.zeros((n_nodes, n_features, n_bins), dtype=np.float32)
    hess_hist = np.zeros((n_nodes, n_features, n_bins), dtype=np.float32)

    for i in range(n_samples):
        node = node_indices[i]
        if node >= 0:
            g = gradients[i]
            h = hessians[i]
            for f in range(n_features):
                bin_idx = binned_X[i, f]
                grad_hist[node, f, bin_idx] += g
                hess_hist[node, f, bin_idx] += h

    return grad_hist, hess_hist


@njit(parallel=True, cache=True)
def _compute_gains_and_find_splits(
    grad_hist: np.ndarray,
    hess_hist: np.ndarray,
    reg_lambda: float,
    gamma: float,
    min_child_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute gains and find best splits for all nodes in parallel.

    Returns:
        best_features: Best feature index for each node
        best_thresholds: Best threshold (bin index) for each node
        best_gains: Best gain for each node
        valid_splits: Whether each node has a valid split
    """
    n_nodes, n_features, n_bins = grad_hist.shape

    best_features = np.zeros(n_nodes, dtype=np.int32)
    best_thresholds = np.zeros(n_nodes, dtype=np.int32)
    best_gains = np.full(n_nodes, -np.inf, dtype=np.float32)
    valid_splits = np.zeros(n_nodes, dtype=np.bool_)

    # Parallel over nodes
    for node in prange(n_nodes):
        node_best_gain = -np.inf
        node_best_feature = 0
        node_best_threshold = 0

        for f in range(n_features):
            # Compute cumulative sums for this feature
            total_grad = 0.0
            total_hess = 0.0
            for b in range(n_bins):
                total_grad += grad_hist[node, f, b]
                total_hess += hess_hist[node, f, b]

            # Skip if not enough samples
            if total_hess < min_child_weight:
                continue

            # Parent score
            parent_score = (total_grad * total_grad) / (total_hess + reg_lambda)

            # Try each split point
            left_grad = 0.0
            left_hess = 0.0

            for b in range(n_bins - 1):  # Don't split after last bin
                left_grad += grad_hist[node, f, b]
                left_hess += hess_hist[node, f, b]
                right_grad = total_grad - left_grad
                right_hess = total_hess - left_hess

                # Check min_child_weight constraint
                if left_hess < min_child_weight or right_hess < min_child_weight:
                    continue

                # Compute gain
                left_score = (left_grad * left_grad) / (left_hess + reg_lambda)
                right_score = (right_grad * right_grad) / (right_hess + reg_lambda)
                gain = 0.5 * (left_score + right_score - parent_score) - gamma

                if gain > node_best_gain:
                    node_best_gain = gain
                    node_best_feature = f
                    node_best_threshold = b

        best_gains[node] = node_best_gain
        best_features[node] = node_best_feature
        best_thresholds[node] = node_best_threshold
        valid_splits[node] = node_best_gain > 0

    return best_features, best_thresholds, best_gains, valid_splits


@njit(cache=True)
def _compute_leaf_values(
    gradients: np.ndarray,
    hessians: np.ndarray,
    node_indices: np.ndarray,
    n_nodes: int,
    reg_lambda: float,
) -> np.ndarray:
    """Compute optimal leaf values for all nodes."""
    grad_sums = np.zeros(n_nodes, dtype=np.float32)
    hess_sums = np.zeros(n_nodes, dtype=np.float32)

    for i in range(len(gradients)):
        node = node_indices[i]
        if node >= 0:
            grad_sums[node] += gradients[i]
            hess_sums[node] += hessians[i]

    values = np.zeros(n_nodes, dtype=np.float32)
    for node in range(n_nodes):
        denom = hess_sums[node] + reg_lambda
        if denom > 1e-10:
            values[node] = -grad_sums[node] / denom

    return values


@njit(cache=True)
def _split_samples(
    X: np.ndarray,
    node_indices: np.ndarray,
    splits: np.ndarray,
    bin_edges: np.ndarray,
    max_node_id: int,
) -> np.ndarray:
    """Split samples into left/right children based on split decisions.

    Optimized with node lookup table for O(n_samples) complexity.
    """
    n_samples = len(node_indices)
    n_splits = len(splits)

    # Build lookup tables: node_id -> split info
    # -1 means no split for this node
    node_features = np.full(max_node_id + 1, -1, dtype=np.int32)
    node_thresholds = np.zeros(max_node_id + 1, dtype=np.float32)
    node_left = np.zeros(max_node_id + 1, dtype=np.int32)
    node_right = np.zeros(max_node_id + 1, dtype=np.int32)

    for s in range(n_splits):
        node_id = int(splits[s, 0])
        feature = int(splits[s, 1])
        threshold_bin = int(splits[s, 2])
        left_child = int(splits[s, 3])
        right_child = int(splits[s, 4])

        node_features[node_id] = feature
        node_thresholds[node_id] = bin_edges[feature, threshold_bin + 1]
        node_left[node_id] = left_child
        node_right[node_id] = right_child

    # Single pass over samples
    new_indices = node_indices.copy()
    for i in range(n_samples):
        node = node_indices[i]
        if node >= 0 and node_features[node] >= 0:
            feature = node_features[node]
            if X[i, feature] <= node_thresholds[node]:
                new_indices[i] = node_left[node]
            else:
                new_indices[i] = node_right[node]

    return new_indices


@njit(parallel=True, cache=True)
def _compute_histograms_parallel(
    binned_X: np.ndarray,
    g_np: np.ndarray,
    h_np: np.ndarray,
    compact_indices: np.ndarray,
    n_active: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute histograms in parallel over features."""
    n_samples, n_features = binned_X.shape

    grad_hist = np.zeros((n_active, n_features, n_bins), dtype=np.float32)
    hess_hist = np.zeros((n_active, n_features, n_bins), dtype=np.float32)

    # Parallel over features - each feature's histogram is independent
    for f in prange(n_features):
        for i in range(n_samples):
            cidx = compact_indices[i]
            if cidx >= 0:
                bin_idx = binned_X[i, f]
                grad_hist[cidx, f, bin_idx] += g_np[i]
                hess_hist[cidx, f, bin_idx] += h_np[i]

    return grad_hist, hess_hist


@njit(parallel=True, cache=True)
def _compute_hist_direct(
    binned_X: np.ndarray,
    g_np: np.ndarray,
    h_np: np.ndarray,
    compact_indices: np.ndarray,
    needs_compute: np.ndarray,
    grad_hist: np.ndarray,
    hess_hist: np.ndarray,
) -> None:
    """Compute histograms in parallel for nodes that need direct computation."""
    n_samples, n_features = binned_X.shape

    # Parallel over features
    for f in prange(n_features):
        for i in range(n_samples):
            cidx = compact_indices[i]
            if cidx >= 0 and needs_compute[cidx]:
                bin_idx = binned_X[i, f]
                grad_hist[cidx, f, bin_idx] += g_np[i]
                hess_hist[cidx, f, bin_idx] += h_np[i]


@njit(cache=True)
def _build_tree_core(
    X_np: np.ndarray,
    binned_X: np.ndarray,
    g_np: np.ndarray,
    h_np: np.ndarray,
    bin_edges: np.ndarray,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    min_child_weight: float,
    reg_lambda: float,
    gamma: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Core tree building logic with histogram subtraction optimization."""
    n_samples, n_features = X_np.shape
    max_nodes = 2 ** (max_depth + 1) - 1

    # Pre-allocate tree arrays
    feature_indices = np.full(max_nodes, -1, dtype=np.int32)
    thresholds = np.zeros(max_nodes, dtype=np.float32)
    left_children = np.full(max_nodes, -1, dtype=np.int32)
    right_children = np.full(max_nodes, -1, dtype=np.int32)
    values = np.zeros(max_nodes, dtype=np.float32)
    is_leaf = np.ones(max_nodes, dtype=np.bool_)

    # Parent tracking for histogram subtraction
    node_parent = np.full(max_nodes, -1, dtype=np.int32)
    node_is_left = np.zeros(max_nodes, dtype=np.bool_)

    # Saved histograms for subtraction (only need to save parent histograms)
    saved_grad_hist = np.zeros((max_nodes, n_features, n_bins), dtype=np.float32)
    saved_hess_hist = np.zeros((max_nodes, n_features, n_bins), dtype=np.float32)

    # Node assignments
    node_indices = np.zeros(n_samples, dtype=np.int32)

    next_node_id = 1
    n_nodes_created = 1

    # Build level by level
    for _depth in range(max_depth):
        # Find unique active nodes and count samples per node
        node_counts = np.zeros(max_nodes, dtype=np.int32)
        for i in range(n_samples):
            node = node_indices[i]
            if node >= 0:
                node_counts[node] += 1

        # Collect active nodes
        active_nodes = np.zeros(max_nodes, dtype=np.int32)
        n_active = 0
        for node in range(max_nodes):
            if node_counts[node] > 0:
                active_nodes[n_active] = node
                n_active += 1

        if n_active == 0:
            break

        # Create compact index mapping
        node_to_compact = np.full(max_nodes, -1, dtype=np.int32)
        for idx in range(n_active):
            node_to_compact[active_nodes[idx]] = idx

        compact_indices = np.empty(n_samples, dtype=np.int32)
        for i in range(n_samples):
            node = node_indices[i]
            if node >= 0:
                compact_indices[i] = node_to_compact[node]
            else:
                compact_indices[i] = -1

        # Compute histograms with subtraction trick
        grad_hist = np.zeros((n_active, n_features, n_bins), dtype=np.float32)
        hess_hist = np.zeros((n_active, n_features, n_bins), dtype=np.float32)

        # Track which nodes need direct computation vs subtraction
        needs_compute = np.ones(n_active, dtype=np.bool_)

        for idx in range(n_active):
            node = active_nodes[idx]
            parent = node_parent[node]

            # Use subtraction for right children (parent histogram - left sibling)
            if parent >= 0 and not node_is_left[node]:
                left_sibling = left_children[parent]
                # Subtract: right = parent - left
                for f in range(n_features):
                    for b in range(n_bins):
                        grad_hist[idx, f, b] = (
                            saved_grad_hist[parent, f, b]
                            - saved_grad_hist[left_sibling, f, b]
                        )
                        hess_hist[idx, f, b] = (
                            saved_hess_hist[parent, f, b]
                            - saved_hess_hist[left_sibling, f, b]
                        )
                needs_compute[idx] = False

        # Direct computation for nodes that need it (root and left children)
        # Call parallel histogram function
        _compute_hist_direct(
            binned_X, g_np, h_np, compact_indices, needs_compute, grad_hist, hess_hist
        )

        # Find best splits and process them
        any_split = False
        for idx in range(n_active):
            orig_node = active_nodes[idx]
            n_node_samples = node_counts[orig_node]

            if n_node_samples < min_samples_split:
                is_leaf[orig_node] = True
                continue

            # Find best split for this node
            best_gain = -1e10
            best_feature = 0
            best_bin = 0

            for f in range(n_features):
                total_grad = 0.0
                total_hess = 0.0
                for b in range(n_bins):
                    total_grad += grad_hist[idx, f, b]
                    total_hess += hess_hist[idx, f, b]

                if total_hess < min_child_weight:
                    continue

                parent_score = (total_grad * total_grad) / (total_hess + reg_lambda)

                left_grad = 0.0
                left_hess = 0.0
                for b in range(n_bins - 1):
                    left_grad += grad_hist[idx, f, b]
                    left_hess += hess_hist[idx, f, b]
                    right_grad = total_grad - left_grad
                    right_hess = total_hess - left_hess

                    if left_hess < min_child_weight or right_hess < min_child_weight:
                        continue

                    left_score = (left_grad * left_grad) / (left_hess + reg_lambda)
                    right_score = (right_grad * right_grad) / (right_hess + reg_lambda)
                    gain = 0.5 * (left_score + right_score - parent_score) - gamma

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = f
                        best_bin = b

            if best_gain <= 0:
                is_leaf[orig_node] = True
                continue

            # Get threshold
            thresh = bin_edges[best_feature, best_bin + 1]

            # Count left/right samples
            n_left = 0
            n_right = 0
            for i in range(n_samples):
                if node_indices[i] == orig_node:
                    if X_np[i, best_feature] <= thresh:
                        n_left += 1
                    else:
                        n_right += 1

            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                is_leaf[orig_node] = True
                continue

            # Save histogram for this node (will be parent of children)
            for f in range(n_features):
                for b in range(n_bins):
                    saved_grad_hist[orig_node, f, b] = grad_hist[idx, f, b]
                    saved_hess_hist[orig_node, f, b] = hess_hist[idx, f, b]

            # Create split
            left_idx = next_node_id
            right_idx = next_node_id + 1
            next_node_id += 2
            n_nodes_created += 2

            feature_indices[orig_node] = best_feature
            thresholds[orig_node] = thresh
            left_children[orig_node] = left_idx
            right_children[orig_node] = right_idx
            is_leaf[orig_node] = False

            # Track parent-child relationships
            node_parent[left_idx] = orig_node
            node_parent[right_idx] = orig_node
            node_is_left[left_idx] = True
            node_is_left[right_idx] = False

            # Update sample assignments
            for i in range(n_samples):
                if node_indices[i] == orig_node:
                    if X_np[i, best_feature] <= thresh:
                        node_indices[i] = left_idx
                    else:
                        node_indices[i] = right_idx

            any_split = True

        if not any_split:
            break

    # Compute leaf values
    grad_sums = np.zeros(max_nodes, dtype=np.float32)
    hess_sums = np.zeros(max_nodes, dtype=np.float32)
    for i in range(n_samples):
        node = node_indices[i]
        if node >= 0:
            grad_sums[node] += g_np[i]
            hess_sums[node] += h_np[i]

    for node in range(n_nodes_created):
        if is_leaf[node]:
            denom = hess_sums[node] + reg_lambda
            if denom > 1e-10:
                values[node] = -grad_sums[node] / denom

    return (
        feature_indices,
        thresholds,
        left_children,
        right_children,
        values,
        is_leaf,
        n_nodes_created,
    )


def build_xgboost_tree_numba(
    X: mx.array | np.ndarray,
    gradients: mx.array | np.ndarray,
    hessians: mx.array | np.ndarray,
    max_depth: int = 6,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    min_child_weight: float = 1.0,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    gamma: float = 0.0,
    n_bins: int = 256,
    binned_X: np.ndarray | None = None,
    bin_edges: np.ndarray | None = None,
    X_np: np.ndarray | None = None,
) -> TreeArrays:
    """Build XGBoost tree using Numba JIT-compiled functions."""
    _ = reg_alpha  # Not used

    # Convert inputs to numpy
    if X_np is None:
        if isinstance(X, mx.array):
            X_np = np.array(X, dtype=np.float32)
        else:
            X_np = np.ascontiguousarray(X, dtype=np.float32)

    if isinstance(gradients, mx.array):
        g_np = np.array(gradients, dtype=np.float32)
    else:
        g_np = np.ascontiguousarray(gradients, dtype=np.float32)

    if isinstance(hessians, mx.array):
        h_np = np.array(hessians, dtype=np.float32)
    else:
        h_np = np.ascontiguousarray(hessians, dtype=np.float32)

    # Compute binning if not provided
    if binned_X is None or bin_edges is None:
        bin_edges = _compute_bin_edges_numba(X_np, n_bins)
        binned_X = _bin_features_numba(X_np, bin_edges)

    # Call the JIT-compiled core
    (
        feature_indices,
        thresholds,
        left_children,
        right_children,
        values,
        is_leaf,
        n_nodes,
    ) = _build_tree_core(
        X_np,
        binned_X,
        g_np,
        h_np,
        bin_edges,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_child_weight,
        reg_lambda,
        gamma,
        n_bins,
    )

    return TreeArrays(
        feature_indices=mx.array(feature_indices[:n_nodes]),
        thresholds=mx.array(thresholds[:n_nodes]),
        left_children=mx.array(left_children[:n_nodes]),
        right_children=mx.array(right_children[:n_nodes]),
        values=mx.array(values[:n_nodes]),
        is_leaf=mx.array(is_leaf[:n_nodes]),
        n_nodes=n_nodes,
    )


# Pre-compile functions on import (optional warmup)
def _warmup_numba():
    """Warmup Numba JIT compilation."""
    X = np.random.randn(100, 10).astype(np.float32)
    g = np.random.randn(100).astype(np.float32)
    h = np.ones(100, dtype=np.float32)

    edges = _compute_bin_edges_numba(X, 32)
    binned = _bin_features_numba(X, edges)
    node_idx = np.zeros(100, dtype=np.int32)

    _compute_histogram_parallel(binned, g, h, node_idx, 1, 32)
    _compute_histogram_sequential(binned, g, h, node_idx, 1, 32)
