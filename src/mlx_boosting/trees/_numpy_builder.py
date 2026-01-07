"""NumPy-based XGBoost tree builder for maximum speed.

This module uses numpy for all internal computations to avoid
MLX/numpy conversion overhead. Only converts to MLX at the end.
"""

import mlx.core as mx
import numpy as np

from mlx_boosting.trees._tree_structure import TreeArrays


def _compute_bin_edges_numpy(X: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute bin edges using numpy quantiles."""
    n_features = X.shape[1]
    bin_edges = np.zeros((n_features, n_bins + 1), dtype=np.float32)

    quantiles = np.linspace(0, 1, n_bins + 1)

    for f in range(n_features):
        bin_edges[f] = np.quantile(X[:, f], quantiles)
        # Adjust first and last edges
        bin_edges[f, 0] = -np.inf
        bin_edges[f, -1] = np.inf

    return bin_edges


def _bin_features_numpy(X: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Bin features using numpy digitize."""
    n_samples, n_features = X.shape
    binned = np.zeros((n_samples, n_features), dtype=np.int32)

    for f in range(n_features):
        binned[:, f] = np.digitize(X[:, f], bin_edges[f, 1:-1])

    return binned


def _compute_histogram_numpy(
    binned_X: np.ndarray,
    gradients: np.ndarray,
    hessians: np.ndarray,
    node_indices: np.ndarray,
    n_nodes: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute gradient and Hessian histograms for all nodes at once.

    Args:
        binned_X: Binned features (n_samples, n_features)
        gradients: Gradient values (n_samples,)
        hessians: Hessian values (n_samples,)
        node_indices: Which node each sample belongs to (n_samples,)
        n_nodes: Number of active nodes
        n_bins: Number of histogram bins

    Returns:
        grad_hist: (n_nodes, n_features, n_bins)
        hess_hist: (n_nodes, n_features, n_bins)
    """
    n_samples, n_features = binned_X.shape

    grad_hist = np.zeros((n_nodes, n_features, n_bins), dtype=np.float32)
    hess_hist = np.zeros((n_nodes, n_features, n_bins), dtype=np.float32)

    # Process each node
    for node in range(n_nodes):
        node_mask = node_indices == node
        if not np.any(node_mask):
            continue

        node_binned = binned_X[node_mask]
        node_grads = gradients[node_mask]
        node_hess = hessians[node_mask]

        for f in range(n_features):
            grad_hist[node, f] = np.bincount(
                node_binned[:, f], weights=node_grads, minlength=n_bins
            )[:n_bins]
            hess_hist[node, f] = np.bincount(
                node_binned[:, f], weights=node_hess, minlength=n_bins
            )[:n_bins]

    return grad_hist, hess_hist


def _compute_gains_numpy(
    grad_hist: np.ndarray,
    hess_hist: np.ndarray,
    reg_lambda: float,
    gamma: float,
) -> np.ndarray:
    """Compute XGBoost gains for all nodes/features/splits."""
    # Cumulative sums
    grad_cumsum = np.cumsum(grad_hist, axis=2)
    hess_cumsum = np.cumsum(hess_hist, axis=2)

    # Total statistics
    total_grad = grad_cumsum[:, :, -1:]
    total_hess = hess_cumsum[:, :, -1:]

    # Left/right splits
    G_L = grad_cumsum
    H_L = hess_cumsum
    G_R = total_grad - G_L
    H_R = total_hess - H_L

    # XGBoost gain formula
    eps = 1e-10
    score_L = (G_L**2) / (H_L + reg_lambda + eps)
    score_R = (G_R**2) / (H_R + reg_lambda + eps)
    score_total = (total_grad**2) / (total_hess + reg_lambda + eps)

    gains = 0.5 * (score_L + score_R - score_total) - gamma

    return gains, hess_cumsum


def _find_best_splits_numpy(
    gains: np.ndarray,
    hess_cumsum: np.ndarray,
    bin_edges: np.ndarray,
    min_child_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find best split for each node."""
    n_nodes, n_features, n_bins = gains.shape

    # Child weights
    total_hess = hess_cumsum[:, :, -1:]
    H_L = hess_cumsum
    H_R = total_hess - H_L

    # Valid splits mask
    valid_mask = (min_child_weight <= H_L) & (min_child_weight <= H_R)
    # Can't split at last bin
    valid_mask[:, :, -1] = False

    # Mask invalid gains
    masked_gains = np.where(valid_mask, gains, -np.inf)

    # Find best per node
    flat_gains = masked_gains.reshape(n_nodes, -1)
    best_flat_idx = np.argmax(flat_gains, axis=1)
    best_gains = np.max(flat_gains, axis=1)

    # Convert to feature/bin
    best_features = best_flat_idx // n_bins
    best_bins = best_flat_idx % n_bins

    # Get thresholds
    best_thresholds = np.array(
        [bin_edges[best_features[n], best_bins[n] + 1] for n in range(n_nodes)],
        dtype=np.float32,
    )

    valid_splits = best_gains > 0

    return best_features, best_thresholds, best_gains, valid_splits


def build_xgboost_tree_numpy(
    X: mx.array | np.ndarray,
    gradients: mx.array,
    hessians: mx.array,
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
    """Build XGBoost tree using numpy for all internal operations.

    Args:
        X: Features (used only if X_np not provided)
        gradients: Gradient values
        hessians: Hessian values
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        min_child_weight: Minimum sum of Hessians in child
        reg_lambda: L2 regularization
        reg_alpha: L1 regularization
        gamma: Minimum gain for split
        n_bins: Number of histogram bins
        binned_X: Pre-computed binned features (optional, for speed)
        bin_edges: Pre-computed bin edges (optional, for speed)
        X_np: Pre-computed numpy array of X (optional, for speed)

    Returns:
        TreeArrays structure containing the built tree.
    """
    # Convert gradients/hessians to numpy
    g_np = np.array(gradients, dtype=np.float32)
    h_np = np.array(hessians, dtype=np.float32)

    # Use precomputed X_np if provided, otherwise convert
    if X_np is None:
        X_np = np.array(X, dtype=np.float32)

    # Use precomputed binning if provided
    if binned_X is None or bin_edges is None:
        bin_edges = _compute_bin_edges_numpy(X_np, n_bins)
        binned_X = _bin_features_numpy(X_np, bin_edges)

    X_np_for_split = X_np

    n_samples, n_features = binned_X.shape
    max_nodes = 2 ** (max_depth + 1) - 1

    # Pre-allocate tree arrays
    feature_indices = np.full((max_nodes,), -1, dtype=np.int32)
    thresholds = np.zeros((max_nodes,), dtype=np.float32)
    left_children = np.full((max_nodes,), -1, dtype=np.int32)
    right_children = np.full((max_nodes,), -1, dtype=np.int32)
    values = np.zeros((max_nodes,), dtype=np.float32)
    is_leaf = np.ones((max_nodes,), dtype=bool)

    # Node assignments: which node each sample belongs to (-1 = done)
    node_indices = np.zeros(n_samples, dtype=np.int32)

    # Track nodes and stats
    active_nodes = [0]
    next_node_idx = 1

    node_stats = {
        0: {
            "n_samples": n_samples,
            "sum_grad": float(np.sum(g_np)),
            "sum_hess": float(np.sum(h_np)),
        }
    }

    for _depth in range(max_depth):
        if not active_nodes:
            break

        n_active = len(active_nodes)
        node_map = {node: idx for idx, node in enumerate(active_nodes)}

        # Remap node_indices to 0..n_active-1 for active nodes
        np.isin(node_indices, active_nodes)
        remapped = np.zeros_like(node_indices)
        for node, idx in node_map.items():
            remapped[node_indices == node] = idx

        # Compute histograms for all active nodes
        grad_hist, hess_hist = _compute_histogram_numpy(
            binned_X, g_np, h_np, remapped, n_active, n_bins
        )

        # Compute gains
        gains, hess_cumsum = _compute_gains_numpy(
            grad_hist, hess_hist, reg_lambda, gamma
        )

        # Find best splits
        best_features, best_thresholds_arr, best_gains, valid_splits = (
            _find_best_splits_numpy(gains, hess_cumsum, bin_edges, min_child_weight)
        )

        # Process each node
        next_level = []

        for orig_node in active_nodes:
            idx = node_map[orig_node]
            stats = node_stats[orig_node]

            # Check stopping conditions
            should_stop = (
                stats["n_samples"] < min_samples_split
                or stats["n_samples"] < 2 * min_samples_leaf
                or stats["sum_hess"] < 2 * min_child_weight
                or not valid_splits[idx]
            )

            if should_stop:
                # Make leaf
                denom = stats["sum_hess"] + reg_lambda
                if denom < 1e-10:
                    denom = 1e-10

                if reg_alpha > 0:
                    g = stats["sum_grad"]
                    if g > reg_alpha:
                        leaf_val = -(g - reg_alpha) / denom
                    elif g < -reg_alpha:
                        leaf_val = -(g + reg_alpha) / denom
                    else:
                        leaf_val = 0.0
                else:
                    leaf_val = -stats["sum_grad"] / denom

                values[orig_node] = leaf_val
                is_leaf[orig_node] = True
                node_indices[node_indices == orig_node] = -1
            else:
                # Apply split
                feat = int(best_features[idx])
                thresh = float(best_thresholds_arr[idx])

                left_idx = next_node_idx
                right_idx = next_node_idx + 1
                next_node_idx += 2

                # Update tree
                feature_indices[orig_node] = feat
                thresholds[orig_node] = thresh
                left_children[orig_node] = left_idx
                right_children[orig_node] = right_idx
                is_leaf[orig_node] = False

                # Split samples
                node_mask = node_indices == orig_node
                goes_left = X_np_for_split[node_mask, feat] <= thresh

                # Update indices
                sample_indices = np.where(node_mask)[0]
                node_indices[sample_indices[goes_left]] = left_idx
                node_indices[sample_indices[~goes_left]] = right_idx

                # Compute child stats
                left_samples = node_indices == left_idx
                right_samples = node_indices == right_idx

                node_stats[left_idx] = {
                    "n_samples": int(np.sum(left_samples)),
                    "sum_grad": float(np.sum(g_np[left_samples])),
                    "sum_hess": float(np.sum(h_np[left_samples])),
                }
                node_stats[right_idx] = {
                    "n_samples": int(np.sum(right_samples)),
                    "sum_grad": float(np.sum(g_np[right_samples])),
                    "sum_hess": float(np.sum(h_np[right_samples])),
                }

                next_level.extend([left_idx, right_idx])

        active_nodes = next_level

    # Finalize remaining nodes as leaves
    for node in active_nodes:
        stats = node_stats[node]
        denom = stats["sum_hess"] + reg_lambda
        if denom < 1e-10:
            denom = 1e-10

        if reg_alpha > 0:
            g = stats["sum_grad"]
            if g > reg_alpha:
                leaf_val = -(g - reg_alpha) / denom
            elif g < -reg_alpha:
                leaf_val = -(g + reg_alpha) / denom
            else:
                leaf_val = 0.0
        else:
            leaf_val = -stats["sum_grad"] / denom

        values[node] = leaf_val
        is_leaf[node] = True

    # Convert to MLX only at the end
    return TreeArrays(
        feature_indices=mx.array(feature_indices),
        thresholds=mx.array(thresholds),
        left_children=mx.array(left_children),
        right_children=mx.array(right_children),
        values=mx.array(values),
        is_leaf=mx.array(is_leaf),
        default_left=None,
        n_nodes=next_node_idx,
    )
