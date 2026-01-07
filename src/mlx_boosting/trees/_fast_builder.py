"""Fast XGBoost tree builder with batched operations.

This module implements a highly optimized level-wise tree builder that:
1. Uses node_assignments array instead of per-node masks
2. Batches histogram computation for all nodes at once
3. Vectorizes gain computation across all features/bins/nodes
4. Minimizes Python overhead
"""

import mlx.core as mx
import numpy as np

from mlx_boosting.trees._tree_builder import bin_features_mlx, compute_bin_edges_mlx
from mlx_boosting.trees._tree_structure import TreeArrays


def compute_max_nodes(max_depth: int) -> int:
    """Compute maximum number of nodes for a tree of given depth."""
    return 2 ** (max_depth + 1) - 1


def _compute_histograms_vectorized(
    binned_X: mx.array,
    gradients: mx.array,
    hessians: mx.array,
    node_assignments: mx.array,
    active_nodes: list[int],
    n_features: int,
    n_bins: int,
) -> tuple[mx.array, mx.array]:
    """Compute histograms for active nodes using vectorized one-hot encoding."""
    binned_X.shape[0]
    n_active = len(active_nodes)

    if n_active == 0:
        return mx.zeros((0, n_features, n_bins)), mx.zeros((0, n_features, n_bins))

    # Pre-allocate
    grad_hist = mx.zeros((n_active, n_features, n_bins), dtype=mx.float32)
    hess_hist = mx.zeros((n_active, n_features, n_bins), dtype=mx.float32)

    # Process each node
    for idx, node in enumerate(active_nodes):
        node_mask = (node_assignments == node).astype(mx.float32)  # (n_samples,)

        # Vectorized across features
        for f in range(n_features):
            bins_f = binned_X[:, f]  # (n_samples,)

            # One-hot encode bins: (n_samples, n_bins)
            bins_onehot = (bins_f[:, None] == mx.arange(n_bins)[None, :]).astype(
                mx.float32
            )

            # Weighted sum with node mask
            # grad_contrib[b] = sum_i(mask[i] * onehot[i,b] * grad[i])
            weighted_grads = (
                bins_onehot * (node_mask * gradients)[:, None]
            )  # (n_samples, n_bins)
            weighted_hess = bins_onehot * (node_mask * hessians)[:, None]

            grad_hist = grad_hist.at[idx, f, :].add(mx.sum(weighted_grads, axis=0))
            hess_hist = hess_hist.at[idx, f, :].add(mx.sum(weighted_hess, axis=0))

    return grad_hist, hess_hist


def _compute_all_gains_fast(
    grad_hist: mx.array,
    hess_hist: mx.array,
    reg_lambda: float,
    gamma: float,
) -> mx.array:
    """Compute XGBoost gains for all nodes/features/splits at once."""
    # Cumulative sums for left child statistics
    grad_cumsum = mx.cumsum(grad_hist, axis=2)  # (n_nodes, n_features, n_bins)
    hess_cumsum = mx.cumsum(hess_hist, axis=2)

    # Total statistics
    total_grad = grad_cumsum[:, :, -1:]  # (n_nodes, n_features, 1)
    total_hess = hess_cumsum[:, :, -1:]

    # Left child stats
    G_L = grad_cumsum
    H_L = hess_cumsum

    # Right child stats
    G_R = total_grad - G_L
    H_R = total_hess - H_L

    # XGBoost gain formula
    eps = 1e-10
    score_L = (G_L**2) / (H_L + reg_lambda + eps)
    score_R = (G_R**2) / (H_R + reg_lambda + eps)
    score_total = (total_grad**2) / (total_hess + reg_lambda + eps)

    gains = 0.5 * (score_L + score_R - score_total) - gamma

    return gains, hess_cumsum


def _find_best_splits_fast(
    gains: mx.array,
    hess_cumsum: mx.array,
    bin_edges: mx.array,
    min_child_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find best split for each node."""
    n_nodes, n_features, n_bins = gains.shape

    # Compute H_L and H_R for min_child_weight check
    total_hess = hess_cumsum[:, :, -1:]
    H_L = hess_cumsum
    H_R = total_hess - H_L

    # Valid splits: sufficient child weight and not last bin
    valid_mask = (min_child_weight <= H_L) & (min_child_weight <= H_R)
    last_bin_mask = mx.arange(n_bins) < (n_bins - 1)
    valid_mask = valid_mask & last_bin_mask[None, None, :]

    # Mask invalid gains
    neg_inf = mx.full(gains.shape, -1e30, dtype=gains.dtype)
    masked_gains = mx.where(valid_mask, gains, neg_inf)

    # Find best split per node
    flat_gains = masked_gains.reshape(n_nodes, -1)  # (n_nodes, n_features * n_bins)
    best_flat_idx = mx.argmax(flat_gains, axis=1)  # (n_nodes,)
    best_gains = mx.max(flat_gains, axis=1)  # (n_nodes,)

    # Convert to numpy for indexing
    mx.eval(best_flat_idx, best_gains)
    best_flat_idx_np = np.array(best_flat_idx)
    best_gains_np = np.array(best_gains)

    # Convert flat index to feature and bin
    best_features_np = best_flat_idx_np // n_bins
    best_bins_np = best_flat_idx_np % n_bins

    # Get thresholds
    bin_edges_np = np.array(bin_edges)
    best_thresholds_np = np.zeros(n_nodes, dtype=np.float32)
    for node in range(n_nodes):
        f = best_features_np[node]
        b = best_bins_np[node]
        best_thresholds_np[node] = bin_edges_np[f, b + 1]

    valid_splits_np = best_gains_np > 0

    return best_features_np, best_thresholds_np, best_gains_np, valid_splits_np


def build_xgboost_tree_fast(
    X: mx.array,
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
) -> TreeArrays:
    """Build XGBoost tree with optimized batched operations."""
    n_samples, n_features = X.shape
    max_nodes = compute_max_nodes(max_depth)

    # Pre-allocate tree arrays
    feature_indices = np.full((max_nodes,), -1, dtype=np.int32)
    thresholds = np.zeros((max_nodes,), dtype=np.float32)
    left_children = np.full((max_nodes,), -1, dtype=np.int32)
    right_children = np.full((max_nodes,), -1, dtype=np.int32)
    values = np.zeros((max_nodes,), dtype=np.float32)
    is_leaf = np.ones((max_nodes,), dtype=bool)

    # Pre-compute binning
    bin_edges = compute_bin_edges_mlx(X, n_bins)
    binned_X = bin_features_mlx(X, bin_edges)
    mx.eval(bin_edges, binned_X)

    # Node assignments array
    node_assignments = mx.zeros((n_samples,), dtype=mx.int32)

    # Track active nodes and their statistics
    active_nodes = [0]
    next_node_idx = 1

    # Pre-compute total stats
    total_grad = float(mx.sum(gradients))
    total_hess = float(mx.sum(hessians))

    node_n_samples = {0: n_samples}
    node_total_grad = {0: total_grad}
    node_total_hess = {0: total_hess}

    for _depth in range(max_depth):
        if not active_nodes:
            break

        len(active_nodes)

        # Compute histograms for all active nodes
        grad_hist, hess_hist = _compute_histograms_vectorized(
            binned_X,
            gradients,
            hessians,
            node_assignments,
            active_nodes,
            n_features,
            n_bins,
        )
        mx.eval(grad_hist, hess_hist)

        # Compute gains
        gains, hess_cumsum = _compute_all_gains_fast(
            grad_hist, hess_hist, reg_lambda, gamma
        )
        mx.eval(gains, hess_cumsum)

        # Find best splits
        best_features_np, best_thresholds_np, best_gains_np, valid_splits_np = (
            _find_best_splits_fast(gains, hess_cumsum, bin_edges, min_child_weight)
        )

        # Process splits
        next_level_nodes = []

        for idx, node in enumerate(active_nodes):
            n_node = node_n_samples[node]
            node_grad = node_total_grad[node]
            node_hess = node_total_hess[node]

            # Check stopping conditions
            should_stop = (
                n_node < min_samples_split
                or n_node < 2 * min_samples_leaf
                or node_hess < 2 * min_child_weight
                or not valid_splits_np[idx]
            )

            if should_stop:
                # Make leaf
                denom = node_hess + reg_lambda
                if denom < 1e-10:
                    denom = 1e-10

                if reg_alpha > 0:
                    if node_grad > reg_alpha:
                        leaf_val = -(node_grad - reg_alpha) / denom
                    elif node_grad < -reg_alpha:
                        leaf_val = -(node_grad + reg_alpha) / denom
                    else:
                        leaf_val = 0.0
                else:
                    leaf_val = -node_grad / denom

                values[node] = leaf_val
                is_leaf[node] = True

                # Mark samples as done
                node_mask = node_assignments == node
                node_assignments = mx.where(
                    node_mask,
                    mx.full((n_samples,), -1, dtype=mx.int32),
                    node_assignments,
                )
            else:
                # Apply split
                feat = int(best_features_np[idx])
                thresh = float(best_thresholds_np[idx])

                left_idx = next_node_idx
                right_idx = next_node_idx + 1
                next_node_idx += 2

                # Update tree structure
                feature_indices[node] = feat
                thresholds[node] = thresh
                left_children[node] = left_idx
                right_children[node] = right_idx
                is_leaf[node] = False

                # Split samples
                node_mask = node_assignments == node
                feature_values = X[:, feat]
                goes_left = feature_values <= thresh

                node_assignments = mx.where(
                    node_mask & goes_left,
                    mx.full((n_samples,), left_idx, dtype=mx.int32),
                    node_assignments,
                )
                node_assignments = mx.where(
                    node_mask & ~goes_left,
                    mx.full((n_samples,), right_idx, dtype=mx.int32),
                    node_assignments,
                )
                mx.eval(node_assignments)

                # Compute child stats
                left_mask = node_assignments == left_idx
                right_mask = node_assignments == right_idx

                n_left = int(mx.sum(left_mask.astype(mx.float32)))
                n_right = int(mx.sum(right_mask.astype(mx.float32)))

                left_grad = float(mx.sum(mx.where(left_mask, gradients, 0.0)))
                left_hess = float(mx.sum(mx.where(left_mask, hessians, 0.0)))
                right_grad = float(mx.sum(mx.where(right_mask, gradients, 0.0)))
                right_hess = float(mx.sum(mx.where(right_mask, hessians, 0.0)))

                node_n_samples[left_idx] = n_left
                node_n_samples[right_idx] = n_right
                node_total_grad[left_idx] = left_grad
                node_total_grad[right_idx] = right_grad
                node_total_hess[left_idx] = left_hess
                node_total_hess[right_idx] = right_hess

                next_level_nodes.extend([left_idx, right_idx])

        active_nodes = next_level_nodes

    # Finalize remaining active nodes as leaves
    for node in active_nodes:
        node_grad = node_total_grad[node]
        node_hess = node_total_hess[node]

        denom = node_hess + reg_lambda
        if denom < 1e-10:
            denom = 1e-10

        if reg_alpha > 0:
            if node_grad > reg_alpha:
                leaf_val = -(node_grad - reg_alpha) / denom
            elif node_grad < -reg_alpha:
                leaf_val = -(node_grad + reg_alpha) / denom
            else:
                leaf_val = 0.0
        else:
            leaf_val = -node_grad / denom

        values[node] = leaf_val
        is_leaf[node] = True

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
