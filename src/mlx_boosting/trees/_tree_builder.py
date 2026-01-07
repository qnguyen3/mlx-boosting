"""GPU-accelerated iterative tree building using MLX.

This module implements breadth-first tree construction that processes
nodes level by level, enabling efficient GPU utilization.
"""

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import numpy as np

from mlx_boosting.trees._split_finder import (
    DEFAULT_N_BINS,
    HistogramData,
    SplitInfo,
    XGBoostHistogramData,
    compute_batch_histogram_counts,
    compute_batch_histogram_sums,
    compute_batch_variance_gains,
    compute_child_histograms_smart,
    compute_histogram_counts_fast,
    compute_histogram_data,
    compute_leaf_value,
    compute_xgboost_child_histograms_smart,
    compute_xgboost_histogram_data,
    compute_xgboost_leaf_value,
    find_batch_best_splits,
    find_best_split_exact,
    find_best_split_histogram,
    find_best_split_with_histogram,
    find_xgboost_best_split,
    find_xgboost_best_split_with_histogram,
    find_xgboost_best_split_with_missing,
)
from mlx_boosting.trees._tree_structure import (
    TreeArrays,
    compute_max_nodes,
)


@dataclass
class NodeInfo:
    """Information about a node to be processed.

    Attributes:
        node_idx: Index of this node in the tree arrays.
        depth: Depth of this node (root = 0).
        sample_mask: Boolean mask of samples belonging to this node.
        histogram: Cached histogram data for histogram subtraction optimization.
    """

    node_idx: int
    depth: int
    sample_mask: mx.array
    histogram: HistogramData | None = None


def compute_bin_edges_mlx(X: mx.array, n_bins: int) -> mx.array:
    """Compute histogram bin edges using MLX (GPU-accelerated).

    Uses sort + evenly-spaced indexing as a percentile approximation.
    Edges are computed as midpoints between consecutive sorted values
    to ensure splits separate samples correctly.

    Args:
        X: Features of shape (n_samples, n_features).
        n_bins: Number of bins per feature.

    Returns:
        Bin edges of shape (n_features, n_bins + 1).
    """
    n_samples, n_features = X.shape

    # Sort each feature column
    X_sorted = mx.sort(X, axis=0)  # (n_samples, n_features)

    # Compute indices for percentile positions
    # linspace(0, n_samples-1, n_bins+1) gives evenly spaced indices
    percentile_positions = mx.linspace(0, n_samples - 1, n_bins + 1)
    indices = mx.clip(percentile_positions.astype(mx.int32), 0, n_samples - 1)

    # Take values at those positions: (n_bins+1, n_features)
    bin_edges = mx.take(X_sorted, indices, axis=0)

    # Compute midpoints between consecutive edges for interior edges
    # This ensures splits separate samples correctly
    # First edge stays at min, last edge stays at max
    # Interior edges become midpoints
    left_edges = bin_edges[:-1, :]  # (n_bins, n_features)
    right_edges = bin_edges[1:, :]  # (n_bins, n_features)
    midpoints = (left_edges + right_edges) / 2.0  # (n_bins, n_features)

    # Reconstruct bin_edges with first edge and midpoints
    first_edge = bin_edges[:1, :]  # (1, n_features)
    bin_edges = mx.concatenate(
        [first_edge, midpoints], axis=0
    )  # (n_bins+1, n_features)

    # Transpose to (n_features, n_bins+1)
    bin_edges = mx.transpose(bin_edges)

    return bin_edges


def bin_features_mlx(X: mx.array, bin_edges: mx.array) -> mx.array:
    """Bin features using MLX vectorized comparisons (GPU-accelerated).

    Implements searchsorted logic: for each value, count how many bin edges
    it exceeds using vectorized comparisons.

    Args:
        X: Features of shape (n_samples, n_features).
        bin_edges: Bin edges of shape (n_features, n_bins + 1).

    Returns:
        Binned features of shape (n_samples, n_features), int32.
    """
    n_samples, n_features = X.shape
    n_bins = bin_edges.shape[1] - 1

    # Process all features using vectorized comparison
    # For each feature f, we compare X[:, f] against bin_edges[f, 1:]
    # to find which bin each sample falls into

    binned_list = []
    for f in range(n_features):
        # Interior edges (exclude the minimum edge): (n_bins,)
        interior_edges = bin_edges[f, 1:]
        # Feature values: (n_samples,)
        feature_vals = X[:, f]

        # Vectorized comparison: (n_samples, n_bins)
        # For each sample, count how many interior edges it exceeds
        comparison = feature_vals[:, None] >= interior_edges[None, :]

        # Sum to get bin index: (n_samples,)
        bin_indices = mx.sum(comparison.astype(mx.int32), axis=1)

        # Clamp to valid range [0, n_bins-1]
        bin_indices = mx.clip(bin_indices, 0, n_bins - 1)

        binned_list.append(bin_indices)

    # Stack all features: (n_samples, n_features)
    binned = mx.stack(binned_list, axis=1)

    return binned


def build_regression_tree(
    X: mx.array,
    y: mx.array,
    max_depth: int = 6,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    n_bins: int = DEFAULT_N_BINS,
    split_method: Literal["histogram", "exact"] = "histogram",
) -> TreeArrays:
    """Build a regression tree using iterative breadth-first approach.

    This implementation processes all nodes at each depth level,
    enabling better GPU utilization than recursive approaches.

    Args:
        X: Features of shape (n_samples, n_features).
        y: Targets of shape (n_samples,).
        max_depth: Maximum depth of the tree.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required in a leaf node.
        n_bins: Number of histogram bins for split finding.
        split_method: "histogram" for large datasets, "exact" for small.

    Returns:
        Fitted TreeArrays structure.
    """
    n_samples = X.shape[0]
    max_nodes = compute_max_nodes(max_depth)

    # Use numpy arrays during construction (mutable)
    feature_indices = np.full((max_nodes,), -1, dtype=np.int32)
    thresholds = np.zeros((max_nodes,), dtype=np.float32)
    left_children = np.full((max_nodes,), -1, dtype=np.int32)
    right_children = np.full((max_nodes,), -1, dtype=np.int32)
    values = np.zeros((max_nodes,), dtype=np.float32)
    is_leaf = np.ones((max_nodes,), dtype=bool)

    # Pre-compute binning for histogram method using MLX (GPU)
    if split_method == "histogram":
        bin_edges = compute_bin_edges_mlx(X, n_bins)
        binned_X = bin_features_mlx(X, bin_edges)
        mx.eval(bin_edges, binned_X)  # Evaluate once at start
    else:
        bin_edges = None
        binned_X = None

    # Initialize with root node
    root_mask = mx.ones((n_samples,), dtype=mx.bool_)
    nodes_to_process = [NodeInfo(node_idx=0, depth=0, sample_mask=root_mask)]

    next_node_idx = 1  # Next available node index

    # Process nodes breadth-first
    while nodes_to_process:
        current_node = nodes_to_process.pop(0)
        node_idx = current_node.node_idx
        depth = current_node.depth
        sample_mask = current_node.sample_mask

        # Count samples in this node
        n_node_samples = int(mx.sum(sample_mask.astype(mx.float32)))

        # Check stopping conditions
        should_stop = (
            depth >= max_depth
            or n_node_samples < min_samples_split
            or n_node_samples < 2 * min_samples_leaf
        )

        if should_stop:
            # Make this a leaf node
            leaf_value = compute_leaf_value(y, sample_mask)
            values[node_idx] = leaf_value
            is_leaf[node_idx] = True
            continue

        # Find best split
        if split_method == "histogram":
            split_info = find_best_split_histogram(
                X=X,
                y=y,
                sample_mask=sample_mask,
                bin_edges=bin_edges,
                binned_X=binned_X,
                n_bins=n_bins,
                min_samples_leaf=min_samples_leaf,
            )
        else:
            split_info = find_best_split_exact(
                X=X,
                y=y,
                sample_mask=sample_mask,
                min_samples_leaf=min_samples_leaf,
            )

        # Check if valid split was found
        if not split_info.is_valid or split_info.gain <= 0:
            # Make this a leaf node
            leaf_value = compute_leaf_value(y, sample_mask)
            values[node_idx] = leaf_value
            is_leaf[node_idx] = True
            continue

        # Apply split
        left_child_idx = next_node_idx
        right_child_idx = next_node_idx + 1
        next_node_idx += 2

        # Update tree arrays (numpy, mutable)
        feature_indices[node_idx] = split_info.feature
        thresholds[node_idx] = split_info.threshold
        left_children[node_idx] = left_child_idx
        right_children[node_idx] = right_child_idx
        is_leaf[node_idx] = False

        # Create child masks
        feature_values = X[:, split_info.feature]
        goes_left = feature_values <= split_info.threshold

        left_mask = sample_mask & goes_left
        right_mask = sample_mask & ~goes_left

        # Queue children for processing
        nodes_to_process.append(
            NodeInfo(node_idx=left_child_idx, depth=depth + 1, sample_mask=left_mask)
        )
        nodes_to_process.append(
            NodeInfo(node_idx=right_child_idx, depth=depth + 1, sample_mask=right_mask)
        )

    # Convert numpy arrays to MLX arrays
    return TreeArrays(
        feature_indices=mx.array(feature_indices),
        thresholds=mx.array(thresholds),
        left_children=mx.array(left_children),
        right_children=mx.array(right_children),
        values=mx.array(values),
        is_leaf=mx.array(is_leaf),
        n_nodes=next_node_idx,
    )


def build_regression_tree_levelwise(
    X: mx.array,
    y: mx.array,
    max_depth: int = 6,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    n_bins: int = DEFAULT_N_BINS,
) -> TreeArrays:
    """Build regression tree with level-wise batched processing.

    Processes all nodes at each depth level simultaneously for better
    GPU utilization. Combines level-wise batching with histogram subtraction.

    Args:
        X: Features of shape (n_samples, n_features).
        y: Targets of shape (n_samples,).
        max_depth: Maximum depth of the tree.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required in a leaf node.
        n_bins: Number of histogram bins for split finding.

    Returns:
        Fitted TreeArrays structure.
    """
    n_samples = X.shape[0]
    max_nodes = compute_max_nodes(max_depth)

    # Use numpy arrays during construction (mutable)
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

    # Compute root histogram
    root_mask = mx.ones((n_samples,), dtype=mx.bool_)
    root_histogram = compute_histogram_data(binned_X, y, root_mask, n_bins)

    # Initialize level 0 (root)
    current_level = [
        NodeInfo(node_idx=0, depth=0, sample_mask=root_mask, histogram=root_histogram)
    ]

    next_node_idx = 1

    # Process level by level
    for depth in range(max_depth + 1):
        if not current_level:
            break

        n_level_nodes = len(current_level)

        # For small levels, process individually with histogram subtraction
        if n_level_nodes <= 4:
            next_level = []
            for node in current_level:
                node_idx = node.node_idx
                sample_mask = node.sample_mask
                histogram = node.histogram

                n_node_samples = int(mx.sum(sample_mask.astype(mx.float32)))

                # Check stopping conditions
                should_stop = (
                    depth >= max_depth
                    or n_node_samples < min_samples_split
                    or n_node_samples < 2 * min_samples_leaf
                )

                if should_stop:
                    leaf_value = compute_leaf_value(y, sample_mask)
                    values[node_idx] = leaf_value
                    is_leaf[node_idx] = True
                    continue

                # Find best split using cached histogram
                if histogram is not None:
                    split_info = find_best_split_with_histogram(
                        bin_edges, histogram, n_bins, min_samples_leaf
                    )
                else:
                    split_info = find_best_split_histogram(
                        X, y, sample_mask, bin_edges, binned_X, n_bins, min_samples_leaf
                    )

                if not split_info.is_valid or split_info.gain <= 0:
                    leaf_value = compute_leaf_value(y, sample_mask)
                    values[node_idx] = leaf_value
                    is_leaf[node_idx] = True
                    continue

                # Apply split
                left_child_idx = next_node_idx
                right_child_idx = next_node_idx + 1
                next_node_idx += 2

                feature_indices[node_idx] = split_info.feature
                thresholds[node_idx] = split_info.threshold
                left_children[node_idx] = left_child_idx
                right_children[node_idx] = right_child_idx
                is_leaf[node_idx] = False

                # Create child masks
                feature_values = X[:, split_info.feature]
                goes_left = feature_values <= split_info.threshold

                # Compute child histograms using subtraction optimization
                if histogram is not None:
                    left_hist, right_hist = compute_child_histograms_smart(
                        binned_X, y, sample_mask, goes_left, histogram, n_bins
                    )
                else:
                    left_mask = sample_mask & goes_left
                    right_mask = sample_mask & ~goes_left
                    left_hist = compute_histogram_data(binned_X, y, left_mask, n_bins)
                    right_hist = compute_histogram_data(binned_X, y, right_mask, n_bins)

                left_mask = sample_mask & goes_left
                right_mask = sample_mask & ~goes_left

                next_level.append(
                    NodeInfo(left_child_idx, depth + 1, left_mask, left_hist)
                )
                next_level.append(
                    NodeInfo(right_child_idx, depth + 1, right_mask, right_hist)
                )

            current_level = next_level

        else:
            # For larger levels, use batched processing
            masks = mx.stack([node.sample_mask for node in current_level])

            # Compute histograms for all nodes at once
            sum_hist = compute_batch_histogram_sums(binned_X, y, masks, n_bins)
            count_hist = compute_batch_histogram_counts(binned_X, masks, n_bins)

            # Compute gains for all nodes at once
            gains = compute_batch_variance_gains(sum_hist, count_hist, min_samples_leaf)

            # Find best splits for all nodes
            splits = find_batch_best_splits(gains, bin_edges, n_bins)

            # Process each node's split decision
            next_level = []

            for i, (node, split) in enumerate(zip(current_level, splits, strict=False)):
                node_idx = node.node_idx
                sample_mask = node.sample_mask

                n_node_samples = int(mx.sum(sample_mask.astype(mx.float32)))

                should_stop = (
                    depth >= max_depth
                    or n_node_samples < min_samples_split
                    or n_node_samples < 2 * min_samples_leaf
                    or not split.is_valid
                    or split.gain <= 0
                )

                if should_stop:
                    leaf_value = compute_leaf_value(y, sample_mask)
                    values[node_idx] = leaf_value
                    is_leaf[node_idx] = True
                    continue

                # Apply split
                left_child_idx = next_node_idx
                right_child_idx = next_node_idx + 1
                next_node_idx += 2

                feature_indices[node_idx] = split.feature
                thresholds[node_idx] = split.threshold
                left_children[node_idx] = left_child_idx
                right_children[node_idx] = right_child_idx
                is_leaf[node_idx] = False

                # Create child masks
                feature_values = X[:, split.feature]
                goes_left = feature_values <= split.threshold

                left_mask = sample_mask & goes_left
                right_mask = sample_mask & ~goes_left

                # Build child histograms from batch data for subtraction
                node_histogram = HistogramData(
                    sum_hist=sum_hist[i],
                    count_hist=count_hist[i],
                    total_sum=mx.sum(sum_hist[i]),
                    total_count=mx.sum(count_hist[i]),
                )
                left_hist, right_hist = compute_child_histograms_smart(
                    binned_X, y, sample_mask, goes_left, node_histogram, n_bins
                )

                next_level.append(
                    NodeInfo(left_child_idx, depth + 1, left_mask, left_hist)
                )
                next_level.append(
                    NodeInfo(right_child_idx, depth + 1, right_mask, right_hist)
                )

            current_level = next_level

    # Convert numpy arrays to MLX arrays
    return TreeArrays(
        feature_indices=mx.array(feature_indices),
        thresholds=mx.array(thresholds),
        left_children=mx.array(left_children),
        right_children=mx.array(right_children),
        values=mx.array(values),
        is_leaf=mx.array(is_leaf),
        n_nodes=next_node_idx,
    )


def build_classification_tree(
    X: mx.array,
    y: mx.array,
    n_classes: int,
    max_depth: int = 6,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    n_bins: int = DEFAULT_N_BINS,
    criterion: Literal["gini", "entropy"] = "gini",
) -> TreeArrays:
    """Build a classification tree using iterative breadth-first approach.

    Args:
        X: Features of shape (n_samples, n_features).
        y: Class labels of shape (n_samples,), values in [0, n_classes).
        n_classes: Number of classes.
        max_depth: Maximum depth of the tree.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required in a leaf node.
        n_bins: Number of histogram bins for split finding.
        criterion: Split criterion ("gini" or "entropy").

    Returns:
        Fitted TreeArrays structure with class probabilities as values.
    """
    n_samples = X.shape[0]
    max_nodes = compute_max_nodes(max_depth)

    # Use numpy arrays during construction (mutable)
    feature_indices = np.full((max_nodes,), -1, dtype=np.int32)
    thresholds = np.zeros((max_nodes,), dtype=np.float32)
    left_children = np.full((max_nodes,), -1, dtype=np.int32)
    right_children = np.full((max_nodes,), -1, dtype=np.int32)
    values = np.zeros((max_nodes, n_classes), dtype=np.float32)
    is_leaf = np.ones((max_nodes,), dtype=bool)

    # Pre-compute binning using MLX (GPU)
    bin_edges = compute_bin_edges_mlx(X, n_bins)
    binned_X = bin_features_mlx(X, bin_edges)
    mx.eval(bin_edges, binned_X)  # Evaluate once at start

    # Initialize with root node
    root_mask = mx.ones((n_samples,), dtype=mx.bool_)
    nodes_to_process = [NodeInfo(node_idx=0, depth=0, sample_mask=root_mask)]

    next_node_idx = 1

    while nodes_to_process:
        current_node = nodes_to_process.pop(0)
        node_idx = current_node.node_idx
        depth = current_node.depth
        sample_mask = current_node.sample_mask

        n_node_samples = int(mx.sum(sample_mask.astype(mx.float32)))

        # Check stopping conditions
        should_stop = (
            depth >= max_depth
            or n_node_samples < min_samples_split
            or n_node_samples < 2 * min_samples_leaf
        )

        # Compute class distribution for this node (GPU)
        class_probs = compute_class_distribution_mlx(y, sample_mask, n_classes)
        mx.eval(class_probs)

        # Check if pure node (all same class)
        if not should_stop:
            is_pure = float(mx.max(class_probs)) > 0.999
            should_stop = should_stop or is_pure

        if should_stop:
            # Make this a leaf node with class probabilities
            values[node_idx] = np.array(class_probs)
            is_leaf[node_idx] = True
            continue

        # Find best split using classification criterion (GPU)
        split_info = find_best_classification_split_mlx(
            X=X,
            y=y,
            sample_mask=sample_mask,
            bin_edges=bin_edges,
            binned_X=binned_X,
            n_bins=n_bins,
            n_classes=n_classes,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
        )

        if not split_info.is_valid or split_info.gain <= 0:
            values[node_idx] = np.array(class_probs)
            is_leaf[node_idx] = True
            continue

        # Apply split
        left_child_idx = next_node_idx
        right_child_idx = next_node_idx + 1
        next_node_idx += 2

        feature_indices[node_idx] = split_info.feature
        thresholds[node_idx] = split_info.threshold
        left_children[node_idx] = left_child_idx
        right_children[node_idx] = right_child_idx
        is_leaf[node_idx] = False

        feature_values = X[:, split_info.feature]
        goes_left = feature_values <= split_info.threshold

        left_mask = sample_mask & goes_left
        right_mask = sample_mask & ~goes_left

        nodes_to_process.append(
            NodeInfo(node_idx=left_child_idx, depth=depth + 1, sample_mask=left_mask)
        )
        nodes_to_process.append(
            NodeInfo(node_idx=right_child_idx, depth=depth + 1, sample_mask=right_mask)
        )

    # Convert to MLX arrays
    return TreeArrays(
        feature_indices=mx.array(feature_indices),
        thresholds=mx.array(thresholds),
        left_children=mx.array(left_children),
        right_children=mx.array(right_children),
        values=mx.array(values),
        is_leaf=mx.array(is_leaf),
        n_nodes=next_node_idx,
    )


def compute_class_distribution_mlx(
    y: mx.array, sample_mask: mx.array, n_classes: int
) -> mx.array:
    """Compute class probability distribution for a leaf node (GPU).

    Args:
        y: Class labels of shape (n_samples,).
        sample_mask: Boolean mask for samples in this node.
        n_classes: Number of classes.

    Returns:
        Class probabilities of shape (n_classes,).
    """
    # Create class range: (n_classes,)
    class_range = mx.arange(n_classes)

    # Broadcast comparison: y (n_samples, 1) == class_range (1, n_classes)
    # Result: (n_samples, n_classes) bool
    class_matches = y[:, None] == class_range[None, :]

    # Apply sample mask: (n_samples, 1)
    valid_matches = class_matches & sample_mask[:, None]

    # Sum over samples to get counts: (n_classes,)
    counts = mx.sum(valid_matches.astype(mx.float32), axis=0)

    total = mx.sum(counts)
    probs = counts / mx.maximum(total, 1.0)

    return probs


def find_best_classification_split_mlx(
    X: mx.array,
    y: mx.array,
    sample_mask: mx.array,
    bin_edges: mx.array,
    binned_X: mx.array,
    n_bins: int,
    n_classes: int,
    min_samples_leaf: int,
    criterion: Literal["gini", "entropy"],
) -> SplitInfo:
    """Find best split for classification using Gini or entropy (GPU).

    All computations run on GPU using MLX vectorized operations.

    Args:
        X: Features of shape (n_samples, n_features).
        y: Class labels of shape (n_samples,).
        sample_mask: Boolean mask for samples in this node.
        bin_edges: Pre-computed bin edges, shape (n_features, n_bins+1).
        binned_X: Pre-computed binned features, shape (n_samples, n_features).
        n_bins: Number of bins.
        n_classes: Number of classes.
        min_samples_leaf: Minimum samples in leaf.
        criterion: "gini" or "entropy".

    Returns:
        SplitInfo with best split.
    """
    # Compute class histogram for each bin for each feature
    # Shape: (n_features, n_bins, n_classes)
    class_hist = compute_class_histogram_fast(
        binned_X, y, sample_mask, n_bins, n_classes
    )

    # Cumulative sums for left side: (n_features, n_bins, n_classes)
    left_class_counts = mx.cumsum(class_hist, axis=1)

    # Total class counts per feature: (n_features, 1, n_classes)
    total_class_counts = mx.sum(class_hist, axis=1, keepdims=True)

    # Right side counts
    right_class_counts = total_class_counts - left_class_counts

    # Total samples per side: (n_features, n_bins)
    left_counts = mx.sum(left_class_counts, axis=2)
    right_counts = mx.sum(right_class_counts, axis=2)

    # Compute impurity
    if criterion == "gini":
        left_impurity = compute_gini_mlx(left_class_counts, left_counts)
        right_impurity = compute_gini_mlx(right_class_counts, right_counts)
    else:
        left_impurity = compute_entropy_mlx(left_class_counts, left_counts)
        right_impurity = compute_entropy_mlx(right_class_counts, right_counts)

    # Parent impurity
    parent_class_dist = compute_class_distribution_mlx(y, sample_mask, n_classes)

    if criterion == "gini":
        parent_impurity = 1.0 - mx.sum(parent_class_dist**2)
    else:
        parent_impurity = -mx.sum(
            mx.where(
                parent_class_dist > 0,
                parent_class_dist * mx.log(parent_class_dist + 1e-10),
                0.0,
            )
        )

    # Information gain = parent_impurity - weighted_child_impurity
    total = left_counts + right_counts
    weighted_impurity = (
        left_counts * left_impurity + right_counts * right_impurity
    ) / mx.maximum(total, 1.0)

    gains = parent_impurity - weighted_impurity

    # Invalidate splits with too few samples
    valid_mask = (left_counts >= min_samples_leaf) & (right_counts >= min_samples_leaf)
    gains = mx.where(valid_mask, gains, -mx.inf)

    # Evaluate and find best
    mx.eval(gains)

    best_gain = float(mx.max(gains))

    if best_gain <= 0 or np.isinf(best_gain):
        return SplitInfo(feature=-1, threshold=0.0, gain=best_gain, is_valid=False)

    flat_gains = gains.flatten()
    flat_idx = int(mx.argmax(flat_gains))
    best_feature = flat_idx // n_bins
    best_bin = flat_idx % n_bins
    best_threshold = float(bin_edges[best_feature, best_bin + 1])

    return SplitInfo(
        feature=best_feature,
        threshold=best_threshold,
        gain=best_gain,
        is_valid=True,
    )


def compute_class_histogram_fast(
    binned_X: mx.array,
    y: mx.array,
    sample_mask: mx.array,
    n_bins: int,
    n_classes: int,
) -> mx.array:
    """Compute class counts per bin per feature using numpy for speed.

    Uses numpy's optimized bincount for histogram computation, which is
    much faster than MLX broadcasting for this operation.

    Args:
        binned_X: Binned features, shape (n_samples, n_features).
        y: Class labels, shape (n_samples,).
        sample_mask: Boolean mask for samples in this node.
        n_bins: Number of bins.
        n_classes: Number of classes.

    Returns:
        Class histogram of shape (n_features, n_bins, n_classes).
    """
    # Convert to numpy for fast histogram computation
    binned_np = np.array(binned_X)
    y_np = np.array(y)
    mask_np = np.array(sample_mask)

    n_samples, n_features = binned_np.shape

    # Get valid sample indices
    valid_indices = np.where(mask_np)[0]
    y_valid = y_np[valid_indices]

    # Compute histogram using bincount (much faster than broadcasting)
    class_hist = np.zeros((n_features, n_bins, n_classes), dtype=np.float32)

    for c in range(n_classes):
        class_samples = valid_indices[y_valid == c]
        if len(class_samples) > 0:
            binned_class = binned_np[class_samples]
            for f in range(n_features):
                counts = np.bincount(binned_class[:, f], minlength=n_bins)
                class_hist[f, :, c] = counts[:n_bins]

    return mx.array(class_hist)


def compute_gini_mlx(class_counts: mx.array, total_counts: mx.array) -> mx.array:
    """Compute Gini impurity (GPU).

    Args:
        class_counts: Shape (..., n_classes).
        total_counts: Shape (...).

    Returns:
        Gini impurity values.
    """
    probs = class_counts / mx.maximum(total_counts[..., None], 1.0)
    gini = 1.0 - mx.sum(probs**2, axis=-1)
    return gini


def compute_entropy_mlx(class_counts: mx.array, total_counts: mx.array) -> mx.array:
    """Compute entropy (GPU).

    Args:
        class_counts: Shape (..., n_classes).
        total_counts: Shape (...).

    Returns:
        Entropy values.
    """
    probs = class_counts / mx.maximum(total_counts[..., None], 1.0)
    log_probs = mx.where(probs > 0, mx.log(probs + 1e-10), 0.0)
    entropy = -mx.sum(probs * log_probs, axis=-1)
    return entropy


# =============================================================================
# XGBoost Tree Builder
# =============================================================================


@dataclass
class XGBoostNodeInfo:
    """Information about an XGBoost node to be processed.

    Attributes:
        node_idx: Index of this node in the tree arrays.
        depth: Depth of this node (root = 0).
        sample_mask: Boolean mask of samples belonging to this node.
        histogram: Cached XGBoost histogram data for histogram subtraction.
    """

    node_idx: int
    depth: int
    sample_mask: mx.array
    histogram: XGBoostHistogramData | None = None


def build_xgboost_tree_levelwise(
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
    n_bins: int = DEFAULT_N_BINS,
    colsample_bylevel: float = 1.0,
    colsample_bynode: float = 1.0,
    nan_mask: mx.array | None = None,
) -> TreeArrays:
    """Build XGBoost tree with level-wise batched processing.

    Uses second-order optimization (gradient and Hessian), regularization,
    and supports missing value handling.

    Args:
        X: Features of shape (n_samples, n_features).
        gradients: First-order derivatives, shape (n_samples,).
        hessians: Second-order derivatives, shape (n_samples,).
        max_depth: Maximum depth of the tree.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required in a leaf node.
        min_child_weight: Minimum sum of Hessians in each child.
        reg_lambda: L2 regularization parameter.
        reg_alpha: L1 regularization parameter.
        gamma: Minimum loss reduction for split (pruning threshold).
        n_bins: Number of histogram bins for split finding.
        colsample_bylevel: Fraction of features to sample per level.
        colsample_bynode: Fraction of features to sample per node.
        nan_mask: Boolean mask where True indicates NaN values,
            shape (n_samples, n_features). If None, no missing value handling.

    Returns:
        Fitted TreeArrays structure with default_left for missing values.
    """
    n_samples, n_features = X.shape
    max_nodes = compute_max_nodes(max_depth)

    # Use numpy arrays during construction (mutable)
    feature_indices = np.full((max_nodes,), -1, dtype=np.int32)
    thresholds = np.zeros((max_nodes,), dtype=np.float32)
    left_children = np.full((max_nodes,), -1, dtype=np.int32)
    right_children = np.full((max_nodes,), -1, dtype=np.int32)
    values = np.zeros((max_nodes,), dtype=np.float32)
    is_leaf = np.ones((max_nodes,), dtype=bool)
    default_left = np.ones((max_nodes,), dtype=bool)

    # Pre-compute binning
    bin_edges = compute_bin_edges_mlx(X, n_bins)
    binned_X = bin_features_mlx(X, bin_edges)
    mx.eval(bin_edges, binned_X)

    # Compute root histogram
    root_mask = mx.ones((n_samples,), dtype=mx.bool_)
    root_histogram = compute_xgboost_histogram_data(
        binned_X, gradients, hessians, root_mask, n_bins
    )

    # Initialize level 0 (root)
    current_level = [
        XGBoostNodeInfo(
            node_idx=0, depth=0, sample_mask=root_mask, histogram=root_histogram
        )
    ]

    next_node_idx = 1
    has_missing = nan_mask is not None

    # Process level by level
    for depth in range(max_depth + 1):
        if not current_level:
            break

        len(current_level)

        # Sample features for this level
        if colsample_bylevel < 1.0:
            n_features_level = max(1, int(n_features * colsample_bylevel))
            level_features = np.random.choice(
                n_features, n_features_level, replace=False
            )
            level_features = mx.array(level_features)
        else:
            level_features = None

        # Process nodes individually (histogram subtraction works best this way)
        next_level = []
        for node in current_level:
            node_idx = node.node_idx
            sample_mask = node.sample_mask
            histogram = node.histogram

            n_node_samples = int(mx.sum(sample_mask.astype(mx.float32)))

            # Get total Hessian for this node
            total_hess = (
                float(histogram.total_hess)
                if histogram
                else float(mx.sum(mx.where(sample_mask, hessians, 0.0)))
            )

            # Check stopping conditions
            should_stop = (
                depth >= max_depth
                or n_node_samples < min_samples_split
                or n_node_samples < 2 * min_samples_leaf
                or total_hess < 2 * min_child_weight
            )

            if should_stop:
                if histogram is not None:
                    leaf_value = compute_xgboost_leaf_value(
                        histogram.total_grad,
                        histogram.total_hess,
                        reg_lambda,
                        reg_alpha,
                    )
                else:
                    total_grad = mx.sum(mx.where(sample_mask, gradients, 0.0))
                    total_hess_mx = mx.sum(mx.where(sample_mask, hessians, 0.0))
                    leaf_value = compute_xgboost_leaf_value(
                        total_grad, total_hess_mx, reg_lambda, reg_alpha
                    )
                values[node_idx] = leaf_value
                is_leaf[node_idx] = True
                continue

            # Sample features for this node
            if colsample_bynode < 1.0:
                base_features = (
                    level_features
                    if level_features is not None
                    else np.arange(n_features)
                )
                n_features_node = max(1, int(len(base_features) * colsample_bynode))
                node_features = np.random.choice(
                    base_features
                    if isinstance(base_features, np.ndarray)
                    else np.array(base_features),
                    n_features_node,
                    replace=False,
                )
                node_features = mx.array(node_features)
            else:
                node_features = level_features

            # Find best split
            if has_missing:
                split_info = find_xgboost_best_split_with_missing(
                    binned_X,
                    gradients,
                    hessians,
                    sample_mask,
                    nan_mask,
                    bin_edges,
                    n_bins,
                    reg_lambda=reg_lambda,
                    gamma=gamma,
                    min_samples_leaf=min_samples_leaf,
                    min_child_weight=min_child_weight,
                    feature_indices=node_features,
                )
            elif histogram is not None:
                # Use count histogram for min_samples_leaf check
                count_hist = compute_histogram_counts_fast(
                    binned_X, sample_mask, n_bins
                )
                split_info = find_xgboost_best_split_with_histogram(
                    histogram,
                    bin_edges,
                    n_bins,
                    reg_lambda=reg_lambda,
                    gamma=gamma,
                    min_samples_leaf=min_samples_leaf,
                    min_child_weight=min_child_weight,
                    count_hist=count_hist,
                )
            else:
                split_info = find_xgboost_best_split(
                    binned_X,
                    gradients,
                    hessians,
                    sample_mask,
                    bin_edges,
                    n_bins,
                    reg_lambda=reg_lambda,
                    gamma=gamma,
                    min_samples_leaf=min_samples_leaf,
                    min_child_weight=min_child_weight,
                    feature_indices=node_features,
                )

            if not split_info.is_valid or split_info.gain <= 0:
                if histogram is not None:
                    leaf_value = compute_xgboost_leaf_value(
                        histogram.total_grad,
                        histogram.total_hess,
                        reg_lambda,
                        reg_alpha,
                    )
                else:
                    total_grad = mx.sum(mx.where(sample_mask, gradients, 0.0))
                    total_hess_mx = mx.sum(mx.where(sample_mask, hessians, 0.0))
                    leaf_value = compute_xgboost_leaf_value(
                        total_grad, total_hess_mx, reg_lambda, reg_alpha
                    )
                values[node_idx] = leaf_value
                is_leaf[node_idx] = True
                continue

            # Apply split
            left_child_idx = next_node_idx
            right_child_idx = next_node_idx + 1
            next_node_idx += 2

            feature_indices[node_idx] = split_info.feature
            thresholds[node_idx] = split_info.threshold
            left_children[node_idx] = left_child_idx
            right_children[node_idx] = right_child_idx
            is_leaf[node_idx] = False
            default_left[node_idx] = split_info.default_left

            # Create child masks
            feature_values = X[:, split_info.feature]
            goes_left = feature_values <= split_info.threshold

            # Handle missing values in child mask creation
            if has_missing:
                feat_nan_mask = nan_mask[:, split_info.feature]
                if split_info.default_left:
                    goes_left = goes_left | feat_nan_mask
                else:
                    goes_left = goes_left & ~feat_nan_mask

            # Compute child histograms using subtraction optimization
            if histogram is not None:
                left_hist, right_hist = compute_xgboost_child_histograms_smart(
                    binned_X,
                    gradients,
                    hessians,
                    sample_mask,
                    goes_left,
                    histogram,
                    n_bins,
                )
            else:
                left_mask = sample_mask & goes_left
                right_mask = sample_mask & ~goes_left
                left_hist = compute_xgboost_histogram_data(
                    binned_X, gradients, hessians, left_mask, n_bins
                )
                right_hist = compute_xgboost_histogram_data(
                    binned_X, gradients, hessians, right_mask, n_bins
                )

            left_mask = sample_mask & goes_left
            right_mask = sample_mask & ~goes_left

            next_level.append(
                XGBoostNodeInfo(left_child_idx, depth + 1, left_mask, left_hist)
            )
            next_level.append(
                XGBoostNodeInfo(right_child_idx, depth + 1, right_mask, right_hist)
            )

        current_level = next_level

    # Convert to MLX arrays
    return TreeArrays(
        feature_indices=mx.array(feature_indices),
        thresholds=mx.array(thresholds),
        left_children=mx.array(left_children),
        right_children=mx.array(right_children),
        values=mx.array(values),
        is_leaf=mx.array(is_leaf),
        default_left=mx.array(default_left),
        n_nodes=next_node_idx,
    )
