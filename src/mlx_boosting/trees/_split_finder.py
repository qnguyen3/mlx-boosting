"""GPU-accelerated split finding algorithms using MLX.

This module contains vectorized implementations for finding optimal splits
in decision trees, designed to maximize GPU utilization on Apple Silicon.
"""

from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx
import numpy as np

# Default number of histogram bins
DEFAULT_N_BINS: int = 256


@dataclass
class HistogramData:
    """Cached histogram data for a node.

    Used for histogram subtraction optimization where child histograms
    can be computed as parent - sibling instead of from scratch.

    Attributes:
        sum_hist: Sum of target values per bin, shape (n_features, n_bins).
        count_hist: Count of samples per bin, shape (n_features, n_bins).
        total_sum: Total sum of targets in this node.
        total_count: Total count of samples in this node.
    """

    sum_hist: mx.array  # (n_features, n_bins)
    count_hist: mx.array  # (n_features, n_bins)
    total_sum: mx.array  # scalar
    total_count: mx.array  # scalar


class SplitInfo(NamedTuple):
    """Information about a split decision.

    Attributes:
        feature: Index of the feature to split on.
        threshold: Threshold value for the split.
        gain: Information gain from the split.
        is_valid: Whether a valid split was found.
    """

    feature: int
    threshold: float
    gain: float
    is_valid: bool


def _compute_histogram_sums_impl(
    binned_X: mx.array,
    values: mx.array,
    sample_mask: mx.array,
    bin_range: mx.array,
) -> mx.array:
    """Core histogram sum computation (JIT compiled).

    Args:
        binned_X: Binned features of shape (n_samples, n_features), int32.
        values: Values to sum, shape (n_samples,), float32.
        sample_mask: Boolean mask for samples in this node.
        bin_range: Range [0, n_bins), shape (n_bins,).

    Returns:
        Histogram sums of shape (n_features, n_bins), float32.
    """
    # Broadcast comparison: (n_samples, n_features, n_bins)
    bin_matches = binned_X[:, :, None] == bin_range[None, None, :]

    # Apply sample mask and weight by values
    valid_matches = bin_matches & sample_mask[:, None, None]
    weighted = mx.where(valid_matches, values[:, None, None], 0.0)

    # Sum over samples
    return mx.sum(weighted, axis=0)


# JIT compile the core function
_compute_histogram_sums_compiled = mx.compile(_compute_histogram_sums_impl)


def compute_histogram_sums_fast(
    binned_X: mx.array,
    values: mx.array,
    sample_mask: mx.array,
    n_bins: int,
) -> mx.array:
    """Compute sum of values in each histogram bin using numpy bincount.

    Uses numpy's optimized bincount for much faster histogram computation.

    Args:
        binned_X: Binned features of shape (n_samples, n_features), int32.
        values: Values to sum (e.g., targets), shape (n_samples,), float32.
        sample_mask: Boolean mask for samples in this node.
        n_bins: Number of bins.

    Returns:
        Histogram sums of shape (n_features, n_bins), float32.
    """
    binned_np = np.array(binned_X)
    values_np = np.array(values)
    mask_np = np.array(sample_mask)

    n_samples, n_features = binned_np.shape

    # Get valid samples
    valid_idx = np.where(mask_np)[0]
    binned_valid = binned_np[valid_idx]
    values_valid = values_np[valid_idx]

    # Compute weighted histogram using bincount
    hist_sums = np.zeros((n_features, n_bins), dtype=np.float32)

    for f in range(n_features):
        hist_sums[f] = np.bincount(
            binned_valid[:, f], weights=values_valid, minlength=n_bins
        )[:n_bins]

    return mx.array(hist_sums)


def compute_histogram_counts_fast(
    binned_X: mx.array,
    sample_mask: mx.array,
    n_bins: int,
) -> mx.array:
    """Compute count of samples in each histogram bin using numpy bincount.

    Uses numpy's optimized bincount for much faster histogram computation.

    Args:
        binned_X: Binned features of shape (n_samples, n_features), int32.
        sample_mask: Boolean mask for samples in this node.
        n_bins: Number of bins.

    Returns:
        Histogram counts of shape (n_features, n_bins), float32.
    """
    binned_np = np.array(binned_X)
    mask_np = np.array(sample_mask)

    n_samples, n_features = binned_np.shape

    # Get valid samples
    valid_idx = np.where(mask_np)[0]
    binned_valid = binned_np[valid_idx]

    # Compute histogram using bincount
    hist_counts = np.zeros((n_features, n_bins), dtype=np.float32)

    for f in range(n_features):
        hist_counts[f] = np.bincount(binned_valid[:, f], minlength=n_bins)[:n_bins]

    return mx.array(hist_counts)


def compute_histogram_data(
    binned_X: mx.array,
    y: mx.array,
    sample_mask: mx.array,
    n_bins: int,
) -> HistogramData:
    """Compute complete histogram data for a node.

    This is the primary entry point for histogram computation.
    Returns structured data suitable for caching and subtraction.

    Args:
        binned_X: Binned features of shape (n_samples, n_features), int32.
        y: Target values of shape (n_samples,).
        sample_mask: Boolean mask for samples in this node.
        n_bins: Number of histogram bins.

    Returns:
        HistogramData containing sum and count histograms with totals.
    """
    sum_hist = compute_histogram_sums_fast(binned_X, y, sample_mask, n_bins)
    count_hist = compute_histogram_counts_fast(binned_X, sample_mask, n_bins)

    total_sum = mx.sum(mx.where(sample_mask, y, 0.0))
    total_count = mx.sum(sample_mask.astype(mx.float32))

    return HistogramData(
        sum_hist=sum_hist,
        count_hist=count_hist,
        total_sum=total_sum,
        total_count=total_count,
    )


def subtract_histograms(
    parent: HistogramData,
    child: HistogramData,
) -> HistogramData:
    """Compute sibling histogram via subtraction.

    Given parent histogram and one child's histogram, compute the other
    child's histogram in O(n_features × n_bins) time instead of
    O(n_samples × n_features).

    Args:
        parent: Parent node histogram data.
        child: Known child node histogram data.

    Returns:
        Histogram data for the sibling child.
    """
    return HistogramData(
        sum_hist=parent.sum_hist - child.sum_hist,
        count_hist=parent.count_hist - child.count_hist,
        total_sum=parent.total_sum - child.total_sum,
        total_count=parent.total_count - child.total_count,
    )


def compute_child_histograms_smart(
    binned_X: mx.array,
    y: mx.array,
    parent_mask: mx.array,
    goes_left: mx.array,
    parent_histogram: HistogramData,
    n_bins: int,
) -> tuple[HistogramData, HistogramData]:
    """Compute child histograms using subtraction optimization.

    Computes the smaller child's histogram directly (fewer samples = faster),
    then subtracts from parent to get the larger child's histogram.
    This reduces histogram computation for the larger child from
    O(n_samples) to O(n_features × n_bins).

    Args:
        binned_X: Binned features of shape (n_samples, n_features), int32.
        y: Target values of shape (n_samples,).
        parent_mask: Parent node's sample mask.
        goes_left: Boolean mask for samples going left.
        parent_histogram: Pre-computed parent histogram.
        n_bins: Number of histogram bins.

    Returns:
        Tuple of (left_histogram, right_histogram).
    """
    left_mask = parent_mask & goes_left
    right_mask = parent_mask & ~goes_left

    n_left = int(mx.sum(left_mask.astype(mx.float32)))
    n_right = int(mx.sum(right_mask.astype(mx.float32)))

    # Compute smaller child directly, get larger via subtraction
    if n_left <= n_right:
        left_hist = compute_histogram_data(binned_X, y, left_mask, n_bins)
        right_hist = subtract_histograms(parent_histogram, left_hist)
    else:
        right_hist = compute_histogram_data(binned_X, y, right_mask, n_bins)
        left_hist = subtract_histograms(parent_histogram, right_hist)

    return left_hist, right_hist


def _compute_histogram_counts_impl(
    binned_X: mx.array,
    sample_mask: mx.array,
    bin_range: mx.array,
) -> mx.array:
    """Core histogram count computation (JIT compiled).

    Args:
        binned_X: Binned features of shape (n_samples, n_features), int32.
        sample_mask: Boolean mask for samples in this node.
        bin_range: Range [0, n_bins), shape (n_bins,).

    Returns:
        Histogram counts of shape (n_features, n_bins), float32.
    """
    # Broadcast comparison
    bin_matches = binned_X[:, :, None] == bin_range[None, None, :]

    # Apply sample mask
    valid_matches = bin_matches & sample_mask[:, None, None]

    # Sum over samples
    return mx.sum(valid_matches.astype(mx.float32), axis=0)


# JIT compile the core function
_compute_histogram_counts_compiled = mx.compile(_compute_histogram_counts_impl)


def compute_histogram_counts_mlx(
    binned_X: mx.array,
    sample_mask: mx.array,
    n_bins: int,
    feature_batch_size: int = 64,
) -> mx.array:
    """Compute count of samples in each histogram bin for each feature (GPU).

    Uses vectorized broadcasting for GPU acceleration.
    Processes features in batches to avoid memory issues with large datasets.

    Args:
        binned_X: Binned features of shape (n_samples, n_features), int32.
        sample_mask: Boolean mask for samples in this node.
        n_bins: Number of bins.
        feature_batch_size: Number of features to process at once.

    Returns:
        Histogram counts of shape (n_features, n_bins), float32.
    """
    n_samples, n_features = binned_X.shape
    bin_range = mx.arange(n_bins)

    # For smaller feature counts, process all at once
    if n_features <= feature_batch_size:
        return _compute_histogram_counts_compiled(binned_X, sample_mask, bin_range)

    # Process features in batches for large feature counts
    hist_parts = []
    for f_start in range(0, n_features, feature_batch_size):
        f_end = min(f_start + feature_batch_size, n_features)
        binned_batch = binned_X[:, f_start:f_end]
        batch_hist = _compute_histogram_counts_compiled(
            binned_batch, sample_mask, bin_range
        )
        hist_parts.append(batch_hist)

    return mx.concatenate(hist_parts, axis=0)


def compute_variance_reduction_gains_mlx(
    sum_hist: mx.array,
    count_hist: mx.array,
    total_sum: mx.array,
    total_count: mx.array,
    min_samples_leaf: int = 1,
) -> mx.array:
    """Compute variance reduction gains for all possible splits (GPU).

    Fully vectorized across features and bins using cumulative sums.

    Args:
        sum_hist: Sum of targets per bin, shape (n_features, n_bins).
        count_hist: Count per bin, shape (n_features, n_bins).
        total_sum: Total sum of targets in the node.
        total_count: Total count of samples in the node.
        min_samples_leaf: Minimum samples required in each leaf.

    Returns:
        Gains for each split, shape (n_features, n_bins).
    """
    # Cumulative sums give left-side statistics for each potential split
    left_sum = mx.cumsum(sum_hist, axis=1)
    left_count = mx.cumsum(count_hist, axis=1)

    # Right side = Total - Left
    right_sum = total_sum - left_sum
    right_count = total_count - left_count

    # Compute means (avoid division by zero)
    left_mean = left_sum / mx.maximum(left_count, 1.0)
    right_mean = right_sum / mx.maximum(right_count, 1.0)
    parent_mean = total_sum / mx.maximum(total_count, 1.0)

    # Variance reduction = weighted variance decrease
    parent_score = total_count * parent_mean**2
    left_score = left_count * left_mean**2
    right_score = right_count * right_mean**2

    gains = left_score + right_score - parent_score

    # Invalidate splits with too few samples
    valid_mask = (left_count >= min_samples_leaf) & (right_count >= min_samples_leaf)
    gains = mx.where(valid_mask, gains, -mx.inf)

    return gains


def find_best_split_histogram(
    X: mx.array,
    y: mx.array,
    sample_mask: mx.array,
    bin_edges: mx.array,
    binned_X: mx.array,
    n_bins: int,
    min_samples_leaf: int = 1,
) -> SplitInfo:
    """Find the best split using histogram-based approach (GPU-accelerated).

    All computations run on GPU using MLX vectorized operations.

    Args:
        X: Features of shape (n_samples, n_features).
        y: Targets of shape (n_samples,).
        sample_mask: Boolean mask for samples in this node.
        bin_edges: Pre-computed bin edges, shape (n_features, n_bins+1).
        binned_X: Pre-computed binned features, shape (n_samples, n_features).
        n_bins: Number of bins.
        min_samples_leaf: Minimum samples required in each leaf.

    Returns:
        SplitInfo with best feature, threshold, and gain.
    """
    # Compute histograms using fast numpy bincount
    sum_hist = compute_histogram_sums_fast(binned_X, y, sample_mask, n_bins)
    count_hist = compute_histogram_counts_fast(binned_X, sample_mask, n_bins)

    # Total statistics (GPU)
    total_sum = mx.sum(mx.where(sample_mask, y, 0.0))
    total_count = mx.sum(sample_mask.astype(mx.float32))

    # Early exit check
    mx.eval(total_count)
    if float(total_count) < 2 * min_samples_leaf:
        return SplitInfo(feature=-1, threshold=0.0, gain=-np.inf, is_valid=False)

    # Compute gains for all splits using MLX (GPU)
    gains = compute_variance_reduction_gains_mlx(
        sum_hist, count_hist, total_sum, total_count, min_samples_leaf
    )

    # Evaluate and find maximum gain
    mx.eval(gains)

    best_gain = float(mx.max(gains))

    if best_gain <= 0 or np.isinf(best_gain):
        return SplitInfo(feature=-1, threshold=0.0, gain=best_gain, is_valid=False)

    # Get indices of maximum
    flat_gains = gains.flatten()
    flat_idx = int(mx.argmax(flat_gains))
    best_feature = flat_idx // n_bins
    best_bin = flat_idx % n_bins

    # Convert bin index to threshold value
    best_threshold = float(bin_edges[best_feature, best_bin + 1])

    return SplitInfo(
        feature=best_feature,
        threshold=best_threshold,
        gain=best_gain,
        is_valid=True,
    )


def find_best_split_with_histogram(
    bin_edges: mx.array,
    histogram: HistogramData,
    n_bins: int,
    min_samples_leaf: int = 1,
) -> SplitInfo:
    """Find best split using pre-computed histogram data.

    This function avoids recomputing histograms when they're already
    available from parent node processing.

    Args:
        bin_edges: Pre-computed bin edges, shape (n_features, n_bins+1).
        histogram: Pre-computed histogram data for this node.
        n_bins: Number of bins.
        min_samples_leaf: Minimum samples required in each leaf.

    Returns:
        SplitInfo with best feature, threshold, and gain.
    """
    # Early exit check
    mx.eval(histogram.total_count)
    if float(histogram.total_count) < 2 * min_samples_leaf:
        return SplitInfo(feature=-1, threshold=0.0, gain=-np.inf, is_valid=False)

    # Compute gains using pre-computed histogram
    gains = compute_variance_reduction_gains_mlx(
        histogram.sum_hist,
        histogram.count_hist,
        histogram.total_sum,
        histogram.total_count,
        min_samples_leaf,
    )

    mx.eval(gains)
    best_gain = float(mx.max(gains))

    if best_gain <= 0 or np.isinf(best_gain):
        return SplitInfo(feature=-1, threshold=0.0, gain=best_gain, is_valid=False)

    # Get indices of maximum
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


# =============================================================================
# Level-wise batched histogram functions
# =============================================================================


def compute_batch_histogram_sums(
    binned_X: mx.array,
    y: mx.array,
    sample_masks: mx.array,
    n_bins: int,
) -> mx.array:
    """Compute histogram sums for multiple nodes simultaneously.

    Args:
        binned_X: Binned features, shape (n_samples, n_features).
        y: Target values, shape (n_samples,).
        sample_masks: Stacked masks, shape (n_nodes, n_samples).
        n_bins: Number of histogram bins.

    Returns:
        Batch histogram sums, shape (n_nodes, n_features, n_bins).
    """
    n_nodes = sample_masks.shape[0]
    n_features = binned_X.shape[1]

    binned_np = np.array(binned_X)
    y_np = np.array(y)
    masks_np = np.array(sample_masks)

    batch_sums = np.zeros((n_nodes, n_features, n_bins), dtype=np.float32)

    for node_idx in range(n_nodes):
        mask = masks_np[node_idx]
        valid_idx = np.where(mask)[0]
        if len(valid_idx) == 0:
            continue

        binned_valid = binned_np[valid_idx]
        y_valid = y_np[valid_idx]

        for f in range(n_features):
            batch_sums[node_idx, f] = np.bincount(
                binned_valid[:, f], weights=y_valid, minlength=n_bins
            )[:n_bins]

    return mx.array(batch_sums)


def compute_batch_histogram_counts(
    binned_X: mx.array,
    sample_masks: mx.array,
    n_bins: int,
) -> mx.array:
    """Compute histogram counts for multiple nodes simultaneously.

    Args:
        binned_X: Binned features, shape (n_samples, n_features).
        sample_masks: Stacked masks, shape (n_nodes, n_samples).
        n_bins: Number of histogram bins.

    Returns:
        Batch histogram counts, shape (n_nodes, n_features, n_bins).
    """
    n_nodes = sample_masks.shape[0]
    n_features = binned_X.shape[1]

    binned_np = np.array(binned_X)
    masks_np = np.array(sample_masks)

    batch_counts = np.zeros((n_nodes, n_features, n_bins), dtype=np.float32)

    for node_idx in range(n_nodes):
        mask = masks_np[node_idx]
        valid_idx = np.where(mask)[0]
        if len(valid_idx) == 0:
            continue

        binned_valid = binned_np[valid_idx]

        for f in range(n_features):
            batch_counts[node_idx, f] = np.bincount(
                binned_valid[:, f], minlength=n_bins
            )[:n_bins]

    return mx.array(batch_counts)


def compute_batch_variance_gains(
    sum_hist: mx.array,
    count_hist: mx.array,
    min_samples_leaf: int = 1,
) -> mx.array:
    """Compute variance reduction gains for batched nodes.

    Fully vectorized across nodes, features, and bins using cumulative sums.

    Args:
        sum_hist: Shape (n_nodes, n_features, n_bins).
        count_hist: Shape (n_nodes, n_features, n_bins).
        min_samples_leaf: Minimum samples in each leaf.

    Returns:
        Gains for each split, shape (n_nodes, n_features, n_bins).
    """
    # Cumulative sums for left-side statistics
    left_sum = mx.cumsum(sum_hist, axis=2)  # (n_nodes, n_features, n_bins)
    left_count = mx.cumsum(count_hist, axis=2)

    # Total statistics per node per feature
    total_sum = mx.sum(sum_hist, axis=2, keepdims=True)  # (n_nodes, n_features, 1)
    total_count = mx.sum(count_hist, axis=2, keepdims=True)

    # Right side = Total - Left
    right_sum = total_sum - left_sum
    right_count = total_count - left_count

    # Compute means (avoid division by zero)
    left_mean = left_sum / mx.maximum(left_count, 1.0)
    right_mean = right_sum / mx.maximum(right_count, 1.0)
    parent_mean = total_sum / mx.maximum(total_count, 1.0)

    # Variance reduction = weighted variance decrease
    parent_score = total_count * parent_mean**2
    left_score = left_count * left_mean**2
    right_score = right_count * right_mean**2

    gains = left_score + right_score - parent_score

    # Invalidate splits with too few samples
    valid_mask = (left_count >= min_samples_leaf) & (right_count >= min_samples_leaf)
    gains = mx.where(valid_mask, gains, -mx.inf)

    return gains


def find_batch_best_splits(
    gains: mx.array,
    bin_edges: mx.array,
    n_bins: int,
) -> list[SplitInfo]:
    """Find best split for each node in batch.

    Args:
        gains: Shape (n_nodes, n_features, n_bins).
        bin_edges: Shape (n_features, n_bins + 1).
        n_bins: Number of bins.

    Returns:
        List of SplitInfo, one per node.
    """
    n_nodes = gains.shape[0]
    mx.eval(gains)

    splits = []
    for node_idx in range(n_nodes):
        node_gains = gains[node_idx]  # (n_features, n_bins)
        best_gain = float(mx.max(node_gains))

        if best_gain <= 0 or np.isinf(best_gain):
            splits.append(
                SplitInfo(feature=-1, threshold=0.0, gain=best_gain, is_valid=False)
            )
            continue

        flat_gains = node_gains.flatten()
        flat_idx = int(mx.argmax(flat_gains))
        best_feature = flat_idx // n_bins
        best_bin = flat_idx % n_bins
        best_threshold = float(bin_edges[best_feature, best_bin + 1])

        splits.append(
            SplitInfo(
                feature=best_feature,
                threshold=best_threshold,
                gain=best_gain,
                is_valid=True,
            )
        )

    return splits


def find_best_split_exact(
    X: mx.array,
    y: mx.array,
    sample_mask: mx.array,
    min_samples_leaf: int = 1,
    n_thresholds: int = 100,
) -> SplitInfo:
    """Find the best split using exact threshold search.

    This approach uses MLX vectorized operations for GPU acceleration.

    Args:
        X: Features of shape (n_samples, n_features).
        y: Targets of shape (n_samples,).
        sample_mask: Boolean mask for samples in this node.
        min_samples_leaf: Minimum samples required in each leaf.
        n_thresholds: Number of candidate thresholds per feature.

    Returns:
        SplitInfo with best feature, threshold, and gain.
    """
    n_samples, n_features = X.shape

    total_count = float(mx.sum(sample_mask.astype(mx.float32)))

    if total_count < 2 * min_samples_leaf:
        return SplitInfo(feature=-1, threshold=0.0, gain=-np.inf, is_valid=False)

    # Generate candidate thresholds for each feature
    feature_mins = mx.min(X, axis=0)
    feature_maxs = mx.max(X, axis=0)

    # Create threshold grid: (n_features, n_thresholds)
    t = mx.linspace(0.01, 0.99, n_thresholds)
    thresholds = (
        t[None, :] * (feature_maxs - feature_mins)[:, None] + feature_mins[:, None]
    )

    # Vectorized comparison: (n_samples, n_features, n_thresholds)
    goes_left = X[:, :, None] <= thresholds[None, :, :]

    # Apply sample mask
    valid_samples = sample_mask[:, None, None]

    # Count samples going left/right
    n_left = mx.sum((goes_left & valid_samples).astype(mx.float32), axis=0)
    n_right = mx.sum((~goes_left & valid_samples).astype(mx.float32), axis=0)

    # Sum of y values going left/right
    y_expanded = y[:, None, None]
    sum_left = mx.sum(mx.where(goes_left & valid_samples, y_expanded, 0.0), axis=0)
    sum_right = mx.sum(mx.where(~goes_left & valid_samples, y_expanded, 0.0), axis=0)

    # Compute means
    mean_left = sum_left / mx.maximum(n_left, 1.0)
    mean_right = sum_right / mx.maximum(n_right, 1.0)

    total_sum = mx.sum(mx.where(sample_mask, y, 0.0))
    parent_mean = total_sum / total_count

    # Variance reduction gain
    parent_score = total_count * parent_mean**2
    gains = n_left * mean_left**2 + n_right * mean_right**2 - parent_score

    # Invalidate splits with too few samples
    valid_mask = (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
    gains = mx.where(valid_mask, gains, -mx.inf)

    # Evaluate and find best
    mx.eval(gains)

    best_gain = float(mx.max(gains))

    if best_gain <= 0 or np.isinf(best_gain):
        return SplitInfo(feature=-1, threshold=0.0, gain=best_gain, is_valid=False)

    # Get indices of maximum
    flat_gains = gains.flatten()
    flat_idx = int(mx.argmax(flat_gains))
    best_feature = flat_idx // n_thresholds
    best_threshold_idx = flat_idx % n_thresholds
    best_threshold = float(thresholds[best_feature, best_threshold_idx])

    return SplitInfo(
        feature=best_feature,
        threshold=best_threshold,
        gain=best_gain,
        is_valid=True,
    )


def compute_leaf_value(y: mx.array, sample_mask: mx.array) -> float:
    """Compute the prediction value for a leaf node.

    For regression, this is the mean of target values.

    Args:
        y: Targets of shape (n_samples,).
        sample_mask: Boolean mask for samples in this node.

    Returns:
        Mean target value.
    """
    masked_sum = mx.sum(mx.where(sample_mask, y, 0.0))
    count = mx.sum(sample_mask.astype(mx.float32))
    value = masked_sum / mx.maximum(count, 1.0)
    mx.eval(value)
    return float(value)


# =============================================================================
# XGBoost-specific functions
# =============================================================================


@dataclass
class XGBoostHistogramData:
    """Histogram data for XGBoost (gradients and Hessians).

    Used for XGBoost split finding where gains are computed using
    both first-order (gradient) and second-order (Hessian) derivatives.

    Attributes:
        grad_hist: Sum of gradients per bin, shape (n_features, n_bins).
        hess_hist: Sum of Hessians per bin, shape (n_features, n_bins).
        total_grad: Total sum of gradients in this node.
        total_hess: Total sum of Hessians in this node.
    """

    grad_hist: mx.array  # (n_features, n_bins)
    hess_hist: mx.array  # (n_features, n_bins)
    total_grad: mx.array  # scalar
    total_hess: mx.array  # scalar


class XGBoostSplitInfo(NamedTuple):
    """Information about an XGBoost split decision.

    Attributes:
        feature: Index of the feature to split on.
        threshold: Threshold value for the split.
        gain: XGBoost gain from the split.
        is_valid: Whether a valid split was found.
        default_left: Whether missing values should go left (True) or right.
    """

    feature: int
    threshold: float
    gain: float
    is_valid: bool
    default_left: bool = True


def compute_xgboost_histogram_data(
    binned_X: mx.array,
    gradients: mx.array,
    hessians: mx.array,
    sample_mask: mx.array,
    n_bins: int,
) -> XGBoostHistogramData:
    """Compute XGBoost histogram data (gradients and Hessians).

    Args:
        binned_X: Binned features of shape (n_samples, n_features), int32.
        gradients: First-order derivatives, shape (n_samples,).
        hessians: Second-order derivatives, shape (n_samples,).
        sample_mask: Boolean mask for samples in this node.
        n_bins: Number of histogram bins.

    Returns:
        XGBoostHistogramData with gradient and Hessian histograms.
    """
    grad_hist = compute_histogram_sums_fast(binned_X, gradients, sample_mask, n_bins)
    hess_hist = compute_histogram_sums_fast(binned_X, hessians, sample_mask, n_bins)

    total_grad = mx.sum(mx.where(sample_mask, gradients, 0.0))
    total_hess = mx.sum(mx.where(sample_mask, hessians, 0.0))

    return XGBoostHistogramData(
        grad_hist=grad_hist,
        hess_hist=hess_hist,
        total_grad=total_grad,
        total_hess=total_hess,
    )


def subtract_xgboost_histograms(
    parent: XGBoostHistogramData,
    child: XGBoostHistogramData,
) -> XGBoostHistogramData:
    """Compute sibling XGBoost histogram via subtraction.

    Args:
        parent: Parent node histogram data.
        child: Known child node histogram data.

    Returns:
        XGBoost histogram data for the sibling child.
    """
    return XGBoostHistogramData(
        grad_hist=parent.grad_hist - child.grad_hist,
        hess_hist=parent.hess_hist - child.hess_hist,
        total_grad=parent.total_grad - child.total_grad,
        total_hess=parent.total_hess - child.total_hess,
    )


def compute_xgboost_child_histograms_smart(
    binned_X: mx.array,
    gradients: mx.array,
    hessians: mx.array,
    parent_mask: mx.array,
    goes_left: mx.array,
    parent_histogram: XGBoostHistogramData,
    n_bins: int,
) -> tuple[XGBoostHistogramData, XGBoostHistogramData]:
    """Compute XGBoost child histograms using subtraction optimization.

    Computes the smaller child's histogram directly, then subtracts
    from parent to get the larger child's histogram.

    Args:
        binned_X: Binned features of shape (n_samples, n_features).
        gradients: First-order derivatives, shape (n_samples,).
        hessians: Second-order derivatives, shape (n_samples,).
        parent_mask: Parent node's sample mask.
        goes_left: Boolean mask for samples going left.
        parent_histogram: Pre-computed parent histogram.
        n_bins: Number of histogram bins.

    Returns:
        Tuple of (left_histogram, right_histogram).
    """
    left_mask = parent_mask & goes_left
    right_mask = parent_mask & ~goes_left

    n_left = int(mx.sum(left_mask.astype(mx.float32)))
    n_right = int(mx.sum(right_mask.astype(mx.float32)))

    if n_left <= n_right:
        left_hist = compute_xgboost_histogram_data(
            binned_X, gradients, hessians, left_mask, n_bins
        )
        right_hist = subtract_xgboost_histograms(parent_histogram, left_hist)
    else:
        right_hist = compute_xgboost_histogram_data(
            binned_X, gradients, hessians, right_mask, n_bins
        )
        left_hist = subtract_xgboost_histograms(parent_histogram, right_hist)

    return left_hist, right_hist


def compute_xgboost_gains(
    grad_hist: mx.array,
    hess_hist: mx.array,
    total_grad: mx.array,
    total_hess: mx.array,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    min_samples_leaf: int = 1,
    min_child_weight: float = 1.0,
    count_hist: mx.array | None = None,
) -> mx.array:
    """Compute XGBoost split gains for all possible splits.

    Uses the XGBoost gain formula:
    Gain = 0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ

    Args:
        grad_hist: Sum of gradients per bin, shape (n_features, n_bins).
        hess_hist: Sum of Hessians per bin, shape (n_features, n_bins).
        total_grad: Total gradient sum in the node.
        total_hess: Total Hessian sum in the node.
        reg_lambda: L2 regularization parameter.
        gamma: Minimum loss reduction for split (pruning threshold).
        min_samples_leaf: Minimum samples in each leaf (if count_hist provided).
        min_child_weight: Minimum sum of Hessians in each leaf.
        count_hist: Optional count histogram for min_samples_leaf check.

    Returns:
        Gains for each split, shape (n_features, n_bins).
    """
    # Cumulative sums give left-side statistics for each potential split
    left_grad = mx.cumsum(grad_hist, axis=1)
    left_hess = mx.cumsum(hess_hist, axis=1)

    # Right side = Total - Left
    right_grad = total_grad - left_grad
    right_hess = total_hess - left_hess

    # XGBoost gain formula: 0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ
    left_score = (left_grad**2) / (left_hess + reg_lambda)
    right_score = (right_grad**2) / (right_hess + reg_lambda)
    parent_score = (total_grad**2) / (total_hess + reg_lambda)

    gains = 0.5 * (left_score + right_score - parent_score) - gamma

    # Invalidate splits with insufficient Hessian (min_child_weight)
    valid_mask = (left_hess >= min_child_weight) & (right_hess >= min_child_weight)

    # Also check min_samples_leaf if count histogram provided
    if count_hist is not None:
        left_count = mx.cumsum(count_hist, axis=1)
        right_count = mx.sum(count_hist, axis=1, keepdims=True) - left_count
        valid_mask = (
            valid_mask
            & (left_count >= min_samples_leaf)
            & (right_count >= min_samples_leaf)
        )

    gains = mx.where(valid_mask, gains, -mx.inf)

    return gains


def compute_xgboost_leaf_value(
    total_grad: mx.array,
    total_hess: mx.array,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
) -> float:
    """Compute XGBoost optimal leaf value.

    Uses the formula: w* = -G / (H + λ)
    With L1 regularization: w* = -sign(G) * max(0, |G| - α) / (H + λ)

    Args:
        total_grad: Total gradient sum in the leaf.
        total_hess: Total Hessian sum in the leaf.
        reg_lambda: L2 regularization parameter.
        reg_alpha: L1 regularization parameter.

    Returns:
        Optimal leaf prediction value.
    """
    total_grad = float(total_grad)
    total_hess = float(total_hess)

    # Add minimum value to prevent division by zero
    denominator = total_hess + reg_lambda
    if denominator < 1e-10:
        denominator = 1e-10

    if reg_alpha > 0:
        # Soft thresholding for L1 regularization
        if total_grad > reg_alpha:
            return -(total_grad - reg_alpha) / denominator
        elif total_grad < -reg_alpha:
            return -(total_grad + reg_alpha) / denominator
        else:
            return 0.0
    else:
        return -total_grad / denominator


def find_xgboost_best_split(
    binned_X: mx.array,
    gradients: mx.array,
    hessians: mx.array,
    sample_mask: mx.array,
    bin_edges: mx.array,
    n_bins: int,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    min_samples_leaf: int = 1,
    min_child_weight: float = 1.0,
    feature_indices: mx.array | None = None,
) -> XGBoostSplitInfo:
    """Find the best XGBoost split using histogram-based approach.

    Args:
        binned_X: Binned features of shape (n_samples, n_features), int32.
        gradients: First-order derivatives, shape (n_samples,).
        hessians: Second-order derivatives, shape (n_samples,).
        sample_mask: Boolean mask for samples in this node.
        bin_edges: Pre-computed bin edges, shape (n_features, n_bins+1).
        n_bins: Number of bins.
        reg_lambda: L2 regularization parameter.
        gamma: Minimum loss reduction for split.
        min_samples_leaf: Minimum samples in each leaf.
        min_child_weight: Minimum sum of Hessians in each leaf.
        feature_indices: Optional subset of features to consider.

    Returns:
        XGBoostSplitInfo with best feature, threshold, gain, and default direction.
    """
    # Subset features if specified
    if feature_indices is not None:
        binned_X = binned_X[:, feature_indices]
        bin_edges = bin_edges[feature_indices]

    # Compute histograms
    grad_hist = compute_histogram_sums_fast(binned_X, gradients, sample_mask, n_bins)
    hess_hist = compute_histogram_sums_fast(binned_X, hessians, sample_mask, n_bins)
    count_hist = compute_histogram_counts_fast(binned_X, sample_mask, n_bins)

    total_grad = mx.sum(mx.where(sample_mask, gradients, 0.0))
    total_hess = mx.sum(mx.where(sample_mask, hessians, 0.0))

    # Early exit check
    mx.eval(total_hess)
    if float(total_hess) < 2 * min_child_weight:
        return XGBoostSplitInfo(
            feature=-1, threshold=0.0, gain=-np.inf, is_valid=False, default_left=True
        )

    # Compute gains for all splits
    gains = compute_xgboost_gains(
        grad_hist,
        hess_hist,
        total_grad,
        total_hess,
        reg_lambda=reg_lambda,
        gamma=gamma,
        min_samples_leaf=min_samples_leaf,
        min_child_weight=min_child_weight,
        count_hist=count_hist,
    )

    mx.eval(gains)

    best_gain = float(mx.max(gains))

    if best_gain <= 0 or np.isinf(best_gain):
        return XGBoostSplitInfo(
            feature=-1, threshold=0.0, gain=best_gain, is_valid=False, default_left=True
        )

    # Get indices of maximum
    flat_gains = gains.flatten()
    flat_idx = int(mx.argmax(flat_gains))
    best_feature = flat_idx // n_bins
    best_bin = flat_idx % n_bins

    # Map back to original feature index if subset was used
    if feature_indices is not None:
        best_feature = int(feature_indices[best_feature])
        best_threshold = float(bin_edges[best_feature, best_bin + 1])
    else:
        best_threshold = float(bin_edges[best_feature, best_bin + 1])

    return XGBoostSplitInfo(
        feature=best_feature,
        threshold=best_threshold,
        gain=best_gain,
        is_valid=True,
        default_left=True,  # Default direction, will be updated by missing value handler
    )


def find_xgboost_best_split_with_missing(
    binned_X: mx.array,
    gradients: mx.array,
    hessians: mx.array,
    sample_mask: mx.array,
    nan_mask: mx.array,
    bin_edges: mx.array,
    n_bins: int,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    min_samples_leaf: int = 1,
    min_child_weight: float = 1.0,
    feature_indices: mx.array | None = None,
) -> XGBoostSplitInfo:
    """Find best XGBoost split with missing value handling (sparsity-aware).

    For each candidate split, tries both directions for missing values
    and chooses the direction with higher gain.

    Args:
        binned_X: Binned features of shape (n_samples, n_features), int32.
            NaN values should be binned to a special bin (e.g., n_bins-1).
        gradients: First-order derivatives, shape (n_samples,).
        hessians: Second-order derivatives, shape (n_samples,).
        sample_mask: Boolean mask for samples in this node.
        nan_mask: Boolean mask where True indicates NaN, shape (n_samples, n_features).
        bin_edges: Pre-computed bin edges, shape (n_features, n_bins+1).
        n_bins: Number of bins.
        reg_lambda: L2 regularization parameter.
        gamma: Minimum loss reduction for split.
        min_samples_leaf: Minimum samples in each leaf.
        min_child_weight: Minimum sum of Hessians in each leaf.
        feature_indices: Optional subset of features to consider.

    Returns:
        XGBoostSplitInfo with best feature, threshold, gain, and learned default direction.
    """
    # First, find best split ignoring missing values (using non-missing only)
    # Create mask excluding missing values
    nan_mask_np = np.array(nan_mask)
    sample_mask_np = np.array(sample_mask)

    n_features = binned_X.shape[1]
    if feature_indices is not None:
        feature_list = list(np.array(feature_indices))
    else:
        feature_list = list(range(n_features))

    best_overall_gain = -np.inf
    best_feature = -1
    best_threshold = 0.0
    best_default_left = True

    np.array(binned_X)
    np.array(gradients)
    np.array(hessians)
    bin_edges_np = np.array(bin_edges)

    for feat_idx in feature_list:
        # Get non-missing samples for this feature
        has_value = ~nan_mask_np[:, feat_idx] & sample_mask_np
        is_missing = nan_mask_np[:, feat_idx] & sample_mask_np

        n_valid = np.sum(has_value)
        n_missing = np.sum(is_missing)

        if n_valid < 2 * min_samples_leaf:
            continue

        # Compute histogram for non-missing samples only
        valid_mask = mx.array(has_value)
        binned_feat = binned_X[:, feat_idx : feat_idx + 1]

        grad_hist_feat = compute_histogram_sums_fast(
            binned_feat, gradients, valid_mask, n_bins
        )[0]
        hess_hist_feat = compute_histogram_sums_fast(
            binned_feat, hessians, valid_mask, n_bins
        )[0]

        # Total stats for non-missing
        total_grad_valid = float(mx.sum(mx.where(valid_mask, gradients, 0.0)))
        total_hess_valid = float(mx.sum(mx.where(valid_mask, hessians, 0.0)))

        # Stats for missing values
        if n_missing > 0:
            missing_mask = mx.array(is_missing)
            grad_missing = float(mx.sum(mx.where(missing_mask, gradients, 0.0)))
            hess_missing = float(mx.sum(mx.where(missing_mask, hessians, 0.0)))
        else:
            grad_missing = 0.0
            hess_missing = 0.0

        # Try each split threshold
        grad_hist_np = np.array(grad_hist_feat)
        hess_hist_np = np.array(hess_hist_feat)

        left_grad_cum = np.cumsum(grad_hist_np)
        left_hess_cum = np.cumsum(hess_hist_np)

        for bin_idx in range(n_bins - 1):
            left_grad = left_grad_cum[bin_idx]
            left_hess = left_hess_cum[bin_idx]
            right_grad = total_grad_valid - left_grad
            right_hess = total_hess_valid - left_hess

            # Check min_child_weight (without missing)
            if left_hess < min_child_weight or right_hess < min_child_weight:
                continue

            total_grad = total_grad_valid + grad_missing
            total_hess = total_hess_valid + hess_missing

            # Try missing -> left
            left_grad_with_missing = left_grad + grad_missing
            left_hess_with_missing = left_hess + hess_missing

            if (
                left_hess_with_missing >= min_child_weight
                and right_hess >= min_child_weight
            ):
                gain_left = (
                    0.5
                    * (
                        (left_grad_with_missing**2)
                        / (left_hess_with_missing + reg_lambda)
                        + (right_grad**2) / (right_hess + reg_lambda)
                        - (total_grad**2) / (total_hess + reg_lambda)
                    )
                    - gamma
                )
            else:
                gain_left = -np.inf

            # Try missing -> right
            right_grad_with_missing = right_grad + grad_missing
            right_hess_with_missing = right_hess + hess_missing

            if (
                left_hess >= min_child_weight
                and right_hess_with_missing >= min_child_weight
            ):
                gain_right = (
                    0.5
                    * (
                        (left_grad**2) / (left_hess + reg_lambda)
                        + (right_grad_with_missing**2)
                        / (right_hess_with_missing + reg_lambda)
                        - (total_grad**2) / (total_hess + reg_lambda)
                    )
                    - gamma
                )
            else:
                gain_right = -np.inf

            # Choose best direction
            if gain_left >= gain_right and gain_left > best_overall_gain:
                best_overall_gain = gain_left
                best_feature = feat_idx
                best_threshold = float(bin_edges_np[feat_idx, bin_idx + 1])
                best_default_left = True
            elif gain_right > gain_left and gain_right > best_overall_gain:
                best_overall_gain = gain_right
                best_feature = feat_idx
                best_threshold = float(bin_edges_np[feat_idx, bin_idx + 1])
                best_default_left = False

    if best_overall_gain <= 0 or np.isinf(best_overall_gain):
        return XGBoostSplitInfo(
            feature=-1,
            threshold=0.0,
            gain=best_overall_gain,
            is_valid=False,
            default_left=True,
        )

    return XGBoostSplitInfo(
        feature=best_feature,
        threshold=best_threshold,
        gain=best_overall_gain,
        is_valid=True,
        default_left=best_default_left,
    )


def find_xgboost_best_split_with_histogram(
    histogram: XGBoostHistogramData,
    bin_edges: mx.array,
    n_bins: int,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    min_samples_leaf: int = 1,
    min_child_weight: float = 1.0,
    count_hist: mx.array | None = None,
) -> XGBoostSplitInfo:
    """Find best XGBoost split using pre-computed histogram.

    Args:
        histogram: Pre-computed XGBoostHistogramData.
        bin_edges: Pre-computed bin edges, shape (n_features, n_bins+1).
        n_bins: Number of bins.
        reg_lambda: L2 regularization parameter.
        gamma: Minimum loss reduction for split.
        min_samples_leaf: Minimum samples in each leaf.
        min_child_weight: Minimum sum of Hessians in each leaf.
        count_hist: Optional count histogram for min_samples_leaf check.

    Returns:
        XGBoostSplitInfo with best feature, threshold, gain.
    """
    # Early exit check
    mx.eval(histogram.total_hess)
    if float(histogram.total_hess) < 2 * min_child_weight:
        return XGBoostSplitInfo(
            feature=-1, threshold=0.0, gain=-np.inf, is_valid=False, default_left=True
        )

    # Compute gains for all splits
    gains = compute_xgboost_gains(
        histogram.grad_hist,
        histogram.hess_hist,
        histogram.total_grad,
        histogram.total_hess,
        reg_lambda=reg_lambda,
        gamma=gamma,
        min_samples_leaf=min_samples_leaf,
        min_child_weight=min_child_weight,
        count_hist=count_hist,
    )

    mx.eval(gains)

    best_gain = float(mx.max(gains))

    if best_gain <= 0 or np.isinf(best_gain):
        return XGBoostSplitInfo(
            feature=-1, threshold=0.0, gain=best_gain, is_valid=False, default_left=True
        )

    # Get indices of maximum
    flat_gains = gains.flatten()
    flat_idx = int(mx.argmax(flat_gains))
    best_feature = flat_idx // n_bins
    best_bin = flat_idx % n_bins

    best_threshold = float(bin_edges[best_feature, best_bin + 1])

    return XGBoostSplitInfo(
        feature=best_feature,
        threshold=best_threshold,
        gain=best_gain,
        is_valid=True,
        default_left=True,
    )


# =============================================================================
# Batch XGBoost functions for level-wise processing
# =============================================================================


def compute_batch_xgboost_histogram_sums(
    binned_X: mx.array,
    values: mx.array,
    sample_masks: mx.array,
    n_bins: int,
) -> mx.array:
    """Compute histogram sums for multiple nodes in batch.

    Args:
        binned_X: Binned features of shape (n_samples, n_features).
        values: Values to sum (gradients or Hessians), shape (n_samples,).
        sample_masks: Stacked masks of shape (n_nodes, n_samples).
        n_bins: Number of bins.

    Returns:
        Histogram sums of shape (n_nodes, n_features, n_bins).
    """
    n_nodes = sample_masks.shape[0]
    n_features = binned_X.shape[1]

    binned_np = np.array(binned_X)
    values_np = np.array(values)
    masks_np = np.array(sample_masks)

    result = np.zeros((n_nodes, n_features, n_bins), dtype=np.float32)

    for node_idx in range(n_nodes):
        mask = masks_np[node_idx]
        valid_idx = np.where(mask)[0]

        if len(valid_idx) == 0:
            continue

        binned_valid = binned_np[valid_idx]
        values_valid = values_np[valid_idx]

        for f in range(n_features):
            result[node_idx, f] = np.bincount(
                binned_valid[:, f],
                weights=values_valid,
                minlength=n_bins,
            )[:n_bins]

    return mx.array(result)


def compute_batch_xgboost_gains(
    grad_hist: mx.array,
    hess_hist: mx.array,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
    min_child_weight: float = 1.0,
) -> mx.array:
    """Compute XGBoost gains for multiple nodes in batch.

    Args:
        grad_hist: Gradient histograms, shape (n_nodes, n_features, n_bins).
        hess_hist: Hessian histograms, shape (n_nodes, n_features, n_bins).
        reg_lambda: L2 regularization parameter.
        gamma: Minimum loss reduction for split.
        min_child_weight: Minimum sum of Hessians in each child.

    Returns:
        Gains of shape (n_nodes, n_features, n_bins).
    """
    # Total statistics per node
    total_grad = mx.sum(grad_hist, axis=2, keepdims=True)  # (n_nodes, n_features, 1)
    total_hess = mx.sum(hess_hist, axis=2, keepdims=True)

    # Cumulative sums for left child
    left_grad = mx.cumsum(grad_hist, axis=2)
    left_hess = mx.cumsum(hess_hist, axis=2)

    # Right child
    right_grad = total_grad - left_grad
    right_hess = total_hess - left_hess

    # XGBoost gain
    left_score = (left_grad**2) / (left_hess + reg_lambda)
    right_score = (right_grad**2) / (right_hess + reg_lambda)
    parent_score = (total_grad**2) / (total_hess + reg_lambda)

    gains = 0.5 * (left_score + right_score - parent_score) - gamma

    # Invalidate splits with insufficient Hessian
    valid_mask = (left_hess >= min_child_weight) & (right_hess >= min_child_weight)
    gains = mx.where(valid_mask, gains, -mx.inf)

    return gains


def find_batch_xgboost_best_splits(
    gains: mx.array,
    bin_edges: mx.array,
    n_bins: int,
) -> list[XGBoostSplitInfo]:
    """Find best XGBoost split for each node in batch.

    Args:
        gains: Gain matrices, shape (n_nodes, n_features, n_bins).
        bin_edges: Bin edges, shape (n_features, n_bins+1).
        n_bins: Number of bins.

    Returns:
        List of XGBoostSplitInfo for each node.
    """
    mx.eval(gains)
    n_nodes = gains.shape[0]

    results = []
    gains_np = np.array(gains)
    bin_edges_np = np.array(bin_edges)

    for node_idx in range(n_nodes):
        node_gains = gains_np[node_idx]
        best_gain = float(np.max(node_gains))

        if best_gain <= 0 or np.isinf(best_gain):
            results.append(
                XGBoostSplitInfo(
                    feature=-1,
                    threshold=0.0,
                    gain=best_gain,
                    is_valid=False,
                    default_left=True,
                )
            )
            continue

        flat_idx = int(np.argmax(node_gains.flatten()))
        best_feature = flat_idx // n_bins
        best_bin = flat_idx % n_bins
        best_threshold = float(bin_edges_np[best_feature, best_bin + 1])

        results.append(
            XGBoostSplitInfo(
                feature=best_feature,
                threshold=best_threshold,
                gain=best_gain,
                is_valid=True,
                default_left=True,
            )
        )

    return results
