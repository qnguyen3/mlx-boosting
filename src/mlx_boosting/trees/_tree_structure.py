"""Tree data structures for GPU-efficient storage and access."""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class TreeArrays:
    """Tree stored as parallel arrays for GPU-efficient access.

    This Structure-of-Arrays (SoA) representation enables:
    - Coalesced memory access during tree traversal
    - Vectorized prediction across all samples
    - Index-based navigation (no pointer chasing)

    Attributes:
        feature_indices: Which feature to split on (-1 for leaf nodes).
        thresholds: Split threshold values.
        left_children: Index of left child (-1 for leaf nodes).
        right_children: Index of right child (-1 for leaf nodes).
        values: Prediction values for leaf nodes.
        is_leaf: Boolean mask indicating leaf nodes.
        default_left: For XGBoost, whether missing values go left (True) or right.
        n_nodes: Actual number of nodes used in the tree.
    """

    feature_indices: mx.array  # (max_nodes,) int32
    thresholds: mx.array  # (max_nodes,) float32
    left_children: mx.array  # (max_nodes,) int32
    right_children: mx.array  # (max_nodes,) int32
    values: mx.array  # (max_nodes,) or (max_nodes, n_classes) float32
    is_leaf: mx.array  # (max_nodes,) bool
    default_left: mx.array | None = (
        None  # (max_nodes,) bool - for XGBoost missing values
    )
    n_nodes: int = 0


def create_empty_tree(
    max_nodes: int, n_outputs: int = 1, with_default_left: bool = False
) -> TreeArrays:
    """Create an empty tree structure with pre-allocated arrays.

    Args:
        max_nodes: Maximum number of nodes (typically 2^(max_depth+1) - 1).
        n_outputs: Number of output values per node (1 for regression,
            n_classes for classification probabilities).
        with_default_left: Whether to include default_left array for XGBoost
            missing value handling.

    Returns:
        Empty TreeArrays structure ready for filling.
    """
    values_shape = (max_nodes,) if n_outputs == 1 else (max_nodes, n_outputs)

    return TreeArrays(
        feature_indices=mx.full((max_nodes,), -1, dtype=mx.int32),
        thresholds=mx.zeros((max_nodes,), dtype=mx.float32),
        left_children=mx.full((max_nodes,), -1, dtype=mx.int32),
        right_children=mx.full((max_nodes,), -1, dtype=mx.int32),
        values=mx.zeros(values_shape, dtype=mx.float32),
        is_leaf=mx.ones((max_nodes,), dtype=mx.bool_),
        default_left=mx.ones((max_nodes,), dtype=mx.bool_)
        if with_default_left
        else None,
        n_nodes=0,
    )


def compute_max_nodes(max_depth: int) -> int:
    """Compute maximum nodes for a complete binary tree.

    Args:
        max_depth: Maximum depth of the tree.

    Returns:
        Maximum number of nodes possible.
    """
    return 2 ** (max_depth + 1) - 1
