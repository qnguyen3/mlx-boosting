"""Data utilities for MLX Boosting."""

import mlx.core as mx
import numpy as np


def to_mlx_array(data: np.ndarray | mx.array | list) -> mx.array:
    """Convert input data to MLX array.

    Args:
        data: Input data as numpy array, MLX array, or list.

    Returns:
        MLX array.

    Raises:
        TypeError: If input type is not supported.
    """
    if isinstance(data, mx.array):
        return data
    if isinstance(data, np.ndarray):
        return mx.array(data)
    if isinstance(data, list):
        return mx.array(data)
    raise TypeError(f"Unsupported data type: {type(data)}")
