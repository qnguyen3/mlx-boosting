"""Evaluation metrics for MLX Boosting."""

import mlx.core as mx


def mse(y_true: mx.array, y_pred: mx.array) -> mx.array:
    """Compute Mean Squared Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MSE value.
    """
    return mx.mean((y_true - y_pred) ** 2)


def rmse(y_true: mx.array, y_pred: mx.array) -> mx.array:
    """Compute Root Mean Squared Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        RMSE value.
    """
    return mx.sqrt(mse(y_true, y_pred))


def mae(y_true: mx.array, y_pred: mx.array) -> mx.array:
    """Compute Mean Absolute Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MAE value.
    """
    return mx.mean(mx.abs(y_true - y_pred))


def accuracy(y_true: mx.array, y_pred: mx.array) -> mx.array:
    """Compute classification accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy value.
    """
    return mx.mean(y_true == y_pred)
