"""Regression loss functions."""

import mlx.core as mx


class MSELoss:
    """Mean Squared Error loss for regression tasks."""

    @staticmethod
    def loss(y_true: mx.array, y_pred: mx.array) -> mx.array:
        """Compute MSE loss.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.

        Returns:
            Mean squared error.
        """
        return mx.mean((y_true - y_pred) ** 2)

    @staticmethod
    def gradient(y_true: mx.array, y_pred: mx.array) -> mx.array:
        """Compute gradient of MSE loss.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.

        Returns:
            Gradient with respect to predictions.
        """
        return y_pred - y_true

    @staticmethod
    def hessian(y_true: mx.array, y_pred: mx.array) -> mx.array:
        """Compute hessian of MSE loss.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.

        Returns:
            Hessian (constant 1 for MSE).
        """
        return mx.ones_like(y_pred)
