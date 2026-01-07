"""Classification loss functions."""

import mlx.core as mx


class LogLoss:
    """Logistic loss for binary classification tasks."""

    @staticmethod
    def loss(y_true: mx.array, y_pred: mx.array) -> mx.array:
        """Compute log loss.

        Args:
            y_true: Ground truth labels (0 or 1).
            y_pred: Predicted log-odds.

        Returns:
            Log loss value.
        """
        prob = mx.sigmoid(y_pred)
        prob = mx.clip(prob, 1e-15, 1 - 1e-15)
        return -mx.mean(y_true * mx.log(prob) + (1 - y_true) * mx.log(1 - prob))

    @staticmethod
    def gradient(y_true: mx.array, y_pred: mx.array) -> mx.array:
        """Compute gradient of log loss.

        Args:
            y_true: Ground truth labels (0 or 1).
            y_pred: Predicted log-odds.

        Returns:
            Gradient with respect to predictions.
        """
        prob = mx.sigmoid(y_pred)
        return prob - y_true

    @staticmethod
    def hessian(y_true: mx.array, y_pred: mx.array) -> mx.array:
        """Compute hessian of log loss.

        Args:
            y_true: Ground truth labels (unused).
            y_pred: Predicted log-odds.

        Returns:
            Hessian values.
        """
        prob = mx.sigmoid(y_pred)
        return prob * (1 - prob)
