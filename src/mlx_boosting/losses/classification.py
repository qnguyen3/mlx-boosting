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


class SoftmaxLoss:
    """Softmax cross-entropy loss for multi-class classification.

    For K classes, XGBoost builds K trees per boosting iteration.
    Each tree fits the gradient for one class.
    """

    def __init__(self, n_classes: int) -> None:
        """Initialize softmax loss.

        Args:
            n_classes: Number of classes.
        """
        self.n_classes = n_classes

    def loss(self, y_true: mx.array, y_pred: mx.array) -> mx.array:
        """Compute softmax cross-entropy loss.

        Args:
            y_true: Ground truth class labels (integers), shape (n_samples,).
            y_pred: Predicted raw scores, shape (n_samples, n_classes).

        Returns:
            Mean cross-entropy loss.
        """
        # Softmax for numerical stability
        y_pred_max = mx.max(y_pred, axis=1, keepdims=True)
        y_pred_stable = y_pred - y_pred_max
        exp_pred = mx.exp(y_pred_stable)
        softmax_probs = exp_pred / mx.sum(exp_pred, axis=1, keepdims=True)

        # Clip for numerical stability
        softmax_probs = mx.clip(softmax_probs, 1e-15, 1.0)

        # Cross-entropy: -log(p_k) for true class k
        n_samples = y_true.shape[0]
        # Create indices for gathering
        row_indices = mx.arange(n_samples)
        true_class_probs = softmax_probs[row_indices, y_true.astype(mx.int32)]

        return -mx.mean(mx.log(true_class_probs))

    def gradient(self, y_true: mx.array, y_pred: mx.array) -> mx.array:
        """Compute gradient of softmax cross-entropy loss.

        For XGBoost, we need gradients for each class separately.
        g_ik = p_k - 1 if k is the true class, else p_k

        Args:
            y_true: Ground truth class labels, shape (n_samples,).
            y_pred: Predicted raw scores, shape (n_samples, n_classes).

        Returns:
            Gradient with shape (n_samples, n_classes).
        """
        # Softmax probabilities
        y_pred_max = mx.max(y_pred, axis=1, keepdims=True)
        y_pred_stable = y_pred - y_pred_max
        exp_pred = mx.exp(y_pred_stable)
        probs = exp_pred / mx.sum(exp_pred, axis=1, keepdims=True)

        # Gradient: p_k - y_k (one-hot encoded)
        # For true class: p_k - 1
        # For other classes: p_k
        n_samples = y_true.shape[0]
        grad = probs.astype(mx.float32)

        # Subtract 1 from the true class
        mx.arange(n_samples)
        # Use scatter-like operation
        one_hot = mx.zeros((n_samples, self.n_classes), dtype=mx.float32)
        one_hot = mx.where(
            mx.arange(self.n_classes)[None, :] == y_true[:, None].astype(mx.int32),
            mx.ones_like(one_hot),
            mx.zeros_like(one_hot),
        )
        grad = grad - one_hot

        return grad

    def hessian(self, y_true: mx.array, y_pred: mx.array) -> mx.array:
        """Compute hessian of softmax cross-entropy loss.

        For XGBoost, the diagonal Hessian is: h_ik = p_k * (1 - p_k)

        Args:
            y_true: Ground truth class labels (unused in computation).
            y_pred: Predicted raw scores, shape (n_samples, n_classes).

        Returns:
            Hessian values with shape (n_samples, n_classes).
        """
        # Softmax probabilities
        y_pred_max = mx.max(y_pred, axis=1, keepdims=True)
        y_pred_stable = y_pred - y_pred_max
        exp_pred = mx.exp(y_pred_stable)
        probs = exp_pred / mx.sum(exp_pred, axis=1, keepdims=True)

        # Diagonal Hessian: p_k * (1 - p_k)
        hess = probs * (1.0 - probs)

        # Ensure minimum value for numerical stability
        hess = mx.maximum(hess, 1e-6)

        return hess
