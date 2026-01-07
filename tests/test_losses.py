"""Tests for loss functions."""

import mlx.core as mx

from mlx_boosting.losses import MSELoss
from mlx_boosting.losses.classification import LogLoss


class TestMSELoss:
    """Tests for MSE loss."""

    def test_loss_zero(self) -> None:
        """Test MSE loss when predictions match targets."""
        y_true = mx.array([1.0, 2.0, 3.0])
        y_pred = mx.array([1.0, 2.0, 3.0])
        loss = MSELoss.loss(y_true, y_pred)
        mx.eval(loss)
        assert float(loss) == 0.0

    def test_loss_nonzero(self) -> None:
        """Test MSE loss with prediction errors."""
        y_true = mx.array([1.0, 2.0, 3.0])
        y_pred = mx.array([2.0, 3.0, 4.0])
        loss = MSELoss.loss(y_true, y_pred)
        mx.eval(loss)
        assert float(loss) == 1.0

    def test_gradient(self) -> None:
        """Test MSE gradient computation."""
        y_true = mx.array([1.0, 2.0, 3.0])
        y_pred = mx.array([2.0, 3.0, 4.0])
        grad = MSELoss.gradient(y_true, y_pred)
        mx.eval(grad)
        expected = mx.array([1.0, 1.0, 1.0])
        assert mx.allclose(grad, expected)

    def test_hessian(self) -> None:
        """Test MSE hessian computation."""
        y_true = mx.array([1.0, 2.0, 3.0])
        y_pred = mx.array([2.0, 3.0, 4.0])
        hess = MSELoss.hessian(y_true, y_pred)
        mx.eval(hess)
        expected = mx.ones(3)
        assert mx.allclose(hess, expected)


class TestLogLoss:
    """Tests for log loss."""

    def test_gradient_shape(self) -> None:
        """Test log loss gradient has correct shape."""
        y_true = mx.array([0.0, 1.0, 1.0, 0.0])
        y_pred = mx.array([0.1, 0.9, 0.8, 0.2])
        grad = LogLoss.gradient(y_true, y_pred)
        mx.eval(grad)
        assert grad.shape == y_true.shape

    def test_hessian_positive(self) -> None:
        """Test log loss hessian is always positive."""
        y_true = mx.array([0.0, 1.0, 1.0, 0.0])
        y_pred = mx.array([-1.0, 1.0, 0.5, -0.5])
        hess = LogLoss.hessian(y_true, y_pred)
        mx.eval(hess)
        assert mx.all(hess > 0)
