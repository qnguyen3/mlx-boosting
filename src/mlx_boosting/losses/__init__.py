"""Loss functions for MLX Boosting."""

from mlx_boosting.losses.classification import LogLoss, SoftmaxLoss
from mlx_boosting.losses.regression import MSELoss

__all__ = ["MSELoss", "LogLoss", "SoftmaxLoss"]
