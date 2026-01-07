"""Base classes for MLX Boosting estimators."""

from abc import ABC, abstractmethod
from typing import Any

import mlx.core as mx


class BaseEstimator(ABC):
    """Abstract base class for all MLX Boosting estimators.

    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword arguments.
    """

    @abstractmethod
    def fit(self, X: mx.array, y: mx.array) -> "BaseEstimator":
        """Fit the model to training data.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).

        Returns:
            Self for method chaining.
        """

    @abstractmethod
    def predict(self, X: mx.array) -> mx.array:
        """Make predictions on new data.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """

    def get_params(self) -> dict[str, Any]:
        """Get parameters for this estimator.

        Returns:
            Parameter names mapped to their values.
        """
        return {
            key: getattr(self, key)
            for key in self.__init__.__code__.co_varnames[1:]
            if hasattr(self, key)
        }

    def set_params(self, **params: Any) -> "BaseEstimator":
        """Set parameters for this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            Self for method chaining.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
