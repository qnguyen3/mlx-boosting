"""Tests for Gradient Boosting Decision Trees."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_boosting import GradientBoostingClassifier, GradientBoostingRegressor


class TestGradientBoostingRegressor:
    """Tests for GradientBoostingRegressor."""

    def test_fit_basic(self) -> None:
        """Test basic fitting."""
        X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=mx.float32)
        y = mx.array([1.0, 2.0, 3.0, 4.0])

        model = GradientBoostingRegressor(n_estimators=10, max_depth=3)
        model.fit(X, y)

        assert len(model.trees_) == 10
        assert model.n_features_in_ == 2

    def test_predict_shape(self) -> None:
        """Test prediction shape."""
        X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=mx.float32)
        y = mx.array([1.0, 2.0, 3.0, 4.0])

        model = GradientBoostingRegressor(n_estimators=5, max_depth=2)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (4,)

    def test_predict_not_fitted(self) -> None:
        """Test error when predicting before fitting."""
        model = GradientBoostingRegressor()
        X = mx.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

    def test_train_score_decreasing(self) -> None:
        """Test that training loss generally decreases."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 5).astype(np.float32))
        y = mx.array(np.random.randn(100).astype(np.float32))

        model = GradientBoostingRegressor(
            n_estimators=20, learning_rate=0.1, max_depth=4
        )
        model.fit(X, y)

        # Loss should decrease overall
        assert model.train_score_[-1] < model.train_score_[0]

    def test_overfitting_capability(self) -> None:
        """Test model can overfit training data."""
        # Use more data points for better fitting
        np.random.seed(42)
        X = mx.array(np.linspace(0, 5, 20).reshape(-1, 1).astype(np.float32))
        y = mx.array((np.linspace(0, 5, 20) ** 2).astype(np.float32))

        model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.2, max_depth=4
        )
        model.fit(X, y)

        preds = model.predict(X)
        mx.eval(preds)

        # Should fit training data reasonably well
        mse = float(mx.mean((preds - y) ** 2))
        assert mse < 10.0  # Reasonable fit

    def test_numpy_input(self) -> None:
        """Test with numpy input."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        model = GradientBoostingRegressor(n_estimators=5, max_depth=2)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (4,)

    def test_subsample(self) -> None:
        """Test with subsampling."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 5).astype(np.float32))
        y = mx.array(np.random.randn(100).astype(np.float32))

        model = GradientBoostingRegressor(n_estimators=10, subsample=0.8, max_depth=3)
        model.fit(X, y)

        assert len(model.trees_) == 10

    def test_get_params(self) -> None:
        """Test get_params method."""
        model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.05)
        params = model.get_params()

        assert params["n_estimators"] == 50
        assert params["learning_rate"] == 0.05


class TestGradientBoostingClassifier:
    """Tests for GradientBoostingClassifier."""

    def test_fit_basic(self) -> None:
        """Test basic fitting."""
        X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=mx.float32)
        y = mx.array([0, 0, 1, 1])

        model = GradientBoostingClassifier(n_estimators=10, max_depth=3)
        model.fit(X, y)

        assert len(model.trees_) == 10
        assert model.n_features_in_ == 2
        assert model.classes_ is not None

    def test_predict_shape(self) -> None:
        """Test prediction shape."""
        X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=mx.float32)
        y = mx.array([0, 0, 1, 1])

        model = GradientBoostingClassifier(n_estimators=5, max_depth=2)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (4,)

    def test_predict_proba_shape(self) -> None:
        """Test predict_proba shape."""
        X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=mx.float32)
        y = mx.array([0, 0, 1, 1])

        model = GradientBoostingClassifier(n_estimators=5, max_depth=2)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (4, 2)

    def test_predict_proba_sums_to_one(self) -> None:
        """Test probabilities sum to 1."""
        X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=mx.float32)
        y = mx.array([0, 0, 1, 1])

        model = GradientBoostingClassifier(n_estimators=5, max_depth=2)
        model.fit(X, y)

        proba = model.predict_proba(X)
        sums = mx.sum(proba, axis=1)
        mx.eval(sums)

        assert mx.allclose(sums, mx.ones((4,)))

    def test_predict_not_fitted(self) -> None:
        """Test error when predicting before fitting."""
        model = GradientBoostingClassifier()
        X = mx.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

    def test_binary_only(self) -> None:
        """Test error for non-binary classification."""
        X = mx.array([[1, 2], [3, 4], [5, 6]])
        y = mx.array([0, 1, 2])  # 3 classes

        model = GradientBoostingClassifier()

        with pytest.raises(ValueError, match="binary"):
            model.fit(X, y)

    def test_train_score_decreasing(self) -> None:
        """Test that training loss generally decreases."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 5).astype(np.float32))
        y = mx.array((np.random.randn(100) > 0).astype(np.int32))

        model = GradientBoostingClassifier(
            n_estimators=20, learning_rate=0.1, max_depth=4
        )
        model.fit(X, y)

        # Loss should decrease overall
        assert model.train_score_[-1] < model.train_score_[0]

    def test_classification_accuracy(self) -> None:
        """Test reasonable classification accuracy."""
        # Simple linearly separable data
        np.random.seed(42)
        X_class0 = np.random.randn(50, 2).astype(np.float32) + np.array([[-2, -2]])
        X_class1 = np.random.randn(50, 2).astype(np.float32) + np.array([[2, 2]])
        X = mx.array(np.vstack([X_class0, X_class1]))
        y = mx.array(np.array([0] * 50 + [1] * 50))

        model = GradientBoostingClassifier(
            n_estimators=30, learning_rate=0.1, max_depth=4
        )
        model.fit(X, y)

        preds = model.predict(X)
        mx.eval(preds)

        accuracy = float(mx.mean((preds == y).astype(mx.float32)))
        assert accuracy > 0.9  # Should achieve high accuracy on easy data

    def test_numpy_input(self) -> None:
        """Test with numpy input."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])

        model = GradientBoostingClassifier(n_estimators=5, max_depth=2)
        model.fit(X, y)

        preds = model.predict(X)
        proba = model.predict_proba(X)

        assert preds.shape == (4,)
        assert proba.shape == (4, 2)


class TestLevelwiseOptimization:
    """Tests verifying level-wise optimization produces correct results."""

    def test_levelwise_tree_correctness(self) -> None:
        """Test that level-wise tree produces reasonable predictions."""
        np.random.seed(42)
        X = mx.array(np.random.randn(200, 10).astype(np.float32))
        y = mx.array(np.random.randn(200).astype(np.float32))

        model = GradientBoostingRegressor(n_estimators=10, max_depth=5)
        model.fit(X, y)

        preds = model.predict(X)
        mx.eval(preds)

        # Predictions should be reasonable (not NaN, not extreme)
        assert not mx.isnan(preds).any()
        assert float(mx.max(mx.abs(preds))) < 100

    def test_histogram_subtraction_correctness(self) -> None:
        """Test that histogram subtraction produces valid results."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 5).astype(np.float32))
        y = mx.array(np.random.randn(100).astype(np.float32))

        # Train with optimized level-wise builder
        model = GradientBoostingRegressor(n_estimators=5, max_depth=4)
        model.fit(X, y)

        # Should reduce loss
        assert model.train_score_[-1] < model.train_score_[0]
