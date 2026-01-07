"""Tests for XGBoost implementation."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_boosting import XGBoostClassifier, XGBoostRegressor


class TestXGBoostRegressor:
    """Tests for XGBoostRegressor."""

    def test_fit_basic(self) -> None:
        """Test basic fitting."""
        X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=mx.float32)
        y = mx.array([1.0, 2.0, 3.0, 4.0])

        model = XGBoostRegressor(n_estimators=10, max_depth=3)
        model.fit(X, y)

        assert len(model.trees_) == 10
        assert model.n_features_in_ == 2

    def test_predict_shape(self) -> None:
        """Test prediction shape."""
        X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=mx.float32)
        y = mx.array([1.0, 2.0, 3.0, 4.0])

        model = XGBoostRegressor(n_estimators=5, max_depth=2)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (4,)

    def test_predict_not_fitted(self) -> None:
        """Test error when predicting before fitting."""
        model = XGBoostRegressor()
        X = mx.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

    def test_train_score_decreasing(self) -> None:
        """Test that training loss generally decreases."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 5).astype(np.float32))
        y = mx.array(np.random.randn(100).astype(np.float32))

        model = XGBoostRegressor(n_estimators=20, learning_rate=0.1, max_depth=4)
        model.fit(X, y)

        # Loss should decrease overall
        assert model.train_score_[-1] < model.train_score_[0]

    def test_overfitting_capability(self) -> None:
        """Test model can overfit training data."""
        np.random.seed(42)
        X = mx.array(np.linspace(0, 5, 20).reshape(-1, 1).astype(np.float32))
        y = mx.array((np.linspace(0, 5, 20) ** 2).astype(np.float32))

        model = XGBoostRegressor(n_estimators=100, learning_rate=0.2, max_depth=4)
        model.fit(X, y)

        preds = model.predict(X)
        mx.eval(preds)

        # Should fit training data reasonably well
        mse = float(mx.mean((preds - y) ** 2))
        assert mse < 10.0

    def test_numpy_input(self) -> None:
        """Test with numpy input."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        model = XGBoostRegressor(n_estimators=5, max_depth=2)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (4,)

    def test_subsample(self) -> None:
        """Test with row subsampling."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 5).astype(np.float32))
        y = mx.array(np.random.randn(100).astype(np.float32))

        model = XGBoostRegressor(n_estimators=10, subsample=0.8, max_depth=3)
        model.fit(X, y)

        assert len(model.trees_) == 10

    def test_colsample_bytree(self) -> None:
        """Test with feature subsampling per tree."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 10).astype(np.float32))
        y = mx.array(np.random.randn(100).astype(np.float32))

        model = XGBoostRegressor(n_estimators=10, colsample_bytree=0.5, max_depth=3)
        model.fit(X, y)

        assert len(model.trees_) == 10
        # Check that feature subsets were used
        for feat_idx in model._feature_indices:
            assert len(feat_idx) <= 5  # 50% of 10 features

    def test_regularization_lambda(self) -> None:
        """Test L2 regularization effect."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 5).astype(np.float32))
        y = mx.array(np.random.randn(100).astype(np.float32))

        # No regularization
        model_no_reg = XGBoostRegressor(
            n_estimators=20, reg_lambda=0.0, learning_rate=0.3
        )
        model_no_reg.fit(X, y)

        # With L2 regularization
        model_l2 = XGBoostRegressor(n_estimators=20, reg_lambda=10.0, learning_rate=0.3)
        model_l2.fit(X, y)

        # Regularization should result in higher training loss (less overfitting)
        assert model_l2.train_score_[-1] >= model_no_reg.train_score_[-1] * 0.5

    def test_regularization_alpha(self) -> None:
        """Test L1 regularization effect."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 5).astype(np.float32))
        y = mx.array(np.random.randn(100).astype(np.float32))

        # With L1 regularization
        model = XGBoostRegressor(n_estimators=20, reg_alpha=1.0, learning_rate=0.3)
        model.fit(X, y)

        # Model should still train
        assert model.train_score_[-1] < model.train_score_[0]

    def test_gamma_min_gain(self) -> None:
        """Test gamma (minimum gain) parameter."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 5).astype(np.float32))
        y = mx.array(np.random.randn(100).astype(np.float32))

        # High gamma should reduce tree complexity
        model = XGBoostRegressor(n_estimators=10, gamma=10.0, max_depth=4)
        model.fit(X, y)

        assert len(model.trees_) == 10

    def test_min_child_weight(self) -> None:
        """Test min_child_weight parameter."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 5).astype(np.float32))
        y = mx.array(np.random.randn(100).astype(np.float32))

        # High min_child_weight should limit tree depth
        model = XGBoostRegressor(n_estimators=10, min_child_weight=10.0, max_depth=6)
        model.fit(X, y)

        assert len(model.trees_) == 10

    def test_get_params(self) -> None:
        """Test get_params method."""
        model = XGBoostRegressor(n_estimators=50, learning_rate=0.05, reg_lambda=2.0)
        params = model.get_params()

        assert params["n_estimators"] == 50
        assert params["learning_rate"] == 0.05
        assert params["reg_lambda"] == 2.0


class TestXGBoostClassifier:
    """Tests for XGBoostClassifier."""

    def test_fit_binary_basic(self) -> None:
        """Test basic binary classification fitting."""
        X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=mx.float32)
        y = mx.array([0, 0, 1, 1])

        model = XGBoostClassifier(n_estimators=10, max_depth=3)
        model.fit(X, y)

        assert len(model.trees_) == 10
        assert model.n_features_in_ == 2
        assert model.n_classes_ == 2
        assert model.classes_ is not None

    def test_predict_shape_binary(self) -> None:
        """Test binary prediction shape."""
        X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=mx.float32)
        y = mx.array([0, 0, 1, 1])

        model = XGBoostClassifier(n_estimators=5, max_depth=2)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (4,)

    def test_predict_proba_shape_binary(self) -> None:
        """Test binary predict_proba shape."""
        X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=mx.float32)
        y = mx.array([0, 0, 1, 1])

        model = XGBoostClassifier(n_estimators=5, max_depth=2)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (4, 2)

    def test_predict_proba_sums_to_one(self) -> None:
        """Test probabilities sum to 1."""
        X = mx.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=mx.float32)
        y = mx.array([0, 0, 1, 1])

        model = XGBoostClassifier(n_estimators=5, max_depth=2)
        model.fit(X, y)

        proba = model.predict_proba(X)
        sums = mx.sum(proba, axis=1)
        mx.eval(sums)

        assert mx.allclose(sums, mx.ones((4,)), atol=1e-5)

    def test_predict_not_fitted(self) -> None:
        """Test error when predicting before fitting."""
        model = XGBoostClassifier()
        X = mx.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

    def test_multiclass_basic(self) -> None:
        """Test multi-class classification."""
        X = mx.array(
            [
                [1, 2],
                [2, 1],
                [1, 1],
                [5, 6],
                [6, 5],
                [5, 5],
                [10, 10],
                [11, 10],
                [10, 11],
            ],
            dtype=mx.float32,
        )
        y = mx.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        model = XGBoostClassifier(n_estimators=20, max_depth=3)
        model.fit(X, y)

        assert model.n_classes_ == 3
        assert len(model.trees_) == 20
        # Each iteration should have 3 trees (one per class)
        for tree_list in model.trees_:
            assert len(tree_list) == 3

    def test_multiclass_predict_proba_shape(self) -> None:
        """Test multi-class predict_proba shape."""
        X = mx.array(
            [[1, 2], [2, 1], [5, 6], [6, 5], [10, 10], [11, 10]], dtype=mx.float32
        )
        y = mx.array([0, 0, 1, 1, 2, 2])

        model = XGBoostClassifier(n_estimators=10, max_depth=2)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (6, 3)

    def test_multiclass_proba_sums_to_one(self) -> None:
        """Test multi-class probabilities sum to 1."""
        X = mx.array(
            [[1, 2], [2, 1], [5, 6], [6, 5], [10, 10], [11, 10]], dtype=mx.float32
        )
        y = mx.array([0, 0, 1, 1, 2, 2])

        model = XGBoostClassifier(n_estimators=10, max_depth=2)
        model.fit(X, y)

        proba = model.predict_proba(X)
        sums = mx.sum(proba, axis=1)
        mx.eval(sums)

        assert mx.allclose(sums, mx.ones((6,)), atol=1e-5)

    def test_train_score_decreasing(self) -> None:
        """Test that training loss generally decreases."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 5).astype(np.float32))
        y = mx.array((np.random.randn(100) > 0).astype(np.int32))

        model = XGBoostClassifier(n_estimators=20, learning_rate=0.1, max_depth=4)
        model.fit(X, y)

        # Loss should decrease overall
        assert model.train_score_[-1] < model.train_score_[0]

    def test_classification_accuracy_binary(self) -> None:
        """Test reasonable binary classification accuracy."""
        np.random.seed(42)
        X_class0 = np.random.randn(50, 2).astype(np.float32) + np.array([[-2, -2]])
        X_class1 = np.random.randn(50, 2).astype(np.float32) + np.array([[2, 2]])
        X = mx.array(np.vstack([X_class0, X_class1]))
        y = mx.array(np.array([0] * 50 + [1] * 50))

        model = XGBoostClassifier(n_estimators=30, learning_rate=0.1, max_depth=4)
        model.fit(X, y)

        preds = model.predict(X)
        mx.eval(preds)

        accuracy = float(mx.mean((preds == y).astype(mx.float32)))
        assert accuracy > 0.9

    def test_classification_accuracy_multiclass(self) -> None:
        """Test reasonable multi-class classification accuracy."""
        np.random.seed(42)
        # Create 3 well-separated clusters
        X_class0 = np.random.randn(30, 2).astype(np.float32) + np.array([[0, 0]])
        X_class1 = np.random.randn(30, 2).astype(np.float32) + np.array([[5, 5]])
        X_class2 = np.random.randn(30, 2).astype(np.float32) + np.array([[10, 0]])
        X = mx.array(np.vstack([X_class0, X_class1, X_class2]))
        y = mx.array(np.array([0] * 30 + [1] * 30 + [2] * 30))

        model = XGBoostClassifier(n_estimators=30, learning_rate=0.1, max_depth=4)
        model.fit(X, y)

        preds = model.predict(X)
        mx.eval(preds)

        accuracy = float(mx.mean((preds == y).astype(mx.float32)))
        assert accuracy > 0.85

    def test_numpy_input(self) -> None:
        """Test with numpy input."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])

        model = XGBoostClassifier(n_estimators=5, max_depth=2)
        model.fit(X, y)

        preds = model.predict(X)
        proba = model.predict_proba(X)

        assert preds.shape == (4,)
        assert proba.shape == (4, 2)

    def test_subsample(self) -> None:
        """Test with row subsampling."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 5).astype(np.float32))
        y = mx.array((np.random.randn(100) > 0).astype(np.int32))

        model = XGBoostClassifier(n_estimators=10, subsample=0.8, max_depth=3)
        model.fit(X, y)

        assert len(model.trees_) == 10

    def test_colsample_bytree(self) -> None:
        """Test with feature subsampling per tree."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 10).astype(np.float32))
        y = mx.array((np.random.randn(100) > 0).astype(np.int32))

        model = XGBoostClassifier(n_estimators=10, colsample_bytree=0.5, max_depth=3)
        model.fit(X, y)

        assert len(model.trees_) == 10
        # Check that feature subsets were used
        for feat_idx in model._feature_indices:
            assert len(feat_idx) <= 5  # 50% of 10 features


class TestXGBoostMissingValues:
    """Tests for XGBoost missing value handling."""

    def test_regressor_with_missing_values(self) -> None:
        """Test regressor handles missing values."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        # Introduce missing values
        X[10:20, 0] = np.nan
        X[30:40, 2] = np.nan
        X = mx.array(X)
        y = mx.array(np.random.randn(100).astype(np.float32))

        model = XGBoostRegressor(n_estimators=10, max_depth=3)
        model.fit(X, y)

        # Should train without errors
        assert len(model.trees_) == 10

        # Should predict without errors
        preds = model.predict(X)
        assert preds.shape == (100,)
        assert not mx.isnan(preds).any()

    def test_classifier_with_missing_values(self) -> None:
        """Test classifier handles missing values."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        # Introduce missing values
        X[10:20, 0] = np.nan
        X[30:40, 2] = np.nan
        X = mx.array(X)
        y = mx.array((np.random.randn(100) > 0).astype(np.int32))

        model = XGBoostClassifier(n_estimators=10, max_depth=3)
        model.fit(X, y)

        # Should train without errors
        assert len(model.trees_) == 10

        # Should predict without errors
        preds = model.predict(X)
        proba = model.predict_proba(X)
        assert preds.shape == (100,)
        assert proba.shape == (100, 2)
        assert not mx.isnan(proba).any()

    def test_predict_with_missing_not_in_training(self) -> None:
        """Test prediction with missing values not seen in training."""
        np.random.seed(42)
        # Train without missing values
        X_train = mx.array(np.random.randn(100, 5).astype(np.float32))
        y_train = mx.array(np.random.randn(100).astype(np.float32))

        model = XGBoostRegressor(n_estimators=10, max_depth=3)
        model.fit(X_train, y_train)

        # Predict with missing values
        X_test = np.random.randn(20, 5).astype(np.float32)
        X_test[5:10, 0] = np.nan
        X_test = mx.array(X_test)

        preds = model.predict(X_test)
        assert preds.shape == (20,)
        assert not mx.isnan(preds).any()


class TestXGBoostCorrectness:
    """Tests for verifying XGBoost algorithm correctness."""

    def test_xgboost_gain_formula(self) -> None:
        """Test that XGBoost gain formula is being applied correctly."""
        np.random.seed(42)
        X = mx.array(np.random.randn(200, 10).astype(np.float32))
        y = mx.array(np.random.randn(200).astype(np.float32))

        # Train model
        model = XGBoostRegressor(n_estimators=10, max_depth=5, reg_lambda=1.0)
        model.fit(X, y)

        # Predictions should be reasonable (not NaN, not extreme)
        preds = model.predict(X)
        mx.eval(preds)
        assert not mx.isnan(preds).any()
        assert float(mx.max(mx.abs(preds))) < 100

    def test_leaf_values_with_regularization(self) -> None:
        """Test that leaf values are computed with regularization."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 5).astype(np.float32))
        y = mx.array(np.random.randn(100).astype(np.float32))

        # With strong regularization, leaf values should be smaller
        model_high_reg = XGBoostRegressor(n_estimators=5, max_depth=3, reg_lambda=100.0)
        model_high_reg.fit(X, y)

        model_low_reg = XGBoostRegressor(n_estimators=5, max_depth=3, reg_lambda=0.001)
        model_low_reg.fit(X, y)

        # Check leaf values - high reg should have smaller magnitude values
        high_reg_max_leaf = max(
            float(mx.max(mx.abs(tree.values))) for tree in model_high_reg.trees_
        )
        low_reg_max_leaf = max(
            float(mx.max(mx.abs(tree.values))) for tree in model_low_reg.trees_
        )

        assert high_reg_max_leaf < low_reg_max_leaf

    def test_predictions_reasonable_range(self) -> None:
        """Test predictions are in reasonable range."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 5).astype(np.float32))
        # Target with known range
        y = mx.array((3 * np.random.randn(100) + 10).astype(np.float32))

        model = XGBoostRegressor(n_estimators=50, learning_rate=0.1, max_depth=4)
        model.fit(X, y)

        preds = model.predict(X)
        mx.eval(preds)

        # Predictions should be somewhat close to target mean
        pred_mean = float(mx.mean(preds))
        target_mean = float(mx.mean(y))
        assert abs(pred_mean - target_mean) < 5.0

    def test_learning_rate_effect(self) -> None:
        """Test that learning rate affects convergence."""
        np.random.seed(42)
        X = mx.array(np.random.randn(100, 5).astype(np.float32))
        y = mx.array(np.random.randn(100).astype(np.float32))

        # High learning rate
        model_high_lr = XGBoostRegressor(n_estimators=10, learning_rate=0.5)
        model_high_lr.fit(X, y)

        # Low learning rate
        model_low_lr = XGBoostRegressor(n_estimators=10, learning_rate=0.01)
        model_low_lr.fit(X, y)

        # High learning rate should fit faster initially
        assert model_high_lr.train_score_[5] < model_low_lr.train_score_[5]
