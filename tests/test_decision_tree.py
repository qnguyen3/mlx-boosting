"""Tests for Decision Tree implementations."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_boosting.trees import DecisionTreeClassifier, DecisionTreeRegressor


class TestDecisionTreeRegressor:
    """Tests for DecisionTreeRegressor."""

    def test_fit_basic(self) -> None:
        """Test basic fitting on simple data."""
        X = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = mx.array([1.0, 2.0, 3.0, 4.0])

        model = DecisionTreeRegressor(max_depth=3)
        model.fit(X, y)

        assert model.tree_ is not None
        assert model.n_features_in_ == 2

    def test_predict_shape(self) -> None:
        """Test prediction output shape."""
        X = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = mx.array([1.0, 2.0, 3.0, 4.0])

        model = DecisionTreeRegressor(max_depth=3)
        model.fit(X, y)

        predictions = model.predict(X)
        mx.eval(predictions)

        assert predictions.shape == (4,)

    def test_predict_not_fitted(self) -> None:
        """Test error when predicting without fitting."""
        model = DecisionTreeRegressor()
        X = mx.array([[1.0, 2.0]])

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

    def test_perfect_fit_depth1(self) -> None:
        """Test perfect fit on linearly separable data with depth 1."""
        # Data perfectly separable by feature 0 at threshold 2.5
        X = mx.array([[1.0], [2.0], [3.0], [4.0]])
        y = mx.array([0.0, 0.0, 1.0, 1.0])

        model = DecisionTreeRegressor(max_depth=2, min_samples_leaf=1)
        model.fit(X, y)

        predictions = model.predict(X)
        mx.eval(predictions)

        # Check predictions are reasonable (close to means of each group)
        assert float(predictions[0]) < 0.5
        assert float(predictions[1]) < 0.5
        assert float(predictions[2]) > 0.5
        assert float(predictions[3]) > 0.5

    def test_single_sample_leaf(self) -> None:
        """Test with min_samples_leaf=1 allows single sample leaves."""
        X = mx.array([[1.0], [2.0], [3.0]])
        y = mx.array([1.0, 2.0, 3.0])

        model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=1)
        model.fit(X, y)

        predictions = model.predict(X)
        mx.eval(predictions)

        # With enough depth, should fit training data exactly
        mse = float(mx.mean((predictions - y) ** 2))
        assert mse < 0.5  # Should be close to perfect

    def test_numpy_input(self) -> None:
        """Test that numpy arrays are accepted."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0])

        model = DecisionTreeRegressor(max_depth=3)
        model.fit(X, y)

        predictions = model.predict(X)
        mx.eval(predictions)

        assert predictions.shape == (4,)

    def test_list_input(self) -> None:
        """Test that lists are accepted."""
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        y = [1.0, 2.0, 3.0, 4.0]

        model = DecisionTreeRegressor(max_depth=3)
        model.fit(X, y)

        predictions = model.predict(X)
        mx.eval(predictions)

        assert predictions.shape == (4,)

    def test_1d_features(self) -> None:
        """Test that 1D features are reshaped correctly."""
        X = mx.array([1.0, 2.0, 3.0, 4.0])
        y = mx.array([1.0, 2.0, 3.0, 4.0])

        model = DecisionTreeRegressor(max_depth=3)
        model.fit(X, y)

        assert model.n_features_in_ == 1

    def test_exact_split_method(self) -> None:
        """Test exact split method."""
        X = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = mx.array([1.0, 2.0, 3.0, 4.0])

        model = DecisionTreeRegressor(max_depth=3, split_method="exact")
        model.fit(X, y)

        predictions = model.predict(X)
        mx.eval(predictions)

        assert predictions.shape == (4,)

    def test_min_samples_split(self) -> None:
        """Test min_samples_split constraint."""
        X = mx.array([[1.0], [2.0], [3.0], [4.0]])
        y = mx.array([1.0, 2.0, 3.0, 4.0])

        # With min_samples_split=5, root should be a leaf
        model = DecisionTreeRegressor(max_depth=5, min_samples_split=5)
        model.fit(X, y)

        predictions = model.predict(X)
        mx.eval(predictions)

        # All predictions should be the mean
        mean_y = float(mx.mean(y))
        for pred in predictions:
            assert abs(float(pred) - mean_y) < 0.01

    def test_get_params(self) -> None:
        """Test get_params method."""
        model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=3)
        params = model.get_params()

        assert params["max_depth"] == 5
        assert params["min_samples_leaf"] == 3


class TestDecisionTreeClassifier:
    """Tests for DecisionTreeClassifier."""

    def test_fit_basic(self) -> None:
        """Test basic fitting on simple data."""
        X = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = mx.array([0, 0, 1, 1])

        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X, y)

        assert model.tree_ is not None
        assert model.n_features_in_ == 2
        assert model.n_classes_ == 2

    def test_predict_shape(self) -> None:
        """Test prediction output shape."""
        X = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = mx.array([0, 0, 1, 1])

        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X, y)

        predictions = model.predict(X)
        mx.eval(predictions)

        assert predictions.shape == (4,)

    def test_predict_proba_shape(self) -> None:
        """Test probability prediction output shape."""
        X = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = mx.array([0, 0, 1, 1])

        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X, y)

        proba = model.predict_proba(X)
        mx.eval(proba)

        assert proba.shape == (4, 2)

    def test_predict_proba_sums_to_one(self) -> None:
        """Test that probabilities sum to 1."""
        X = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = mx.array([0, 0, 1, 1])

        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X, y)

        proba = model.predict_proba(X)
        mx.eval(proba)

        sums = mx.sum(proba, axis=1)
        mx.eval(sums)

        for s in sums:
            assert abs(float(s) - 1.0) < 0.01

    def test_predict_not_fitted(self) -> None:
        """Test error when predicting without fitting."""
        model = DecisionTreeClassifier()
        X = mx.array([[1.0, 2.0]])

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

    def test_multiclass(self) -> None:
        """Test multiclass classification."""
        X = mx.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y = mx.array([0, 0, 1, 1, 2, 2])

        model = DecisionTreeClassifier(max_depth=5)
        model.fit(X, y)

        assert model.n_classes_ == 3

        proba = model.predict_proba(X)
        mx.eval(proba)

        assert proba.shape == (6, 3)

    def test_gini_criterion(self) -> None:
        """Test Gini criterion."""
        X = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = mx.array([0, 0, 1, 1])

        model = DecisionTreeClassifier(max_depth=3, criterion="gini")
        model.fit(X, y)

        predictions = model.predict(X)
        mx.eval(predictions)

        assert predictions.shape == (4,)

    def test_entropy_criterion(self) -> None:
        """Test entropy criterion."""
        X = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = mx.array([0, 0, 1, 1])

        model = DecisionTreeClassifier(max_depth=3, criterion="entropy")
        model.fit(X, y)

        predictions = model.predict(X)
        mx.eval(predictions)

        assert predictions.shape == (4,)

    def test_numpy_input(self) -> None:
        """Test that numpy arrays are accepted."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([0, 0, 1, 1])

        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X, y)

        predictions = model.predict(X)
        mx.eval(predictions)

        assert predictions.shape == (4,)


class TestTreeStructure:
    """Tests for tree structure internals."""

    def test_tree_arrays_created(self) -> None:
        """Test that tree arrays are properly created."""
        X = mx.array([[1.0], [2.0], [3.0], [4.0]])
        y = mx.array([1.0, 2.0, 3.0, 4.0])

        model = DecisionTreeRegressor(max_depth=2)
        model.fit(X, y)

        tree = model.tree_
        assert tree is not None
        assert tree.feature_indices is not None
        assert tree.thresholds is not None
        assert tree.left_children is not None
        assert tree.right_children is not None
        assert tree.values is not None
        assert tree.is_leaf is not None

    def test_leaf_nodes_marked(self) -> None:
        """Test that leaf nodes are properly marked."""
        X = mx.array([[1.0], [2.0], [3.0], [4.0]])
        y = mx.array([1.0, 1.0, 2.0, 2.0])

        model = DecisionTreeRegressor(max_depth=1)
        model.fit(X, y)

        tree = model.tree_
        assert tree is not None

        # Root should not be a leaf (if split was made)
        # Children should be leaves
        mx.eval(tree.is_leaf)

        # At least some nodes should be leaves
        assert mx.any(tree.is_leaf)


class TestLargerDatasets:
    """Tests with larger datasets to verify GPU performance."""

    def test_medium_dataset_regression(self) -> None:
        """Test regression on medium-sized dataset."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        X = mx.array(np.random.randn(n_samples, n_features).astype(np.float32))
        y = mx.array(np.random.randn(n_samples).astype(np.float32))

        model = DecisionTreeRegressor(max_depth=6, n_bins=128)
        model.fit(X, y)

        predictions = model.predict(X)
        mx.eval(predictions)

        assert predictions.shape == (n_samples,)

    def test_medium_dataset_classification(self) -> None:
        """Test classification on medium-sized dataset."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        n_classes = 3

        X = mx.array(np.random.randn(n_samples, n_features).astype(np.float32))
        y = mx.array(np.random.randint(0, n_classes, n_samples))

        model = DecisionTreeClassifier(max_depth=6, n_bins=128)
        model.fit(X, y)

        predictions = model.predict(X)
        proba = model.predict_proba(X)
        mx.eval(predictions, proba)

        assert predictions.shape == (n_samples,)
        assert proba.shape == (n_samples, n_classes)
