"""MLX Boosting - GPU-accelerated boosting algorithms for Apple Silicon."""

from mlx_boosting.base import BaseEstimator
from mlx_boosting.gbdt import GradientBoostingClassifier, GradientBoostingRegressor
from mlx_boosting.trees import DecisionTreeClassifier, DecisionTreeRegressor
from mlx_boosting.xgboost import XGBoostClassifier, XGBoostRegressor

__version__ = "1.0.0"
__all__ = [
    "BaseEstimator",
    "DecisionTreeRegressor",
    "DecisionTreeClassifier",
    "GradientBoostingRegressor",
    "GradientBoostingClassifier",
    "XGBoostRegressor",
    "XGBoostClassifier",
    "__version__",
]
