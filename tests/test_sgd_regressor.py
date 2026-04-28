import numpy as np
import pytest
from simpleml.linear_models import SGDRegressor


def test_sgd_regressor_learns_simple_linear_relationship():
    X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
    y = np.array([2, 4, 6, 8, 10, 12], dtype=float)

    model = SGDRegressor(
        learning_rate=0.01,
        max_iter=1000,
        batch_size=2,
        l2_ratio=0.0,
        random_state=42
    )

    model.fit(X, y)
    preds = model.predict(X)

    mse = np.mean((y - preds) ** 2)

    assert mse < 1.0
    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert model.n_features_in_ == 1


def test_sgd_regressor_predict_before_fit_raises_error():
    model = SGDRegressor()

    with pytest.raises(ValueError):
        model.predict(np.array([[1], [2]]))