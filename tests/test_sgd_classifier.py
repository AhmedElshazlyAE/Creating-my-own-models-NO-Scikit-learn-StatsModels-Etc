import numpy as np
import pytest
from simpleml.linear_models import SGDClassifier


def test_sgd_classifier_learns_simple_binary_problem():
    X = np.array([
        [-2],
        [-1],
        [0],
        [1],
        [2],
        [3]
    ], dtype=float)

    y = np.array([0, 0, 0, 1, 1, 1])

    model = SGDClassifier(
        learning_rate=0.1,
        max_iter=1000,
        batch_size=2,
        l2_ratio=0.0,
        random_state=42
    )

    model.fit(X, y)
    preds = model.predict(X)

    accuracy = np.mean(preds == y)

    assert accuracy >= 0.8
    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert model.n_features_in_ == 1


def test_sgd_classifier_predict_before_fit_raises_error():
    model = SGDClassifier()

    with pytest.raises(ValueError):
        model.predict(np.array([[1], [2]]))


def test_sgd_classifier_predict_proba_outputs_values_between_zero_and_one():
    X = np.array([[-2], [-1], [1], [2]], dtype=float)
    y = np.array([0, 0, 1, 1])

    model = SGDClassifier(
        learning_rate=0.1,
        max_iter=500,
        batch_size=2,
        l2_ratio=0.0,
        random_state=42
    )

    model.fit(X, y)
    probs = model.predict_proba(X)

    assert np.all(probs >= 0)
    assert np.all(probs <= 1)