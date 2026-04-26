import numpy as np
from simpleml.preprocessing import StandardScaler
from simpleml.linear_models import SGDRegressor, SGDClassifier
from simpleml.pipeline import make_pipeline


def test_pipeline_with_regressor_can_fit_and_predict():
    X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
    y = np.array([2, 4, 6, 8, 10, 12], dtype=float)

    pipe = make_pipeline(
        StandardScaler(),
        SGDRegressor(
            learning_rate=0.01,
            max_iter=1000,
            batch_size=2,
            l2_ratio=0.0,
            random_state=42
        )
    )

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert preds.shape == y.shape


def test_pipeline_named_steps_contains_steps():
    pipe = make_pipeline(
        StandardScaler(),
        SGDRegressor()
    )

    assert "standardscaler" in pipe.named_steps
    assert "sgdregressor" in pipe.named_steps


def test_pipeline_with_classifier_predict_proba():
    X = np.array([[-2], [-1], [1], [2]], dtype=float)
    y = np.array([0, 0, 1, 1])

    pipe = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            learning_rate=0.1,
            max_iter=500,
            batch_size=2,
            l2_ratio=0.0,
            random_state=42
        )
    )

    pipe.fit(X, y)
    probs = pipe.predict_proba(X)

    assert probs.shape == y.shape
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)