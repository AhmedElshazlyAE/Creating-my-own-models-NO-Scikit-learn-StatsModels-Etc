import numpy as np
from simpleml.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    regression_report,
)


def test_regression_metrics_perfect_prediction():
    y = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4])

    assert mean_squared_error(y, y_pred) == 0
    assert mean_absolute_error(y, y_pred) == 0
    assert root_mean_squared_error(y, y_pred) == 0
    assert r2_score(y, y_pred) == 1


def test_regression_report_returns_dictionary():
    y = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 5])

    report = regression_report(y, y_pred)

    assert isinstance(report, dict)
    assert "R2 Score" in report
    assert "Mean Squared Error" in report
    assert "Mean Absolute Error" in report