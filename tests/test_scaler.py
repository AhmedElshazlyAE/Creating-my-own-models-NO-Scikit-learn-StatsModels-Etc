import numpy as np
from simpleml.preprocessing import StandardScaler


def test_standard_scaler_fit_transform_mean_zero_std_one():
    X = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40]
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled.mean(axis=0), [0, 0])
    assert np.allclose(X_scaled.std(axis=0), [1, 1])


def test_standard_scaler_handles_constant_column():
    X = np.array([
        [1, 5],
        [2, 5],
        [3, 5]
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert np.allclose(X_scaled[:, 1], [0, 0, 0])
    assert scaler.std_[1] == 1.0