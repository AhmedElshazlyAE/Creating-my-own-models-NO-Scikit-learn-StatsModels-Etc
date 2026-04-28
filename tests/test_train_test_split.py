import numpy as np
from simpleml.model_selection import train_test_split


def test_train_test_split_shapes():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    assert X_train.shape == (8, 2)
    assert X_test.shape == (2, 2)
    assert y_train.shape == (8,)
    assert y_test.shape == (2,)


def test_train_test_split_reproducible_with_random_state():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    split1 = train_test_split(X, y, test_size=0.3, random_state=42)
    split2 = train_test_split(X, y, test_size=0.3, random_state=42)

    for a, b in zip(split1, split2):
        assert np.array_equal(a, b)