# simpleml/model_selection/selection.py
# Here we are defining functions for model selection, such as train-test splitting.

import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    X : array-like
        Feature data.
    y : array-like
        Target values.
    test_size : float or int, default=0.2
        If float, represents the proportion of samples used for testing.
        If int, represents the exact number of test samples.
    random_state : int, optional
        Seed for reproducible shuffling.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = np.array(X)
    y = np.array(y)

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")

    n_samples = len(X)

    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("test_size as a float must be between 0 and 1.")
        n_test = int(n_samples * test_size)
    elif isinstance(test_size, int):
        if not 0 < test_size < n_samples:
            raise ValueError("test_size as an int must be between 1 and n_samples - 1.")
        n_test = test_size
    else:
        raise TypeError("test_size must be a float or an int.")

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test