# simpleml/model_selection/selection.py
# Here we are defining functions for model selection, such as train-test splitting.

import numpy as np
def train_test_split(X, y, test_size=0.2, random_seed=None):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    - X: Features dataset (numpy array or pandas DataFrame)
    - y: Target variable (numpy array or pandas Series)
    - test_size: Proportion of the dataset to include in the test split (default is 0.2)
    - random_seed: Seed for random number generator for reproducibility (default is None)

    Returns:
    - X_train, X_test, y_train, y_test: Split datasets
    """
    X = np.array(X)
    y = np.array(y)
    if random_seed is not None:
        np.random.seed(random_seed)

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_size = int(n_samples * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test