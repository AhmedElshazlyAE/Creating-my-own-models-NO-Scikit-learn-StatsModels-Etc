# scaling/_scalers.py
# Here we are defining different scaling techniques.

import numpy as np

def z_score_standardization(X, Data):
    """Standardizes features by removing the mean and scaling to unit variance.
    Useful when the data follows a Gaussian/Normal distribution.
    Formula: X_scaled = (x - u) / s
    """
    X = np.asarray(X, dtype=float)   
    Data   = np.asarray(Data, dtype=float)  
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (Data - mean) / std


def min_max_scaling(X, Data):
    """Scales features to a fixed range (usually 0 to 1).
    Useful when the distribution is not Gaussian and when preserving
    zero entries in sparse data is important.
    Formula: X_scaled = (X - X.min) / (X.max - X.min)
    """
    X = np.asarray(X, dtype=float)   
    Data   = np.asarray(Data, dtype=float)  
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    return (Data - min_val) / (max_val - min_val)


def robust_scaling(X, Data):
    """Scales features using statistics that are robust to outliers.
    Useful when the data contains outliers and you want to reduce their influence.
    Formula: X_scaled = (X - median) / (Q75 - Q25)
    """
    X = np.asarray(X, dtype=float)   
    Data   = np.asarray(Data, dtype=float)  
    median = X.median(axis=0)
    q75 = X.quantile(0.75, axis=0)
    q25 = X.quantile(0.25, axis=0)  
    return (Data - median) / (q75 - q25)


def max_abs_scaling(X, Data):
    """Scales features by their maximum absolute value.
    Useful for data that is already centered at zero without outliers.
    Formula: X_scaled = X / max(abs(X))
    """
    X = np.asarray(X, dtype=float)
    Data   = np.asarray(Data, dtype=float)
    abs_max = X.abs().max(axis=0)
    return Data / abs_max