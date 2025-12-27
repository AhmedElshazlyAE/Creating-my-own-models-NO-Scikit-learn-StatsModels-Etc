# scaling/_scalers.py
# Here we are defining different scaling techniques.

def z_score_standardization(X, Data):
    """Standardizes features by removing the mean and scaling to unit variance.
    Useful when the data follows a Gaussian/Normal distribution.
    Formula: X_scaled = (x - u) / s
    """
    return (Data - X.mean()) / X.std()


def min_max_scaling(X, Data):
    """Scales features to a fixed range (usually 0 to 1).
    Useful when the distribution is not Gaussian and when preserving
    zero entries in sparse data is important.
    Formula: X_scaled = (X - X.min) / (X.max - X.min)
    """
    return (Data - X.min()) / (X.max() - X.min())


def robust_scaling(X, Data):
    """Scales features using statistics that are robust to outliers.
    Useful when the data contains outliers and you want to reduce their influence.
    Formula: X_scaled = (X - median) / (Q75 - Q25)
    """
    return (Data - X.median()) / (X.quantile(0.75) - X.quantile(0.25))


def max_abs_scaling(X, Data):
    """Scales features by their maximum absolute value.
    Useful for data that is already centered at zero without outliers.
    Formula: X_scaled = X / max(abs(X))
    """
    return Data / X.abs().max()