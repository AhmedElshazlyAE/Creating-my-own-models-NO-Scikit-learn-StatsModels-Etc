# metrics/_regression.py
# Here we are defining regression metrics.

def r2_score(y, y_pred):
    """
    R2 score provides an indication of goodness of fit for regression models.
    It explains the variance and represents the proportion of the variance in the 
    dependent variable that is predictable from the independent variable(s).
    """
    sum_of_squared_errors = ((y - y_pred) ** 2).sum()
    total_sum_of_squares = ((y - y.mean()) ** 2).sum()
    r2 = 1 - sum_of_squared_errors / total_sum_of_squares
    return r2


def mean_squared_error(y, y_pred):
    """
    Mean squared error is used to measure the average of the squares of the errors.
    It is robust to outliers and easy to interpret because it uses the same unit 
    as the target variable.
    """
    mse = ((y - y_pred) ** 2).mean()
    return mse


def mean_absolute_error(y, y_pred):
    """
    Mean absolute error measures the average magnitude of the errors in a set of predictions,
    without considering their direction. It is easy to understand and interpret.
    """
    mae = (abs(y - y_pred)).mean()
    return mae


def root_mean_squared_error(y, y_pred):
    """
    Root mean squared error is the square root of the average of squared differences 
    between prediction and actual observation. It gives a relatively high weight to large errors.
    """
    rmse = (((y - y_pred) ** 2).mean()) ** 0.5
    return rmse


def mean_absolute_percentage_error(y, y_pred):
    """
    Mean absolute percentage error measures the average magnitude of errors in percentage terms.
    It is scale-independent and easy to interpret.
    """
    mape = (abs((y - y_pred) / y)).mean() * 100
    return mape


def regression_report(y, y_pred):
    return {
        "R2 Score": round(r2_score(y, y_pred), 4),
        "Mean Squared Error": round(mean_squared_error(y, y_pred), 4),
        "Mean Absolute Error": round(mean_absolute_error(y, y_pred), 4),
        "Root Mean Squared Error": round(root_mean_squared_error(y, y_pred), 4),
        "Mean Absolute Percentage Error": round(mean_absolute_percentage_error(y, y_pred), 4)
    }
