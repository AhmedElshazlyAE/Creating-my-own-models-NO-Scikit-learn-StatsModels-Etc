# linear_models/_Linear_Regression.py
# Multiple Linear Regression
# y = b0 + b1x1 + b2x2 + ... + bnxn 
import numpy as np
import math
from .. import optimizers as opt
from ..metrics import mean_squared_error
from ..operations import sum_of_products
from ..scaling import z_score_standardization


class LinearRegression:
    n = 0  # Data Row Size
    b_count = 0  # Number of params 

    def __init__(self, lr=0.001, itterations=10000, optimizer="gradient_descent", patience=5, 
                 early_stopping=True):
        self.lr = lr
        self.itterations = itterations
        self.optimizer = optimizer
        self.patience = patience
        self.early_stopping = early_stopping
        self.fit = LinearRegression.Fit(self)

    def beta_dv(self, X, y, y_pred):
        dB = np.zeros(self.b_count)
        for i in range(self.b_count):
            dB[i] = (2 / self.n) * ((y_pred - y) * X[i]).sum()
        return dB

    class Fit:
        def __init__(self, model):
            self.model = model
            self.b_count = self.model.b_count
            self.lr = self.model.lr
            self.itterations = self.model.itterations
            self.optimizer = self.model.optimizer
            self.patience = self.model.patience
            self.early_stopping = self.model.early_stopping
            self.beta_dv = self.model.beta_dv
            
        def __call__(self, X, y):
            self.X = X
            self.y = y
            self.n = len(y)

            self.scaled_X = np.vstack([np.ones(self.n), z_score_standardization(self.X, self.X)])
            self.b_count = self.scaled_X.shape[0]
            self.B = np.zeros(self.b_count)

            self.model.b_count = self.b_count
            self.model.n = self.n
            
            patience_idx = 0
            best_loss = math.inf  # Highest Possible Loss for early stopping
            best_prams = self.B
            y_pred = sum_of_products(self.B, self.scaled_X, self.n)
            
            for i in range(self.itterations):
                dB = self.beta_dv(self.scaled_X, self.y, y_pred)
                if self.optimizer == "gradient_descent":
                    self.B = opt.gradient_descent(self.B, dB, self.lr)

                y_pred = sum_of_products(self.B, self.scaled_X, self.n)
                loss = mean_squared_error(self.y, y_pred)

                if round(loss, 3) < round(best_loss, 3):
                    best_prams = self.B
                    best_loss = loss
                    patience_idx = 0
                else:
                    patience_idx += 1
                    
                if patience_idx + 1 >= self.patience and self.model.early_stopping:
                    break

            self.B = best_prams
            return self
            
        def predict(self, X):
            X = np.vstack([np.ones(len(X)), z_score_standardization(self.X, X)])
            return sum_of_products(self.B, X, X.shape[1])

        def _coeff(self):
            return self.B
