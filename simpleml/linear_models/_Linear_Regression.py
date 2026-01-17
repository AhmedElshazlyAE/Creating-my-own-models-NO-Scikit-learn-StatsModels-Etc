# linear_models/_Linear_Regression.py
# Multiple Linear Regression
# y = b0 + b1x1 + b2x2 + ... + bnxn 
import numpy as np
import math
from .. import optimizers as opt
from ..metrics import mean_squared_error
from ..operations import sum_of_products
from ..scaling import z_score_standardization
from ..schedulers import inverse_scaling

class LinearRegression:
    n = 0  # Data Row Size
    b_count = 0  # Number of params 

    def __init__(self, lr="invscaling", itterations=1000, optimizer="sgd", patience=10, 
                 early_stopping=True, batch_size=32, epochs_per_decay=10):
        self.lr = lr
        self.itterations = itterations
        self.optimizer = optimizer
        self.patience = patience
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.epochs_per_decay = epochs_per_decay
        self.fit = LinearRegression.Fit(self)

    def beta_dv(self, X, y, y_pred):
        dB = np.zeros(self.b_count)
        for i in range(self.b_count):
            dB[i] = (2 / self.n) * ((y_pred - y) * X[:,i]).sum()
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
            self.batch_size = self.model.batch_size
            self.epochs_per_decay = self.model.epochs_per_decay
        
        
        def sgd_train(self, X, y):
            patience_idx = 0
            best_loss = math.inf  # Highest Possible Loss for early stopping
            best_prams = self.B
            self.batch_size = self.batch_size if self.batch_size < len(X) else len(X)
            steps_per_epochs = len(X) // self.batch_size
            learning_rate = 1.0 if self.lr == "invscaling" else self.lr
            decayed_lr = learning_rate

            for i in range(self.itterations):
                start = (i * self.batch_size) % len(X)
                end = start + self.batch_size
                y_pred = sum_of_products(self.B, X[start:end])
                
                dB = self.beta_dv(X[start:end], y[start:end], y_pred)
                self.B = opt.gradient_descent(self.B, dB, decayed_lr)
                loss = mean_squared_error(y[start:end], y_pred)

                decayed_lr = inverse_scaling(learning_rate, self.epochs_per_decay, steps_per_epochs, i) if self.lr == "invscaling" else learning_rate

                if round(loss, 4) < round(best_loss, 4):
                    best_prams = self.B
                    best_loss = loss
                    patience_idx = 0
                else:
                    patience_idx += 1
                    
                if patience_idx + 1 >= self.patience and self.model.early_stopping:
                    break


            print(i)

            return best_prams
        
        def adagd_train(self, X, y):
            G = np.zeros(self.b_count)
            patience_idx = 0
            best_loss = math.inf  # Highest Possible Loss for early stopping
            best_prams = self.B
            y_pred = sum_of_products(self.B, X)
            for i in range(self.itterations):
                dB = self.beta_dv(X, y, y_pred)
                G += dB ** 2
                self.B = opt.adaptive_gradient(self.B, dB, G, self.lr)

                y_pred = sum_of_products(self.B, X)
                loss = mean_squared_error(y, y_pred)

                if round(loss, 3) < round(best_loss, 3):
                    best_prams = self.B
                    best_loss = loss
                    patience_idx = 0
                else:
                    patience_idx += 1
                    
                if patience_idx + 1 >= self.patience and self.model.early_stopping:
                    break

            return best_prams



        def __call__(self, X, y):
            self.X = X
            self.y = y
            self.n = len(y)

            self.scaled_X = np.hstack([np.ones((len(X), 1)),  z_score_standardization(self.X, X)])
            self.b_count = self.scaled_X.shape[1]
            self.B = np.zeros(self.b_count)

            self.model.b_count = self.b_count
            self.model.n = self.n
            
            if self.optimizer == "sgd":
                self.B = self.sgd_train(self.scaled_X, self.y)
            elif self.optimizer == "adaptive_gradient":
                self.B = self.adagd_train(self.scaled_X, self.y)

            return self
            
        def predict(self, X):
            X_scaled = z_score_standardization(self.X, X)
            X = np.hstack([np.ones((len(X), 1)),  X_scaled])
            try:
                return sum_of_products(self.B, X)
            except:
                return sum_of_products(self.B, X, 1)
            

        def _coeff(self):
            return self.B

        def residuals(self, y, y_pred):
            return y - y_pred