# linear_models/_Linear_Regression.py
# Multiple Linear Regression
# y = b0 + b1x1 + b2x2 + ... + bnxn 
import numpy as np
import math
from .. import optimizers as opt
from ..metrics import mean_squared_error
from ..scaling import z_score_standardization
from ..schedulers import inverse_scaling
from ..regularizers import l2
from ..model_selection import train_test_split

class LinearRegression:
    n = 0  # Data Row Size
    w_count = 0  # Number of params 
    intercept = 0

    def __init__(self, lr="invscaling", itterations=1000, optimizer="sgd", patience=10, 
                 early_stopping=True, batch_size=32, epochs_per_decay=10, fit_intercept=True, l2_ratio = 0.01,
                 random_state=None):
        self.lr = lr
        self.itterations = itterations
        self.optimizer = optimizer
        self.patience = patience
        self.early_stopping = early_stopping
        self.batch_size = batch_size
        self.epochs_per_decay = epochs_per_decay
        self.fit_intercept = fit_intercept
        self.l2_ratio = l2_ratio
        self.random_state = random_state
        self.fit = LinearRegression.Fit(self)
        
        
    def intercept_dv(self, batch_size, y, y_pred):
        db = (2 / batch_size) * (y_pred - y).sum()
        return db

    def weights_dv(self, batch_size, X, y, y_pred, W):
        dw = np.zeros(self.w_count)
        residuals = y_pred - y
        inverse_size = 2 / batch_size
        l2_term = l2(self.l2_ratio, W)  # Example regularization strength
        for i in range(self.w_count):
            dw[i] = inverse_size * (residuals * X[:,i]).sum() + l2_term[i]
        return dw

    class Fit:
        def __init__(self, model):
            self.model = model
            self.w_count = self.model.w_count
            self.intercept = self.model.intercept
            self.n = self.model.n
            self.lr = self.model.lr
            self.itterations = self.model.itterations
            self.optimizer = self.model.optimizer
            self.patience = self.model.patience
            self.early_stopping = self.model.early_stopping
            self.weights_dv = self.model.weights_dv
            self.intercept_dv = self.model.intercept_dv
            self.batch_size = self.model.batch_size
            self.epochs_per_decay = self.model.epochs_per_decay
            self.fit_intercept = self.model.fit_intercept
            self.random_state = self.model.random_state
            self.l2_ratio = self.model.l2_ratio
            
        
        def sgd_train(self, X, y):
            patience_idx = 0
            best_loss = math.inf  # Highest Possible Loss for early stopping
            best_prams = self.W
            self.batch_size = self.batch_size if self.batch_size < len(X) else len(X)
            
            learning_rate = 1.0 if self.lr == "invscaling" else self.lr
            decayed_lr = learning_rate
            updates = 0
            
            rng = np.random.default_rng(self.random_state)
            # Epochs Loop
            for i in range(self.itterations):
                # Mini-Batch Gradient Descent
                perm = rng.permutation(len(X))
                
                steps_per_epochs = math.ceil(len(X)/self.batch_size)
                
                # Mini-Batch Loop
                for start in range(0, math.ceil(len(X)/self.batch_size)):
                    batch_idx = perm[start * self.batch_size: (start + 1) * self.batch_size]
                    X_batch = X[batch_idx]
                    y_batch = y[batch_idx]
                    updates += 1

                    current_batch_size = len(X_batch)
                
                    y_pred = self.predict(X_batch)

                    dw = self.weights_dv(current_batch_size, X_batch, y_batch, y_pred, self.W)
                    db = self.intercept_dv(current_batch_size, y_batch, y_pred)

                    self.W = opt.gradient_descent(self.W, dw, decayed_lr)
                    self.intercept = opt.gradient_descent(self.intercept, db, decayed_lr)
        
                    decayed_lr = inverse_scaling(learning_rate, self.epochs_per_decay, steps_per_epochs, updates) if self.lr == "invscaling" else learning_rate
                
                y_pred = self.predict(self.X_train) if not self.early_stopping else self.predict(self.X_val)
                loss = mean_squared_error(y, y_pred) if not self.early_stopping else mean_squared_error(self.y_val, y_pred)

                if loss < best_loss:
                    best_prams = np.concatenate([[self.intercept], self.W.copy()])
                    best_loss = loss
                    patience_idx = 0
                else:
                    patience_idx += 1
                    
                if patience_idx + 1 >= self.patience and self.model.early_stopping:
                    break
            
            print("Val Loss:", best_loss)
            if not self.early_stopping:
                best_prams = np.concatenate([[self.intercept], self.W.copy()])
            return best_prams
        
        def adagd_train(self, X, y):
            G = np.zeros(self.b_count)
            patience_idx = 0
            best_loss = math.inf  # Highest Possible Loss for early stopping
            best_prams = self.W
            y_pred = self.predict(X)
            for i in range(self.itterations):
                dw = self.weights_dv(X, y, y_pred)
                G += dw ** 2
                self.W = opt.adaptive_gradient(self.W, dw, G, self.lr)

                y_pred = self.predict(X, scaling=False)
                loss = mean_squared_error(y, y_pred)

                if round(loss, 3) < round(best_loss, 3):
                    best_prams = self.W
                    best_loss = loss
                    patience_idx = 0
                else:
                    patience_idx += 1
                    
                if patience_idx + 1 >= self.patience and self.model.early_stopping:
                    break

            return best_prams



        def __call__(self, X, y):
            self.X = np.asarray(X, dtype=float)
            self.y = y
            self.n = len(y)
            
            if self.early_stopping:
                X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=self.random_state)
                self.X_train = X_train
                self.y_train = y_train
                self.X_val = X_val
                self.y_val = y_val
            else:
                self.X_train = X
                self.y_train = y
            
            
            self.w_count = self.X_train.shape[1]
            self.W = np.zeros(self.w_count)

            self.model.w_count = self.w_count
            self.model.n = self.n
            
            
            if self.optimizer == "sgd":
                params = self.sgd_train(self.X_train, self.y_train)
                self.W = params[1:]
                self.intercept = params[0]
            elif self.optimizer == "adaptive_gradient":
                self.W = self.adagd_train(self.X_train, self.y)
            return self
            
        def predict(self, X):          
            X = np.asarray(X, dtype=float)  
            Y = np.zeros(len(X))
            for i in range(self.w_count):
                Y += self.W[i] * X[:, i]
            return Y + self.intercept
        
        def _coeff(self):
            return np.concatenate([[self.intercept], self.W])

        def residuals(self, y, y_pred):
            return y - y_pred