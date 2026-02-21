# linear_models/_Linear_Regression.py
# Multiple Linear Regression
# y = b0 + b1x1 + b2x2 + ... + bnxn 

# Importing Numpy and math we will only be using these for the library 
import numpy as np
import math
# Importing from prexisting packages in the libary
from .. import optimizers as opt
from ..metrics import mean_squared_error
from ..schedulers import inverse_scaling
from ..regularizers import l2
from ..model_selection import train_test_split

# Creating a class to be able to instantiate a linear regression object for repeatability
class SGDRegressor:
    def __init__(self, lr="invscaling", max_itterations=1000, patience=10, 
                 early_stopping=False, batch_size=32, epochs_per_decay=10, fit_intercept=True, l2_ratio = 0.01,
                 random_state=None): 
        
        self.lr = lr # learning rate
        self.max_itterations = max_itterations # number of itteration (epochs)
        self.patience = patience # early stopping patience
        self.early_stopping = early_stopping
        self.batch_size = batch_size # batches for sgd
        self.epochs_per_decay = epochs_per_decay # number of epochs per learning rate decay
        self.fit_intercept = fit_intercept # if set to false the regressor will not fit an intercept during training
        self.l2_ratio = l2_ratio 
        self.random_state = random_state
        
        # initializing the trainer for the model which will be used to fit the model and train it using the sgd algorithm
        self._trainer = SGDRegressor.Fit(self)  
        self.fitted_ = None  
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        
    # function to calculate the derivative of the intercept with respect to the loss function (MSE)
    # 2 / n * SUM(y_pred - y)
    def intercept_dv(self, batch_size, y, y_pred):
        db = (2 / batch_size) * (y_pred - y).sum()
        return db
    
    # function to calculate the derivative of the weights with respect to the loss function (MSE)
    # 2 / n * SUM((y_pred - y) * X) + l2
    def weights_dv(self, batch_size, X, y, y_pred, W):
        features_count = X.shape[1]
        
        # setting an array with the size of the number of weights equal to 0
        dw = np.zeros(features_count)
        residuals = y_pred - y
        inverse_size = 2 / batch_size
        
        # calculating the l2 regularizer using the l2 function from regularizer package
        l2_term = l2(self.l2_ratio, W)
        
        # inverse_size * residuals @ X is the vectorized form of the sum of (y_pred - y) * X for all samples in the batch and adding the l2 term to it to get the final weights derivative
        dw = inverse_size * residuals @ X + l2_term

        return dw

    class Fit:
        def __init__(self, model):
            # Initializing variables
            self.model = model
            self.w_count = 0
            self.intercept = 0
            self.lr = self.model.lr
            self.max_itterations = self.model.max_itterations
            self.patience = self.model.patience
            self.early_stopping = self.model.early_stopping
            self.weights_dv = self.model.weights_dv
            self.intercept_dv = self.model.intercept_dv
            self.batch_size = self.model.batch_size
            self.epochs_per_decay = self.model.epochs_per_decay
            self.fit_intercept = self.model.fit_intercept
            self.random_state = self.model.random_state
            self.l2_ratio = self.model.l2_ratio
        
        def _predict_internal(self, X):
            X = np.asarray(X, dtype=float)
            return X @ self.W + self.intercept
            
        # this function trains a mini-batch sgd regressor model and return the new weights and intercept
        def sgd_train(self, X, y):
            n = len(X) # number of samples in the dataset
            patience_idx = 0 # current patience itteration for early stopping
            best_loss = math.inf  # Highest Possible Loss for early stopping
            best_weights = self.W
            self.batch_size = self.batch_size if self.batch_size < n else n # if the batch size is greater than the size of the dataset set it to the whole dataset size
            learning_rate = 1.0 if self.lr == "invscaling" else self.lr # set lr to 1 if inverse scaling is enabled
            decayed_lr = learning_rate
            updates = 0
            
            rng = np.random.default_rng(self.random_state) # setting a default random state
            # Epochs Loop
            for epoch in range(self.max_itterations):
                # Mini-Batch Gradient Descent
                
                # permutation table with the length of the dataset (random generated array) to make the batches
                # consistant in size but randomized in order
                perm = rng.permutation(n)
                
                # number of steps or batches per each epoch
                steps_per_epochs = math.ceil(n / self.batch_size)
                
                # Mini-Batch Loop
                for batch in range(0, steps_per_epochs):
                    # array with a batch from the dataset
                    batch_idx = perm[batch * self.batch_size: (batch + 1) * self.batch_size]
                    X_batch = X[batch_idx]
                    y_batch = y[batch_idx]
                    updates += 1

                    current_batch_size = len(X_batch)
                    
                    y_pred = self._predict_internal(X_batch)
                    
                    # getting weights and intercept derivatives
                    dw = self.weights_dv(current_batch_size, X_batch, y_batch, y_pred, self.W)
                    db = self.intercept_dv(current_batch_size, y_batch, y_pred) if self.fit_intercept else 0
                    
                    # updating weights and intercept
                    self.W = opt.gradient_descent(self.W, dw, decayed_lr)
                    self.intercept = opt.gradient_descent(self.intercept, db, decayed_lr) if self.fit_intercept else 0
                    
                    # decaying lr
                    decayed_lr = inverse_scaling(learning_rate, self.epochs_per_decay, steps_per_epochs, updates) if self.lr == "invscaling" else learning_rate
                
                # calculating the loss for early stopping, if enabled, using the validation set if early stopping is enabled otherwise using the training set
                if self.early_stopping:
                    y_pred = self._predict_internal(self.X_val)
                    loss = mean_squared_error(self.y_val, y_pred)
                else:
                    y_pred = self._predict_internal(self.X_train)
                    loss = mean_squared_error(self.y_train, y_pred)
                
                # if current loss is better than best loss reset patience and save the new loss as the best loss
                if loss < best_loss:
                    best_weights = self.W.copy()
                    best_loss = loss
                    patience_idx = 0
                else:
                    patience_idx += 1
                
                # if patience reached maximum patience break out of the loop and stop the training
                if patience_idx + 1 >= self.patience and self.model.early_stopping:
                    break
            
            print("Val Loss:", best_loss)
            if not self.early_stopping:
                best_weights = self.W.copy()

            return np.hstack([self.intercept, best_weights])

        
        # call function for initializing the training
        def __call__(self, X, y):
            self.model.fitted = self
            self.X = np.asarray(X, dtype=float)
            self.y = np.asarray(y, dtype=float)
            
            # split the training data into validation and training 80/20 ratio for early stopping
            if self.early_stopping:
                X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=self.random_state)
                self.X_train = X_train
                self.y_train = y_train
                self.X_val = X_val
                self.y_val = y_val
            else:
                self.X_train = self.X
                self.y_train = self.y
            
            self.w_count = self.X_train.shape[1] # number of features in the dataset
            self.W = np.zeros(self.w_count) # initializing weights to 0
            
            # training the model using the sgd algorithm and getting the final weights and intercept
            betas = self.sgd_train(self.X_train, self.y_train)
            self.W = betas[1:]
            self.intercept = betas[0]
                
            
            self.model.fitted = self
            return self
            
    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")

        X = np.asarray(X, dtype=float)
        
        # X @ self.coef_ is the dot product of X and the weights adding the intercept to get the final prediction
        return X @ self.coef_ + self.intercept_

    def fit(self, X, y):
        trainer = self._trainer(X, y)     
        self.fitted_ = trainer            
        self.coef_ = trainer.W.copy()
        self.intercept_ = trainer.intercept
        self.n_features_in_ = trainer.w_count
        return self