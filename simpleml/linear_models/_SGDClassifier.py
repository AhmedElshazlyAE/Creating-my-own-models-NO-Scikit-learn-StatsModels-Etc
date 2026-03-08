# linear_models/_SGDClassifier.py
# Stochastic Gradient Descent Classifier Implementation from scratch using Numpy only
# with support for mini-batch training, learning rate scheduling, and early stopping
# y = sigmoid(X @ W + b) where W are the weights and b is the intercept

# Importing Numpy and math we will only be using these for the library 
import numpy as np
import math
# Importing from prexisting packages in the libary
from .. import optimizers as opt
from ..schedulers import inverse_scaling
from ..regularizers import l2
from ..model_selection import train_test_split

# Creating a class to be able to instantiate a sgd classifier object for repeatability
class SGDClassifier:
    def __init__(self, lr="invscaling", max_itter=1000, patience=10, 
                 early_stopping=False, batch_size=32, epochs_per_decay=10, fit_intercept=True, l2_ratio = 0.01,
                 random_state=None): 
        
        self.lr = lr # learning rate
        self.max_itter = max_itter # number of itteration (epochs)
        self.patience = patience # early stopping patience
        self.early_stopping = early_stopping
        self.batch_size = batch_size # batches for sgd
        self.epochs_per_decay = epochs_per_decay # number of epochs per learning rate decay
        self.fit_intercept = fit_intercept # if set to false the regressor will not fit an intercept during training
        self.l2_ratio = l2_ratio 
        self.random_state = random_state
        
        # initializing the trainer for the model which will be used to fit the model and train it using the sgd algorithm
        self._trainer = SGDClassifier.Fit(self)  
        self.fitted_ = None  
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
    
    """ 
    function to calculate the sigmoid of a value in a stable way to avoid overflow issues with large values of z
    - for z >= 0: sigmoid(z) = 1 / (1 + e ^ (-z))
    - for z < 0: sigmoid(z) = exp(z) / (1 + e ^ z)
    """
    def stable_sigmoid(self, z):
        return np.where(z >= 0, 
                    1 / (1 + np.exp(-z)), 
                    np.exp(z) / (1 + np.exp(z)))
        
    # function to calculate the probability of the positive class for a given input X using the fitted model
    def predict_proba(self, X):
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        logits = self.decision_function(X)
        return self.stable_sigmoid(logits)
    
    # function to predict the class labels for a given input X based on the predicted probabilities
    # using predict_proba function and a threshold to determine the class labels default threshold is 0.5
    # example: if proba >= 0.5 predict class 1 otherwise predict class 0

    def predict(self, X, threshold=0.5):
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet. Call fit(X, y) first.")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    # logit function to calculate the decision function for a given input X using the fitted model
    def decision_function(self, X):
        return X @ self.coef_ + self.intercept_

    class Fit:
        def __init__(self, model):
            # Initializing variables
            self.model = model
            self.w_count = 0
            self.intercept = 0
            self.lr = self.model.lr
            self.max_itter = self.model.max_itter
            self.patience = self.model.patience
            self.early_stopping = self.model.early_stopping
            self.batch_size = self.model.batch_size
            self.epochs_per_decay = self.model.epochs_per_decay
            self.fit_intercept = self.model.fit_intercept
            self.random_state = self.model.random_state
            self.l2_ratio = self.model.l2_ratio
            self.stable_sigmoid = self.model.stable_sigmoid
        
        # same as predict_proba but without checking if the model is fitted and without X type checking and
        # reshaping to speed up training since this will be called multiple times
        def _predict_proba_internal(self, X):
            logits = self._decision_function_internal(X)
            return self.stable_sigmoid(logits)
        
        """
        function to calculate the binary cross entropy loss in a stable way to avoid overflow
        issues with large values
        bce = (1 / sample_size) * SUM(log(1 + e ^ (-probabilities)) - y * probabilities)
        """
        def stable_binary_cross_entropy(self, y, proba):
            return np.mean(np.logaddexp(0, proba) - y * proba)
        
        # same as decision_function but without checking if the model is fitted and type checking etc, just
        # like _predict_proba_internal this is to speed up training since this will be called multiple times
        def _decision_function_internal(self, X):
            return X @ self.W + self.intercept
        
        """
        function to calculate the derivative of the intercept with respect to the loss function (MSE)
        db0/dE = 1 / sample_size * SUM(probabilities - y)
        """
        def intercept_dv(self, y, proba):
            return np.mean(proba - y)
        
        """
        function to calculate the derivative of the weights with respect to the loss function (BCE) with
        l2 regularization term added by defult
        dw/dE =1 / sample_size * X^T @ (probabilities - y) + l2
        """
        def weights_dv(self, X, y, proba, W):
            residuals = proba - y
            l2_term = l2(self.l2_ratio, W)
            dw = (X.T @ residuals) / len(y) + l2_term
            return dw
        
            
        # this function trains a mini-batch sgd classifier model and return the new weights and intercept
        def sgd_train(self, X, y):
            n = len(y) # number of samples in the dataset
            patience_idx = 0 # current patience itteration for early stopping
            best_loss = math.inf  # Highest Possible Loss for early stopping
            best_weights = self.W
            self.batch_size = self.batch_size if self.batch_size < n else n # if the batch size is greater than the size of the dataset set it to the whole dataset size
            learning_rate = 1.0 if self.lr == "invscaling" else self.lr # set lr to 1 if inverse scaling is enabled
            decayed_lr = learning_rate
            updates = 0
            
            rng = np.random.default_rng(self.random_state) # setting a default random state
            # Epochs Loop
            for epoch in range(self.max_itter):
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
                    
                    # calculating the predicted probabilities for the batch using the current weights and intercept
                    proba = self._predict_proba_internal(X_batch)
                    
                    # getting weights and intercept derivatives
                    dw = self.weights_dv(X_batch, y_batch, proba, self.W)
                    db = self.intercept_dv(y_batch, proba) if self.fit_intercept else 0
                    
                    # updating weights and intercept
                    self.W = opt.gradient_descent(self.W, dw, decayed_lr)
                    self.intercept = opt.gradient_descent(self.intercept, db, decayed_lr) if self.fit_intercept else 0
                    
                    # decaying lr
                    decayed_lr = inverse_scaling(learning_rate, self.epochs_per_decay, steps_per_epochs, updates) if self.lr == "invscaling" else learning_rate
                
                # calculating the loss for early stopping, if enabled, using the validation set if early stopping is enabled otherwise using the training set
                if self.early_stopping:
                    proba = self._predict_proba_internal(self.X_val)
                    loss = self.stable_binary_cross_entropy(self.y_val, proba).mean()
                else:
                    proba = self._predict_proba_internal(self.X_train)
                    loss = self.stable_binary_cross_entropy(self.y_train, proba).mean()
                
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
            
            if not self.early_stopping:
                best_weights = self.W.copy()

            return np.hstack([self.intercept, best_weights])

        
        # call function for initializing the training
        def __call__(self, X, y):
            self.model.fitted = self
            self.X = np.asarray(X, dtype=float)
            self.y = np.asarray(y, dtype=float)

            if self.X.ndim == 1:
                self.X = self.X.reshape(-1, 1)

            if self.y.ndim != 1:
                self.y = self.y.ravel()

            if len(self.X) != len(self.y):
                raise ValueError("X and y must contain the same number of samples.")
            
            if self.early_stopping:
                X_train, X_val, y_train, y_val = train_test_split(
                    self.X, self.y, test_size=0.2, random_state=self.random_state
                )
                self.X_train = X_train
                self.y_train = y_train
                self.X_val = X_val
                self.y_val = y_val
            else:
                self.X_train = self.X
                self.y_train = self.y
            
            self.w_count = self.X_train.shape[1]
            self.W = np.zeros(self.w_count)

            betas = self.sgd_train(self.X_train, self.y_train)
            self.W = betas[1:]
            self.intercept = betas[0]

            self.model.fitted = self
            return self
    
    def fit(self, X, y):
        trainer = self._trainer(X, y)     
        self.fitted_ = trainer            
        self.coef_ = trainer.W.copy()
        self.intercept_ = trainer.intercept
        self.n_features_in_ = trainer.w_count
        return self

