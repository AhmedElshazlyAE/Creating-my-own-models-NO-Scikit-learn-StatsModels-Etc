# operations/_operation.py
# Here we are defining different operations for predictions.
import numpy as np

def sum_of_products(B, X, n): # Sum Of Products
    Y = np.zeros(n)
    for i in range(len(B)):
        Y += B[i] * X[i]
    return Y

def product_of_sums(B, X, n): # Product Of Sums
    Y = np.ones(n)
    for i in range(len(B)):
        Y *= (B[i] + X[i])
    return Y