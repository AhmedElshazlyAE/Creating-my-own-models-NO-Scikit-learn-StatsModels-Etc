# optimizers/_optimizer.py
# Here we are defining different optimization algorithms.


def gradient_descent(W, dW, learning_rate):
    """
    Performs a single step of gradient descent optimization.
    W: Current parameters (weights).
    dW: Gradient of the loss with respect to parameters.
    learning_rate: Learning rate.
    """
    return W - learning_rate * dW

def adaptive_gradient(W, dW, G, learning_rate):
    """
    Performs a single step of adaptive gradient optimization (Adagrad).
    W: Current parameters (weights).
    dW: Gradient of the loss with respect to parameters.
    G: Accumulated squared gradients.
    learning_rate: Learning rate.
    """
    epsilon = 1 * 10**(-8)
    step = learning_rate / (G ** .5 + epsilon)
    return W - step * dW