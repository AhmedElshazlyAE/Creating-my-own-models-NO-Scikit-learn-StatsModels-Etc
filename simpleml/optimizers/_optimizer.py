# optimizers/_optimizer.py
# Here we are defining different optimization algorithms.


def gradient_descent(W, dW, lr):
    """
    Performs a single step of gradient descent optimization.
    W: Current parameters (weights).
    dW: Gradient of the loss with respect to parameters.
    lr: Learning rate.
    """
    return W - lr * dW

def adaptive_gradient(W, dW, G, lr):
    """
    Performs a single step of adaptive gradient optimization (Adagrad).
    B: Current parameters (weights).
    dW: Gradient of the loss with respect to parameters.
    G: Accumulated squared gradients.
    lr: Learning rate.
    """
    epsilon = 1 * 10**(-8)
    step = lr / (G ** .5 + epsilon)
    return W - step * dW