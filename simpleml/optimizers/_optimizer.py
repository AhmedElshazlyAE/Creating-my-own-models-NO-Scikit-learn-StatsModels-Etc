# optimizers/_optimizer.py
# Here we are defining different optimization algorithms.

def lr_decay(initial_lr, epochs, steps_per_epochs, iteration):
    """
    Applies learning rate decay.
    initial_lr: Initial learning rate.
    decay_rate: Rate at which to decay the learning rate.
    iteration: Current iteration number.
    """
    tau = steps_per_epochs * epochs
    return initial_lr / (1 + iteration / tau)

def gradient_descent(B, dB, lr):
    """
    Performs a single step of gradient descent optimization.
    B: Current parameters (weights).
    dB: Gradient of the loss with respect to parameters.
    lr: Learning rate.
    """
    return B - lr * dB

def adaptive_gradient(B, dB, G, lr):
    """
    Performs a single step of adaptive gradient optimization (Adagrad).
    B: Current parameters (weights).
    dB: Gradient of the loss with respect to parameters.
    G: Accumulated squared gradients.
    lr: Learning rate.
    """
    epsilon = 1 * 10**(-8)
    step = lr / (G ** .5 + epsilon)
    return B - step * dB