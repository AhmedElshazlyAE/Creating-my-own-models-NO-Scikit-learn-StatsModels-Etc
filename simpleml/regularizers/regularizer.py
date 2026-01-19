def l2(ratio, W):
    """
    Computes the L2 regularization term and its gradient.
    ratio: Regularization strength (lambda).
    W: Model weights.
    """
    # l2 = (ratio / 2) * sum(W^2)
    # dJ/dW = ratio * W
    return ratio * W