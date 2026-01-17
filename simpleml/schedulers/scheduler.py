# simpleml/schedulers/scheduler.py
# Here we are defining different scheduling algorithms.

def  inverse_scaling(initial_lr, epochs, steps_per_epochs, updates):
    """
    Performs inverse scaling learning rate decay.
    initial_lr: Initial learning rate.
    epochs: Number of epochs after which learning rate is decayed.
    steps_per_epochs: Number of steps in each epoch.
    updates: Current update number.
    """
    tau = steps_per_epochs * epochs
    return initial_lr / (1 + updates / tau)