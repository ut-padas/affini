import numpy as np


def one_d_example(x):
    return np.atleast_2d(np.sin(2 * np.pi * x) / np.exp(x))


def grad_one_d_example(x):
    return np.atleast_3d(
        (2 * np.pi * np.cos(2 * np.pi * x) - np.sin(2 * np.pi * x)) / np.exp(x)
    )
