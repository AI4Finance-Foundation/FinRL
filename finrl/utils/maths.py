import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Returns the returns softmax. For numerical stability we subtract the max
    which will cancel out in the equation.

    See:
        * https://en.wikipedia.org/wiki/Softmax_function
        * https://github.com/tensorflow/tensorflow/blob/85c8b2a817f95a3e979ecd1ed95bff1dc1335cff/tensorflow/python/keras/activations.py#L78
        * https://stackoverflow.com/a/38250088

    Parameters
    ----------
    x : pd.Series or np.ndarray

    Returns
    -------
    softmax : array-like
    """

    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum(axis=0)
