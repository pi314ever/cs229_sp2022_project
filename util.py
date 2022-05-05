import numpy as np

#*** util.py
# Summary: Library of utility functions for various functions
#
# Functions:
#   softmax(): Computes the softmax for a 2d array along an axis
#   sigmoid(): Computes the element-wise sigmoid for an nd array.
#***

import logging, sys # For debugging purposes
FORMAT = "[%(levelname)s:%(filename)s:%(lineno)3s - %(funcName)20s()] %(message)s"
logging.basicConfig(format=FORMAT, stream=sys.stderr)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def softmax(x, ax=1, debug=False):
    """
    Compute softmax function for a batch of input values with overflow protection.

    Args:
        x:  A 2d numpy float array of shape (n x m)
        ax: Axis which indexes the batches (i.e. ax = 1 means the softmax is row-wise)

    Returns:
        A 2d numpy float array containing the softmax results (n x m)
    """
    # 2d array
    assert(len(x.shape) == 2)
    if ax:
        x = x.T
    if debug:
        logger.debug(f'Shape of x {x.shape}')
    x_max = np.max(x, axis=0)
    if debug:
        logger.debug(f'Shape of xmax {x_max.shape}')
    den = np.sum(np.exp(x - x_max), axis=0)
    result = np.exp(x - x_max) / den
    assert(result.shape == x.shape)
    if ax:
        result = result.T
    return result

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    return np.reciprocal(1 + np.exp(-x))