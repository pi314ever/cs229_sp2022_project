import numpy as np
import pandas as pd
import re

#*** util.py
# Summary: Library of utility functions for various functions and classes
#
# Functions:
#   softmax(): Computes the softmax for a 2d array along an axis
#   sigmoid(): Computes the element-wise sigmoid for an nd array.
#   load_csv(): Loads dataset from a csv file.
#   word_dict(): Creates dictionary mapping from words to index given messages
#   split(): Splits messages into words by spaces and newlines
#
# Classes:
#   model(): Base model with basic model parameters and structure
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

def load_csv(filename):
    return pd.read_csv(filename)

def word_dict(text_data):
    mapping = dict()
    idx = 0
    for text in text_data:
        words = split(text)
        for word in words:
            if word.lower() not in mapping:
                mapping[word.lower()] = idx
                idx += 1
    return mapping

def word_mat(text_data, mapping):
    mat = np.zeros((len(text_data), len(mapping)))
    for i, text in enumerate(text_data):
        words = split(text)
        for word in words:
            mat[i, mapping[word.lower()]] += 1
    return mat

def split(message:str):
    return re.split(' |\r|\n', message)

class model:
    def __init__(self, filename = None, verbose = False):
        """
        Call super().__init__() after all parameters necessary for load_params and init_params are created.

        Args:
            filename (str, optional): File to load parameters from. Defaults to None.
            verbose (bool, optional): Toggles verbose printouts. Defaults to False.
        """
        self.verbose = verbose
        if filename is not None:
            self.load_params(filename)
        else:
            self.init_params()
    def init_params(self):
        logger.warning('init_params function not implemented yet.')
    def load_params(self, *args, **kwargs):
        logger.warning('load_params function not implemented yet.')
        logger.info(f'Parameters provided: {args} {kwargs}')
    def fit(self, *args, **kwargs):
        logger.warning('Fit function not implemented yet.')
        logger.info(f'Parameters provided: {args} {kwargs}')
    def predict(self, *args, **kwargs):
        logger.warning('Predict function not implemented yet.')
        logger.info(f'Parameters provided: {args} {kwargs}')
    def save(self, *args, **kwargs):
        logger.warning('Save function not implemented yet.')
        logger.info(f'Parameters provided: {args} {kwargs}')
    def accuracy(self, *args, **kwargs):
        logger.warning('Accuracy function not implemented yet.')
        logger.info(f'Parameters provided: {args} {kwargs}')

