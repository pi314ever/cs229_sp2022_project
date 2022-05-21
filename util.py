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
FORMAT = "[%(levelname)s:%(filename)s:%(lineno)3s] %(funcName)s(): %(message)s"
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
    tmp = re.sub('â€™', "'",message)
    return re.sub(r'[^a-zA-Z0-9_\']+', ' ', tmp).split()

def load_dataset(min_words = 3, pooled=False, by_books=False):
    """
    Loads dataset from main dataset.

    Arguments:
        min_words (int): Minimum number of words in dataset to be considered
        pooled (bool): Whether or not books are pooled into 3 catagories only
        by_books (bool): Whether or not dataset is pooled by books

    Returns:
        _type_: _description_
    """
    # Loads data and processes
    raw_data = load_csv('../cs229_sp22_dataset/full_processed_dataset.csv')
    if by_books:
        raw_data = raw_data.groupby('isbn').agg({'page_word_count':'sum', 'level':'max','page_num':'max','page_text':'sum'})
        pass
    valid_data = raw_data.loc[raw_data['page_word_count'] > min_words]
    text_data = np.array(valid_data['page_text'])
    level = np.array(valid_data['level'])
    n = len(level)
    if pooled:
        pools = [['A','B','C','D'],['E','F','G','H','I','J'],['K','L','M','N']]
    else:
        # Obtain unique levels
        pools = list(set(level))
        pools.sort()
        pools = [[pools[i]] for i in range(len(pools))]
    # Maps letter to index
    level_map = dict()
    for i, pool in enumerate(pools):
        for element in pool:
            level_map[element] = i
    # Generates levels matrix (list of one hot vectors)
    levels = np.zeros((n, len(pools)))
    for i in range(n):
        levels[i, level_map[level[i]]] = 1.
    # Generate word matrix
    word_map = word_dict(text_data)
    matrix = word_mat(text_data, word_map)
    # Shuffle data
    # np.random.seed(100)
    perm = np.random.shuffle(np.arange(text_data.shape[0]))
    matrix = matrix[perm, :].squeeze()
    levels = levels[perm, :].squeeze()
    return matrix, levels, level_map

def load_dataset_pooled(**kwargs):
    return load_dataset(pooled=True, **kwargs)

# def load_dataset_books(pooled = True):
#     raw_data = load_csv('../cs229_sp22_dataset/full_processed_dataset.csv')
#     valid_data = raw_data.loc[raw_data['page_word_count'] > 10]
#     text_data = np.array(valid_data['page_text'])
#     level = np.array(valid_data['level'])
#     n = len(level)


def train_test_split(c, matrix, levels):
    n = matrix.shape[0]
    train_data = matrix[:int(n * c), :]
    train_levels = levels[:int(n * c), :]
    test_data = matrix[int(n * c) + 1:, :]
    test_levels = levels[int(n * c) + 1:, :]
    return train_data, train_levels, test_data, test_levels

class classification_model:
    def __init__(self, filename = None, **kwargs):
        """
        Call super().__init__() after all parameters necessary for load_params and init_params are created.

        Args:
            filename (str, optional): File to load parameters from. Defaults to None.
        """
        if filename is not None:
            self.load_params(filename, **kwargs)
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
    def accuracy(self, output, labels):
        """
        Defines accuracy of output given labels. Returns accuracies for each individual class and overall accuracy

        Args:
            output (2d array): Array of model outputs
            labels (2d array): Array of corresponding labels

        Returns:
            accuracy "acc" (1d list): 1d array of [acc_class_0, ..., acc_class_n, acc_overall]
        """
        if self.verbose:
            logger.info('Default accuracy module')
        assert(output.shape == labels.shape)
        acc = []
        for i in range(self.num_classes):
            acc.append(sum(np.logical_and(np.argmax(output, axis=1) == i, np.argmax(labels, axis=1) == i)) * 1. / sum(labels[:,i]))
        acc.append(sum(np.argmax(output, axis=1) == np.argmax(labels, axis=1)) * 1. / labels.shape[0])
        return acc
    def is_valid(self, data = None, labels = None):
        """
        Checks data and labels are valid

        Args:
            data (2d array, optional): Data points to be considered. Defaults to None.
            labels (2d array, optional): Labels to be considered. Defaults to None.

        Returns:
            Returns None

        Exceptions:
            Throws an exception if the input parameters are of invalid shape.
        """
        if self.verbose:
            logger.info('Default is_valid module')
        if data is not None:
            nd, dim = data.shape
            assert dim == self.num_features, 'Data features does not match declared number of features'
        if labels is not None:
            nl, o = labels.shape
            assert o == self.num_classes, 'Label classes does not match declared number of classes'
        if data is not None and labels is not None:
            assert nd == nl, 'Number of data points does not match number of label points'
        pass
    def predict_one_hot(self, data):
        """
        Computes prediction based on weights (Array of one-hot vectors)
        """
        if self.verbose:
            logger.info('Default predict_one_hot module')
        output = self.predict(data)
        pred = np.zeros_like(output)
        for i in range(output.shape[0]):
            pred[i, np.argmax(output[i,:])] = 1
        return pred

# Sample (bare minimum) class using this base model:
class sample_model(classification_model):
    #*** MUST IMPLEMENT THESE METHODS ***#
    def __init__(*args, filename=None,**kwargs):
        # Initialize important parameters
        # MUST HAVE members:
        #   self.verbose (bool)
        #   self.num_features (int)
        #   self.num_classes  (int)
        # Load dataset using base model init method
        super().__init__(filename)
    def init_params(self, *args):
        # Initialize model parameters here
        pass
    def load_params(self, filename, *args, **kwargs):
        # Load parameters from file(s) here
        pass
    def fit(self, *args, **kwargs):
        # Fit the model here
        pass
    def predict(self, *args, **kwargs):
        # Predict the model output here
        pass
    def save(self, filename, *args, **kwargs):
        # Save parameters to file(s) here
        pass

    #*** Can modify the exact implementation of these methods to your desire
    def accuracy(self, output, labels):
        # Usually fine to use default provided definitions here
        return super().accuracy(output, labels)
    def is_valid(self, data=None, labels=None):
        # Usually fine to use default provided definitions her
        super().is_valid(data, labels)
    def predict_one_hot(self, data):
        # Usually fine to use default provided definitions her
        return super().predict_one_hot(data)


