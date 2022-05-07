from enum import unique
import numpy as np
from time import time
import re
import matplotlib.pyplot as plt

#*** neural_network.py
# Summary: Contains a class for generating a 2-layer fully connected neural network
#***

import util

import logging, sys # For debugging purposes
FORMAT = "[%(levelname)s:%(filename)s:%(lineno)3s - %(funcName)20s()] %(message)s"
logging.basicConfig(format=FORMAT, stream=sys.stderr)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#** Logger usage:
# logger.debug(): For all parameters useful in debugging (i.e. matrix shapes, important scalars, etc.)
# logger.info(): For all information on what the solver is doing
# logger.warning(): For all information that might cause known instability (i.e. underflow/overflow, etc.). Can also be used for places with implementations to-be-defined
# logger.error(): For notifying failed attempts at calculation (i.e. any exception, bad data, etc.)
#***

class two_layer_neural_network(util.model):
    """
    Two layered fully connected neural network with Batch Gradient Descent optimizer

    Architecture:
        Features (Input) -> Sigmoid (Hidden)-> Softmax (Output)

    All data must be shaped as (num_examples, num_features)
    All labels must be shaped as (num_examples, num_classes)
    """
    def __init__(self, num_features:int, num_hidden:int, num_classes:int, reg=0, filenames = None, verbose = False):
        """
        Initializes neural network

        Args:
            num_features (int): Number of features to consider
            num_hidden (int): Number of hidden layers
            num_classes (int): Number of classes to identify between
            regularized (float, optional): Regularization constant for the weights. Defaults to 0.
            filenames (list of str, optional): File location where the dataset of weights can be loaded. Order: [W1, W2, b1, b2]. Defaults to None (no pre-loaded parameters).
            verbose (bool, optional): Toggles verbose printouts.
        """
        if verbose:
            logger.info('Initializing two layer neural network')
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.reg = reg
        # Load parameters
        super().__init__(filenames, verbose)
    def init_params(self):
        if self.verbose:
            logger.info('Initializing weights and biases')
        # Initialize weights
        # np.random.seed(100) # For reproducibility
        self.W = [np.random.normal(0,1, (self.num_hidden, self.num_features)), np.random.normal(0,1,(self.num_classes, self.num_hidden))]
        self.b = [np.zeros((self.num_hidden, 1)), np.zeros((self.num_classes, 1))]
    def load_params(self, filenames, **kwargs):
        """
        Load parameters with np.loadtxt()

        Args:
            filenames (list of str): File location where the dataset of weights can be loaded. Order: [W1, W2, b1, b2].
            **kwargs: Keyword arguments to be passed to np.loadtxt()

        Raises:
            e: Assertion errors for mismatched shape
        """
        if self.verbose:
            logger.info(f'Loading dataset from {filenames}')
        assert(len(filenames) == 4)
        self.W = [np.array([]), np.array([])]
        self.b = [np.array([]), np.array([])]
        self.W[0] = np.loadtxt(filenames[0], ndmin=2, **kwargs)
        self.W[1] = np.loadtxt(filenames[1], ndmin=2, **kwargs)
        self.b[0] = np.loadtxt(filenames[2], ndmin=2, **kwargs)
        self.b[1] = np.loadtxt(filenames[3], ndmin=2, **kwargs)
        # Confirm parameters are of the right shape
        try:
            assert(self.W[0].shape == (self.num_hidden, self.num_features))
            assert(self.W[1].shape == (self.num_classes, self.num_hidden))
            assert(self.b[0].shape == (self.num_hidden, 1))
            assert(self.b[1].shape == (self.num_classes, 1))
        except Exception as e:
            logger.error('Failed to load files, mismatched shape')
            raise e
    def save(self, filenames, **kwargs):
        """
        Saves parameters to filenames using np.savetxt()

        Args:
            filenames (list of str): File location where the dataset of weights can be saved. Order: [W1, W2, b1, b2].
            **kwargs: Keyword arguments to be passed to np.savetxt()
        """
        if self.verbose:
            logger.info(f'Saving parameters to {filenames}')
        assert(len(filenames) == 4)
        np.savetxt(filenames[0], self.W[0], **kwargs)
        np.savetxt(filenames[1], self.W[1], **kwargs)
        np.savetxt(filenames[2], self.b[0], **kwargs)
        np.savetxt(filenames[3], self.b[1], **kwargs)
    def fit(self, train_data, train_labels, batch_size, num_epochs = 30, learning_rate = 5., dev_data = None, dev_labels = None):
        """
        Fits neural network based on training data and training labels using batch gradient descent (Can convert to GD if batch size = number of examples)

        Args:
            train_data (2d array)
            train_labels (2d array)
            batch_size (int)
            num_epochs (int, optional): Number of epochs for training. Defaults to 30.
            learning_rate (float, optional): Batch GD learning rate. Defaults to 5.
            dev_data (2d array, optional): Development data, for use in comparing incremental increases. Defaults to None.
            dev_labels (2d array, optional): Developmental labels, for use in comparing incremental increases. Defaults to None.

        Returns:
            cost_train: Cost history for training data
            accuracy_train: Accuracy history for training data
            cost_dev: Cost history for dev data (if provided)
            accuracy_dev: Accuracy history of dev data (if provided)
        """
        logger.info(f'Fitting neural network with {self.num_features} features to {self.num_classes} classes with {self.reg} regularization and {self.num_hidden} hidden nodes')
        # Check for proper dimensions
        self.is_valid(train_data, train_labels)
        has_dev = dev_data is not None and dev_labels is not None
        if has_dev:
            self.is_valid(dev_data, dev_labels)
            cost_dev = []
            accuracy_dev = []
        cost_train = []
        accuracy_train = []
        begin = time()
        for epoch in range(num_epochs):
            # Shuffle training data
            perm = np.random.shuffle(np.arange(train_data.shape[0]))
            train_data = train_data[perm, :].squeeze()
            train_labels = train_labels[perm, :].squeeze()
            if self.verbose:
                logger.info(f'Epoch {epoch + 1} of {num_epochs}')
            # Perform gradient descent
            self.gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size)
            # Gather current epoch information
            _, output, cost = self.forward_prop(train_data, train_labels)
            cost_train.append(cost)
            accuracy_train.append(self.accuracy(output, train_labels))
            # Gather dev dataset epoch information
            if has_dev:
                _, output, cost = self.forward_prop(dev_data, dev_labels)
                cost_dev.append(cost)
                accuracy_dev.append(self.accuracy(output, dev_labels))
        end = time()
        if self.verbose:
            logger.info(f'Training took {(end - begin)/60} minutes')
        if has_dev:
            return cost_train, accuracy_train, cost_dev, accuracy_dev
        else:
            return cost_train, accuracy_train
    def accuracy(self, output, labels):
        return sum(np.argmax(output, axis=1) == np.argmax(labels, axis=1)) * 1. / labels.shape[0]
    def gradient_descent_epoch(self, data, labels, learning_rate, batch_size):
        self.is_valid(data, labels)
        n = data.shape[0]
        num_iters = int(np.floor(n / batch_size))
        if num_iters == 0:
            # Batch size larger than input size
            num_iters = 1
            batch_size = n
        for i in range(num_iters):
            self.backward_prop(data[batch_size*i:batch_size*(i+1), :], labels[batch_size*i:batch_size*(i+1),:], learning_rate)
    def is_valid(self, data = None, labels = None):
        """
        Checks data and labels are valid

        Args:
            data (_type_, optional): _description_. Defaults to None.
            labels (_type_, optional): _description_. Defaults to None.
        """
        if data is not None:
            nd, dim = data.shape
            assert dim == self.num_features, 'Data features does not match declared number of features'
        if labels is not None:
            nl, o = labels.shape
            assert o == self.num_classes, 'Label classes does not match declared number of classes'
        if data is not None and labels is not None:
            assert nd == nl, 'Number of data points does not match number of label points'
    def forward_prop(self, data, labels=None):
        """
        Calculates forward propagation layers and loss given data and labels

        Args:
            data (2d array)
            labels (2d array, optional)

        Returns:
            hidden (2d array): Hidden layer activations for each case
            output (2d array): Output of the neural network (after softmax)
            loss (float): Average loss for the predicted output (if labels are given, else returns 0)
        """
        self.is_valid(data=data)
        hidden = util.sigmoid((self.W[0] @ data.T + self.b[0]).T)
        output = util.softmax((self.W[1] @ hidden.T + self.b[1]).T)
        if labels is not None:
            loss = self.loss(labels, output)
        else:
            loss = 0
        return hidden, output, loss
    def loss(self, labels, output):
        """
        Calculates loss given labels and model output

        Args:
            labels (_type_): _description_
            output (_type_): _description_

        Returns:
            _type_: _description_
        """
        return -np.sum(labels * np.log(output + 1e-20)) / labels.shape[0]
    def backward_prop(self, data, labels, learning_rate):
        """
        Performs backward propagation given data and labels

        Args:
            data (2d array)
            labels (2d array)
        """
        # Forward prop values
        hidden, output, _ = self.forward_prop(data, labels)
        n = data.shape[0]
        dCEdz2 = labels - output
        dCEdz1 = (dCEdz2 @ self.W[1]) * hidden * (1 - hidden)
        self.W[0] -= learning_rate * (-dCEdz1.T @ data / n + self.reg * self.W[0])
        self.W[1] -= learning_rate * (-dCEdz2.T @ hidden / n + self.reg * self.W[1])
        self.b[0] -= -(learning_rate * (np.average(dCEdz1, axis=0))).reshape(self.b[0].shape)
        self.b[1] -= -(learning_rate * (np.average(dCEdz2, axis=0))).reshape(self.b[1].shape)
    def predict(self, data):
        """
        Computes prediction based on weights (Array of one-hot vectors)
        """
        output = self.forward_prop(data)[1]
        pred = np.zeros_like(output)
        for i in range(output.shape[0]):
            pred[i, np.argmax(output[i,:])] = 1
        return pred

# Testing function
if __name__ == '__main__':
    folder = './neural_network_parameters/'
    filenames = [folder + 'W1.txt.gz', folder + 'W2.txt.gz',folder + 'b1.txt.gz',folder + 'b2.txt.gz']
    raw_data = util.load_csv('../cs229_sp22_dataset/full_processed_dataset.csv')
    valid_data = raw_data.loc[raw_data['page_word_count'] > 10]
    # print(valid_data)
    text_data = np.array(valid_data['page_text'])
    level = np.array(valid_data['level'])
    n = len(level)
    unique_levels = list(set(level))
    unique_levels.sort()
    level_map = dict()
    for i, letter in enumerate(unique_levels):
        level_map[letter] = i
    levels = np.zeros((n, len(level_map)))
    for i  in range(n):
        levels[i, level_map[level[i]]] = 1.
    word_dict = util.word_dict(text_data)
    matrix = util.word_mat(text_data, word_dict)
    # print(matrix)
    # Shuffle data
    # Shuffle training data
    # np.random.seed(100)
    perm = np.random.shuffle(np.arange(text_data.shape[0]))
    matrix = matrix[perm, :].squeeze()
    levels = levels[perm, :].squeeze()
    # Train nn
    nn = two_layer_neural_network(len(word_dict), 300, len(unique_levels),reg=0.02, verbose=True)
    c = 0.75
    train_data = matrix[:int(n * c), :]
    train_levels = levels[:int(n * c), :]
    test_data = matrix[int(n * c) + 1:, :]
    test_levels = levels[int(n * c) + 1:, :]
    # nn.load_params(filenames)
    epochs = 100
    cost_train, accuracy_train, cost_dev, accuracy_dev = nn.fit(train_data, train_levels, batch_size=n, num_epochs=epochs, dev_data=test_data, dev_labels=test_levels,learning_rate=0.4)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    nn.save(filenames)
    t = np.arange(epochs)
    if True:
        ax1.plot(t, cost_train,'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        ax1.legend()

        ax2.plot(t, accuracy_train,'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./test.pdf')

