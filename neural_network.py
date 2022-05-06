import numpy as np
from time import time

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
    Two layered fully connected neural network

    Architecture:
        Features (Input) -> Sigmoid (Hidden)-> Softmax (Output)

    All data must be shaped as (num_examples, num_features)
    All labels must be shaped as (num_examples, num_classes)
    """
    def __init__(self, num_features:int, num_hidden:int, num_classes:int, regularized=False, filename = None, verbose = False):
        """
        Initializes neural network

        Args:
            num_features (int): Number of features to consider
            num_hidden (int): Number of hidden layers
            num_classes (int): Number of classes to identify between
            regularized (bool, optional): Whether or not a weight regularizer is implemented. Defaults to False.
            filename (str, optional): File location where the dataset of weights can be loaded. Defaults to None.
            verbose (bool, optional): Toggles verbose printouts.
        """
        if verbose:
            logger.info('Initializing two layer neural network')
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.regularized = regularized
        super().__init__(filename, verbose)
    def init_params(self):
        if self.verbose:
            logger.info('Initializing weights and biases')
        # Initialize weights
        np.random.seed(100)
        self.W = [np.random.normal(0,1, (self.num_hidden, self.num_features)), np.random.normal(0,1,(self.num_classes, self.num_hidden))]
        self.b = [np.zeros((self.num_hidden, 1)), np.zeros((self.num_classes, 1))]
    def fit(self, train_data, train_labels, batch_size, num_epochs = 30, learning_rate = 5, dev_data = None, dev_labels = None):
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
            train_data = train_data[perm, :]
            train_labels = train_labels[perm, :]
            if self.verbose:
                logger.info(f'Epoch {epoch} of {num_epochs}')
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
    def accuracy(self, output, labels):
        return sum(np.argmax(output, axis=1) == np.argmax(output, axis=1)) * 1. / labels.shape[0]
    def gradient_descent_epoch(self, data, labels, learning_rate, batch_size):
        self.is_valid(data, labels)
        n = data.shape[0]
        num_iters = int(np.floor(n / batch_size))
        if num_iters == 0:
            # Batch size larger than input size
            num_iters = 1
            batch_size = n
        for i in range(num_iters):
            self.backward_prop(data[batch_size*i:batch_size*(i+1), :], labels[batch_size*i:batch_size*(i+1),:])
    def is_valid(self, data = None, labels = None):
        """
        Checks data and labels are valid

        Args:
            data (_type_, optional): _description_. Defaults to None.
            labels (_type_, optional): _description_. Defaults to None.
        """
        if data is not None and labels is not None:
            nd, dim = data.shape
            nl, o = labels.shape
            assert nd == nl, 'Number of data points does not match number of label points'
            assert dim == self.num_features, 'Data features does not match declared number of features'
            assert o == self.num_classes, 'Label classes does not match declared number of classes'
    def forward_prop(self, data, labels):
        hidden = util.sigmoid((self.W[0] @ data.T + self.b[0]).T)
        output = util.softmax()
        pass
    def backward_prop(data, labels):
        pass
    def predict(self, data):
        """
        Computes prediction based on weights (Array of one-hot vectors)
        """
        pass
    def load_dataset(self, filename):
        pass

# Testing function
if __name__ == '__main__':
    nn = two_layer_neural_network(5, 3, 10, verbose=True)
    nn.save()
    arr = np.array([[1, 2, 3, 2, 1],[2, 3, 2, 1, 0],[3, 1, 9, 2, 1]])
    pass