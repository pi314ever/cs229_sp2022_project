import numpy as np

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

class two_layer_neural_network:
    """
    Two layered fully connected neural network

    Architecture:
        Features (Input) -> Sigmoid (Hidden)-> Softmax (Output)
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
        self.verbose = verbose
        if filename is not None:
            if self.verbose:
                logger.info('Loading dataset')
            self.load_dataset(filename)
        else:
            if self.verbose:
                logger.info('Initializing weights and biases')
            # Initialize weights
            np.random.seed(100)
            self.W = [np.random.normal(0,1, (num_hidden, num_features)), np.random.normal(0,1,(num_classes, num_hidden))]
            self.b = [np.zeros((num_hidden, 1)), np.zeros((num_classes, 1))]
    def fit(self, data, labels, epochs, batch_size):
        pass
    def forward_prop(self, data, labels):
        hidden = util.sigmoid((self.W[0] @ data.T + self.b[0]).T)
        output = util.softmax()
        pass
    def predict(self, data):
        """
        Computes prediction based on weights

        """
    def load_dataset(self, filename):
        pass

# Testing function
if __name__ == '__main__':
    nn = two_layer_neural_network(2, 6, 1)
    pass