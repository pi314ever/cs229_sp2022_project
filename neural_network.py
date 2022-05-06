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
        if dev_data is not None and dev_labels is not None:
            self.is_valid(dev_data, dev_labels)
            cost_dev = []
            accuracy_dev = []
        cost_train = []
        accuracy_train = []
        for epoch in range(num_epochs):
            if self.verbose:
                logger.info(f'Epoch {epoch} of {num_epochs}')
            self.gradient_descent_epoch(train_data, train_labels, learning_rate)
            _, output, cost = self.forward_prop(train_data, train_labels)

            cost_train.append(cost)
            accuracy_train.append()
    def output_predict(self, output):
        return
    def gradient_descent_epoch(data, labels, learning_rate, batch_size):
        pass
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
    def predict(self, data):
        """
        Computes prediction based on weights
        """
        pass
    def load_dataset(self, filename):
        pass

# Testing function
if __name__ == '__main__':
    nn = two_layer_neural_network(5, 3, 10, verbose=True)
    nn.save()
    pass