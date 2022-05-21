import numpy as np
from time import time
import matplotlib.pyplot as plt

#*** neural_network.py
# Summary: Contains a class for generating a 2-layer fully connected neural network
#***

import util

import logging, sys # For debugging purposes
FORMAT = "[%(levelname)s:%(filename)s:%(lineno)3s] %(funcName)s(): %(message)s"
logging.basicConfig(format=FORMAT, stream=sys.stderr)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#** Logger usage:
# logger.debug(): For all parameters useful in debugging (i.e. matrix shapes, important scalars, etc.)
# logger.info(): For all information on what the solver is doing
# logger.warning(): For all information that might cause known instability (i.e. underflow/overflow, etc.). Can also be used for places with implementations to-be-defined
# logger.error(): For notifying failed attempts at calculation (i.e. any exception, bad data, etc.)
#***

# Hyperparameters
epochs = 500
lr = 0.01
reg = 0.15
n_hidden = 200
batch_size = 1000
var_lr = False
# Filenames for saving parameters
folder = f'./neural_network_parameters/test_E{epochs}_LR{lr:.2e}_R{reg:.2e}_H{n_hidden}_'
# folder = './neural_network_parameters/'
filenames = [folder + 'W1.txt.gz', folder + 'W2.txt.gz',folder + 'b1.txt.gz',folder + 'b2.txt.gz']
figure_filename = './test.pdf'

plot = True
save = False
load = False

class two_layer_neural_network(util.classification_model):
    """
    Two layered fully connected neural network with Batch Gradient Descent optimizer

    Architecture:
        Features (Input) -> Sigmoid (Hidden)-> Softmax (Output)

    All data must be shaped as (num_examples, num_features)
    All labels must be shaped as (num_examples, num_classes)
    """
    def __init__(self, num_features:int, num_hidden:int, num_classes:int, reg=0, filenames = None, verbose = False, **kwargs):
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
        self.verbose = verbose
        # Load parameters
        super().__init__(filenames, **kwargs)
    def init_params(self):
        if self.verbose:
            logger.info('Default initializing weights and biases')
        # Initialize weights
        # np.random.seed(100) # For reproducibility
        sigma = 1
        self.W = [np.random.normal(0,sigma, (self.num_hidden, self.num_features)), np.random.normal(0,sigma,(self.num_classes, self.num_hidden))]
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
        try:
            assert(len(filenames) == 4)
            self.W = [np.array([]), np.array([])]
            self.b = [np.array([]), np.array([])]
            self.W[0] = np.loadtxt(filenames[0], ndmin=2, **kwargs)
            self.W[1] = np.loadtxt(filenames[1], ndmin=2, **kwargs)
            self.b[0] = np.loadtxt(filenames[2], ndmin=2, **kwargs)
            self.b[1] = np.loadtxt(filenames[3], ndmin=2, **kwargs)
        except:
            logger.warning('Failed to load dataset, performing default initialization.')
            self.init_params()
            pass
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
    def fit(self, train_data, train_labels, batch_size, num_epochs = 30, learning_rate = 5., dev_data = None, dev_labels = None, var_lr = False):
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
            var_lr (bool): Whether or not the learning rate is decreasing

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
        try:
            for epoch in range(num_epochs):
                # Shuffle training data
                perm = np.random.shuffle(np.arange(train_data.shape[0]))
                train_data = train_data[perm, :].squeeze()
                train_labels = train_labels[perm, :].squeeze()
                if self.verbose:
                    logger.info(f'Epoch {epoch + 1} of {num_epochs}')
                if var_lr:
                    learning_rate /= np.log(np.log(0.1 * epoch + 1) + 1) + 1
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
        except KeyboardInterrupt:
            logger.info('Keyboard interrupted, stopping training process.')
            pass
        except Exception as e:
            raise e
        end = time()
        if self.verbose:
            logger.info(f'Training took {(end - begin)/60} minutes')
        if has_dev:
            return np.array(cost_train), np.array(accuracy_train), np.array(cost_dev), np.array(accuracy_dev)
        else:
            return np.array(cost_train), np.array(accuracy_train)
    def accuracy(self, output, labels):
        assert(output.shape == labels.shape)
        acc = []
        for i in range(self.num_classes):
            acc.append(sum(np.logical_and(np.argmax(output, axis=1) == i, np.argmax(labels, axis=1) == i)) * 1. / sum(labels[:,i]))
        acc.append(sum(np.argmax(output, axis=1) == np.argmax(labels, axis=1)) * 1. / labels.shape[0])
        return acc
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
        self.W[0] -= learning_rate * (-dCEdz1.T @ data / n + 2 * self.reg * self.W[0])
        self.W[1] -= learning_rate * (-dCEdz2.T @ hidden / n + 2 * self.reg * self.W[1])
        self.b[0] -= -(learning_rate * (np.average(dCEdz1, axis=0))).reshape(self.b[0].shape)
        self.b[1] -= -(learning_rate * (np.average(dCEdz2, axis=0))).reshape(self.b[1].shape)
    def predict(self, data):
        return self.forward_prop(data)[1]
    def predict_one_hot(self, data):
        """
        Computes prediction based on weights (Array of one-hot vectors)
        """
        output = self.predict(data)
        pred = np.zeros_like(output)
        for i in range(output.shape[0]):
            pred[i, np.argmax(output[i,:])] = 1
        return pred

# Testing function
def main():
    # Gather data
    matrix, levels, level_map = util.load_dataset(pooled=True, by_books=True)
    n, n_features = matrix.shape
    _, n_levels = levels.shape
    c = 0.75
    train_data, train_levels, test_data, test_levels = util.train_test_split(c, matrix, levels)
    num_class = [sum(levels[:,i]) for i in range(n_levels)]
    for i in range(n_levels):
        print(f'Number of class {i}: {num_class[i]}')

    # Train nn
    nn = two_layer_neural_network(n_features, n_hidden, n_levels,reg=reg, verbose=True)
    if load:
        nn.load_params(filenames)
    cost_train, accuracy_train, cost_dev, accuracy_dev = nn.fit(train_data, train_levels, batch_size=batch_size, num_epochs=epochs, dev_data=test_data, dev_labels=test_levels,learning_rate=lr, var_lr = var_lr)
    if save:
        nn.save(filenames)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    if plot:
        ax1.plot(np.arange(len(cost_train)), cost_train,'r', label='train')
        ax1.plot(np.arange(len(cost_dev)), cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        ax1.legend()

        labels = list(np.arange(nn.num_classes))
        labels.append('all')
        train_labels = [f'train {labels[i]}' for i in range(len(labels))]
        dev_labels = [f'dev {labels[i]}' for i in range(len(labels))]

        ax2.plot(np.arange(len(accuracy_train)), accuracy_train[:,:-1],':', label=train_labels[:-1])
        ax2.plot(np.arange(len(accuracy_train)), accuracy_train[:,-1],'r', label=train_labels[-1], linewidth=2)
        ax2.plot(np.arange(len(accuracy_dev)), accuracy_dev[:,:-1], '--', label=dev_labels[:-1])
        ax2.plot(np.arange(len(accuracy_dev)), accuracy_dev[:,-1],'b', label=dev_labels[-1],linewidth=2)
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig(figure_filename)

if __name__ == '__main__':
    main()

