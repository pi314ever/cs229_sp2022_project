import numpy as np
from time import time
import matplotlib.pyplot as plt

#*** neural_network.py
# Summary: Contains a class for generating a 2-layer fully connected neural network
#***

import util

import logging # For debugging purposes
# import sys
if __name__ == '__main__':
    FORMAT = "[%(levelname)s:%(filename)s:%(lineno)3s] %(funcName)s(): %(message)s"
    logging.basicConfig(filename='./neural_network_files/nn.log', filemode='a',format=FORMAT, level=logging.INFO) # stream=sys.stderr
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#** Logger usage:
# logger.debug(): For all parameters useful in debugging (i.e. matrix shapes, important scalars, etc.)
# logger.info(): For all information on what the solver is doing
# logger.warning(): For all information that might cause known instability (i.e. underflow/overflow, etc.). Can also be used for places with implementations to-be-defined
# logger.error(): For notifying failed attempts at calculation (i.e. any exception, bad data, etc.)
#***

if __name__ == '__main__':
    # Hyperparameters
    epochs = 500
    lr = 0.0015
    reg = 0.01
    n_hidden = 100
    layers = 5
    batch_size = 100
    var_lr = False
    # Filenames for saving parameters
    header = f'./neural_network_files/test_E{epochs}_LR{lr:.2e}_R{reg:.2e}_H{n_hidden}_'
    # folder = './neural_network_parameters/'
    # filenames = [folder + 'W1.txt.gz', folder + 'W2.txt.gz',folder + 'b1.txt.gz',folder + 'b2.txt.gz']
    figure_filename = './test.pdf'

    plot = True
    save = False
    load = False

class n_layer_neural_network(util.classification_model):
    """
    N layered fully connected neural network with Batch Gradient Descent optimizer

    Architecture:
        Features (Input) -> Activation (N Hidden)-> Softmax (Output)

    All data must be shaped as (num_examples, num_features)
    All labels must be shaped as (num_examples, num_classes)
    """
    def __init__(self, num_features:int, num_hidden:int, num_hidden_layers:int, num_classes:int, activation_func, d_activation_func, reg=0, filenames = None, verbose = False, **kwargs):
        """
        Initializes neural network

        Args:
            num_features (int): Number of features to consider
            num_hidden (int): Number of hidden nodes per layer
            num_hidden_layers (int): Number of hidden layers
            num_classes (int): Number of classes to identify between
            activation_func (list of lambda):
            regularized (float, optional): Regularization constant for the weights. Defaults to 0.
            filenames (list of str, optional): File location where the dataset of weights can be loaded. Order: [W1, W2, b1, b2]. Defaults to None (no pre-loaded parameters).
            verbose (bool, optional): Toggles verbose printouts.
        """
        if verbose:
            logger.info(f'Initializing {num_hidden_layers + 1} layer neural network')
        assert num_hidden_layers == len(activation_func), 'Improper length of activation functions'
        assert num_hidden_layers == len(d_activation_func), 'Improper length of derivative activation functions'
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_hidden_layers = num_hidden_layers
        self.act = activation_func
        self.dact = d_activation_func
        self.reg = reg
        self.verbose = verbose
        self.err = nn_error()
        # Load parameters
        super().__init__(filenames, **kwargs)
    def init_params(self):
        if self.verbose:
            logger.info('Default initializing weights and biases')
        # Initialize weights
        sigma = 1
        rng = np.random.default_rng(100)
        self.W = [rng.normal(0,sigma, (self.num_hidden, self.num_features))]
        self.b = [np.zeros((self.num_hidden, 1))]
        for _ in range(self.num_hidden_layers - 1):
            self.W.append(rng.normal(0, sigma, (self.num_hidden, self.num_hidden)))
            self.b.append(np.zeros((self.num_hidden, 1)))
        self.W.append(rng.normal(0,sigma,(self.num_classes, self.num_hidden)))
        self.b.append(np.zeros((self.num_classes, 1)))
    def load_params(self, header:str, **kwargs):
        """
        Load parameters with np.loadtxt()

        Args:
            header (str): File location where the dataset of weights can be loaded.
            **kwargs: Keyword arguments to be passed to np.loadtxt()

        Raises:
            e: Assertion errors for mismatched shape
        """
        W_filenames = [f'{header}W{i}.txt.gz' for i in range(self.num_hidden_layers)]
        b_filenames = [f'{header}b{i}.txt.gz' for i in range(self.num_hidden_layers + 1)]
        if self.verbose:
            logger.info(f'Loading dataset from {header}')
        try:
            self.W = [np.loadtxt(W_filenames[i], ndmin=2, **kwargs) for i in range(self.num_hidden_layers + 1)]
            self.b = [np.loadtxt(b_filenames[i], ndmin=2, **kwargs) for i in range(self.num_hidden_layers + 1)]
        except:
            logger.warning('Failed to load dataset, performing default initialization.')
            self.init_params()
            pass
        # Confirm parameters are of the right shape
        try:
            assert(self.W[0].shape == (self.num_hidden, self.num_features))
            assert(self.W[-1].shape == (self.num_classes, self.num_hidden))
            assert(self.b[0].shape == (self.num_hidden, 1))
            assert(self.b[-1].shape == (self.num_classes, 1))
            for i in range(1, self.num_hidden_layers):
                assert(self.W[i].shape == (self.num_hidden, self.num_hidden))
                assert(self.b[i].shape == (self.num_hidden, 1))
        except Exception as e:
            logger.error('Failed to load files, mismatched shape')
            raise e
    def save(self, header:str, **kwargs):
        """
        Saves parameters to filenames using np.savetxt()

        Args:
            header (str): Header to the file location where the dataset of weights can be saved. ex: header='test_' -> 'test_W1.txt.gz' for the file that contains W1.
            **kwargs: Keyword arguments to be passed to np.savetxt()
        """
        W_filenames = [f'{header}W{i}.txt.gz' for i in range(self.num_hidden_layers)]
        b_filenames = [f'{header}b{i}.txt.gz' for i in range(self.num_hidden_layers)]
        if self.verbose:
            logger.info(f'Saving parameters to {header}')
        for i in range(self.num_hidden_layers + 1):
            np.savetxt(W_filenames[i], self.W[i], **kwargs)
            np.savetxt(b_filenames[i], self.b[i], **kwargs)
    def fit(self, train_data, train_labels, batch_size, num_epochs = 30, learning_rate = 5., dev_data = None, dev_labels = None, var_lr = False, print_epochs = False):
        """
        Fits neural network based on training data and training labels using batch gradient descent (Can convert to GD if batch size = number of examples)

        Aliases:
            nt: Number of training examples
            nd: Number of dev examples
            d: num_features
            c: num_classes

        Args:
            train_data (nt x d array)
            train_labels (nt x c array)
            batch_size (int)
            num_epochs (int, optional): Number of epochs for training. Defaults to 30.
            learning_rate (float, optional): Batch GD learning rate. Defaults to 5.
            dev_data (nd x d array, optional): Development data, for use in comparing incremental increases. Defaults to None.
            dev_labels (nd x c array, optional): Developmental labels, for use in comparing incremental increases. Defaults to None.
            var_lr (bool): Whether or not the learning rate is decreasing

        Returns:
            cost_train (epochs x 1 np array): Cost history for training data
            accuracy_train (epochs x (c+1) np array): Accuracy history for training data
            cost_dev (epochs x 1 np array): Cost history for dev data (if provided)
            accuracy_dev (epochs x (c+1) np array): Accuracy history of dev data (if provided)
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
            if self.verbose:
                logger.info('Start training')
            for epoch in range(num_epochs):
                # Shuffle training data
                perm = np.random.shuffle(np.arange(train_data.shape[0]))
                train_data = train_data[perm, :].squeeze()
                train_labels = train_labels[perm, :].squeeze()
                if print_epochs:
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
                # Check for termination conditions after several iterations
                if epoch > 20:
                    if sum(np.abs(np.array(accuracy_dev)[-19:,-1] - np.array(accuracy_dev)[-20:-1,-1])) > 1.5:
                        # Error is fluctuating too much
                        self.err.set_code(1)
                        break
                    if np.abs(cost_dev[-1] - cost_dev[-2]) < 1e-4:
                        # Loss stabilized
                        self.err.set_code(2, train_cost = cost_train[-1], train_acc = accuracy_train[-1][-1],dev_cost = cost_dev[-1], dev_acc = accuracy_dev[-1][-1])
                        break
        except KeyboardInterrupt:
            logger.info('Keyboard interrupted, stopping training process.')
            self.err.set_code(100, iter = epoch + 1)
            pass
        except Exception as e:
            raise e
        end = time()
        if self.verbose:
            logger.info(f'Training took {(end - begin)/60:.2f} minutes, average {(end - begin) / (epoch + 1):.6f} sec / epoch over {epoch + 1} epochs')
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
            loss (float): Average loss for the predicted output (if labels are given, else returns -1)
        """
        self.is_valid(data=data)
        hidden = []
        for i, func in enumerate(self.act):
            if i == 0:
                # First step, use data
                hidden.append(func((self.W[0] @ data.T + self.b[0]).T))
            else:
                hidden.append(func((self.W[i] @ hidden[i-1].T + self.b[i]).T))
        output = util.softmax((self.W[-1] @ hidden[-1].T + self.b[-1]).T)
        if labels is not None:
            loss = self.loss(labels, output)
        else:
            loss = -1
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
        dCEdzi = labels - output
        for i in np.arange(self.num_hidden_layers, -1, -1):
            self.b[i] -= learning_rate * (np.average(dCEdzi, axis=0)).reshape(self.b[i].shape)
            if i == 0:
                self.W[i] -= learning_rate * (-dCEdzi.T @ data / n + 2 * self.reg * self.W[i])
            else:
                self.W[i] -= learning_rate * (-dCEdzi.T @ hidden[i - 1] / n + 2 * self.reg * self.W[i])
                dCEdzi = (dCEdzi @ self.W[i]) * self.dact[i - 1](hidden[i - 1])
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

# To preserve previous API
class two_layer_neural_network(n_layer_neural_network):
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
        # Load parameters
        super().__init__(num_features, num_hidden, 1, num_classes, [util.sigmoid], [util.dsigmoid], reg, filenames, verbose, **kwargs)
class nn_error:
    def __init__(self) -> None:
        self.code = 0
    def set_code(self, code:int, **kwargs) -> None:
        self.code = code
        self.kwargs = kwargs
        pass
    def __repr__(self) -> str:
        if self.code == 0:
            return 'No errors found'
        if self.code == 1:
            return 'ERROR: Accuracy is fluctuating too much, try reducing learning rate for more stability.'
        if self.code == 2:
            return f'Stabilized with train cost {self.kwargs["train_cost"]} and dev cost {self.kwargs["dev_cost"]}\n\ttrain accuracy {self.kwargs["train_acc"]} and dev accuracy {self.kwargs["dev_acc"]}'
        if self.code == 100:
            return f'ERROR: Keyboard interrupted at iteration {self.kwargs["iter"]}. Model may not have converged yet.'

# Testing function
def main():
    # Gather data
    matrix = np.loadtxt('./neural_network_files/matrix.txt.gz')
    levels = np.loadtxt('./neural_network_files/levels.txt.gz')
    n, n_features = matrix.shape
    _, n_levels = levels.shape
    c = 0.6
    train_data, train_levels, dev_data, dev_levels, test_data, test_levels = util.train_test_split(matrix, levels, c)
    num_class = [sum(levels[:,i]) for i in range(n_levels)]
    for i in range(n_levels):
        print(f'Number of class {i}: {num_class[i]}')

    # Train nn
    # nn = two_layer_neural_network(n_features, n_hidden, n_levels,reg=reg, verbose=True)
    nn = n_layer_neural_network(n_features, n_hidden, layers, n_levels, [util.sigmoid] * layers, [util.dsigmoid] * layers, reg, verbose=True)
    if load:
        nn.load_params(header)
    cost_train, accuracy_train, cost_dev, accuracy_dev = nn.fit(train_data, train_levels, batch_size=batch_size, num_epochs=epochs, dev_data=dev_data, dev_labels=dev_levels,learning_rate=lr, var_lr = var_lr, print_epochs=True)
    print(nn.err)
    print(f'Final training accuracy: {accuracy_train[-1,-1]}, dev accuracy: {accuracy_dev[-1,-1]}')
    pred = nn.predict_one_hot(matrix)
    # Prediction matrix
    levels_all = util.load_dataset(pooled=False)[1]
    print((pred.T @ levels_all).astype(int))
    print((pred.T @ levels).astype(int))
    print(np.linalg.det(pred.T @ levels))
    if save:
        nn.save(header)
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
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
        ax2.plot(np.arange(len(accuracy_dev)), accuracy_dev[:,:-1], '--', label=dev_labels[:-1])
        ax2.plot(np.arange(len(accuracy_train)), accuracy_train[:,-1],'r', label=train_labels[-1], linewidth=2)
        ax2.plot(np.arange(len(accuracy_dev)), accuracy_dev[:,-1],'b', label=dev_labels[-1],linewidth=2)
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig(figure_filename)

if __name__ == '__main__':
    main()

