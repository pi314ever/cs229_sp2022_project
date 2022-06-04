import numpy as np
# import matplotlib.pyplot as plt

#*** naive_bayes.py
# Summary: Contains a class for naive bayes multi-class classifier
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

# Filenames for saving parameters
folder = './bayes_params/'
filenames = [folder + 'W1.txt.gz', folder + 'W2.txt.gz',folder + 'b1.txt.gz',folder + 'b2.txt.gz']

class naive_bayes_model(util.classification_model):
    """
    Naive bayes multinomial event model for multi-class classification
    """
    def __init__(self, num_features:int, num_classes:int, filenames = None, verbose = False, **kwargs):
        """
        Initializes neural network

        Args:
            num_features (int): Number of features to consider
            num_classes (int): Number of classes to identify between
            filenames (list of str, optional): File location where the dataset of weights can be loaded. Order: [W1, W2, b1, b2]. Defaults to None (no pre-loaded parameters).
            verbose (bool, optional): Toggles verbose printouts.
        """
        if verbose:
            logger.info('Initializing naive bayes model')
        self.num_classes = num_classes
        self.num_features = num_features
        self.verbose = verbose

        # Load parameters
        super().__init__(filenames, **kwargs)
    def init_params(self):
        """
        Initializes the list of phi(x|y=i) and phi(y) parameters
        """
        if self.verbose:
            logger.info('Default initializing parameters')
        self.phix = [np.zeros((self.num_features,)) for _ in range(self.num_classes)]
        self.phiy = np.zeros((self.num_features,))
        pass
    def load_params(self, filenames, **kwargs):
        """
        Loads corresponding probabilities from files in the order [phi(x|y=0), ... phi(x|y=num_classes), phi(y)]
        """
        self.init_params()
        if self.verbose:
            logger.info(f'Loading dataset from {filenames}')
        try:
            assert(len(filenames) == self.num_classes + 1)
            for i, filename in enumerate(filenames[:-1]):
                self.phix[i] = np.loadtxt(filename, **kwargs)
            self.phiy = np.loadtxt(filename[-1], **kwargs)
        except:
            logger.warning('Failed to load dataset, using initial parameters')
            self.init_params()
            pass
    def fit(self, matrix, labels):
        """
        Fit the naive bayes model given a training matrix and training labels.

        Args:
            matrix (2d array): A numpy array containing word counts for the training data
            labels (2d array): The binary (0 or 1) labels for that training data

        Returns:
            Accuracy of the training set after fitting
        """
        if self.verbose:
            logger.info(f'Fitting {self.num_features} features to {self.num_classes} classes with {matrix.shape[0]} datapoints.')
        self.is_valid(matrix, labels)

        for i in range(self.num_classes):
            matrix_class = matrix[labels[:,i]==1, :]
            word_count = np.sum(matrix_class, axis=0)
            self.phix[i] = (1 + word_count) / (self.num_features + np.sum(matrix_class))

        self.phiy= np.mean(labels, axis=0)
        pred = self.predict(matrix)
        return self.accuracy(pred, labels)
    def predict(self, matrix):
        if self.verbose:
            logger.info('Predicting output of Naive Bayes')
        self.is_valid(data=matrix)
        n = matrix.shape[0]
        pred = np.zeros((n, self.num_classes))
        for i in range(self.num_classes):
            pred[:,i] = np.log(self.phiy[i] + 1e-20) + np.sum(np.log(self.phix[i] * matrix + 1e-20), axis=1)
        return pred
    def save(self, filenames, **kwargs):
        if self.verbose:
            logger.info(f'Saving parameters to {filenames}')
        try:
            assert(len(filenames) == self.num_classes + 1)
            for i in range(self.num_classes):
                np.savetxt(filenames[i], self.phix[i], **kwargs)
            np.savetxt(filenames[-1], self.phiy, **kwargs)
        except Exception as e:
            logger.error('Failed to write to file, error message:')
            print(e)
            pass
        pass
    def accuracy(self, output, labels):
        return super().accuracy(output, labels)
    def is_valid(self, data=None, labels=None):
        super().is_valid(data, labels)
    def predict_one_hot(self, data):
        return super().predict_one_hot(data)

def main():
    matrix, levels, level_map = util.load_dataset_pooled()
    n, n_features = matrix.shape
    _, n_levels = levels.shape
    c = 0.6
    train_data, train_levels, dev_data, dev_levels, test_data, test_levels = util.train_test_split(matrix, levels, c)
    # print("train data shape", train_data.shape)
    # print("test data shape", test_data.shape)
    #print("test data shape labels", test_levels.shape)
    # print("train levels mean", np.mean(train_levels, axis=0))
    # print("test levels mean", np.mean(test_levels, axis=0))
    # print("levels mean", np.mean(levels, axis=0))
    # print("n", n)
    #print("n features", n_features)
    #print("level shape", levels.shape)
    #print(np.mean(test_levels, axis=0))
    nb = naive_bayes_model(n_features, n_levels, verbose=True)
    print(nb.fit(train_data, train_levels))
    print(nb.accuracy(nb.predict(test_data), test_levels))

    # fit_model= fit_naive_bayes_model(train_data, train_levels)
    # nb_predict= predict_from_naive_bayes_model(fit_model, test_data)
    # nb_accuracy= accuracy(nb_predict, test_levels)
    # print("accuracy", nb_accuracy)
    # nb_accuracy_levels= accuracy_levels(nb_predict, test_levels)
    # print("accuracy by levels", nb_accuracy_levels)


# Testing function
if __name__ == '__main__':
    main()
