from multiprocessing import pool
import util
import numpy as np
from neural_network import two_layer_neural_network
from neural_network import logger
from os.path import exists

def single_test(train_data, train_levels, dev_data, dev_levels, epochs, lr, reg, n_hidden, batch_size):
    n_features = train_data.shape[1]
    n_levels = train_levels.shape[1]
    nn = two_layer_neural_network(n_features, n_hidden, n_levels,reg=reg, verbose=True)
    cost_train, accuracy_train, cost_dev, accuracy_dev = nn.fit(train_data, train_levels, batch_size=batch_size, num_epochs=epochs, dev_data=dev_data, dev_labels=dev_levels,learning_rate=lr)
    return nn, cost_train, accuracy_train, cost_dev, accuracy_dev

def is_fluctuating(accuracy_history) -> bool:
    """
    Determines if the training run is fluctuating

    Args:
        accuracy_history (list of floats): History of accuracy reported by the fitting history

    Returns:
        True if the accuracy history has a total variance (TV) of greater than 1.5
        False otherwise
    """
    TV = sum(np.abs(accuracy_history[:-1] - accuracy_history[1:]))
    if TV > 1.5:
        return True
    else:
        return False

def multi_test(train_data, train_levels, dev_data, dev_levels, test_data, test_levels, max_epoch, lr_range, reg_range, hidden_range, batch_sizes):

    pass


def main():
    # Hyperparameter ranges
    max_epoch = 500
    lr_steps = [0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1]
    reg_steps = [0, 0.2]
    hidden_steps = [10, 100, 200, 300]
    batch_sizes = [1, 100, 10000]
    # Load dataset
    matrix, levels, level_map = util.load_dataset(pooled=True, by_books=False, vectorizer=False)
    train_data, train_levels, dev_data, dev_levels, test_data, test_levels = util.train_test_split(matrix, levels)
    # Perform multi test on regular word model
    multi_test(train_data, train_levels, dev_data, dev_levels, test_data, test_levels, max_epoch, lr_steps, reg_steps, hidden_steps, batch_sizes)

    if not exists('./neural_network_files/matrix.txt.gz'):
        logger.info('Generating dataset from pretrained model')
        matrix, levels, level_map = util.load_dataset(pooled=True, by_books=False, vectorizer=True)
        np.savetxt('./neural_network_files/matrix.txt.gz', matrix)
        np.savetxt('./neural_network_files/levels.txt.gz', levels)
    else:
        logger.info('Loading dataset from files')
        matrix = np.loadtxt('./neural_network_files/matrix.txt.gz')
        levels = np.loadtxt('./neural_network_files/levels.txt.gz')



if __name__ == '__main__':
    main()
