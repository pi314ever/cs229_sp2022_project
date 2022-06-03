from multiprocessing import pool
import util
import numpy as np
from neural_network import two_layer_neural_network
from os.path import exists
from os import mkdir
import pandas as pd
import matplotlib.pyplot as plt
import re

if __name__ == '__main__':
    import logging # For debugging purposes
    # import sys
    FORMAT = "[%(levelname)s:%(filename)s:%(lineno)3s] %(funcName)s(): %(message)s"
    logging.basicConfig(filename='./neural_network_files/nn_hyperparameter_run.log',format=FORMAT, level=logging.INFO) # stream=sys.stderr

logger = logging.getLogger(__name__)

def single_test(train_data, train_levels, dev_data, dev_levels, epochs, lr, reg, n_hidden, batch_size):
    n_features = train_data.shape[1]
    n_levels = train_levels.shape[1]
    logger.info(f'Single test for neural network with {n_features} features, {n_hidden} hidden neurons, and {n_levels} classes.')
    logger.info(f'\tHyperparameters: lr = {lr}, reg = {reg}, batch_size = {batch_size}, max_epochs = {epochs}')
    nn = two_layer_neural_network(n_features, n_hidden, n_levels,reg=reg, verbose=True)
    cost_train, accuracy_train, cost_dev, accuracy_dev = nn.fit(train_data, train_levels, batch_size=batch_size, num_epochs=epochs, dev_data=dev_data, dev_labels=dev_levels,learning_rate=lr)
    if nn.err.code:
        print(nn.err)
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
    if TV > 2.5:
        return True
    else:
        return False

def multi_test(type, train_data, train_levels, dev_data, dev_levels, test_data, test_levels, max_epoch, lrs, regs, hiddens, batch_sizes):
    filename = f'./neural_network_files/{type}_summary.csv'
    df_dict = {'hidden':[], 'batch_size':[],'reg':[],'lr':[],'train_cost':[],'train_acc_total': [], 'dev_cost':[],
               'dev_acc_total':[], 'test_acc_total':[], 'err':[]}
    num_classes = train_levels.shape[1]
    for i in range(num_classes):
        df_dict[f'train_acc_{i}'] = []
        df_dict[f'dev_acc_{i}'] = []
        df_dict[f'test_acc_{i}'] = []

    for batch_size in batch_sizes:
        for lr in lrs:
            for reg in regs:
                for hidden in hiddens:
                    key = f'H{hidden}B{batch_size}L{lr}R{reg}'
                    logger.info(f'Testing {key}')
                    plot_file = f'./neural_network_files/plots/{type}_{key}.png'
                    sub_key = re.sub(r"\.",r"_",key)
                    save_path = f'./neural_network_files/{type}_{sub_key}/'
                    # Perform single test
                    if not exists(plot_file):
                        nn, cost_train, accuracy_train, cost_dev, accuracy_dev = single_test(train_data, train_levels, dev_data, dev_levels, max_epoch, lr, reg, hidden, batch_size)
                        # Plot and save to file
                        if not exists(save_path):
                            mkdir(save_path)
                        nn.save(save_path)
                        test_acc = nn.accuracy(nn.predict(test_data), test_levels)
                        plot(cost_train, accuracy_train, cost_dev, accuracy_dev, nn.num_classes, plot_file, type, key)
                        # Save to lists
                        logger.info('Saving to dictionary')
                        df_dict['hidden'].append(hidden)
                        df_dict['reg'].append(reg)
                        df_dict['lr'].append(lr)
                        df_dict['batch_size'].append(batch_size)
                        df_dict['train_cost'].append(cost_train[-1])
                        df_dict['train_acc_total'].append(accuracy_train[-1,-1])
                        df_dict['dev_cost'].append(cost_dev[-1])
                        df_dict['dev_acc_total'].append(accuracy_dev[-1,-1])
                        df_dict['test_acc_total'].append(test_acc[-1])
                        df_dict['err'].append(nn.err.code)
                        for i in range(num_classes):
                            df_dict[f'train_acc_{i}'].append(accuracy_train[-1,i])
                            df_dict[f'dev_acc_{i}'].append(accuracy_dev[-1,i])
                            df_dict[f'test_acc_{i}'].append(test_acc[i])

    # Save summary to csv file
    if not exists(filename):
        logger.info('Saving to file')
        df = pd.DataFrame(df_dict)
        df.to_csv(filename)
        logger.info(f'Top 5 by dev set:\n{df.sort_values(by="dev_acc_total").head()}')
        logger.info(f'Top 5 by test set:\n{df.sort_values(by="test_acc_total").head()}')



def plot(cost_train, accuracy_train, cost_dev, accuracy_dev, num_classes, filename, type, key):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title(f'{type} with KEY {key}')
    ax1.plot(np.arange(len(cost_train)), cost_train,'r', label='train')
    ax1.plot(np.arange(len(cost_dev)), cost_dev, 'b', label='dev')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.legend()

    labels = list(np.arange(num_classes))
    labels.append('all')
    train_labels = [f'train {labels[i]}' for i in range(len(labels))]
    dev_labels = [f'dev {labels[i]}' for i in range(len(labels))]

    ax2.plot(np.arange(len(accuracy_train)), accuracy_train[:,:-1],':', label=train_labels[:-1])
    ax2.plot(np.arange(len(accuracy_dev)), accuracy_dev[:,:-1], '--', label=dev_labels[:-1])
    ax2.plot(np.arange(len(accuracy_train)), accuracy_train[:,-1],'r', label=train_labels[-1], linewidth=2)
    ax2.plot(np.arange(len(accuracy_dev)), accuracy_dev[:,-1],'b', label=dev_labels[-1],linewidth=2)
    ax2.set_xlabel('epochs')
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('accuracy')
    ax2.legend()

    fig.savefig(filename)


def main():
    # Hyperparameter ranges
    max_epoch = 500
    lr_steps = [0.05, 0.1, 0.4]
    reg_steps = [0, 0.001, 0.005, 0.01, 0.05]
    hidden_steps = [10, 100, 300]
    batch_sizes = [100]

    if not exists('./neural_network_files/plots/'):
        mkdir('./neural_network_files/plots/')

    # Load dataset
    matrix, levels, level_map = util.load_dataset(pooled=True, by_books=False, vectorizer=False)
    train_data, train_levels, dev_data, dev_levels, test_data, test_levels = util.train_test_split(matrix, levels)

    # Perform multi test on regular word model
    multi_test('Regular',train_data, train_levels, dev_data, dev_levels, test_data, test_levels, max_epoch, lr_steps, reg_steps, hidden_steps, batch_sizes)

    print('======================================')

    # Hyperparameter ranges
    max_epoch = 500
    lr_steps = [0.4, 0.6, 0.8]
    reg_steps = [0, 0.0001, 0.0005, 0.01]
    hidden_steps = [10, 100, 300]
    batch_sizes = [100]

    # Load from pretrained model
    if not exists('./neural_network_files/matrix.txt.gz'):
        logger.info('Generating dataset from pretrained model')
        matrix, levels, level_map = util.load_dataset(pooled=True, by_books=False, vectorizer=True)
        np.savetxt('./neural_network_files/matrix.txt.gz', matrix)
        np.savetxt('./neural_network_files/levels.txt.gz', levels)
    else:
        logger.info('Loading dataset from files')
        matrix = np.loadtxt('./neural_network_files/matrix.txt.gz')
        levels = np.loadtxt('./neural_network_files/levels.txt.gz')
    train_data, train_levels, dev_data, dev_levels, test_data, test_levels = util.train_test_split(matrix, levels)

    # Perform test with pretrained model
    multi_test('Vectorized',train_data, train_levels, dev_data, dev_levels, test_data, test_levels, max_epoch, lr_steps, reg_steps, hidden_steps, batch_sizes)



if __name__ == '__main__':
    main()
