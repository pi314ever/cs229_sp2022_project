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

# Filenames for saving parameters
folder = './bayes_param/'
filenames = [folder + 'W1.txt.gz', folder + 'W2.txt.gz',folder + 'b1.txt.gz',folder + 'b2.txt.gz']

# class naive_bayes_model(util.model):
#     """
#     Bayes

    
#     """
#     def __init__(self, num_features:int, num_hidden:int, num_classes:int, reg=0, filenames = None, verbose = False):
#         """
#         Initializes neural network

#         Args:
#             num_features (int): Number of features to consider
#             num_hidden (int): Number of hidden layers
#             num_classes (int): Number of classes to identify between
#             regularized (float, optional): Regularization constant for the weights. Defaults to 0.
#             filenames (list of str, optional): File location where the dataset of weights can be loaded. Order: [W1, W2, b1, b2]. Defaults to None (no pre-loaded parameters).
#             verbose (bool, optional): Toggles verbose printouts.
#         """
#         if verbose:
#             logger.info('Initializing two layer neural network')
#         #self.num_classes = num_classes
#         #self.num_features = num_features
        
#         # Load parameters
#         super().__init__(filenames, verbose)
    
#     def init_params(self):
#         if self.verbose:
#             logger.info('Initializing weights and biases')
#         # Initialize weights
#         # np.random.seed(100) # For reproducibility
#         #self.W = [np.random.normal(0,1, (self.num_hidden, self.num_features)), np.random.normal(0,1,(self.num_classes, self.num_hidden))]
#         #self.b = [np.zeros((self.num_hidden, 1)), np.zeros((self.num_classes, 1))]
#     def load_params(self, filenames, **kwargs):
#         """
#         Load parameters with np.loadtxt()

#         Args:
#             filenames (list of str): File location where the dataset of weights can be loaded. Order: [W1, W2, b1, b2].
#             **kwargs: Keyword arguments to be passed to np.loadtxt()

        # Raises:
        #     e: Assertion errors for mismatched shape
        # """
        # if self.verbose:
        #     logger.info(f'Loading dataset from {filenames}')
        # assert(len(filenames) == 4)
        # self.W = [np.array([]), np.array([])]
        # self.b = [np.array([]), np.array([])]
        # self.W[0] = np.loadtxt(filenames[0], ndmin=2, **kwargs)
        # self.W[1] = np.loadtxt(filenames[1], ndmin=2, **kwargs)
        # self.b[0] = np.loadtxt(filenames[2], ndmin=2, **kwargs)
        # self.b[1] = np.loadtxt(filenames[3], ndmin=2, **kwargs)
        # # Confirm parameters are of the right shape
        # try:
        #     assert(self.W[0].shape == (self.num_hidden, self.num_features))
        #     assert(self.W[1].shape == (self.num_classes, self.num_hidden))
        #     assert(self.b[0].shape == (self.num_hidden, 1))
        #     assert(self.b[1].shape == (self.num_classes, 1))
        # except Exception as e:
        #     logger.error('Failed to load files, mismatched shape')
        #     raise e
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
    # np.savetxt(filenames[0], self.W[0], **kwargs)
    # np.savetxt(filenames[1], self.W[1], **kwargs)
    # np.savetxt(filenames[2], self.b[0], **kwargs)
    # np.savetxt(filenames[3], self.b[1], **kwargs)

def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    n= matrix.shape[0]
    cols= matrix.shape[1]

    matrix_y0= np.zeros((n,cols))
    matrix_y1= np.zeros((n,cols))
    matrix_y2= np.zeros((n,cols))
    ysum= np.sum(labels, axis=0)
    phi_y= np.mean(labels, axis=0)


    for i in range(n):
        matrix_y0[i,:]= labels[i,0]*matrix[i,:]
        matrix_y1[i,:]= labels[i,1]*matrix[i,:]
        matrix_y2[i,:]= labels[i,2]*matrix[i,:]

    x1y1= np.sum(matrix_y1, axis=0)
    x1y0= np.sum(matrix_y0, axis=0)
    x1y2= np.sum(matrix_y2, axis=0)
    

    phi_x_y0= (1+x1y0)/ (cols+ np.sum(matrix_y0))
    phi_x_y1= (1+x1y1)/ (cols+ np.sum(matrix_y1))
    phi_x_y2= (1+x1y2)/ (cols+ np.sum(matrix_y2))


    model=[]
    model= [phi_x_y0, phi_x_y1, phi_x_y2, phi_y]
    print("phi_x_y1 in bayes",phi_x_y1.shape)
    print("phi_x_y0 in bayes",phi_x_y0.shape)
    print("phi_x_y2 in bayes",phi_x_y2.shape)
    print("phi", phi_y.shape)
    print("phi0", phi_y)
    #print(model[1])
    return(model)





def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    phi_x_y0=model[0]
    phi_x_y1 = model[1]
    phi_x_y2 = model[2]
    phi_y= model[3]
    
    

    n= matrix.shape[0]
    cols= phi_y.shape[0]
    proby= np.zeros((n,cols))
    #sumcols= np.zeros(n)

    prediction= np.zeros((n,cols))
    proby[:,0]= np.log(phi_y[0]) + np.sum((np.log(phi_x_y0) * matrix), axis=1)
    proby[:,1]= np.log(phi_y[1]) + np.sum((np.log(phi_x_y1) * matrix), axis=1)
    proby[:,2]= np.log(phi_y[2]) + np.sum((np.log(phi_x_y2) * matrix), axis=1)
    #sumcols= np.sum(proby, axis=1)
    #print("sum dim", sumcols.shape)
    #proby[:,0]= prob_levels0/(prob_levels0+prob_levels1+ prob_levels2)
    # proby[:,1]= prob_levels1/(prob_levels0+prob_levels1+ prob_levels2)
    # proby[:,2]= prob_levels2/(prob_levels0+prob_levels1+ prob_levels2)
    #proby[:,0]= prob_levels0
    #proby[:,1]= prob_levels1
    #proby[:,2]= prob_levels2
    #print("proby", proby[:,0])
    #print("proby0", prob_levels0)



    for i in range(n):
        #print(proby[i])
        max=np.argmax(proby[i])
        #print(i, max)
        prediction[i,max]=1
        

    print("prediction", prediction.shape)
    print(np.mean(prediction, axis=0))
    return(prediction)

    #print("prob0", prob_levels0)


    #proby=1/(1+np.exp(den-num))


    #for i in range(phi_x_y1.shape):

    #prob_levels[i,0]= np.log(phi_y[0]) + np.sum((np.log(phi_x_y1) * matrix), axis=1)
    
    #den= np.log(1-phi_y) + np.sum((np.log(phi_x_y0)* matrix), axis=1)
    
    
    
    #num=1
    #den1=1
    #den2=1
    #for i in range(d):
    #    num= np.log(phi_x_y1*phi_y
    #    den1= den1*phi_x_y1*phi_y
    #    den2= den2*phi_x_y0*(1-phi_y)
    #print("num", num)
    #print("denom", den)
    #print("prediction", proby)
    
    # predicty= np.zeros(len(proby))
    # predicty[proby>=0.5]=1
    # #for i in range(len(proby)):
    #  #   if(proby[i]>=0.5):
    #  #       predicty[i]= 1
    # print("prediction")
    # print(predicty.shape)
    # 




def accuracy(output, labels):
    return(sum(np.argmax(output, axis=1) == np.argmax(labels, axis=1)) * 1. / labels.shape[0])
     

def accuracy_levels(output, labels):
    accuracy_levels=np.zeros(3)
    print(labels.shape[0], "label shape")
    print(output.shape[0], "output shape")
    sum=0
    for i in range(labels.shape[0]):
        #print(labels[i], "labels")
        #print(output[i], "output")
        if(np.argmax(output[i]) == np.argmax(labels[i]) and  (np.argmax(labels[i])==0)):
            sum+=1
            accuracy_levels[0]+=1
        if(np.argmax(output[i]) == np.argmax(labels[i]) and  (np.argmax(labels[i])==1)):
            accuracy_levels[1]+=1
            sum+=1
        if(np.argmax(output[i]) == np.argmax(labels[i]) and  (np.argmax(labels[i])==2)):
            accuracy_levels[2]+=1
            sum+=1

    labels_count= np.sum(labels, axis=0)

    print("sum accuracy", sum)
    print("sum accuracy by levels", accuracy_levels)
    print("count by levels", labels_count)

    accuracy_levels[0]/=labels_count[0]
    accuracy_levels[1]/=labels_count[1]
    accuracy_levels[2]/=labels_count[2]
    #print("mean accuracy in loop", sum/labels.shape[0])
    return(accuracy_levels)




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

# def predict(self, data):
#     """
#     Computes prediction based on weights (Array of one-hot vectors)
#     """
#     output = self.forward_prop(data)[1]
#     pred = np.zeros_like(output)
#     for i in range(output.shape[0]):
#         pred[i, np.argmax(output[i,:])] = 1
#     return pred

def main():
    matrix, levels, level_map = util.load_dataset_pooled()
    n, n_features = matrix.shape
    _, n_levels = levels.shape
    c = 0.75
    train_data, train_levels, test_data, test_levels = util.train_test_split(c, matrix, levels)
    print("train data shape", train_data.shape)
    print("test data shape", test_data.shape)
    #print("test data shape labels", test_levels.shape)
    print("train levels mean", np.mean(train_levels, axis=0))
    print("test levels mean", np.mean(test_levels, axis=0))
    print("levels mean", np.mean(levels, axis=0))
    print("n", n)
    #print("n features", n_features)
    #print("level shape", levels.shape)
    #print(np.mean(test_levels, axis=0))

    fit_model= fit_naive_bayes_model(train_data, train_levels)
    nb_predict= predict_from_naive_bayes_model(fit_model, test_data)
    nb_accuracy= accuracy(nb_predict, test_levels)
    print("accuracy", nb_accuracy)
    nb_accuracy_levels= accuracy_levels(nb_predict, test_levels)
    print("accuracy by levels", nb_accuracy_levels)





# Testing function
if __name__ == '__main__':
    # Gather data
    main()
    


    # Train nn
    #nn = two_layer_neural_network(n_features, 300, n_levels,reg=0.04, verbose=True)
    # nn.load_params(filenames)
    #epochs = 200
    #cost_train, accuracy_train, cost_dev, accuracy_dev = nn.fit(train_data, train_levels, batch_size=n, num_epochs=epochs, dev_data=test_data, dev_labels=test_levels,learning_rate=0.1)
    # nn.save(filenames)
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # t = np.arange(epochs)
    # if True:
    #     ax1.plot(t, cost_train,'r', label='train')
    #     ax1.plot(t, cost_dev, 'b', label='dev')
    #     ax1.set_xlabel('epochs')
    #     ax1.set_ylabel('loss')
    #     ax1.legend()

    #     ax2.plot(t, accuracy_train,'r', label='train')
    #     ax2.plot(t, accuracy_dev, 'b', label='dev')
    #     ax2.set_xlabel('epochs')
    #     ax2.set_ylabel('accuracy')
    #     ax2.legend()

    #     fig.savefig('./test.pdf')
