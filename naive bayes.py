import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

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

class naive_bayes(util.model):
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
        #self.num_classes = num_classes
        #self.num_features = num_features
        #self.num_hidden = num_hidden
        #self.regularized = regularized
        super().__init__(filename, verbose)

    def init_params(self):
        if self.verbose:
            logger.info('Initializing weights and biases')
        # Initialize weights
        #np.random.seed(100)
        #self.W = [np.random.normal(0,1, (self.num_hidden, self.num_features)), np.random.normal(0,1,(self.num_classes, self.num_hidden))]
        #self.b = [np.zeros((self.num_hidden, 1)), np.zeros((self.num_classes, 1))]

    def get_words(message):
    """Get the normalized list of words from a message string.
    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
        message_split= np.char.lower(message)
    #message_split= message.lower()
        message_split= np.char.split(message_split, sep=" ")
    #message_split= message_split.split(" ")
        return(message_split)


    def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

     Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
        message_split= get_words(messages)
        iteration=0
        worddict= {}
        freq={}
    
        for i in range(message_split.shape[0]):
            for word in message_split[i]:
                if (word in freq): 
                    freq[word] += 1
                    if (freq[word]==5):
                        worddict[word]=iteration
                        iteration+=1
                else: 
                    freq[word] = 1
        
        return(worddict)
    # *** END CODE HERE ***


    def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    """
        # *** START CODE HERE ***
        rows= len(messages)
        #print("rows")
        #print(rows)
        cols= len(word_dictionary)
        words_message= np.zeros((rows, cols), dtype=int)
        for i in range(rows):
            #print(messages[i])
            message_split= get_words(messages[i])
            #print(message_split.item())
            for words in list(message_split.item()):
                if words in word_dictionary:
                    words_message[i,word_dictionary[words]] +=1  
                    
        return(words_message)
    
    # *** END CODE HERE ***
    

    def fit(matrix, labels):
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

        


        phi_y= np.mean(labels)
        matrix_y1= matrix[labels==1]
        matrix_y0= matrix[labels==0]
        x1y1= np.sum(matrix_y1, axis=0)
        x1y0= np.sum(matrix_y0, axis=0)
        phi_x_y1= (1+x1y1)/ (cols+ np.sum(matrix_y1, axis=None))
        phi_x_y0= (1+x1y0)/ (cols+ np.sum(matrix_y0, axis=None))
        model=[]
        model= [phi_x_y1, phi_x_y0, phi_y]
        print("phi_x_y1 in bayes",phi_x_y1.shape)
        print("model0", phi_x_y0.shape)
        print("phi0", phi_y)
        #print(model[1])
    return(model)
    # *** END CODE HERE ***

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
        phi_x_y1=model[0]
        phi_x_y0 = model[1]
        phi_y= model[2]
        #for i in range(phi_x_y1.shape):
        num= np.log(phi_y) + np.sum((np.log(phi_x_y1) * matrix), axis=1)
        den= np.log(1-phi_y) + np.sum((np.log(phi_x_y0)* matrix), axis=1)
        
        proby=1/(1+np.exp(den-num))
        
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
        
        predicty= np.zeros(len(proby))
        predicty[proby>=0.5]=1
        #for i in range(len(proby)):
         #   if(proby[i]>=0.5):
         #       predicty[i]= 1
        print("prediction")
        print(predicty.shape)
        return(predicty)





    def output_predict(self, output):
        phi_x_y1=model[0]
        phi_x_y0 = model[1]
        phi_y= model[2]
        #for i in range(phi_x_y1.shape):
        num= np.log(phi_y) + np.sum((np.log(phi_x_y1) * matrix), axis=1)
        den= np.log(1-phi_y) + np.sum((np.log(phi_x_y0)* matrix), axis=1)
        
        proby=1/(1+np.exp(den-num))
        
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
        
        predicty= np.zeros(len(proby))
        predicty[proby>=0.5]=1
        #for i in range(len(proby)):
         #   if(proby[i]>=0.5):
         #       predicty[i]= 1
        print("prediction")
        print(predicty.shape)
        return(predicty)

        
    
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
    

    def load_dataset(self, filename):
        # Load headers
        df = pd.read_csv(filename, header= None)
        df= df.rename(columns={0:"levels", 1: "pages"})
        training_data, testing_data = train_test_split(df, test_size=0.4, random_state=25)
        train_pages= training_data['pages']
        train_labels= training_data['levels']
        test_pages= testing_data['pages']
        test_labels= testing_data['levels']

    return train_pages, train_labels, test_pages, test_labels



    def write_json(filename, value):
    """Write the provided value as JSON to the given filename"""
    with open(filename, 'w') as f:
        json.dump(value, f)

    


def main():
    train_pages, train_labels, test_pages, test_labels = load_dataset('/Users/radhika/Documents/Stanford readings/Spring2022/CS229/CS229project/cs229_sp22_dataset/level_to_page.csv')
    dictionary = create_dictionary(train_pages)
    print('Size of dictionary: ', len(dictionary))
    util.write_json('spam_dictionary', dictionary)
    train_matrix = transform_text(train_pages, dictionary)
    print("train matrix")
    print(train_matrix.shape)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    #val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_pages, dictionary)

     #model = fit_naive_bayes_model(matrix, labels)
    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)
    #print("naive bayes model", naive_bayes_model.shape)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

        

# Testing function
if __name__ == '__main__':
    main()

    #nn = two_layer_neural_network(5, 3, 10, verbose=True)
    #nn.save()
    #pass