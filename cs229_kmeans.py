import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
# import matplotlib.pyplot as plt

#*** naive_bayes.py
# Summary: Contains a class for naive bayes multi-class classifier
#***

import util

#** Logger usage:
# logger.debug(): For all parameters useful in debugging (i.e. matrix shapes, important scalars, etc.)
# logger.info(): For all information on what the solver is doing
# logger.warning(): For all information that might cause known instability (i.e. underflow/overflow, etc.). Can also be used for places with implementations to-be-defined
# logger.error(): For notifying failed attempts at calculation (i.e. any exception, bad data, etc.)
#***

"""
    Naive bayes multinomial event model for multi-class classification
    """

def tfidf(matrix):

    """"
    Args:
        term frequency
    Returns:
        tf idf score for each word in a page

    """
    tf= np.log(1+ matrix) # log of term frequency - how often a word appears in a page
    Nwords= matrix.shape[1]
    Npages=matrix.shape[0]
    matchk = np.zeros((matrix.shape[0],matrix.shape[1]))
    matchk[matrix>0]= 1 # 1 or 0 depending on whether word is in page
    df= np.sum(matchk, axis=0) #Number of pages a word appears in or document frequency
    idf= np.log(Npages/ df) #inverse document frequency - log (number of pages/ number of pages words appears in)
    tf_idf= tf * idf
    return tf, tf_idf


    


def kmean_cluster(matrix, num_cluster=3):
    """"
    Args:
    Number of clusters
    Returns:
    Sentences mapped to classification levels
    """
    labels= KMeans(n_clusters= num_cluster, random_state=0).fit_predict(matrix)
    return labels

def xtab(*cols):
    uniq_vals_all_cols, idx = zip( *(np.unique(col, return_inverse=True) for col in cols) )
    shape_xt = [uniq_vals_col.size for uniq_vals_col in uniq_vals_all_cols]
    xt = np.zeros(shape_xt, dtype='uint')
    wt=1
    np.add.at(xt, idx, wt)
    return uniq_vals_all_cols, xt

def main():

    matrix, grade_level, levels, level_map = util.load_dataset_pooled()
    n, n_features = matrix.shape
    _, n_levels = levels.shape
    c = 0.75
    print(n_levels)
    print("matrix shape", matrix.shape)
    print("grade_level",grade_level.shape, grade_level)

    ## K means no change in matrix
    labels_tf= kmean_cluster(matrix,n_levels)
    print(labels_tf.shape)
    print(levels.shape)
    print(np.unique(labels_tf))

    ## K modified terms matrix
    matrix_tf, matrix_tfidf= tfidf(matrix)
    print("tfidf", matrix_tfidf.shape, matrix_tfidf)
    labels_tfidf= kmean_cluster(matrix_tfidf,n_levels)
    print(np.unique(labels_tfidf))



    uv, xt = xtab(labels_tf, grade_level)
    print("simple term frequency matrix")
    print(uv)
    print(xt)


    uvidf, xtidf = xtab(labels_tfidf, grade_level)
    print("term frequency  matrix")
    print(uvidf)
    print(xtidf)

    





    #plt.scatter(labels_tf, grade_level)
    #plt.show()
    # pd.crosstab(labels_tf, grade_level)
    # foo = pd.Categorical(['a', 'b'], categories=['a', 'b', 'c'])
    # bar = pd.Categorical(['d', 'e'], categories=['d', 'e', 'f'])
    # pd.crosstab(foo, bar)

    

    


# Testing function
if __name__ == '__main__':
    main()
