import numpy as np
from sklearn.cluster import KMeans

#*** cs229_kmeans.py
# Summary: Contains and tests k-means on various representations of the dataset
#***

import util

#** Logger usage:
# logger.debug(): For all parameters useful in debugging (i.e. matrix shapes, important scalars, etc.)
# logger.info(): For all information on what the solver is doing
# logger.warning(): For all information that might cause known instability (i.e. underflow/overflow, etc.). Can also be used for places with implementations to-be-defined
# logger.error(): For notifying failed attempts at calculation (i.e. any exception, bad data, etc.)
#***


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
        matrix (n x d np array): Matrix of n examples of dimension d
        num_cluster (int): Number of clusters for kmeans
    Returns:
        labels (size n array): Vector of indices for each example to the classification
    """
    labels= KMeans(n_clusters= num_cluster, random_state=0).fit_predict(matrix)
    return labels

def xtab(*cols):
    uniq_vals_all_cols, idx = zip( *(np.unique(col, return_inverse=True) for col in cols) )
    shape_xt = [uniq_vals_col.size for uniq_vals_col in uniq_vals_all_cols]
    xt = np.zeros(shape_xt, dtype='uint')
    wt=1
    print(shape_xt, idx)
    np.add.at(xt, idx, wt)
    return uniq_vals_all_cols, xt

def index2matrix(vec):
    """
    Creates a matrix of one hot vectors out of a vector of indices

    Args:
        vec (1d iterable): Vector of indices

    Returns:
        matrix: Matrix of one-hot vectors
    """
    out = np.zeros((len(vec), max(vec) + 1))
    for i, idx in enumerate(vec):
        out[i, idx] = 1
    return out

def main():
    matrix, levels, _ = util.load_dataset()
    k = 14
    print(k)
    print("matrix shape", matrix.shape)
    print("grade_level",levels.shape)

    ## K means no change in matrix
    labels_tf= kmean_cluster(matrix,k)
    print(labels_tf.shape)
    print(levels.shape)
    # print(np.unique(labels_tf))

    ## K modified terms matrix
    matrix_tf, matrix_tfidf= tfidf(matrix)
    # print("tfidf", matrix_tfidf.shape)
    labels_tfidf= kmean_cluster(matrix_tfidf,k)
    # print(np.unique(labels_tfidf))

    print("simple term frequency matrix")
    print(index2matrix(labels_tf).T @ levels)
    # uv, xt = xtab(labels_tf, levels)
    # print(uv)
    # print(xt)

    print("term frequency matrix")
    print(index2matrix(labels_tfidf).T @ levels)
    # uvidf, xtidf = xtab(labels_tfidf, levels)
    # print(uvidf)
    # print(xtidf)

    # Using vectorizer
    matrix = np.loadtxt('./neural_network_files/matrix.txt.gz')

    ## K means no change in matrix
    labels_tf= kmean_cluster(matrix,k)
    print("Vectorized inputs")
    print(index2matrix(labels_tf).T @ levels)


# Testing function
if __name__ == '__main__':
    main()
