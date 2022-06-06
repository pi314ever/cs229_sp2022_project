import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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



def barplt_pooled(kmeans_matrix, k, algtype):
    
    labels = np.arange(k)
    x = np.arange(len(labels))  # the label locations
    barWidth = 0.25  # the width of the bars

    fig, ax = plt.subplots()

    bars1= kmeans_matrix[:,0]
    bars2= kmeans_matrix[:,1]
    bars3= kmeans_matrix[:,2]

    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]


    # rects1 = ax.bar(x - width/3, kmeans_matrix[], width, label='TF')
    # rects2 = ax.bar(x + width/3, women_means, width, label='TF-IDF')
    # rects2 = ax.bar(x + width/3, women_means, width, label='BERT-vec')

    # Make the plot
    plt.bar(r1, bars1,  width=barWidth, edgecolor='white', label='K', align="center")
    plt.bar(r2, bars2,  width=barWidth, edgecolor='white', label='G1', align="center")
    plt.bar(r3, bars3,  width=barWidth, edgecolor='white', label='G2', align="center")


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of pages')
    plot_title = f"Algorithm, {algtype}: {k} clusters."
    ax.set_title(plot_title)
    # ax.set_title('Algorithm %s: %iclusters' %{"s":algtype, "i":str(k)})
    ax.set_xticks(x, labels)
    ax.set_xlabel('Kmeans clusters')
    ax.legend()

    for bar in ax.patches:
      # The text annotation for each bar should be its height.
      bar_value = bar.get_height()
      # Format the text with commas to separate thousands. You can do
      # any type of formatting here though.
      text = f'{bar_value:,}'
      # This will give the middle of each bar on the x-axis.
      text_x = bar.get_x() + bar.get_width() / 2
      # get_y() is where the bar starts so we add the height to it.
      text_y = bar.get_y() + bar_value
      # If we want the text to be the same color as the bar, we can
      # get the color like so:
      bar_color = bar.get_facecolor()
      # If you want a consistent color, you can just set it as a constant, e.g. #222222
      ax.text(text_x, text_y, text, ha='center', va='bottom', color=bar_color,
              size=12)

    

    fig.tight_layout()
    # key = f'H{hidden}B{batch_size}L{lr}R{reg}'
    #                 logger.info(f'Testing {key}')
    #                 plot_file = f'./neural_network_files/plots/{type}_{key}.png'
    #                 sub_key = re.sub(r"\.",r"_",key)
    #                 save_path = f'./neural_network_files/{type}_{sub_key}/'
    pltname = f"./kmeans_{k}clusters_{algtype}.png"
    print(pltname)
    plt.savefig(pltname, format= 'png') 


def main():
    np.set_printoptions(suppress=True)
    
    
    #k = 3

    for k in range(2,5):
        
        print(k)
        matrix, levels, _ = util.load_dataset_pooled()
        print("matrix shape", matrix.shape)
        print("grade_level",levels.shape)
        ## K means no change in matrix
        labels_tf= kmean_cluster(matrix,k)

        ## K modified terms matrix
        matrix_tf, matrix_tfidf= tfidf(matrix)
        # print("tfidf", matrix_tfidf.shape)
        labels_tf= kmean_cluster(matrix_tf,k)
        # print(np.unique(labels_tfidf))

        ## Columns are original levels, rows are K means clusters
        kmeans_tf= (index2matrix(labels_tf).T @ levels).round(decimals=2)

        print("simple term frequency matrix")
        print(kmeans_tf)
        barplt_pooled(kmeans_tf, k, "TF")
        # uv, xt = xtab(labels_tf, levels)
        # print(uv)
        # print(xt)

        print("term frequency idf matrix")
        labels_tfidf= kmean_cluster(matrix_tfidf,k)
        print("labels_tfidf",index2matrix(labels_tfidf).shape)
        kmeans_tfidf= (index2matrix(labels_tfidf).T @ levels).round(decimals=2)
        print(kmeans_tfidf)
        barplt_pooled(kmeans_tfidf, k, "TF-IDF")
        # uvidf, xtidf = xtab(labels_tfidf, levels)
        # print(uvidf)
        # print(xtidf)

        # Using vectorizer
        matrix = np.loadtxt('./neural_network_files/matrix.txt.gz')

        ## K means no change in matrix
        labels_vec= kmean_cluster(matrix,k)
        kmeans_vec= (index2matrix(labels_vec).T @ levels).round(decimals=2)
        print("Vectorized inputs")
        print(kmeans_vec)
        barplt_pooled(kmeans_vec, k, "BERT vector")



# Testing function
if __name__ == '__main__':
    main()
