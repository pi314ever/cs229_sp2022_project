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

def pgperlevel(levels):
    return np.sum(levels, axis=0)

def wordsperlevel(matrix, levels):
    wordsperpage= np.sum(matrix, axis=1)
    print("wordsperpage", wordsperpage.shape)
    wordsperlevel= (levels.T * wordsperpage)
    print("wordsperlevel", wordsperlevel.shape, wordsperlevel[:,1])
    print("wordsperpage", wordsperpage[1,])
    print("levels", levels[1,:])
    wordsperpage= np.nanmean(np.where(wordsperlevel!=0,wordsperlevel,np.nan),1)

    return wordsperpage




def main():
    matrix, levels, _ = util.load_dataset()
    print(levels.shape)
    print(matrix.shape)
    levelcount=pgperlevel(levels)
    print(levelcount)
    # np.savetxt('pagesperlevel.txt', levelcount,fmt='%i')
    wrdpage= wordsperlevel(matrix, levels)


    matrix_pooled, levels_pooled, _ = util.load_dataset_pooled()
    print("pooled")
    print(levels_pooled.shape)
    print(matrix_pooled.shape)
    levelcount_p=pgperlevel(levels_pooled)
    print(levelcount_p)
    wrdpage_p= wordsperlevel(matrix_pooled, levels_pooled)
    print("pooled words per page", wrdpage_p)
    # np.savetxt('wordsperlevelpooled.txt', wrdpage_p,fmt='%i')



# Testing function
if __name__ == '__main__':
    main()
