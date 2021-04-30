from scipy import stats
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

def jacknife_correlation(data):
    """ Function to calculate single-trial beta-series correlation using the jackknife approach.
    
        Parameters
        data: 2D numpy array, trial-number x region-number
              beta values of each trial
        
        Return
        JC: 3D numpy array, trial-number x region-number x region-number
            connectivity matrix of each single-trial, zscore-transformed across trials
    """
    trial_num, roi_num = data.shape
    JC = []
    for itrl in range(trial_num):
        idx = np.arange(trial_num) != itrl
        curr = data[idx]
        r = 1 - pairwise_distances(curr.T, metric='correlation')
        r = -r
        np.fill_diagonal(r, np.nan)
        JC.append(r)
    JC = np.stack(JC, axis=0)
    JC = 0.5 * np.log((1 + JC) / (1 - JC))
    
    JC = stats.zscore(JC, axis=0)
    return JC
