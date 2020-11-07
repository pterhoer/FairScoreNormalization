# Author: Jan Niklas Kolf, 2020

import numpy as np

def eer_threshold(far, tar, thresholds):
    values = np.abs(1 - tar - far)
    eer_idx = np.argmin(values)
    return thresholds[eer_idx]    

def fnmr_threshold(far, tar, thresholds):
    fnmr_idx = np.argmin(np.abs(far - 0.001))
    return thresholds[fnmr_idx]

def fnmr_threshold_10powm4(far, tar, thresholds):
    fnmr_idx = np.argmin(np.abs(far - 0.0001))
    return thresholds[fnmr_idx]