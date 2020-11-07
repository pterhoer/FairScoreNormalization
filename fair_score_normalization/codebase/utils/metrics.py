# Author: Philipp Terhörst, Jan Niklas Kolf, 2019-2020

import keras.backend as K
import numpy as np


def gpu_pairwise_cosine(A):
    """ Computes the pairwise cosine similarity matrix
    """
    A = K.l2_normalize(A, axis=-1)
    A_tr = K.transpose(A)
    return K.dot(A, A_tr)

def gpu_calculate_csm(y_pred):
    """ Computes cosine similarity matrix for given data
        y_pred: data, each row contains one sample
    """
    # rename
    X_batch = y_pred
    size = X_batch.shape[0]

    # compute pairwise cosine matrix
    pair_cos = K.placeholder(shape=(size, size))
    pair_cos = K.eval(gpu_pairwise_cosine(X_batch))

    return pair_cos

def gpu_one_against_all(base_emb, comparison):
    """
        Computes the cosine similarity of the base_emb with each other embedding in
        comparison
        
        base_emb: 1D array with size of the embedding (facenet: 128)
        comparison: 2D array, each index in the first dimension points to an embedding
    """
    data_base = np.array([base_emb])

    data_comp = comparison
    
    size = data_comp.shape[0]
    
    pair_cos = K.placeholder(shape=(1, size))
    
    norm_data_comp = K.l2_normalize(data_comp, axis=-1)
    norm_data_base = K.l2_normalize(data_base, axis=-1)
    
    norm_data_comp = K.transpose(norm_data_comp)
    
    pair_cos = K.eval(K.dot(norm_data_base, norm_data_comp))

    K.clear_session()

    return pair_cos[0]

def gpu_gen_against_impgen(genuine_whole, comparison_data):
    K.clear_session()
    
    pair_cos = K.placeholder(shape=(genuine_whole.shape[0], comparison_data.shape[0]))

    norm_gen = K.l2_normalize(genuine_whole, axis=-1)
    
    norm_comp = K.l2_normalize(comparison_data, axis=-1)
    norm_comp = K.transpose(norm_comp)
    
    pair_cos = K.eval(K.dot(norm_gen, norm_comp))
    
    K.clear_session()
    
    return pair_cos