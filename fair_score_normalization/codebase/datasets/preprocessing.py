# Author: Jan Niklas Kolf, 2019-2020
from codebase.datasets import Dataset
from codebase.utils import path_utils as pu

# Foreign imports
import numpy as np

def generate_subject_exclusive_folds(IDs:np.ndarray, num_folds:int=5):
    """
    Generates num_folds amount of folds for cross validation.
    The IDs per fold are randomly selected through permutation.
    

    Parameters
    ----------
    IDs : np.ndarray, 1d of shape (<len>,)
        The numpy array specifying which sample in the dataset has which id.
    num_folds : int, optional
        Amount of folds to be created. The default is 5.

    Raises
    ------
    ValueError
        If the pairwise comparison between folds has samples in both folds,
        a ValueError exception is raised. Due to the fact, that each ID
        is only put in one array, this is not the case.
        For integrity reasons the check is always performed.

    Returns
    -------
    cv_mask : np.ndarray, shape (num_folds, <len IDs>)
        A bool mask specifying which fold contains which sample.
        This mask can be easily applied to the dataset.

    """    
    
    # Creates an empty mask of type bools to be used as sample selector
    cv_mask = np.zeros((num_folds, len(IDs)), dtype=bool)
    
    unique_IDs, unique_counts = np.unique(IDs, return_counts=True)
    
    capacity = np.zeros(num_folds)
       
    # Iterate randomly over unique IDs and select the smallest fold
    # where the ID should be put into
    for idx in np.random.permutation(unique_IDs.shape[0]):
        
        ID = unique_IDs[idx]
        count = unique_counts[idx]
        
        smallest_fold = np.argmin(capacity)
        
        capacity[smallest_fold] += count
        
        # Apply logical or operation to the array to put the ID into the fold
        cv_mask[smallest_fold] = np.logical_or(cv_mask[smallest_fold], IDs == ID)
        
    # Valid check with pairwise comparison of the folds
    from itertools import combinations
    
    for x,y in combinations(list(range(num_folds)), 2):
        if np.sum(np.logical_and(cv_mask[x], cv_mask[y])) > 0:
            raise ValueError("Folds are not unique!")
    
        
    return cv_mask

def get_fold_distribution(constructor : Dataset, embedding_type, num_folds: int):
    """
    Extracts the distribution per subgroup of the dataset
    for each fold.
    Returns dict with subgroup key and percentage / sample amount

    Parameters
    ----------
    constructor : Dataset
        The constructor for the dataset to be inspected.

    Returns
    -------
    dict.
    The amount of samples per subgroup in percentage (key: "%")
    and the raw amount ("#")

    """
    
    distribution = {}
    
    folds = [constructor(embedding_type, [i]) for i in range(num_folds)]
    
    distribution["__subgroups__"] = folds[0].subgroups
    
    subgroups = folds[0].subgroups
    for subgroup in subgroups.keys():
        
        if subgroup.startswith("__"):
            continue
                
        masks = [np.ones(len(folds[i]), dtype=bool) for i in range(num_folds)]
        
        for feature in subgroups[subgroup].keys():
            
            if feature.startswith("__"):
                continue
            
            feature_masks = [np.zeros(len(folds[i]), dtype=bool)
                             for i in range(num_folds)]
            
            for feature_class in subgroups[subgroup][feature]:
                feature_masks = [np.logical_or(
                                    feature_masks[i],
                                    folds[i].features._asdict()[feature] == feature_class
                                 )
                                for i in range(num_folds)
                                ]
                
                
            masks = [np.logical_and(masks[i], feature_masks[i]) for i in range(num_folds)]

        amount_ids = [len(np.unique(folds[i].ids[masks[i]])) for i in range(num_folds)]
        overall_amount_ids = [len(np.unique(folds[i].ids)) for i in range(num_folds)]

        lengths = [len(masks[i]) for i in range(num_folds)]
        sums = [np.sum(masks[i]) for i in range(num_folds)]
                       
        distribution[subgroup] = {}
        
        distribution[subgroup]["samples"] = {}
        distribution[subgroup]["samples"]["%"] = [(sums[i]/lengths[i]) * 100.0 for i in range(num_folds)]
        distribution[subgroup]["samples"]["#"] = [sums[i] for i in range(num_folds)]
        
        distribution[subgroup]["ids"] =  {}
        distribution[subgroup]["ids"]["%"] = [(amount_ids[i]/overall_amount_ids[i])*100.0 
                                              for i in range(num_folds)]
        distribution[subgroup]["ids"]["#"] = [amount_ids[i] for i in range(num_folds)]
        
        
    pu.file_pickle_save(f"data/subgroup_distributions/{folds[0].dataset_name}_{embedding_type}.pkl",
                        distribution)

    return distribution