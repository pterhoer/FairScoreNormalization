# Author: Jan Niklas Kolf, 2020

from codebase.utils import path_utils as pu

import numpy as np
import pandas as pd

from scipy import interp
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc

def cluster_global_eer(gbsn_iterable, num_folds:int=5):
    """
    Returns the calculated Equal-Error-Rates (EER) for
    the given GBSN models returned by the GBSN-Iterator
    

    Parameters
    ----------
    gbsn_iterable :
        An iterator object, returning
        k and the GBSN object for the given k
        and dataset/embedding.
    num_folds : int, optional
        Amount of folds. The default is 5.

    Returns
    -------
    Ks : TYPE
        The cluster size k for the returned
        EERs (normlized and unnormalized).
    EER_unnormalized : 1d array, List[floats]
        Unnormalized Equal-Error-Rates.
    EER_normalized : 2d array, List[floats]
        Normalized Equal-Error-Rates.

    """
    Ks = []
    EER_unnormalized = []
    EER_normalized = []
    
    for k, gbsn in gbsn_iterable:
        Ks.append(k)
        eer_nor = []
        eer_unn = []
        
        for fold in range(num_folds):
            res = pu.file_pickle_load(gbsn.path_test_results(fold, create=False))
            eer_nor.append(res["eer.normalized"])
            eer_unn.append(res["eer.unnormalized"])
            
        EER_unnormalized.append(eer_unn)
        EER_normalized.append(eer_nor)
        
    EER_unnormalized = np.array(EER_unnormalized).mean(axis=0)
    
    return Ks, EER_unnormalized, EER_normalized        


def cluster_subgroup_eer(gbsn_iterable, num_folds=5):
    
    Ks = []
    subgroups_unnormed = {}
    subgroups_normed = {}
    
    for k, gbsn in gbsn_iterable:
        
        Ks.append(k)
        
        results = [pu.file_pickle_load(gbsn.path_test_results(fold, create=False)) 
                   for fold in range(num_folds)]
        
        for subgroup in results[0]["subgroups"].keys():
            
            if not subgroup in subgroups_unnormed.keys():
                subgroups_unnormed[subgroup] = []

            subgroups_unnormed[subgroup].append(
            [
             results[i]["subgroups"][subgroup]["eer.unnormalized"] for i in range(num_folds)
            ])
                
            if not subgroup in subgroups_normed.keys():
                subgroups_normed[subgroup] = []

            subgroups_normed[subgroup].append(
            [
             results[i]["subgroups"][subgroup]["eer.normalized"] \
             for i in range(num_folds)
            ])

    return Ks, subgroups_unnormed, subgroups_normed

def extract_subgroup_information(gbsn_iterable, subgroup_id, num_folds=5):
    
    info = {}

    for k, gbsn in gbsn_iterable:
        
        info[k] = {}
        
        data = [pu.file_pickle_load(gbsn.path_test_results(fold, create=False))["subgroups"]
                for fold in range(num_folds)]
        
        info[k]["amount.ids"] = [d[subgroup_id]["amount.ids"] for d in data]
        info[k]["amount.samples"] = [d[subgroup_id]["amount.samples"] for d in data]
        info[k]["eer.normalized"] = [d[subgroup_id]["eer.normalized"] for d in data]
        info[k]["eer.unnormalized"] = [d[subgroup_id]["eer.unnormalized"] for d in data]
    
    return info

def extract_valid_clusters(gbsn_iterable, num_folds=5):
    
    cluster_info = {}
    
    for k, gbsn in gbsn_iterable:
        
        data = [np.load(gbsn.path_cluster_thresholds(fold))
                for fold in range(num_folds)]
        
        cluster_info[k] = [np.sum(np.logical_not(np.isnan(data[f]))) 
                           for f in range(num_folds)]
    
    return cluster_info

def create_dataframe(Ks, unnormed_data, normed_data):
    
    X = []
    X_ed = []
    Y = []
    CAT = []
    
    for idx in range(len(Ks)):
        
        X.extend([Ks[idx]] * 10)
        Y.extend(unnormed_data)
        CAT.extend(["Unnormalized"] * 5)
        Y.extend(normed_data[idx])
        CAT.extend(["Normalized"] * 5)
        
        X_ed.extend(np.full(10, idx))

    return pd.DataFrame({"x":X, "EER":Y, "":CAT, "x_ed":X_ed})

def calculate_fnmr(gbsn_iterable, fmr=0.001, num_folds=5, recalculate=False):
    
    path = f"data/fnmr_calc/fnmr_precalc_{gbsn_iterable.dataset}_{gbsn_iterable.embeddings}.pkl"
    
    if pu.file_test(path) and not recalculate:
        return pu.file_pickle_load(path)
    
    results = {}
    results["fmr"] = fmr
    
    added = False
    for k, gbsn in gbsn_iterable:
        
        subgroups = list(pu.file_pickle_load(gbsn.path_test_results(0, create=False))["subgroups"].keys())
        subgroups.append("global")
        
        if not added:
            for subgroup in subgroups:    
                results[subgroup] = {}
            added = True
            
        
        for subgroup in subgroups:
            
            results[subgroup][k] = {}
            
            results[subgroup][k]["normalized"] = []
            results[subgroup][k]["unnormalized"] = []
            
            for fold in range(num_folds):
                
                far_unnormed = gbsn.path_test_files("far", fold, subgroup, normalized=False, create=False)
                tar_unnormed = gbsn.path_test_files("tar", fold, subgroup, normalized=False, create=False)
                
                far_normed = gbsn.path_test_files("far", fold, subgroup, normalized=True, create=False)
                tar_normed = gbsn.path_test_files("tar", fold, subgroup, normalized=True, create=False)
                
                if not (pu.file_test(far_unnormed) and pu.file_test(far_normed)
                        and pu.file_test(tar_unnormed) and pu.file_test(tar_normed)):
                    continue
                
                far_un = np.load(far_unnormed)
                tar_un = np.load(tar_unnormed)
                
                fnmr_index = np.argmin(np.abs(far_un - fmr))
                fnmr_un = 1 - tar_un[fnmr_index]
                
                results[subgroup][k]["unnormalized"].append(fnmr_un)
                
                far_nm = np.load(far_normed)
                tar_nm = np.load(tar_normed)
                
                fnmr_index = np.argmin(np.abs(far_nm - fmr))
                fnmr_nm = 1 - tar_nm[fnmr_index]
                
                results[subgroup][k]["normalized"].append(fnmr_nm)
    
    pu.file_pickle_save(path, results)        
    
    return results


def calculate_mean_roc(gbsn_iterable, num_folds=5, precision=100000, recalculate=False):
    
    file_path = f"data/roc_interpolation/{gbsn_iterable.formatted_prefix}{gbsn_iterable.dataset}_{gbsn_iterable.embeddings}.pkl"
    
    if pu.file_test(file_path) and not recalculate:
        return pu.file_pickle_load(file_path)
    
    results = {}
    
    added = False
    for k, gbsn in gbsn_iterable:
        
        subgroups = list(pu.file_pickle_load(gbsn.path_test_results(0, create=False))["subgroups"].keys())
        subgroups.append("global")
        
        if not added:
            for subgroup in subgroups:    
                results[subgroup] = {}
            added = True
            
        
        for subgroup in subgroups:
            
            results[subgroup][k] = {}
            
            results[subgroup][k]["normalized"] = {}
            results[subgroup][k]["unnormalized"] = {}
            
            tars_normed_all = []
            aucs_normed_all = []
            mean_far_normed = np.linspace(0, 1, precision)
            
            tars_unnormed_all = []
            aucs_unnormed_all = []
            mean_far_unnormed = np.linspace(0, 1, precision)
            
            for fold in range(num_folds):
                
                far_unnormed = gbsn.path_test_files("far", fold, subgroup, normalized=False, create=False)
                tar_unnormed = gbsn.path_test_files("tar", fold, subgroup, normalized=False, create=False)
                
                far_normed = gbsn.path_test_files("far", fold, subgroup, normalized=True, create=False)
                tar_normed = gbsn.path_test_files("tar", fold, subgroup, normalized=True, create=False)
                
                if not (pu.file_test(far_unnormed) and pu.file_test(far_normed)
                        and pu.file_test(tar_unnormed) and pu.file_test(tar_normed)):
                    continue

                # Unnormed                
                far_un = np.load(far_unnormed)
                tar_un = np.load(tar_unnormed)
                
                tars_unnormed_all.append(interp(mean_far_unnormed, far_un, tar_un))
                tars_unnormed_all[-1][0] = 0.0
            
                aucs_unnormed_all.append(auc(far_un, tar_un))


                # Normed
                far_nm = np.load(far_normed)
                tar_nm = np.load(tar_normed)
                
                tars_normed_all.append(interp(mean_far_normed, far_nm, tar_nm))
                tars_normed_all[-1][0] = 0.0
                
                aucs_normed_all.append(auc(far_nm, tar_nm))
                
            # Unnormed
            mean_tar_unnormed = np.mean(tars_unnormed_all, axis=0)
            mean_tar_unnormed[-1] = 1.0
            mean_auc_unnormed = auc(mean_far_unnormed, mean_tar_unnormed)

            results[subgroup][k]["unnormalized"]["mean_tar"] = mean_tar_unnormed
            results[subgroup][k]["unnormalized"]["mean_far"] = mean_far_unnormed
            results[subgroup][k]["unnormalized"]["mean_auc"] = mean_auc_unnormed

            results[subgroup][k]["unnormalized"]["std_tar"] = np.std(tars_unnormed_all, axis=0)
            results[subgroup][k]["unnormalized"]["std_auc"] = np.std(aucs_unnormed_all)

            # Normed
            mean_tar_normed = np.mean(tars_normed_all, axis=0)
            mean_tar_normed[-1] = 1.0
            mean_auc_normed = auc(mean_far_normed, mean_tar_normed)
            
            results[subgroup][k]["normalized"]["mean_tar"] = mean_tar_normed
            results[subgroup][k]["normalized"]["mean_far"] = mean_far_normed
            results[subgroup][k]["normalized"]["mean_auc"] = mean_auc_normed
            
            results[subgroup][k]["normalized"]["std_tar"] = np.std(tars_normed_all, axis=0)
            results[subgroup][k]["normalized"]["std_auc"] = np.std(aucs_normed_all)
            
    pu.file_pickle_save(file_path, results)
    
    return results
    
    
def calculate_tar_at_far(gbsn_iterator, far_location, num_folds=5, precision=100000, recalculate=True):
    
    roc_data = calculate_mean_roc(gbsn_iterator, recalculate=recalculate)
    
    results = {}
    
    for subgroup in roc_data.keys():
        
        if subgroup.startswith("__"):
            continue
        
        results["k"] = list(roc_data[subgroup].keys())
        
        results[subgroup] = {}
        results[subgroup]["normalized"] = []
        results[subgroup]["unnormalized"] = []
        
        sg_data = roc_data[subgroup]
        
        for k in sg_data.keys():
            
            for key in ["normalized", "unnormalized"]:
                
                mean_tar = sg_data[k][key]["mean_tar"]
                mean_far = sg_data[k][key]["mean_far"]
                
                min_loc = np.argmin(np.abs(mean_far - far_location))
                
                results[subgroup][key].append(mean_tar[min_loc])
                
    return results
   
            
