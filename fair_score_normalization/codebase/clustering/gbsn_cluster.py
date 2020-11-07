# Author: Jan Niklas Kolf, 2020

# Own imports
from codebase.clustering.methods import ClusteringMethod
from codebase.clustering.threshold_functions import fnmr_threshold
from codebase.datasets import Dataset, CVDataset
from codebase.utils import path_utils as pu

# Foreign imports
from sklearn.metrics import roc_curve
from tqdm import tqdm
import numpy as np
import sys


class GBSN:
    
    def __init__(self,
                 method:ClusteringMethod,
                 model_name:str = "GBSN",
                 save_location:str ="./data/models/",
                 threshold_method = fnmr_threshold,
                 gbsn_type="gbsn"
                 ):

        self._method = method
        self._model_name = model_name
        self._model_location = f"{save_location}{model_name}.{gbsn_type}/"
        pu.make_path(self._model_location)
        
        self._threshold_method = threshold_method
        
    @property
    def clustering_method(self):
        return self._method
    
    @property
    def model_location(self):
        return self._model_location
    
    def train(self, cv_dataset : CVDataset):
        
        dataset = cv_dataset.train_dataset
        
        self.clustering_method.reset()
        cluster_assignment, existing_clusters = self.clustering_method.fit(dataset)
        
        """
            Calculate Threshold per Cluster
        """
        cluster_results = np.empty((self.clustering_method.cluster_amount,))
        for k in tqdm(
                        range(self.clustering_method.cluster_amount),
                        unit="bin",
                        disable=False,
                        desc="Calculate Cluster",
                        position=1,
                        file=sys.stdout,
                        leave=False):    
            # Cluster was removed due to reclustering
            if not existing_clusters[k]:
                cluster_results[k] = np.nan
                continue
            
            cluster_mask = cluster_assignment == k
            gen_imp_labels, scores, row_indices = dataset.generate_scores(cluster_mask,
                                                                          bin_size=200)
            
            # Skip the calculation if the cluster is empty or it does not contain any
            # imposter/genuine pairs.
            if len(gen_imp_labels) == 0 or len(np.unique(gen_imp_labels)) != 2:
                raise ValueError(f"Invalid scores generated. Dataset {dataset.dataset_name}, k={k}")

            far, tar, thresholds = roc_curve(gen_imp_labels, scores)

            cluster_results[k] = self._threshold_method(far, tar, thresholds)
            
            del far, tar, thresholds
            
        del k

        """
            Calculate Global-Threshold
        """             
        gen_imp_labels, scores, row_indices = dataset.generate_scores(bin_size=200)
        far, tar, thresholds = roc_curve(gen_imp_labels, scores)

        global_thr = self._threshold_method(far, tar, thresholds)
        
        # Saving all 3 files
        pu.file_np_save(self.path_cluster_assignment(
                        cv_dataset.current_test_fold, testing=False
                        ), cluster_assignment)

        pu.file_np_save(self.path_cluster_thresholds(
                        cv_dataset.current_test_fold
                        ), cluster_results)
        
        pu.file_np_save(self.path_global_threshold(
                        cv_dataset.current_test_fold
                        ), global_thr)

        return global_thr, cluster_results
    
            
    def test(self,
             cv_dataset : CVDataset,
             global_threshold : float,
             cluster_thresholds : np.array,
             multiplicator=0.5):
        
        self.multiplicator = multiplicator
        dataset = cv_dataset.test_dataset
        
        cluster_assignment = self.clustering_method\
                             .predict_cluster(dataset.embeddings)
                             
                             
        results = {}
        results["multiplicator"] = multiplicator 
        
        gen_imp_labels, scores, row_indices = dataset.generate_scores(
                                              bin_size=50, imposter_count=2000
                                              )
        
        eer, fnmr = self._calculate_eer(
                                        gen_imp_labels, 
                                        scores, 
                                        test_fold=cv_dataset.current_test_fold,
                                        group="global"
                                      )   
         
        results["eer.unnormalized"] = eer
        results["fnmr.unnormalized"] = fnmr
        
        
        eer, fnmr = self._calculate_eer_normalized(
                                        gen_imp_labels, 
                                        scores,
                                        row_indices,
                                        cluster_assignment,
                                        cluster_thresholds,
                                        global_threshold,
                                        test_fold=cv_dataset.current_test_fold,
                                        group="global"
                                     )  
        
        results["eer.normalized"] = eer
        results["fnmr.normalized"] = fnmr

        del gen_imp_labels, scores, row_indices

        results["subgroups"] = {}

        subgroups = dataset.subgroups
        # We iterate over the subgroups dictionary
        for subgroup in tqdm(
                                subgroups.keys(),
                                unit="subgroup",
                                disable=False,
                                desc="Calculate Subgroups",
                                position=1,
                                file=sys.stdout,
                                leave=False
                         ):
            # Skip comment keys
            if subgroup.startswith("__"):
                continue
            
            # Mask is build
            mask = np.ones(len(dataset), dtype=bool)            
            for feature in subgroups[subgroup].keys():
                if feature.startswith("__"):
                    continue
                
                feature_mask = np.zeros(len(dataset), dtype=bool)                
                ft_data = dataset.features._asdict()[feature]                
                for ft_class in subgroups[subgroup][feature]:
                    feature_mask = np.logical_or(feature_mask, 
                                                 ft_data == ft_class
                                                 )
                mask = np.logical_and(mask, feature_mask)
                
            gen_imp_labels, scores, row_indices = dataset.generate_scores(
                                                    mask_remaining_samples=mask
                                                    )            
            res = {}
            ids_subgroup = dataset.ids[mask]
            
            res["amount.samples"] = len(ids_subgroup)
            res["amount.ids"] = len(np.unique(ids_subgroup))
            
            
            if len(gen_imp_labels) == 0 or len(np.unique(gen_imp_labels)) != 2:
                res["eer.unnormalized"] = np.nan
                res["fnmr.unnormalized"] = np.nan
                res["eer.normalized"] = np.nan
                res["fnmr.normalized"] = np.nan
                results["subgroups"][subgroup] = res
                continue
            
            
            eer, fnmr = self._calculate_eer(gen_imp_labels, 
                                               scores,
                                               test_fold=cv_dataset.current_test_fold,
                                               group=subgroup)

            res["eer.unnormalized"] = eer
            res["fnmr.unnormalized"] = fnmr
    
            eer, fnmr = self._calculate_eer_normalized(
                                                    gen_imp_labels, 
                                                    scores,
                                                    row_indices,
                                                    cluster_assignment,
                                                    cluster_thresholds,
                                                    global_threshold,
                                                    test_fold=cv_dataset.current_test_fold,
                                                    group=subgroup)
            
            res["eer.normalized"] = eer
            res["fnmr.normalized"] = fnmr
            
            results["subgroups"][subgroup] = res
            
            del gen_imp_labels, scores, row_indices
            
        pu.file_pickle_save(self.path_test_results(cv_dataset.current_test_fold), results)
        
        pu.file_np_save(self.path_cluster_assignment(
                        cv_dataset.current_test_fold, testing=True
                        ), cluster_assignment)
        
        return results
        
        
    def _calculate_eer(self, gen_imp_labels, scores, test_fold:int=None, group:str="global"):
        """
        Calculates the Equal-Error-Rate and False-None-Match-Rate for given
        genuine/imposter labels and scores.

        Parameters
        ----------
        gen_imp_labels : array, 1d
            Array labeling if a score is a genuine comparison
            or an imposter comparison.
            1: Genuine Comparison
            0: Imposter Comparison
        scores : array, 1d
            Comparison scores between two samples.

        Returns
        -------
        eer, float
            Equal-Error-Rate.
        fnmr, dictionary
            False-Non-Match-Rate

        """
        far, tar, thresholds = roc_curve(gen_imp_labels, scores)
        values = np.abs(1 - tar - far)
        eer_idx = np.argmin(values)

        eer = far[eer_idx]
        
        fnmr = {}
        fnmr["10e1"] = 1 - tar[np.argmin(np.abs(far - 0.1))]
        fnmr["10e2"] = 1 - tar[np.argmin(np.abs(far - 0.01))]
        fnmr["10e3"] = 1 - tar[np.argmin(np.abs(far - 0.001))]
        fnmr["10e4"] = 1 - tar[np.argmin(np.abs(far - 0.0001))]
        fnmr["10e5"] = 1 - tar[np.argmin(np.abs(far - 0.00001))]
        fnmr["10e6"] = 1 - tar[np.argmin(np.abs(far - 0.000001))]
        
        if not test_fold is None:
            
            pu.file_np_save(self.path_test_files(
                    "far", test_fold, group, normalized=False
                    ), far)        
            pu.file_np_save(self.path_test_files(
                    "tar", test_fold, group, normalized=False
                    ), tar)
            pu.file_np_save(self.path_test_files(
                    "thr", test_fold, group, normalized=False
                    ), thresholds)
            pu.file_np_save(self.path_test_files(
                    "eer", test_fold, group, normalized=False
                    ), eer)

        return eer, fnmr
    
    def _calculate_eer_normalized(self,
                                  gen_imp_labels, 
                                  scores,
                                  row_indices,
                                  cluster_assignment,
                                  cluster_thresholds,
                                  global_threshold,
                                  test_fold:int=None,
                                  group:str="global"):

        clusters_left = cluster_assignment[row_indices[:,0]]
        scores_left = cluster_thresholds[clusters_left] - global_threshold
        del clusters_left

        
        clusters_right = cluster_assignment[row_indices[:,1]]
        scores_right = cluster_thresholds[clusters_right] - global_threshold
        del clusters_right


        # Scores are now the normalized scores
        scores_normed = scores - np.multiply(self.multiplicator, scores_left + scores_right)
        
        del scores_left, scores_right

        if np.sum(np.isnan(scores_normed)) > 0:
            raise ValueError("NaN Value for Score detected.")
        
        far, tar, thresholds = roc_curve(gen_imp_labels, scores_normed)
        values = np.abs(1 - tar - far)
        eer_idx = np.argmin(values)
        
        eer_normalized = far[eer_idx]
        
        fnmr = {}
        fnmr["10e1"] = 1 - tar[np.argmin(np.abs(far - 0.1))]
        fnmr["10e2"] = 1 - tar[np.argmin(np.abs(far - 0.01))]
        fnmr["10e3"] = 1 - tar[np.argmin(np.abs(far - 0.001))]
        fnmr["10e4"] = 1 - tar[np.argmin(np.abs(far - 0.0001))]
        fnmr["10e5"] = 1 - tar[np.argmin(np.abs(far - 0.00001))]
        fnmr["10e6"] = 1 - tar[np.argmin(np.abs(far - 0.000001))]
        
        if not test_fold is None:
            
            pu.file_np_save(self.path_test_files(
                    "far", test_fold, group, normalized=True
                    ), far)        
            pu.file_np_save(self.path_test_files(
                    "tar", test_fold, group, normalized=True
                    ), tar)
            pu.file_np_save(self.path_test_files(
                    "thr", test_fold, group, normalized=True
                    ), thresholds)
            pu.file_np_save(self.path_test_files(
                    "eer", test_fold, group, normalized=True
                    ), eer_normalized)

        return eer_normalized, fnmr
    
    
    def path_cluster_assignment(self, test_fold:int, testing=True, create=True):
        ext = "testing" if testing else "training"
        p = f"{self._model_location}test_fold={test_fold}/{ext}/"
        
        if create:
            pu.make_path(p)
            
        return f"{p}{ext}_cluster_assignment.npy"
    
    def path_cluster_thresholds(self, test_fold:int, create=True):
        p = f"{self._model_location}test_fold={test_fold}/training/"

        if create:
            pu.make_path(p)
            
        return f"{p}training_cluster_thresholds.npy"
    
    def path_global_threshold(self, test_fold:int, create=True):
        p = f"{self._model_location}test_fold={test_fold}/training/"
        
        if create:
            pu.make_path(p)
            
        return f"{p}training_global_threshold.npy"
    
    def path_test_files(self, file_type:str, test_fold:int, group:str, normalized:bool=False, create=True):
        
        normed = "normalized" if normalized else "unnormalized"
        
        p = f"{self._model_location}test_fold={test_fold}/testing/{group}/"
        
        if create:
            pu.make_path(p)
        
        return f"{p}testing_{group}_{normed}_{file_type}.npy"
    
    def path_test_results(self, test_fold:int, create=True):
        p = f"{self._model_location}test_fold={test_fold}/testing/"
        
        if create:
            pu.make_path(p)
            
        return f"{p}testing_results.pkl"