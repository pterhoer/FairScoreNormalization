# Author: Jan Niklas Kolf, 2020

# own imports
from codebase.datasets import Dataset

# Foreign imports
from abc import ABC, abstractmethod

import sklearn.cluster as skcluster
import numpy as np

class ClusteringMethod(ABC):
    def __init__(self, param):
        self._param = param
        self._obj = None
        self._existing_clusters = None

    @abstractmethod
    def fit(self, dataset : Dataset):
        pass
    
    @abstractmethod
    def predict_cluster(self, data):
        pass
    
    @abstractmethod
    def transform(self, data):
        pass
    
    @property
    def model(self):
        return self._obj
    
    @property
    @abstractmethod
    def cluster_amount(self):
        pass
    
    @property
    @abstractmethod
    def method_name(self):
        pass
    
    def reset(self):
        self._obj = None
        self._existing_clusters = None

    def _validate_cluster(self,
                          cluster_assignment,
                          dataset : Dataset,
                          max_k : int,
                          min_ids :int = 5,
                          min_samples :int = 20):
        
        violations = {}
        
        # Penalty function
        def penalty(k):
            
            IDs, counts = np.unique(dataset.ids[cluster_assignment == k], return_counts=True)
            
            P = 0
            try:
                P1 = int(np.expm1(min_samples - np.sum(counts)))
                P2 = int(np.expm1(min_ids - IDs.shape[0]))
                P3 = int(np.expm1(np.sum(counts < 2)))
    
                P = P1+P2+P3
            except:
                P = np.inf

            if P > 0:
                violations[k] = {}
                violations[k]["ids"] = IDs
                violations[k]["counts"] = counts
                
                violations[k]["sample_count"] = np.sum(counts)
                violations[k]["ids_violating"] = np.sum(counts < 2)
            
            return P
            
        # Calculate penalty for each cluster
        penalties = np.empty((max_k, 2), dtype=float)
        for k in range(max_k):
            penalties[k, 0] = k
            penalties[k, 1] = penalty(k)
        
        # Sort the penalties by their values (descending)
        penalties = penalties[penalties[:,1].argsort()[::-1]]
        
        # Create an array which stores if a cluster is still valid
        # or if it was removed
        existing_clusters = np.ones((max_k, ), dtype=bool)
        k_range = np.arange(max_k)
        # We iterate over the penalties
        for k, initial_p in penalties:
            k = int(k)

            # If the initial p was below or equal to zero, the cluster
            # is a valid cluster. Because we sorted descending, only
            # valid cluster will follow. -> break           
            if initial_p <= 0:
                break
            # The reclustering of previous cluster could have resulted
            # in making in a following invalid cluster valid. Therefore
            # we have to check if cluster k became valid after calculating
            # the initial penalty. -> continue, because other previously
            # invalid clusters may be still invalid
            if penalty(k) <= 0:
                continue
            
            # Get the data for this cluster
            k_IDs = violations[k]["ids"]
            k_counts  = violations[k]["counts"]
            
            k_samplesize = violations[k]["sample_count"]
            k_ids_violating = violations[k]["ids_violating"]
            
            # If we only have to remove some IDs (because of too few samples)
            # we just remove those IDs
            if k_samplesize - k_ids_violating >= min_samples and \
               k_IDs.shape[0] - k_ids_violating >= min_ids:
                remove_ids = k_IDs[k_counts < 2]
            else: # Too many violations, remove cluster
                remove_ids = k_IDs
                # Set cluster to not existent
                existing_clusters[k] = False
            
            # We can not use existing_clusters array because it is not always set to false.
            nearest_cluster_mask = np.empty(existing_clusters.shape, dtype=bool)
            nearest_cluster_mask[:] = existing_clusters[:]
            nearest_cluster_mask[k] = False
            
            # Iterate over each ID which needs to be removed
            for ID in remove_ids:
                # Get the samples for this ID in this cluster
                sample_mask = np.logical_and(dataset.ids == ID,
                                             cluster_assignment == k)
                # Get the indices for the dataset array
                sample_indices = np.argwhere(sample_mask == True).squeeze(axis=1)
                # Get the nearest clusters for these samples
                tf = self.transform(dataset.embeddings[sample_indices])
                # Get the nearest valid cluster indices
                nearest_clusters = np.argmin(tf[:, nearest_cluster_mask], axis=1)
                # Cluster index to cluster ID
                cluster_choice = k_range[nearest_cluster_mask][nearest_clusters]
                # Count if some new clusters are present multiple times
                cluster_ids, cluster_count = np.unique(cluster_choice, return_counts=True)
                
                # Iterate over clusters which are the nearest cluster for multiple
                # samples. Because in this case we do not have to check for samples
                # of this ID in other clusters
                iter_stop = len(cluster_ids)
                for cluster_id in cluster_ids[cluster_count >= 2]:
                    cluster_mask = sample_indices[cluster_choice == cluster_id]
                    # Place these samples to the new cluster
                    cluster_assignment[cluster_mask] = cluster_id
                    iter_stop -= 1
                    
                # There was no ID with only one sample,
                # so just continue
                if iter_stop <= 0:
                    continue
        
                # There are some samples which could not be sorted, so check where we
                # can put these samples. Because we do not want to create invalid clusters,
                # we have to place these samples in the nearest cluster which already has
                # at least one sample from this ID. Here we create masks to check for this.
                clusters_with_id = np.unique(cluster_assignment[dataset.ids == ID])
                clusters_with_id = clusters_with_id[clusters_with_id != k]
                
                clusters_with_id_mask = np.zeros((max_k,), dtype=bool)
                clusters_with_id_mask[clusters_with_id] = True
            
                # Iterate over choices/samples which could 
                for cluster_id in cluster_ids[cluster_count < 2]:
                    cid_mask = cluster_choice == cluster_id
                    sample_index = sample_indices[cid_mask]
                    
                    # There may be no other clusters with a sample of this ID, so we
                    # take the first sample and put them to the nearest neighbor. But only if
                    # there are at least 2 samples of this ID. Then the other sample will be placed
                    # automatically in the same cluster with the first sample.
                    # This is not that bad because all the samples of this ID are placed
                    # together in the current cluster, so they at least belong together.
                    if len(clusters_with_id) == 0 and len(cluster_ids[cluster_count < 2]) > 1:
                        cluster_idx = np.argmin(tf[cid_mask, nearest_cluster_mask])
                        nearest_for_id = k_range[nearest_cluster_mask][cluster_idx]
                        cluster_assignment[sample_index] = nearest_for_id
                        clusters_with_id = np.array([nearest_for_id])
                        clusters_with_id_mask[nearest_for_id] = True
                    # We can place a sample in another cluster
                    elif not cluster_id in clusters_with_id:
        
                        valid_for_id = np.logical_and(nearest_cluster_mask,
                                                      clusters_with_id_mask)
                        
                        
                        cluster_idx = np.argmin(tf[cid_mask, valid_for_id])
                        nearest_for_id = k_range[valid_for_id][cluster_idx]
                        
                        
                        cluster_assignment[sample_index] = nearest_for_id
                    # The chosen cluster is a cluster which already has a sample
                    # of this id                        
                    else:
                        cluster_assignment[sample_index] = cluster_id
                                    
        self._existing_clusters = existing_clusters                           
        return cluster_assignment, existing_clusters                
                        
                     
                        
                    
                    
                    

class KMeansClustering(ClusteringMethod):
    
    def __init__(self, k, set_kwargs=None):
        super().__init__(k)
        self.k = k
        self._set_kwargs = set_kwargs
        
    @property
    def cluster_amount(self):
        return self.k
    
    @property
    def method_name(self):
        return f"kMeans_{self.cluster_amount}"
    
    def fit(self, dataset : Dataset):
        if self._set_kwargs is None:
            self._obj = skcluster.KMeans(n_clusters=self.k,
                                       random_state=0).fit(dataset.embeddings)
        else:            
            self._obj = skcluster.KMeans(n_clusters=self.k,
                                       **self._set_kwargs).fit(dataset.embeddings)
            
            
        cluster_assignment = self.predict_cluster(dataset.embeddings)
        cluster_assignment, self._existing_clusters = self._validate_cluster(cluster_assignment, 
                                                                       dataset, 
                                                                       self.cluster_amount)
        
        return cluster_assignment, self._existing_clusters
        
    def predict_cluster(self, data):
        if self._obj is None:
            return ValueError("KMeans need to be fitted before it can predict the cluster.")
        # If all clusters are valid we can just predict the data        
        if self._existing_clusters is None or \
           np.sum(self._existing_clusters) == len(self._existing_clusters):
            return self.model.predict(data)
        # If we encountered several invalid clusters while doing fit(), we need to
        # make sure that no sample was clustered into an invalid cluster
        transformations = self.transform(data)   
        # Just set the distance to invalid clusters to infinite
        transformations[:, np.logical_not(self._existing_clusters)] = np.inf
        # Take argmin
        return np.argmin(transformations, axis=1)
        
    
    def transform(self, data):
        if self._obj is None:
            return ValueError("KMeans need to be fitted before it can transform the data.")
        
        return self.model.transform(data)