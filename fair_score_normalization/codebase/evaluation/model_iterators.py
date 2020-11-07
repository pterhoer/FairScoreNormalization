# Author: Jan Niklas Kolf, 2020
from codebase.clustering import GBSN

from pathlib import Path
from typing import List

"""
Each iterator should be added to the __init__.py file!
"""

class ExampleIterator:
    
    def __init__(self, embeddings, ks: List[int], multiplicator=None, prefix=None):
        """
        These Iterators are used for evaluating and plotting the models.
        It makes it easier to autmatically create model names for a given
        list of cluster amounts k.
        The style of models is the same:
            <dataset name>_<embedding type>_k<cluster amount>
        or if a multiplicator is used
            <dataset name>_<embedding type>_k<cluster amount>_<multiplicator>

        Parameters
        ----------
        embeddings : str
            embedding type used.
        ks : List[int]
            Cluster amounts per list.
        multiplicator : TYPE, optional
            DESCRIPTION. The default is None.
        prefix : str, optional
            Prefix for a model, e.g. if you run a dataset multiple
            times with different settings, a prefix can specify
            which run you want to evaluate. The default is None.

        """
        
        
        self._index = -1
        self._ks = ks
        self._emb = embeddings
        self._multiplicator = multiplicator
        self._prefix = prefix
    
    @property
    def embeddings(self):
        return self._emb
    
    @property
    def dataset(self):
        return "example"
    
    @property
    def formatted_prefix(self):
        if self._prefix is None:
            return ""
        
        return self._prefix+"_"
    
    def __iter__(self):
        return self
    
    def __next__(self):
        
        self._index += 1
        
        if self._index >= len(self._ks):
            raise StopIteration
        
        k = self._ks[self._index]
        
        if self._multiplicator is None:
            mdl = f"example_{self._emb}_k{k}"
        else:
            mdl = f"example_{self._emb}_k{k}_{self._multiplicator}"
            
        if not self._prefix is None:
            mdl = f"{self._prefix}_{mdl}"
            
        gbsn = GBSN(None, mdl)
        
        if len([f for f in Path(gbsn._model_location).iterdir() if f.is_dir()]) == 0:
            raise ValueError("No valid GBSN found!")
        
        return k, gbsn
