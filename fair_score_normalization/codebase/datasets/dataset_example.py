# Author: Jan Niklas Kolf, 2020

# Own imports
from codebase.utils import path_utils as pu
from codebase.datasets import preprocessing as pp
from codebase.datasets import Dataset


# Foreign imports
from typing import List
from collections import namedtuple

from pathlib import Path

import numpy as np

import json

class ExampleDataset(Dataset):

    """
    The created dataset needs to be added to the __init__.py file:
    codebase.datasets.__init__.py
    
    """
    
    def __init__(self,
                 embedding_type,
                 folds:List[int],
                 data_path:str="./path/to/where/you/store/your/data/"):
        # Call the super class
        super().__init__(embedding_type, folds, data_path)

        # Just for this example we create a cross-validation mask
        # Normally, this mask would be created with the static method
        # generate_cv_mask
        # and then saved to disk and then simply loaded here via numpy
        basic_cv_mask = np.zeros((5, # pretend we have 5 folds
                                  50), # pretend we have 50 samples
                                  dtype=bool)
        
        # Set the sample mask to true for a given fold
        for f in range(5):
            basic_cv_mask[f, f*10:(f+1)*10] = True
        

        # Create a bool mask with size of amount of samples
        cv_mask = np.zeros((basic_cv_mask.shape[1],), dtype=bool)

        # folds is a list containing the fold number as index
        # If we have 5 folds, the index will be in [0, 4].
        # If this dataset should use folds 0-3 e.g. for training
        # then folds = [0,1,2,3]
        for f in folds:
            # Combine the mask to one whole mask, covering multiple folds
            cv_mask = np.logical_or(cv_mask, basic_cv_mask[f])
            

        # Create 50 random embeddings with size 128 (just like facenet)
        self._embeddings    = np.random.randn(50, 128)[cv_mask]

        # Only one sample per id
        self._ids           = np.arange(50)[cv_mask]

        self._filenames     = None # Path to image per sample

        # Create 50 random ages between 1 and 8, so we have 8 age groups
        self._features_age  = np.random.randint(1, 8, size=50)[cv_mask]
        # Create 50 random gender values
        self._features_gender = np.random.randint(0, 1, size=50)[cv_mask]
        # Create 50 random ethnicity values
        self._features_ethnicity = np.random.randint(0, 1, size=50)[cv_mask]

        # Create a namedtuple for every feature, in this case only age
        # If you want more, e.g. gender of ethnicity, add them to the list
        # in the namedtuple constructor:
        # ["age", "gender", "ethnicity]
        self._feature_constructor = namedtuple("ExampleFeature",
                                               ["age", "gender", "ethnicity"])
        
        # Create a namedtuple constructor for an Example-Item
        self._item_constructor = namedtuple("ExampleItem", ["id",
                                                            "embedding",
                                                            "embedding_type",
                                                            "filename",
                                                            "feature"
                                                            ]
                                            )
        
        with open(f"data/subgroups_example.json", "r") as f:
            self._subgroups = json.load(f)

    @staticmethod
    def generate_cv_mask(num_folds,
                         embedding_type,
                         data_path:str="./path_to_dataset/"):
        """
        Creates a Cross-Validation mask.

        Parameters
        ----------
        num_folds : int
            The amount of folds, e.g. 5.
        embedding_type : str,
            The embedding type, e.g. facenet.
        data_path : str, optional
            The path to the stored dataset.
        Returns
        -------
        None.

        """

        # Load your IDs here, we just create 50 as an example
        IDs = np.arange(50)
        # Call the function which creates the mask
        mask = pp.generate_subject_exclusive_folds(IDs, num_folds)
        # Save the mask where you want to have it
        pu.file_np_save(f"cv-mask.npy", mask)


    def __getitem__(self, index):
        """
        Returns a single item, it's ID, embedding, the embedding type, filename,
        in this case None, and the features which is stored as a namedtuple again.

        """
        features = self._feature_constructor(
                                                self._features_age[index],
                                                self._features_gender[index],
                                                self._features_ethnicity[index]
                                             )

        return self._item_constructor(
                                        self.ids[index],
                                        self.embeddings[index],
                                        self.embedding_type,
                                        None, # Filename omitted
                                        features
                                     )


    def __len__(self):
        # Returns the length of the dataset
        return self._ids.shape[0]
    
    def remove_singletons(self):
        # Removes IDs with only one sample
        # Removes them from all data arrays
        uni_IDs, counts = np.unique(self.ids, return_counts=True)
        
        mask = np.zeros(self.__len__(), dtype=bool)
        for ID in uni_IDs[counts < 2]:
           mask = np.logical_or(mask, self.ids == ID)
        
        mask = np.logical_not(mask)
        
        self._embeddings = self._embeddings[mask]
        self._ids = self._ids[mask]
        self._features_age = self._features_age[mask]
        self._features_gender = self._features_gender[mask]
        self._features_ethnicity = self._features_ethnicity[mask]
        
    @property
    def embeddings(self):
        return self._embeddings

    @property
    def filenames(self):
        return self._filenames

    @property
    def ids(self):
        return self._ids

    @property
    def features(self):
        return self._feature_constructor(
                                         self._features_age,
                                         self._features_gender,
                                         self._features_ethnicity
                                         )

    @property
    def dataset_name(self):
        return "example"

    @property
    def subgroups(self):
        return self._subgroups

