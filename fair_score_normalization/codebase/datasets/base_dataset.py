# Author: Jan Niklas Kolf, 2020

# Own imports
from codebase.utils import metrics
from codebase.utils import path_utils as pu

# Foreign imports
from abc import ABC, abstractmethod
from typing import List
from collections import namedtuple
from itertools import product
from tqdm import tqdm

import numpy as np
import sys

EMBEDDING_TYPES = namedtuple("FEATURES", ["ARCFACE", "FACENET", "VGG"])("arcface", "facenet", "vgg")

class Dataset(ABC):


    def __init__(self, embedding_type, folds:List[int], data_path:str):
        if embedding_type not in EMBEDDING_TYPES:
            raise ValueError("Unknown features type specified!")

        self._folds = folds
        self._embedding_type = embedding_type
        self._data_path = data_path
        
        self._css_matrix = None
        self._rows = None
        
    @staticmethod
    @abstractmethod
    def generate_cv_mask(num_folds:int,
                         embedding_type:str,
                         data_path:str):
        # This static method is used to calculate a cross validation mask
        pass

    @abstractmethod
    def __getitem__(self, index):
        # Returns the sample at index
        pass

    @abstractmethod
    def __len__(self):
        # Returns the lenght of the dataset
        pass

    @abstractmethod
    def remove_singletons(self):
        # Removes IDs with only one sample
        pass

    @property
    @abstractmethod
    def embeddings(self):
        # Returns the embeddings, e.g. facenet, arcface
        pass

    @property
    @abstractmethod
    def filenames(self):
        # Filenames of each image
        pass

    @property
    @abstractmethod
    def ids(self):
        # The corresponding id of a sample
        pass

    @property
    @abstractmethod
    def dataset_name(self):
        # Returns name of the dataset as string
        pass

    @property
    @abstractmethod
    def features(self):
        # Returns features, e.g. age, gender, ethnicity etc.
        pass

    @property
    @abstractmethod
    def subgroups(self):
        # Combination of subgroups, e.g. if you wish to
        # combine several ages into one group.
        # Dict
        pass

    @property
    def embedding_type(self):
        return self._embedding_type

    @property
    def folds(self):
        return self._folds

    def calculate_css(self, row_selection, column_selection):

        if isinstance(row_selection, int):
            return metrics.gpu_one_against_all(self.embeddings[row_selection],
                                               self.embeddings[column_selection])

        return metrics.gpu_gen_against_impgen(self.embeddings[row_selection],
                                              self.embeddings[column_selection])


    def generate_scores(self,   mask_remaining_samples=None, 
                                silent:bool=False,
                                bin_size:int=500, 
                                imposter_count:int=2500,
                                method = "dynamic"):
        """
        Generates genuine/imposter scores for EER calculation.
        Mask is used to mask out samples which should not be considered.

        Parameters
        ----------
        mask_remaining_samples : numpy ndarray, type bool, default is None
            This is used to mask out samples which should
            not be considered during score generation.
            A sample can be masked out e.g. if the sample did not reach a score threshold.

            If None is given, a full-true mask with the size of this dataset is created.
            Then all samples are used for the score generation.

        silent : bool, optional
            DESCRIPTION. The default is False.
        bin_size : TYPE, optional
            DESCRIPTION. The default is 2000.
        imposter_count : TYPE, optional
            DESCRIPTION. The default is 2000.

        Returns
        -------
        gen_imp_labels : TYPE
            0/1 matrix signaling an imposter/genuine comparison.
        scores : TYPE
            Cosine Similarity of each comparison.

        """
        if method == "dynamic":
            return self._generate_dynamic_scores(mask_remaining_samples,
                                                 silent=silent,
                                                 bin_size=bin_size,
                                                 imposter_count=imposter_count)
        
        else:
            return self._generate_static_scores(mask_remaining_samples,
                                                silent=silent,
                                                imposter_count = imposter_count)
        

    def _generate_static_scores(self,   mask_remaining_samples=None, 
                                        silent:bool=False,
                                        imposter_count:int=2500):
        """
        Calculates Cosine-Similarity Scores in a static way, which
        means that it utilizes a pre computed cosine-similarity matrix (dense).
        Only feasible vor small datasets.

        Parameters
        ----------
        mask_remaining_samples : np.ndarray of type bool, optional
            True=Sample should be used for calculation. 
            The default is None. If None, every sample in the dataset
            is considered.
        silent : bool, optional
            If True, TQDM is not displayed. The default is False.
        imposter_count : int, optional
            Min amount of imposters used for every sample. The default is 2500.

        Returns
        -------
        css_ids : np.array
            The ID for each sample, 1=Genuine comparison, 0=imposter comparison.
        css_scores : np.array
            Scores for each sample.
        row_idx : np.array
            Row of each sample, which means index in dataset-array.

        """
        if self._css_matrix is None:
            css_path = f"{self._data_path}css_matrix_{self.embedding_type}_"+\
                       f"{''.join(str(s) for s in sorted(self.folds))}.npy"
            if pu.file_test(css_path):
                self._css_matrix = np.load(css_path)
            else:
                self._css_matrix = metrics.gpu_calculate_csm(self.embeddings)
                pu.file_np_save(css_path, self._css_matrix)
        
        if self._rows is None:
            self._rows = np.arange(self.__len__(), dtype=int)
        
        if mask_remaining_samples is None:
            mask_remaining_samples = np.ones((self.__len__(),), dtype=bool)

        
        css_scores = np.array([], dtype=float)
        css_ids = np.array([], dtype=int)
        css_rows_left = np.array([], dtype=int)
        css_rows_right = np.array([], dtype=int)
        
        for row in  tqdm(
                        self._rows[mask_remaining_samples],
                        unit="row",
                        disable=silent,
                        desc=f"{self.dataset_name}/{self.embedding_type} - Score Gen.",
                        position=2,
                        file=sys.stdout,
                        leave=False):

            ID_mask = self.ids == self.ids[row]
            
            # Genuine
            genuine_mask = np.logical_and(ID_mask, mask_remaining_samples)
            genuine_scores = self._css_matrix[row, genuine_mask]
            
            css_scores = np.append(css_scores, genuine_scores)
            
            css_ids = np.append(
                                css_ids, 
                                np.full((len(genuine_scores),), 1, dtype=int)
                               )
            
            css_rows_left = np.append(
                                        css_rows_left,
                                        np.full((len(genuine_scores),), row, dtype=int)
                                     )
            
            css_rows_right = np.append(
                                        css_rows_right,
                                        self._rows[genuine_mask]
                                      )
            
            del genuine_mask, genuine_scores
            
            # Imposter
            imposter_mask = np.logical_and(
                                    np.logical_not(ID_mask), 
                                    mask_remaining_samples
                                           )
            
            imposter_amount = np.sum(imposter_mask)
            if imposter_amount > imposter_count:
                imposter_amount = imposter_count
                
            imposter_choice = np.random.choice(self._rows[imposter_mask],
                                               imposter_amount,
                                               replace=False)
            
            imposter_scores = self._css_matrix[row, imposter_choice]
            

            css_scores = np.append(css_scores, imposter_scores)
            
            css_ids = np.append(
                            css_ids,
                            np.full((imposter_amount,), 0, dtype=int)
                        )
            
            css_rows_left = np.append(
                                css_rows_left,
                                np.full((imposter_amount,), row, dtype=int)
                            )
            
            css_rows_right = np.append(
                                css_rows_right,
                                imposter_choice
                            )
            # End for
        
        row_idx = np.column_stack((css_rows_left, css_rows_right)).astype(int)
        
        return css_ids, css_scores, row_idx
            

    def _generate_dynamic_scores(self,  mask_remaining_samples=None, 
                                        silent:bool=False,
                                        bin_size:int=500, 
                                        imposter_count:int=2500):
        # Mask is None, therefore create a full true mask
        if mask_remaining_samples is None:
            mask_remaining_samples = np.ones((self.__len__(),), dtype=bool)
        elif np.sum(mask_remaining_samples) == 0:
            return [], [], []
            

        combinations = self._css_get_combinations(mask_remaining_samples, bin_size)

        scores, gen_imp_labels = np.array([]), np.array([])

        row_indices = []

        for collection_idx in tqdm(
                                    range(len(combinations)),
                                    unit="bin",
                                    disable=silent,
                                    desc=f"{self.dataset_name}/{self.embedding_type} - Score Gen.",
                                    position=2,
                                    file=sys.stdout,
                                    leave=False):

            ID_batch = combinations[collection_idx]

            if len(ID_batch) == 0:
                continue

            gim, c_rows, c_columns = self._css_batch_ids(ID_batch,
                                                         mask_remaining_samples,
                                                         imposter_count)

            score_data = self.calculate_css(c_rows, c_columns)

            gen_imp_labels = np.append(gen_imp_labels, gim.flatten())
            scores = np.append(scores, score_data.flatten())

            row_indices.extend(list(product(c_rows, c_columns)))

        return gen_imp_labels, scores, np.array(row_indices)


    def _css_get_combinations(self, mask_remaining_samples, bin_size=2000):
        """
        Creates bins of minimum bin_size amount of samples.
        All samples/ids in one bin are calculated together on a GPU run.
        If only one ID per bin should be used, set
        bin_size=0 or bin_size=1
    

        Parameters
        ----------
        mask_remaining_samples : np.array of type bool
            Mask-Index = True, sample is used.
            There is no default value for the mask, nor is it checked.
        bin_size : int, optional
            Amount of samples per bin. 
            bin_size=0 or bin_size=1 if only one id should be used for a run.
            The default is 2000.

        Returns
        -------
        combinations : list of lists of ints
            DESCRIPTION.

        """
        
        # Get amount of samples per id.
        # Throw out ids with only one sample, as no genuine comparison is possible
        # for them
        uniques, counts = np.unique(self.ids[mask_remaining_samples], return_counts=True)
        uniques, counts = uniques[counts > 1], counts[counts > 1]

        combinations = [[]]
        count = 0

        # Randomize shuffling
        iters = np.arange(len(uniques))
        np.random.shuffle(iters)

        # Combine ids to bins
        for collection_idx in iters:
            if count > bin_size:
                count = 0
                combinations.append([])

            combinations[-1].append(uniques[collection_idx])
            count += counts[collection_idx]

        return combinations

    def _css_batch_ids(self, ID_batch, mask_remaining_samples, imposter_count=2000, debug=False):
        # Sort the IDs
        ID_batch = np.sort(ID_batch).astype(int)
        # Create a mask to check which data is a possible imposter (ID of Imp is not in ID_batch)
        mask_possible_imposter = np.ones(self.__len__(), dtype=bool)
        # Get an array for the row selection of genuine data
        row_selection_genuine = np.array([])
        # Create an array to store which row index corresponds to which ID
        gen_identifier = np.array([])

        rows = np.arange(self.__len__(), dtype=int)

        for ID in ID_batch:
            # Select valid rows of that ID
            row_selection = rows[np.logical_and(mask_remaining_samples, self.ids == ID)]
            # Append the selection to our genuine row collection
            row_selection_genuine = np.append(row_selection_genuine, row_selection)
            # Append the ID to the identifier array so we know which row corresponds to which ID
            gen_identifier = np.append(gen_identifier, np.full(len(row_selection), ID, dtype=int))
            # Remove the ID from possible imposter
            mask_possible_imposter = np.logical_and(mask_possible_imposter, self.ids != ID)

        del row_selection, ID, ID_batch

        mask_possible_imposter = np.logical_and(mask_possible_imposter, mask_remaining_samples)

        # Get the maximum available imposter amaount, capped by given imposter_count variable
        imp_max = np.sum(mask_possible_imposter)
        imposter_count = imp_max if imp_max < imposter_count else imposter_count
        del imp_max

        # Selection randomly additional imposter_count amount of imposter samples
        if imposter_count == 0:
            row_selection_imposter = []    
        else:
            row_selection_imposter = np.random.choice(
                                                    rows[mask_possible_imposter],
                                                    size=imposter_count,
                                                    replace=False
                                                 )

        # These rows are now selected for the comparison
        row_selection_comp = np.append(row_selection_genuine, row_selection_imposter)

        gen_imp_mask = np.zeros((len(gen_identifier), len(row_selection_comp)), dtype=bool)

        if debug:
            id_mapping = np.zeros((len(gen_identifier), len(row_selection_comp)), dtype=bool)

        for idx in range(gen_identifier.shape[0]):

            ID = gen_identifier[idx]
            gen_imp_mask[idx] = np.append(gen_identifier == ID, np.zeros(imposter_count, dtype=bool))

            if debug:
                id_mapping[idx] = np.full(len(row_selection_comp), ID, dtype=int)

        if debug:
            return id_mapping, gen_imp_mask, row_selection_genuine, row_selection_comp


        # gen_imp_mask specifies which comparison value is a genuine or an imposter comparison
        # row_selection_genuine selectes the rows of the cosine similarity matrix
        # row_selection_comp selects the columns per row of the css matrix
        return gen_imp_mask, row_selection_genuine.astype(int), row_selection_comp.astype(int)

class CVDataset:

    def __init__(self,
                 dts_constructor:Dataset,
                 embedding_type:str,
                 fold_count:int=5
                 ):
        """
        Cross Validation Dataset
        Used to iterate over fold_count amount of folds of each dataset.
        The used dataset is given with its constructor method.
        The dataset needs to implement the loading of a list of folds.
        See example dataset in codebase.datasets.dataset_example.py

        Parameters
        ----------
        dts_constructor : Dataset
            The dataset to be used.
        embedding_type : str
            The type of embeddings used. Is passed to dts_constructor.
        fold_count : int, optional
            Amount of folds to use. The default is 5.

        Returns
        -------
        Created instance of CVDataset.

        """


        self._embedding_type = embedding_type
        self._fold_count = fold_count

      
        self._train_folds = None
        self._test_dataset = None
        self._train_dataset = None
        self._current_test_fold = -1
        
        self._dts_constructor = dts_constructor

    @property
    def embedding_type(self):
        return self._embedding_type
    
    @property
    def dataset_name(self):
        return self._dts_constructor.dataset_name

    @property
    def amount_folds(self):
        return self._fold_count
    
    @property
    def train_dataset(self):
        return self._train_dataset
    
    @property
    def test_dataset(self):
        return self._test_dataset
    
    @property
    def current_test_fold(self):
        return self._current_test_fold

        
    def __iter__(self):
        return self
    
    def __next__(self):
        # Increments to the next fold
        self._current_test_fold += 1
        
        if self._current_test_fold < self._fold_count:
            
            # Creates a list of ints of folds used for training
            self._train_folds = [
                                 f for f in range(self.amount_folds) \
                                 if f != self._current_test_fold
                                ]
            # Uses the constructor to create the training dataset
            self._train_dataset = self._dts_constructor(
                                                self._embedding_type,
                                                self._train_folds
                                       )
            # Removes ids with only one sample
            self._train_dataset.remove_singletons()
            
            # Create test dataset
            self._test_dataset  = self._dts_constructor(
                                                self._embedding_type,
                                                [self._current_test_fold]
                                       )
                        
            return self
        
        else:
            raise StopIteration
        
        