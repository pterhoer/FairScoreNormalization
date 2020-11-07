from .base_dataset import Dataset, CVDataset, EMBEDDING_TYPES
from .dataset_example import ExampleDataset
from . import preprocessing

"""
Add your datasets to this dictionary.
"""
DATASETS = {
            "example" : ExampleDataset
            }