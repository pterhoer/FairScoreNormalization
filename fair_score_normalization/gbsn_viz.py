# Author: Jan Niklas Kolf, 2020

from codebase.datasets import DATASETS, EMBEDDING_TYPES
from codebase.clustering import GBSN
from codebase.utils import path_utils as pu
from codebase.evaluation import evaluate_models as em
from codebase.evaluation.model_iterators import ITERATOR
from codebase.visualization import lineplots as lp

from codebase.modes import modes_visualization as mv

# Foreign imports
from argparse import ArgumentParser
import numpy as np
import json
import csv


if __name__ == "__main__":
    
    parser = ArgumentParser()    
    parser.add_argument("--file", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--embeddings", type=str)
    parser.add_argument("--ks", default=[5,10,15])

    args = parser.parse_args()
        
    ITER = ITERATOR[args.dataset]
    
    EMB = EMBEDDING_TYPES._asdict()[args.embeddings]


    with open(args.file, "r") as f:
        specs = json.load(f)
    del f
    
    plot_name = f"{args.dataset}_{args.embeddings}"
    
    mv.plot_fnmr(ITER(args.embeddings, specs["k"], multiplicator=None, prefix=None)
                 , specs["k"], plot_name)
    
