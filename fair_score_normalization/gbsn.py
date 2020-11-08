# Author: Jan Niklas Kolf, 2020

# Own imports
from codebase.datasets import DATASETS
from codebase.datasets import CVDataset, EMBEDDING_TYPES
from codebase.datasets import preprocessing as pp

from codebase.clustering.methods import KMeansClustering
from codebase.clustering import GBSN

# Foreign imports
from argparse import ArgumentParser
import json

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4' 

if __name__ == "__main__":
    
    parser = ArgumentParser()    
    parser.add_argument("--mode", default="gbsn", choices=["gbsn", "batch", "pp-cv"])
    parser.add_argument("--file", type=str)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--embeddings", type=str)
    parser.add_argument("--cluster-method", default="kmeans", choices=["kmeans"])
    parser.add_argument("--k", default=100, type=int)

    args = parser.parse_args()
    
    
    DT = DATASETS[args.dataset]
    
    EMB = EMBEDDING_TYPES._asdict()[args.embeddings]
    
    CLUSTERING =    {
                    "kmeans":KMeansClustering    
                    }[args.cluster_method]
    
    K = args.k
    
    MDL_NAME = args.model_name if len(args.model_name) > 0 \
                               else f"{args.dataset}_{args.embeddings}_k{K}"
                               

    if args.mode == "gbsn":
        # Run the model on the given dataset with cross validation
        gbsn = GBSN(CLUSTERING(K), model_name=MDL_NAME)
        
        for CVD in CVDataset(DT, EMB, fold_count=args.folds):
            global_thr, cluster_thresholds = gbsn.train(CVD)
            gbsn.test(CVD, global_thr, cluster_thresholds)
                
    elif args.mode == "batch":
        
        with open(args.file, "r") as f:
            data = json.load(f)
            
        for K in data["ks"]:
            
            MDL_NAME = args.model_name if len(args.model_name) > 0 \
                               else f"{args.dataset}_{args.embeddings}_k{K}"
                               
            gbsn = GBSN(CLUSTERING(K), model_name=MDL_NAME)
        
            for CVD in CVDataset(DT, EMB, fold_count=args.folds):
                global_thr, cluster_thresholds = gbsn.train(CVD)
                gbsn.test(CVD, global_thr, cluster_thresholds)
        
    
    elif args.mode == "pp-cv":
        DT.generate_cv_mask(args.folds,
                            EMB)
        