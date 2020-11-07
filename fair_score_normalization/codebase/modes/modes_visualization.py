# Author: Jan Niklas Kolf, 2020

# Own imports
from codebase.evaluation import evaluate_models as em
from codebase.visualization import lineplots as lp

# Foreign imports
from typing import List
import pandas as pd
import numpy as np

def plot_eer(gbsn_iterable,
             num_folds:int=5,
             plot_name:str=None, 
             display=False,
             loc="lower right"):
    
    r = em.cluster_global_eer(gbsn_iterable, 
                              num_folds=num_folds
                              )
    
    dataframe = em.create_dataframe(r[0], r[1], r[2])
    
    lp.plot_eer_over_k(dataframe, r[0], 
                       plot_name=plot_name, 
                       display=display,
                       loc=loc)

def plot_fnmr(gbsn_iterable,
              ks,
              plot_name:str=None,
              plot_name_addition:str=None,
              display=True,
              loc="upper center",
              plot_settings=None):
    
    results = em.calculate_fnmr(gbsn_iterable)

    X = []
    X_ed = []
    Y = []
    CAT = []
    
    for idx in range(len(ks)):
        k = ks[idx]
        X.extend([k]*10)
        Y.extend(results["global"][k]["normalized"])
        CAT.extend(["Normalized"]*5)
        Y.extend(results["global"][ks[0]]["unnormalized"])
        CAT.extend(["Unnormalized"]*5)
        
        X_ed.extend(np.full(10, idx))
        
    df = pd.DataFrame({"x":X, "EER":Y, "":CAT, "x_ed":X_ed})


    lp.plot_fnmr_over_k(
                    df,
                    ks,
                    plot_name,
                    display,
                    loc,
                    plot_name_addition,
                    plot_settings
       )
    
    
def plot_subgroups(gbsn_iterable,
                   num_folds:int=5,
                   plot_prefix:str=None,
                   save_plots:bool=True,
                   display=False,
                   loc="lower right"):
    
    r = em.cluster_subgroup_eer(gbsn_iterable, 
                                num_folds=num_folds
                                )
    
    for subgroup in r[2].keys():
        
        dataframe = em.create_dataframe(r[0], r[1][subgroup][0], r[2][subgroup])
        
        if save_plots:
            plot_addition = "_subgroup_"+subgroup if not plot_prefix is None \
                        else subgroup
        else:
            plot_addition = None
            plot_prefix = None
            
        lp.plot_eer_over_k(dataframe, r[0], 
                            plot_name=plot_prefix,
                            display=display,
                            loc=loc,
                            plot_name_addition=plot_addition)