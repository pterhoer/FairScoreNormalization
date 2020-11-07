# Author: Jan Niklas Kolf, 2020

from codebase.utils import path_utils as pu

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def setup(size=2.1):
    sns.set(font_scale=size)
    sns.set_style("whitegrid")
    sns.set_style({'legend.frameon': True})
    
    
def plot_eer_over_k(dataframe, 
                    Ks, 
                    plot_name:str=None,
                    display=False,
                    loc="lower right",
                    plot_name_addition:str=None):
    
    pu.make_path("data/plots/eer/")
    setup(2.5)
    
    plt.figure(num=None, figsize=(17, 6), dpi=80, facecolor="w", edgecolor="k")
    ax = sns.lineplot(x="x_ed", y="EER", hue="", data=dataframe)
    
    ax.set_ylabel("Equal Error Rate")
    ax.set_xlabel("Individuality parameter $k$")
    
    plt.xticks(np.arange(len(Ks)), Ks, rotation="45")
    plt.xlim([0, len(Ks)-1])
    
    plt.legend(loc=loc)
    plt.tight_layout()

    if not plot_name is None:
        path = f"data/plots/eer/{plot_name}/"
        pu.make_path(path)
        
        if not plot_name_addition is None:
            plt.savefig(f"{path}lp_eer_over_k_{plot_name}{plot_name_addition}.png", bbox_inches="tight")
        else:
            plt.savefig(f"{path}lp_eer_over_k_{plot_name}.png", bbox_inches="tight")
    
    if display:
        plt.show()
    else:
        plt.clf()
        plt.close()
    # done
    
def plot_fnmr_over_k(dataframe,
                     Ks,
                     plot_name:str=None,
                     display=True,
                     loc="upper center",
                     plot_name_addition:str=None,
                     plot_settings=None):
    
    pu.make_path("data/plots/fnmr/")
    setup(2.3)
    
    plt.figure(num=None, figsize=(7, 4), dpi=80, facecolor="w", edgecolor="k")
    #plt.title(f"{plot_name} {plot_name_addition}")
    ax = sns.lineplot(x="x_ed", y="EER", hue="", data=dataframe)
    
    ax.set_ylabel("FNMR")
    ax.set_xlabel("Individuality parameter $k$")
    
    plt.xticks(np.arange(len(Ks)), Ks, rotation="45")
    plt.xlim([0, len(Ks)-1])
    
    if plot_settings is not None:
        
        if "ylim" in plot_settings.keys():
            plt.ylim(plot_settings["ylim"])
    
    plt.legend(loc=loc, prop={'size': 15})
    plt.tight_layout()

    if not plot_name is None:
        path = f"data/plots/fnmr/"
        
        if not plot_name_addition is None:
            plt.savefig(f"{path}lp_eer_over_k_{plot_name}{plot_name_addition}.png", bbox_inches="tight")
        else:
            plt.savefig(f"{path}lp_eer_over_k_{plot_name}.png", bbox_inches="tight")
    
    if display:
        plt.show()
    else:
        plt.clf()
        plt.close()
    # done