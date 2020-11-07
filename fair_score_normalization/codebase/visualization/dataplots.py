# Author: Jan Niklas Kolf, 2020

from codebase.utils import path_utils as pu


import matplotlib.pyplot as plt
import numpy as np

from typing import Dict

def plot_subgroup_distribution(subgroup_data:Dict, 
                               dataset_name:str,
                               embedding_type:str,
                               type_samples=True,
                               type_percentage=True,
                               display=False):
    
    pu.make_path("data/plots/distributions/")
    
    num_type = "samples" if type_samples else "ids"
    
    selector = "%" if type_percentage else "#"
    name = "percentage" if type_percentage else "absolute"
    
    grouping = subgroup_data["__subgroups__"]["__grouping__"]
    
    
    for subgroup in grouping.keys():

        if subgroup.startswith("__"):
            continue
        
    
        data = [[f"{subgroup.title()} [{selector}]", "Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4"]]
        for key in  grouping[subgroup]["keys"]:

            row = [subgroup_data["__subgroups__"][key]["__displayname__"]]
            row.extend([str(round(s, 2)) for s in subgroup_data[key][num_type][selector]])
            
            data.append(row)
    
        fig, ax = plt.subplots()

        # hide axes
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        data = np.array(data)
        
        ax.table(cellText=data, loc='center', cellLoc="center")
        
        
        
        fig.tight_layout()
        
        path = "D:/jkolf/group-based-score-normalization/data/plots/distributions/"
        ext = f"{dataset_name}_{embedding_type}/{num_type}_{name}/"
        pu.make_path(path+ext)
        
        plt.savefig(f"{path}{ext}dist_subgroup_{dataset_name}_{embedding_type}_{subgroup}_{name}_{num_type}.png", 
                    bbox_inches="tight", dpi=200)

        if display:
            plt.show()
        else:
            plt.clf()
            plt.close()
        

def plot_valid_cluster_amount(cluster_data : Dict,
                              dataset_name:str,
                              embedding_type:str):
    
    table_data = [["Cluster Size", "Valid @ Fold 0", "Valid @ Fold 1", "Valid @ Fold 2", "Valid @ Fold 3", "Valid @ Fold 4"]]
    
    for k in cluster_data.keys():
        
        valid_absolute = [str(k)]
        valid_absolute.extend([str(t) for t in cluster_data[k]])
    
        table_data.append(valid_absolute)
    
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    
    
    ax.table(cellText=table_data, loc='center', cellLoc="center")
    
    fig.tight_layout()
    
    path = "D:/jkolf/group-based-score-normalization/data/plots/valid_clusters/"
    ext = f"{dataset_name}_{embedding_type}/"
    pu.make_path(path+ext)
    
    plt.savefig(f"{path}{ext}valid_clusters_{dataset_name}_{embedding_type}.png", 
                bbox_inches="tight", dpi=200)

if __name__ == "__main__":

    for dts in ["adience", "colorferet"]:
        for ft in ["facenet", "arcface"]:

            pkl = pu.file_pickle_load(f"D:/jkolf/group-based-score-normalization/data/subgroup_distributions/{dts}_{ft}.pkl")
            plot_subgroup_distribution(pkl, dts, ft, type_percentage=True, type_samples=True)
            plot_subgroup_distribution(pkl, dts, ft, type_percentage=True, type_samples=False)
            plot_subgroup_distribution(pkl, dts, ft, type_percentage=False, type_samples=True)
            plot_subgroup_distribution(pkl, dts, ft, type_percentage=False, type_samples=False)
    
