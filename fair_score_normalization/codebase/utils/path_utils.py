# Author: Jan Niklas Kolf, 2020

from pathlib import Path
from numpy import save as np_save
from pickle import dump as pickle_dump
from pickle import load as pickle_load

def file_test(path: str):
    file = Path(path)
    return file.exists()

def folder_test(path: str):
    path = Path(path)
    return path.is_dir()

def file_np_save(path: str, np_array):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np_save(path, np_array)

def file_pickle_save(path:str, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle_dump(data, f)

def file_pickle_load(path:str):
    if file_test(path):
        with open(path, "rb") as f:
            t = pickle_load(f)
        return t

    print(f"Pickle-File {path} could not be found!")
    return None

def dict_merge(dict_base, dict_new, mode="replace"):

    if mode == "replace":
        for key in dict_new.keys():
            dict_base[key] = dict_new[key]

        return dict_base


def make_path(path:str):
    Path(path).mkdir(parents=True, exist_ok=True)