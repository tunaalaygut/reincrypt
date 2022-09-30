import numpy as np
import os
import json


def str_to_ndarray(str: str) -> np.ndarray:
    rows = str.split('\n')
    arr = np.ndarray((len(rows), len(rows[0].split())))
    
    for idx, row in enumerate(rows):
        arr[idx, :] = row.split()
    
    return arr


def read_config(config_filename: str, output_dir="output"):
    config = {}

    with open(f"config/{config_filename}.json", "r+") as f:
        config = json.load(f)
    config["experiment_name"] = config_filename

    os.makedirs(output_dir, exist_ok=True)
    
    return config
